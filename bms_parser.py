"""
[BMS 읽기] BMS 파서 (SP 전용)

BMS 파일 -> Chart(노트 목록 + 키음 이벤트 + 메타데이터)
시간 변환은 마디 위치를 Fraction(분수)으로 계산해서 부동소수점 오차를 피하고,
BPM 변경(채널 03/08), STOP(채널 09), 마디 길이 변경(채널 02)을 모두 반영한다.

사용 예:
    python bms_parser.py path/to/chart.bms
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import bisect
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

# ---------------------------------------------------------------------------
# 레인 매핑 (SP 1P): 0 = 스크래치, 1~7 = 건반
# ---------------------------------------------------------------------------
LANE_MAP = {"16": 0, "11": 1, "12": 2, "13": 3, "14": 4, "15": 5, "18": 6, "19": 7}
# 롱노트 채널(LNTYPE 방식)은 일반 채널 + 40 (11 -> 51, 16 -> 56 ...)
LN_MAP = {"5" + k[1]: v for k, v in LANE_MAP.items()}
# 키음이 들어있는 채널 (SP 기준): BGM, 1P 노트, 투명 노트, 롱노트
SOUND_CHANNELS = {"01"} | set(LANE_MAP) | {f"3{i}" for i in range(1, 10)}

DATA_RE = re.compile(r"^#(\d{3})([0-9A-Za-z]{2}):\s*(.*)$")
HEADER_RE = re.compile(r"^#([A-Za-z0-9_]+)(?:\s+(.*))?$")


@dataclass
class Note:
    time: float                    # 노트 시작 시각(초)
    lane: int                      # 0=스크래치, 1~7=건반
    end_time: float | None = None  # 롱노트면 끝 시각, 일반 노트면 None
    wav: int = 0                   # 키음 번호
    beat: float = -1.0             # 곡 시작부터의 박(4분음표) 위치 (-1 = 모름)
    mbeat: float = -1.0            # 그 노트가 있는 마디 안에서의 박 위치 (0 = 마디 첫 박)

    @property
    def is_long(self) -> bool:
        return self.end_time is not None


@dataclass
class Chart:
    path: str = ""
    title: str = ""
    artist: str = ""
    genre: str = ""
    level: int | None = None
    player: int = 1
    init_bpm: float = 130.0
    notes: list[Note] = field(default_factory=list)
    sounds: list[tuple[float, int]] = field(default_factory=list)  # (시각, wav번호)
    wav_files: dict[int, str] = field(default_factory=dict)        # wav번호 -> 파일명
    num_measures: int = 0
    duration: float = 0.0
    is_dp: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def read_text(path: str | Path) -> str:
    """BMS는 인코딩이 제각각(대부분 Shift-JIS)이라 순서대로 시도한다."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp932", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace")


def split_tokens(data: str) -> list[str]:
    data = data.strip()
    return [data[i:i + 2] for i in range(0, len(data) - len(data) % 2, 2)]


def b36(tok: str) -> int:
    return int(tok, 36)


def to_fraction(text: str) -> Fraction:
    return Fraction(text.strip()).limit_denominator(100000)


# ---------------------------------------------------------------------------
# 타이밍 테이블
# ---------------------------------------------------------------------------
class TimingMap:
    """beat(4/4 기준 4박 = 1마디) -> 초 변환기. BPM 변경과 STOP 포함."""

    def __init__(self, init_bpm: float, changes: dict[Fraction, dict]):
        # changes[beat] = {"bpm": float|None, "stop": float(1/192 단위)|0}
        if Fraction(0) not in changes:
            changes[Fraction(0)] = {"bpm": None, "stop": 0.0}
        self.beats: list[Fraction] = sorted(changes)
        self.bpms: list[float] = []
        self.t_after: list[float] = []   # STOP이 끝난 직후 시각

        prev_beat = Fraction(0)
        prev_bpm = init_bpm
        prev_time_after = 0.0
        for beat in self.beats:
            t_before = prev_time_after + float(beat - prev_beat) * 60.0 / prev_bpm
            bpm = changes[beat]["bpm"] if changes[beat]["bpm"] is not None else prev_bpm
            if bpm <= 0:
                raise ValueError(f"지원하지 않는 BPM 값: {bpm}")
            stop_units = changes[beat]["stop"]
            stop_sec = (stop_units / 192.0) * 4.0 * 60.0 / bpm if stop_units else 0.0
            self.bpms.append(bpm)
            self.t_after.append(t_before + stop_sec)
            prev_beat, prev_bpm, prev_time_after = beat, bpm, t_before + stop_sec

    def to_time(self, beat: Fraction) -> float:
        # beat보다 "엄격히 앞선" 마지막 타이밍 포인트 -> 같은 beat의 노트는 STOP 이전에 재생
        i = bisect.bisect_left(self.beats, beat) - 1
        if i < 0:
            return 0.0
        return self.t_after[i] + float(beat - self.beats[i]) * 60.0 / self.bpms[i]


# ---------------------------------------------------------------------------
# 메인 파서
# ---------------------------------------------------------------------------
def parse_bms(path: str | Path) -> Chart:
    chart = Chart(path=str(path))
    text = read_text(path)

    headers: dict[str, str] = {}
    bpm_table: dict[int, float] = {}    # #BPMxx / #EXBPMxx
    stop_table: dict[int, float] = {}   # #STOPxx
    lnobj: int | None = None
    lntype = 1
    entries: list[tuple[int, str, str]] = []  # (마디, 채널, 데이터)

    # --- 1) 줄 단위 파싱 (#RANDOM/#IF는 1번 분기만 채택) ---
    rand_val = 1
    active = True
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue

        m = DATA_RE.match(line)
        if m:
            if active:
                entries.append((int(m.group(1)), m.group(2).upper(), m.group(3)))
            continue

        h = HEADER_RE.match(line)
        if not h:
            continue
        key = h.group(1).upper()
        val = (h.group(2) or "").strip()

        if key in ("RANDOM", "SETRANDOM"):
            rand_val = 1
        elif key == "IF":
            active = val.isdigit() and int(val) == rand_val
        elif key == "ELSE":
            active = not active
        elif key in ("ENDIF", "END"):
            active = True
        elif not active:
            continue
        elif key.startswith("EXBPM") and len(key) == 7:
            bpm_table[b36(key[5:])] = float(val)
        elif key.startswith("BPM") and len(key) == 5:
            bpm_table[b36(key[3:])] = float(val)
        elif key.startswith("STOP") and len(key) == 6:
            stop_table[b36(key[4:])] = float(val)
        elif key.startswith("WAV") and len(key) == 5:
            chart.wav_files[b36(key[3:])] = val
        elif key == "LNOBJ":
            lnobj = b36(val)
        elif key == "LNTYPE":
            lntype = int(val) if val.isdigit() else 1
        else:
            headers[key] = val

    chart.title = headers.get("TITLE", "")
    chart.artist = headers.get("ARTIST", "")
    chart.genre = headers.get("GENRE", "")
    try:
        chart.init_bpm = float(headers.get("BPM", "130"))
    except ValueError:
        chart.init_bpm = 130.0
    try:
        chart.level = int(float(headers.get("PLAYLEVEL", "")))
    except ValueError:
        chart.level = None
    try:
        chart.player = int(headers.get("PLAYER", "1"))
    except ValueError:
        chart.player = 1

    if lntype != 1:
        chart.warnings.append(f"LNTYPE {lntype}은 정식 지원하지 않음(1처럼 처리)")

    # DP 패턴 감지 (채널 21~29 / 61~69에 노트가 있거나 PLAYER 2,3)
    dp_channels = {f"2{i}" for i in range(1, 10)} | {f"6{i}" for i in range(1, 10)}
    chart.is_dp = chart.player in (2, 3) or any(
        ch in dp_channels and any(t != "00" for t in split_tokens(d)) for _, ch, d in entries
    )
    if chart.is_dp:
        chart.warnings.append("DP 채보로 판단됨 (SP 전용 파이프라인에서는 제외 권장)")

    if not entries:
        return chart

    # --- 2) 마디 길이 / 마디 시작 beat ---
    max_measure = max(m for m, _, _ in entries)
    chart.num_measures = max_measure + 1
    measure_len: dict[int, Fraction] = {}
    for meas, ch, data in entries:
        if ch == "02":
            try:
                measure_len[meas] = to_fraction(data)
            except (ValueError, ZeroDivisionError):
                chart.warnings.append(f"마디 {meas}: 잘못된 길이 '{data}'")

    measure_beats: list[Fraction] = []   # 각 마디가 차지하는 beat 수
    measure_start: list[Fraction] = []   # 각 마디 시작 beat
    acc = Fraction(0)
    for m in range(max_measure + 2):
        measure_start.append(acc)
        b = 4 * measure_len.get(m, Fraction(1))
        measure_beats.append(b)
        acc += b

    def pos_to_beat(meas: int, i: int, n: int) -> Fraction:
        return measure_start[meas] + Fraction(i, n) * measure_beats[meas]

    # --- 3) 타이밍 포인트 (BPM / STOP) ---
    changes: dict[Fraction, dict] = {}

    def point(beat: Fraction) -> dict:
        return changes.setdefault(beat, {"bpm": None, "stop": 0.0})

    for meas, ch, data in entries:
        if ch not in ("03", "08", "09"):
            continue
        toks = split_tokens(data)
        n = len(toks)
        for i, tok in enumerate(toks):
            if tok == "00":
                continue
            beat = pos_to_beat(meas, i, n)
            if ch == "03":                      # 16진수 직접 BPM
                point(beat)["bpm"] = float(int(tok, 16))
            elif ch == "08":                    # 확장 BPM
                if b36(tok) in bpm_table:
                    point(beat)["bpm"] = bpm_table[b36(tok)]
                else:
                    chart.warnings.append(f"정의되지 않은 EXBPM {tok}")
            else:                               # STOP
                if b36(tok) in stop_table:
                    point(beat)["stop"] += stop_table[b36(tok)]
                else:
                    chart.warnings.append(f"정의되지 않은 STOP {tok}")

    timing = TimingMap(chart.init_bpm, changes)
    chart.timing = timing                                   # 임의 박 -> 초 변환용 (build_grid.py 등)
    chart.measure_starts = [float(x) for x in measure_start] + [float(acc)]   # 마디 시작 박 (+ 끝)

    # --- 4) 노트 수집 ---
    # taps[lane] = [(beat, wav)], ln_marks[lane] = [(beat, wav)]
    taps: dict[int, list[tuple[Fraction, int]]] = {l: [] for l in range(8)}
    ln_marks: dict[int, list[tuple[Fraction, int]]] = {l: [] for l in range(8)}
    sound_beats: list[tuple[Fraction, int]] = []

    for meas, ch, data in entries:
        is_lane = ch in LANE_MAP
        is_ln = ch in LN_MAP
        if not (is_lane or is_ln or ch in SOUND_CHANNELS):
            continue
        toks = split_tokens(data)
        n = len(toks)
        for i, tok in enumerate(toks):
            if tok == "00":
                continue
            wav = b36(tok)
            beat = pos_to_beat(meas, i, n)
            if is_lane:
                taps[LANE_MAP[ch]].append((beat, wav))
            elif is_ln:
                ln_marks[LN_MAP[ch]].append((beat, wav))
            elif ch in SOUND_CHANNELS:           # BGM(01), 투명 노트(3x)
                sound_beats.append((beat, wav))

    notes: list[Note] = []
    import bisect
    _starts = [float(x) for x in measure_start]

    def in_measure(beat) -> float:
        m = max(0, bisect.bisect_right(_starts, float(beat) + 1e-9) - 1)
        return float(beat) - _starts[m]

    for lane in range(8):
        lane_taps = sorted(taps[lane])

        # 4-a) LNOBJ 방식: 같은 레인에서 LNOBJ 값의 오브젝트가 직전 노트의 끝
        built: list[dict] = []
        for beat, wav in lane_taps:
            if lnobj is not None and wav == lnobj and built and built[-1]["end"] is None:
                built[-1]["end"] = beat           # LNOBJ 끝 오브젝트는 키음 재생 X
            else:
                built.append({"beat": beat, "wav": wav, "end": None})
                sound_beats.append((beat, wav))
        for b in built:
            notes.append(Note(
                time=timing.to_time(b["beat"]),
                lane=lane,
                end_time=timing.to_time(b["end"]) if b["end"] is not None else None,
                wav=b["wav"], beat=float(b["beat"]), mbeat=in_measure(b["beat"]),
            ))

        # 4-b) 5x 채널 방식: 같은 레인에서 시작/끝이 번갈아 나옴
        marks = sorted(ln_marks[lane])
        if len(marks) % 2 == 1:
            chart.warnings.append(f"레인 {lane}: 롱노트 시작/끝 짝이 맞지 않음(마지막 무시)")
            marks = marks[:-1]
        for k in range(0, len(marks), 2):
            (sb, swav), (eb, _) = marks[k], marks[k + 1]
            notes.append(Note(
                time=timing.to_time(sb), lane=lane,
                end_time=timing.to_time(eb), wav=swav, beat=float(sb), mbeat=in_measure(sb),
            ))
            sound_beats.append((sb, swav))

    notes.sort(key=lambda n: (n.time, n.lane))
    chart.notes = notes
    chart.sounds = sorted((timing.to_time(b), w) for b, w in sound_beats)

    last = max(
        [n.end_time if n.end_time is not None else n.time for n in notes]
        + [t for t, _ in chart.sounds] + [0.0]
    )
    chart.duration = last
    return chart


# ---------------------------------------------------------------------------
# CLI: 요약 출력
# ---------------------------------------------------------------------------
def summarize(chart: Chart, head: int = 10) -> None:
    n_ln = sum(1 for n in chart.notes if n.is_long)
    print(f"파일     : {chart.path}")
    print(f"제목     : {chart.title} / {chart.artist}")
    print(f"레벨     : {chart.level}   초기 BPM: {chart.init_bpm}   플레이어: {chart.player}")
    print(f"마디 수  : {chart.num_measures}   길이: {chart.duration:.1f}초")
    print(f"노트 수  : {len(chart.notes)} (롱노트 {n_ln})   키음 이벤트: {len(chart.sounds)}")
    print(f"WAV 정의 : {len(chart.wav_files)}개   DP 여부: {chart.is_dp}")
    for w in chart.warnings:
        print(f"[경고] {w}")
    print(f"\n처음 {head}개 노트 (시각, 레인, 롱노트끝):")
    for n in chart.notes[:head]:
        end = f"{n.end_time:.3f}" if n.is_long else "-"
        print(f"  {n.time:8.3f}s  lane={n.lane}  end={end}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python bms_parser.py <chart.bms>")
        sys.exit(1)
    summarize(parse_bms(sys.argv[1]))
