"""
[저장, STEP 7] 노트 목록(시각, 레인) -> BMS 파일  (BPM 고정, 4/4 박자 고정)

구조:
    #WAV01 = 곡 전체 오디오 (BGM 채널 01, 0번 마디 맨 앞에서 재생)
    #WAV02 = 무음 WAV (모든 노트가 이 키음을 가리킴 - 노트를 쳐도 소리가 안 겹침)

시간 -> 마디 위치:
    beat = 시각 * BPM / 60,  마디 = beat / 4.  마디 시작은 시각 0 (= 오디오 맨 앞)
마디마다 분할 수(N)를 따로 고른다:
    DIVS 중 "모든 노트의 반올림 오차가 tol_ms 이하"인 가장 작은 N (없으면 192).
    -> 정리된 마디는 16분/48분 같은 간결한 격자로, 어긋난 마디는 촘촘한 격자로 쓴다.

verify_written()은 쓴 파일을 bms_parser로 다시 읽어서 시각/레인이 의도와 같은지 확인한다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from bms_parser import LANE_MAP, parse_bms

LANE_TO_CH = {lane: ch for ch, lane in LANE_MAP.items()}   # 0->"16", 1->"11", ... 7->"19"
DIVS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 192)
BGM_ID = "01"
KEY_ID = "02"


def _clean(text: str) -> str:
    return re.sub(r"[\r\n#]+", " ", str(text)).strip()


def quantize(times, lanes, bpm: float, tol_ms: float = 10.0, divs=DIVS):
    """노트를 마디 격자에 맞춘다
    반환: grid[(마디, 레인)] = 칸 번호 집합, div_of[마디] = 분할 수, quant = [(맞춘 시각, 레인, 원래 시각)], dropped = 겹쳐서 버린 노트 수"""
    meas_dur = 240.0 / bpm
    if any(t < -1e-9 for t in times):
        raise ValueError("음수 시각의 노트가 있어요 (앞쪽 패딩이 필요)")

    by_meas: dict[int, list[tuple[float, int, float]]] = {}
    for t, lane in zip(times, lanes):
        t = max(float(t), 0.0)
        x = t / meas_dur
        m = int(math.floor(x + 1e-12))
        by_meas.setdefault(m, []).append((t, int(lane), x - m))

    # 1) 마디마다 분할 수 선택
    div_of: dict[int, int] = {}
    for m, items in by_meas.items():
        fracs = [f for _, _, f in items]
        div_of[m] = divs[-1]
        for N in divs:
            err_ms = max(abs(f * N - round(f * N)) for f in fracs) / N * meas_dur * 1000.0
            if err_ms <= tol_ms:
                div_of[m] = N
                break

    # 2) 칸에 배치 (마디 끝으로 반올림되면 다음 마디 0번 칸으로 넘김)
    placed: list[tuple[int, int, int, float]] = []          # (마디, 레인, 칸, 원래 시각)
    for m, items in sorted(by_meas.items()):
        N = div_of[m]
        for t, lane, f in items:
            s, mm = int(round(f * N)), m
            if s >= N:
                s, mm = 0, m + 1
            placed.append((mm, lane, s, t))
    for mm, _, _, _ in placed:
        div_of.setdefault(mm, divs[0])

    grid: dict[tuple[int, int], set[int]] = {}
    quant: list[tuple[float, int, float]] = []
    dropped = 0
    for mm, lane, s, t in placed:
        slots = grid.setdefault((mm, lane), set())
        if s in slots:
            dropped += 1
            continue
        slots.add(s)
        quant.append(((mm + s / div_of[mm]) * meas_dur, lane, t))
    return grid, div_of, quant, dropped


def write_bms(path, times, lanes, bpm: float, title: str = "AI chart", artist: str = "",
              bgm_wav: str = "song.wav", key_wav: str = "silence.wav", level: int = 0,
              tol_ms: float = 10.0, divs=DIVS, rank: int = 2, total: float = 300.0, subtitle: str = ""):
    """BMS 파일을 쓰고 (quant, stats)를 돌려준다"""
    grid, div_of, quant, dropped = quantize(times, lanes, bpm, tol_ms, divs)
    n_meas = max(div_of) + 1
    if n_meas > 999:
        raise ValueError(f"마디가 {n_meas}개라 BMS 한도(999)를 넘어요. BPM이 너무 낮거나 곡이 너무 길어요.")

    lines = [
        "*---------------------- HEADER FIELD", "",
        "#PLAYER 1", "#GENRE AI Generated", f"#TITLE {_clean(title)}"]
    if str(subtitle).strip():
        lines.append(f"#SUBTITLE {_clean(subtitle)}")
    lines += [f"#ARTIST {_clean(artist)}".rstrip(),
        f"#BPM {bpm:g}", f"#PLAYLEVEL {int(level)}", f"#RANK {int(rank)}", f"#TOTAL {float(total):g}", "#LNTYPE 1", "",
        f"#WAV{BGM_ID} {_clean(bgm_wav)}", f"#WAV{KEY_ID} {_clean(key_wav)}", "",
        "*---------------------- MAIN DATA FIELD", "",
        f"#000{BGM_ID}:{BGM_ID}",                       # 곡 전체 오디오: 0번 마디 맨 앞
    ]
    for mm in sorted(div_of):
        N = div_of[mm]
        for lane in sorted({ln for (m2, ln) in grid if m2 == mm}, key=lambda l: LANE_TO_CH[l]):
            toks = ["00"] * N
            for s in grid[(mm, lane)]:
                toks[s] = KEY_ID
            lines.append(f"#{mm:03d}{LANE_TO_CH[lane]}:{''.join(toks)}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932", errors="replace"))

    errs = [abs(qt - ot) * 1000.0 for qt, _, ot in quant]
    stats = {
        "n_in": len(times), "n_out": len(quant), "dropped_dup": dropped,
        "max_err_ms": max(errs) if errs else 0.0,
        "mean_err_ms": sum(errs) / len(errs) if errs else 0.0,
        "div_hist": dict(sorted(Counter(div_of[m] for m in {mm for mm, _ in grid}).items())),
        "n_measures": n_meas,
    }
    return quant, stats


def verify_written(path, quant, tol_sec: float = 1e-6) -> dict:
    """쓴 BMS를 파서로 다시 읽어, 노트 시각/레인이 quant와 일치하는지 확인."""
    chart = parse_bms(path)
    got = sorted((n.lane, n.time) for n in chart.notes)
    exp = sorted((lane, qt) for qt, lane, _ in quant)
    ok = len(got) == len(exp) and all(
        gl == el and abs(gt - et) <= tol_sec for (gl, gt), (el, et) in zip(got, exp))
    max_diff = max((abs(gt - et) for (_, gt), (_, et) in zip(got, exp)), default=0.0) * 1000.0
    return {"ok": ok, "n_expected": len(exp), "n_parsed": len(got), "max_diff_ms": max_diff,
            "warnings": chart.warnings}
