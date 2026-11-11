"""
[학습 준비, 생성에서는 오디오 읽기에 씀] BMS 패턴의 키음 이벤트를 합쳐서 곡 오디오(WAV)로 렌더링한다.

사용 예:
    python render_audio.py "data\\bms\\[SHIKI] Air\\air7god.bme" --out data\\audio
    -> data\\audio\\[SHIKI] Air\\air7god.wav          (곡 오디오, 학습용)
       data\\audio\\[SHIKI] Air\\air7god_clicks.wav   (노트 위치에 클릭음을 얹은 검증용)

검증용 파일을 들으면서 클릭음이 곡의 드럼/멜로디와 맞는지 확인하면 된다.
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import argparse
from pathlib import Path

import numpy as np

from bms_parser import Chart, parse_bms

try:
    import soundfile as sf
except ImportError:  # soundfile이 없으면 scipy로 WAV만 처리
    sf = None

AUDIO_EXTS = (".wav", ".ogg", ".mp3", ".flac")


# ---------------------------------------------------------------------------
# 입출력
# ---------------------------------------------------------------------------
def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """(samples[N, 2] float32, sample_rate). 모노는 스테레오로 복제."""
    if sf is not None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    else:
        from scipy.io import wavfile
        sr, raw = wavfile.read(str(path))
        if raw.dtype == np.int16:
            data = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            data = raw.astype(np.float32) / 2147483648.0
        elif raw.dtype == np.uint8:
            data = (raw.astype(np.float32) - 128.0) / 128.0
        else:
            data = raw.astype(np.float32)
        if data.ndim == 1:
            data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2], int(sr)


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        sf.write(str(path), data, sr, subtype="PCM_16")
    else:
        from scipy.io import wavfile
        wavfile.write(str(path), sr, (np.clip(data, -1, 1) * 32767).astype(np.int16))


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr_in, sr_out)
        return resample_poly(data, sr_out // g, sr_in // g, axis=0).astype(np.float32)
    except ImportError:  # 품질은 낮지만 동작은 하는 폴백
        n_out = int(round(len(data) * sr_out / sr_in))
        x_old = np.linspace(0, 1, len(data), endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        return np.stack([np.interp(x_new, x_old, data[:, c]) for c in range(2)], axis=1).astype(np.float32)


def build_file_index(folder: Path) -> dict[str, Path]:
    """stem(소문자) -> 실제 파일. #WAV가 .wav로 적혀 있어도 .ogg 등을 찾아준다."""
    idx: dict[str, Path] = {}
    files = sorted((p for p in folder.rglob("*") if p.is_file()),     # 하위 폴더(sound/ 등)도 포함, 얕은 쪽 우선
                   key=lambda q: len(q.relative_to(folder).parts))
    for ext in AUDIO_EXTS:  # 확장자 우선순위
        for p in files:
            if p.suffix.lower() == ext:
                idx.setdefault(p.stem.lower(), p)
    return idx


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------
def render_chart(chart: Chart, folder: Path, sr: int = 44100,
                 tail_sec: float = 2.0) -> tuple[np.ndarray, dict]:
    """키음 이벤트를 배치해 믹스다운. 반환: (audio[N,2] float32, 통계)."""
    idx = build_file_index(folder)
    cache: dict[int, np.ndarray | None] = {}
    stats = {"events": len(chart.sounds), "missing": 0, "failed": 0, "cut": 0}

    def load(wav_id: int) -> np.ndarray | None:
        if wav_id in cache:
            return cache[wav_id]
        name = chart.wav_files.get(wav_id)
        arr = None
        if name is not None:
            path = idx.get(Path(name).stem.lower())
            if path is not None:
                try:
                    data, file_sr = read_audio(path)
                    arr = resample(data, file_sr, sr)
                except Exception:  # noqa: BLE001  (깨진/미지원 포맷)
                    stats["failed"] += 1
        cache[wav_id] = arr
        return arr

    # 같은 WAV 번호가 다시 재생되면 이전 재생은 끊긴다 (BMS 플레이어 동작: 폴리포니 1)
    by_wav: dict[int, list[float]] = {}
    for t, w in chart.sounds:
        by_wav.setdefault(w, []).append(t)
    next_start: dict[tuple[int, float], float | None] = {}
    for w, ts in by_wav.items():
        ts.sort()
        for i, t in enumerate(ts):
            next_start[(w, t)] = ts[i + 1] if i + 1 < len(ts) else None

    total = int((chart.duration + tail_sec) * sr)
    out = np.zeros((total, 2), dtype=np.float32)
    fade = max(1, int(0.005 * sr))  # 끊을 때 5ms 페이드아웃

    for t, w in chart.sounds:
        samp = load(w)
        if samp is None:
            stats["missing"] += 1
            continue
        start = int(round(t * sr))
        seg = samp
        nxt = next_start[(w, t)]
        if nxt is not None:
            limit = int(round((nxt - t) * sr))
            if limit < len(seg):
                seg = seg[:limit].copy()
                if len(seg) > fade:
                    seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
                stats["cut"] += 1
        end = min(start + len(seg), total)
        if end > start:
            out[start:end] += seg[: end - start]

    peak = float(np.abs(out).max())
    stats["peak_before_norm"] = round(peak, 3)
    if peak > 0.98:
        out *= 0.98 / peak
    return out, stats


def make_click(sr: int, freq: float, ms: float = 25.0) -> np.ndarray:
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    c = np.sin(2 * np.pi * freq * t) * np.exp(-t * 120)
    return np.repeat(c[:, None].astype(np.float32), 2, axis=1)


def add_clicks(audio: np.ndarray, chart: Chart, sr: int) -> np.ndarray:
    """노트 시작 위치에 클릭음: 스크래치=낮은 음, 건반=높은 음."""
    out = audio * 0.6
    clicks = {0: make_click(sr, 900), 1: make_click(sr, 2400)}
    for n in chart.notes:
        c = clicks[0 if n.lane == 0 else 1]
        s = int(round(n.time * sr))
        e = min(s + len(c), len(out))
        if e > s:
            out[s:e] += 0.5 * c[: e - s]
    peak = float(np.abs(out).max())
    return out * (0.98 / peak) if peak > 0.98 else out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chart", help="차분 파일 경로 (.bms/.bme/.bml)")
    ap.add_argument("--out", default="data/audio", help="출력 폴더")
    ap.add_argument("--sr", type=int, default=44100, help="샘플레이트")
    ap.add_argument("--no-clicks", action="store_true", help="클릭 검증 파일 생략")
    args = ap.parse_args()

    chart_path = Path(args.chart)
    chart = parse_bms(chart_path)
    print(f"{chart.title}  노트 {len(chart.notes)}개, 키음 이벤트 {len(chart.sounds)}개, "
          f"{chart.duration:.1f}초")

    audio, stats = render_chart(chart, chart_path.parent, sr=args.sr)
    print(f"렌더링 완료: 누락 {stats['missing']}, 읽기 실패 {stats['failed']}, "
          f"재생 끊김 처리 {stats['cut']}, 정규화 전 피크 {stats['peak_before_norm']}")

    out_dir = Path(args.out) / chart_path.parent.name
    wav_path = out_dir / f"{chart_path.stem}.wav"
    write_wav(wav_path, audio, args.sr)
    print(f"저장: {wav_path}")

    if not args.no_clicks:
        click_path = out_dir / f"{chart_path.stem}_clicks.wav"
        write_wav(click_path, add_clicks(audio, chart, args.sr), args.sr)
        print(f"저장: {click_path}  <- 이걸 들으면서 클릭음이 곡과 맞는지 확인")


if __name__ == "__main__":
    main()
