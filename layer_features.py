"""
[학습 준비] 학습 패턴이 쓴 키음 파일마다 소리 특징(크기, 대역, 어택)을 재서 작은 CSV로 만든다

사용:
    python layer_features.py --processed data/processed --out layer_features.zip --workers 3
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import argparse
import csv
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

COLS = ["chart_id", "wav_id", "name", "found", "dur", "peak_db", "on_db", "body_db",
        "f_low", "f_lmid", "f_hmid", "f_high", "centroid", "t_peak_ms"]


def _read_head(path: Path, max_sec: float = 3.0):
    """앞부분 max_sec초만 모노로 읽기 -> (samples, sr, 전체 길이 초). soundfile 없으면 WAV만."""
    try:
        import soundfile as sf
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            n = f.frames
            data = f.read(frames=min(n, int(sr * max_sec)), dtype="float32", always_2d=True)
        return data.mean(axis=1), sr, n / sr
    except ImportError:
        pass
    from scipy.io import wavfile
    sr, raw = wavfile.read(str(path))
    if raw.dtype == np.int16:
        x = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        x = raw.astype(np.float32) / 2147483648.0
    elif raw.dtype == np.uint8:
        x = (raw.astype(np.float32) - 128.0) / 128.0
    else:
        x = raw.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x[: int(sr * max_sec)], sr, len(raw) / sr


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-9)))


def features(path: Path) -> dict:
    x, sr, dur = _read_head(path)
    n100 = max(1, int(0.1 * sr))
    head = np.zeros(n100, np.float32)
    head[: min(n100, len(x))] = x[:n100]
    body = x[: int(sr * 1.0)] if len(x) else np.zeros(1, np.float32)
    spec = np.abs(np.fft.rfft(head * np.hanning(n100))) ** 2
    fr = np.fft.rfftfreq(n100, 1.0 / sr)
    tot = spec.sum() + 1e-12
    bands = [spec[fr < 150].sum(), spec[(fr >= 150) & (fr < 1000)].sum(),
             spec[(fr >= 1000) & (fr < 5000)].sum(), spec[fr >= 5000].sum()]
    hop = max(1, int(0.005 * sr))
    seg = x[: int(0.3 * sr)]
    if len(seg) >= hop:
        env = np.sqrt((seg[: len(seg) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))
        t_peak = float(np.argmax(env) * 5.0)
    else:
        t_peak = 0.0
    return {"found": 1, "dur": round(dur, 4),
            "peak_db": round(_db(float(np.abs(x).max()) if len(x) else 0.0), 2),
            "on_db": round(_db(float(np.sqrt((head ** 2).mean()))), 2),
            "body_db": round(_db(float(np.sqrt((body ** 2).mean()))), 2),
            "f_low": round(bands[0] / tot, 4), "f_lmid": round(bands[1] / tot, 4),
            "f_hmid": round(bands[2] / tot, 4), "f_high": round(bands[3] / tot, 4),
            "centroid": round(float((fr * spec).sum() / tot), 1), "t_peak_ms": t_peak}


def one_chart(chart_id: str, src: str) -> tuple[str, list[dict], list[str]]:
    from bms_parser import parse_bms
    from render_audio import build_file_index
    src_p = Path(src)
    c = parse_bms(src_p)
    used = {w for _, w in c.sounds} | {n.wav for n in c.notes}
    idx = build_file_index(src_p.parent)
    rows, errs = [], []
    for w in sorted(used):
        name = c.wav_files.get(w)
        row = {"chart_id": chart_id, "wav_id": w, "name": name or "", "found": 0}
        p = idx.get(Path(name).stem.lower()) if name else None
        if p is not None:
            try:
                row.update(features(p))
            except Exception as e:  # noqa: BLE001
                errs.append(f"{chart_id}\t{name}\t{e!r}")
        rows.append(row)
    return chart_id, rows, errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed", help="index.csv가 있는 폴더")
    ap.add_argument("--out", default="layer_features.zip")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    index = Path(a.processed) / "index.csv"
    if not index.exists():
        raise SystemExit(f"{index} 가 없어요. --processed 경로를 확인해 주세요.")
    with open(index, encoding="utf-8-sig", newline="") as fp:
        rows = [{k.strip().lstrip("﻿"): v for k, v in r.items() if k} for r in csv.DictReader(fp)]
    jobs = []
    for r in rows:
        src = r.get("source", "")
        cid = r.get("chart_id") or f"{Path(src).parent.name}__{Path(src).stem}"
        if src and Path(src).exists():
            jobs.append((cid, src))

    part = Path(a.out).with_name("layer_features_partial.csv")
    done = set()
    if part.exists():
        with open(part, encoding="utf-8", newline="") as fp:
            done = {r["chart_id"] for r in csv.DictReader(fp)}
    todo = [j for j in jobs if j[0] not in done]
    print(f"차분 {len(jobs)}개 중 남은 것 {len(todo)}개 (이미 끝낸 것 {len(done)}개), workers {a.workers}")

    errs_all: list[str] = []
    t0 = time.time()
    new_file = not part.exists()
    with open(part, "a", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=COLS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = {ex.submit(one_chart, cid, src): cid for cid, src in todo}
            for k, f in enumerate(as_completed(futs), 1):
                cid = futs[f]
                try:
                    _, rs, errs = f.result()
                    w.writerows(rs)
                    fp.flush()
                    errs_all += errs
                    miss = sum(1 for r in rs if not r["found"])
                    note = f", 못 찾거나 못 읽은 키음 {miss}" if miss else ""
                except Exception as e:  # noqa: BLE001
                    errs_all.append(f"{cid}\t(차분 전체)\t{e!r}")
                    note = " 실패"
                el = time.time() - t0
                eta = el / k * (len(todo) - k)
                print(f"[{k}/{len(todo)}] {cid}{note}  (지난 {el / 60:.1f}분, 남은 약 {eta / 60:.1f}분)")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(part, "layer_features.csv")
        z.writestr("errors.txt", "\n".join(errs_all))
    print(f"끝: {a.out} ({Path(a.out).stat().st_size / 1e6:.1f}MB), 읽기 실패 키음 {len(errs_all)}개")
    print("이 zip 파일만 보내 주면 돼요. (다시 처음부터 재려면 layer_features_partial.csv를 지우고 실행)")


if __name__ == "__main__":
    main()
