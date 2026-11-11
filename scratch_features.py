"""
[학습 준비] 사람 패턴의 줄마다 '그 순간 소리 특징'(전처리된 멜에서)과 스크래치 여부를 뽑아 작은 파일로 만든다

사용:
    python scratch_features.py --processed data/processed --out scratch_features.zip
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
import io
import time
import zipfile
from pathlib import Path

import numpy as np

FPS = 100
BANDS = (0.0, 150.0, 1000.0, 5000.0, 1e9)
FEATS = [f"{k}_{b}" for k in ("lvl", "flux", "pre", "post") for b in ("low", "lmid", "hmid", "high")] + ["tail_high"]


def mel_centers(n_mels: int, fmin: float = 30.0, fmax: float = 16000.0) -> np.ndarray:
    m = np.linspace(2595.0 * np.log10(1 + fmin / 700.0), 2595.0 * np.log10(1 + fmax / 700.0), n_mels + 2)
    return (700.0 * (10 ** (m / 2595.0) - 1))[1:-1]


def _win_mean(x: np.ndarray, f: np.ndarray, a: int, b: int) -> np.ndarray:
    """x (T, k)의 프레임 f+a ~ f+b-1 평균 (구간이 곡 밖이면 잘라서)"""
    T = len(x)
    cs = np.vstack([np.zeros((1, x.shape[1]), np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    lo = np.clip(f + a, 0, T)
    hi = np.clip(f + b, 0, T)
    n = np.maximum(hi - lo, 1)[:, None]
    return ((cs[hi] - cs[lo]) / n).astype(np.float32)


def row_features(mel: np.ndarray, times: np.ndarray) -> np.ndarray:
    """mel (T, n_mels, 0~1), 줄 시각(초) -> (n, 17)"""
    m = np.asarray(mel, dtype=np.float32)
    T = len(m)
    hz = mel_centers(m.shape[1])
    groups = [(hz >= BANDS[i]) & (hz < BANDS[i + 1]) for i in range(4)]
    lv = np.stack([m[:, g].mean(1) for g in groups], 1)                          # (T, 4) 대역 평균 크기
    d = np.zeros_like(m)
    d[1:] = np.maximum(m[1:] - m[:-1], 0.0)
    fx = np.stack([d[:, g].sum(1) for g in groups], 1)                           # (T, 4) 대역 상승량
    f = np.clip(np.rint(np.asarray(times) * FPS).astype(np.int64), 0, max(T - 1, 0))
    pk_l = np.max(np.stack([lv[np.clip(f + k, 0, T - 1)] for k in range(0, 4)]), 0)
    pk_f = np.max(np.stack([fx[np.clip(f + k, 0, T - 1)] for k in range(-2, 3)]), 0)
    pre = _win_mean(lv, f, -12, -2)
    post = _win_mean(lv, f, 10, 50)
    tail = _win_mean(lv[:, 3:4], f, 50, 100)
    return np.concatenate([pk_l, pk_f, pre, post, tail], 1).astype(np.float32)


def one(chart_id: str, src: str, mel_path: Path):
    from bms_parser import parse_bms
    c = parse_bms(Path(src))
    rows: dict[float, list[int]] = {}
    for n in c.notes:
        r = rows.setdefault(round(n.time, 3), [0, 0])
        if n.lane == 0:
            r[0] = 1
        else:
            r[1] += 1
    t = np.array(sorted(rows), dtype=np.float64)
    if len(t) == 0:
        return None
    mel = np.load(mel_path)
    feats = row_features(mel, t)
    scr = np.array([rows[x][0] for x in t], np.int8)
    keys = np.array([rows[x][1] for x in t], np.int8)
    return t.astype(np.float32), scr, keys, feats.astype(np.float16), float(c.init_bpm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="scratch_features.zip")
    a = ap.parse_args()
    proc = Path(a.processed)
    index = proc / "index.csv"
    if not index.exists():
        raise SystemExit(f"{index} 가 없어요. --processed 경로를 확인해 주세요.")
    with open(index, encoding="utf-8-sig", newline="") as fp:
        rows = [{k.strip().lstrip("﻿"): v for k, v in r.items() if k} for r in csv.DictReader(fp)]
    ids, T, S, K, F, C, BPM = [], [], [], [], [], [], []
    miss = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cid, src = r.get("chart_id", ""), r.get("source", "")
        mp = proc / f"{cid}.mel.npy"
        if not (cid and src and Path(src).exists() and mp.exists()):
            miss.append(cid or src)
            continue
        try:
            res = one(cid, src, mp)
        except Exception as e:  # noqa: BLE001
            miss.append(f"{cid} ({e!r})")
            continue
        if res is None:
            continue
        t, s, k, f, bpm = res
        ids.append(cid)
        T.append(t), S.append(s), K.append(k), F.append(f), C.append(np.full(len(t), len(ids) - 1, np.int16))
        BPM.append(bpm)
        if i % 20 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] {cid}  (지난 {(time.time() - t0) / 60:.1f}분)")
    buf = io.BytesIO()
    np.savez_compressed(buf, chart_ids=np.array(ids), bpm=np.array(BPM, np.float32), chart=np.concatenate(C),
                        t=np.concatenate(T), scr=np.concatenate(S), keys=np.concatenate(K),
                        feats=np.concatenate(F), feat_names=np.array(FEATS))
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        z.writestr("scratch_features.npz", buf.getvalue())
        z.writestr("missing.txt", "\n".join(miss))
    print(f"끝: 차분 {len(ids)}개, 줄 {sum(len(x) for x in T)}개 -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f}MB)"
          + (f", 못 읽은 차분 {len(miss)}개 (missing.txt)" if miss else ""))


if __name__ == "__main__":
    main()
