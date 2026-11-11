"""
[모델, STEP 6] 스크래치 자리 모델: 소리(멜)와 리듬 맥락으로 '사람이 스크래치를 놓을 만한 줄'의 확률을 내고, 그 순서로 스크래치를 놓는다
(--scratch-model, --scratch-place)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FPS = 100
BANDS = (0.0, 150.0, 1000.0, 5000.0, 1e9)
MEL_FEATS = [f"{k}_{b}" for k in ("lvl", "flux", "pre", "post") for b in ("low", "lmid", "hmid", "high")] + ["tail_high"]
RHY = ["pos", "gap_prev", "gap_next", "dens_b", "dens_a", "ddens"]
REL = ["lrel", "frel", "trel", "jump", "hrel", "lowrel"]
ORDER = RHY + ["n_" + c for c in MEL_FEATS] + REL + ["keys"]


def mel_centers(n_mels: int, fmin: float = 30.0, fmax: float = 16000.0) -> np.ndarray:
    m = np.linspace(2595.0 * np.log10(1 + fmin / 700.0), 2595.0 * np.log10(1 + fmax / 700.0), n_mels + 2)
    return (700.0 * (10 ** (m / 2595.0) - 1))[1:-1]


def _win_mean(x, f, a, b):
    T = len(x)
    cs = np.vstack([np.zeros((1, x.shape[1]), np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    lo, hi = np.clip(f + a, 0, T), np.clip(f + b, 0, T)
    return ((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)[:, None]).astype(np.float32)


def mel_row_features(mel: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """mel (T, n_mels, 0~1), 줄의 멜 프레임 -> (n, 17)  (scratch_features.py와 같은 계산)"""
    m = np.asarray(mel, dtype=np.float32)
    T = len(m)
    hz = mel_centers(m.shape[1])
    groups = [(hz >= BANDS[i]) & (hz < BANDS[i + 1]) for i in range(4)]
    lv = np.stack([m[:, g].mean(1) for g in groups], 1)
    d = np.zeros_like(m)
    d[1:] = np.maximum(m[1:] - m[:-1], 0.0)
    fx = np.stack([d[:, g].sum(1) for g in groups], 1)
    f = np.clip(np.asarray(frames, dtype=np.int64), 0, max(T - 1, 0))
    pk_l = np.max(np.stack([lv[np.clip(f + k, 0, T - 1)] for k in range(0, 4)]), 0)
    pk_f = np.max(np.stack([fx[np.clip(f + k, 0, T - 1)] for k in range(-2, 3)]), 0)
    return np.concatenate([pk_l, pk_f, _win_mean(lv, f, -12, -2), _win_mean(lv, f, 10, 50),
                           _win_mean(lv[:, 3:4], f, 50, 100)], 1).astype(np.float32)


def feature_table(mel_feats: np.ndarray, t_chart: np.ndarray, bpm: float, keys: np.ndarray) -> np.ndarray:
    """학습 때(sf_eval)와 같은 순서(ORDER)의 특징 행렬"""
    F = np.asarray(mel_feats, dtype=np.float64)
    T = np.asarray(t_chart, dtype=np.float64)
    n = len(T)
    B = 60.0 / bpm
    gp = np.minimum(np.r_[99.0, np.diff(T)] / B, 8)
    gn = np.minimum(np.r_[np.diff(T), 99.0] / B, 8)
    q = (T / B) % 4
    qi = np.round(q * 4) % 16
    on = np.abs(q * 4 - np.round(q * 4)) < 0.15
    pos = np.where(~on, 5, np.where(qi == 0, 0, np.where(qi == 8, 1, np.where(qi % 4 == 0, 2, np.where(qi % 2 == 0, 3, 4)))))
    i = np.arange(n)
    lo = np.searchsorted(T, T - 2 * B)
    hi = np.searchsorted(T, T + 2 * B, side="right")
    db, da = i - lo, hi - i - 1
    loud = F[:, 0:4].mean(1)
    flux = F[:, 4:8].sum(1)
    lrel = np.array([loud[k] - np.median(loud[lo[k]:hi[k]]) for k in i])
    frel = np.log(np.array([flux[k] / (np.median(flux[lo[k]:hi[k]]) + 1e-3) for k in i]) + 1e-3)
    trel = F[:, 16] - np.median(F[:, 16])
    jump = F[:, 12:16].mean(1) - F[:, 8:12].mean(1)
    hrel = F[:, 3] - np.median(F[:, 3])
    lowrel = F[:, 0] - np.median(F[:, 0])
    norm = np.empty_like(F)
    for c in range(F.shape[1]):
        med = np.median(F[:, c])
        norm[:, c] = np.log((F[:, c] + 1e-3) / (med + 1e-3)) if 4 <= c < 8 else F[:, c] - med
    cols = [pos, gp, gn, db, da, da - db] + [norm[:, c] for c in range(17)] + [lrel, frel, trel, jump, hrel, lowrel,
                                                                              np.asarray(keys, np.float64)]
    return np.stack(cols, 1).astype(np.float32)


class TreeModel:
    """sklearn HistGradientBoostingClassifier를 numpy 배열로 옮긴 것 (scratch_pos.npz)"""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.feat, self.thr = z["feat"], z["thr"]
        self.left, self.right = z["left"], z["right"]
        self.leaf, self.value, self.mgl = z["leaf"].astype(bool), z["value"], z["mgl"].astype(bool)
        self.start = z["start"]
        self.base = float(z["base"])
        self.order = [str(x) for x in z["order"]]
        if self.order != ORDER:
            raise ValueError("스크래치 모델의 특징 순서가 코드와 달라요 (scratch_pos.npz를 이 코드와 같은 판으로)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        raw = np.full(n, self.base)
        rows = np.arange(n)
        for s in self.start[:-1]:
            node = np.full(n, s, np.int64)
            for _ in range(64):
                act = ~self.leaf[node]
                if not act.any():
                    break
                nd = node[act]
                x = X[rows[act], self.feat[nd]]
                go_left = np.where(np.isnan(x), self.mgl[nd], x <= self.thr[nd])
                node[act] = s + np.where(go_left, self.left[nd], self.right[nd])
            raw += self.value[node]
        return 1.0 / (1.0 + np.exp(-raw))


def export_sklearn(model, path: str | Path) -> None:
    """학습 쪽(sklearn 있는 곳)에서만 씀: HistGradientBoostingClassifier -> npz"""
    feat, thr, left, right, leaf, value, mgl, start = [], [], [], [], [], [], [], [0]
    for it in model._predictors:
        nodes = it[0].nodes
        feat.append(nodes["feature_idx"]), thr.append(nodes["num_threshold"])
        left.append(nodes["left"]), right.append(nodes["right"]), leaf.append(nodes["is_leaf"])
        value.append(nodes["value"]), mgl.append(nodes["missing_go_to_left"])
        start.append(start[-1] + len(nodes))
    np.savez_compressed(path, feat=np.concatenate(feat).astype(np.int16), thr=np.concatenate(thr).astype(np.float64),
                        left=np.concatenate(left).astype(np.int32), right=np.concatenate(right).astype(np.int32),
                        leaf=np.concatenate(leaf).astype(np.uint8), value=np.concatenate(value).astype(np.float64),
                        mgl=np.concatenate(mgl).astype(np.uint8), start=np.array(start, np.int64),
                        base=np.array(float(np.ravel(model._baseline_prediction)[0])), order=np.array(ORDER))


def place_scratch(t_chart, frames, masks, mel, bpm: float, model: TreeModel, target: int,
                  strength: float = 1.0, seed: int = 0, min_gap_beats: float = 0.25):
    """확률이 높은 줄부터 스크래치를 target개 놓는다
    반환: (masks, 확률, 보고)"""
    masks = [int(m) for m in masks]
    n = len(masks)
    if n == 0 or target <= 0 and not any(m & 1 for m in masks):
        return masks, np.zeros(n), {}
    t = np.asarray(t_chart, dtype=np.float64)
    keys = np.array([bin(m & 0xFE).count("1") for m in masks])
    X = feature_table(mel_row_features(mel, frames), t, bpm, keys)
    p = model.predict(X)
    rng = np.random.default_rng(seed + 223)
    had = np.array([m & 1 for m in masks], bool)
    fixed = had & ((keys == 0) | (rng.random(n) >= strength))         # 다시 정하지 않고 남기는 스크래치
    B = 60.0 / bpm
    chosen = list(np.flatnonzero(fixed))
    taken = np.sort(t[chosen]) if chosen else np.zeros(0)
    for i in np.argsort(-p, kind="stable"):
        if len(chosen) >= target:
            break
        if fixed[i] or keys[i] == 0:
            continue
        j = np.searchsorted(taken, t[i])
        if (j > 0 and t[i] - taken[j - 1] < min_gap_beats * B - 1e-6) or \
                (j < len(taken) and taken[j] - t[i] < min_gap_beats * B - 1e-6):
            continue
        chosen.append(int(i))
        taken = np.insert(taken, j, t[i])
    sel = np.zeros(n, bool)
    sel[chosen] = True
    for i in range(n):
        masks[i] = (masks[i] | 1) if sel[i] else (masks[i] & ~1 if keys[i] > 0 else masks[i])
    pos = X[:, 0].astype(int)
    rep = {"before": int(had.sum()), "after": int(sel.sum()), "kept": int((had & sel).sum()),
           "pos": {k: float(np.mean(pos[sel] == k)) if sel.any() else 0.0 for k in range(6)}}
    return masks, p, rep
