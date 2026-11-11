"""
[평가, 생성에서는 봉우리 고르기에 씀] 노트 타이밍 평가 (numpy만 사용, torch 불필요)

모델이 내놓은 프레임별 확률 -> 피크 검출 -> 노트 시각 -> 정답과 비교(F1)

용어:
    프레임 = 10ms.  tol=5 -> ±50ms,  tol=2 -> ±20ms 안에 있으면 맞은 것으로 센다.
    micro F1 = 채널(레인)별 TP/FP/FN을 전부 합산한 뒤 계산.
    union F1 = 레인을 무시하고 "어떤 레인이든 노트가 있는 시각"만 비교 (노트 시각 모델 지표).
"""
from __future__ import annotations

import numpy as np


def peak_pick(prob: np.ndarray, threshold: float, width: int = 1) -> np.ndarray:
    """prob[T] -> 노트 프레임 인덱스(오름차순)"""
    T = len(prob)
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    padded = np.pad(prob, (width, width), constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * width + 1)
    is_peak = (prob >= threshold) & (prob >= windows.max(axis=1))
    idx = np.flatnonzero(is_peak)
    # 완전히 평평한 구간에서 같은 값이 연속으로 잡히는 경우: 앞의 것만 남김
    keep: list[int] = []
    for i in idx:
        if not keep or i - keep[-1] > width:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def match_count(pred: np.ndarray, true: np.ndarray, tol: int) -> int:
    """정렬된 두 시각 배열에서 |차이| <= tol 인 쌍을 일대일로 최대한 맞춘 개수(TP)."""
    i = j = tp = 0
    while i < len(pred) and j < len(true):
        d = int(pred[i]) - int(true[j])
        if abs(d) <= tol:
            tp += 1
            i += 1
            j += 1
        elif d < 0:      # 예측이 정답보다 tol 넘게 앞 -> 오검출
            i += 1
        else:            # 정답이 예측보다 tol 넘게 앞 -> 놓침
            j += 1
    return tp


def prf(tp: int, n_pred: int, n_true: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate_charts(probs: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                    thresholds, tols=(5, 2), width: int = 1) -> list[dict]:
    """패턴별 (C, T) 확률/정답 -> 임계값마다 F1 요약
    반환: 임계값마다 dict (thr, n_pred, n_true, f1_t5, ...)"""
    tols = tuple(tols)
    # 정답 시각은 임계값과 무관하므로 미리 뽑아 둔다
    true_ch = {cid: [np.flatnonzero(l[c]) for c in range(l.shape[0])] for cid, l in labels.items()}
    true_un = {cid: np.flatnonzero(l.max(axis=0) > 0) for cid, l in labels.items()}

    results = []
    for thr in thresholds:
        tp = {t: 0 for t in tols}
        un_tp = {t: 0 for t in tols}
        n_pred = n_true = un_pred = un_true = 0
        for cid, p in probs.items():
            peaks = [peak_pick(p[c], thr, width) for c in range(p.shape[0])]
            for c, pk in enumerate(peaks):
                n_pred += len(pk)
                n_true += len(true_ch[cid][c])
                for t in tols:
                    tp[t] += match_count(pk, true_ch[cid][c], t)
            un_pk = peak_pick(p.max(axis=0), thr, width)
            un_pred += len(un_pk)
            un_true += len(true_un[cid])
            for t in tols:
                un_tp[t] += match_count(un_pk, true_un[cid], t)

        row = {"thr": round(float(thr), 3), "n_pred": n_pred, "n_true": n_true,
               "union_pred": un_pred, "union_true": un_true}
        for t in tols:
            pp, rr, ff = prf(tp[t], n_pred, n_true)
            row.update({f"p_t{t}": pp, f"r_t{t}": rr, f"f1_t{t}": ff})
            up, ur, uf = prf(un_tp[t], un_pred, un_true)
            row.update({f"union_p_t{t}": up, f"union_r_t{t}": ur, f"union_f1_t{t}": uf})
        results.append(row)
    return results


def best_by(results: list[dict], key: str = "union_f1_t5") -> dict:
    return max(results, key=lambda r: r[key])
