"""
[규칙, STEP 3] 확신이 약해서 빠진 32분, 24분 노트를 되살린다 (--fill32)

임계값보다 낮지만 low 이상인 봉우리 중, 앞뒤 확실한 노트의 한가운데에 있는 것만 더한다.
--fill32-sound: 분리 소리가 있으면 그 자리에 어택이 있을 때만 더한다.
"""
from __future__ import annotations

import numpy as np


LAST_SKIP = 0      # 분리 소리 어택이 없어 안 더한 수 (마지막 호출)


def fill32(probs: np.ndarray, frames: np.ndarray, bpm: float, low: float = 0.25, sure: float = 0.6,
           fps: int = 100, stems=None, ratio: float = 0.0) -> tuple[np.ndarray, int]:
    """16분 두 노트 사이 -> 32분, 8분 셋잇단(12분) 두 노트 사이 -> 24분 (16비트 이상 채우기, 48분은 안 함)."""
    global LAST_SKIP
    LAST_SKIP = 0
    from metrics import peak_pick
    p = np.asarray(probs, dtype=np.float32)
    kept = np.unique(np.asarray(frames, dtype=np.int64))
    if len(kept) < 2 or low <= 0:
        return kept, 0
    beat = 60.0 / bpm * fps
    gaps = ((beat / 4, 0.15), (beat / 3, 0.1))       # (16분, 허용), (8분 셋잇단, 허용) 길이(프레임) - 16분 기준은 예전과 같음
    cand = np.setdiff1d(peak_pick(p, low, width=1), kept)
    add = []
    for c in cand:
        j = int(np.searchsorted(kept, c))
        if j == 0 or j >= len(kept):
            continue
        a, b = kept[j - 1], kept[j]
        if p[a] < sure or p[b] < sure:
            continue
        for g, tol in gaps:
            if abs((b - a) - g) <= tol * g and abs(c - (a + b) / 2) <= 0.2 * g:
                add.append(int(round((a + b) / 2)))   # 봉우리 시각이 아니라 두 노트 한가운데에 둔다
                break
    if add and stems is not None and ratio > 0:
        from stem_follow import onset_at
        fa = np.asarray(add, dtype=np.int64)
        ok = np.zeros(len(fa), dtype=bool)
        for k_ in ("drums", "bass", "other", "vocals"):
            if k_ in stems:
                ok |= onset_at(np.asarray(stems[k_], dtype=np.float32), fa, ratio=ratio)[0]
        LAST_SKIP = int((~ok).sum())
        add = [int(x) for x in fa[ok]]
    if not add:
        return kept, 0
    return np.unique(np.r_[kept, np.array(add, dtype=np.int64)]), len(add)
