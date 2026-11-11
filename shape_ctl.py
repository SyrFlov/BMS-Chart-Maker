"""
[규칙, STEP 6] --shape-run: 같은 동시치기 모양 이어짐 끊기

자리 모델 뒤에서 (1) 같은 모양 3줄 이상 사슬은 셋째 줄부터 모양을 바꿔 끊고,
(2) 앞 줄과 같은 모양 비율이 사람 상위 25% 곡(21%)보다 높으면 넘친 만큼 바꾼다. 건반 수와 스크래치는 그대로.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

SAME_P75 = 0.21          # 앞 줄과 같은 모양 (사람 곡별 75%)
CHAIN_P50 = 9.6          # 3줄+ 사슬 / 폭타 1000줄 (사람 곡별 중앙값)


def _keys(m: int) -> list[int]:
    return [k for k in range(1, 8) if (int(m) >> k) & 1]


def _shape(m: int):
    ks = _keys(m)
    return tuple(k - ks[0] for k in ks) if ks else ()


def _stream_pairs(t_chart, masks, bpm: float, min_fill: int = 12):
    """폭타 마디(16분 칸 min_fill+/16) 안 16분 줄의 (16분 번호 -> 줄 번호), 폭타 16분 줄 수"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q = t / B * 4.0
    on = np.abs(q - np.round(q)) < 1e-3
    qi = np.round(q).astype(np.int64)
    pos = {}
    for i in np.flatnonzero(on):
        if int(masks[i]) & 0xFE:
            pos[int(qi[i])] = int(i)
    meas = {}
    for x in pos:
        meas[x // 16] = meas.get(x // 16, 0) + 1
    pos = {x: i for x, i in pos.items() if meas[x // 16] >= min_fill}
    return pos


def shape_stats(t_chart, masks, bpm: float):
    """(앞 줄과 같은 모양 비율, 3줄+ 사슬 / 폭타 1000줄, 사슬 목록[줄 번호 목록], 같은 모양 쌍 목록[(앞, 뒤)])"""
    pos = _stream_pairs(t_chart, masks, bpm)
    multi = lambda i: bin(int(masks[i]) & 0xFE).count("1") >= 2
    n = same = 0
    pairs, chains, cur = [], [], []
    for x in sorted(pos):
        i = pos[x]
        if (x - 1) in pos and multi(i) and multi(pos[x - 1]):
            n += 1
            if _shape(masks[i]) == _shape(masks[pos[x - 1]]):
                same += 1
                pairs.append((pos[x - 1], i))
                if not cur:
                    cur = [pos[x - 1]]
                cur.append(i)
                continue
        if len(cur) >= 3:
            chains.append(cur)
        cur = []
    if len(cur) >= 3:
        chains.append(cur)
    rate = same / n if n else 0.0
    per1k = len(chains) / max(len(pos), 1) * 1000.0
    return rate, per1k, chains, pairs, n, len(pos)


def _reshape_row(t, masks, i: int, avoid_shapes, jack_ms: float, rng) -> bool:
    """i줄 건반을 같은 수로, avoid_shapes와 다른 모양으로 (연타 없이). 바꿨으면 True"""
    from thick_ctl import _near_mask
    m = int(masks[i])
    old = _keys(m)
    k = len(old)
    if k < 2:
        return False
    near = _near_mask(t, masks, i, jack_ms + 5.0)
    c_old = float(np.mean(old))
    best, best_s = None, None
    for comb in combinations(range(1, 8), k):
        if any((near >> x) & 1 for x in comb):
            continue
        sh = tuple(x - comb[0] for x in comb)
        if sh in avoid_shapes:
            continue
        s = (k - len(set(comb) & set(old))) + 0.5 * abs(float(np.mean(comb)) - c_old) + 0.01 * rng.random()
        if best_s is None or s < best_s:
            best, best_s = comb, s
    if best is None:
        return False
    masks[i] = (m & 1) | sum(1 << x for x in best)
    return True


def shape_run_fix(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0):
    """반환: (masks, 보고 dict)"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    r0, c0, _, _, n0, ns = shape_stats(t, masks, bpm)
    rep = {"before": (r0, c0), "after": (r0, c0), "chain_rows": 0, "pair_rows": 0, "pairs": n0}
    if strength <= 0 or n0 < 20:
        return masks, rep
    rng = np.random.default_rng(seed + 1010)
    pos = _stream_pairs(t, masks, bpm)
    q_of = {i: x for x, i in pos.items()}

    def nb_shapes(i):
        x = q_of.get(i)
        out = set()
        for d in (-1, 1):
            j = pos.get(x + d) if x is not None else None
            if j is not None and bin(masks[j] & 0xFE).count("1") >= 2:
                out.add(_shape(masks[j]))
        return out

    chain_t = CHAIN_P50 / max(strength, 1e-3)
    same_t = SAME_P75 / max(strength, 1e-3) if strength < 1 else SAME_P75
    # (1) 3줄+ 사슬: 셋째 줄부터 (긴 사슬부터)
    for _ in range(400):
        r, c, chains, _, _, _ = shape_stats(t, masks, bpm)
        if c <= chain_t or not chains:
            break
        chains.sort(key=len, reverse=True)
        done = False
        for ch in chains:
            for i in ch[2::2]:
                if _reshape_row(t, masks, i, nb_shapes(i), jack_ms, rng):
                    rep["chain_rows"] += 1
                    done = True
                    break
            if done:
                break
        if not done:
            break
    # (2) 앞 줄과 같은 모양 비율이 사람 75% 곡보다 높으면 넘친 만큼 둘째 줄을
    for _ in range(2000):
        r, c, _, pairs, n, _ = shape_stats(t, masks, bpm)
        if r <= same_t or not pairs:
            break
        a, b = pairs[int(rng.integers(len(pairs)))]
        if _reshape_row(t, masks, b, nb_shapes(b), jack_ms, rng):
            rep["pair_rows"] += 1
        else:
            # 못 바꾸면 앞 줄을
            if _reshape_row(t, masks, a, nb_shapes(a), jack_ms, rng):
                rep["pair_rows"] += 1
    r1, c1, _, _, _, _ = shape_stats(t, masks, bpm)
    rep["after"] = (r1, c1)
    return masks, rep
