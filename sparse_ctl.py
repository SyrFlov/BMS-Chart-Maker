"""
[규칙, STEP 5] 쉬는(성긴) 구간의 단타가 한 건반(특히 4번)에 몰리는 것을 푼다 (기본 켬, --no-sparse-spread로 끔)

크로노미아 v6 생성: 성긴 마디(8줄 이하)의 단타 143개 중 4번이 55개(38%). 고르게 쓰면 14% 정도.
레인 모델은 앞뒤 줄을 보고 레인을 고르는데, 성긴 구간은 앞뒤 줄이 멀어서 '무난한 가운데'로 쏠린다.

규칙: 앞뒤 줄과 모두 8분 이상 떨어진 단타(건반 1개, 스크래치 없음)가
      바로 앞 성긴 단타 2개 중 하나와 같은 건반이거나, 그 건반을 이미 고른 몫의 1.5배 넘게 썼으면
      다른 건반으로 옮긴다.
      새 건반은 앞뒤 줄의 건반을 피하고(연타 방지), 지금까지 성긴 단타에 덜 쓴 건반을 우선한다.
스트림이나 동시치기는 건드리지 않는다. numpy만 사용.
"""
from __future__ import annotations

import numpy as np


def _keys(m: int) -> list[int]:
    return [b for b in range(1, 8) if (int(m) >> b) & 1]


def spread_sparse_singles(t_chart, masks, bpm: float, seed: int = 0, history: int = 2):
    """반환: (새 masks 리스트, 옮긴 줄 수, 성긴 단타 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 3:
        return masks, 0, 0
    eighth = 60.0 / bpm / 2 - 1e-3
    rng = np.random.default_rng(seed + 7)
    used = np.zeros(8)
    recent: list[int] = []
    moved = total = 0
    for i in range(n):
        m = masks[i]
        ks = _keys(m)
        gp = t[i] - t[i - 1] if i > 0 else 1e9
        gn = t[i + 1] - t[i] if i < n - 1 else 1e9
        if len(ks) != 1 or (m & 1) or gp < eighth or gn < eighth:
            continue
        total += 1
        k = ks[0]
        over = used[k] > 1.5 * used[1:].sum() / 7 + 1        # 한 건반이 고른 몫의 1.5배를 넘으면
        if k in recent[-history:] or over:
            block = set(recent[-history:]) | (set(_keys(masks[i - 1])) if i > 0 else set()) \
                | (set(_keys(masks[i + 1])) if i < n - 1 else set())
            cands = [c for c in range(1, 8) if c not in block]
            if cands:
                w = 1.0 / (1.0 + used[cands])
                k = int(rng.choice(cands, p=w / w.sum()))
                masks[i] = (m & 1) | (1 << k)
                moved += 1
        used[k] += 1
        recent.append(k)
    return masks, moved, total


def key_share(masks, t_chart, bpm):
    """확인용: 성긴 단타의 건반별 비율 (1..7)"""
    t = np.asarray(t_chart, dtype=np.float64)
    eighth = 60.0 / bpm / 2 - 1e-3
    c = np.zeros(8)
    for i, m in enumerate(masks):
        ks = _keys(m)
        gp = t[i] - t[i - 1] if i > 0 else 1e9
        gn = t[i + 1] - t[i] if i < len(masks) - 1 else 1e9
        if len(ks) == 1 and not (m & 1) and gp >= eighth and gn >= eighth:
            c[ks[0]] += 1
    return c[1:] / max(c.sum(), 1)


if __name__ == "__main__":
    # 자체 점검: 성긴 구간에 4번만 20번 -> 고르게 퍼지고, 앞뒤 줄과 같은 건반(연타)은 없어야 함
    bpm = 150.0
    t = np.arange(20) * (60 / bpm)                     # 4분 간격 = 성긴 구간
    masks = [1 << 4] * 20
    new, moved, tot = spread_sparse_singles(t, masks, bpm)
    ks = [_keys(m)[0] for m in new]
    print("전:", [4] * 20, "\n후:", ks, f"\n옮김 {moved}/{tot}, 건반 비율 {np.round(key_share(new, t, bpm), 2)}")
    assert all(a != b for a, b in zip(ks, ks[1:])), "연속 같은 건반이 남았어요"
    assert max(np.bincount(ks)) <= 6
    # 스트림(16분)은 건드리지 않음
    t2 = np.arange(16) * (60 / bpm / 4)
    new2, moved2, _ = spread_sparse_singles(t2, [1 << 4, 1 << 2] * 8, bpm)
    assert moved2 == 0 and new2 == [1 << 4, 1 << 2] * 8
    print("OK")



def stream_end_chords(t_chart, masks, bpm: float, plus: int = 1, seed: int = 0, max_notes: int = 6,
                      jack_ms: float = 100.0):
    """16분 연타(3줄 이상)가 끝나고 8분 이상 쉬는 마지막 줄을 바로 앞 줄보다 plus개 이상 많은 동시치기로 (--stream-end-chord)
    반환: (새 masks, 바꾼 줄 수, 연타 끝 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if plus <= 0 or n < 4:
        return masks, 0, 0
    B = 60.0 / bpm
    s16, s8 = B / 4 * 1.15, B / 2 * 0.95
    rng = np.random.default_rng(seed + 11)
    changed = ends = 0
    for i in range(2, n):
        gn = t[i + 1] - t[i] if i < n - 1 else 1e9
        if not (t[i] - t[i - 1] <= s16 and t[i - 1] - t[i - 2] <= s16 and gn >= s8):
            continue
        ends += 1
        m = masks[i]
        need = min(bin(masks[i - 1] & 0xFE).count("1") + plus, max_notes)   # 건반만 셈 (스크래치는 상한에 안 넣음)
        if bin(m & 0xFE).count("1") >= need:
            continue
        keys = _keys(m)
        near_next = i < n - 1 and gn * 1000.0 < jack_ms
        block = int(masks[i - 1]) | (int(masks[i + 1]) if near_next else 0) | m
        while bin(m & 0xFE).count("1") < need:
            cands = [c for c in range(1, 8) if not (block >> c) & 1]
            if not cands:
                break                                     # 연타를 피할 자리가 없으면 거기까지만
            if keys:
                d = np.array([min(abs(c - k) for k in keys) for c in cands], dtype=np.float64)
                w = np.where(d >= 2, d, 0.1)
            else:
                w = np.ones(len(cands))
            c = int(rng.choice(cands, p=w / w.sum()))
            m |= 1 << c
            block |= 1 << c
            keys.append(c)
        if m != masks[i]:
            masks[i] = m
            changed += 1
    return masks, changed, ends
