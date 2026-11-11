"""
[규칙, STEP 3 ~ 6] 동시치기 두께와 배치를 다듬는 규칙 모음

--thick: 소리가 꽉 찬 구간은 두꺼운 동시치기 스트림, 비어 있는 구간은 얇은 스트림으로.
노트를 지우지 않고 줄마다 건반 수만 바꾼다. numpy만 사용.
"""
# 용어는 보고서의 용어 표를 따른다: 줄, 자리, 8분 자리와 뒷자리, 쌍 합, 폭타, 연타, 트릴, 계단, 데님.
# 이 파일에서만 쓰는 말:
#   기둥            8분 전이나 1박 전 줄과 같은 건반을 계속 쓰는 것 (1 _ 1 _ 1 _)
#   축              8분 자리 3줄 이상이 같은 건반으로 이어지는 것
#   사다리 폭타      16분 줄은 그대로 두고 두께만 1+1 ~ 2+2로 줄인 폭타
#   반쯤 빈 16비트   8분 자리는 차 있고 16분 뒷자리가 1~5줄만 있는 마디
#   드문드문         곡에서 가장 조용한 구간에 쓰는, 줄이 가장 적은 마디
#   울타리 모드      사람 곡별 값의 10~90% 범위를 벗어난 만큼만 고치는 모드 (--guard)
from __future__ import annotations

import numpy as np



def _near_mask(t_chart, masks, i: int, ms: float) -> int:
    """i번째 줄 앞뒤 ms 안에 있는 모든 줄(바로 옆 줄만이 아니라 32분 두 줄 전/후까지)의 건반 합 (연타 금지 확인용)"""
    t = np.asarray(t_chart, dtype=np.float64)
    lo = int(np.searchsorted(t, t[i] - ms / 1000.0, side="left"))
    hi = int(np.searchsorted(t, t[i] + ms / 1000.0, side="right"))
    out = 0
    for j in range(lo, hi):
        if j != i:
            out |= int(masks[j])
    return out & 0xFE

def _keys(m: int) -> list[int]:
    return [b for b in range(1, 8) if (int(m) >> b) & 1]


def _rank01(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def intensity_curve(mel: np.ndarray, fps: int = 100, smooth_sec: float = 2.0) -> np.ndarray:
    """(T,) 0~1: 곡 안에서 이 순간이 얼마나 '꽉 찬' 소리인가 (음량 순위와 어택 양 순위의 평균)"""
    m = np.asarray(mel, dtype=np.float32)
    w = max(1, int(smooth_sec * fps))
    k = np.ones(w, dtype=np.float32) / w

    def smooth(v):
        return np.convolve(np.pad(v, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")

    loud = smooth(m.mean(axis=1))
    flux = np.zeros(len(m), dtype=np.float32)
    flux[1:] = np.maximum(m[1:] - m[:-1], 0.0).sum(axis=1)
    busy = smooth(flux)
    return (0.5 * _rank01(loud) + 0.5 * _rank01(busy)).astype(np.float32)


def apply_thickness(t_chart, frames, masks, mel, bpm: float, strength: float = 0.7, lo: float = 1.0,
                    hi: float = 3.2, gain: float = 1.0, max_keys: int = 5, seed: int = 0, jack_ms: float = 100.0,
                    accent=None):
    """반환: (새 masks, 바꾼 줄 수, 대상 줄 수, (세기 하위 25% 평균 두께 전/후, 상위 25% 전/후))"""
    t = np.asarray(t_chart, dtype=np.float64)
    f = np.asarray(frames, dtype=np.int64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 4 or strength <= 0:
        return masks, 0, 0, None
    inten = intensity_curve(mel)[np.clip(f, 0, len(mel) - 1)]
    if accent is None:
        from lane_v2 import accent_features
        accent = accent_features(mel, f)
    acc = np.asarray(accent, dtype=np.float64).mean(axis=1)
    acc = acc - np.median(acc)                      # 곡 평균은 세기 곡선이 정하고, 어택은 줄 사이 상대 차이만
    eighth = 60.0 / bpm / 2 * 1.05
    gp = np.r_[1e9, np.diff(t)]
    gn = np.r_[np.diff(t), 1e9]
    target_rows = (gp <= eighth) | (gn <= eighth)
    if target_rows.sum() >= 8:                       # 대상 줄끼리 다시 순위: 가장 빈 연타 = 0, 가장 꽉 찬 연타 = 1
        inten = inten.astype(np.float64).copy()
        inten[target_rows] = _rank01(inten[target_rows])
    rng = np.random.default_rng(seed + 31)
    before = np.array([len(_keys(m)) for m in masks], dtype=np.float64)
    changed = 0
    for i in np.flatnonzero(target_rows):
        m = masks[i]
        ks = _keys(m)
        if not ks:
            continue
        goal = lo + (hi - lo) * float(inten[i]) + gain * float(acc[i])
        x = (1 - strength) * len(ks) + strength * goal
        want = int(np.clip(np.floor(x) + (rng.random() < x - np.floor(x)), 1, max_keys))   # 평균이 목표가 되게 확률 반올림
        if want < len(ks):
            drop = rng.choice(ks, size=len(ks) - want, replace=False)
            for b in drop:
                m &= ~(1 << int(b))
        elif want > len(ks):
            block = m
            if i > 0 and gp[i] * 1000.0 < jack_ms * 1.5:
                block |= masks[i - 1]
            if i < n - 1 and gn[i] * 1000.0 < jack_ms * 1.5:
                block |= masks[i + 1]
            block |= _near_mask(t_chart, masks, i, jack_ms * 1.5)
            while len(ks) < want:
                cands = [c for c in range(1, 8) if not (block >> c) & 1]
                if not cands:
                    break
                d = np.array([min(abs(c - k) for k in ks) for c in cands], dtype=np.float64)
                w = np.where(d >= 2, d, 0.2)
                c = int(rng.choice(cands, p=w / w.sum()))
                m |= 1 << c
                block |= 1 << c
                ks.append(c)
        if m != masks[i]:
            masks[i] = m
            changed += 1
    after = np.array([len(_keys(m)) for m in masks], dtype=np.float64)
    sel = target_rows
    q1, q3 = np.percentile(inten[sel], [25, 75]) if sel.any() else (0, 1)
    lo_m, hi_m = sel & (inten <= q1), sel & (inten >= q3)
    stats = (before[lo_m].mean() if lo_m.any() else np.nan, after[lo_m].mean() if lo_m.any() else np.nan,
             before[hi_m].mean() if hi_m.any() else np.nan, after[hi_m].mean() if hi_m.any() else np.nan)
    return masks, changed, int(sel.sum()), stats


if __name__ == "__main__":
    # 자체 점검: 앞 절반 조용, 뒤 절반 꽉 찬 가짜 곡 -> 뒤쪽이 더 두꺼워지고, 늘린 건반은 앞뒤 줄과 연타가 아니어야 함
    rng = np.random.default_rng(0)
    T = 3000
    mel = np.zeros((T, 96), np.float32)
    mel[:1500] = 0.2 + 0.02 * rng.random((1500, 96))
    mel[1500:] = 0.5 + 0.2 * rng.random((1500, 96))
    bpm = 150.0
    t = np.arange(0, 29.5, 60 / bpm / 4)
    fr = np.rint(t * 100).astype(int)
    masks = [(1 << (1 + i % 7)) | (1 << (1 + (i + 3) % 7)) for i in range(len(t))]
    acc = np.zeros((len(t), 4))
    new, ch, tot, st = apply_thickness(t, fr, masks, mel, bpm, accent=acc)
    print(f"바꾼 줄 {ch}/{tot} | 조용한 구간 두께 {st[0]:.2f} -> {st[1]:.2f}, 꽉 찬 구간 {st[2]:.2f} -> {st[3]:.2f}")
    assert st[1] < st[0] and st[3] > st[2]
    for i in range(1, len(new)):
        added = new[i] & ~masks[i]
        assert not (added & new[i - 1]) and (i == len(new) - 1 or not (added & new[i + 1])), i
    print("OK")


def adjacent_ratio(masks) -> float:
    """동시치기(건반 2개 이상) 중 붙은 건반(23, 56 등)이 있는 줄의 비율"""
    ch = [m for m in masks if len(_keys(m)) >= 2]
    if not ch:
        return float("nan")
    return float(np.mean([any(b - a == 1 for a, b in zip(_keys(m), _keys(m)[1:])) for m in ch]))


def spread_chords(t_chart, masks, prob: float = 0.6, seed: int = 0, jack_ms: float = 100.0):
    """붙어 있는 건반이 든 동시치기(23, 56, 345 등)를 prob 확률로 벌린다 (--spread-chords)
    반환: (새 masks, 바꾼 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    rng = np.random.default_rng(seed + 47)
    changed = 0
    for i in range(n):
        ks = _keys(masks[i])
        if len(ks) < 2 or len(ks) > 4 or rng.random() >= prob:
            continue
        pairs = [(a, b) for a, b in zip(ks, ks[1:]) if b - a == 1]
        if not pairs:
            continue
        block = masks[i]
        if i > 0 and (t[i] - t[i - 1]) * 1000.0 < jack_ms * 1.5:
            block |= masks[i - 1]
        if i < n - 1 and (t[i + 1] - t[i]) * 1000.0 < jack_ms * 1.5:
            block |= masks[i + 1]
        block |= _near_mask(t, masks, i, jack_ms * 1.5)
        m = masks[i]
        for a, b in pairs:
            if not ((m >> a) & 1 and (m >> b) & 1):
                continue
            best = None
            for mv in ((a, b), (b, a)) if rng.random() < 0.5 else ((b, a), (a, b)):   # mv[0]를 옮김
                rest = [k for k in _keys(m) if k != mv[0]]
                for c in sorted(range(1, 8), key=lambda c: abs(c - mv[0])):
                    if (block >> c) & 1 or c in rest:
                        continue
                    if all(abs(c - r) >= 2 for r in rest):
                        if best is None or abs(c - mv[0]) < best[2]:
                            best = (mv[0], c, abs(c - mv[0]))
                        break
            if best is not None:
                m = (m & ~(1 << best[0])) | (1 << best[1])
                block |= 1 << best[1]
        if m != masks[i]:
            masks[i] = m
            changed += 1
    return masks, changed


def _gap_class(m: int) -> int:
    """동시치기의 간격 성격: 1 = 붙은 건반 있음, 2 = 가장 좁은 간격 2칸, 3 = 3칸 이상"""
    ks = _keys(m)
    g = min(b - a for a, b in zip(ks, ks[1:]))
    return min(g, 3)



def _style(m: int) -> tuple[int, int]:
    """동시치기 성격 = (건반 수, 간격 성격)"""
    return len(_keys(m)), _gap_class(m)


def style_persistence(t_chart, masks, bpm: float, key=None) -> float:
    """확인용: 동시치기가 3개 이상인 반 마디끼리, 다음 반 마디도 같은 성격(가장 많은 것)인 비율. key 기본 = (건반 수, 간격)"""
    import collections
    key = key or _style
    t = np.asarray(t_chart, dtype=np.float64)
    beats = t / (60.0 / bpm)
    gp = np.r_[9.0, np.diff(beats)]
    gn = np.r_[np.diff(beats), 9.0]
    st = (gp <= 0.26) | (gn <= 0.26)
    half = np.floor(beats / 2 + 1e-6).astype(int)
    by = collections.defaultdict(list)
    for h, m, s_ in zip(half, masks, st):
        if s_ and len(_keys(m)) >= 2:
            by[int(h)].append(key(m))
    modes = [collections.Counter(by[h]).most_common(1)[0][0] for h in sorted(by) if len(by[h]) >= 3]
    return float(np.mean([a == b for a, b in zip(modes, modes[1:])])) if len(modes) > 1 else float("nan")


def chord_style(t_chart, masks, bpm: float, keep: float = 0.5, seed: int = 0, jack_ms: float = 100.0):
    """반 마디마다 동시치기 성격(건반 수 + 간격: 붙음 / 2칸 / 3칸 이상)을 정하고, keep 확률로 다음 반 마디에도 유지 (--chord-style)
    반환: (새 masks, 바꾼 줄 수)"""
    import collections
    from itertools import combinations
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 4 or keep <= 0:
        return masks, 0
    B = 60.0 / bpm
    half = np.floor(t / B / 2 + 1e-6).astype(int)
    gp = np.r_[1e9, np.diff(t)]
    gn = np.r_[np.diff(t), 1e9]
    near = (gp <= B / 2 * 1.05) | (gn <= B / 2 * 1.05)
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    # 8분 자리 줄만: 16분 뒷자리까지 같은 두께로 맞추면 '센 줄-약한 줄' 대비가 사라지고
    # 123 456 127 356 처럼 이상해짐. 뒷자리는 beat_contrast가 얇게 정리.
    cand = [i for i in range(n) if near[i] and on8[i] and 2 <= len(_keys(masks[i])) <= 5]
    by = collections.defaultdict(list)
    for i in cand:
        by[int(half[i])].append(i)
    rng = np.random.default_rng(seed + 59)
    style, prev = {}, None
    for h in sorted(by):
        own = collections.Counter(_style(masks[i]) for i in by[h]).most_common(1)[0][0]
        style[h] = prev if (prev is not None and rng.random() < keep) else own
        prev = style[h]
    changed = 0
    for h, ids in by.items():
        size, gap = style[h]
        for i in ids:
            m = masks[i]
            if _style(m) == (size, gap):
                continue
            ks = _keys(m)
            block = 0
            if i > 0 and gp[i] * 1000.0 < jack_ms * 1.5:
                block |= masks[i - 1]
            if i < n - 1 and gn[i] * 1000.0 < jack_ms * 1.5:
                block |= masks[i + 1]
            block |= _near_mask(t, masks, i, jack_ms * 1.5)
            best, best_cost = None, 1e9
            for combo in combinations([c for c in range(1, 8) if not (block >> c) & 1], size):
                if min(min(b - a for a, b in zip(combo, combo[1:])), 3) != gap:
                    continue
                cost = (sum(min(abs(c - k) for k in ks) for c in combo)
                        + sum(min(abs(c - k) for c in combo) for k in ks) + 0.01 * rng.random())
                if cost < best_cost:
                    best, best_cost = combo, cost
            if best is not None:
                nm = m & 1
                for c in best:
                    nm |= 1 << c
                masks[i] = nm
                changed += 1
    return masks, changed


def beat_contrast(t_chart, masks, bpm: float, prob: float = 0.7, seed: int = 0):
    """16분 연타에서 8분 자리는 두껍게, 16분 뒷자리는 얇게 (--beat-contrast)
    반환: (masks, 줄인 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 3 or prob <= 0:
        return masks, 0
    q = t / (60.0 / bpm) * 4.0
    qi = np.round(q).astype(np.int64)
    on = np.abs(q - qi) < 1e-3
    rng = np.random.default_rng(seed + 71)
    cnt = 0
    for i in range(1, n - 1):
        if not (on[i] and on[i - 1] and on[i + 1] and qi[i] % 2 == 1 and qi[i] - qi[i - 1] == 1 and qi[i + 1] - qi[i] == 1):
            continue
        ks = _keys(masks[i])
        lo = min(len(_keys(masks[i - 1])), len(_keys(masks[i + 1])))
        if len(ks) < 2 or len(ks) < lo or (len(ks) == 2 and lo == 2) or rng.random() >= prob:   # 2개 연속(데님)은 denim_cap이 맡음
            continue
        want = max(1, lo - 1)
        if want >= len(ks):
            continue
        keep = rng.choice(ks, size=want, replace=False)
        nm = masks[i] & 1
        for k in keep:
            nm |= 1 << int(k)
        masks[i] = nm
        cnt += 1
    return masks, cnt


def beat_sizes(t_chart, masks, bpm: float):
    """확인용: 16분 연속 구간의 (8분 자리 평균 건반 수, 뒷자리 평균)"""
    t = np.asarray(t_chart, dtype=np.float64)
    q = t / (60.0 / bpm) * 4.0
    qi = np.round(q).astype(np.int64)
    on = np.abs(q - qi) < 1e-3
    n = len(masks)
    ev, od = [], []
    for i in range(n):
        if not on[i]:
            continue
        nb = (i > 0 and on[i - 1] and qi[i] - qi[i - 1] == 1) or (i < n - 1 and on[i + 1] and qi[i + 1] - qi[i] == 1)
        if nb:
            (ev if qi[i] % 2 == 0 else od).append(len(_keys(masks[i])))
    return (float(np.mean(ev)) if ev else float("nan"), float(np.mean(od)) if od else float("nan"))


def denim_runs(t_chart, masks, bpm: float) -> list[int]:
    """확인용: 16분으로 붙은 2개짜리 동시치기 연속(데님, 13 24 13 / 13 24 35 46)의 길이 목록"""
    t = np.asarray(t_chart, dtype=np.float64)
    q = t / (60.0 / bpm) * 4.0
    qi = np.round(q).astype(np.int64)
    on = np.abs(q - qi) < 1e-3
    out, L = [], 0
    for i in range(len(masks)):
        two = on[i] and len(_keys(masks[i])) == 2
        if two and L and on[i - 1] and qi[i] - qi[i - 1] == 1:
            L += 1
        else:
            if L >= 2:
                out.append(L)
            L = 1 if two else 0
    if L >= 2:
        out.append(L)
    return out


def denim_cap(t_chart, masks, bpm: float, lo: int = 4, hi: int = 8, seed: int = 0):
    """데님(2개 동시치기 16분 연속)이 너무 길면 끊는다 (--denim-max)
    반환: (masks, 끊은 곳 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if hi <= 0 or n < 3:
        return masks, 0
    q = t / (60.0 / bpm) * 4.0
    qi = np.round(q).astype(np.int64)
    on = np.abs(q - qi) < 1e-3
    rng = np.random.default_rng(seed + 83)
    cut, L, lim = 0, 0, int(rng.integers(lo, hi + 1))
    for i in range(n):
        two = on[i] and len(_keys(masks[i])) == 2
        if two and L and on[i - 1] and qi[i] - qi[i - 1] == 1:
            L += 1
        else:
            if L >= 2:
                lim = int(rng.integers(lo, hi + 1))
            L = 1 if two else 0
            continue
        if L > lim:
            j = i if qi[i] % 2 == 1 or i + 1 >= n else i + 1          # 뒷자리 줄을 단타로
            if j != i and not (on[j] and qi[j] - qi[i] == 1 and len(_keys(masks[j])) == 2):
                j = i
            ks = _keys(masks[j])
            masks[j] = (masks[j] & 1) | (1 << int(rng.choice(ks)))
            cut += 1
            L = 0 if j == i else 1
            lim = int(rng.integers(lo, hi + 1))
    return masks, cut


def alt_cap(t_chart, masks, max_rows: int = 3, seed: int = 0, jack_ms: float = 100.0, gap_s: float = 0.4):
    """묶음 교대(A-B-A-B: 1357 246 1357, 13 24 13)를 후처리 뒤에 다시 max_rows줄로 제한 (--max-alt-rows와 같은 값)
    반환: (masks, 바꾼 줄 수)"""
    from plan_feats import is_alt
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if max_rows < 2 or n < 3:
        return masks, 0
    rng = np.random.default_rng(seed + 97)
    L, cnt = 0, 0
    for i in range(2, n):
        a, b, c = masks[i - 2] & 0xFE, masks[i - 1] & 0xFE, masks[i] & 0xFE
        if not (is_alt(a, b, c) and t[i] - t[i - 2] <= gap_s):
            L = 0
            continue
        L = L + 1 if L else 3
        if L <= max_rows:
            continue
        block = a | b
        if i + 1 < n and (t[i + 1] - t[i]) * 1000.0 < jack_ms * 1.5:
            block |= masks[i + 1]
        block |= _near_mask(t, masks, i, jack_ms * 1.5)
        same = [k for k in _keys(c) if (a >> k) & 1]
        k_out = int(rng.choice(same))
        free = [k for k in range(1, 8) if not ((block | c) >> k) & 1]
        nm = masks[i] & ~(1 << k_out)
        if free:
            d = np.array([abs(k - k_out) for k in free], dtype=np.float64)
            k_in = free[int(np.argmin(d + 0.01 * rng.random(len(free))))]
            nm |= 1 << k_in
        masks[i] = nm
        if is_alt(a, b, nm & 0xFE):                     # 그래도 교대면 하나 더 뺌 (최소 1개는 남김)
            ks = [k for k in _keys(nm) if (a >> k) & 1]
            if ks and len(_keys(nm)) > 1:
                masks[i] = nm & ~(1 << int(rng.choice(ks)))
        cnt += 1
        L = 0
    return masks, cnt


def chord_count_keep(t_chart, masks, bpm: float, keep: float = 0.5, seed: int = 0, jack_ms: float = 100.0):
    """반 마디 동안 8분 자리 동시치기 '개수'를 이어 가게 (--count-keep)
    반환: (masks, 바꾼 줄 수)"""
    import collections
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 4 or keep <= 0:
        return masks, 0
    B = 60.0 / bpm
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    gp = np.r_[1e9, np.diff(t)]
    gn = np.r_[np.diff(t), 1e9]
    near = (gp <= B / 2 * 1.05) | (gn <= B / 2 * 1.05)
    half = np.floor(t / B / 2 + 1e-6).astype(int)
    by = collections.defaultdict(list)
    for i in range(n):
        if near[i] and on8[i] and len(_keys(masks[i])) >= 2:
            by[int(half[i])].append(i)
    rng = np.random.default_rng(seed + 113)
    target, prev = {}, None
    for h in sorted(by):
        own = collections.Counter(len(_keys(masks[i])) for i in by[h]).most_common(1)[0][0]
        target[h] = prev if (prev is not None and rng.random() < keep) else own
        prev = target[h]
    changed = 0
    for h, ids in by.items():
        want = target[h]
        for i in ids:
            ks = _keys(masks[i])
            if abs(len(ks) - want) != 1:
                continue
            if len(ks) > want:
                if want < 2:
                    continue
                adj = [k for k in ks if (k - 1 in ks) or (k + 1 in ks)]
                k = int(rng.choice(adj if adj else ks))
                masks[i] &= ~(1 << k)
                changed += 1
            else:
                block = masks[i]
                if i > 0 and gp[i] * 1000.0 < jack_ms * 1.5:
                    block |= masks[i - 1]
                if i < n - 1 and gn[i] * 1000.0 < jack_ms * 1.5:
                    block |= masks[i + 1]
                block |= _near_mask(t, masks, i, jack_ms * 1.5)
                free = [c for c in range(1, 8) if not (block >> c) & 1]
                far = [c for c in free if all(abs(c - k) >= 2 for k in ks)]
                pool = far or free
                if not pool:
                    continue
                masks[i] |= 1 << int(rng.choice(pool))
                changed += 1
    return masks, changed


def add_scratch(t_chart, frames, masks, mel, bpm: float, per_measure: float = 0.7, seed: int = 0,
                min_gap_beats: float = 0.5, fb: float = 0.05, ob: float = 0.05, pos_weight: float = 0.0):
    """스크래치가 너무 적을 때 사람 빈도로 채운다 (--scratch-add, 마디당 줄 수 목표)
    반환: (masks, 더한 줄 수)"""
    from accent3_ctl import band_strength
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n == 0 or per_measure <= 0:
        return masks, 0
    beats = t / (60.0 / bpm)
    q = beats * 4.0
    qi = np.round(q).astype(np.int64)
    on = np.abs(q - qi) < 1e-3
    meas = len(set((qi // 16).tolist()))
    want = int(round(per_measure * meas)) - sum(m & 1 for m in masks)
    if want <= 0:
        return masks, 0
    s = band_strength(mel, np.asarray(frames, dtype=np.int64))
    r = np.empty(n); r[np.argsort(s, kind="mergesort")] = np.arange(n); r /= max(n - 1, 1)
    score = r + fb * (on & (qi % 16 == 0)) + ob * (on & (qi % 4 == 0)) + 0.1 * np.array([len(_keys(m)) for m in masks])
    cand = on & (qi % 2 == 0) & (np.array([len(_keys(m)) for m in masks]) > 0)
    have = sorted(beats[i] for i in range(n) if masks[i] & 1)
    added = 0

    def put(i):
        nonlocal added
        if masks[i] & 1:
            return False
        j = np.searchsorted(have, beats[i])
        if (j > 0 and beats[i] - have[j - 1] < min_gap_beats) or (j < len(have) and have[j] - beats[i] < min_gap_beats):
            return False
        masks[i] |= 1
        have.insert(j, beats[i])
        added += 1
        return True

    if pos_weight > 0:
        # --scratch-pos: 줄마다 독립 확률 p = c x 자리 가중치 x (0.5 + 점수). c는 기대 개수가 want가 되게 맞춤.
        # 자리 가중치(lane_v2.scratch_weight, x pos_weight 제곱): 앞 1박 안에 스크래치가 없으면 '처음 놓는' 값
        #, 있으면 '이어지는' 값
        # -> 자리별로 뽑히는 비율이 가중치 비율 그대로 (점수 순 상위 N개는 첫 박에만 몰림)
        from lane_v2 import scratch_weight
        codes = np.where(on, qi % 16, 1)
        w = scratch_weight(codes) ** pos_weight
        wf = scratch_weight(codes, follow=True) ** pos_weight
        base = np.where(cand & (np.array(masks) & 1 == 0), w * (0.5 + score), 0.0)
        if base.sum() <= 0:
            return masks, 0
        lo, hi = 0.0, 1e6
        for _ in range(60):                                   # sum(min(1, c*base)) = want 인 c 찾기
            c = (lo + hi) / 2
            if np.minimum(1.0, c * base).sum() < want:
                lo = c
            else:
                hi = c
        rng = np.random.default_rng(seed + 131)
        u = rng.random(n)
        for i in np.argsort(beats, kind="stable"):            # 시간 순: 앞 1박 안에 스크래치가 있으면 이어지는 가중치
            if added >= want:
                break
            if base[i] <= 0:
                continue
            j = np.searchsorted(have, beats[i])
            near = j > 0 and beats[i] - have[j - 1] <= 1.0 + 1e-3
            p = min(1.0, hi * base[i] * (wf[i] / w[i] if near else 1.0))
            if u[i] < p:
                put(int(i))
        return masks, added
    score = np.where(cand, score, -1)
    score = score + 1e-3 * np.random.default_rng(seed + 127).random(n)
    for i in np.argsort(-score):
        if added >= want or score[i] < 0:
            break
        put(int(i))
    return masks, added


def chord_flow(t_chart, frames, masks, mel, bpm: float, prob: float = 0.5, seed: int = 0,
               jack_ms: float = 100.0, same_sim: float = 0.9):
    """16분 연타 속 8분 자리 동시치기가 방향을 이어 가게 (--flow)
    반환: (masks, 바꾼 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 5 or prob <= 0:
        return masks, 0
    q = t / (60.0 / bpm) * 4.0
    qi = np.round(q).astype(np.int64)
    on = np.abs(q - qi) < 1e-3
    pos = {int(qi[i]): i for i in range(n) if on[i]}
    m = np.asarray(mel, dtype=np.float32)
    fr = np.asarray(frames, dtype=np.int64)

    def feat(i):
        f = int(np.clip(fr[i], 0, len(m) - 1))
        d = m[f:f + 4].mean(0); d = d - d.mean()
        return d / (np.linalg.norm(d) + 1e-9)

    from itertools import combinations
    rng = np.random.default_rng(seed + 131)
    changed = 0
    for p in sorted(pos):
        if p % 2:
            continue
        a, b, c = pos.get(p - 4), pos.get(p - 2), pos[p]
        if a is None or b is None or not all((p - k) in pos for k in (1, 3)):
            continue
        A, Bk, C = _keys(masks[a]), _keys(masks[b]), _keys(masks[c])
        if min(len(A), len(Bk), len(C)) < 2:
            continue
        d1 = np.mean(Bk) - np.mean(A)
        d2 = np.mean(C) - np.mean(Bk)
        if d1 == 0 or d1 * d2 > 0:
            continue
        if float(feat(c) @ feat(a)) >= same_sim or rng.random() >= prob:
            continue
        block = 0
        if c > 0 and (t[c] - t[c - 1]) * 1000.0 < jack_ms * 1.5:
            block |= masks[c - 1]
        if c < n - 1 and (t[c + 1] - t[c]) * 1000.0 < jack_ms * 1.5:
            block |= masks[c + 1]
        block |= _near_mask(t, masks, c, jack_ms * 1.5)
        goal = np.mean(Bk) + d1                     # B에서 A->B만큼 한 번 더 (75 64 -> 53)
        best, cost_b = None, 1e9
        for combo in combinations([k for k in range(1, 8) if not (block >> k) & 1], len(C)):
            cen = float(np.mean(combo))
            if (cen - np.mean(Bk)) * d1 <= 0.4 * abs(d1):      # 같은 방향으로 충분히 가야 함
                continue
            cost = abs(cen - goal) + 0.3 * sum(1 for k in combo if k not in C) \
                + 0.5 * sum(1 for x, y in zip(combo, combo[1:]) if y - x == 1) - 0.5 * sum(1 for x, y in zip(C, C[1:]) if y - x == 1)
            if cost < cost_b:
                best, cost_b = combo, cost
        if best is None:
            continue
        nm = sum(1 << k for k in best)
        masks[c] = (masks[c] & 1) | nm
        changed += 1
    return masks, changed


_LEFT, _RIGHT = 0b1110, 0b11100000          # 1~3번 왼손, 5~7번 오른손, 4번은 가운데(어느 손도 안 셈)


def _hand_load(m: int) -> tuple[int, int]:
    return bin(m & _LEFT).count("1"), bin(m & _RIGHT).count("1")


def hand32(t_chart, masks, bpm: float, prob: float = 0.8, seed: int = 0, jack_ms: float = 100.0):
    """32분 사이 칸(16분 격자 사이) 줄이 앞뒤 32분 이웃 줄과 같은 손에 몰리면 반대 손으로 옮김 (--hand32)
    반환: (masks, 옮긴 줄 수)"""
    masks = [int(m) for m in masks]
    if prob <= 0 or len(masks) < 2:
        return masks, 0
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    g32 = B / 8.0
    q = t / B * 8.0
    off = (np.abs(q - np.round(q)) < 1e-3) & (np.round(q).astype(np.int64) % 2 == 1)
    rng = np.random.default_rng(seed + 181)
    from itertools import combinations
    moved = 0
    n = len(masks)
    for i in np.flatnonzero(off):
        nb = [j for j in (i - 1, i + 1) if 0 <= j < n and abs(abs(t[j] - t[i]) - g32) < g32 * 0.15]
        if not nb:
            continue
        ks = [k for k in _keys(masks[i]) if k != 4]
        if not ks or any(k == 4 for k in _keys(masks[i])):
            continue
        hl = sum(_hand_load(masks[j])[0] for j in nb)
        hr = sum(_hand_load(masks[j])[1] for j in nb)
        side_l = all(k <= 3 for k in ks)
        side_r = all(k >= 5 for k in ks)
        if not ((side_l and hl > hr) or (side_r and hr > hl)):
            continue
        if rng.random() >= prob:
            continue
        block = _near_mask(t, masks, i, jack_ms)
        for j in nb:
            block |= masks[j]
        cand = [k for k in ((5, 6, 7, 4) if side_l else (1, 2, 3, 4)) if not (block >> k) & 1]
        if len(cand) < len(ks):
            continue
        mirror = [8 - k for k in ks]
        best, cb = None, 1e9
        for combo in combinations(cand, len(ks)):
            c = sum(min(abs(x - y) for y in mirror) for x in combo) + 0.01 * rng.random()
            if c < cb:
                best, cb = combo, c
        masks[i] = (masks[i] & 1) | sum(1 << k for k in best)
        moved += 1
    return masks, moved

def jack16(t_chart, masks, bpm: float, prob: float = 0.5, seed: int = 0):
    """16분 간격 연타(같은 건반이 16분 뒤에 또) 줄이기 (--jack16)
    반환: (masks, 옮긴 건반 수)"""
    masks = [int(m) for m in masks]
    if prob <= 0 or len(masks) < 2:
        return masks, 0
    t = np.asarray(t_chart, dtype=np.float64)
    win = 60.0 / bpm / 4.0 * 1.15
    rng = np.random.default_rng(seed + 191)
    moved = 0
    for i in range(1, len(masks)):
        lo = int(np.searchsorted(t, t[i] - win, side="left"))
        before = 0
        for j in range(lo, i):
            before |= masks[j]
        clash = masks[i] & before & 0xFE
        if not clash:
            continue
        for k in _keys(clash):
            if rng.random() >= prob:
                continue
            block = _near_mask(t, masks, i, win * 1000.0) | masks[i]
            free = [c for c in range(1, 8) if not (block >> c) & 1]
            if not free:
                continue
            d = np.array([abs(c - k) for c in free], dtype=np.float64) + 0.01 * rng.random(len(free))
            c = free[int(np.argmin(d))]
            masks[i] = (masks[i] & ~(1 << k)) | (1 << c)
            moved += 1
    return masks, moved

# 8분 간격 겹침, 16분 같은 건반 3줄 연속 울타리
# 8분 간격 겹침 줄 비율, 줄당 겹친 건반 = 곡별 75% 지점
# 16분 3줄 연속 = 16분 간격 줄 1000개당 곡별 중앙값
JACK8_BPM = (140.0, 165.0, 190.0, 215.0)
JACK8_ROWS = (0.53, 0.46, 0.48, 0.42)
JACK8_KEYS = (0.95, 0.84, 0.73, 0.63)
TRI16_K = (7.7, 0.0, 0.0, 0.0)
# 줄이 두꺼우면 우연히 겹치는 양 자체가 커서 위 울타리(건반 수 기준)를 못 맞추고 하한까지 밀게 됨
# 그래서 '우연히 겹칠 양 대비' 사람 곡별 중앙값도 울타리로 같이 씀:
#   줄당 겹친 건반 / (k1*k2/7): ~150 0.70, 150~180 0.63, 180~200 0.57, 200+ 0.56
#   겹친 줄 /: ~150 0.73, 150~180 0.63, 180~200 0.56, 200+ 0.59
# 둘 중 느슨한 쪽까지만 고침 (보통 두께면 건반 수 울타리, 두꺼우면 우연 대비 울타리)
JACK8_REL_K = (0.70, 0.63, 0.57, 0.56)
JACK8_REL_R = (0.73, 0.63, 0.56, 0.59)


def jack8_fence(bpm: float):
    """BPM -> (8분 간격 겹침 줄 비율 상한, 줄당 겹친 건반 상한, 16분 3줄 연속 상한 / 16분 간격 줄 1000개)"""
    f = lambda ys: float(np.interp(bpm, JACK8_BPM, ys))
    return f(JACK8_ROWS), f(JACK8_KEYS), f(TRI16_K)


def jack8_stats(t_chart, masks, bpm: float):
    """(8분 간격 줄 수, 겹침 줄 비율, 줄당 겹친 건반, 16분 간격 줄 수, 16분 3줄 연속 곳 수). 스크래치만 있는 줄은 뺌"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    idx = [i for i in range(len(masks)) if int(masks[i]) & 0xFE]
    e8 = o8 = k8 = e16 = tri = 0
    for a in range(1, len(idx)):
        p, c = idx[a - 1], idx[a]
        o = int(masks[p]) & int(masks[c]) & 0xFE
        if abs(t[c] - t[p] - B / 2) <= B / 16:
            e8 += 1; o8 += o != 0; k8 += bin(o).count("1")
        elif abs(t[c] - t[p] - B / 4) <= B / 32:
            e16 += 1
            if a >= 2 and abs(t[p] - t[idx[a - 2]] - B / 4) <= B / 32 and o & int(masks[idx[a - 2]]):
                tri += 1
    return e8, o8 / max(e8, 1), k8 / max(e8, 1), e16, tri


def jack8_fix(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0):
    """8분 간격 겹침 줄이기 (--jack8)
    반환: (masks, 보고 dict)"""
    masks = [int(m) for m in masks]
    st0 = jack8_stats(t_chart, masks, bpm)
    rep = {"rows": 0, "keys": 0, "tri": 0, "before": st0, "after": st0, "fence": jack8_fence(bpm)}
    if strength <= 0 or len(masks) < 8:
        return masks, rep
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    lim_r, lim_k, lim_t = rep["fence"]
    rng = np.random.default_rng(seed + 211)
    idx = [i for i in range(len(masks)) if masks[i] & 0xFE]

    def move(c, ks, avoid, soft=0):
        """c줄의 건반 ks를 avoid, c줄, jack_ms 안 줄에 없는 가장 가까운 건반으로 (soft 건반은 될 수 있으면 피함). 옮긴 수"""
        block = masks[c] | avoid | _near_mask(t, masks, c, jack_ms + 5.0)
        n_ = 0
        for k in ks:
            free = [x for x in range(1, 8) if not (block >> x) & 1]
            if not free:
                break
            d = np.array([abs(x - k) + (0.5 if (x - 4) * (k - 4) < 0 else 0.0) + (8.0 if (soft >> x) & 1 else 0.0)
                          for x in free]) + 0.01 * rng.random(len(free))
            x = free[int(np.argmin(d))]
            masks[c] = (masks[c] & ~(1 << k)) | (1 << x)
            block |= 1 << x
            n_ += 1
        return n_

    # (1) 8분 간격 겹침: 8분 간격으로 이어진 줄 묶음(사슬)마다 앞에서부터 - 뒷줄 건반은 될 수 있으면 피하고, 못 피하면 뒷줄로
    #     겹침을 밀어 사슬 끝에서 풀림. 겹침이 진한 사슬부터, 울타리 안에 들면 멈춤
    is8 = [abs(t[idx[a]] - t[idx[a - 1]] - B / 2) <= B / 16 for a in range(1, len(idx))]
    chains, a = [], 0
    while a < len(is8):
        if not is8[a]:
            a += 1
            continue
        b_ = a
        while b_ + 1 < len(is8) and is8[b_ + 1]:
            b_ += 1
        chains.append([(idx[x], idx[x + 1]) for x in range(a, b_ + 1)])
        a = b_ + 1
    ov = lambda p, c: masks[p] & masks[c] & 0xFE
    n8 = sum(len(ch) for ch in chains)
    from math import comb
    kc = lambda i: bin(masks[i] & 0xFE).count("1")
    ch_k = sum(kc(p) * kc(c) / 7.0 for ch in chains for p, c in ch)                  # 우연히 겹칠 건반 (건반 수는 안 바뀜)
    ch_r = sum((1.0 - comb(7 - kc(p), kc(c)) / comb(7, kc(c))) if 7 - kc(p) >= kc(c) else 1.0
               for ch in chains for p, c in ch)                                        # 우연히 겹칠 줄
    rel_k = float(np.interp(bpm, JACK8_BPM, JACK8_REL_K))
    rel_r = float(np.interp(bpm, JACK8_BPM, JACK8_REL_R))
    cap_r = max(lim_r * n8, rel_r * ch_r)
    cap_k = max(lim_k * n8, rel_k * ch_k)
    rep["chance"] = (ch_k / max(n8, 1), ch_r / max(n8, 1), cap_k / max(n8, 1), cap_r / max(n8, 1))

    def over():
        r_ = sum(1 for ch in chains for p, c in ch if ov(p, c)) - cap_r
        k_ = sum(bin(ov(p, c)).count("1") for ch in chains for p, c in ch) - cap_k
        return r_ * strength > 1e-9 or k_ * strength > 1e-9

    dens = lambda ch: sum(bin(ov(p, c)).count("1") for p, c in ch) / len(ch)
    order = sorted(range(len(chains)), key=lambda q: (-dens(chains[q]), rng.random()))
    touched = set()
    for q in order:
        if not over():
            break
        ch = chains[q]
        for z, (p, c) in enumerate(ch):
            o = ov(p, c)
            if not o:
                continue
            nx = ch[z + 1][1] if z + 1 < len(ch) else None
            n_ = move(c, _keys(o), masks[p], soft=masks[nx] if nx is not None else 0)
            if n_:
                touched.add(c); rep["keys"] += n_
    rep["rows"] = len(touched)
    # (2) 16분 같은 건반 3줄 연속
    tri = [(idx[a - 2], idx[a - 1], idx[a]) for a in range(2, len(idx))
           if abs(t[idx[a]] - t[idx[a - 1]] - B / 4) <= B / 32 and abs(t[idx[a - 1]] - t[idx[a - 2]] - B / 4) <= B / 32]
    n16 = sum(1 for a in range(1, len(idx)) if abs(t[idx[a]] - t[idx[a - 1]] - B / 4) <= B / 32)
    have = [x for x in tri if masks[x[0]] & masks[x[1]] & masks[x[2]] & 0xFE]
    ex_t = int(round((len(have) - lim_t / 1000.0 * n16) * strength))
    if ex_t > 0:
        for q in rng.permutation(len(have)):
            if ex_t <= 0:
                break
            a, b, c = have[int(q)]
            s = masks[a] & masks[b] & masks[c] & 0xFE
            n_ = move(b, _keys(s), masks[a] | masks[c]) if s else 0
            if n_:
                rep["tri"] += 1; rep["keys"] += n_; ex_t -= 1
    rep["after"] = jack8_stats(t_chart, masks, bpm)
    return masks, rep

def chord_cap(t_chart, masks, max_chord: int = 6, seed: int = 0, allow=None):
    """마지막 안전장치: 한 줄의 건반 수(스크래치는 안 셈, 4+S 허용)가 max_chord를 넘으면 건반을 뺌
    반환: (masks, 고친 줄 수)"""
    masks = [int(m) for m in masks]
    if max_chord <= 0 or max_chord >= 7:
        return masks, 0
    t = np.asarray(t_chart, dtype=np.float64)
    rng = np.random.default_rng(seed + 197)
    fixed = 0
    allow = allow or {}                              # 폭타 끝 줄은 상한 + --end-over까지 허용
    for i in range(len(masks)):
        cap_i = max(max_chord, allow.get(i, 0))
        if len(_keys(masks[i])) <= cap_i:
            continue
        near = 0
        for j in (i - 1, i + 1):
            if 0 <= j < len(masks) and abs(t[j] - t[i]) < 0.2:
                near |= masks[j]
        while len(_keys(masks[i])) > cap_i:
            ks = _keys(masks[i])
            rep = [k for k in ks if (near >> k) & 1]
            inner = ks[1:-1] if len(ks) > 2 else ks
            adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
            k = int(rng.choice(rep or adj or inner))
            masks[i] &= ~(1 << k)
        fixed += 1
    return masks, fixed

def jack_fix(t_chart, masks, jack_ms: float = 100.0, seed: int = 0):
    """마지막 안전장치: jack_ms 안에서 같은 건반(스크래치 제외)이 다시 나오면 뒷줄 건반을 가까운 빈 건반으로 옮김
    반환: (masks, 고친 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    if jack_ms <= 0:
        return masks, 0
    rng = np.random.default_rng(seed + 137)
    fixed = 0
    for i in range(len(masks)):
        lo = int(np.searchsorted(t, t[i] - jack_ms / 1000.0, side="left"))
        before = 0
        for j in range(lo, i):
            before |= masks[j]
        clash = masks[i] & before & 0xFE
        if not clash:
            continue
        block = _near_mask(t, masks, i, jack_ms) | masks[i]
        for k in _keys(clash):
            free = [c for c in range(1, 8) if not (block >> c) & 1]
            if free:
                d = np.array([abs(c - k) for c in free], dtype=np.float64) + 0.01 * rng.random(len(free))
                c = free[int(np.argmin(d))]
                masks[i] = (masks[i] & ~(1 << k)) | (1 << c)
                block |= 1 << c
            elif len(_keys(masks[i])) > 1:
                masks[i] &= ~(1 << k)
            else:                                    # 단타라 못 옮기면 앞쪽 동시치기에서 그 건반을 뺌
                for j in range(i - 1, lo - 1, -1):
                    if (masks[j] >> k) & 1 and len(_keys(masks[j])) > 1:
                        masks[j] &= ~(1 << k)
                        break
        fixed += 1
    return masks, fixed


def count_smooth(t_chart, masks, bpm: float, prob: float = 0.8, seed: int = 0, jack_ms: float = 100.0,
                 max_jump: int = 2):
    """16분 연타 속 8분 자리 동시치기 건반 수를 고르게 (--count-smooth)
    반환: (masks, 바꾼 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 8 or prob <= 0:
        return masks, 0
    B = 60.0 / bpm
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    rng = np.random.default_rng(seed + 151)
    s16 = B / 4
    cnt = lambda m: len(_keys(m))

    def add_one(i):
        ks = _keys(masks[i])
        block = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)      # +5ms: 프레임 반올림 여유 (jack_fix와 같은 기준)
        pm = _prev8_mask(t, masks, i, B)
        free = [c for c in range(1, 8) if not ((block | pm) >> c) & 1 and (c - 1 not in ks) and (c + 1 not in ks)]
        if not free:                                                   # 기둥(8분 전/1박 전 건반)은 되도록 피함
            free = [c for c in range(1, 8) if not (block >> c) & 1 and (c - 1 not in ks) and (c + 1 not in ks)]
        if not free:
            free = [c for c in range(1, 8) if not (block >> c) & 1]
        if not free:
            return False
        if ks:
            d = np.array([min(abs(c - k) for k in ks) for c in free], dtype=np.float64)
            w = np.where(d >= 2, 1.0, 0.2) + 0.01
        else:
            w = np.ones(len(free))
        masks[i] |= 1 << int(rng.choice(free, p=w / w.sum()))
        return True

    def drop_one(i):
        ks = _keys(masks[i])
        if len(ks) <= 1:
            return False
        pm = _prev8_mask(t, masks, i, B)
        rep = [k for k in ks if (pm >> k) & 1]           # 기둥(8분 전/1박 전과 같은 건반)부터
        inner = ks[1:-1] if len(ks) > 2 else ks
        adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
        k = int(rng.choice(rep or adj or inner))
        masks[i] &= ~(1 << k)
        return True

    changed = set()
    # 8분 자리 줄이 8분 간격으로 끊김 없이 이어지는 구간 (16분 뒷자리는 있든 없든: --thin-weak로 뒷자리가 빠져도 잡힘)
    e8 = [k for k in range(n) if on8[k]]
    runs, cur = [], e8[:1]
    for a, b in zip(e8, e8[1:]):
        if abs((t[b] - t[a]) - 2 * s16) <= s16 * 0.15:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    for e in runs:
        if len(e) < 4 or np.mean([cnt(masks[k]) for k in e]) < 2.0:
            continue
        for a, b, c in zip(e, e[1:], e[2:]):
            ca, cb, cc = cnt(masks[a]), cnt(masks[b]), cnt(masks[c])
            lo = max(1, min(ca, cc) - 1)
            hi = max(ca, cc) + 1
            lo = max(lo, ca - max_jump)
            hi = min(hi, ca + max_jump)
            want = min(max(cb, lo), hi)
            if want == cb or rng.random() >= prob:
                continue
            while cnt(masks[b]) < want and add_one(b):
                pass
            while cnt(masks[b]) > want and drop_one(b):
                pass
            if cnt(masks[b]) != cb:
                changed.add(b)
    return masks, len(changed)


def thin_boost(t_chart, masks, bpm: float, dropped_t, prob: float = 0.7, seed: int = 0, jack_ms: float = 100.0,
               max_notes: int = 6):
    """--thin-weak로 16분 뒷자리를 덜어낸 자리 바로 옆 8분 자리 줄을 건반 하나 더 두껍게 (--thin-boost)
    반환: (masks, 바꾼 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    dt = np.sort(np.asarray(dropped_t, dtype=np.float64))
    if len(dt) == 0 or prob <= 0 or len(masks) == 0:
        return masks, 0
    B = 60.0 / bpm
    s16 = B / 4
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    rng = np.random.default_rng(seed + 163)
    changed = 0
    for i in np.flatnonzero(on8):
        ks = _keys(masks[i])
        if not ks or len(ks) >= max_notes:
            continue
        hit = False
        for side in (-1, 1):
            x = t[i] + side * s16
            j = int(np.searchsorted(dt, x))
            for k in (j - 1, j):
                if 0 <= k < len(dt) and abs(dt[k] - x) <= s16 * 0.15:
                    hit = True
        if not hit or rng.random() >= prob:
            continue
        block = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)
        pm = _prev8_mask(t, masks, i, B)
        free = [c for c in range(1, 8) if not ((block | pm) >> c) & 1] or \
            [c for c in range(1, 8) if not (block >> c) & 1]      # 기둥(8분 전/1박 전 건반)은 되도록 피함
        if not free:
            continue
        d = np.array([min(abs(c - k) for k in ks) for c in free], dtype=np.float64)
        w = np.where(d >= 2, 1.0, 0.15) + 0.01
        masks[i] |= 1 << int(rng.choice(free, p=w / w.sum()))
        changed += 1
    return masks, changed


def _prev8_mask(t, masks, i, B):
    """i번째 줄의 8분 전 + 1박 전 줄 건반 (기둥 확인용)"""
    out = 0
    for dt in (B / 2, B):
        j = int(np.searchsorted(t, t[i] - dt - 1e-4))
        if j < i and abs(t[i] - t[j] - dt) <= B / 16:
            out |= masks[j]
    return out & 0xFE


def row_kind(t_chart, bpm: float) -> np.ndarray:
    """줄마다 구간 종류: 16 = 16분 연타 속(앞뒤 16분 안에 줄이 있음), 8 = 8분 동시치기 구간(16분 이웃 없이 8분 이웃),
    0 = 그 밖(4비트 이하로 성긴 곳)"""
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(t)
    B = 60.0 / bpm
    gp = np.r_[1e9, np.diff(t)]
    gn = np.r_[np.diff(t), 1e9]
    g = np.minimum(gp, gn)
    out = np.zeros(n, dtype=np.int64)
    out[g <= B / 4 * 1.15] = 16
    out[(g > B / 4 * 1.15) & (g <= B / 2 * 1.1)] = 8
    return out


def stream_cap(t_chart, masks, bpm: float, prob: float = 0.3, big: int = 4, seed: int = 0):
    """16분 연타 속 8분 자리(박, 8분 뒷박) 줄 중 big건반 이상인 큰 동시치기에서 건반마다 prob 확률로 덜어냄 (--stream-cap)
    반환: (masks, 바꾼 줄 수)"""
    masks = [int(m) for m in masks]
    if prob <= 0:
        return masks, 0
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    kind = row_kind(t, bpm)
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    rng = np.random.default_rng(seed + 171)
    changed = 0
    for i in np.flatnonzero((kind == 16) & on8):
        ks = _keys(masks[i])
        if len(ks) < big:
            continue
        k_drop = min(int(rng.binomial(len(ks), min(prob, 1.0))), len(ks) - 1)
        for _ in range(k_drop):
            pm = _prev8_mask(t, masks, i, B)
            rep = [k for k in ks if (pm >> k) & 1]
            inner = ks[1:-1] if len(ks) > 2 else ks
            adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
            k = int(rng.choice(rep or adj or inner))
            masks[i] &= ~(1 << k)
            ks.remove(k)
        changed += k_drop > 0
    return masks, changed


def eighth_chords(t_chart, masks, bpm: float, overlap: float = 0.6, big: float = 0.4, max_chord: int = 6,
                  seed: int = 0, jack_ms: float = 100.0, diff=None, same_prob: float = 0.0):
    """8분 동시치기 구간(16분 이웃 없는 8분 줄) 다듬기
    반환: (masks, 옮긴 줄 수, 키운 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 2 or (overlap <= 0 and big <= 0):
        return masks, 0, 0
    B = 60.0 / bpm
    kind = row_kind(t, bpm)
    rng = np.random.default_rng(seed + 173)
    moved = grown = 0

    dif = np.ones(n, dtype=bool) if diff is None else np.asarray(diff, dtype=bool)

    def nb(i, sound=False):
        """앞뒤 8분 줄 건반 (+ jack_ms 안). sound면 같은 소리인 이웃은 빼서 '겹쳐도 되는' 반복으로 둠"""
        m = 0
        if i > 0 and t[i] - t[i - 1] <= B / 2 * 1.1 and (not sound or dif[i]):
            m |= masks[i - 1]
        if i < n - 1 and t[i + 1] - t[i] <= B / 2 * 1.1 and (not sound or dif[i + 1]):
            m |= masks[i + 1]
        return (m | _near_mask(t, masks, i, jack_ms + 5.0)) & 0xFE

    def pick(free, ks):
        if ks:
            d = np.array([min(abs(c - k) for k in ks) for c in free], dtype=np.float64)
            w = np.where(d >= 2, 1.0, 0.15) + 0.01
        else:
            w = np.ones(len(free))
        return int(rng.choice(free, p=w / w.sum()))

    for i in np.flatnonzero(kind == 8):
        ks = _keys(masks[i])
        if not ks:
            continue
        around = nb(i)
        bad = nb(i, sound=True)                       # 다른 소리인 이웃과 겹친 건반
        shared = [k for k in ks if (bad >> k) & 1]
        p_move = overlap
        if not shared and same_prob > 0:              # 같은 소리 이웃과만 겹치면 same_prob로
            shared = [k for k in ks if (around >> k) & 1]
            p_move = same_prob
        if shared and p_move > 0 and rng.random() < p_move:
            did = False
            for k in shared:
                free = [c for c in range(1, 8) if not ((around | masks[i]) >> c) & 1]
                if not free:
                    break
                c = pick(free, [x for x in ks if x != k])
                masks[i] = (masks[i] & ~(1 << k)) | (1 << c)
                ks = _keys(masks[i])
                did = True
            moved += did
        if big > 0 and len(ks) >= 2:
            did = False
            while len(ks) < max_chord and rng.random() < big:
                free = [c for c in range(1, 8) if not ((nb(i) | masks[i]) >> c) & 1]
                if not free:
                    break
                masks[i] |= 1 << pick(free, ks)
                ks = _keys(masks[i])
                did = True
            grown += did
    return masks, moved, grown


def sound_diff(mel, frames, pct: float = 50.0) -> np.ndarray:
    """줄마다 '바로 앞 줄과 다른 소리인가' (곡 안 이웃 유사도가 pct 백분위 아래면 다른 소리). jack_ctl과 같은 방식"""
    from jack_ctl import onset_profiles
    fr = np.asarray(frames, dtype=np.int64)
    out = np.ones(len(fr), dtype=bool)
    if len(fr) < 3:
        return out
    P = onset_profiles(mel, fr)
    s = (P[1:] * P[:-1]).sum(1)
    out[1:] = s < np.percentile(s, pct)
    return out


def pillar_fix(t_chart, masks, bpm: float, prob: float = 0.5, seed: int = 0, jack_ms: float = 100.0):
    """16분 연타 속 8분 자리 줄이 8분 전 / 1박 전 줄과 같은 건반을 쓰면(기둥: 1 _ 1 _ 1 _) prob 확률로 가까운 빈 건반으로 옮김 (--pillar-fix)
    반환: (masks, 바꾼 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    if n < 3 or prob <= 0:
        return masks, 0
    B = 60.0 / bpm
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    kind = row_kind(t, bpm)
    rng = np.random.default_rng(seed + 179)
    changed = 0
    for i in np.flatnonzero(on8 & (kind == 16)):
        def row_at(dt):
            j = int(np.searchsorted(t, t[i] - dt - 1e-4))
            return masks[j] if j < i and abs(t[i] - t[j] - dt) <= B / 16 else 0
        m8, m4 = row_at(B / 2), row_at(B)            # 8분 전, 1박 전 (같은 8분 자리 기둥)
        shared = [k for k in _keys(masks[i]) if ((m8 | m4) >> k) & 1]
        if not shared or rng.random() >= prob:
            continue
        did = False
        for k in shared:
            near = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)
            free = [c for c in range(1, 8) if not ((near | m8 | m4) >> c) & 1]
            if not free:                              # 둘 다 피할 자리가 없으면 8분 전 줄만 피함
                if not (m8 >> k) & 1:
                    continue
                free = [c for c in range(1, 8) if not ((near | m8) >> c) & 1]
            if not free:
                continue
            d = np.array([abs(c - k) for c in free], dtype=np.float64) + 0.01 * rng.random(len(free))
            c = free[int(np.argmin(d))]
            masks[i] = (masks[i] & ~(1 << k)) | (1 << c)
            did = True
        changed += did
    return masks, changed


# 16분 연타 속 8분 자리 줄 건반 수 목표 (5 = 5개 이상)
ON8_HUMAN = {1: 0.04, 2: 0.24, 3: 0.44, 4: 0.24, 5: 0.04}


def on8_match(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, target=None, max_chord: int = 7,
              jack_ms: float = 100.0):
    """16분 연타 속 8분 자리 줄의 건반 수 분포를 사람 쪽으로 (--on8-match)
    반환: (masks, 뺀 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    if strength <= 0 or len(masks) < 8:
        return masks, 0
    tg = dict(ON8_HUMAN if target is None else target)
    B = 60.0 / bpm
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    rows = [int(i) for i in np.flatnonzero(on8 & (row_kind(t, bpm) == 16))]
    if len(rows) < 8:
        return masks, 0
    rng = np.random.default_rng(seed + 181)
    cnt = lambda i: min(len(_keys(masks[i])), 5)
    pos = {i: k for k, i in enumerate(rows)}
    changed = 0
    for c in (5, 4, 3, 2):
        have = [i for i in rows if cnt(i) == c]
        excess = int(round((len(have) - tg[c] * len(rows)) * strength))
        if excess <= 0:
            continue
        w = []
        for i in have:
            k = pos[i]
            nb = [rows[j] for j in (k - 1, k + 1) if 0 <= j < len(rows)]
            w.append(1.0 + sum(1 for j in nb if cnt(j) < c))
        w = np.array(w, dtype=np.float64)
        pick = rng.choice(len(have), size=min(excess, len(have)), replace=False, p=w / w.sum())
        for p in pick:
            i = have[int(p)]
            ks = _keys(masks[i])
            if len(ks) <= 1:
                continue
            pm = _prev8_mask(t, masks, i, B)
            rep_ = [k for k in ks if (pm >> k) & 1]
            inner = ks[1:-1] if len(ks) > 2 else ks
            adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
            masks[i] &= ~(1 << int(rng.choice(rep_ or adj or inner)))
            changed += 1
    # 목표보다 얇은 쪽이 많으면 올림. 위쪽(c개 초과)이 목표보다 모자란 만큼만.
    # 더할 건반: 앞뒤 jack_ms 안 줄과 기둥(8분 전/1박 전)을 피하고, 기존 건반과 두 칸 이상 떨어진 쪽 우선. max_chord까지.
    for c in (1, 2, 3, 4):
        have = [i for i in rows if len(_keys(masks[i])) == c]
        above = sum(1 for i in rows if cnt(i) > c)
        want_above = sum(v for k, v in tg.items() if k > c) * len(rows)
        excess = int(round(min(len(have) - tg[c] * len(rows), want_above - above) * strength))
        if excess <= 0 or c >= max_chord:
            continue
        w = []
        for i in have:
            k = pos[i]
            nb = [rows[j] for j in (k - 1, k + 1) if 0 <= j < len(rows)]
            w.append(1.0 + sum(1 for j in nb if cnt(j) > c))
        w = np.array(w, dtype=np.float64)
        pick = rng.choice(len(have), size=min(excess, len(have)), replace=False, p=w / w.sum())
        for p in pick:
            i = have[int(p)]
            ks = _keys(masks[i])
            near = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)
            pm = _prev8_mask(t, masks, i, B)
            free = [k for k in range(1, 8) if not ((near | pm) >> k) & 1] or \
                [k for k in range(1, 8) if not (near >> k) & 1]
            if not free:
                continue
            d = np.array([min(abs(k - x) for x in ks) for k in free], dtype=np.float64) if ks else np.ones(len(free))
            wk = np.where(d >= 2, 1.0, 0.25) + 0.01
            masks[i] |= 1 << int(rng.choice(free, p=wk / wk.sum()))
            changed += 1
    return masks, changed


OFF_HUMAN = {1: 0.36, 2: 0.47, 3: 0.15, 4: 0.02}    # ★20 13개: 16분 연타 속 뒷자리(16분 사이 칸) 줄 건반 수


def off_match(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
              target=None):
    """16분 연타 속 뒷자리 줄 건반 수를 ★20 분포(1개 36% / 2개 47% / 3개 15% / 4개 2%)로 올림 (--off-match)
    반환: (masks, 더한 줄 수)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    if strength <= 0 or len(masks) < 8:
        return masks, 0
    tg = dict(OFF_HUMAN if target is None else target)
    B = 60.0 / bpm
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    rows = [int(i) for i in np.flatnonzero(~on8 & (row_kind(t, bpm) == 16)) if masks[i] & 0xFE]
    if len(rows) < 8:
        return masks, 0
    rng = np.random.default_rng(seed + 191)
    n = len(masks)
    cnt = lambda i: len(_keys(masks[i]))
    changed = 0
    for c in (1, 2, 3):
        have = [i for i in rows if cnt(i) == c]
        excess = int(round((len(have) - tg.get(c, 0.0) * len(rows)) * strength))
        if excess <= 0:
            continue
        w = np.array([1.0 + 2.0 * sum(1 for j in (i - 1, i + 1) if 0 <= j < n and len(_keys(masks[j])) >= 3)
                      for i in have], dtype=np.float64)
        pick = rng.choice(len(have), size=min(excess, len(have)), replace=False, p=w / w.sum())
        for p in pick:
            i = have[int(p)]
            ks = _keys(masks[i])
            near = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)
            pm = _prev8_mask(t, masks, i, B)
            free = [k for k in range(1, 8) if not ((near | pm) >> k) & 1] or \
                [k for k in range(1, 8) if not (near >> k) & 1]
            if not free:
                continue
            d = np.array([min(abs(k - x) for x in ks) for k in free], dtype=np.float64) if ks else np.ones(len(free))
            wk = np.where(d >= 2, 1.0, 0.25) + 0.01
            masks[i] |= 1 << int(rng.choice(free, p=wk / wk.sum()))
            changed += 1
    return masks, changed


# ---------------------------------------------------------------------------
# 셋잇단 연속 (6비트 = 4분 셋잇단, 24비트 = 16분 셋잇단) 두께 다듬기 (--trip-match)
# 사람 ★ 패턴 277개 (같은 간격으로 3줄 이상 이어지고 16분 격자 밖 줄이 섞인 곳):
#   12비트 826곳: 이웃 같은 개수 47%, 차이 2+ 66% -> 보통 구간과 같아서 건드리지 않음
# ---------------------------------------------------------------------------
TRIP24_HUMAN = {1: 0.61, 2: 0.25, 3: 0.10, 4: 0.04}


def triplet_runs(t_chart, bpm: float, period: float, min_len: int = 3, tol: float = 0.06):
    """간격이 period박(±tol)으로 min_len줄 이상 이어지고 16분 격자 밖 줄이 하나라도 있는 구간 [(시작, 끝)] (끝 포함)"""
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(t)
    B = 60.0 / bpm
    if n < min_len:
        return []
    g = np.diff(t) / B
    ok = np.abs(g - period) <= tol * period
    q16 = t / B * 4.0
    off16 = np.abs(q16 - np.round(q16)) > 0.08
    out, i = [], 0
    while i < n - 1:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n - 1 and ok[j]:
            j += 1
        if j - i + 1 >= min_len and off16[i:j + 1].any():
            out.append((i, j))
        i = j
    return out


def trip_match(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, max_chord: int = 6,
               jack_ms: float = 100.0):
    """셋잇단 연속 두께를 사람 쪽으로 (--trip-match, 0이면 끔)
    반환: (masks, 6비트 바꾼 줄, 24비트 바꾼 줄, 6비트 연속 수, 24비트 연속 수)"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(masks)
    if strength <= 0 or n < 3:
        return masks, 0, 0, 0, 0
    rng = np.random.default_rng(seed + 233)
    cnt = lambda i: len(_keys(masks[i]))

    def nbm(i):
        m = _near_mask(t, masks, i, jack_ms + 5.0)
        if i > 0:
            m |= masks[i - 1]
        if i < n - 1:
            m |= masks[i + 1]
        return m & 0xFE

    def add_one(i):
        ks = _keys(masks[i])
        if len(ks) >= max_chord:
            return False
        free = [c for c in range(1, 8) if not ((nbm(i) | masks[i]) >> c) & 1]
        if not free:
            return False
        if ks:
            d = np.array([min(abs(c - k) for k in ks) for c in free], dtype=np.float64)
            w = np.where(d >= 2, 1.0, 0.15) + 0.01
        else:
            w = np.ones(len(free))
        masks[i] |= 1 << int(rng.choice(free, p=w / w.sum()))
        return True

    def drop_one(i):
        ks = _keys(masks[i])
        if len(ks) <= 1:
            return False
        nb = nbm(i)
        rep = [k for k in ks if (nb >> k) & 1]
        inner = ks[1:-1] if len(ks) > 2 else ks
        adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
        masks[i] &= ~(1 << int(rng.choice(rep or adj or inner)))
        return True

    # 6비트
    r6 = triplet_runs(t, bpm, 2.0 / 3.0)
    ch6 = set()
    for a, b in r6:
        idx = list(range(a, b + 1))
        c = [cnt(i) for i in idx]
        tg = int(np.floor(np.mean(c) + 0.5))
        tg = max(1, min(tg, max_chord))
        for i in idx:
            d = cnt(i) - tg
            if abs(d) >= 2:
                goal = tg + (1 if d > 0 else -1)
            elif abs(d) == 1 and rng.random() < strength * 0.6:
                goal = tg
            else:
                continue
            if abs(d) >= 2 and rng.random() >= strength:
                continue
            while cnt(i) < goal and add_one(i):
                ch6.add(i)
            while cnt(i) > goal and drop_one(i):
                ch6.add(i)
    # 24비트
    r24 = triplet_runs(t, bpm, 1.0 / 6.0)
    rows = sorted({i for a, b in r24 for i in range(a, b)} - ch6)          # 마지막 줄(착지) 뺌
    ch24 = 0
    if rows:
        for k in (4, 3, 2):
            cur = [i for i in rows if min(cnt(i), 4) >= k]
            want = sum(v for kk, v in TRIP24_HUMAN.items() if kk >= k) * len(rows)
            ex = int(round((len(cur) - want) * strength))
            if ex <= 0:
                continue
            for i in rng.permutation(cur)[:ex]:
                ch24 += drop_one(int(i))
    return masks, len(ch6), ch24, len(r6), len(r24)


# ---------------------------------------------------------------------------
# 폭타 뼈대 / 마디 반복 / 대각선 (--skel-swap, --bar-repeat, --diag)
# 사람 ★20 34개 vs 생성 5곡, shape_stats.py 기준 중앙값:
#   폭타 속 8분 자리 > 바로 뒤 뒷자리 69% / 생성 40%, 뒷자리가 더 두꺼움 3% / 26%
#   한 마디 전 같은 칸 건반 수 같음 62% / 34%, 모양 같음 13% / 3%, 닮은 모양(반전, 평행이동 포함) 31% / 11%
# ---------------------------------------------------------------------------
def _mirror(m: int) -> int:
    return (m & 1) | sum(1 << (8 - k) for k in range(1, 8) if (m >> k) & 1)


def _shift(m: int, d: int) -> int:
    """건반을 d칸 평행이동 (1~7 밖으로 나가면 0). 스크래치는 그대로"""
    ks = [k + d for k in _keys(m)]
    if not ks or min(ks) < 1 or max(ks) > 7:
        return 0
    return (m & 1) | sum(1 << k for k in ks)


def _near_ex(t, masks, i: int, ms: float, ex=()) -> int:
    """_near_mask와 같지만 ex에 든 줄도 뺌 (두 줄을 같이 바꿀 때)"""
    t = np.asarray(t, dtype=np.float64)
    lo = int(np.searchsorted(t, t[i] - ms / 1000.0, side="left"))
    hi = int(np.searchsorted(t, t[i] + ms / 1000.0, side="right"))
    out = 0
    for j in range(lo, hi):
        if j != i and j not in ex:
            out |= int(masks[j])
    return out & 0xFE


def _grid16(t, bpm: float):
    """16분 격자 위 줄만 {칸 번호: 줄 번호} (t = 0이 마디 시작)"""
    q = np.asarray(t, dtype=np.float64) / (60.0 / bpm) * 4.0
    qi = np.round(q).astype(np.int64)
    return {int(qi[i]): i for i in np.flatnonzero(np.abs(q - qi) < 1e-3)}


def skel_swap(t_chart, masks, bpm: float, prob: float = 1.0, seed: int = 0, jack_ms: float = 100.0):
    """폭타 뼈대 (--skel-swap): 16분 연타 속 8분 자리 줄이 바로 뒤(16분) 뒷자리 줄보다 얇으면 prob 확률로 두 줄의 건반을 맞바꿈
    반환: (masks, 바꾼 쌍 수)"""
    masks = [int(m) for m in masks]
    if prob <= 0 or len(masks) < 4:
        return masks, 0
    t = np.asarray(t_chart, dtype=np.float64)
    kind = row_kind(t, bpm)
    pos = _grid16(t, bpm)
    rng = np.random.default_rng(seed + 239)
    n_sw = 0
    for p in sorted(pos):
        i, j = pos[p], pos.get(p + 1)
        if p % 2 or j is None or kind[i] != 16:
            continue
        a, b = masks[i] & 0xFE, masks[j] & 0xFE
        if bin(b).count("1") <= bin(a).count("1") or rng.random() >= prob:
            continue
        opts = [(b, a), (_mirror(b) & 0xFE, _mirror(a) & 0xFE)]
        for d in rng.permutation([-2, -1, 1, 2]):               # 그래도 겹치면 맞바꾼 모양을 함께 평행이동
            sb, sa = _shift(b, int(d)), _shift(a, int(d))
            if sb and sa:
                opts.append((sb, sa))
        for ni, nj in opts:
            if ni & _near_ex(t, masks, i, jack_ms + 5.0, (j,)) or nj & _near_ex(t, masks, j, jack_ms + 5.0, (i,)):
                continue
            masks[i] = (masks[i] & 1) | ni
            masks[j] = (masks[j] & 1) | nj
            n_sw += 1
            break
    return masks, n_sw


def _row_feat(mel, frames) -> np.ndarray:
    """줄마다 소리 벡터 (곡 기준으로 대역마다 표준화한 멜, 앞뒤 2프레임 평균)"""
    m = np.asarray(mel, dtype=np.float32)
    z = (m - m.mean(0)) / (m.std(0) + 1e-6)
    fr = np.clip(np.asarray(frames, dtype=np.int64), 0, len(m) - 1)
    out = np.stack([z[np.clip(fr + d, 0, len(m) - 1)] for d in (-2, -1, 0, 1, 2)]).mean(0)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)


def bar_repeat(t_chart, frames, masks, mel, bpm: float, prob: float = 0.6, shape: float = 0.35, seed: int = 0,
               jack_ms: float = 100.0, max_chord: int = 7, rhythm_sim: float = 0.6, sound_sim: float = 0.5):
    """마디 반복 (--bar-repeat): 리듬과 소리가 한 마디 전과 비슷한 마디에서, 같은 칸 줄마다 prob 확률로 한 마디 전 줄의 건반 수를 따라감
    반환: (masks, 비슷한 마디 수, 건반 수 맞춘 줄, 모양 따라간 줄)"""
    masks = [int(m) for m in masks]
    if prob <= 0 or mel is None or len(frames) != len(masks) or len(masks) < 16:
        return masks, 0, 0, 0
    t = np.asarray(t_chart, dtype=np.float64)
    pos = _grid16(t, bpm)
    feat = _row_feat(mel, frames)
    rng = np.random.default_rng(seed + 241)
    by = {}
    for p, i in pos.items():
        by.setdefault(p // 16, {})[p % 16] = i
    cnt = lambda i: len(_keys(masks[i]))

    def fits(i, m):
        return not (m & _near_ex(t, masks, i, jack_ms + 5.0))

    def resize(i, goal):
        while cnt(i) < goal:
            ks = _keys(masks[i])
            free = [c for c in range(1, 8) if not ((_near_ex(t, masks, i, jack_ms + 5.0) | masks[i]) >> c) & 1]
            if not free:
                return
            d = np.array([min((abs(c - k) for k in ks), default=2) for c in free], dtype=np.float64)
            w = np.where(d >= 2, 1.0, 0.15) + 0.01
            masks[i] |= 1 << int(rng.choice(free, p=w / w.sum()))
        while cnt(i) > goal:
            ks = _keys(masks[i])
            inner = ks[1:-1] if len(ks) > 2 else ks
            adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
            masks[i] &= ~(1 << int(rng.choice(adj or inner)))

    n_bar = n_cnt = n_shape = 0
    for b in sorted(by):
        cur, prev = by[b], by.get(b - 1)
        if not prev or len(cur) < 4 or len(prev) < 4:
            continue
        common = sorted(set(cur) & set(prev))
        if len(common) / len(set(cur) | set(prev)) < rhythm_sim:
            continue
        if float(np.mean([feat[cur[s]] @ feat[prev[s]] for s in common])) < sound_sim:
            continue
        n_bar += 1
        for s in common:
            i, j = cur[s], prev[s]
            src = masks[j] & 0xFE
            if not src or rng.random() >= prob:
                continue
            if rng.random() < shape:
                opts = [src, _mirror(src) & 0xFE] + [m for m in (_shift(src, d) for d in rng.permutation([-1, 1])) if m]
                done = False
                for m in opts:
                    if m == masks[i] & 0xFE:
                        done = True
                        break
                    if fits(i, m):
                        masks[i] = (masks[i] & 1) | m
                        n_shape += 1
                        done = True
                        break
                if done:
                    continue
            goal = min(len(_keys(src)), max_chord)
            if cnt(i) != goal:
                resize(i, goal)
                n_cnt += 1
    return masks, n_bar, n_cnt, n_shape


def diag(t_chart, masks, bpm: float, prob: float = 0.15, seed: int = 0, jack_ms: float = 100.0):
    """대각선 (--diag): 16분 연타에서 두 줄 간격인 줄끼리 건반 수가 같으면 prob 확률로 한 칸 평행이동한 모양으로
    반환: (masks, 바꾼 줄 수)"""
    masks = [int(m) for m in masks]
    if prob <= 0 or len(masks) < 4:
        return masks, 0
    t = np.asarray(t_chart, dtype=np.float64)
    pos = _grid16(t, bpm)
    rng = np.random.default_rng(seed + 251)
    n_d = 0
    last = {}
    for p in sorted(pos):
        i, k = pos[p], pos.get(p + 2)
        if k is None or p + 1 not in pos:
            continue
        a, c = masks[i] & 0xFE, masks[k] & 0xFE
        if not a or len(_keys(a)) != len(_keys(c)) or rng.random() >= prob:
            continue
        d0 = last.get(p)
        dirs = [d0, -d0] if d0 else list(rng.permutation([-1, 1]))
        for d in dirs:
            m = _shift(a, int(d))
            if m and m != c and not (m & _near_ex(t, masks, k, jack_ms + 5.0)):
                masks[k] = (masks[k] & 1) | m
                last[p + 2] = int(d)
                n_d += 1
                break
    return masks, n_d


# ---------------------------------------------------------------------------
# 32분, 24분 단타 / 연속 모양 / 박 안 두 8분 자리 / 가운데 몰린 4건반 (--fine-single, --fine-shape, --beat-pair, --edge4)
#   박 안 첫, 둘째 8분 자리 건반 수: 같음 56% / 첫째가 큼 29% / 둘째가 큼 15% (생성 같음 34%)
# ---------------------------------------------------------------------------
def _fine_gap(g: float, B: float) -> int:
    """간격(초) -> 32 / 24 / 0"""
    if abs(g - B / 8) <= 0.15 * B / 8:
        return 32
    if abs(g - B / 6) <= 0.15 * B / 6:
        return 24
    return 0


def fine_single(t_chart, masks, bpm: float, target: float = 0.9, strength: float = 1.0, seed: int = 0,
                jack_ms: float = 100.0):
    """32분, 24분 채운 줄을 단타로 (--fine-single)
    반환: (masks, 단타로 만든 줄 수, 대상 줄 수)"""
    masks = [int(m) for m in masks]
    if strength <= 0 or len(masks) < 3:
        return masks, 0, 0
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q16 = t / B * 4.0
    off16 = np.abs(q16 - np.round(q16)) > 0.08
    rows = []
    for i in range(1, len(t) - 1):
        if off16[i] and masks[i] & 0xFE:
            k = _fine_gap(t[i] - t[i - 1], B)
            if k and _fine_gap(t[i + 1] - t[i], B) == k:
                rows.append(i)
    if not rows:
        return masks, 0, 0
    multi = [i for i in rows if len(_keys(masks[i])) >= 2]
    single = 1.0 - len(multi) / len(rows)
    f = min(1.0, max(0.0, (target - single) / max(1.0 - single, 1e-9))) * strength
    rng = np.random.default_rng(seed + 257)
    ctr = lambda m: float(np.mean(_keys(m))) if m & 0xFE else 4.0
    n = 0
    for i in multi:
        if rng.random() >= f:
            continue
        ks = _keys(masks[i])
        near = _near_mask(t, masks, i, jack_ms + 5.0)
        ok = [k for k in ks if not (near >> k) & 1] or ks
        lo, hi = sorted((ctr(masks[i - 1]), ctr(masks[i + 1])))
        w = np.array([2.0 if lo < k < hi else 1.0 for k in ok])
        masks[i] = (masks[i] & 1) | (1 << int(rng.choice(ok, p=w / w.sum())))
        n += 1
    return masks, n, len(rows)


# 277곡 단타 5줄+ 연속(916곳)으로 다시 잼. 예전 ★20 13곡 값은
# 좁은 자리 반복(432432), 좌우 교대가 사람보다 몇 배 많았음
FINE_SHAPES = {"stair": 0.56, "alt": 0.07, "zig": 0.02, "narrow": 0.04, "none": 0.31}
# 계단 계열을 고를 확률은 간격마다 다르게, 나머지는 위 비율대로 나눔
FINE_STAIR_P = {24: 0.52, 32: 0.35}
# 계단 계열 안 종류: 한 줄 / 감아 도는(4567 123) / 겹계단(5 3 6 4) / 꺾이는
STAIR_KINDS = {32: {"line": 0.46, "wrap": 0.23, "double": 0.26, "bent": 0.05},
               24: {"line": 0.58, "wrap": 0.10, "double": 0.13, "bent": 0.19}}


def fine_weights(gap: int, base=None) -> dict:
    """간격(24 / 32)별 연속 모양 비율: 계단 = FINE_STAIR_P, 나머지는 base 비율대로 남은 몫을 나눔"""
    w = dict(FINE_SHAPES if base is None else base)
    ps = FINE_STAIR_P.get(gap, w["stair"])
    rest = sum(v for k, v in w.items() if k != "stair")
    return {k: (ps if k == "stair" else v * (1.0 - ps) / rest) for k, v in w.items()}


def _stair_seq(sub: str, L: int, rng) -> list[int]:
    """계단 계열 단타 순서"""
    d = int(rng.choice([-1, 1]))
    if sub == "wrap":                                     # 연속 안에서 끝을 꼭 넘도록 시작 자리를 끝 쪽에서
        s = int(rng.integers(max(1, 9 - L), 8)) if d > 0 else int(rng.integers(1, min(7, L - 1) + 1))
        return [(s - 1 + d * j) % 7 + 1 for j in range(L)]
    if sub == "double":                                   # 짝수 줄 s, s+d, s+2d ... / 홀수 줄은 두 칸 뒤
        span = (L - 1) // 2 + 2
        if span > 6:
            return []
        lo = 3 if d > 0 else 1 + (L - 1) // 2
        hi = 7 - (L - 1) // 2 if d > 0 else 5
        if lo > hi:
            return []
        s = int(rng.integers(lo, hi + 1))
        return [s + d * (j // 2) - (2 * d if j % 2 else 0) for j in range(L)]
    if sub == "bent" and L >= 4:                          # turn번째 줄에서 방향을 바꿈(바로 앞 건반은 다시 안 밟고 한 칸 건너뜀)
        turn = int(rng.integers(2, L - 1))
        up, down = turn, L - turn
        for _ in range(10):
            s = int(rng.integers(1, 8))
            seq = [s + d * j for j in range(turn)]
            k = seq[-1] - d * 2
            seq += [k - d * j for j in range(down)]
            if all(1 <= x <= 7 for x in seq):
                return seq
        return []
    # line (L > 7이면 끝에서 되돌아감)
    if L <= 7:
        s = int(rng.integers(1, 8 - (L - 1))) if d > 0 else int(rng.integers(L, 8))
        return [s + d * j for j in range(L)]
    s, out, k = (1 if d > 0 else 7), [], 0
    k = s
    for _ in range(L):
        out.append(k)
        if not 1 <= k + d <= 7:
            d = -d
        k += d
    return out


def _fine_seq(kind: str, L: int, rng, gap: int = 32) -> tuple[list[int], str]:
    """연속 모양 하나의 단타 건반 순서 (길이 L)와 세부 이름"""
    if kind == "stair":
        sw = STAIR_KINDS.get(gap, STAIR_KINDS[32])
        subs = list(sw)
        sub = subs[int(rng.choice(len(subs), p=np.array([sw[k] for k in subs]) / sum(sw.values())))]
        return _stair_seq(sub, L, rng), "stair_" + sub
    if kind == "alt":                                     # 왼손(1~3)과 오른손(5~7)이 번갈아, 손마다 세 건반을 돌아가며
        lh, rh = list(rng.permutation([1, 2, 3])), list(rng.permutation([5, 6, 7]))
        first = int(rng.integers(2))
        return [int((lh if (j + first) % 2 == 0 else rh)[(j // 2) % 3]) for j in range(L)], kind
    if kind == "zig":                                     # 두 칸 가고 한 칸 돌아오기 (2 4 3 5 4 6 ...)
        d = int(rng.choice([-1, 1]))
        s = int(rng.integers(1, 8))
        out, k = [], s
        for j in range(L):
            out.append(k)
            step = 2 * d if j % 2 == 0 else -d
            if not 1 <= k + step <= 7:
                d = -d
                step = -step
            k += step
        return out, kind
    if kind == "narrow":                                  # 붙은 세 건반을 돌아가며 (3 4 5 3 4 5)
        c = int(rng.integers(1, 6))
        cyc = list(rng.permutation([c, c + 1, c + 2]))
        return [int(cyc[j % 3]) for j in range(L)], kind
    return [], kind


def fine_shape(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
               weights=None):
    """24분, 32분 3줄 이상 연속을 모양 하나로 (--fine-shape)
    반환: (masks, 바꾼 연속 수, 연속 수, 모양별 개수)"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(masks)
    if strength <= 0 or n < 3:
        return masks, 0, 0, {}
    B = 60.0 / bpm
    q16 = t / B * 4.0
    off16 = np.abs(q16 - np.round(q16)) > 0.08
    probs = {}
    for gap in (24, 32):                                  # 계단을 고를 확률이 간격마다 다름
        wt = fine_weights(gap, weights)
        names = list(wt)
        p = np.array([wt[k] for k in names], dtype=np.float64)
        probs[gap] = (names, p / p.sum())
    rng = np.random.default_rng(seed + 263)
    runs, i = [], 0
    while i < n - 1:
        k = _fine_gap(t[i + 1] - t[i], B)
        if not k:
            i += 1
            continue
        j = i + 1
        while j < n - 1 and _fine_gap(t[j + 1] - t[j], B) == k:
            j += 1
        R = [x for x in range(i, j + 1) if off16[x] or i < x < j]
        if j - i + 1 >= 3 and off16[i:j + 1].any() and len(R) >= 3:
            runs.append((R, k))
        i = j
    done, used = 0, {}
    for R, gap in runs:
        if rng.random() >= strength:
            continue
        names, p = probs[gap]
        kind = names[int(rng.choice(len(names), p=p))]
        if kind == "none":
            used["none"] = used.get("none", 0) + 1
            continue
        old = [masks[x] for x in R]
        for _ in range(6):
            seq, sub = _fine_seq(kind, len(R), rng, gap)
            if not seq:
                continue
            for x, k in zip(R, seq):
                masks[x] = (masks[x] & 1) | (1 << k)
            if all(not (masks[x] & _near_mask(t, masks, x, jack_ms + 5.0)) for x in R):
                done += 1
                used[sub] = used.get(sub, 0) + 1
                break
            for x, m in zip(R, old):
                masks[x] = m
    return masks, done, len(runs), used


def beat_pair(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
              max_chord: int = 7, target=(0.41, 0.40, 0.15)):
    """박 안 두 8분 자리 건반 수 맞추기 (--beat-pair)
    반환: (masks, 맞춘 쌍 수, 쌍 수)"""
    masks = [int(m) for m in masks]
    if strength <= 0 or len(masks) < 8:
        return masks, 0, 0
    t = np.asarray(t_chart, dtype=np.float64)
    kind = row_kind(t, bpm)
    pos = _grid16(t, bpm)
    cnt = lambda i: len(_keys(masks[i]))
    pairs = [(pos[p], pos[p + 2]) for p in sorted(pos)
             if p % 4 == 0 and p + 2 in pos and kind[pos[p]] == 16 and kind[pos[p + 2]] == 16
             and masks[pos[p]] & 0xFE and masks[pos[p + 2]] & 0xFE]
    if len(pairs) < 8:
        return masks, 0, len(pairs)
    rng = np.random.default_rng(seed + 269)
    ab = [pr for pr in pairs if cnt(pr[0]) > cnt(pr[1])]
    ba = [pr for pr in pairs if cnt(pr[0]) < cnt(pr[1])]
    pick = []
    for grp, tg in ((ab, target[1]), (ba, target[2])):
        ex = int(round((len(grp) - tg * len(pairs)) * strength))
        if ex > 0:
            pick += [grp[int(x)] for x in rng.choice(len(grp), size=min(ex, len(grp)), replace=False)]
    n_done = 0
    for a, b in pick:
        goal = min(cnt(a), max_chord)
        nxt = pos.get(int(round(t[b] / (60.0 / bpm) * 4.0)) + 1)
        if nxt is not None and goal < cnt(nxt):
            continue
        while cnt(b) < goal:
            ks = _keys(masks[b])
            free = [c for c in range(1, 8) if not ((_near_mask(t, masks, b, jack_ms + 5.0) | masks[b]) >> c) & 1]
            if not free:
                break
            d = np.array([min((abs(c - k) for k in ks), default=2) for c in free], dtype=np.float64)
            w = np.where(d >= 2, 1.0, 0.15) + 0.01
            masks[b] |= 1 << int(rng.choice(free, p=w / w.sum()))
        while cnt(b) > goal:
            ks = _keys(masks[b])
            pm = masks[b - 1] & 0xFE if b > 0 else 0
            rep = [k for k in ks if (pm >> k) & 1]
            inner = ks[1:-1] if len(ks) > 2 else ks
            adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
            masks[b] &= ~(1 << int(rng.choice(rep or adj or inner)))
        n_done += cnt(b) == goal
    return masks, n_done, len(pairs)


def edge4(t_chart, masks, prob: float = 0.6, seed: int = 0, jack_ms: float = 100.0, target: float = 0.07):
    """가운데 몰린 4건반이 사람보다 많을 때만 줄이기 (--edge4)
    반환: (masks, 옮긴 줄 수)"""
    masks = [int(m) for m in masks]
    if prob <= 0:
        return masks, 0
    t = np.asarray(t_chart, dtype=np.float64)
    rng = np.random.default_rng(seed + 271)
    four = [i for i, m in enumerate(masks) if len(_keys(m)) == 4]
    cen = [i for i in four if 1 not in _keys(masks[i]) and 7 not in _keys(masks[i])
           and sum(k + 1 in _keys(masks[i]) for k in _keys(masks[i])) >= 2]
    ex = int(round((len(cen) - target * len(four)) * min(prob, 1.0)))
    if ex <= 0:
        return masks, 0
    n = 0
    for i in rng.permutation(cen):
        if n >= ex:
            break
        i = int(i)
        m = masks[i]
        ks = _keys(m)
        near = _near_mask(t, masks, i, jack_ms + 5.0)
        opts = [(ks[0], 1), (ks[-1], 7)]
        for a, b in (opts if rng.random() < 0.5 else opts[::-1]):
            if not (near >> b) & 1:
                masks[i] = (m & ~(1 << a)) | (1 << b)
                n += 1
                break
    return masks, n


# ---------------------------------------------------------------------------
# 채운 단타 축 풀기 / 쉼 뒤 폭타 재시작 (--fill-axis, --restart-shift)
#   16분 연타가 1/2~1박 쉬고 다시 시작할 때 끝 줄과 시작 줄이 건반을 나눠 씀: 사람 49% / 생성 63~70%
# ---------------------------------------------------------------------------
def fill_axis(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
              window_beats: float = 1.0):
    """16분 격자 밖 단타(32, 24분 채움)가 앞 window_beats박 안 다른 채운 단타와 같은 건반이면(축) strength 확률로 가장 가까운 다른 건반으로 옮김
    반환: (masks, 옮긴 줄 수, 대상 줄 수)"""
    masks = [int(m) for m in masks]
    n = len(masks)
    if strength <= 0 or n < 3:
        return masks, 0, 0
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q16 = t / B * 4.0
    off16 = np.abs(q16 - np.round(q16)) > 0.08
    rng = np.random.default_rng(seed + 271)
    fills = [i for i in range(n) if off16[i] and len(_keys(masks[i])) == 1]
    moved = 0
    for a, i in enumerate(fills):
        recent = [j for j in fills[:a] if t[i] - t[j] <= window_beats * B + 1e-6]
        used = 0
        for j in recent:
            used |= masks[j] & 0xFE
        k = _keys(masks[i])[0]
        if not (used >> k) & 1 or rng.random() >= strength:
            continue
        block = used | _near_mask(t, masks, i, jack_ms + 5.0)
        if i > 0:
            block |= masks[i - 1]
        if i < n - 1:
            block |= masks[i + 1]
        free = [c for c in range(1, 8) if not (block >> c) & 1]
        if not free:
            continue
        d = np.array([abs(c - k) for c in free], dtype=np.float64) + 0.01 * rng.random(len(free))
        c = free[int(np.argmin(d))]
        masks[i] = (masks[i] & 1) | (1 << c)
        moved += 1
    return masks, moved, len(fills)


def restart_shift(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
                  target: float = 0.49):
    """16분 연타가 1/2~1박 쉬고 다시 시작할 때, 시작 줄이 쉬기 전 끝 줄과 건반을 나눠 쓰는 비율을 target(사람 49%)까지 줄임
    반환: (masks, 옮긴 곳, 재시작 수)"""
    masks = [int(m) for m in masks]
    n = len(masks)
    if strength <= 0 or n < 8:
        return masks, 0, 0
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q = B / 4
    is16 = lambda a, b: abs((t[b] - t[a]) - q) <= 0.15 * q
    spots = []
    for i in range(2, n - 3):
        g = t[i + 1] - t[i]
        if B / 2 - 0.1 * q <= g <= B + 0.1 * q and is16(i - 2, i - 1) and is16(i - 1, i) \
                and is16(i + 1, i + 2) and is16(i + 2, i + 3) and masks[i] & 0xFE and masks[i + 1] & 0xFE:
            spots.append(i)
    if not spots:
        return masks, 0, 0
    over = [i for i in spots if masks[i] & masks[i + 1] & 0xFE]
    ex = int(round((len(over) - target * len(spots)) * strength))
    if ex <= 0:
        return masks, 0, len(spots)
    rng = np.random.default_rng(seed + 277)
    moved = 0
    for i in rng.permutation(over):                 # 겹침이 다 풀린 곳만 세서 ex곳까지
        if moved >= ex:
            break
        s = int(i) + 1
        old = masks[s]
        for k in [k for k in _keys(masks[s]) if (masks[s - 1] >> k) & 1]:
            block = masks[s - 1] | masks[s + 1] | masks[s] | _near_mask(t, masks, s, jack_ms + 5.0)
            free = [c for c in range(1, 8) if not (block >> c) & 1]
            if not free:
                break
            d = np.array([abs(c - k) for c in free], dtype=np.float64) + 0.01 * rng.random(len(free))
            c = free[int(np.argmin(d))]
            masks[s] = (masks[s] & ~(1 << k)) | (1 << c)
        if masks[s] & masks[s - 1] & 0xFE:
            masks[s] = old                           # 다 못 풀면 그대로
        else:
            moved += 1
    return masks, moved, len(spots)


# ---------------------------------------------------------------------------
# 구간 강약 (--section-dyn)
# 사람 ★277개(scratch_features 멜 크기, 2~8마디 창): 소리 크기 vs 줄당 건반 순위상관 중앙 +0.55~+0.60
#   소리 크기 vs 줄 수는 +0.14 -> 사람은 크기를 줄 수보다 동시치기 크기로 나타냄
# ---------------------------------------------------------------------------
def section_loudness(t_chart, row_loud, bpm: float, smooth: int = 1) -> np.ndarray:
    """마디별 구간 소리 크기: 그 마디 줄들의 노트 순간 크기 평균을 앞뒤 smooth마디로 평균"""
    t = np.asarray(t_chart, dtype=np.float64)
    L = np.asarray(row_loud, dtype=np.float64)
    bar = 4 * 60.0 / bpm
    meas = np.floor(t / bar + 1e-6).astype(np.int64)
    n_meas = int(meas.max()) + 1 if len(meas) else 0
    out = np.full(n_meas, np.nan)
    for k in range(n_meas):
        sel = meas == k
        if sel.any():
            out[k] = float(L[sel].mean())
    sm = out.copy()
    for k in range(n_meas):
        w = out[max(0, k - smooth):k + smooth + 1]
        w = w[~np.isnan(w)]
        sm[k] = w.mean() if len(w) and not np.isnan(out[k]) else np.nan
    return sm


def section_dyn(t_chart, masks, bpm: float, meas_loud, strength: float = 1.0, seed: int = 0, max_chord: int = 7,
                jack_ms: float = 100.0, depth: float = 0.9):
    """구간 강약 (--section-dyn)
    반환: (masks, 뺀 건반 수, 더한 건반 수, (조용한 1/3 줄당, 큰 1/3 줄당) 전, 후)"""
    masks = [int(m) for m in masks]
    n = len(masks)
    t = np.asarray(t_chart, dtype=np.float64)
    if strength <= 0 or n < 16:
        return masks, 0, 0, None, None
    B = 60.0 / bpm
    bar = 4 * B
    meas = np.floor(t / bar + 1e-6).astype(np.int64)
    loud = np.asarray(meas_loud, dtype=np.float64)
    cnt = lambda i: len(_keys(masks[i]))
    used = sorted(k for k in set(meas.tolist()) if 0 <= k < len(loud) and not np.isnan(loud[k]))
    if len(used) < 6:
        return masks, 0, 0, None, None
    lv = np.array([loud[k] for k in used])
    rank = {k: r for k, r in zip(used, (np.argsort(np.argsort(lv)) / max(len(lv) - 1, 1)))}
    rows_of = {k: np.flatnonzero(meas == k) for k in used}
    rows_of = {k: [int(i) for i in v if masks[i] & 0xFE] for k, v in rows_of.items()}
    allc = [cnt(i) for k in used for i in rows_of[k]]
    mu = float(np.mean(allc)) if allc else 1.0

    def thirds():
        q = sorted(used, key=lambda k: rank[k])
        third = max(1, len(q) // 3)
        f = lambda ks: float(np.mean([cnt(i) for k in ks for i in rows_of[k]] or [0]))
        return f(q[:third]), f(q[-third:])
    before = thirds()
    rng = np.random.default_rng(seed + 281)
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 0.08) & (np.round(q16).astype(np.int64) % 2 == 0)
    onb = (np.abs(q16 - np.round(q16)) < 0.08) & (np.round(q16).astype(np.int64) % 4 == 0)
    removed = added = 0
    for k in used:
        R = rows_of[k]
        if not R:
            continue
        goal = mu * (1.0 + strength * depth * (rank[k] - 0.5))
        need = int(round((goal - np.mean([cnt(i) for i in R])) * len(R)))
        if need < 0:
            for _ in range(-need):
                cand = [i for i in R if cnt(i) >= 2]
                if not cand:
                    break
                top = max(cnt(i) for i in cand)
                cand = [i for i in cand if cnt(i) == top]
                cand.sort(key=lambda i: (onb[i], on8[i]))          # 뒷자리부터 덜어냄
                i = cand[0] if rng.random() < 0.7 else cand[int(rng.integers(len(cand)))]
                ks = _keys(masks[i])
                nb = ((masks[i - 1] if i > 0 else 0) | (masks[i + 1] if i < n - 1 else 0)) & 0xFE
                rep = [x for x in ks if (nb >> x) & 1]
                inner = ks[1:-1] if len(ks) > 2 else ks
                adj = [x for x in inner if (x - 1 in ks) or (x + 1 in ks)]
                masks[i] &= ~(1 << int(rng.choice(rep or adj or inner)))
                removed += 1
        elif need > 0:
            for _ in range(need):
                cand = [i for i in R if cnt(i) < max_chord]
                if not cand:
                    break
                low = min(cnt(i) for i in cand)
                pool = [i for i in cand if cnt(i) <= low + 1]
                pool.sort(key=lambda i: (-int(onb[i]), -int(on8[i]), cnt(i)))   # 박, 8분 자리부터 키움
                did = False
                for i in pool[:6]:
                    block = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)
                    if i > 0:
                        block |= masks[i - 1]
                    if i < n - 1:
                        block |= masks[i + 1]
                    free = [c for c in range(1, 8) if not (block >> c) & 1]
                    if not free:
                        continue
                    ks = _keys(masks[i])
                    d = np.array([min(abs(c - x) for x in ks) if ks else 3 for c in free], dtype=np.float64)
                    w = np.where(d >= 2, 1.0, 0.3) + 0.01
                    masks[i] |= 1 << int(rng.choice(free, p=w / w.sum()))
                    added += 1
                    did = True
                    break
                if not did:
                    break
    return masks, removed, added, before, thirds()


# ---------------------------------------------------------------------------
# 폭타 끝 묵직하게 다시 (--end-fat, --end-over) / 8비트 동시치기 이어짐 방식 (--eighth-trans)
# 폭타 끝(16분 3줄+ 뒤 8분 이상 쉼) 사람 277 / ★20: 끝 줄 >= 앞 1박 8분 자리 최대 66 / 61%, > 21 / 12%, 4건반+ 31%
# 같은 개수 동시치기가 8분~1박 간격으로 이어질 때: 그대로 10/5, 하나만 이동 16/9, 여럿 바뀜 29/24,
# ---------------------------------------------------------------------------
def stream_end_fat(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, max_chord: int = 7,
                   over: int = 1, jack_ms: float = 100.0, ge_target: float = 0.66, gt_target: float = 0.2):
    """폭타 끝 줄(16분 3줄+ 뒤 8분 이상 쉼)을 앞 1박 안 8분 자리 최대 동시치기 이상으로 (--end-fat)
    반환: (masks, 키운 줄 수, 끝 수, {끝 줄 번호: 허용 상한})"""
    masks = [int(m) for m in masks]
    n = len(masks)
    t = np.asarray(t_chart, dtype=np.float64)
    if strength <= 0 or n < 6:
        return masks, 0, 0, {}
    B = 60.0 / bpm
    s16 = B / 4 * 1.15
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 0.08) & (np.round(q16).astype(np.int64) % 2 == 0)
    cnt = lambda i: len(_keys(masks[i]))
    lim = min(7, max_chord + max(0, over))
    ends = []
    for i in range(2, n):
        gn = t[i + 1] - t[i] if i < n - 1 else 1e9
        if t[i] - t[i - 1] <= s16 and t[i - 1] - t[i - 2] <= s16 and gn >= B / 2 * 0.95 and masks[i] & 0xFE:
            body = [cnt(j) for j in range(max(0, i - 8), i) if t[i] - t[j] <= B + 1e-6 and on8[j]]
            if body:
                ends.append((i, max(body)))
    if not ends:
        return masks, 0, 0, {}
    rng = np.random.default_rng(seed + 283)
    ge = sum(cnt(i) >= bm for i, bm in ends) / len(ends)
    p_up = min(1.0, max(0.0, (ge_target - ge) / max(1.0 - ge, 1e-9))) * strength
    allow, grown = {}, 0

    def add_to(i, goal):
        nonlocal grown
        did = False
        gn = t[i + 1] - t[i] if i < n - 1 else 1e9
        while cnt(i) < goal:
            block = masks[i] | masks[i - 1] | _near_mask(t, masks, i, jack_ms + 5.0)
            if i < n - 1 and gn * 1000.0 < jack_ms + 5.0:
                block |= masks[i + 1]
            free = [c for c in range(1, 8) if not (block >> c) & 1]
            if not free:
                break
            masks[i] |= 1 << int(rng.choice(free))
            did = True
        grown += did
    for i, bm in ends:
        allow[i] = lim
        if cnt(i) < bm and rng.random() < p_up:
            add_to(i, min(bm, lim))
    gt = sum(cnt(i) > bm for i, bm in ends) / len(ends)
    eq = [(i, bm) for i, bm in ends if cnt(i) == bm and bm < lim]
    if eq:
        p_plus = min(1.0, max(0.0, (gt_target - gt) * len(ends) / len(eq))) * strength
        for i, bm in eq:
            if rng.random() < p_plus:
                add_to(i, bm + 1)
    return masks, grown, len(ends), allow


TRANS_HUMAN = {"same": 0.08, "one": 0.13, "par": 0.18, "new": 0.36, "sev": 0.25}


def eighth_trans(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
                 diff=None, chain: float = 0.3, target=None):
    """8분~1박 간격으로 같은 개수(2+) 동시치기가 이어질 때(16분 이웃 없는 줄끼리) 다음 줄을 사람처럼 바꿈 (--eighth-trans)
    반환: (masks, 바꾼 줄 수, 대상 쌍 수, 방식별 개수)"""
    masks = [int(m) for m in masks]
    n = len(masks)
    t = np.asarray(t_chart, dtype=np.float64)
    if strength <= 0 or n < 4:
        return masks, 0, 0, {}
    B = 60.0 / bpm
    s16 = B / 4 * 1.15
    tg = dict(TRANS_HUMAN if target is None else target)
    dif = np.ones(n, bool) if diff is None else np.asarray(diff, bool)
    has16 = np.zeros(n, bool)
    for i in range(n):
        if (i > 0 and t[i] - t[i - 1] <= s16) or (i < n - 1 and t[i + 1] - t[i] <= s16):
            has16[i] = True
    rng = np.random.default_rng(seed + 293)
    cnt = lambda m: len(_keys(m))
    used, changed, pairs = {}, 0, 0
    last_kind, last_dir = None, 0

    def shift(m, s):
        ks = [k + s for k in _keys(m)]
        return sum(1 << k for k in ks) if all(1 <= k <= 7 for k in ks) else None
    for i in range(1, n):
        j = i - 1
        g = t[i] - t[j]
        a, c = masks[j] & 0xFE, masks[i] & 0xFE
        if not (B / 2 * 0.95 <= g <= B * 1.05) or has16[i] or has16[j] or cnt(a) < 2 or cnt(a) != cnt(c):
            last_kind = None
            continue
        pairs += 1
        if rng.random() >= strength:
            last_kind = None
            continue
        k = cnt(a)
        w = dict(tg)
        if not dif[i]:
            w["same"] *= 3.0
            w["one"] *= 2.0
        else:
            w["same"] *= 0.3
        if k < 3:
            w["sev"] = 0.0                                   # 2건반은 '여럿 바뀜'이 없음 (1개 공유 = 하나만 이동)
        if 7 - k < k:
            w["new"] = 0.0
        names = [x for x in w if w[x] > 0]
        p = np.array([w[x] for x in names])
        first = "one" if (last_kind == "one" and rng.random() < chain) else names[int(rng.choice(len(names), p=p / p.sum()))]
        order = [first] + [x for x in sorted(names, key=lambda x: -w[x]) if x != first]   # 못 놓으면 다음 방식으로
        near = _near_mask(t, masks, i, jack_ms + 5.0) & ~a   # 바로 앞 줄(8분 이상 떨어짐)은 따로 판단
        if g * 1000.0 < jack_ms + 5.0:
            near |= a
        nxt = masks[i + 1] & 0xFE if i < n - 1 and (t[i + 1] - t[i]) * 1000.0 < jack_ms + 5.0 else 0
        done = False
        for kind in order:
            cands = []
            if kind == "same":
                cands = [a]
            elif kind == "one":
                dirs = [last_dir] if last_kind == "one" and last_dir and kind == first else [-1, 1]
                for x in rng.permutation(_keys(a)):
                    for d in rng.permutation(dirs):
                        y = int(x) + int(d)
                        if 1 <= y <= 7 and not (a >> y) & 1:
                            cands.append(((a & ~(1 << int(x))) | (1 << y), int(d)))
            elif kind == "par":
                for sft in rng.permutation([-2, -1, 1, 2, -3, 3]):
                    m2 = shift(a, int(sft))
                    if m2 is not None and not (m2 & a):      # 통째로 옮긴 뒤 안 겹치는 것만 (12 -> 23은 '하나만 이동')
                        cands.append(m2)
            elif kind == "new":
                free = [x for x in range(1, 8) if not (a >> x) & 1]
                for _ in range(10):
                    pick = sorted(int(x) for x in rng.choice(free, size=k, replace=False))
                    cands.append(sum(1 << x for x in pick))
            else:                                             # 여럿 바뀜: 1~k-2개만 남기고 나머지 새로
                for _ in range(10):
                    keep = int(rng.integers(1, k - 1))
                    kept = [int(x) for x in rng.choice(_keys(a), size=keep, replace=False)]
                    free = [x for x in range(1, 8) if not (a >> x) & 1]
                    if k - keep > len(free):
                        continue
                    add = [int(x) for x in rng.choice(free, size=k - keep, replace=False)]
                    cands.append(sum(1 << x for x in kept + add))
            for cd in cands:
                m2, d = (cd if isinstance(cd, tuple) else (cd, 0))
                if m2 & near or m2 & nxt:
                    continue
                if m2 != c:
                    masks[i] = (masks[i] & 1) | m2
                    changed += 1
                used[kind] = used.get(kind, 0) + 1
                last_kind, last_dir, done = kind, d, True
                break
            if done:
                break
        if not done:
            last_kind = None
    return masks, changed, pairs, used


# ---------------- 폭타 끝 모양, 두께, 16분 '앞 줄 품고 더하기' 막기 ----------------
# 사람 277개: 16분 3줄+ 뒤 8분 이상 쉬는 끝 줄. short = 다음 줄까지 1박 이하(대부분 16분 한 칸 빔), long = 1박 넘게 쉼
END_K_HUMAN = {"short": {1: .133, 2: .228, 3: .339, 4: .206, 5: .078, 6: .016},
               "long": {1: .395, 2: .233, 3: .206, 4: .110, 5: .043, 6: .012}}
# 끝 줄 모양(8분 앞 줄 대비): 이동 = 무게중심 1.5칸 이상 옮김(7 -> 1 같은 대각), 벌어짐 = 폭 2칸 이상 넓어짐(4 -> 17),
# 모임 = 폭 2칸 이상 좁아짐(17 -> 4), 비슷 = 그 밖, 그대로 = 같은 건반
END_SHAPE_HUMAN = {"short": {"diag": .328, "sim": .289, "wide": .233, "narrow": .096, "same": .053},
                   "long": {"diag": .377, "sim": .256, "same": .155, "wide": .149, "narrow": .062}}
SHAPE_KO = {"diag": "이동", "sim": "비슷", "wide": "벌어짐", "narrow": "모임", "same": "그대로"}


def end_shape_class(P: int, E: int):
    a, b = _keys(P), _keys(E)
    if not a or not b:
        return None
    if (P & 0xFE) == (E & 0xFE):
        return "same"
    ca, cb = float(np.mean(a)), float(np.mean(b))
    wa, wb = max(a) - min(a), max(b) - min(b)
    if abs(cb - ca) >= 1.5:
        return "diag"
    if wb <= wa - 2:
        return "narrow"
    if wb >= wa + 2:
        return "wide"
    return "sim"


def stream_end_rows(t_chart, masks, bpm: float):
    """16분 3줄 뒤 8분 이상 쉬는 끝 줄. 반환 [(i, 'short'|'long', 8분 앞 줄 번호)]"""
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(t)
    B = 60.0 / bpm
    s16 = B / 4 * 1.15
    out = []
    for i in range(3, n):
        gn = t[i + 1] - t[i] if i < n - 1 else 1e9
        if not (t[i] - t[i - 1] <= s16 and t[i - 1] - t[i - 2] <= s16 and t[i - 2] - t[i - 3] <= s16):
            continue
        if gn < B / 2 * 0.95 or not (int(masks[i]) & 0xFE):
            continue
        j = int(np.argmin(np.abs(t[:i] - (t[i] - B / 2))))
        p = j if abs(t[j] - (t[i] - B / 2)) < B / 16 else i - 1
        out.append((i, "short" if gn <= B * 1.05 else "long", p))
    return out


def end_shape(t_chart, masks, bpm: float, loud=None, strength: float = 1.0, seed: int = 0, max_chord: int = 6,
              over: int = 1, jack_ms: float = 100.0):
    """--end-shape: 폭타 끝 줄의 건반 수와 모양을 사람처럼
    반환: (masks, 바꾼 줄 수, 끝 수, {끝 줄: 허용 상한}, {모양: 수}, 끝 줄 평균 건반 (전, 후))"""
    from itertools import combinations
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    ends = stream_end_rows(t, masks, bpm)
    if strength <= 0 or not ends:
        return masks, 0, len(ends), {}, {}, (0.0, 0.0)
    rng = np.random.default_rng(seed + 727)
    lim = min(7, max_chord + max(0, over))
    lo = np.asarray(loud, dtype=np.float64) if loud is not None and len(loud) == len(masks) else rng.random(len(masks))
    before = float(np.mean([len(_keys(masks[i])) for i, _, _ in ends]))
    target = {}
    for gc in ("short", "long"):
        idx = [i for i, g, _ in ends if g == gc]
        if not idx:
            continue
        r = _rank01(lo[idx] + rng.random(len(idx)) * 1e-6)
        ks = sorted(END_K_HUMAN[gc])
        cum = np.cumsum([END_K_HUMAN[gc][k] for k in ks])
        cum = cum / cum[-1]
        for i, q in zip(idx, r):
            target[i] = min(lim, ks[int(np.searchsorted(cum, q, side="left"))])
    allow, changed, used = {}, 0, {}
    for i, gc, p in ends:
        allow[i] = lim
        if rng.random() >= strength:
            continue
        k = target[i]
        P = masks[p]
        block = masks[i - 1] | _near_mask(t, masks, i, jack_ms + 5.0)
        free = [c for c in range(1, 8) if not (block >> c) & 1]
        if len(free) < k:
            k = len(free)
        if k <= 0:
            continue
        w = dict(END_SHAPE_HUMAN[gc])
        cands = {}
        for comb in combinations(free, k):
            m = sum(1 << c for c in comb)
            c = end_shape_class(P, m)
            if c:
                cands.setdefault(c, []).append(m)
        kinds = [c for c in w if c in cands]
        if not kinds:
            continue
        # 모양마다 될 수 있는 끝 줄이 달라(1개짜리는 벌어짐이 거의 안 됨) 확률로만 고르면 이동이 넘치고 벌어짐이 모자람
        # -> 지금까지 고른 비율이 사람 비율보다 가장 모자란 모양을 고름 (같으면 사람 비율로 무작위)
        tot_ = sum(used.values()) + 1
        need = np.array([w[c] * tot_ - used.get(c, 0) for c in kinds]) + rng.random(len(kinds)) * 0.3
        c = kinds[int(np.argmax(need))]
        cur = masks[i] & 0xFE
        best = max(bin(m & cur).count("1") for m in cands[c])          # 지금 건반을 되도록 남김
        pick = [m for m in cands[c] if bin(m & cur).count("1") == best]
        new = int(pick[int(rng.integers(len(pick)))])
        if new != cur:
            masks[i] = (masks[i] & 1) | new
            changed += 1
        used[c] = used.get(c, 0) + 1
    after = float(np.mean([len(_keys(masks[i])) for i, _, _ in ends]))
    return masks, changed, len(ends), allow, used, (before, after)


def superset_fix(t_chart, masks, bpm: float, seed: int = 0, jack_ms: float = 100.0):
    """--superset-fix: 16분 간격에서 뒷줄이 앞 줄(2개+)을 그대로 품고 건반을 더한 모양(45 -> 457)을 푼다
    반환: (masks, 고친 줄 수, 찾은 수)"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    s16 = 60.0 / bpm / 4 * 1.15
    rng = np.random.default_rng(seed + 731)
    fixed = found = 0
    for i in range(1, len(masks)):
        if t[i] - t[i - 1] > s16:
            continue
        A, Bm = masks[i - 1] & 0xFE, masks[i] & 0xFE
        if bin(A).count("1") < 2 or (A & Bm) != A or Bm == A:
            continue
        found += 1
        block = A | Bm | _near_mask(t, masks, i, jack_ms + 5.0)
        if i < len(masks) - 1 and t[i + 1] - t[i] <= s16:
            block |= masks[i + 1]
        free = [c for c in range(1, 8) if not (block >> c) & 1]
        if not free:
            continue
        sh = _keys(A)
        a = int(rng.choice(sh))
        b = min(free, key=lambda c: (abs(c - a), rng.random()))
        masks[i] = (masks[i] & ~(1 << a)) | (1 << b)
        fixed += 1
    return masks, fixed, found


# ---------------- 박 단위 반복(모티프) ----------------
# 생성 8% / 6% / 0% -> 박마다 두께, 모양이 다 달라 축 옆이 난잡해 보임.
def beat_motif(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, max_chord: int = 6,
               jack_ms: float = 100.0, cnt_target: float = 0.30, exact_target: float = 0.04, passes: int = 3):
    """리듬(박 안 줄 자리)이 같은 바로 앞 박(없으면 2박 전)의 줄별 건반 수를 따라가게 하고, 일부는 줄 모양까지 그대로 옮겨 사람처럼 박 단위 반복을 만든다
    반환: (masks, 수 맞춘 박, 그대로 옮긴 박, 대상 박, (수패턴 같음 전, 후))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(t)
    if strength <= 0 or n < 8:
        return masks, 0, 0, 0, (0.0, 0.0)
    B = 60.0 / bpm
    s16 = B / 4 * 1.15
    slot = np.rint(t / B * 48).astype(np.int64)
    ok = np.abs(t / B * 48 - slot) < 0.3
    beats = {}
    for i in range(n):
        if ok[i]:
            beats.setdefault(int(slot[i] // 48), []).append(i)
    sig = {b: tuple(int(slot[i] % 48) for i in ix) for b, ix in beats.items()}
    FULL = (0, 12, 24, 36)
    cnt = lambda i: len(_keys(masks[i]))
    cpat = lambda b: [cnt(i) for i in beats[b]]
    rng = np.random.default_rng(seed + 733)

    def ref_of(b):
        for lag in (1, 2):
            p = b - lag
            if p in beats and sig[p] == sig[b] and len(beats[b]) >= 2:
                return p
        return None

    pairs = [(b, ref_of(b)) for b in sorted(beats)]
    pairs = [(b, p) for b, p in pairs if p is not None]
    if not pairs:
        return masks, 0, 0, 0, (0.0, 0.0)
    grp = lambda b: sig[b] == FULL

    def rate(g=None):
        a = s_ = 0
        for b in beats:
            p = b - 1
            if p in beats and sig[p] == sig[b] and len(beats[b]) >= 2 and (g is None or grp(b) == g):
                a += 1
                s_ += cpat(p) == cpat(b)
        return s_ / max(a, 1), a

    def blocked(i):
        bl = _near_mask(t, masks, i, jack_ms + 5.0)
        if i > 0 and t[i] - t[i - 1] <= s16:
            bl |= masks[i - 1]
        if i < n - 1 and t[i + 1] - t[i] <= s16:
            bl |= masks[i + 1]
        return bl

    r0 = rate()[0]
    n_cnt = n_ex = 0
    q_ex = exact_target * strength
    for b, p in pairs:                                   # 줄 모양까지 그대로 (연타가 생기면 되돌림)
        if rng.random() >= q_ex:
            continue
        src, dst = beats[p], beats[b]
        old = [masks[i] for i in dst]
        for i, j in zip(dst, src):
            masks[i] = (masks[i] & 1) | (masks[j] & 0xFE)
        if any((masks[i] & 0xFE) & blocked(i) for i in dst):
            for i, m in zip(dst, old):
                masks[i] = m
        else:
            n_ex += 1
    for _ in range(passes):
        for g in (True, False):
            rg, ag = rate(g)
            if ag == 0:
                continue
            diff = [(b, p) for b, p in pairs if grp(b) == g and cpat(p) != cpat(b)]
            q = min(1.0, max(0.0, (cnt_target - rg) * ag / max(len(diff), 1))) * strength
            for b, p in diff:
                if rng.random() >= q:
                    continue
                src, dst = beats[p], beats[b]
                old = [masks[i] for i in dst]
                done = True
                for i, j in zip(dst, src):
                    goal = min(max(1, cnt(j)), max_chord)
                    while cnt(i) > goal:
                        ks = _keys(masks[i])
                        nb = (masks[i - 1] if i > 0 else 0) | (masks[i + 1] if i < n - 1 else 0)
                        pref = [k for k in ks if (nb >> k) & 1] or ks
                        masks[i] &= ~(1 << int(rng.choice(pref)))
                    while cnt(i) < goal:
                        bl = blocked(i) | masks[i]
                        free = [c for c in range(1, 8) if not (bl >> c) & 1]
                        pref = [c for c in free if (masks[j] >> c) & 1] or free   # 앞 박 같은 자리 줄 건반 먼저
                        if not pref:
                            done = False
                            break
                        masks[i] |= 1 << int(rng.choice(pref))
                    if not done:
                        break
                if done:
                    n_cnt += 1
                else:
                    for i, m in zip(dst, old):
                        masks[i] = m
    return masks, n_cnt, n_ex, len(pairs), (r0, rate()[0])


# 생성 4% / 10% / 8% -> 줄마다 새 모양이라 축 옆이 난잡
def eighth_shape(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
                 t_e2: float = 0.12, t_s2: float = 0.25, t_s1: float = 0.18):
    """--eighth-shape: 8분 자리 줄을 사람처럼 반복, 교대 구조로
    반환: (masks, {종류: 바꾼 수}, 대상 줄 수, (전 비율 e2, s2, s1), (후 비율))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(t)
    if strength <= 0 or n < 8:
        return masks, {}, 0, (0, 0, 0), (0, 0, 0)
    B = 60.0 / bpm
    s16 = B / 4 * 1.15
    q = t / B * 2.0
    on8 = np.abs(q - np.round(q)) < 0.04
    idx8 = [i for i in range(n) if on8[i] and (masks[i] & 0xFE)]
    pos = {int(round(q[i])): i for i in idx8}
    trip = [(pos[k], pos[k - 1], pos[k - 2]) for k in sorted(pos) if k - 1 in pos and k - 2 in pos]
    rng = np.random.default_rng(seed + 739)

    def norm(m):
        k = _keys(m)
        return tuple(x - k[0] for x in k) if k else ()

    def rates():
        if not trip:
            return (0.0, 0.0, 0.0)
        e2 = s2 = s1 = 0
        for i, a, b in trip:
            mi, ma, mb = masks[i] & 0xFE, masks[a] & 0xFE, masks[b] & 0xFE
            e2 += mi == mb
            s2 += norm(mi) == norm(mb)
            s1 += norm(mi) == norm(ma)
        L = len(trip)
        return (e2 / L, s2 / L, s1 / L)

    def blocked(i):
        bl = _near_mask(t, masks, i, jack_ms + 5.0)
        if i > 0 and t[i] - t[i - 1] <= s16:
            bl |= masks[i - 1]
        if i < n - 1 and t[i + 1] - t[i] <= s16:
            bl |= masks[i + 1]
        return bl & 0xFE

    def shifts(m, bl, avoid_same=True):
        ks = _keys(m)
        out = []
        for d in range(-6, 7):
            if d == 0 and avoid_same:
                continue
            nk = [k + d for k in ks]
            if min(nk) < 1 or max(nk) > 7:
                continue
            mm = sum(1 << k for k in nk)
            if not (mm & bl):
                out.append(mm)
        return out

    r0 = rates()
    L = len(trip)
    elig = lambda i, j: len(_keys(masks[i])) == len(_keys(masks[j]))
    ne2 = sum(1 for i, a, b in trip if elig(i, b) and (masks[i] & 0xFE) != (masks[b] & 0xFE))
    ns2 = sum(1 for i, a, b in trip if elig(i, b) and norm(masks[i]) != norm(masks[b]))
    ns1 = sum(1 for i, a, b in trip if elig(i, a) and norm(masks[i]) != norm(masks[a]))
    q_e2 = min(1.0, max(0.0, (t_e2 - r0[0]) * L / max(ne2, 1))) * strength
    q_s2 = min(1.0, max(0.0, (t_s2 - max(r0[1], t_e2)) * L / max(ns2, 1))) * strength
    q_s1 = min(1.0, max(0.0, (t_s1 - r0[2]) * L / max(ns1, 1))) * strength
    used = {}
    for i, a, b in trip:
        mi = masks[i] & 0xFE
        bl = blocked(i)
        u = rng.random()
        new = None
        if elig(i, b) and mi != (masks[b] & 0xFE) and u < q_e2 and not ((masks[b] & 0xFE) & bl):
            new, kind = masks[b] & 0xFE, "e2"
        elif elig(i, b) and norm(mi) != norm(masks[b]) and rng.random() < q_s2:
            c = shifts(masks[b], bl)
            if c:
                best = max(bin(m & mi).count("1") for m in c)
                c = [m for m in c if bin(m & mi).count("1") == best]
                new, kind = int(c[int(rng.integers(len(c)))]), "s2"
        if new is None and elig(i, a) and norm(mi) != norm(masks[a]) and rng.random() < q_s1:
            c = shifts(masks[a], bl)
            if c:
                best = max(bin(m & mi).count("1") for m in c)
                c = [m for m in c if bin(m & mi).count("1") == best]
                new, kind = int(c[int(rng.integers(len(c)))]), "s1"
        if new is not None and new != mi:
            masks[i] = (masks[i] & 1) | new
            used[kind] = used.get(kind, 0) + 1
    return masks, used, L, r0, rates()


# ---------------- 1, 7번 축 풀기 ----------------
# 8분 자리 3줄 연속 같은 건반(축) 중 1, 7번 비율: 사람 277 39% / ★20 37% / 생성 54%
# 1, 7번 사용량은 비슷(36 vs 33%)한데 1, 7번이 몇 줄씩 이어져 양 끝 기둥처럼 보임. 축 수 자체는 생성이 더 적음(38 vs 52/100줄)
def _edge_axis_once(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
                    target: float = 0.40):
    """1, 7번 축이 target(사람 약 40%)보다 많으면, 3줄 넘게 이어지는 곳의 가운데 줄을 안쪽 빈 건반으로 옮긴다
    반환: (masks, 옮긴 수, (1, 7번 축 비율 전, 후))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(t)
    if strength <= 0 or n < 8:
        return masks, 0, (0.0, 0.0)
    B = 60.0 / bpm
    s16 = B / 4 * 1.15
    q = t / B * 2.0
    on8 = np.abs(q - np.round(q)) < 0.04
    pos = {int(round(q[i])): i for i in range(n) if on8[i]}
    keys8 = sorted(pos)
    rng = np.random.default_rng(seed + 751)

    def share():
        c = np.zeros(8)
        for k in keys8:
            if k + 1 in pos and k + 2 in pos:
                a = masks[pos[k]] & masks[pos[k + 1]] & masks[pos[k + 2]]
                for b in range(1, 8):
                    if (a >> b) & 1:
                        c[b] += 1
        tot = c[1:].sum()
        return ((c[1] + c[7]) / tot if tot else 0.0), c

    s0, c0 = share()
    if s0 <= target:
        return masks, 0, (s0, s0)
    # 1, 7번 런 찾기
    runs = []
    for e in (1, 7):
        cur = []
        for k in keys8:
            i = pos[k]
            if (masks[i] >> e) & 1 and (not cur or k == cur[-1][0] + 1):
                cur.append((k, i))
            else:
                if len(cur) >= 3:
                    runs.append((e, [i_ for _, i_ in cur]))
                cur = [(k, i)] if (masks[i] >> e) & 1 else []
        if len(cur) >= 3:
            runs.append((e, [i_ for _, i_ in cur]))
    rng.shuffle(runs)
    tot = c0[1:].sum()
    excess = (c0[1] + c0[7]) - target * tot          # 대략 없애야 할 1, 7번 축 수
    inner = {1: (2, 3), 7: (6, 5)}                  # 가까운 안쪽만 (멀리 튀면 모양이 깨짐)
    moved = 0
    for e, idx in runs:
        if excess <= 0:
            break
        if rng.random() >= strength:
            continue
        # 런 안에서 가운데 줄들(양 끝 제외)을 하나 걸러 옮김: 3줄 -> 가운데 1개, 5줄 -> 2, 4번째
        for j in range(1, len(idx) - 1, 2):
            i = idx[j]
            if not (masks[i] >> e) & 1:
                continue
            bl = _near_mask(t, masks, i, jack_ms + 5.0) | masks[i]
            if i > 0 and t[i] - t[i - 1] <= s16:
                bl |= masks[i - 1]
            if i < n - 1 and t[i + 1] - t[i] <= s16:
                bl |= masks[i + 1]
            for c in inner[e]:
                if not (bl >> c) & 1:
                    masks[i] = (masks[i] & ~(1 << e)) | (1 << c)
                    moved += 1
                    excess -= min(2, len(idx) - 2)       # 가운데 하나를 빼면 그 줄이 낀 3연속이 최대 3개 깨짐
                    break
    return masks, moved, (s0, share()[0])


def edge_axis(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
              target: float = 0.40, passes: int = 4):
    """_edge_axis_once를 1, 7번 축 비율이 target 아래로 갈 때까지 최대 passes번 (옮기면 런이 쪼개져 남은 런을 다시 찾음).
    반환: (masks, 옮긴 수, (전, 후))"""
    s0 = None
    total = 0
    s1 = 0.0
    for k in range(passes):
        masks, mv, (a, s1) = _edge_axis_once(t_chart, masks, bpm, strength, seed + 17 * k, jack_ms, target)
        s0 = a if s0 is None else s0
        total += mv
        if mv == 0 or s1 <= target:
            break
    return masks, total, (s0 or 0.0, s1)


# ---------------- 폭타 8분+16분 쌍 합 일정하게 ----------------
# 바쁜 마디(16분 칸 8+/16)에서 '8분 자리 줄 + 바로 뒤 16분 줄' 건반 합의 마디 안 표준편차:
# 사람 ★20 0.71 / 생성 0.93. 사람 폭타는 쌍 합이 5, 6처럼 거의 일정하게 이어짐.
def pair_even(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, max_chord: int = 6,
              jack_ms: float = 100.0, min_fill: int = 8, pair_min: int = 0, targets=None, keep_var: bool = False):
    """바쁜 마디마다 쌍 합을 그 마디 중앙값(목표)에 맞춘다
    반환: (masks, 바꾼 줄 수, (표준편차 중앙값 전, 후))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(t)
    if strength <= 0 or n < 8:
        return masks, 0, (0.0, 0.0)
    B = 60.0 / bpm
    s16 = B / 4 * 1.15
    sl = np.rint(t / B * 48).astype(np.int64)
    ok = np.abs(t / B * 48 - sl) < 0.3
    at = {int(sl[i]): i for i in range(n) if ok[i]}
    cnt = lambda i: len(_keys(masks[i]))
    rng = np.random.default_rng(seed + 761)

    def meas_pairs(w):
        out = []
        for j in range(8):
            a, b = at.get(w + 24 * j), at.get(w + 24 * j + 12)
            if a is not None:
                out.append((a, b))
        return out

    meas = sorted(set(int(x) // 192 for x in at))
    busy = [m for m in meas if sum((m * 192 + 12 * j) in at for j in range(16)) >= min_fill]

    def stds():
        v = []
        for m in busy:
            ps = [cnt(a) + (cnt(b) if b is not None else 0) for a, b in meas_pairs(m * 192)]
            if len(ps) >= 4:
                v.append(float(np.std(ps)))
        return float(np.median(v)) if v else 0.0

    def free_keys(i):
        bl = _near_mask(t, masks, i, jack_ms + 5.0) | masks[i]
        if i > 0 and t[i] - t[i - 1] <= s16:
            bl |= masks[i - 1]
        if i < n - 1 and t[i + 1] - t[i] <= s16:
            bl |= masks[i + 1]
        return [c for c in range(1, 8) if not (bl >> c) & 1]

    def add(i):
        if cnt(i) >= max_chord:
            return False
        fr = free_keys(i)
        if not fr:
            return False
        ks = _keys(masks[i])
        # 이미 있는 건반 옆을 피하고 고르게: 가장 가까운 기존 건반과 2칸 이상 떨어진 곳 먼저
        far = [c for c in fr if not ks or min(abs(c - k) for k in ks) >= 2] or fr
        masks[i] |= 1 << int(rng.choice(far))
        return True

    def rem(i):
        if cnt(i) <= 1:
            return False
        ks = _keys(masks[i])
        nb = (masks[i - 1] if i > 0 else 0) | (masks[i + 1] if i < n - 1 else 0)
        pref = [k for k in ks if (nb >> k) & 1] or ks
        masks[i] &= ~(1 << int(rng.choice(pref)))
        return True

    s0 = stds()
    changed = set()
    for m in busy:
        prs = meas_pairs(m * 192)
        ps = [cnt(a) + (cnt(b) if b is not None else 0) for a, b in prs]
        if len(ps) < 4:
            continue
        T = int(round(float(np.median(ps))))
        if targets is not None:                       # 4마디 묶음 소리 크기로 정한 목표 (--pair-avg)
            if m not in targets:
                continue
            T = int(targets[m])
        elif pair_min > 0 and T < pair_min:
            T = min(pair_min, T + 1)
        shift = T - int(round(float(np.median(ps)))) if keep_var else 0
        for (a, b), s in zip(prs, ps):
            if keep_var:
                # 마디 쌍 합 높낮이(편차)는 남기고 마디 전체를 목표 쪽으로 옮김 + 목표에서 2 이상 벗어난 쌍만 1 안으로.
                # 예전처럼 모든 쌍을 목표에 딱 맞추면 마디 안 표준편차 0.78 -> 0.33(사람 0.71)이 되어
                # 꽉 찬 박 건반 수 패턴이 2-1-2-1 / 2-1-1-1로 몰림
                goal = s + shift
                if abs(goal - T) >= 2:
                    goal = T + (1 if goal > T else -1)
                if goal == s:
                    continue
            else:
                d = s - T
                if d == 0:
                    continue
                if abs(d) == 1 and rng.random() >= 0.5 * strength:
                    continue
                goal = T if (abs(d) == 1 or targets is not None) else T + (1 if d > 0 else -1)
            for _ in range(abs(s - goal)):
                cur = cnt(a) + (cnt(b) if b is not None else 0)
                if cur < goal:
                    did = (b is not None and cnt(b) < cnt(a) - 1 and add(b)) or add(a) or (b is not None and add(b))
                    if did:
                        changed.add(a); changed.add(b)
                elif cur > goal:
                    did = (b is not None and cnt(b) >= cnt(a) and rem(b)) or rem(a) or (b is not None and rem(b))
                    if did:
                        changed.add(a); changed.add(b)
    changed.discard(None)
    return masks, len(changed), (s0, stds())


# ---------------- 폭타 쌍 합 평균 입력 (구간 = 4마디 묶음, 크기 = lowmid) ----------------
# pair_level_check: 쌍 합 4마디 묶음과 상관이 full, bgm 둘 다 높은 기준은
# lowmid. db, 드럼 층, 초당 어택 수는 약함.
# 묶음 안 표준편차 4마디 0.40 < 8마디 0.45~0.50. 곡 안 마디 쌍 합 하위~상위 10% 폭 2.55, 곡 평균 10/50/90% 2.65/3.7/4.63.
def block_lowmid(mel, bpm: float, pad: float, n_meas: int, block: int = 4, fps: int = 100):
    """4마디 묶음마다 16분 격자 시각(±2프레임 최댓값)의 저, 중음(40~2000Hz) mel 평균. 반환 (묶음 수,)"""
    from scratch_model import mel_centers
    m = np.asarray(mel, dtype=np.float32)
    hz = mel_centers(m.shape[1])
    lm = m[:, (hz >= 40) & (hz < 2000)].mean(axis=1)
    T = len(lm)
    B = 60.0 / bpm
    nb = (n_meas + block - 1) // block
    out = np.full(nb, np.nan)
    for b in range(nb):
        cells = np.arange(b * block * 16, min((b + 1) * block, n_meas) * 16)
        fr = np.rint((cells * B / 4 - pad) * fps).astype(np.int64)
        fr = fr[(fr >= 0) & (fr < T)]
        if len(fr):
            out[b] = float(np.mean([lm[max(0, f - 2):f + 3].max() for f in fr]))
    return out


def pair_targets(t_chart, masks, bpm: float, blk_loud, pair_avg: float, spread: float = 3.2, block: int = 4,
                 seed: int = 0, min_fill: int = 8, jitter: float = 0.2):
    """바쁜 마디가 있는 4마디 묶음을 소리 크기 순위대로 세워 목표 쌍 합 = pair_avg + spread * (순위 - 0.5)로 정한다"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    sl = np.rint(t / B * 48).astype(np.int64)
    occ = set(sl[np.abs(t / B * 48 - sl) < 0.3].tolist())
    meas = sorted(set((sl // 192).tolist()))
    busy = [m for m in meas if sum((m * 192 + 12 * j) in occ for j in range(16)) >= min_fill]
    blocks = sorted(set(m // block for m in busy if m // block < len(blk_loud) and np.isfinite(blk_loud[m // block])))
    if not blocks:
        return {}
    lo = np.array([blk_loud[b] for b in blocks])
    r = _rank01(lo) if len(blocks) > 1 else np.array([0.5])
    rng = np.random.default_rng(seed + 797)
    tgt = {}
    for b, q in zip(blocks, r):
        v = float(np.clip(pair_avg + spread * (q - 0.5), 1.0, 8.0))
        tgt[b] = int(np.floor(v) + (rng.random() < v - np.floor(v)))
    out = {}
    for m in busy:
        if m // block not in tgt:
            continue
        v = tgt[m // block]
        u = rng.random()                              # 묶음 안도 사람처럼 조금 흔들리게 (사람 묶음 안 표준편차 0.40)
        if u < jitter / 2:
            v -= 1
        elif u < jitter:
            v += 1
        out[m] = int(max(1, v))
    return out


# ---------------- 12비트 3연 강-약-중 ----------------
def trip12_accent(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, eq_target: float = 0.3):
    """12비트 3연(박 + 1/3 + 2/3)에서 둘째, 셋째가 첫째보다 두꺼우면 첫째 이하로 줄이고, 셋 다 같은 개수인 곳이 사람보다 많으면 둘째에서 하나 뺌
    반환: (masks, 바꾼 줄 수, 3연 수)"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(masks)
    if strength <= 0 or n < 3:
        return masks, 0, 0
    B = 60.0 / bpm
    pos = {}
    for i in range(n):
        q = t[i] / B * 12.0                                   # 1/12박 격자
        if abs(q - round(q)) < 0.1 and masks[i] & 0xFE:
            pos[int(round(q))] = i
    groups = []
    for p in sorted(pos):
        if p % 12 == 0 and p + 4 in pos and p + 8 in pos and not any(p + x in pos for x in (3, 6, 9)):
            groups.append((pos[p], pos[p + 4], pos[p + 8]))
    if not groups:
        return masks, 0, 0
    rng = np.random.default_rng(seed + 307)
    cnt = lambda i: len(_keys(masks[i]))
    changed = set()

    def drop(i):
        ks = _keys(masks[i])
        if len(ks) <= 1:
            return False
        nb = ((masks[i - 1] if i > 0 else 0) | (masks[i + 1] if i < n - 1 else 0)) & 0xFE
        rep = [x for x in ks if (nb >> x) & 1]
        inner = ks[1:-1] if len(ks) > 2 else ks
        adj = [x for x in inner if (x - 1 in ks) or (x + 1 in ks)]
        masks[i] &= ~(1 << int(rng.choice(rep or adj or inner)))
        changed.add(i)
        return True
    for a, b, c in groups:
        if rng.random() >= strength:
            continue
        if cnt(b) > cnt(a):
            goal = max(1, cnt(a) - 1) if rng.random() < 0.7 else cnt(a)
            while cnt(b) > goal and drop(b):
                pass
        if cnt(c) > cnt(a):
            while cnt(c) > cnt(a) and drop(c):
                pass
    eq = [(a, b, c) for a, b, c in groups if cnt(a) == cnt(b) == cnt(c) and cnt(a) >= 2]
    p = min(1.0, max(0.0, (len(eq) - eq_target * len(groups)) / max(len(eq), 1))) * strength
    for a, b, c in eq:
        if rng.random() < p:
            drop(b)
    return masks, len(changed), len(groups)


def _quarter_drift_at(t, masks, i, B):
    """4비트로 이어지는 세 줄(i-2, i-1, i)이 한쪽 손 범위 안에서 끝 건반으로 밀려가면 그 방향(+1 오른쪽, -1 왼쪽)"""
    if i < 2 or t[i - 1] - t[i - 2] < 0.99 * B or t[i] - t[i - 1] < 0.99 * B:
        return 0
    ks = [_keys(masks[j]) for j in (i - 2, i - 1, i)]
    if any(not k for k in ks):
        return 0
    allk = [x for k in ks for x in k]
    c = [float(np.mean(k)) for k in ks]
    if all(x >= 4 for x in allk) and 7 in ks[2] and c[2] > max(c[0], c[1]):
        return 1
    if all(x <= 3 for x in allk) and 1 in ks[2] and c[2] < min(c[0], c[1]):
        return -1
    return 0


def quarter_drift(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0):
    """--quarter-spread: 4비트 세 줄이 한쪽 손 범위 안에서 가운데 -> 끝으로 밀려가면(5 4 47) 마지막 줄을 반대쪽 손으로
    반환: (masks, 고친 줄 수, (전 비율, 후 비율))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(masks)
    B = 60.0 / bpm

    def rate():
        tot = hit = 0
        for i in range(2, n):
            if t[i - 1] - t[i - 2] >= 0.99 * B and t[i] - t[i - 1] >= 0.99 * B and all(_keys(masks[j]) for j in (i - 2, i - 1, i)):
                tot += 1
                hit += _quarter_drift_at(t, masks, i, B) != 0
        return hit / tot if tot else 0.0

    r0 = rate()
    if strength <= 0 or n < 3:
        return masks, 0, (r0, r0)
    rng = np.random.default_rng(seed + 883)
    fixed = 0
    for i in range(2, n):
        d = _quarter_drift_at(t, masks, i, B)
        if not d or rng.random() >= strength:
            continue
        ks = _keys(masks[i])
        near = _near_mask(t, masks, i, jack_ms + 5.0)
        cands = [[8 - k for k in ks]]                       # 반대쪽 손 (47 -> 41)
        for sh in (1, 2, 3):                                # 가운데 쪽으로 sh칸 (47 -> 36 -> 25)
            cands.append([k - d * sh for k in ks])
        for c in cands:
            if any(not 1 <= k <= 7 for k in c) or len(set(c)) != len(c):
                continue
            nm = (masks[i] & 1) | sum(1 << k for k in c)
            if nm & near & 0xFE:
                continue
            old = masks[i]
            masks[i] = nm
            if _quarter_drift_at(t, masks, i, B) == 0 and (i + 1 >= n or _quarter_drift_at(t, masks, i + 1, B) == 0) \
                    and (i + 2 >= n or _quarter_drift_at(t, masks, i + 2, B) == 0):
                fixed += 1
                break
            masks[i] = old
    return masks, fixed, (r0, rate())


_TRILL_HANDS = ((1, 2, 3), (4, 5, 6, 7))      # 4번은 오른손


def _hand_trill_runs(t, masks, B, min_len: int = 4):
    """한 손 안에서 붙은 두 건반(12, 23, 45, 56, 67)이 16분 이하 간격으로 번갈아 나오는 줄 묶음
    반환: [(시작, 끝, 손, a, c)]"""
    n = len(masks)
    out = []
    for hand in _TRILL_HANDS:
        hs = set(hand)
        for a in hand[:-1]:
            c = a + 1

            def side(x):
                ks = {k for k in range(1, 8) if (int(x) >> k) & 1} & hs
                return 1 if ks == {a} else (2 if ks == {c} else 0)
            i = 0
            while i < n:
                sd = side(masks[i])
                if not sd:
                    i += 1
                    continue
                j = i
                while j + 1 < n and t[j + 1] - t[j] <= 0.26 * B and side(masks[j + 1]) == 3 - side(masks[j]):
                    j += 1
                if j - i + 1 >= min_len:
                    out.append((i, j, hand, a, c))
                i = max(j, i + 1)
    return out


def trill_break(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0):
    """--trill-break: 한 손 붙은 두 건반 트릴(67 67, 12 12 ...) 4줄 이상을 끊음
    반환: (masks, 바꾼 줄 수, (전, 후 1000줄당))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(masks)
    B = 60.0 / bpm
    r0 = len(_hand_trill_runs(t, masks, B)) * 1000.0 / max(n, 1)
    if strength <= 0 or n < 4:
        return masks, 0, (r0, r0)
    rng = np.random.default_rng(seed + 884)
    changed = 0
    for _ in range(3):
        runs = _hand_trill_runs(t, masks, B)
        if not runs:
            break
        for i, j, hand, a, c in runs:
            if rng.random() >= strength:
                continue
            for x in range(i + 2, j + 1, 3):
                cur = a if (masks[x] >> a) & 1 else c
                near = _near_mask(t, masks, x, jack_ms + 5.0)
                free = [k for k in hand if k not in (a, c) and not (masks[x] >> k) & 1 and not (near >> k) & 1]
                if not free:
                    continue
                d = np.array([abs(k - cur) for k in free], dtype=np.float64) + 0.01 * rng.random(len(free))
                k = free[int(np.argmin(d))]
                masks[x] = (masks[x] & ~(1 << cur)) | (1 << k)
                changed += 1
    r1 = len(_hand_trill_runs(t, masks, B)) * 1000.0 / max(n, 1)
    return masks, changed, (r0, r1)


def eighth_audit(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
                 max_chord: int = 7, bp_target=(0.41, 0.40, 0.15), sk_target=(0.64, 0.30, 0.03)):
    """--eighth-audit: 마지막 점검
    반환: (masks, 바꾼 줄 수, {'bp': (전, 후), 'sk': (전, 후)})"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    cnt = lambda i: len(_keys(masks[i]))

    def measure():
        kind = row_kind(t, bpm)
        pos = _grid16(t, bpm)
        bp, sk = [], []
        for p in sorted(pos):
            i = pos[p]
            if kind[i] != 16 or not masks[i] & 0xFE:
                continue
            if p % 4 == 0 and p + 2 in pos and kind[pos[p + 2]] == 16 and masks[pos[p + 2]] & 0xFE:
                bp.append((i, pos[p + 2], pos.get(p + 3)))
            if p % 2 == 0 and p + 1 in pos and masks[pos[p + 1]] & 0xFE:
                sk.append((i, pos[p + 1]))
        return bp, sk

    def ratios(bp, sk):
        r1 = np.mean([cnt(a) == cnt(b) for a, b, _ in bp]) if bp else 0.0
        r2 = np.mean([cnt(a) > cnt(b) for a, b in sk]) if sk else 0.0
        return float(r1), float(r2)

    bp, sk = measure()
    b0, s0 = ratios(bp, sk)
    if strength <= 0 or len(bp) < 8:
        return masks, 0, {"bp": (b0, b0), "sk": (s0, s0)}
    rng = np.random.default_rng(seed + 886)
    changed = 0

    def free_key(i, near_ex=()):
        near = _near_mask(t, masks, i, jack_ms + 5.0) | masks[i]
        ks = _keys(masks[i])
        fr = [c for c in range(1, 8) if not (near >> c) & 1]
        if not fr:
            return None
        d = np.array([min((abs(c - k) for k in ks), default=2) for c in fr], dtype=np.float64)
        w = np.where(d >= 2, 1.0, 0.2) + 0.01
        return int(rng.choice(fr, p=w / w.sum()))

    def drop_key(i):
        ks = _keys(masks[i])
        pm = masks[i - 1] & 0xFE if i > 0 else 0
        rep = [k for k in ks if (pm >> k) & 1]
        inner = ks[1:-1] if len(ks) > 2 else ks
        adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
        return int(rng.choice(rep or adj or inner))

    # 1) 박 안 두 8분 자리: 같음이 넘친 만큼 -> 첫째가 큼
    same = [x for x in bp if cnt(x[0]) == cnt(x[1])]
    ex = int(round((len(same) - bp_target[0] * len(bp)) * min(strength, 1.0)))
    if ex > 0:
        order = rng.permutation(len(same))
        done, flip = 0, 0
        for k in order:
            if done >= ex:
                break
            a, b, nx = same[int(k)]
            if cnt(a) != cnt(b):
                continue
            want_drop = flip % 2 == 0
            ok = False
            for mode in ((0, 1) if want_drop else (1, 0)):
                if mode == 0 and cnt(b) >= 2 and (nx is None or cnt(b) - 1 >= cnt(nx)):
                    masks[b] &= ~(1 << drop_key(b))
                    ok = True
                elif mode == 1 and cnt(a) < max_chord:
                    c = free_key(a)
                    if c is not None:
                        masks[a] |= 1 << c
                        ok = True
                if ok:
                    flip += 1
                    break
            if ok:
                done += 1
                changed += 1

    # 2) 뼈대: '8분이 큼'이 넘친 만큼, 2개+ 차이 쌍에서 하나를 뒤 줄로 (합 그대로)
    bp, sk = measure()
    big = [x for x in sk if cnt(x[0]) > cnt(x[1])]
    ex = int(round((len(big) - sk_target[0] * len(sk)) * min(strength, 1.0)))
    if ex > 0:
        cand = [x for x in big if cnt(x[0]) - cnt(x[1]) >= 2]
        for k in rng.permutation(len(cand))[:ex]:
            i, j = cand[int(k)]
            if cnt(i) - cnt(j) < 2:
                continue
            c = free_key(j)
            if c is None or (masks[i] >> c) & 1 and False:
                continue
            masks[i] &= ~(1 << drop_key(i))
            masks[j] |= 1 << c
            changed += 1
    bp, sk = measure()
    b1, s1 = ratios(bp, sk)
    return masks, changed, {"bp": (b0, b1), "sk": (s0, s1)}


def hand_balance(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
                 ratio: float = 1.42, win_meas: int = 8):
    """--hand-balance: 폭타 밖(16분 연타가 아닌 곳) 한 손 줄의 좌우 쏠림 풀기
    반환: (masks, 바꾼 줄 수, (전, 후 오른손만/왼손만 배))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(masks)
    B = 60.0 / bpm
    kind = row_kind(t, bpm)
    meas = np.floor(t / (4 * B) + 1e-9).astype(np.int64)

    def side(m):
        ks = _keys(m)
        if not ks:
            return 0
        if all(k >= 4 for k in ks):
            return 1
        if all(k <= 3 for k in ks):
            return -1
        return 0

    def flip(m, to_left):
        ks = _keys(m)
        if to_left:
            nk = [8 - k if k != 4 else 3 for k in ks]
        else:
            nk = [8 - k for k in ks]
        if len(set(nk)) != len(nk):
            return None
        return (m & 1) | sum(1 << k for k in nk)

    ns = [i for i in range(n) if kind[i] != 16 and masks[i] & 0xFE]

    def overall():
        r = sum(1 for i in ns if side(masks[i]) == 1)
        l = sum(1 for i in ns if side(masks[i]) == -1)
        return r / max(l, 1)

    r0 = overall()
    if strength <= 0 or len(ns) < 8:
        return masks, 0, (r0, r0)
    rng = np.random.default_rng(seed + 887)
    changed = 0
    R = [i for i in ns if side(masks[i]) == 1]
    L = [i for i in ns if side(masks[i]) == -1]
    if len(R) > ratio * len(L):
        big, to_left, ex = R, True, int(np.floor((len(R) - ratio * len(L)) / (1.0 + ratio) * min(strength, 1.0)))
    elif len(L) > len(R):                                    # 왼손 쏠림은 사람(오른손이 1.42배)과 반대라 왼손이 더 많을 때만 같게
        big, to_left, ex = L, False, int(np.floor((len(L) - len(R)) / 2.0 * min(strength, 1.0)))
    else:
        return masks, 0, (r0, r0)
    # 앞뒤 win_meas마디 안 쏠림이 심한 줄부터 (구간마다 고르게 풀리도록)
    sgn = 1 if to_left else -1
    def local(i):
        w = [j for j in ns if abs(int(meas[j]) - int(meas[i])) <= win_meas // 2]
        return sum(1 for j in w if side(masks[j]) == sgn) - ratio * sum(1 for j in w if side(masks[j]) == -sgn)
    order = sorted(big, key=lambda i: -(local(i) + rng.random()))
    for i in order:
        if ex <= 0:
            break
        if side(masks[i]) != sgn or local(i) <= 0:
            continue
        nm = flip(masks[i], to_left)
        if nm is None or nm & _near_mask(t, masks, i, jack_ms + 5.0) & 0xFE:
            continue
        if (i > 0 and (masks[i - 1] & 0xFE) == (nm & 0xFE)) or (i + 1 < n and (masks[i + 1] & 0xFE) == (nm & 0xFE)):
            continue
        masks[i] = nm
        changed += 1
        ex -= 1
    return masks, changed, (r0, overall())


def fine_density(t_chart, masks, bpm: float, strength: float = 1.0, seed: int = 0, jack_ms: float = 100.0,
                 ratio: float = 1.13, single: float = 0.84, win_meas: int = 4):
    """--fine-density: 폭타 속 32, 24분 줄이 낀 박이 사람보다 빽빽해지는 것 줄이기
    반환: (masks, 뺀 건반 수, {'single': (전, 후), 'ratio': (전, 후)})"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    n = len(masks)
    B = 60.0 / bpm
    kind = row_kind(t, bpm)
    q = t / B * 4.0
    fine = np.abs(q - np.round(q)) > 1e-3
    on8 = (~fine) & (np.round(q).astype(np.int64) % 2 == 0)
    beat = np.floor(t / B + 1e-9).astype(np.int64)
    cnt = lambda i: len(_keys(masks[i]))
    rng = np.random.default_rng(seed + 888)

    def drop_one(i):
        ks = _keys(masks[i])
        pm = masks[i - 1] & 0xFE if i > 0 else 0
        rep = [k for k in ks if (pm >> k) & 1]
        inner = ks[1:-1] if len(ks) > 2 else ks
        adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
        masks[i] &= ~(1 << int(rng.choice(rep or adj or inner)))

    fr = [i for i in range(n) if kind[i] == 16 and fine[i] and masks[i] & 0xFE]

    def single_rate():
        return float(np.mean([cnt(i) == 1 for i in fr])) if fr else 1.0

    beats = {}
    for i in range(n):
        if kind[i] == 16:
            beats.setdefault(int(beat[i]), []).append(i)
    stream_beats = {b: ix for b, ix in beats.items() if all(kind[j] == 16 for j in range(ix[0], ix[-1] + 1))}
    fbeats = [b for b, ix in stream_beats.items() if any(fine[j] for j in ix)]
    plain = {b: sum(cnt(j) for j in ix) for b, ix in stream_beats.items() if not any(fine[j] for j in ix) and len(ix) >= 4}
    bpm_meas = 4

    def ratio_now():
        fk = [sum(cnt(j) for j in stream_beats[b]) for b in fbeats]
        pk = list(plain.values())
        return float(np.mean(fk) / np.mean(pk)) if fk and pk else 0.0

    s0, q0 = single_rate(), ratio_now()
    if strength <= 0 or not fr:
        return masks, 0, {"single": (s0, s0), "ratio": (q0, q0)}
    removed = 0
    multi = [i for i in fr if cnt(i) >= 2]
    ex = int(round((single * len(fr) - sum(cnt(i) == 1 for i in fr)) * min(strength, 1.0)))
    for k in rng.permutation(len(multi)):
        if ex <= 0:
            break
        i = multi[int(k)]
        while cnt(i) >= 2:
            drop_one(i)
            removed += 1
        ex -= 1
    for b in fbeats:
        ix = stream_beats[b]
        near = [pk for bb, pk in plain.items() if abs(bb - b) <= win_meas * bpm_meas]
        if len(near) < 2:
            continue
        goal = np.mean(near) * ratio
        cur = sum(cnt(j) for j in ix)
        if rng.random() >= strength:
            continue
        cand = []
        for j in ix:
            if fine[j]:
                for nb in (j - 1, j + 1):
                    if 0 <= nb < n and kind[nb] == 16 and not fine[nb] and not on8[nb] and nb not in cand:
                        cand.append(nb)
        while cur > goal and cand:
            cand.sort(key=lambda j: -cnt(j))
            j = cand[0]
            if cnt(j) <= 1:
                break
            drop_one(j)
            removed += 1
            cur -= 1
    return masks, removed, {"single": (s0, single_rate()), "ratio": (q0, ratio_now())}


def stream_break_pick(t_chart, row_prob, bpm: float, max_run: int = 16, min_fill: int = 12, mel=None, pad: float = 0.0,
                      fine_cnt=None, skip=()):
    """--stream-break: 16분 폭타 마디가 max_run마디 넘게 쉬지 않으면 8비트 동시치기로 바꿀 마디 묶음을 고른다
    반환: 바꿀 마디 번호 목록"""
    t = np.asarray(t_chart, dtype=np.float64)
    pr = np.asarray(row_prob, dtype=np.float64)
    if max_run <= 0 or len(t) < 32:
        return []
    B = 60.0 / bpm
    sl = np.rint(t / B * 4).astype(np.int64)
    on16 = np.abs(t / B * 4 - sl) < 0.08
    meas = sl // 16
    fill, offp = {}, {}
    for i in np.flatnonzero(on16):
        m = int(meas[i])
        fill.setdefault(m, set()).add(int(sl[i]) % 16)
        if sl[i] % 2 == 1:
            offp.setdefault(m, []).append(pr[i])
    if not fill:
        return []
    ms = list(range(min(fill), max(fill) + 1))
    skip = set(int(x) for x in skip)                 # 사다리로 얇게 한 마디는 폭타 줄기를 끊은 것으로 봄
    sset = {m for m in ms if len(fill.get(m, ())) >= min_fill and m not in skip}
    runs, cur = [], []
    for m in ms:
        if m in sset:
            cur.append(m)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    nov = {}
    if mel is not None:
        mm = np.asarray(mel, dtype=np.float32)
        def mvec(m):
            a_ = int(max(0, (m * 4 * B - pad) * 100)); b_ = int(max(a_ + 1, ((m + 1) * 4 * B - pad) * 100))
            seg = mm[a_:min(b_, len(mm))]
            return seg.mean(0) if len(seg) else None
        vecs = {m: mvec(m) for m in range(ms[0] - 1, ms[-1] + 2)}
        for m in ms:
            u, v = vecs.get(m), vecs.get(m - 1)
            if u is not None and v is not None:
                nov[m] = 1.0 - float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    picks = []
    for run in runs:
        if len(run) <= max_run:
            continue
        sc = {m: float(np.mean(offp[m])) if offp.get(m) else 1.0 for m in run}
        thr = float(np.percentile(list(sc.values()), 35))
        vals = sorted(sc.values())
        rank_sc = lambda v: np.searchsorted(vals, v) / max(len(vals) - 1, 1)
        nv = sorted(nov.get(m, 0.0) for m in run)
        rank_nv = lambda v: np.searchsorted(nv, v) / max(len(nv) - 1, 1)
        pos = {m: k for k, m in enumerate(run)}
        start = 0
        while len(run) - start > max_run:
            lo = start + max(2, max_run // 2)
            hi = min(start + max_run, len(run) - 3)
            best = None
            for k in range(lo, hi + 1):
                m = run[k]
                blocks = []
                if m % 4 == 0:
                    blocks += [[m], [m, m + 1]]
                if m % 4 == 3:
                    blocks += [[m], [m - 1, m]]
                for bl in blocks:
                    if any(x not in pos or pos[x] < lo or pos[x] > len(run) - 3 for x in bl):
                        continue
                    if fine_cnt and any(fine_cnt.get(x, 0) for x in bl):
                        continue                               # 같은 박 자리에 24, 32, 셋잇단이 되풀이되는 마디는 안 끊음
                    if len(bl) == 2 and not all(sc[x] <= thr for x in bl):
                        continue
                    score = float(np.mean([rank_sc(sc[x]) for x in bl])) - 0.6 * rank_nv(nov.get(bl[0], 0.0))
                    if best is None or score < best[0]:
                        best = (score, bl)
            if best is None:                           # 악절 경계가 없으면(박자 어긋남 등) 예전처럼 가장 약한 한 마디
                if lo > hi:
                    break
                ks_ = [kk for kk in range(lo, hi + 1) if not (fine_cnt and fine_cnt.get(run[kk], 0))]
                if not ks_:
                    break
                k = min(ks_, key=lambda kk: sc[run[kk]])
                best = (0.0, [run[k]])
            picks += best[1]
            start = pos[best[1][-1]] + 1
    return sorted(set(picks))


def chord_fill(t_chart, masks, bpm: float, chord_meas, seed: int = 0, max_chord: int = 7, ratio: float = 0.76,
               min_fill: int = 12, ratio_map=None):
    """폭타를 끊고 8비트 동시치기로 바꾼 마디의 건반 수를 주변(앞뒤 4마디) 폭타 마디 건반 수 x ratio(사람 0.76)로 맞춤
    반환: (masks, 바꾼 건반 수, [(마디, 전, 후, 목표)])"""
    masks = [int(m) for m in masks]
    if not chord_meas:
        return masks, 0, []
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    sl = np.rint(t / B * 4).astype(np.int64)
    meas = np.floor(t / (4 * B) + 1e-9).astype(np.int64)
    cnt = lambda i: len(_keys(masks[i]))
    rows_of = {}
    for i in range(len(t)):
        rows_of.setdefault(int(meas[i]), []).append(i)
    stream_tot = {m: sum(cnt(i) for i in ix) for m, ix in rows_of.items()
                  if len({int(sl[i]) % 16 for i in ix}) >= min_fill and m not in chord_meas}
    rng = np.random.default_rng(seed + 890)
    changed, rep = 0, []
    for m in chord_meas:
        ix = [i for i in rows_of.get(m, []) if masks[i] & 0xFE]
        near = []
        for w_ in (4, 8, 16):                           # 8마디 덩어리 가운데는 주변 폭타가 멀어서 넓혀 봄
            near = [v for mm, v in stream_tot.items() if abs(mm - m) <= w_]
            if near:
                break
        if not ix or not near:
            continue
        goal = int(round(np.mean(near) * (ratio_map.get(m, ratio) if ratio_map else ratio)))
        goal = min(goal, max_chord * len(ix))
        cur = sum(cnt(i) for i in ix)
        before = cur
        guard = 0
        while cur != goal and guard < 200:
            guard += 1
            if cur < goal:
                i = min(ix, key=lambda j: (cnt(j), rng.random()))
                if cnt(i) >= max_chord:
                    break
                nb = (masks[i - 1] if i > 0 else 0) | (masks[i + 1] if i + 1 < len(masks) else 0)
                free = [k for k in range(1, 8) if not (masks[i] >> k) & 1]
                pref = [k for k in free if not (nb >> k) & 1] or free
                masks[i] |= 1 << int(rng.choice(pref))
                cur += 1
            else:
                i = max(ix, key=lambda j: (cnt(j), rng.random()))
                if cnt(i) <= 1:
                    break
                ks = _keys(masks[i])
                pm = masks[i - 1] if i > 0 else 0
                rep_k = [k for k in ks if (pm >> k) & 1]
                masks[i] &= ~(1 << int(rng.choice(rep_k or ks)))
                cur -= 1
            changed += 1
        rep.append((m, before, cur, goal))
    return masks, changed, rep


def _meas_fill(t, bpm):
    """마디별 16분 칸 채움 집합과 16분 뒷자리 줄 번호"""
    B = 60.0 / bpm
    sl = np.rint(t / B * 4).astype(np.int64)
    on16 = np.abs(t / B * 4 - sl) < 0.08
    fill, offi = {}, {}
    for i in np.flatnonzero(on16):
        m = int(sl[i] // 16)
        fill.setdefault(m, set()).add(int(sl[i]) % 16)
        if sl[i] % 2 == 1:
            offi.setdefault(m, []).append(i)
    return fill, offi


def section_break_pick(t_chart, row_prob, bpm: float, target: float = 0.64, min_fill: int = 12,
                       mel=None, pad: float = 0.0, max_blocks: int = 4, fine_cnt=None, fine_min: int = 2):
    """--stream-target: 16분 폭타 마디 비율이 target(사람 중앙값 64%)을 넘으면 4마디 악절째 가벼운 구간으로 바꿀 마디를 고른다
    반환: 바꿀 마디 번호 목록"""
    t = np.asarray(t_chart, dtype=np.float64)
    pr = np.asarray(row_prob, dtype=np.float64)
    if target <= 0 or len(t) < 64:
        return []
    B = 60.0 / bpm
    fill, offi = _meas_fill(t, bpm)
    if not fill:
        return []
    ms = list(range(min(fill), max(fill) + 1))
    act = [m for m in ms if fill.get(m)]
    S = {m for m in ms if len(fill.get(m, ())) >= min_fill}
    need = len(S) - target * len(act)
    if need < 3 or len(S) < 16:
        return []
    nov, loud = {}, {}
    if mel is not None:
        mm = np.asarray(mel, dtype=np.float32)
        st_ = _sound_stats(mm, bpm, pad)               # --bass-sound
        vec = {}
        for m in range(ms[0] - 1, ms[-1] + 2):
            a_ = int(max(0, (m * 4 * B - pad) * 100)); b_ = int(max(a_ + 1, ((m + 1) * 4 * B - pad) * 100))
            seg = mm[a_:min(b_, len(mm))]
            if len(seg):
                vec[m] = seg.mean(0)
                loud[m] = _sound_val(seg, st_)
        for m in ms:
            u, v = vec.get(m), vec.get(m - 1)
            if u is not None and v is not None:
                nov[m] = 1.0 - float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    ph = 0
    if nov:
        ph = int(np.argmax([np.mean([nov[m] for m in ms if m % 4 == o and m in nov] or [0]) for o in range(4)]))
    sc = {m: float(np.mean(pr[offi[m]])) if offi.get(m) else 1.0 for m in S}
    def rk(d):
        vs = sorted(d.values())
        return {k: np.searchsorted(vs, v) / max(len(vs) - 1, 1) for k, v in d.items()}
    r_sc = rk(sc)
    r_ld = rk({m: loud.get(m, 0.0) for m in S})
    r_nv = rk({m: nov.get(m, 0.0) for m in ms})
    first_s, last_s = min(S), max(S)
    phrases = {}
    for p in range(ms[0] - 4, ms[-1] + 1):
        if (p - ph) % 4:
            continue
        bl = [p, p + 1, p + 2, p + 3]
        if not all(x in S for x in bl):
            continue
        if p - first_s < 4 or last_s - (p + 3) < 4:              # 곡 첫, 끝 폭타 4마디는 남김
            continue
        if fine_cnt and sum(fine_cnt.get(x, 0) for x in bl) >= fine_min:
            continue                                             # 같은 박 자리에 24, 32, 셋잇단이 되풀이되는 악절은 리듬이 핵심
        phrases[p] = float(np.mean([r_sc[x] + r_ld[x] for x in bl])) - 0.5 * r_nv.get(p, 0.0)
    if not phrases:
        return []
    med = float(np.median(list(phrases.values())))
    picks, blocks = set(), []
    for p in sorted(phrases, key=phrases.get):
        if len(picks) >= need - 1 or len(blocks) >= max_blocks:
            break
        if p in picks:
            continue
        bl = [p, p + 1, p + 2, p + 3]
        # 이웃 악절도 약하면 8마디로 (사람 덩어리 길이 중앙값 8)
        nxt = [q for q in (p + 4, p - 4) if q in phrases and phrases[q] <= med and q not in picks]
        if nxt and len(picks) + 8 <= need + 2:
            q = nxt[0]
            bl = sorted(bl + [q, q + 1, q + 2, q + 3])
        if any(abs(x - y) < 8 for x in bl for y in picks):
            continue
        picks.update(bl)
        blocks.append(bl)
    return sorted(picks)


def stream_thicken(t_chart, masks, bpm: float, target: float = 2.28, max_chord: int = 7, exclude=(),
                   mel=None, pad: float = 0.0, seed: int = 0, min_fill: int = 12):
    """--stream-kpr: 16분 폭타 마디의 줄당 건반을 target(사람 폭타 마디 중앙값 2.28)까지 올림
    반환: (masks, 더한 건반 수, (전 줄당, 후 줄당))"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    if target <= 0 or len(t) < 32:
        return masks, 0, (0.0, 0.0)
    B = 60.0 / bpm
    sl = np.rint(t / B * 4).astype(np.int64)
    on16 = np.abs(t / B * 4 - sl) < 0.08
    fill, _ = _meas_fill(t, bpm)
    ex = set(int(x) for x in exclude)
    S = [m for m, f in fill.items() if len(f) >= min_fill and m not in ex]
    if not S:
        return masks, 0, (0.0, 0.0)
    rows = {}
    for i in range(len(t)):
        m = int(np.floor(t[i] / (4 * B) + 1e-9))
        if m in S and masks[i] & 0xFE:
            rows.setdefault(m, []).append(i)
    cnt = lambda i: len(_keys(masks[i]))
    k0 = sum(cnt(i) for ix in rows.values() for i in ix) / max(sum(len(ix) for ix in rows.values()), 1)
    loud = {}
    if mel is not None:
        mm = np.asarray(mel, dtype=np.float32)
        st_ = _sound_stats(mm, bpm, pad)               # --bass-sound
        for m in rows:
            a_ = int(max(0, (m * 4 * B - pad) * 100)); b_ = int(max(a_ + 1, ((m + 1) * 4 * B - pad) * 100))
            seg = mm[a_:min(b_, len(mm))]
            loud[m] = _sound_val(seg, st_) if len(seg) else 0.0
    vs = sorted(loud.values()) if loud else []
    rng = np.random.default_rng(seed + 920)
    added = 0
    for m, ix in rows.items():
        r = np.searchsorted(vs, loud[m]) / max(len(vs) - 1, 1) if vs else 0.5
        goal = int(round((target + 0.6 * (r - 0.5)) * len(ix)))
        cur = sum(cnt(i) for i in ix)
        guard = 0
        while cur < goal and guard < 100:
            guard += 1
            cand = []
            for i in ix:
                if not on16[i]:
                    continue                                    # 32, 24분 줄은 단타 유지
                cap = min(max_chord, 4 if sl[i] % 2 == 0 else 3)
                if cnt(i) < cap:
                    cand.append((cnt(i) - (0.5 if sl[i] % 2 == 0 else 0.0), rng.random(), i))
            if not cand:
                break
            i = min(cand)[2]
            nb = (masks[i - 1] if i > 0 else 0) | (masks[i + 1] if i + 1 < len(masks) else 0)
            free = [k for k in range(1, 8) if not (masks[i] >> k) & 1]
            pref = [k for k in free if not (nb >> k) & 1] or free
            masks[i] |= 1 << int(rng.choice(pref))
            cur += 1
            added += 1
    k1 = sum(cnt(i) for ix in rows.values() for i in ix) / max(sum(len(ix) for ix in rows.values()), 1)
    return masks, added, (k0, k1)


SECTION_KIND_KO = {"C": "8비트 동시치기", "C1": "8비트 단타", "H": "반쯤 빈 16비트", "P": "드문드문", "L": "사다리 폭타"}
# --ladder: 폭타 사다리 - 16분 줄은 그대로 두고 두께만 (8분 자리 건반, 16분 뒷자리 건반)
LADDER = [(1, 1.0), (2, 1.0), (2, 2.0), (3, 1.5), (3, 2.0), (4, 2.0)]
LADDER_KO = ["1+1", "2+1", "2+2", "3+(1·2)", "3+2", "4+"]
# 반쯤 빈 16비트 16분 뒷자리 수(사람): 1개 27 / 2개 32 / 3개 27 / 4개 10 / 5개+ 4% - 소리 크기와 무관.
# 8분 자리 8줄이 다 있으면 뒷자리 4개부터는 16분 칸 12/16이라 폭타로 셈 -> 1 / 2 / 3개(3개+ 41%)까지
HALF_OFF_Q = [(0.27, 1), (0.59, 2), (1.01, 3)]


def ladder_idx(a: float, b: float) -> int:
    """마디 8분 자리 평균 건반 a, 16분 뒷자리 평균 건반 b -> 가장 가까운 사다리 단계 (kinds.classify와 같은 기준)"""
    if a >= 3.5:
        return 5
    return int(min(range(5), key=lambda j: (LADDER[j][0] - a) ** 2 + (LADDER[j][1] - b) ** 2))


def meas_rungs(t_chart, masks, bpm: float, min_fill: int = 12) -> dict:
    """폭타 마디(16분 칸 min_fill/16 이상)마다 사다리 단계 {마디: 0~5} (스크래치 빼고 셈)"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q = t / B * 4.0
    qi = np.rint(q).astype(np.int64)
    on16 = np.abs(q - qi) < 0.08
    on, off, fill = {}, {}, {}
    for i in np.flatnonzero(on16):
        k = bin(int(masks[i]) & 0xFE).count("1")
        if not k:
            continue
        m, p = int(qi[i] // 16), int(qi[i] % 16)
        fill.setdefault(m, set()).add(p)
        (on if p % 2 == 0 else off).setdefault(m, []).append(k)
    return {m: ladder_idx(float(np.mean(on.get(m, [0]))), float(np.mean(off.get(m, [0]))))
            for m, f in fill.items() if len(f) >= min_fill}


def rank_rung(r: float) -> int:
    """바꾼 구간 사다리 단계 상한 - 곡 안 소리 크기 순위 0.30~0.45 1+1 / 0.45~0.65 2+1 / 그 위 2+2"""
    return 0 if r < 0.45 else (1 if r < 0.65 else 2)


def ladder_plan(lad: dict, t_chart, masks, bpm: float, win: int = 8) -> dict:
    """{마디: (단계 상한 또는 None, 주변 대비 몇 단계 아래)} -> {마디: 단계}"""
    rg = meas_rungs(t_chart, masks, bpm)
    base = {m: v for m, v in rg.items() if m not in lad}
    out = {}
    for m, plan in lad.items():
        cap, down = plan[0], plan[1]
        lo = int(plan[2]) if len(plan) > 2 else 0          # 박 강세 - 8분 자리 소리가 뚜렷하면 최소 2+1
        nb = [v for mm, v in base.items() if abs(mm - m) <= win]
        if not nb:
            nb = [v for mm, v in base.items() if abs(mm - m) <= 2 * win]
        lv = int(np.median(nb)) - int(down) if nb else (cap if cap is not None else 0)
        if cap is not None:
            lv = min(lv, int(cap))
        lv = max(lv, lo)
        if nb:
            lv = min(lv, int(np.median(nb)))                # 주변 폭타보다 두껍게는 안 함
        out[m] = max(0, int(lv))
    return out


def ladder_fill(t_chart, masks, bpm: float, rung: dict, seed: int = 0, max_chord: int = 7, jack_ms: float = 100.0,
                add: bool = True, keep_idx=()):
    """--ladder: 고른 마디 16분 줄 건반 수를 사다리 단계에 맞춤 (줄은 그대로)
    반환: (masks, 바꾼 건반 수, [(마디, 전 줄당, 후 줄당, 단계)])"""
    masks = [int(m) for m in masks]
    if not rung:
        return masks, 0, []
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q = t / B * 4.0
    qi = np.rint(q).astype(np.int64)
    on16 = np.abs(q - qi) < 0.08
    meas = np.floor(t / (4 * B) + 1e-6).astype(np.int64)
    rng = np.random.default_rng(seed + 1060)
    cnt = lambda i: len(_keys(masks[i]))
    changed, rep = 0, []
    kset = set(int(i) for i in keep_idx)            # 치고 쉬기 큰 소리 줄 등은 안 건드림
    for m, lv in sorted(rung.items()):
        ix = [int(i) for i in np.flatnonzero(meas == m) if masks[i] & 0xFE and int(i) not in kset]
        if not ix:
            continue
        a, b = LADDER[min(max(int(lv), 0), 5)]
        k0 = float(np.mean([cnt(i) for i in ix]))
        for i in ix:
            if not on16[i]:
                goal = 1
            elif qi[i] % 2 == 0:
                goal = a
            else:
                goal = int(b) if b == int(b) else (1 if qi[i] % 4 == 1 else 2)
            goal = min(goal, max_chord)
            while cnt(i) > goal:
                ks = _keys(masks[i])
                pm = masks[i - 1] if i > 0 else 0
                nm = masks[i + 1] if i + 1 < len(masks) else 0
                rep_k = [k for k in ks if (pm >> k) & 1] or [k for k in ks if (nm >> k) & 1]
                masks[i] &= ~(1 << int(rng.choice(rep_k or ks)))
                changed += 1
            while add and cnt(i) < goal:
                ks = _keys(masks[i])
                near = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)
                free = [k for k in range(1, 8) if not (near >> k) & 1]
                if not free:
                    break
                d = np.array([min(abs(k - x) for x in ks) for k in free], dtype=np.float64) if ks else np.ones(len(free))
                wk = np.where(d >= 2, 1.0, 0.25) + 0.01
                masks[i] |= 1 << int(rng.choice(free, p=wk / wk.sum()))
                changed += 1
        rep.append((m, k0, float(np.mean([cnt(i) for i in ix])), int(lv)))
    return masks, changed, rep


def half_off_counts(t_chart, row_prob, bpm: float, ms, ref: float) -> dict:
    """반쯤 빈 16비트 마디마다 남길 16분 뒷자리 수(1~3)를 모델 확률 순위와 사람 분포(1개 27 / 2개 32 / 3개+ 41%)로 정한다"""
    t = np.asarray(t_chart, dtype=np.float64)
    pr = np.asarray(row_prob, dtype=np.float64)
    _, offi = _meas_fill(t, bpm)
    sc = {m: float(sum(1 for i in offi.get(m, []) if pr[i] >= ref)) + 0.01 * float(np.mean(pr[offi[m]]) if offi.get(m) else 0.0)
          for m in ms}
    if not sc:
        return {}
    vs = sorted(sc.values())
    out = {}
    for m, v in sc.items():
        r = np.searchsorted(vs, v) / max(len(vs) - 1, 1) if len(vs) > 1 else 0.5
        out[m] = next(k for qq, k in HALF_OFF_Q if r < qq)
    return out
# 사람 189곡(BPM 150~200) 폭타 사이 4마디+ 덩어리 안 마디 종류와 건반 수(주변 폭타 마디 대비 중앙값)
SECTION_KIND_RATIO = {"C": 0.67, "C1": 0.28, "H": 0.65, "P": 0.16}


# --bass-sound: 마디 소리 크기에 저음(멜 0~10번 대역 = 베이스)도 넣음.
#   멜 평균은 대역 96개 중 저음이 11개뿐이라 베이스가 커도 거의 안 움직임.
#   곡 안 z값 (1-w) 전체 + w 저음이 가장 잘 맞음.
#   순위로만 쓰는 곳(악절 종류, 폭타 구간 고르기, 반쯤 빈 마디 채우기, 쉬는 동시치기 크기 등)에만 적용, 곡 소리 폭은 예전 그대로.
SOUND_LOW_W = 0.0
LOW_BANDS = 11


def _low_stats(mm, bpm: float, pad: float):
    """곡 전체 마디 멜 평균(전체, 저음)의 평균, 표준편차 (z값용)"""
    B = 60.0 / bpm
    n = int((len(mm) / 100.0 + pad) / (4 * B)) + 1
    va, vl = [], []
    for m in range(n):
        a_ = int(max(0, (m * 4 * B - pad) * 100)); b_ = int(((m + 1) * 4 * B - pad) * 100)
        seg = mm[a_:min(b_, len(mm))]
        if len(seg) >= 20:
            va.append(float(seg.mean())); vl.append(float(seg[:, :LOW_BANDS].mean()))
    if len(va) < 8:
        return None
    return float(np.mean(va)), float(np.std(va)) + 1e-6, float(np.mean(vl)), float(np.std(vl)) + 1e-6


def _sound_val(seg, st) -> float:
    v = float(seg.mean())
    if SOUND_LOW_W <= 0 or st is None or seg.ndim < 2:
        return v
    return (1.0 - SOUND_LOW_W) * (v - st[0]) / st[1] + SOUND_LOW_W * (float(seg[:, :LOW_BANDS].mean()) - st[2]) / st[3]


def _sound_stats(mm, bpm: float, pad: float):
    return _low_stats(mm, bpm, pad) if SOUND_LOW_W > 0 and mm.ndim == 2 else None


def _meas_loud(mel, bpm: float, pad: float, ms, raw: bool = False):
    """마디 소리 크기 {마디: 값}. raw=False이고 --bass-sound면 곡 안 z값 (1-w) 전체 + w 저음 (순위로만 쓸 것)"""
    B = 60.0 / bpm
    mm = np.asarray(mel, dtype=np.float32)
    st = None if raw else _sound_stats(mm, bpm, pad)
    out = {}
    for m in ms:
        a_ = int(max(0, (m * 4 * B - pad) * 100)); b_ = int(max(a_ + 1, ((m + 1) * 4 * B - pad) * 100))
        seg = mm[a_:min(b_, len(mm))]
        out[m] = (float(seg.mean()) if raw else _sound_val(seg, st)) if len(seg) else 0.0
    return out


def bass_gap_meas(mel, bpm: float, pad: float, n_meas: int, gap: float = 0.3) -> list:
    """로그용: 곡 안 저음 순위가 멜 전체 순위보다 gap 이상 높은 마디 (베이스가 큰데 멜 평균은 조용한 곳)"""
    if mel is None:
        return []
    B = 60.0 / bpm
    mm = np.asarray(mel, dtype=np.float32)
    if mm.ndim != 2:
        return []
    ms, va, vl = [], [], []
    for m in range(n_meas):
        a_ = int(max(0, (m * 4 * B - pad) * 100)); b_ = int(((m + 1) * 4 * B - pad) * 100)
        seg = mm[a_:min(b_, len(mm))]
        if len(seg) >= 20:
            ms.append(m); va.append(float(seg.mean())); vl.append(float(seg[:, :LOW_BANDS].mean()))
    if len(ms) < 8:
        return []
    ra, rl = _rank01(np.array(va)), _rank01(np.array(vl))
    return [m for m, a, b in zip(ms, ra, rl) if b - a >= gap]


def meas_dev_loud(mel, bpm: float, pad: float, n_meas: int, w: float = 0.3) -> dict:
    """--dens-dev: 마디 소리 크기 절대값 (1-w) 멜 전체 평균 + w 저음 평균 (사람 편차 곡선과 같은 기준)"""
    if mel is None:
        return {}
    B = 60.0 / bpm
    mm = np.asarray(mel, dtype=np.float32)
    w = float(np.clip(w, 0.0, 1.0))
    out = {}
    for m in range(n_meas):
        a_ = int(max(0, (m * 4 * B - pad) * 100)); b_ = int(((m + 1) * 4 * B - pad) * 100)
        seg = mm[a_:min(b_, len(mm))]
        if len(seg) >= 20:
            out[m] = (1.0 - w) * float(seg.mean()) + (w * float(seg[:, :LOW_BANDS].mean()) if mm.ndim == 2 else w * float(seg.mean()))
    return out


def section_kinds(sec_meas, bpm: float, n_meas: int, mel=None, pad: float = 0.0, t_chart=None, row_prob=None,
                  ladder: bool = False, info=None):
    """악절째 바꿀 마디(section_break_pick)를 4마디 단위로 나눠 종류를 정함
    반환: ({마디: 종류}, {마디: 곡 전체 마디 중 소리 크기 순위 0~1})"""
    sec = sorted(int(m) for m in sec_meas)
    if not sec:
        return {}, {}
    units, cur = [], []
    for m in sec:
        if cur and (m != cur[-1] + 1 or len(cur) >= 4):
            units.append(cur); cur = []
        cur.append(m)
    if cur:
        units.append(cur)
    loud = _meas_loud(mel, bpm, pad, range(n_meas)) if mel is not None else {m: 0.0 for m in range(n_meas)}
    vs = sorted(loud.values())
    rk = {m: float(np.searchsorted(vs, v) / max(len(vs) - 1, 1)) for m, v in loud.items()}
    off = {}
    ref16 = None
    if t_chart is not None and row_prob is not None:
        fill, offi = _meas_fill(np.asarray(t_chart, dtype=np.float64), bpm)
        pr = np.asarray(row_prob, dtype=np.float64)
        off = {m: float(np.mean(pr[ix])) for m, ix in offi.items() if len(ix)}
        sv = [off[m] for m, f in fill.items() if len(f) >= 12 and m in off and m not in set(sec)]
        ref16 = float(np.median(sv)) if sv else None
        if ladder:
            sva = sorted(off[m] for m, f in fill.items() if len(f) >= 12 and m in off)
            orank = {m: float(np.searchsorted(sva, off[m]) / max(len(sva) - 1, 1)) for m in sec if m in off} if sva else {}
            if info is not None:
                info["ref16"] = float(np.median([pr[i] for m, ix in offi.items() if len(fill.get(m, ())) >= 12 for i in ix])
                                      ) if offi else None
                info["orank"] = orank
    kinds = {}
    for u in units:
        r = float(np.mean([rk.get(m, 0.5) for m in u]))
        o = float(np.mean([off.get(m, 0.0) for m in u])) if off else 0.0
        if ladder:
            orr = float(np.mean([orank.get(m, 0.5) for m in u])) if off else 0.5
            if r < 0.20:
                kd = "P"
            elif r < 0.30:
                kd = "C1"
            elif orr < 0.25:
                kd = "C" if r >= 0.70 else "H"
            else:
                kd = "L"
            for m in u:
                kinds[m] = kd
            continue
        if r < 0.25:
            kd = "P"
        elif r < 0.40:
            kd = "C1"
        elif r >= 0.60 and ref16 is not None and o >= ref16:
            kd = "H"
        else:
            kd = "C"
        for m in u:
            kinds[m] = kd
    return kinds, rk


def section_ratio(kind: str, rank: float) -> float:
    """바꾼 마디 건반 수 = 주변 폭타 마디 x 이 배수. 8비트 동시치기는 소리 크기 순위로 0.49~0.88."""
    if kind == "C":
        return float(np.clip(0.67 + 0.4 * (rank - 0.5), 0.49, 0.88))
    return SECTION_KIND_RATIO.get(kind, 0.76)


def fill_targets(t_chart, bpm: float, min_fill: float, mel=None, pad: float = 0.0, seed: int = 0):
    """--fill-grade: 폭타 잇기를 단계식으로
    반환: {마디: 목표 칸 수}"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    sl = np.rint(t / B * 48).astype(np.int64)
    occ = set(sl.tolist())
    cnt = {}
    for mi in sorted(set((sl // 192).tolist())):
        o = sum(1 for j in range(16) if mi * 192 + 12 * j in occ)
        if o >= min_fill:
            cnt[mi] = o
    if not cnt:
        return {}
    loud = _meas_loud(mel, bpm, pad, cnt) if mel is not None else {m: float(o) for m, o in cnt.items()}
    vs = sorted(loud.values())
    rng = np.random.default_rng(seed + 930)
    tg = {}
    for m, o in cnt.items():
        r = np.searchsorted(vs, loud[m]) / max(len(vs) - 1, 1)
        if r >= 0.36:
            g = 16
        elif r >= 0.19:
            g = int(rng.choice([14, 15]))
        else:
            g = int(rng.choice([12, 13]))
        tg[m] = max(o, g)
    return tg


def accent_rest_pick(t_chart, bpm: float, curve, pad: float = 0.0, ratio: float = 2.0, post: float = 0.6,
                     rest: float = 1.0, fps: int = 100):
    """--accent-rest: 폭타가 이어지다 '따당' 하고 큰 소리가 난 뒤 소리가 잦아드는 곳 -> 치고 잠깐 쉼
    반환: (뺄 줄 번호 배열, 큰 소리 줄 번호 목록, [(큰 소리 시각 초, 세기, 앞 세기, 뒤 최대)])"""
    t = np.asarray(t_chart, dtype=np.float64)
    cv = np.asarray(curve, dtype=np.float64)
    n = len(t)
    if n < 32 or len(cv) < 10 or ratio <= 0:
        return np.zeros(0, dtype=np.int64), [], []
    B = 60.0 / bpm

    def st(x):
        f = int(round((x - pad) * fps))
        lo, hi = max(0, f - 3), min(len(cv), f + 4)
        return float(cv[lo:hi].max()) if hi > lo else 0.0

    def amax(x0, x1):
        a_, b_ = int(round((x0 - pad) * fps)), int(round((x1 - pad) * fps))
        a_, b_ = max(0, a_), min(len(cv), b_)
        return float(cv[a_:b_].max()) if b_ > a_ else 0.0

    S = np.array([st(x) for x in t])
    floor = float(np.percentile(S, 60))
    drop, hits, info = set(), [], []
    last = -1e9
    for i in range(n):
        if S[i] < floor or t[i] - last < 2 * B:
            continue
        prev = np.flatnonzero((t >= t[i] - B - 1e-6) & (t < t[i] - 1e-6))
        if len(prev) < 3:
            continue
        pre_ix = np.flatnonzero((t >= t[i] - 2 * B) & (t < t[i] - 0.1 * B))
        pre = float(np.median(S[pre_ix])) if len(pre_ix) else 0.0
        if S[i] < ratio * max(pre, 1e-6):
            continue
        h, first = i, None
        nxt = np.flatnonzero((t > t[i] + 0.15 * B) & (t <= t[i] + B))
        strong = [j for j in nxt if S[j] >= 0.8 * S[i]]
        if strong:                                     # '따당': 두 번째 큰 소리 기준
            first, h = i, strong[0]
        aft = amax(t[h] + 0.15 * B, t[h] + B)
        if aft > post * S[h] or aft > pre:             # 뒤 1박이 앞 폭타 보통 세기보다도 약해야(소리가 실제로 잦아듦)
            continue
        if first is None:
            # 앞쪽 '따'(큰 소리보다 작지만 폭타 8분 자리보다 확실히 센 소리)도 찾음.
            # 앞 8분 자리보다 확실히 세면 둘 다 치고 사이는 비움. 비슷하면 그냥 폭타.
            back = np.flatnonzero((t >= t[h] - B - 1e-6) & (t <= t[h] - 0.15 * B))
            if len(back):
                j = int(back[np.argmax(S[back])])
                pre8 = float(np.median([st(t[j] - k * B / 2) for k in (1, 2, 3, 4)]))
                if S[j] >= 0.5 * S[h] and S[j] >= 1.4 * max(pre8, 1e-6):
                    first = j
        rest_ix = np.flatnonzero((t > t[h] + 0.02 * B) & (t < t[h] + rest * B - 0.02 * B))
        drop.update(int(j) for j in rest_ix)
        if first is not None:
            mid = np.flatnonzero((t > t[first] + 0.02 * B) & (t < t[h] - 0.02 * B))
            drop.update(int(j) for j in mid if S[j] < 0.6 * S[first])
            hits.append(first)
        hits.append(h)
        info.append((float(t[h]), float(S[h]), pre, aft))
        last = t[h]
    drop -= set(hits)
    return np.array(sorted(drop), dtype=np.int64), hits, info


def accent_chord(masks, hit_idx, max_chord: int = 6, seed: int = 0, t_chart=None, jack_ms: float = 100.0):
    """치고 쉬기 큰 소리 줄을 동시치기로 (최소 3개, 최대 동시치기 안). 앞뒤 jack_ms 안 줄 건반은 피함. 반환 (masks, 더한 건반 수)"""
    masks = [int(m) for m in masks]
    rng = np.random.default_rng(seed + 940)
    t = np.asarray(t_chart, dtype=np.float64) if t_chart is not None else None
    added = 0
    goal = min(3, max_chord)
    for i in hit_idx:
        if i < 0 or i >= len(masks):
            continue
        bl = _near_mask(t, masks, i, jack_ms + 5.0) if t is not None else 0
        while len(_keys(masks[i])) < goal:
            free = [k for k in range(1, 8) if not (masks[i] >> k) & 1 and not (bl >> k) & 1]
            if not free:
                break
            ks = _keys(masks[i])
            far = [k for k in free if not ks or min(abs(k - x) for x in ks) >= 2] or free
            masks[i] |= 1 << int(rng.choice(far))
            added += 1
    return masks, added


def key_scale(masks, scale: float, seed: int = 0, t_chart=None, bpm: float = 150.0, jack_ms: float = 100.0,
              max_chord: int = 6, keep_idx=()):
    """--key-scale: 줄(리듬)은 그대로 두고 동시치기 크기만 scale배로
    반환: (masks, 바꾼 뒤 건반 수)"""
    masks = [int(m) for m in masks]
    rng = np.random.default_rng(seed + 950)
    t = np.asarray(t_chart, dtype=np.float64) if t_chart is not None else None
    B = 60.0 / bpm
    keep = set(int(i) for i in keep_idx)
    cnt = [len(_keys(m)) for m in masks]
    tot = sum(cnt)
    if tot == 0 or abs(scale - 1.0) < 1e-6:
        return masks, tot
    idx = [i for i in range(len(masks)) if cnt[i] > 0 and i not in keep]
    fixed = sum(cnt[i] for i in range(len(masks)) if i in keep)
    if scale < 1.0:
        extra = sum(cnt[i] - 1 for i in idx)
        want = scale * tot - fixed - len(idx)          # 남길 '1개 넘는 몫'
        g = float(np.clip(want / extra, 0.0, 1.0)) if extra > 0 else 1.0
        goal = {i: 1 + int(np.floor((cnt[i] - 1) * g + rng.random())) for i in idx}
    else:
        goal = {i: min(max(cnt[i], int(np.floor(cnt[i] * scale + rng.random()))), max(max_chord, cnt[i])) for i in idx}
    for i in idx:
        c = goal[i]
        while len(_keys(masks[i])) > c:
            ks = _keys(masks[i])
            pm = _prev8_mask(t, masks, i, B) if t is not None else 0
            rep_ = [k for k in ks if (pm >> k) & 1]
            inner = ks[1:-1] if len(ks) > 2 else ks
            adj = [k for k in inner if (k - 1 in ks) or (k + 1 in ks)]
            masks[i] &= ~(1 << int(rng.choice(rep_ or adj or ks)))
        while len(_keys(masks[i])) < c:
            ks = _keys(masks[i])
            near = masks[i] | (_near_mask(t, masks, i, jack_ms + 5.0) if t is not None else 0)
            pm = _prev8_mask(t, masks, i, B) if t is not None else 0
            free = [k for k in range(1, 8) if not ((near | pm) >> k) & 1] or \
                [k for k in range(1, 8) if not (near >> k) & 1]
            if not free:
                break
            d = np.array([min(abs(k - x) for x in ks) for k in free], dtype=np.float64) if ks else np.ones(len(free))
            wk = np.where(d >= 2, 1.0, 0.25) + 0.01
            masks[i] |= 1 << int(rng.choice(free, p=wk / wk.sum()))
    return masks, sum(len(_keys(m)) for m in masks)


# 사람 lv15~22 179곡, 2마디 묶음 밀도(초당 노트, 스크래치 포함) / 곡 최고(상위 5%) 의 분위 0, 5,..., 100% (곡별 중앙값)
DENS_Q = np.linspace(0.0, 1.0, 21)
DENS_HUMAN = np.array([0.08, 0.16, 0.22, 0.30, 0.35, 0.41, 0.49, 0.55, 0.62, 0.67, 0.72,
                       0.77, 0.81, 0.83, 0.87, 0.89, 0.91, 0.94, 0.97, 1.00, 1.07])
# --dens-flat: 소리 편차가 거의 없는 곡은 듬성듬성 대신 조금 쉬는 폭타로.
# 사람 277개를 곡 소리 폭으로 나눠 본 2마디 묶음 밀도/최고 분포.
# 소리 폭 좁은 곡은 가벼운 묶음도 덜 가벼움: 하위 10% 0.38 (넓은 곡 0.18), 드문드문 마디 8% (넓은 곡 18%), 순위상관 -0.45.
DENS_BAND_X = np.array([0.05, 0.077, 0.105, 0.15])       # 소리 폭 ~0.065 / 0.065~0.09 / 0.09~0.12 / 0.12~ 의 대략 가운데
DENS_BAND = np.array([
    [0.19, 0.29, 0.38, 0.43, 0.51, 0.57, 0.64, 0.69, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.87, 0.89, 0.91, 0.93, 0.96, 1.00, 1.09],
    [0.11, 0.21, 0.30, 0.36, 0.42, 0.49, 0.54, 0.59, 0.64, 0.69, 0.74, 0.77, 0.80, 0.83, 0.87, 0.89, 0.92, 0.94, 0.97, 1.00, 1.08],
    [0.08, 0.19, 0.29, 0.36, 0.41, 0.48, 0.53, 0.58, 0.64, 0.69, 0.73, 0.75, 0.79, 0.82, 0.86, 0.88, 0.91, 0.93, 0.97, 1.00, 1.08],
    [0.06, 0.13, 0.18, 0.24, 0.31, 0.37, 0.44, 0.52, 0.58, 0.64, 0.68, 0.73, 0.78, 0.80, 0.84, 0.87, 0.90, 0.93, 0.96, 1.00, 1.09]])


# --dens-dev: 사람 277개 2마디 묶음 밀도/곡 최고(묶음 상위 5%) 중앙값 vs 편차
#   (묶음 소리 = 0.7 멜 전체 평균 + 0.3 저음 평균, 편차 = 묶음 소리 - 곡 묶음 상위 10% 소리, 묶음 13154개)
DEV_X = np.array([-0.30, -0.24, -0.19, -0.15, -0.12, -0.095, -0.075, -0.06, -0.045, -0.03, -0.015, 0.0, 0.015, 0.03])
DEV_MED = np.array([0.16, 0.18, 0.18, 0.21, 0.27, 0.36, 0.44, 0.53, 0.63, 0.73, 0.82, 0.90, 0.98, 1.03])


def dens_curve(loud_range: float) -> np.ndarray:
    """곡 소리 폭에 맞는 사람 2마디 묶음 밀도/최고 분포 (DENS_Q 분위마다, 소리 폭 사이는 선형)"""
    return np.array([np.interp(loud_range, DENS_BAND_X, DENS_BAND[:, j]) for j in range(DENS_BAND.shape[1])])


def loud_range(mel, bpm: float, pad: float, n_meas: int) -> float:
    """곡 소리 폭 = 마디 멜 평균의 10~90% 폭 (사람 277개 분석과 같은 기준)"""
    if mel is None:
        return float("nan")
    v = np.array(list(_meas_loud(mel, bpm, pad, range(n_meas), raw=True).values()))
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.percentile(v, 90) - np.percentile(v, 10)) if len(v) >= 8 else float("nan")


def stream_rows(t_chart, bpm: float, min_fill: int = 12) -> np.ndarray:
    """줄마다 폭타 마디(16분 칸 min_fill/16 이상 참) 안인지 (bool 배열)"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    meas = np.floor(t / (4 * B) + 1e-6).astype(np.int64)
    q16 = t / B * 4.0
    on16 = np.abs(q16 - np.round(q16)) < 1e-3
    qi = np.round(q16).astype(np.int64)
    slots = {}
    for i in np.flatnonzero(on16):
        slots.setdefault(int(meas[i]), set()).add(int(qi[i]) % 16)
    S = {m for m, s_ in slots.items() if len(s_) >= min_fill}
    return np.array([int(m) in S for m in meas], dtype=bool)


# 사람 277개 '최고(2마디 상위 10% 평균)의 0.85배 이상'이 이어진 가장 긴 시간, BPM대 곡별 중앙값(초)
PEAK_RUN_BPM = np.array([135.0, 165.0, 190.0, 215.0])
PEAK_RUN_S = np.array([20.9, 16.4, 12.4, 12.0])


def peak_run_meas(bpm: float) -> int:
    """최고 근처가 이어져도 되는 마디 수 (사람 BPM대 가장 긴 유지 중앙값)"""
    sec = float(np.interp(bpm, PEAK_RUN_BPM, PEAK_RUN_S))
    return max(4, int(round(sec / (240.0 / bpm))))


def dens_shape(t_chart, masks, bpm: float, row_loud, row_prob, strength: float = 1.0, cap: float = 0.0,
               keep_idx=(), seed: int = 0, tol: float = 0.10, max_run: int = 0, protect=None, guard: bool = False,
               brk_keys_ok=None, loud_range=None, dev_loud=None, pulse_meas=None):
    """--dens-shape, --dens-cap: 구간 밀도 분포를 사람처럼
    반환: (남길 줄 bool 배열, masks, 보고 dict)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    keep = np.ones(n, dtype=bool)
    rep = {"rows": 0, "keys": 0, "blocks": 0, "breaks": 0, "pulse_rows": 0}
    prot = np.zeros(n, dtype=bool) if protect is None else np.asarray(protect, dtype=bool)
    if strength <= 0 or n < 64:
        return keep, masks, rep
    rng = np.random.default_rng(seed + 960)
    B = 60.0 / bpm
    bar = 4 * B
    dur = 2 * bar
    blk = np.floor(t / dur + 1e-6).astype(np.int64)
    c = np.array([bin(m & 0xFF).count("1") for m in masks], dtype=np.float64)
    ks = sorted(set(blk.tolist()))
    rows_of = {k: np.flatnonzero(blk == k) for k in ks}
    d = np.array([c[rows_of[k]].sum() / dur for k in ks])
    if len(ks) < 8 or d.max() <= 0:
        return keep, masks, rep
    P = float(np.percentile(d, 95))
    Pref = min(P, cap) if cap > 0 else P
    L = np.asarray(row_loud, dtype=np.float64)
    loud = np.array([float(np.mean(L[rows_of[k]])) for k in ks])
    r = _rank01(0.5 * _rank01(loud) + 0.5 * _rank01(d))
    DH = DENS_HUMAN_HI if guard else DENS_HUMAN                          # 울타리: 사람 상위 10% 곡보다 진할 때만 덜어냄
    flat_k = 1.0
    if loud_range is not None and np.isfinite(loud_range) and not guard:
        # 곡 소리 폭에 맞는 사람 분포 + 소리 폭이 좁으면 줄보다 건반부터 덜어 '조금 쉬는 폭타'로
        DH = dens_curve(float(loud_range))
        flat_k = float(np.clip((float(loud_range) - 0.05) / 0.05, 0.25, 1.0))
        rep["loud_range"] = float(loud_range)
        rep["flat_k"] = flat_k
    tgt = Pref * np.interp(r, DENS_Q, DH) * (1.0 + tol / 2)      # 목표 (약간 여유)
    if dev_loud and not guard:
        # --dens-dev: 편차 기준 목표
        bl = np.array([np.mean([dev_loud[m] for m in (2 * k, 2 * k + 1) if m in dev_loud])
                       if any(m in dev_loud for m in (2 * k, 2 * k + 1)) else np.nan for k in ks])
        ok = np.isfinite(bl)
        if ok.sum() >= 8:
            ref = float(np.percentile(bl[ok], 90))
            dlt = np.where(ok, bl - ref, 0.0)
            t_dev = Pref * np.interp(dlt, DEV_X, DEV_MED) * (1.0 + tol / 2)
            t_new = np.where(dlt >= -0.015, np.maximum(t_dev, tgt), t_dev)
            rep["dev_lo"] = int(np.sum(ok & (t_new < tgt - 1e-9)))
            rep["dev_hi"] = int(np.sum(ok & (t_new > tgt + 1e-9)))
            rep["dev_q10"] = float(np.percentile(dlt[ok], 10))
            tgt = np.where(ok, t_new, tgt)
    trig = tgt * (1.0 + tol / 2)                                         # 이보다 진하면 줄임
    if cap > 0:
        tgt, trig = np.minimum(tgt, cap), np.minimum(trig, cap + 0.5)
    # 최고 근처가 너무 길게 이어지면 끊음
    hi = d >= 0.85 * P
    if max_run <= 0:
        max_run = peak_run_meas(bpm)
    rep["max_run"] = int(max_run)
    mb = max(1, int(max_run) // 2)
    kidx = set(int(x) for x in keep_idx)
    brk = set()        # 끊을 묶음 - protect 줄(폭타, 멜로디)도 brk_keys_ok면 줄은 그대로 두고 건반만 덜 수 있음
    bok = np.zeros(n, dtype=bool) if brk_keys_ok is None else np.asarray(brk_keys_ok, dtype=bool)
    run_ = []
    for bi in range(len(ks) + 1):
        if bi < len(ks) and hi[bi] and (not run_ or ks[bi] == ks[run_[-1]] + 1):
            run_.append(bi)
            continue
        if len(run_) > mb:
            nbr = int(np.ceil(len(run_) / mb)) - 1
            for j in range(1, nbr + 1):
                pos = int(round(j * len(run_) / (nbr + 1)))
                win = run_[max(0, pos - 1):pos + 2]
                pick = min(win, key=lambda x: loud[x])
                tgt[pick] = min(tgt[pick], 0.75 * Pref)
                brk.add(int(pick))
                trig[pick] = min(trig[pick], 0.8 * Pref)
                rep["breaks"] += 1
        run_ = [bi] if bi < len(ks) and hi[bi] else []
    pr = np.asarray(row_prob, dtype=np.float64)
    q16 = t / B * 4.0
    on16 = np.abs(q16 - np.round(q16)) < 1e-3
    qi = np.round(q16).astype(np.int64)
    beat_of = np.floor(t / B + 1e-6).astype(np.int64)
    meas_of = np.floor(t / bar + 1e-6).astype(np.int64)
    pulse = set(int(x) for x in pulse_meas) if pulse_meas else set()
    kset = set(int(i) for i in keep_idx) | set(int(i) for i in np.flatnonzero(prot))
    rep.update(P=P, Pref=Pref, q10_before=float(np.percentile(d / P, 10)))
    d_after = d.copy()
    for bi, k in enumerate(ks):
        if d[bi] <= trig[bi]:
            continue
        E = int(round((d[bi] - tgt[bi]) * dur * strength))
        if E <= 0:
            continue
        rep["blocks"] += 1
        R = [int(i) for i in rows_of[k]]
        share_rows = float(np.clip(1.0 - r[bi], 0.1, 0.8)) * flat_k
        e_rows = int(round(E * share_rows))
        removed = 0
        # 줄 빼기: 박 단위로 16분 뒷자리(32, 24분 포함) 줄 통째 -> 그다음 8분 뒷자리, 확률 낮은 박부터
        for kind in ("off16", "off8"):
            if removed >= e_rows:
                break
            groups = {}
            for i in R:
                if not keep[i] or i in kset:
                    continue
                is_off16 = (not on16[i]) or (qi[i] % 2 == 1)
                is_off8 = on16[i] and qi[i] % 4 == 2
                if kind == "off8" and is_off8 and int(meas_of[i]) in pulse:
                    rep["pulse_rows"] += 1                      # 8분 박 소리가 있는 마디는 8분 뒷자리 줄 지킴
                    continue
                if (kind == "off16" and is_off16) or (kind == "off8" and is_off8):
                    groups.setdefault(int(beat_of[i]), []).append(i)
            order = sorted(groups, key=lambda b_: (float(np.mean(pr[groups[b_]])), rng.random()))
            for b_ in order:
                if removed >= e_rows:
                    break
                for i in groups[b_]:
                    keep[i] = False
                    removed += int(c[i])
                    rep["rows"] += 1
        # 건반 빼기: 큰 동시치기부터 하나씩 (최소 1건반, 스크래치는 그대로)
        need = E - removed
        live = [i for i in R if keep[i] and (i not in kset or (bi in brk and bok[i] and i not in kidx))]
        while need > 0:
            cand = [i for i in live if len(_keys(masks[i])) >= 2]
            if not cand:
                break
            mx = max(len(_keys(masks[i])) for i in cand)
            top = [i for i in cand if len(_keys(masks[i])) == mx]
            i = int(rng.choice(top))
            kk = _keys(masks[i])
            pm = _prev8_mask(t, masks, i, B)
            rep_ = [x for x in kk if (pm >> x) & 1]
            inner = kk[1:-1] if len(kk) > 2 else kk
            adj = [x for x in inner if (x - 1 in kk) or (x + 1 in kk)]
            masks[i] &= ~(1 << int(rng.choice(rep_ or adj or kk)))
            need -= 1
            removed += 1
            rep["keys"] += 1
        d_after[bi] = (d[bi] * dur - removed) / dur
    rep["q10_after"] = float(np.percentile(d_after / max(np.percentile(d_after, 95), 1e-6), 10))
    rep["P_after"] = float(np.percentile(d_after, 95))
    return keep, masks, rep


LIFT_BPM = np.array([135.0, 168.0, 190.0, 220.0])          # BPM대(~150 / 150~180 / 180~200 / 200~)의 대략 가운데
LIFT_CURVE = np.array([
    [1.36, 1.81, 2.13, 2.42, 2.59, 2.73, 2.79, 2.83, 2.90, 2.97, 3.35],
    [1.12, 1.57, 2.02, 2.21, 2.35, 2.44, 2.56, 2.62, 2.69, 2.81, 3.19],
    [1.18, 1.67, 1.93, 2.10, 2.19, 2.25, 2.33, 2.43, 2.52, 2.66, 3.08],
    [1.06, 1.49, 1.73, 1.84, 1.88, 2.00, 2.06, 2.12, 2.25, 2.31, 2.56]])


def stream_lift(t_chart, masks, bpm: float, row_loud, strength: float = 1.0, scale: float = 1.0, seed: int = 0,
                max_chord: int = 6, jack_ms: float = 100.0, keep_idx=(), guard: bool = False):
    """--stream-lift: 얇은, 중간 폭타를 같은 BPM대 사람 두께 분포까지 올림 (줄은 그대로, 건반만 더함, 빼지는 않음)
    반환: (masks, 더한 건반 수, 보고 dict)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(masks)
    rep = {}
    if strength <= 0 or n < 64:
        return masks, 0, rep
    rng = np.random.default_rng(seed + 970)
    B = 60.0 / bpm
    bar = 4 * B
    meas = np.floor(t / bar + 1e-6).astype(np.int64)
    q16 = t / B * 4.0
    on16 = np.abs(q16 - np.round(q16)) < 1e-3
    qi = np.round(q16).astype(np.int64)
    slots = {}
    for i in range(n):
        if on16[i]:
            slots.setdefault(int(meas[i]), set()).add(int(qi[i]) % 16)
    S = sorted(m for m, s in slots.items() if len(s) >= 12)
    if len(S) < 6:
        return masks, 0, rep
    rows_of = {m: [int(i) for i in np.flatnonzero(meas == m)] for m in S}
    cnt = lambda i: bin(masks[i] & 0xFF).count("1")
    kpr = np.array([np.mean([cnt(i) for i in rows_of[m]]) for m in S])
    L = np.asarray(row_loud, dtype=np.float64)
    loud = np.array([float(np.mean(L[rows_of[m]])) for m in S])
    r = _rank01(0.5 * _rank01(loud) + 0.5 * _rank01(kpr))
    LC = LIFT_CURVE_LO if guard else LIFT_CURVE          # 울타리: 사람 하위 10% 곡보다 얇을 때만 올림
    curve = np.array([np.interp(bpm, LIFT_BPM, LC[:, j]) for j in range(LC.shape[1])])
    top = float(np.percentile(kpr, 90))
    nn = np.array([sum(cnt(i) for i in rows_of[m]) for m in S], dtype=np.float64)
    ncap = float(np.percentile(nn, 90))           # 마디 노트 수도 지금 상위 10%까지 (32분 줄이 많은 마디가 최고 밀도를 넘지 않게)
    tgt = np.minimum(np.interp(r, np.linspace(0, 1, len(curve)), curve) * scale, top)
    tgt = kpr + strength * np.maximum(tgt - kpr, 0.0)
    kset = set(int(i) for i in keep_idx)
    added = 0
    for mi, m in enumerate(S):
        R = [i for i in rows_of[m] if i not in kset]
        goal = min(int(np.floor(tgt[mi] * len(rows_of[m]) + 0.5)), max(int(nn[mi]), int(ncap)))
        need = goal - int(nn[mi])
        stuck = set()
        while need > 0:
            cand = [i for i in R if i not in stuck and len(_keys(masks[i])) < max_chord]
            if not cand:
                break
            mn = min(cnt(i) for i in cand)
            low = [i for i in cand if cnt(i) == mn]
            off = [i for i in low if not (on16[i] and qi[i] % 2 == 0)]
            i = int(rng.choice(off or low))
            ks = _keys(masks[i])
            near = masks[i] | _near_mask(t, masks, i, jack_ms + 5.0)
            pm = _prev8_mask(t, masks, i, B)
            free = [k for k in range(1, 8) if not ((near | pm) >> k) & 1] or \
                [k for k in range(1, 8) if not (near >> k) & 1]
            if not free:
                stuck.add(i)
                continue
            d = np.array([min(abs(k - x) for x in ks) for k in free], dtype=np.float64) if ks else np.ones(len(free))
            wk = np.where(d >= 2, 1.0, 0.25) + 0.01
            masks[i] |= 1 << int(rng.choice(free, p=wk / wk.sum()))
            added += 1
            need -= 1
    kpr2 = np.array([np.mean([cnt(i) for i in rows_of[m]]) for m in S])
    rep = {"meas": len(S), "before": tuple(np.percentile(kpr, [10, 50, 90])), "after": tuple(np.percentile(kpr2, [10, 50, 90])),
           "human": (float(np.interp(0.1, np.linspace(0, 1, len(curve)), curve)), float(curve[5]),
                     float(np.interp(0.9, np.linspace(0, 1, len(curve)), curve)))}
    return masks, added, rep


# ---------------- 울타리 모드 (--guard) ----------------
# 밀도, 두께 규칙이 사람 중앙값을 목표로 하지 않고,
# 사람 lv15~22 165곡 곡별 값의 10~90% 범위를 벗어난 만큼만 고침 (범위 안이면 그 곡 개성으로 둠).
ON8_LO = {1: 0.019, 2: 0.103, 3: 0.195, 4: 0.037, 5: 0.0}       # 폭타 속 8분 자리 줄 건반 수 비율, 곡별 하위 10%
ON8_HI = {1: 0.300, 2: 0.426, 3: 0.525, 4: 0.360, 5: 0.141}     # 상위 10%
OFF_LO = {1: 0.209, 2: 0.182, 3: 0.006, 4: 0.0}                  # 폭타 속 뒷자리
OFF_HI = {1: 0.771, 2: 0.600, 3: 0.266, 4: 0.047}
PAIR_LO, PAIR_HI = 3.49, 5.29                                    # 폭타 '8분 자리 + 바로 뒤 16분' 건반 합 곡 평균
LIFT_CURVE_LO = np.array([                                       # 폭타 줄당 노트 곡 안 분위 0~100%, BPM대별 하위 10% 곡
    [1.02, 1.38, 1.52, 1.63, 1.88, 2.02, 2.05, 2.15, 2.33, 2.48, 2.76],
    [1.00, 1.11, 1.30, 1.50, 1.63, 1.72, 1.83, 2.00, 2.13, 2.29, 2.77],
    [1.00, 1.05, 1.36, 1.50, 1.64, 1.91, 2.04, 2.09, 2.20, 2.32, 2.67],
    [1.00, 1.11, 1.19, 1.35, 1.44, 1.49, 1.50, 1.56, 1.79, 1.95, 2.01]])
DENS_HUMAN_HI = np.array([0.28, 0.38, 0.46, 0.53, 0.59, 0.65, 0.71, 0.76, 0.79, 0.82, 0.86,   # 2마디 묶음 밀도/최고, 상위 10% 곡
                          0.88, 0.91, 0.93, 0.94, 0.95, 0.96, 0.98, 0.99, 1.00, 1.19])


def _clip_dist(cur: dict, lo: dict, hi: dict) -> dict:
    """지금 비율을 [lo, hi]로 자른 뒤 합이 1이 되게 범위 안 칸들로 나눠 맞춤 (범위 안이면 지금 그대로)"""
    ks = sorted(cur)
    t = {k: min(max(float(cur[k]), lo[k]), hi[k]) for k in ks}
    for _ in range(30):
        s = sum(t.values())
        if abs(s - 1.0) < 1e-6:
            break
        free = [k for k in ks if (s > 1 and t[k] > lo[k] + 1e-9) or (s < 1 and t[k] < hi[k] - 1e-9)]
        if not free:
            break
        w = sum(t[k] for k in free)
        for k in free:
            share = (t[k] / w) if w > 0 else 1.0 / len(free)
            t[k] = min(max(t[k] - (s - 1.0) * share, lo[k]), hi[k])
    return t


def guard_on8_target(t_chart, masks, bpm: float) -> dict:
    """on8_match와 같은 줄(16분 연타 속 8분 자리)의 지금 건반 수 비율을 사람 10~90% 범위로 자른 목표"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    rows = [int(i) for i in np.flatnonzero(on8 & (row_kind(t, bpm) == 16))]
    if len(rows) < 8:
        return dict(ON8_HUMAN)
    c = np.array([min(len(_keys(masks[i])), 5) for i in rows])
    cur = {k: float(np.mean(c == k)) for k in (1, 2, 3, 4, 5)}
    return _clip_dist(cur, ON8_LO, ON8_HI)


def guard_off_target(t_chart, masks, bpm: float) -> dict:
    """off_match와 같은 줄(16분 연타 속 뒷자리)의 지금 건반 수 비율을 사람 10~90% 범위로 자른 목표"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    q16 = t / B * 4.0
    on8 = (np.abs(q16 - np.round(q16)) < 1e-3) & (np.round(q16).astype(np.int64) % 2 == 0)
    rows = [int(i) for i in np.flatnonzero(~on8 & (row_kind(t, bpm) == 16)) if masks[i] & 0xFE]
    if len(rows) < 8:
        return dict(OFF_HUMAN)
    c = np.array([min(len(_keys(masks[i])), 4) for i in rows])
    cur = {k: float(np.mean(c == k)) for k in (1, 2, 3, 4)}
    return _clip_dist(cur, OFF_LO, OFF_HI)


def pair_natural(t_chart, masks, bpm: float, min_fill: int = 8) -> float:
    """바쁜 마디(16분 칸 min_fill/16+)의 '8분 자리 + 바로 뒤 16분' 건반 합 평균 (지금 그대로)"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    sl = np.rint(t / B * 48).astype(np.int64)
    exact = np.abs(t / B * 48 - sl) < 0.3
    pos = {int(s): i for i, (s, e) in enumerate(zip(sl, exact)) if e}
    meas = {}
    for s in pos:
        if s % 12 == 0:
            meas.setdefault(s // 192, set()).add((s % 192) // 12)
    busy = {m for m, v in meas.items() if len(v) >= min_fill}
    ps = []
    for s, i in pos.items():
        if s % 24 == 0 and s // 192 in busy and (s + 12) in pos:
            ps.append(len(_keys(masks[i])) + len(_keys(masks[pos[s + 12]])))
    return float(np.mean(ps)) if ps else float("nan")


# ---------------- 소리 큰 곳 반쯤 빈 마디 채우기(--half-fill), 4+1 섞기(--four-one) ----------------
# --half-fill 근거 (생성 12곡): 곡 소리 상위 50%에서 반쯤 빈 16비트인 마디 116개, 69개는 앞뒤가 폭타.
# --four-one 근거: 4 단계 중 4+1 30 / 4+2 55 / 4+3+ 15%, 묶음 순위 중앙 4+1 0.77(3+2와 같음) / 4+2 0.82,
#   최고 구간 4+2 26 / 4+1 1% - 폭타 두께 사람 분포(--stream-lift)가 16분 뒷자리부터 건반을 더해서 4+1이 4+2가 됨.
FOUR_ONE_SHARE = 0.35          # 사람 4+1 / (4+1 + 4+2) = 737 / 2097


def half_fill_pick(t_chart, probs, bpm: float, pad: float, fps: int, mel, min_p: float = 0.30, rank_min: float = 0.5,
                   skip=()):
    """--half-fill: 곡 소리 상위 마디 중 앞이나 뒤가 폭타인 반쯤 빈 16비트 마디에서, 모델 확률이 min_p 이상인 빈 16분 뒷자리 칸을 고른다
    반환: (더할 칸 번호[박당 48칸 기준], [(마디, 전 16분 칸 수, 후 칸 수, 소리 순위)])"""
    t = np.asarray(t_chart, dtype=np.float64)
    if min_p <= 0 or len(t) < 32:
        return [], []
    B = 60.0 / bpm
    sl = np.rint(t / B * 48).astype(np.int64)
    occ = set(sl.tolist())
    ms = sorted(set((sl // 192).tolist()))
    loud = _meas_loud(mel, bpm, pad, range(ms[0], ms[-1] + 1)) if mel is not None else {}
    vs = sorted(loud.values())
    rk = {m: float(np.searchsorted(vs, v) / max(len(vs) - 1, 1)) for m, v in loud.items()}
    pr = np.asarray(probs, dtype=np.float64)
    skip = set(int(x) for x in skip)

    def cnt(m, js):
        return sum(1 for j in js if m * 192 + 12 * j in occ)
    f16 = {m: cnt(m, range(16)) for m in range(ms[0] - 1, ms[-1] + 2)}
    add, rep = [], []
    for m in ms:
        if m in skip or rk.get(m, 0.0) < rank_min:
            continue
        on8, off = cnt(m, range(0, 16, 2)), cnt(m, range(1, 16, 2))
        if f16[m] >= 12 or on8 < 6 or not 1 <= off <= 5:
            continue
        if f16.get(m - 1, 0) < 12 and f16.get(m + 1, 0) < 12:
            continue
        got = []
        for j in range(1, 16, 2):
            c = m * 192 + 12 * j
            if c in occ or any((c + d) in occ for d in range(-5, 6) if d):
                continue
            fr = int(round((c * B / 48.0 - pad) * fps))
            p = float(np.max(pr[np.clip(np.arange(fr - 3, fr + 4), 0, len(pr) - 1)]))
            if p >= min_p:
                got.append(c)
        if got:
            add += got
            rep.append((m, f16[m], f16[m] + len(got), rk.get(m, 0.0)))
    return add, rep


def four_one(t_chart, masks, bpm: float, row_loud=None, strength: float = 1.0, seed: int = 0, keep_idx=(), skip=(),
             min_fill: int = 12, targets=None, t_max: int = 5):
    """--four-one: 4 단계 폭타 중 4+1 비율을 사람(35% x strength)까지 올림. 4+2 마디 중 소리가 작은 것부터 16분 뒷자리 줄을 1개씩으로
    반환: (masks, 뺀 건반 수, 보고 dict)"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    if strength <= 0 or len(t) < 64:
        return masks, 0, {}
    B = 60.0 / bpm
    q = t / B * 4.0
    qi = np.rint(q).astype(np.int64)
    on16 = np.abs(q - qi) < 0.08
    kset = set(int(i) for i in keep_idx)
    skip = set(int(x) for x in skip)
    rows, fill = {}, {}
    for i in np.flatnonzero(on16):
        if not masks[i] & 0xFE:
            continue
        m = int(qi[i] // 16)
        rows.setdefault(m, []).append(int(i))
        fill.setdefault(m, set()).add(int(qi[i] % 16))
    cnt = lambda i: len(_keys(masks[i]))

    def ab(m):
        a = [cnt(i) for i in rows[m] if qi[i] % 2 == 0]
        b = [cnt(i) for i in rows[m] if qi[i] % 2 == 1]
        return (float(np.mean(a)) if a else 0.0), (float(np.mean(b)) if b else 0.0)

    def classes():
        c41, c42, c43 = [], [], []
        for m in rows:
            if len(fill[m]) < min_fill:
                continue
            a, b = ab(m)
            if a < 3.5:
                continue
            (c41 if b < 1.5 else (c42 if b < 2.5 else c43)).append(m)
        return c41, c42, c43
    c41, c42, c43 = classes()
    before = (len(c41), len(c42), len(c43))
    need = int(round(FOUR_ONE_SHARE * min(strength, 1.0) * (len(c41) + len(c42)))) - len(c41)
    if need <= 0:
        return masks, 0, {"before": before, "after": before, "meas": []}
    L = np.asarray(row_loud, dtype=np.float64) if row_loud is not None else None
    loud = {m: (float(np.mean(L[rows[m]])) if L is not None else 0.0) for m in c42}
    rng = np.random.default_rng(seed + 1070)
    removed, done = 0, []
    cand = [m for m in c42 if m not in skip]
    blocked = 0
    if targets:
        ok = [m for m in cand if m in targets and int(targets[m]) <= t_max]
        blocked = len(cand) - len(ok)
        cand = ok
    for m in sorted(cand, key=lambda x: loud[x])[:need]:
        for i in rows[m]:
            if qi[i] % 2 == 0 or i in kset:
                continue
            while cnt(i) > 1:
                ks = _keys(masks[i])
                pm = masks[i - 1] if i > 0 else 0
                nm = masks[i + 1] if i + 1 < len(masks) else 0
                rep_k = [k for k in ks if (pm >> k) & 1] or [k for k in ks if (nm >> k) & 1]
                masks[i] &= ~(1 << int(rng.choice(rep_k or ks)))
                removed += 1
        done.append(m)
    c41, c42, c43 = classes()
    return masks, removed, {"before": before, "after": (len(c41), len(c42), len(c43)), "meas": done, "blocked": blocked}


# ---------------------------------------------------------------------------
# --chord32: 32분 뒷자리 줄을 옆 16분 줄 동시치기로 (플레이어 평가 반영)
#   잘 안 들리는 소리에 32분 노트가 찍혀 엇박처럼 들리는 곳이 많음 -> 동시치기로 합침.
#   모델은 롤을 32분마다 봉우리로 실제로 들음 - 소리가 없어서가 아니라 취향 문제라 줄은 지우지 않고
#   건반을 옆 16분 줄에 합쳐 세기(완급)는 남김. 떨림(--trem) 줄은 그대로.
def _merge_mask(a: int, b: int, max_chord: int, avoid: int = 0):
    """a 줄에 b 줄 건반을 더함 (겹치면 비어 있는 가까운 건반, 앞뒤 줄 건반은 되도록 피함). 반환: (새 마스크, 상한 때문에 못 넣은 수)"""
    out = int(a) | (int(b) & 1)
    lost = 0
    for x in _keys(b):
        if len(_keys(out)) >= max_chord:
            lost += 1
            continue
        if not (out >> x) & 1:
            out |= 1 << x
            continue
        free = [y for y in range(1, 8) if not (out >> y) & 1 and not (avoid >> y) & 1] or \
               [y for y in range(1, 8) if not (out >> y) & 1]
        if free:
            out |= 1 << min(free, key=lambda y: (abs(y - x), y))
        else:
            lost += 1
    return out, lost


def chord32(t_chart, masks, bpm: float, max_chord: int = 6, protect=None, strength: float = 0.45, row_prob=None):
    """32분 뒷자리(박의 홀수/8) 줄 중 모델 확률이 strength 미만인 것(약한 소리에 찍힌 노트)을 바로 옆 16분 칸 줄에 합쳐 동시치기로
    반환: (남길 줄 bool, masks, 새 시각, 보고 dict)"""
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    n = len(t)
    keep = np.ones(n, dtype=bool)
    t_new = t.copy()
    rep = {"n32": 0, "merged": 0, "snapped": 0, "kept": 0, "lost": 0, "strong": 0, "meas": set()}
    if strength <= 0 or n == 0:
        return keep, masks, t_new, rep
    B = 60.0 / bpm
    b = t / B
    q8 = np.round(b * 8)
    is32 = (np.abs(b * 8 - q8) < 1e-3) & (q8.astype(np.int64) % 2 == 1)
    q4 = np.round(b * 4)
    on16 = np.abs(b * 4 - q4) < 1e-3
    prot = np.zeros(n, dtype=bool) if protect is None else np.asarray(protect, dtype=bool)
    rep["n32"] = int(is32.sum())
    rep["kept"] = int((is32 & prot).sum())
    weak = np.ones(n, dtype=bool)
    if row_prob is not None and strength < 1.0:
        weak = np.asarray(row_prob, dtype=np.float64) < float(strength)
    rep["strong"] = int((is32 & ~prot & ~weak).sum())
    cell = {int(q4[i]): i for i in range(n) if on16[i]}
    for k in np.flatnonzero(is32 & ~prot & weak):
        c_prev = int(np.floor(b[k] * 4 + 1e-6))
        c_next = c_prev + 1
        cand = [c for c in (c_prev, c_next) if c in cell and keep[cell[c]]]
        if not cand:
            t_new[k] = c_prev * B / 4.0
            cell[c_prev] = k
            rep["snapped"] += 1
            rep["meas"].add(int(c_prev // 16))
            continue
        tgt_c = min(cand, key=lambda c: (c % 2, 0 if c == c_prev else 1))     # 8분 자리 우선, 같으면 앞
        ti = cell[tgt_c]
        avoid = 0
        for c in (tgt_c - 1, tgt_c + 1):
            j = cell.get(c)
            if j is not None and keep[j] and j != k:
                avoid |= masks[j]
        masks[ti], lost = _merge_mask(masks[ti], masks[k], max_chord, avoid)
        keep[k] = False
        rep["merged"] += 1
        rep["lost"] += lost
        rep["meas"].add(int(tgt_c // 16))
    return keep, masks, t_new, rep
