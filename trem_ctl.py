"""
[규칙, STEP 3 ~ 6] --trem: 떨림(드르릉) 살리기

1) 떨림 박 찾기: 약한 봉우리가 촘촘하고 16분 칸 밖 봉우리가 2개 이상인 박이 2박 이상 이어질 때.
2) 그 박은 24분 또는 32분 격자를 다 채운다.
3) 떨림 박 줄은 단타로 두어 구간 밀도를 줄인다.
"""
from __future__ import annotations

import numpy as np

TREM_RATE = 16.0          # 초당 봉우리 수 기준 (strength 1).
TREM_LOW = 0.15           # 봉우리 바닥 (note_diag 후보와 같음)
RES = 96                  # 박당 칸 (24분 16칸마다, 32분 12칸마다)
MIN_STEP_MS = 40.0        # 떨림 격자 최소 간격: 이보다 촘촘한 32분(188BPM+)은 24분으로
# 사람 277개 중 24, 32분이 2박+ 이어진 박 3083개(124곡) 줄 두께 (단타 / 2개 / 3개+)
TREM_THICK = {"head": (0.34, 0.35, 0.31), "16": (0.60, 0.30, 0.10), "fine": (0.88, 0.11, 0.01)}


def beats_from_peaks(t_peaks, bpm: float, strength: float = 1.0, off_min: int = 2, run: int = 2,
                     min_step_ms: float = MIN_STEP_MS):
    """봉우리 시각(차트 시각, 초) -> {박 번호: 분할(6 또는 8)}, 보고"""
    B = 60.0 / bpm
    t = np.sort(np.asarray(t_peaks, dtype=np.float64))
    if strength <= 0 or len(t) < 8:
        return {}, {"cand": 0}
    pos = t / B
    b = np.floor(pos + 1.0 / 16).astype(np.int64)          # 박 끝 1/16 안 봉우리는 다음 박 머리
    frac = pos - b
    rate = TREM_RATE / strength
    cand = {}
    for bb in np.unique(b):
        fr = frac[b == bb]
        if len(fr) / B < rate:
            continue
        if int((np.abs(fr * 4 - np.round(fr * 4)) > 0.15).sum()) < off_min:
            continue
        e32 = float(np.mean(np.abs(fr * 8 - np.round(fr * 8))))       # 격자 칸 대비 거리 (0~0.5)
        e24 = float(np.mean(np.abs(fr * 6 - np.round(fr * 6))))
        if abs(e32 - e24) < 0.03:
            d = 8 if len(fr) >= 7 else 6
        else:
            d = 8 if e32 < e24 else 6
        if d == 8 and B / 8 * 1000.0 < min_step_ms:
            d = 6
        if B / d * 1000.0 < min_step_ms:
            continue
        cand[int(bb)] = d
    if run >= 2:
        beats = {k: v for k, v in cand.items() if (k - 1) in cand or (k + 1) in cand}
    else:
        beats = dict(cand)
    return beats, {"cand": len(cand)}


def trem_beats(probs, bpm: float, pad: float, shift_s: float = 0.0, fps: int = 100, strength: float = 1.0):
    from metrics import peak_pick
    pk = peak_pick(np.asarray(probs, dtype=np.float32), TREM_LOW, width=1)
    return beats_from_peaks(pk / float(fps) + shift_s + pad, bpm, strength)


def skip_rest(beats: dict, accent_t, bpm: float, rest_beats: float) -> dict:
    """치고 쉬기로 비운 박(큰 소리 뒤 rest_beats박)은 빼고 돌려줌"""
    if not accent_t or rest_beats <= 0:
        return beats
    B = 60.0 / bpm
    out = {}
    for bb, d in beats.items():
        s, e = bb * B, (bb + 1) * B
        if any(s < x + rest_beats * B - 1e-6 and e > x + 1e-6 for x in accent_t):
            continue
        out[bb] = d
    return out


def trem_fill(t_chart, frames, bpm: float, pad: float, beats: dict, fps: int = 100, cand_frames=None):
    """떨림 박을 그 격자로 다 채우고 격자 밖 줄은 뺌.
    반환: (t_chart, frames, 남길 원래 줄 bool, 더한 줄 수, 뺀 줄 수, 떨림 칸 집합(박당 96칸 번호))"""
    B = 60.0 / bpm
    t = np.asarray(t_chart, dtype=np.float64)
    f = np.asarray(frames, dtype=np.int64)
    q = np.rint(t / B * RES).astype(np.int64)
    keep = np.ones(len(t), dtype=bool)
    slots, add = set(), []
    occ = set(q.tolist())
    for bb in sorted(beats):
        step = RES // int(beats[bb])
        grid = [bb * RES + k * step for k in range(int(beats[bb]))]
        slots.update(grid)
        inb = (q >= bb * RES) & (q < (bb + 1) * RES)
        keep &= ~(inb & ~np.isin(q, grid))
        add += [g for g in grid if g not in occ]
    t_new = t[keep]
    f_new = f[keep]
    if add:
        t_add = np.array(add, dtype=np.float64) * B / RES
        f_add = np.maximum(np.rint((t_add - pad) * fps).astype(np.int64), 0)
        if cand_frames is not None and len(cand_frames):        # 가까운(20ms 안) 봉우리가 있으면 그 프레임
            cf = np.sort(np.asarray(cand_frames, dtype=np.int64))
            for a in range(len(f_add)):
                j = int(np.searchsorted(cf, f_add[a]))
                near = [int(cf[k]) for k in (j - 1, j) if 0 <= k < len(cf) and abs(int(cf[k]) - f_add[a]) <= 2]
                if near:
                    f_add[a] = min(near, key=lambda x: abs(x - f_add[a]))
        t_new = np.concatenate([t_new, t_add])
        f_new = np.concatenate([f_new, f_add])
        o = np.argsort(t_new, kind="stable")
        t_new, f_new = t_new[o], f_new[o]
    return t_new, f_new, keep, len(add), int((~keep).sum()), slots


def trem_rows(t_chart, bpm: float, slots) -> np.ndarray:
    B = 60.0 / bpm
    q = np.rint(np.asarray(t_chart, dtype=np.float64) / B * RES).astype(np.int64)
    return np.isin(q, np.fromiter(slots, dtype=np.int64)) if slots else np.zeros(len(q), dtype=bool)


def trem_single(t_chart, masks, bpm: float, slots, jack_ms: float = 100.0, head_max: int = 2, seed: int = 0,
                human_thick: bool = True, max_chord: int = 6):
    """떨림 박 줄을 단타로 (박 머리 줄만 최대 head_max개, 원래 동시치기였을 때)
    반환: (masks, 보고 dict)"""
    from thick_ctl import _fine_seq, fine_weights, _near_mask
    B = 60.0 / bpm
    t = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    is_t = trem_rows(t, bpm, slots)
    q = np.rint(t / B * RES).astype(np.int64)
    rng = np.random.default_rng(seed + 990)
    rep = {"rows": int(is_t.sum()), "keys_before": sum(bin(masks[i] & 0xFF).count("1") for i in np.flatnonzero(is_t)),
           "keys_after": 0, "beats": 0, "shaped": 0, "greedy": 0, "jack": 0, "kinds": {}}
    orig = list(masks)
    beat_rows = {}
    for i in np.flatnonzero(is_t):
        beat_rows.setdefault(int(q[i] // RES), []).append(int(i))
    prev_k = None
    for bb in sorted(beat_rows):
        R = beat_rows[bb]
        rep["beats"] += 1
        gap = 32 if len(R) >= 8 else 24
        wt = fine_weights(gap)
        names = [k for k in wt if k != "none"]
        p = np.array([wt[k] for k in names], dtype=np.float64)
        p /= p.sum()
        for x in R:                                       # 일단 단타 자리를 비워 둠 (스크래치는 머리만)
            masks[x] = orig[x] & 1 if q[x] % RES == 0 else 0
        ok = False
        for _ in range(12):
            kind = names[int(rng.choice(len(names), p=p))]
            seq, sub = _fine_seq(kind, len(R), rng, gap)
            if len(seq) != len(R):
                continue
            if prev_k is not None and seq[0] == prev_k:
                continue
            for x, k in zip(R, seq):
                masks[x] = (masks[x] & 1) | (1 << int(k))
            if all(not (masks[x] & _near_mask(t, masks, x, jack_ms + 5.0)) for x in R):
                ok = True
                rep["kinds"][sub] = rep["kinds"].get(sub, 0) + 1
                break
            for x in R:
                masks[x] = masks[x] & 1
        if ok:
            rep["shaped"] += 1
        else:                                             # 줄마다 비는 건반 중 앞 단타에서 1~3칸 옆
            rep["greedy"] += 1
            pk = prev_k
            for x in R:
                near = _near_mask(t, masks, x, jack_ms + 5.0)
                free = [k for k in range(1, 8) if not (near >> k) & 1] or list(range(1, 8))
                k = min(free, key=lambda k: (abs(abs(k - pk) - 2) if pk is not None else abs(k - 4)) + 0.01 * rng.random())
                masks[x] = (masks[x] & 1) | (1 << k)
                pk = k
        h = R[0]
        if not human_thick and q[h] % RES == 0 and head_max >= 2 and len([k for k in range(1, 8) if (orig[h] >> k) & 1]) >= 2:
            near = _near_mask(t, masks, h, jack_ms + 5.0) | masks[h]
            add = [k for k in range(1, 8) if (orig[h] >> k) & 1 and not (near >> k) & 1]
            if add:                                       # 박 머리 강세: 원래 동시치기의 건반 하나 더
                masks[h] |= 1 << add[0]
        last = [k for k in range(1, 8) if (masks[R[-1]] >> k) & 1]
        prev_k = last[0] if last else None
    rep["keys_single"] = sum(bin(masks[i] & 0xFF).count("1") for i in np.flatnonzero(is_t))
    if human_thick:
        rth = np.random.default_rng(seed + 991)
        step16 = RES // 4
        for i in np.flatnonzero(is_t):
            pos = "head" if q[i] % RES == 0 else ("16" if q[i] % step16 == 0 else "fine")
            p_ = TREM_THICK[pos]
            want = 1 + int(rth.choice(3, p=p_))
            want = min(want, max_chord)
            have = _keys_of(masks[i])
            if len(have) >= want:
                continue
            near = _near_mask(t, masks, int(i), jack_ms + 5.0)
            for _ in range(want - len(have)):
                ks = _keys_of(masks[i])
                free = [k for k in range(1, 8) if not ((masks[i] | near) >> k) & 1]
                if not free:
                    break
                far = [k for k in free if all(abs(k - x) >= 2 for x in ks)]
                pool = far or free
                masks[i] |= 1 << int(pool[int(rth.integers(len(pool)))])
                rep["thick_add"] = rep.get("thick_add", 0) + 1
    for i in np.flatnonzero(is_t):
        rep["keys_after"] += bin(masks[i] & 0xFF).count("1")
        if masks[i] & _near_mask(t, masks, i, jack_ms) & 0xFE:
            rep["jack"] += 1
    return masks, rep


def _keys_of(m: int) -> list[int]:
    return [k for k in range(1, 8) if (int(m) >> k) & 1]
