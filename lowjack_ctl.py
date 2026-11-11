"""
[규칙, STEP 6] --jack-allow: 낮은 BPM 16분 2연타 허용

BPM 150 미만에서 폭타 마디 중 건반 수 상위 25% 마디의 이웃 16분 줄 쌍에만 적용한다.
목표 비율은 BPM과 두께가 비슷한 사람 곡에서 구하고, 소리가 비슷한 쌍부터 2연타로 만든다 (건반 수는 그대로).
"""
from __future__ import annotations

from math import comb

import numpy as np

# (앞 줄 건반 수, 뒷 줄 건반 수): (쌍 수, 같은 건반 1개+, 2개+, 3개+) - 사람 277개(256곡), 쌍 20개+ 칸만
HUMAN = (
    # BPM 135 아래 (17곡, 중앙값 118)
    {(1, 3): (28, 0.071, 0.0, 0.0), (1, 4): (40, 0.2, 0.0, 0.0), (1, 5): (37, 0.324, 0.0, 0.0), (1, 6): (21, 0.476, 0.0, 0.0),
     (2, 2): (79, 0.177, 0.0, 0.0), (2, 3): (293, 0.174, 0.017, 0.0), (2, 4): (147, 0.279, 0.034, 0.0),
     (2, 5): (123, 0.512, 0.098, 0.0), (2, 6): (24, 1.0, 0.125, 0.0), (3, 1): (26, 0.0, 0.0, 0.0),
     (3, 2): (275, 0.218, 0.007, 0.0), (3, 3): (440, 0.443, 0.098, 0.005), (3, 4): (225, 0.644, 0.147, 0.009),
     (3, 5): (125, 1.0, 0.44, 0.04), (3, 6): (36, 1.0, 1.0, 0.278), (4, 1): (42, 0.071, 0.0, 0.0),
     (4, 2): (145, 0.372, 0.034, 0.0), (4, 3): (239, 0.762, 0.205, 0.013), (4, 4): (163, 1.0, 0.374, 0.098),
     (4, 5): (56, 1.0, 1.0, 0.357), (4, 6): (39, 1.0, 1.0, 1.0), (5, 1): (41, 0.317, 0.0, 0.0),
     (5, 2): (139, 0.676, 0.065, 0.0), (5, 3): (94, 1.0, 0.351, 0.053), (5, 4): (67, 1.0, 1.0, 0.239),
     (5, 5): (72, 1.0, 1.0, 1.0), (6, 2): (30, 1.0, 0.367, 0.0), (6, 3): (32, 1.0, 1.0, 0.094), (6, 4): (42, 1.0, 1.0, 1.0),
     (6, 6): (30, 1.0, 1.0, 1.0)},
    # BPM 135~150 (22곡, 중앙값 142)
    {(1, 1): (25, 0.28, 0.0, 0.0), (1, 2): (65, 0.015, 0.0, 0.0), (1, 3): (208, 0.038, 0.0, 0.0), (1, 4): (236, 0.03, 0.0, 0.0),
     (1, 5): (93, 0.054, 0.0, 0.0), (1, 6): (27, 0.037, 0.0, 0.0), (2, 1): (41, 0.195, 0.0, 0.0), (2, 2): (306, 0.049, 0.0, 0.0),
     (2, 3): (595, 0.04, 0.003, 0.0), (2, 4): (467, 0.054, 0.0, 0.0), (2, 5): (140, 0.107, 0.007, 0.0),
     (3, 1): (246, 0.033, 0.0, 0.0), (3, 2): (495, 0.073, 0.01, 0.0), (3, 3): (764, 0.096, 0.02, 0.001),
     (3, 4): (441, 0.168, 0.029, 0.005), (3, 5): (27, 1.0, 0.148, 0.0), (4, 1): (246, 0.089, 0.0, 0.0),
     (4, 2): (531, 0.16, 0.03, 0.0), (4, 3): (401, 0.209, 0.045, 0.012), (4, 4): (58, 1.0, 0.483, 0.103),
     (5, 1): (61, 0.115, 0.0, 0.0), (5, 2): (143, 0.175, 0.035, 0.0), (5, 3): (35, 1.0, 0.486, 0.086)},
    # BPM 150+ (217곡) - 150에 가까울수록 이쪽으로 줄여 감
    {(1, 1): (1319, 0.002, 0.0, 0.0), (1, 2): (1485, 0.003, 0.0, 0.0), (1, 3): (2691, 0.005, 0.0, 0.0),
     (1, 4): (2505, 0.006, 0.0, 0.0), (1, 5): (811, 0.006, 0.0, 0.0), (1, 6): (97, 0.01, 0.0, 0.0),
     (2, 1): (1380, 0.007, 0.0, 0.0), (2, 2): (6397, 0.015, 0.001, 0.0), (2, 3): (7326, 0.015, 0.002, 0.0),
     (2, 4): (3529, 0.017, 0.001, 0.0), (2, 5): (479, 0.042, 0.006, 0.0), (3, 1): (2546, 0.008, 0.0, 0.0),
     (3, 2): (7325, 0.022, 0.003, 0.0), (3, 3): (5727, 0.039, 0.014, 0.006), (3, 4): (1203, 0.076, 0.017, 0.0),
     (4, 1): (2574, 0.003, 0.0, 0.0), (4, 2): (3534, 0.019, 0.005, 0.0), (4, 3): (1185, 0.082, 0.026, 0.008),
     (4, 4): (71, 1.0, 0.31, 0.127), (5, 1): (763, 0.008, 0.0, 0.0), (5, 2): (550, 0.049, 0.011, 0.0),
     (5, 3): (21, 1.0, 0.19, 0.048), (6, 1): (91, 0.0, 0.0, 0.0)},
)
# 사람 곡 (BPM, 빽빽한 마디 줄당 건반, 무작위 대비 겹침 비율) - 이웃 16분 쌍 60개+, BPM 175 아래 (277개 중)
SONGS = (
    (50, 2.63, 0), (69, 3.1, 1.047), (90, 2.81, 0.008), (96, 4.75, 0.958), (107, 3.99, 0.976), (110, 3.79, 0.815),
    (115, 2.18, 0.406), (115, 4.91, 0.936), (118, 3.53, 0.566), (120, 3.14, 0.485), (120, 3.62, 0.764),
    (120, 3.65, 0.784), (120, 3.68, 0.741), (122, 3.27, 0.829), (126, 2.9, 0.769), (128, 2.97, 0.313),
    (134, 2.89, 0.295), (135, 1.58, 0.702), (135, 1.92, 0.173), (135, 2.53, 0.039), (135, 2.82, 0.229),
    (135, 3.31, 0.521), (136, 3, 0.074), (139, 2.97, 0.613), (140, 3.07, 0.159), (140, 3.11, 0.009),
    (142, 2.31, 0.007), (142, 2.33, 0.528), (142, 3.2, 0.101), (143, 3.13, 0.065), (145, 2.62, 0), (145, 2.87, 0.326),
    (145, 2.91, 0.217), (145, 2.94, 0.11), (146, 2.61, 0.141), (146, 2.95, 0), (146, 2.97, 0.229), (149, 3.02, 0.491),
    (149, 3.17, 0.005), (150, 2, 0.074), (150, 2.41, 0.327), (150, 2.56, 0.178), (150, 2.58, 0), (150, 2.65, 0.005),
    (150, 2.77, 0.373), (150, 3.17, 0.093), (152, 2.67, 0.224), (153, 2.96, 0), (154, 2.37, 0.014), (154, 2.95, 0.008),
    (154, 3.14, 0.048), (155, 1.85, 0.184), (155, 2.42, 0.206), (155, 2.58, 0), (155, 2.62, 0.074), (155, 2.82, 0.011),
    (155, 2.93, 0.168), (155, 3.19, 0.01), (157, 3.13, 0), (158, 2.86, 0.023), (158, 2.99, 0), (160, 2.1, 0.274),
    (160, 2.34, 0.046), (160, 2.37, 0), (160, 2.74, 0.111), (160, 2.81, 0.074), (160, 3, 0), (161, 2.61, 0.149),
    (161, 2.88, 0.126), (162, 2.1, 0.178), (162, 2.55, 0.053), (162, 2.6, 0.027), (162, 2.68, 0.056), (162, 3.09, 0),
    (163, 2.83, 0.182), (165, 2.64, 0.011), (165, 2.73, 0.079), (165, 2.9, 0), (165, 2.99, 0.096), (165, 3.04, 0.012),
    (165, 3.13, 0), (166, 3, 0), (167, 2.45, 0), (167, 2.65, 0.113), (167, 2.82, 0), (168, 1.78, 0),
    (168, 2.55, 0.044), (170, 2, 0.395), (170, 2.28, 0), (170, 2.46, 0), (170, 2.74, 0), (170, 2.92, 0.011),
    (170, 3.04, 0.026), (171, 2.79, 0.249), (172, 2.37, 0), (172, 2.58, 0.087), (172, 2.74, 0.026), (172, 2.76, 0),
    (172, 2.78, 0.04), (172, 2.78, 0), (172, 2.82, 0.053), (172, 2.85, 0.232), (173, 2.79, 0), (174, 2.44, 0.026),
    (174, 2.89, 0.014),
)

ANCHOR = (118.0, 142.0, 150.0)
RUN3 = (0.20, 0.072, 0.07)        # 같은 건반 16분 연속 중 3번 연속 비율 (2번+ 연속 중, 빽빽한 폭타 마디)
BPM_MAX = 150.0
REF_WIN = ((15.0, 0.4), (20.0, 0.6), (30.0, 0.8))   # 기준 곡 고르기 (BPM ±, 줄당 건반 ±)
REF_MIN = 5


def _keys(m: int) -> list[int]:
    return [k for k in range(1, 8) if (int(m) >> k) & 1]


def _shape(m: int):
    ks = _keys(m)
    return tuple(k - ks[0] for k in ks) if ks else ()


def chance(k1: int, k2: int) -> float:
    """건반 k1개 줄 뒤에 k2개 줄을 무작위로 놓을 때 같은 건반이 하나라도 겹칠 확률"""
    return 1.0 - comb(7 - k1, k2) / comb(7, k2) if 7 - k1 >= k2 else 1.0


def _mix(bpm: float):
    """BPM -> [(표 번호, 무게)]"""
    a0, a1, a2 = ANCHOR
    if bpm <= a0:
        return [(0, 1.0)]
    if bpm <= a1:
        w = (bpm - a0) / (a1 - a0)
        return [(0, 1.0 - w), (1, w)]
    if bpm < a2:
        w = (bpm - a1) / (a2 - a1)
        return [(1, 1.0 - w), (2, w)]
    return [(2, 1.0)]


def human_ratio(bpm: float, kpr: float):
    """BPM, 두께 비슷한 사람 곡의 '무작위 대비 겹침 비율' 중앙값. 반환: (r, 곡 수, BPM 폭, 두께 폭)"""
    for db, dk in REF_WIN:
        sel = [r for b, k, r in SONGS if abs(b - bpm) <= db and abs(k - kpr) <= dk]
        if len(sel) >= REF_MIN:
            return float(np.median(sel)), len(sel), db, dk
    near = sorted(SONGS, key=lambda x: ((x[0] - bpm) / 15.0) ** 2 + ((x[1] - kpr) / 0.4) ** 2)[:REF_MIN]
    return float(np.median([x[2] for x in near])), REF_MIN, 0.0, 0.0


def _cond(bpm: float, k1: int, k2: int):
    """그 칸 사람(모은 표)의 (2개+ / 1개+, 3개+ / 1개+). 칸이 비면 (0, 0)"""
    c2 = c3 = wsum = 0.0
    for b, w in _mix(bpm):
        v = HUMAN[b].get((k1, k2))
        if v is None or v[1] <= 0 or w <= 0:
            continue
        c2 += w * v[2] / v[1]
        c3 += w * v[3] / v[1]
        wsum += w
    return (c2 / wsum, c3 / wsum) if wsum > 0 else (0.0, 0.0)


def human_target(bpm: float, k1: int, k2: int, r: float):
    """(같은 건반 1개+, 2개+, 3개+) 목표 비율"""
    g1 = min(1.0, r * chance(k1, k2))
    c2, c3 = _cond(bpm, k1, k2)
    return np.array([g1, g1 * c2, g1 * c3], dtype=np.float64)


def run3_target(bpm: float) -> float:
    return float(sum(w * RUN3[b] for b, w in _mix(bpm)))


def _dense_pairs(t, masks, bpm: float, min_fill: int = 12, top: float = 0.25, min_meas: int = 4):
    """(16분 번호 -> 줄 번호, 빽빽한 마디 집합, 쌍 목록[(앞 줄, 뒷 줄, 다음 줄 또는 -1)], 빽빽한 마디 줄당 건반)"""
    B = 60.0 / bpm
    q = np.asarray(t, dtype=np.float64) / B * 4.0
    on = np.abs(q - np.round(q)) < 1e-3
    qi = np.round(q).astype(np.int64)
    pos = {}
    for i in np.flatnonzero(on):
        if int(masks[i]) & 0xFE:
            pos[int(qi[i])] = int(i)
    cnt, mk = {}, {}
    for x, i in pos.items():
        cnt[x // 16] = cnt.get(x // 16, 0) + 1
        mk[x // 16] = mk.get(x // 16, 0) + bin(int(masks[i]) & 0xFE).count("1")
    S = [M for M, c in cnt.items() if c >= min_fill]
    if len(S) < min_meas:
        return pos, set(), [], 0.0
    thr = float(np.percentile([mk[M] for M in S], 100.0 * (1.0 - top)))
    D = {M for M in S if mk[M] >= thr}
    pairs = [(pos[x - 1], pos[x], pos.get(x + 1, -1)) for x in sorted(pos) if x // 16 in D and (x - 1) in pos]
    kpr = sum(mk[M] for M in D) / max(sum(cnt[M] for M in D), 1)
    return pos, D, pairs, kpr


def _runs(pos, masks, D):
    """빽빽한 마디 안 같은 건반 16분 연속 (2번, 3번, 4번+) 개수"""
    r = [0, 0, 0]
    xs = sorted(x for x in pos if x // 16 in D)
    for k in range(1, 8):
        run, last = 0, None
        for x in xs + [None]:
            if x is not None and (int(masks[pos[x]]) >> k) & 1:
                if run > 0 and last == x - 1:
                    run += 1
                else:
                    if run >= 2:
                        r[min(run, 4) - 2] += 1
                    run = 1
                last = x
                continue
            if run >= 2:
                r[min(run, 4) - 2] += 1
            run, last = 0, x
    return r


def _run_len(pos, masks, x: int, k: int) -> int:
    """16분 번호 x 줄에 건반 k가 있다고 칠 때 그 건반 16분 연속 길이"""
    n = 1
    y = x - 1
    while y in pos and (int(masks[pos[y]]) >> k) & 1:
        n += 1
        y -= 1
    y = x + 1
    while y in pos and (int(masks[pos[y]]) >> k) & 1:
        n += 1
        y += 1
    return n


def jack_stats(t, masks, bpm: float):
    """빽빽한 폭타 마디 이웃 16분 쌍: (쌍 수, 같은 건반 있음 비율, 사람 기준 비율(같은 두께 구성), 연속 (2,3,4+), 기준 (r, 곡 수), 줄당 건반)"""
    pos, D, pairs, kpr = _dense_pairs(t, masks, bpm)
    if not pairs:
        return 0, 0.0, 0.0, (0, 0, 0), (0.0, 0), kpr
    r, nref, _, _ = human_ratio(bpm, kpr)
    ov = hu = 0.0
    for a, b, _ in pairs:
        k1, k2 = bin(masks[a] & 0xFE).count("1"), bin(masks[b] & 0xFE).count("1")
        ov += bool(masks[a] & masks[b] & 0xFE)
        hu += min(1.0, r * chance(k1, k2))
    return len(pairs), ov / len(pairs), hu / len(pairs), tuple(_runs(pos, masks, D)), (r, nref), kpr


def jack_allow(t_chart, masks, bpm: float, strength: float = 0.0, seed: int = 0, jack_ms: float = 100.0, protect=None,
               prof=None, sound_pct: float = 50.0):
    """prof: 줄마다 '새로 들어온 소리' 모양 (N, n_mels), 길이 1 (jack_ctl.onset_profiles). None이면 소리 안 봄.
    반환: (masks, 보고 dict)"""
    masks = [int(m) for m in masks]
    t = np.asarray(t_chart, dtype=np.float64)
    rep = {"skip": "", "edits": [0, 0, 0], "meas": 0, "sound": None}
    step_ms = 60000.0 / bpm / 4.0
    if strength <= 0:
        rep["skip"] = "off"
        return masks, rep
    if bpm >= BPM_MAX:
        rep["skip"] = f"BPM {bpm:g} >= {BPM_MAX:g}"
        return masks, rep
    if step_ms <= jack_ms + 1.0:
        rep["skip"] = f"16분 {step_ms:.0f}ms <= --no-jack-ms {jack_ms:g}"
        return masks, rep
    from thick_ctl import _near_mask
    s = min(float(strength), 1.0)
    rng = np.random.default_rng(seed + 1030)
    prot = np.zeros(len(masks), dtype=bool) if protect is None else np.asarray(protect, dtype=bool)
    rep["before"] = jack_stats(t, masks, bpm)
    pos, D, pairs, kpr = _dense_pairs(t, masks, bpm)
    rep["meas"] = len(D)
    if not pairs:
        rep["after"] = rep["before"]
        return masks, rep
    r_h, nref, db, dk = human_ratio(bpm, kpr)
    rep["ref"] = (r_h, nref, db, dk, kpr)
    q_of = {i: x for x, i in pos.items()}
    r3_t = run3_target(bpm)
    runs = _runs(pos, masks, D)
    sim = {}
    if prof is not None and len(prof) == len(masks):
        P = np.asarray(prof, dtype=np.float32)
        for a, b, _ in pairs:
            sim[b] = float(P[a] @ P[b])
        thr = float(np.percentile(list(sim.values()), sound_pct))
        rep["sound"] = (thr, sum(v >= thr for v in sim.values()), len(sim))

    for L in (1, 2, 3):
        # 두께 쌍마다 지금 / 목표
        grp = {}
        for a, b, c in pairs:
            k1, k2 = bin(masks[a] & 0xFE).count("1"), bin(masks[b] & 0xFE).count("1")
            if k1 == 1 and k2 == 1:
                continue
            o = bin(masks[a] & masks[b] & 0xFE).count("1")
            g = grp.setdefault((k1, k2), [0, 0, []])
            g[0] += 1
            g[1] += o >= L
            if o == L - 1 and not prot[b] and (not sim or sim[b] >= rep["sound"][0]):
                g[2].append((a, b, c))
        need = {}
        for (k1, k2), (n, cur, cand) in grp.items():
            h = human_target(bpm, k1, k2, r_h)[L - 1]
            if h <= cur / n:
                continue
            want = int(np.floor(n * (cur / n + s * (h - cur / n)) + 0.5))
            if want > cur and cand:
                need[(k1, k2)] = want - cur
        if not need:
            continue
        todo = [(g, p) for g in need for p in grp[g][2]]
        if sim:                                           # 같은 소리일수록 먼저
            order = sorted(range(len(todo)), key=lambda j: -(sim[todo[j][1][1]] + 1e-4 * rng.random()))
        else:
            order = rng.permutation(len(todo))
        for oi in order:
            g, (a, b, c) = todo[int(oi)]
            if need[g] <= 0:
                continue
            ma, mb = masks[a], masks[b]
            if bin(ma & mb & 0xFE).count("1") != L - 1:          # 앞에서 바뀐 쌍
                continue
            mc = masks[c] if c >= 0 else 0
            xb = q_of[b]
            near = _near_mask(t, masks, b, jack_ms + 1.0)
            best, best_s = None, None
            for x in _keys(ma & ~mb):
                if (near >> x) & 1:
                    continue
                rl = _run_len(pos, masks, xb, x)
                if rl >= 4:
                    continue
                if rl == 3 and runs[1] / max(runs[0] + runs[1], 1) >= r3_t:
                    continue
                for y in _keys(mb & ~ma):
                    nb = (mb & ~(1 << y)) | (1 << x)
                    if (g[0] >= 2 and g[1] >= 2
                            and ((nb & 0xFE) & ~ma & 0xFE == 0 or (ma & 0xFE) & ~nb == 0)):   # 앞 줄을 품거나 앞 줄에 들어감 (단타 낀 쌍은 빼고)
                        continue
                    if bin(nb & 0xFE).count("1") >= 2 and (_shape(nb) == _shape(ma) or (mc & 0xFE and _shape(nb) == _shape(mc))):
                        continue
                    sc = abs(x - y) + 0.6 * ((mc >> y) & 1) + 0.3 * ((mc >> x) & 1) + 0.1 * rng.random()
                    if best_s is None or sc < best_s:
                        best, best_s = nb, sc
            if best is None:
                continue
            masks[b] = best
            need[g] -= 1
            rep["edits"][L - 1] += 1
            runs = _runs(pos, masks, D)
    rep["after"] = jack_stats(t, masks, bpm)
    return masks, rep
