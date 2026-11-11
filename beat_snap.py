"""
[규칙, STEP 3] 박자 단위 격자 맞춤 (--snap beat)
"""
from __future__ import annotations

# 기본값은 실제 패턴 22개(BPM 120~330)에 모델 오차(표준편차 6ms), 놓침 4%, 중복 3%, 잡음 5%를 넣어
# 맞춘 값: 정답 칸에 놓인 노트 비율 16분 고정 90.4% -> 박 단위 97.0%

import numpy as np

# 박당 분할 수: (칸 수, 계열, 벌점)  계열 0 = 2진(16분/32분...), 1 = 3진(셋잇단)
DEFAULT_PATTERNS = (
    (4, 0, 0.0),     # 16분
    (3, 1, 2.0),     # 셋잇단 8분
    (6, 1, 4.0),     # 셋잇단 16분
    (8, 0, 7.0),     # 32분
    (12, 1, 14.0),   # 셋잇단 32분 (느린 곡에서만 후보)
    (16, 0, 14.0),   # 64분 (느린 곡에서만 후보)
)


def beat_phase(t: np.ndarray, bpm: float, sixteenth_phase: float) -> float:
    """16분 격자 위상을 알 때, 4개 후보 중 '박(4분음표) 시작'이 어디인지 고른다"""
    g = 60.0 / (bpm * 4)
    best, best_score = sixteenth_phase, -1
    for k in range(4):
        ph = sixteenth_phase + k * g
        d = ((t - ph + 2 * g) % (4 * g)) - 2 * g
        score = int((np.abs(d) < 0.25 * g).sum())
        if score > best_score:
            best, best_score = ph, score
    return best % (4 * g)


def _beat_cost(x: np.ndarray, B: float, d: int, sigma: float, cap: float, merge_pen: float,
               fine_pen: float = 0.0):
    """한 박 안의 상대 위치 x(초)를 d분할에 맞출 때의 비용과 칸 번호"""
    step = B / d
    j = np.rint(x / step).astype(int)               # 0..d (d면 다음 박의 첫 칸)
    e = (x - j * step) / sigma
    c = np.minimum(e * e, cap)
    if d in (6, 8, 12, 16) and fine_pen > 0:
        jc = 2 * np.rint(x / (2 * step)).astype(int)          # 가장 가까운 짝수 칸
        ec = (x - jc * step) / sigma
        cc = np.minimum(ec * ec, cap)
        cf = c + fine_pen * (j % 2 == 1)
        use_c = cc <= cf
        j = np.where(use_c, jc, j)
        c = np.where(use_c, cc, cf)
    n_merge = len(j) - len(np.unique(j))
    return float(c.sum()) + merge_pen * n_merge, j


def snap_beats(t_chart: np.ndarray, bpm: float, min_step_ms: float = 25.0,
               sigma_ms: float = 6.0, cap: float = 9.0, merge_pen: float = 0.5,
               switch_pen: float = 3.0, fine_pen: float = 2.0, patterns=DEFAULT_PATTERNS, beat_allow=None):
    """박 단위 격자 맞춤
    반환: (고유 시각[k], 합쳐진 원래 노트 번호들[k], 정보 dict)"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    pats = [p for p in patterns if B / p[0] * 1000.0 >= min_step_ms] or [patterns[0]]
    sigma = sigma_ms / 1000.0

    beat = np.floor(t / B + 1e-9).astype(int)
    beats = np.unique(beat)
    idx_of = {b: np.flatnonzero(beat == b) for b in beats}

    # 박마다 후보별 비용
    P = len(pats)
    cost = np.zeros((len(beats), P))
    choice = [[None] * P for _ in beats]
    for bi, b in enumerate(beats):
        x = t[idx_of[b]] - b * B
        for pi, (d, _, pen) in enumerate(pats):
            c, j = _beat_cost(x, B, d, sigma, cap, merge_pen, fine_pen)
            cost[bi, pi] = c + pen
            choice[bi][pi] = j
            if beat_allow is not None and d not in beat_allow.get(int(b), (d,)):
                cost[bi, pi] += 1e6

    # 비터비: 연속된 박(사이 빈 박 2개 이하)끼리 계열이 바뀌면 벌점
    fam = np.array([p[1] for p in pats])
    acc = cost[0].copy()
    back = np.zeros((len(beats), P), dtype=int)
    for bi in range(1, len(beats)):
        gap = beats[bi] - beats[bi - 1]
        sw = switch_pen if gap <= 3 else 0.0
        trans = acc[:, None] + sw * (fam[:, None] != fam[None, :])      # [이전, 현재]
        back[bi] = np.argmin(trans, axis=0)
        acc = trans[back[bi], np.arange(P)] + cost[bi]
    path = np.zeros(len(beats), dtype=int)
    path[-1] = int(np.argmin(acc))
    for bi in range(len(beats) - 1, 0, -1):
        path[bi - 1] = back[bi, path[bi]]

    pos = np.empty(len(t))
    hist: dict[int, int] = {}
    for bi, b in enumerate(beats):
        d = pats[path[bi]][0]
        hist[d] = hist.get(d, 0) + 1
        pos[idx_of[b]] = b * B + choice[bi][path[bi]] * (B / d)
    key = np.round(pos, 6)
    uniq, inv = np.unique(key, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    names = {4: "16분", 3: "셋잇단8분", 6: "셋잇단16분", 8: "32분", 12: "셋잇단32분", 16: "64분"}
    i16 = next((pi for pi, p in enumerate(pats) if p[0] == 4), None)
    info = {"cost": float(acc.min()), "beats": {names.get(d, str(d)): n for d, n in sorted(hist.items())},
            "candidates": [names.get(p[0], str(p[0])) for p in pats],
            "beat_div": {int(b): int(pats[path[bi]][0]) for bi, b in enumerate(beats)},
            "beat_gain": {int(b): (float(cost[bi, i16] - cost[bi, path[bi]]) if i16 is not None else 0.0)
                          for bi, b in enumerate(beats)}}
    return uniq, groups, info


FINE48 = (0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42)     # 박당 48칸 중 16분, 32분, 셋잇단, 24분 칸


def _div_cells(d: int) -> set:
    """박당 48칸 중 분할 d의 칸"""
    return {k * 48 // d for k in range(d)} if 48 % d == 0 else set()


def refine_far(t_orig: np.ndarray, t_snap: np.ndarray, groups, bpm: float, far_ms: float = 25.0,
               min_gap_ms: float = 25.0, beat_div=None):
    """--snap-fix: 박 단위 격자에 붙이며 far_ms보다 멀리 옮겨진 노트를 32분, 24분 칸으로 다시 놓는다
    반환: (고유 시각, 묶음, 옮긴 수, 살린 수)"""
    t_orig = np.asarray(t_orig, dtype=np.float64)
    B = 60.0 / bpm
    far, gap = far_ms / 1000.0, min_gap_ms / 1000.0
    times = [float(x) for x in t_snap]
    grps = [list(map(int, g)) for g in groups]
    occ = sorted(times)
    fine = np.array(FINE48, dtype=np.float64) * (B / 48.0)

    base16 = {0, 12, 24, 36}
    cell_cache = {}

    def cells_of(b):
        if beat_div is None:
            return fine
        d = int(beat_div.get(int(b), 4))
        if d not in cell_cache:
            cs = sorted(base16 | (_div_cells(d) & set(FINE48)))
            cell_cache[d] = np.array(cs, dtype=np.float64) * (B / 48.0)
        return cell_cache[d]

    def nearest_fine(x):
        b = np.floor(x / B)
        cands = np.concatenate([(b - 1) * B + cells_of(b - 1), b * B + cells_of(b), (b + 1) * B + cells_of(b + 1)])
        return float(cands[np.argmin(np.abs(cands - x))])

    def free(x, ignore=None):
        import bisect
        i = bisect.bisect_left(occ, x - gap + 1e-9)
        while i < len(occ) and occ[i] < x + gap - 1e-9:
            if ignore is None or abs(occ[i] - ignore) > 1e-9:
                return False
            i += 1
        return True

    jobs = []
    for gi, g in enumerate(grps):
        for m in g:
            d = abs(t_orig[m] - times[gi])
            if d > far:
                jobs.append((d, gi, m))
    jobs.sort(reverse=True)
    n_move = n_split = 0
    new_times, new_groups = [], []
    import bisect
    for d, gi, m in jobs:
        if m not in grps[gi]:
            continue
        x = nearest_fine(t_orig[m])
        dn = abs(x - t_orig[m])
        if dn > far or dn > d - 0.008:
            continue
        if len(grps[gi]) == 1:
            if not free(x, ignore=times[gi]):
                continue
            occ.remove(times[gi])
            bisect.insort(occ, x)
            times[gi] = x
            n_move += 1
        else:
            if not free(x):
                continue
            grps[gi].remove(m)
            bisect.insort(occ, x)
            new_times.append(x)
            new_groups.append([m])
            n_split += 1
    times += new_times
    grps += new_groups
    order = np.argsort(times, kind="stable")
    return (np.array([times[i] for i in order]), [np.array(grps[i], dtype=int) for i in order], n_move, n_split)


TRI_DIVS = (3, 6, 12)      # 셋잇단 계열 (12, 24, 48비트)
B32_DIVS = (8, 16)         # 32분 계열 (32, 64비트)


def _fam(d: int):
    return "tri" if d in TRI_DIVS else ("32" if d in B32_DIVS else None)


def fine_gate(beat_div: dict, level: float = 1.0, share: float = 0.04):
    """--fine-grid: 곡 전체로 셋잇단, 32분 격자를 쓸지 정하고, 박마다 허용 분할을 돌려줌
    반환: (허용 계열 집합, {박: 허용 분할 집합}, 보고 dict)"""
    beats = sorted(int(b) for b in beat_div)
    n = max(len(beats), 1)
    if level <= 0:
        thr = float("inf")
    elif level < 1:
        thr = share / level
    else:
        thr = share * max(2.0 - level, 0.0)
    keep_iso = level >= 1.5
    run, alln, inrun = {"tri": 0, "32": 0}, {"tri": 0, "32": 0}, {}
    for b in beats:
        f = _fam(beat_div[b])
        if f is None:
            continue
        inrun[b] = any(x is not None and _fam(x) == f for x in (beat_div.get(b - 1), beat_div.get(b + 1)))
        alln[f] += 1
        run[f] += int(inrun[b])
    ok = {f for f in ("tri", "32") if run[f] > 0 and run[f] / n >= thr}
    allow, kept = {}, {"tri": 0, "32": 0}
    for b in beats:
        d = int(beat_div[b])
        f = _fam(d)
        if f is None:
            allow[b] = {4, d}
        elif f in ok and (inrun[b] or keep_iso):
            allow[b] = {4, d}
            kept[f] += 1
        else:
            allow[b] = {4}
    rep = {"beats": n, "all": alln, "run": run, "kept": kept, "ok": ok, "thr": thr}
    return ok, allow, rep


# --fine-bpm: 사람 277개 BPM대별 16비트 칸 밖 노트 쓰임(곡 평균, 150~180 대비 배수)을 다듬은 값.
# 사용자: '120의 32비트는 240의 16비트, 120의 24비트는 180의 16비트' - 느린 곡은 32, 24비트를 쉽게, 200+는 32비트 어렵게, 8분 셋잇단 쉽게.
FINE_BPM = np.array([125.0, 142.0, 165.0, 190.0, 215.0, 240.0])
FINE_RATIO = {8: np.array([1.3, 1.3, 1.0, 0.8, 0.25, 0.12]),     # 32분 (64분 16도 같이)
              6: np.array([2.0, 1.4, 1.0, 0.6, 0.6, 0.5]),       # 셋잇단16분 (셋잇단32분 12도 같이)
              3: np.array([0.8, 0.8, 1.0, 0.8, 1.8, 3.0])}       # 셋잇단8분 (4분 셋잇단 6비트 칸도 여기)


def fine_ratio(d: int, bpm: float) -> float:
    """분할 d의 BPM대 사람 쓰임 배수 (150~180 = 1)"""
    key = {16: 8, 12: 6}.get(int(d), int(d))
    if key not in FINE_RATIO:
        return 1.0
    return float(np.interp(bpm, FINE_BPM, FINE_RATIO[key]))


def bpm_patterns(patterns, bpm: float, strength: float = 1.0, beta: float = 1.0, floor: float = 0.5):
    """박 분할 후보 벌점을 BPM대 사람 쓰임에 맞춰 - 벌점 - beta x strength x ln(배수) (16분은 그대로, 최소 floor)"""
    out = []
    for d, fam, pen in patterns:
        if d == 4 or strength <= 0:
            out.append((d, fam, pen))
            continue
        out.append((d, fam, max(floor, pen - beta * strength * float(np.log(fine_ratio(d, bpm))))))
    return tuple(out)


def fine_gate_beat(beat_div: dict, beat_gain: dict, bpm: float, strength: float = 1.0, g0: float = 10.0,
                   alpha: float = 3.0):
    """--fine-bpm: 곡 전체 4% 문턱(fine_gate) 대신 박마다 노래를 따라 정함
    반환: (남긴 계열 집합, {박: 허용 분할 집합}, 보고 dict)"""
    beats = sorted(int(b) for b in beat_div)
    n = max(len(beats), 1)
    allow, kept, alln, run = {}, {"tri": 0, "32": 0}, {"tri": 0, "32": 0}, {"tri": 0, "32": 0}
    thr = {}
    for b in beats:
        d = int(beat_div[b])
        f = _fam(d)
        if f is None:
            allow[b] = {4, d}
            continue
        alln[f] += 1
        inrun = any(x is not None and _fam(x) == f for x in (beat_div.get(b - 1), beat_div.get(b + 1)))
        run[f] += int(inrun)
        G = g0 - alpha * strength * float(np.log(fine_ratio(d, bpm)))
        thr[d] = G
        g = float(beat_gain.get(b, 0.0))
        if g >= G or (inrun and g >= 0.5 * G):
            allow[b] = {4, d}
            kept[f] += 1
        else:
            allow[b] = {4}
    ok = {f for f in ("tri", "32") if kept[f] > 0}
    rep = {"beats": n, "all": alln, "run": run, "kept": kept, "ok": ok, "thr": thr}
    return ok, allow, rep


def periodic_fine(t_chart: np.ndarray, bpm: float, divs=(6,), sigma_ms: float = 6.0, cap: float = 9.0,
                  merge_pen: float = 0.5, need: int = 2, total: float = 10.0, phrase_beats: int = 16):
    """악절(4마디) 안 같은 박 자리에서 되풀이되는 셋잇단16분(24비트), 32분 모양을 찾는다
    반환: ({박 번호: 분할}, {분할: {박: 이득}})"""
    t = np.asarray(t_chart, dtype=np.float64)
    B = 60.0 / bpm
    sigma = sigma_ms / 1000.0
    beat = np.floor(t / B + 1e-9).astype(int)
    gains = {d: {} for d in divs}
    for b in np.unique(beat):
        x = t[beat == b] - b * B
        if len(x) < 2:
            continue
        c4, _ = _beat_cost(x, B, 4, sigma, cap, merge_pen, 0.0)
        for d in divs:
            cd, _ = _beat_cost(x, B, d, sigma, cap, merge_pen, 0.0)
            gains[d][int(b)] = float(c4 - cd)
    best = {}
    for d in divs:
        groups: dict = {}
        for b, v in gains[d].items():
            groups.setdefault((b // phrase_beats, b % 4), []).append((b, v))
        for key, L in groups.items():
            pos = [(b, v) for b, v in L if v > 0]
            tot = sum(v for _, v in pos)
            if len(pos) >= need and tot >= total and (key not in best or tot > best[key][0]):
                best[key] = (tot, d, [b for b, _ in pos])
    out = {}
    for tot, d, bs in best.values():
        for b in bs:
            out[b] = d
    return out, gains


def periodic_tri(t_chart: np.ndarray, bpm: float, **kw):
    """셋잇단16분만 (periodic_fine(divs=(6,))). 반환: (박 집합, {박: 이득})"""
    out, gains = periodic_fine(t_chart, bpm, divs=(6,), **kw)
    return set(out), gains[6]


def choose_beat_phase(t: np.ndarray, bpm: float, sixteenth_phase: float, tie: float = 0.2, **kw) -> float:
    """박 시작 위치(16분 칸 4개 중 하나)를 고른다"""
    t = np.asarray(t, dtype=np.float64)
    B = 60.0 / bpm
    g = B / 4
    best = None
    for k in range(4):
        bp = (sixteenth_phase + k * g) % B
        pad = (-bp) % B
        _, _, info = snap_beats(t + pad, bpm, **kw)
        d = ((t - bp + B / 2) % B) - B / 2
        score = info["cost"] - tie * int((np.abs(d) < 0.25 * g).sum())
        if best is None or score < best[0]:
            best = (score, bp)
    return best[1]

