"""
[모델 보조, STEP 4] 반 마디(2박) 단위 '패턴 계획' 특징 (v4)
"""
from __future__ import annotations

import numpy as np

PLAN_NAMES = ["chord", "alt", "step", "cover", "jack", "move", "scratch"]
PLAN_DIM = len(PLAN_NAMES)
GROUP_GAP = 120          # 이보다 긴 빈틈(프레임, 1.2초)이면 새 묶음


def _beat_of(code: int) -> int:
    if code < 0:
        return -1
    return code // 4 if code < 16 else code - 16


def half_groups(frames: np.ndarray, codes: np.ndarray | None) -> np.ndarray:
    """노트마다 반 마디 묶음 번호 (0부터). codes = 박자 코드(lane_v2.rhythm_code), 없으면 1초 단위로 자름."""
    f = np.asarray(frames, dtype=np.int64)
    n = len(f)
    g = np.zeros(n, dtype=np.int64)
    if n == 0:
        return g
    if codes is None:
        codes = np.full(n, -1)
    gid, prev_beat, start = 0, None, f[0]
    for i in range(n):
        b = _beat_of(int(codes[i]))
        new = False
        if i > 0:
            if f[i] - f[i - 1] > GROUP_GAP:
                new = True
            elif b >= 0 and prev_beat is not None and prev_beat >= 0:
                if b // 2 != prev_beat // 2 or b < prev_beat:      # 다른 반 마디, 또는 다음 마디로 넘어감
                    new = True
            elif b < 0 and f[i] - start > 100:                      # 박자 정보가 없으면 1초 단위
                new = True
        if new:
            gid += 1
            start = f[i]
        g[i] = gid
        if b >= 0:
            prev_beat = b
    return g


def is_alt(a: int, b: int, c: int) -> bool:
    """세 줄(건반 마스크) a(i-2), b(i-1), c(i)가 묶음 교대(A-B-A)인가"""
    pc = lambda k: bin(k).count("1")
    if pc(a) < 2 or pc(b) < 2 or pc(c) < 2 or (c & b) or (a & b):
        return False
    return pc(a & c) / pc(a | c) >= 0.6


def alt_run_ratio(masks, frames, min_run: int = 4, max_gap: int = 40) -> float:
    """묶음 교대(A-B-A-B)가 min_run줄 이상 이어진 줄의 비율. 세 줄이 max_gap 프레임(0.4초) 안이어야 함"""
    k = [int(m) & 0xFE for m in masks]
    tot, L = 0, 0
    for i in range(2, len(k)):
        if is_alt(k[i - 2], k[i - 1], k[i]) and frames[i] - frames[i - 2] <= max_gap:
            L = L + 1 if L else 3
        else:
            tot += L if L >= min_run else 0
            L = 0
    tot += L if L >= min_run else 0
    return tot / max(len(k), 1)


def group_plan(masks: np.ndarray) -> np.ndarray:
    """한 묶음(반 마디)의 마스크들 -> (PLAN_DIM,)"""
    out = np.zeros(PLAN_DIM, dtype=np.float32)
    ms = [int(m) for m in masks]
    if not ms:
        return out
    keys = [m & 0xFE for m in ms]
    kr = [k for k in keys if k]
    out[6] = np.mean([m & 1 for m in ms])
    if not kr:
        return out
    sizes = [bin(k).count("1") for k in kr]
    out[0] = min(np.mean(sizes) / 4.0, 1.5)
    used = 0
    for k in kr:
        used |= k
    out[3] = bin(used).count("1") / 7.0
    if len(kr) >= 2:
        alt = step = jack = 0
        mv = []
        for i in range(2, len(kr)):
            alt += is_alt(kr[i - 2], kr[i - 1], kr[i])
        for a, b in zip(kr[:-1], kr[1:]):
            la = [x for x in range(1, 8) if (a >> x) & 1]
            lb = [x for x in range(1, 8) if (b >> x) & 1]
            step += len(la) == 1 and len(lb) == 1 and abs(la[0] - lb[0]) == 1
            jack += (a & b) != 0
            mv.append(abs(np.mean(la) - np.mean(lb)))
        p = len(kr) - 1
        out[1] = alt / max(len(kr) - 2, 1)
        out[2], out[4] = step / p, jack / p
        out[5] = np.mean(mv) / 6.0
    return out


def plan_features(masks: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(노트별 계획 (n, PLAN_DIM), 묶음별 계획 (G, PLAN_DIM))"""
    G = int(groups.max()) + 1 if len(groups) else 0
    per_g = np.zeros((G, PLAN_DIM), dtype=np.float32)
    for gi in range(G):
        per_g[gi] = group_plan(masks[groups == gi])
    return per_g[groups], per_g


if __name__ == "__main__":
    # 묶음 나누기: 박 코드 0,4(앞 반) | 8,12(뒤 반) | 다음 마디 0 ...
    fr = np.array([0, 10, 20, 30, 40, 50, 300, 310])
    cd = np.array([0, 4, 8, 12, 0, 6, 0, 4])
    assert list(half_groups(fr, cd)) == [0, 0, 1, 1, 2, 2, 3, 3]
    B = lambda *ls: sum(1 << l for l in ls)
    # 1357/246 교대 4줄, 1245/367 교대, 135/246/1357 변형
    p = group_plan(np.array([B(1, 3, 5, 7), B(2, 4, 6), B(1, 3, 5, 7), B(2, 4, 6)]))
    assert abs(p[1] - 1.0) < 1e-6 and p[3] == 1.0 and p[4] == 0
    p = group_plan(np.array([B(1, 2, 4, 5), B(3, 6, 7), B(1, 2, 4, 5), B(3, 6, 7)]))
    assert abs(p[1] - 1.0) < 1e-6
    p = group_plan(np.array([B(1, 3, 5), B(2, 4, 6), B(1, 3, 5, 7)]))
    assert abs(p[1] - 1.0) < 1e-6
    p = group_plan(np.array([B(1, 3), B(2, 4), B(5, 7)]))          # 옮겨 가는 덴님은 교대가 아님
    assert p[1] == 0
    # 계단 1-2-3-4
    p = group_plan(np.array([B(1), B(2), B(3), B(4)]))
    assert abs(p[2] - 1.0) < 1e-6 and p[1] == 0 and abs(p[0] - 0.25) < 1e-6
    # 스크래치 + 연타
    p = group_plan(np.array([B(0, 3), B(3), B(0, 3)]))
    assert abs(p[6] - 2 / 3) < 1e-6 and abs(p[4] - 1.0) < 1e-6
    pn, pg = plan_features(np.array([B(1), B(2), B(1, 3), B(2, 4)]), np.array([0, 0, 1, 1]))
    assert pn.shape == (4, PLAN_DIM) and pg.shape == (2, PLAN_DIM)
    print("plan_feats OK")
