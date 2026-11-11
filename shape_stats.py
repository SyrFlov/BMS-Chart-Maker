"""
[점검] 폭타 뼈대, 마디 반복, 대각선 수치 (사람 패턴 vs 생성 패턴 비교용)

사용:
    python shape_stats.py --label 사람 "BMS_폴더"
    python shape_stats.py --label 생성 dist/BMSChartAI/output
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
from pathlib import Path

import numpy as np

from human_agree import EXTS, parse_rows

KEYS = 0xFE


def cnt(m: int) -> int:
    return bin(m & KEYS).count("1")


def mirror(m: int) -> int:
    return sum(1 << (8 - k) for k in range(1, 8) if (m >> k) & 1)


def shifts(m: int) -> set[int]:
    m &= KEYS
    out = set()
    for d in range(-6, 7):
        if d == 0:
            continue
        s = m << d if d > 0 else m >> -d
        if s and not s & ~KEYS & 0x1FF and cnt(s) == cnt(m) and s <= KEYS:
            out.add(s)
    return out


def grid(rows: dict) -> dict[int, int]:
    """{(마디, 위치): 마스크} -> {16분 칸 번호: 건반 마스크} (16분 격자 위, 건반 있는 줄만)"""
    g = {}
    for (meas, frac), m in rows.items():
        q = frac * 16
        if q.denominator == 1 and m & KEYS:
            g[meas * 16 + int(q)] = m & KEYS
    return g


def stats(rows: dict) -> dict[str, float]:
    g = grid(rows)
    stream = {p for p in g if p - 1 in g or p + 1 in g}
    on8 = [p for p in stream if p % 2 == 0]
    pairs = [(p, p + 1) for p in on8 if p + 1 in g]
    r = {}
    r["8분>뒤"] = np.mean([cnt(g[a]) > cnt(g[b]) for a, b in pairs]) if pairs else np.nan
    r["뒤>8분"] = np.mean([cnt(g[b]) > cnt(g[a]) for a, b in pairs]) if pairs else np.nan
    r["8분단타"] = np.mean([cnt(g[p]) == 1 for p in on8]) if on8 else np.nan
    r["8분4+"] = np.mean([cnt(g[p]) >= 4 for p in on8]) if on8 else np.nan
    per = {}
    for p in g:
        per.setdefault(p // 16, []).append(p)
    same_n, same_s, sim = [], [], []
    for meas, ps in per.items():
        prev = per.get(meas - 1)
        if not prev or len(ps) < 4 or len(prev) < 4:
            continue
        for p in ps:
            q = p - 16
            if q not in g:
                continue
            a, b = g[p], g[q]
            same_n.append(cnt(a) == cnt(b))
            same_s.append(a == b)
            sim.append(a == b or a == mirror(b) or a in shifts(b))
    r["마디수같음"] = np.mean(same_n) if same_n else np.nan
    r["마디모양같음"] = np.mean(same_s) if same_s else np.nan
    r["마디닮음"] = np.mean(sim) if sim else np.nan
    dg = [g[p + 2] in ((g[p] << 1) & KEYS, g[p] >> 1) and cnt(g[p + 2]) == cnt(g[p])
          for p in stream if p + 1 in g and p + 2 in g]
    r["대각선"] = np.mean(dg) if dg else np.nan
    r["줄"] = len(g)
    # 8분 자리 평균, 한 박 단어(16분 4칸 다 폭타), 박 안 두 8분 자리
    r["8분평균"] = np.mean([cnt(g[p]) for p in on8]) if on8 else np.nan
    words, pair = [], []
    for p in stream:
        if p % 4 == 0 and all(p + k in stream for k in range(4)):
            words.append("".join(str(min(cnt(g[p + k]), 9)) for k in range(4)))
        if p % 4 == 0 and p + 2 in stream:
            pair.append(np.sign(cnt(g[p]) - cnt(g[p + 2])))
    r["4242류"] = np.mean([w in ("4242", "4141", "4232") for w in words]) if words else np.nan
    r["들쭉박"] = np.mean([w in ("4231", "2131", "2111") for w in words]) if words else np.nan
    r["박안같음"] = np.mean([s == 0 for s in pair]) if pair else np.nan
    r["박안첫>"] = np.mean([s > 0 for s in pair]) if pair else np.nan
    four = [m for m in g.values() if cnt(m) == 4]
    r["2356"] = np.mean([m == 0b1101100 for m in four]) if four else np.nan
    r["1357"] = np.mean([m == 0b10101010 for m in four]) if four else np.nan
    # 32분, 24분 (16분 격자 밖, 앞뒤 같은 간격 이웃)
    fr = fine_rows(rows)
    for kind in (32, 24):
        xs = [cnt(m) == 1 for kk, m, _ in fr if kk == kind]
        r[f"{kind}단타"] = np.mean(xs) if xs else np.nan
    shp = [classify(ks) for ks in fine_run_keys(rows)]
    for name in SHAPES:
        r[name] = np.mean([s == name for s in shp]) if shp else np.nan
    r["연속수"] = len(shp)
    return r


SHAPES = ["계단", "교대", "지그재그", "좁은반복", "규칙없음"]


def _beats(rows: dict):
    """{(마디, 위치): 마스크} -> 시간 순 [(박 위치 Fraction, 마스크)] (4/4 가정, 마디 = 4박)"""
    from fractions import Fraction
    return sorted((Fraction(meas * 4) + frac * 4, m) for (meas, frac), m in rows.items() if m & KEYS)


def _fine_kind(g) -> int:
    """간격(박) -> 32 / 24 / 0"""
    if abs(g - 1 / 8) <= 0.15 / 8:
        return 32
    if abs(g - 1 / 6) <= 0.15 / 6:
        return 24
    return 0


def fine_rows(rows: dict):
    """16분 격자 밖 줄 중 앞뒤 간격이 같은(32분 또는 24분) 줄 [(종류, 마스크, 번호)]"""
    b = _beats(rows)
    out = []
    for i in range(1, len(b) - 1):
        x = b[i][0] * 4
        if x.denominator == 1:
            continue
        g1, g2 = float(b[i][0] - b[i - 1][0]), float(b[i + 1][0] - b[i][0])
        k = _fine_kind(g1)
        if k and _fine_kind(g2) == k:
            out.append((k, b[i][1], i))
    return out


def fine_run_keys(rows: dict):
    """24분, 32분 같은 간격 3줄 이상 연속(16분 격자 밖 줄 포함)마다 줄 중심(건반 평균 위치) 목록"""
    b = _beats(rows)
    n = len(b)
    out, i = [], 0
    while i < n - 1:
        k = _fine_kind(float(b[i + 1][0] - b[i][0]))
        if not k:
            i += 1
            continue
        j = i + 1
        while j < n - 1 and _fine_kind(float(b[j + 1][0] - b[j][0])) == k:
            j += 1
        seg = list(range(i, j + 1))
        off = [x for x in seg if (b[x][0] * 4).denominator != 1]
        R = [x for x in seg if (b[x][0] * 4).denominator != 1 or i < x < j]
        if len(seg) >= 3 and off and len(R) >= 3:
            out.append([float(np.mean([k for k in range(1, 8) if (b[x][1] >> k) & 1])) for x in R])
        i = j
    return out


def classify(ks: list[float]) -> str:
    """줄 중심 목록 -> 교대(손이 번갈아) / 계단(한쪽으로) / 지그재그(방향이 번갈아) / 좁은반복(폭 2칸 안) / 규칙없음"""
    d = [b - a for a, b in zip(ks, ks[1:])]
    hand = ["L" if k <= 3.5 else "R" if k >= 4.5 else "C" for k in ks]
    if "C" not in hand and all(a != b for a, b in zip(hand, hand[1:])):
        return "교대"
    if all(x > 0 for x in d) or all(x < 0 for x in d):
        return "계단"
    if all(x != 0 for x in d) and all((a > 0) != (b > 0) for a, b in zip(d, d[1:])):
        return "지그재그"
    if max(ks) - min(ks) <= 2:
        return "좁은반복"
    return "규칙없음"


COLS = ["8분>뒤", "뒤>8분", "8분단타", "8분4+", "마디수같음", "마디모양같음", "마디닮음", "대각선"]
COLS2 = ["32단타", "24단타", "8분평균", "4242류", "들쭉박", "박안같음", "박안첫>", "2356", "1357"] + SHAPES


def files_of(paths):
    for p in map(Path, paths):
        if p.is_file():
            yield p
        else:
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true", help="차분별 줄은 빼고 요약만")
    ap.add_argument("--set", type=int, default=1, help="1 = 뼈대·반복·대각선 / 2 = 32·24분, 8분 자리 두께, 박, 4건반, 연속 모양")
    a = ap.parse_args()
    global COLS
    if a.set == 2:
        COLS = COLS2
    res = []
    print(f"{'차분':<34}" + "".join(f"{c:>9}" for c in COLS))
    for f in files_of(a.paths):
        rows = parse_rows(f)
        if rows is None:
            continue
        s = stats(rows)
        if s["줄"] < 300:
            continue
        res.append(s)
        if not a.quiet:
            name = (f.parent.name + "/" + f.stem)[-34:]
            print(f"{name:<34}" + "".join(f"{s[c]:>9.2f}" if c == "8분평균" else f"{s[c]:>9.0%}" for c in COLS))
    if res:
        M = np.array([[s[c] for c in COLS] for s in res], dtype=np.float64)
        print(f"{(a.label + ' 중앙값 (' + str(len(res)) + '개)'):<34}" +
              "".join(f"{v:>9.2f}" if c == "8분평균" else f"{v:>9.0%}" for c, v in zip(COLS, np.nanmedian(M, 0))))
        if "계단" in COLS:                        # 연속 모양은 패턴마다가 아니라 전체 합으로도
            tot = sum(s["연속수"] for s in res)
            print(f"{'  연속 모양 전체 (' + str(tot) + '곳)':<34}" + "".join(
                f"{sum(s[c] * s['연속수'] for s in res if s['연속수']) / max(tot, 1):>9.0%}" if c in SHAPES else f"{'':>9}"
                for c in COLS))
        print(f"{'  10~90%':<34}" + "".join(f"{lo:>4.0%}~{hi:<4.0%}"[:9].rjust(9)
                                             for lo, hi in zip(np.nanpercentile(M, 10, 0), np.nanpercentile(M, 90, 0))))


if __name__ == "__main__":
    main()
