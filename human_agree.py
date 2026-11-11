"""
[점검] 같은 곡을 다른 사람이 찍은 패턴끼리 줄 모양이 얼마나 같은가 (레인 모델 정확도의 '상한' 감)

사용:
    python human_agree.py --bms-dir data/bms data/bms_removed
    python human_agree.py --bms-dir "D:/bms/곡_폴더"          # 곡 폴더 하나만
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
import collections
import itertools
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

EXTS = {".bms", ".bme", ".bml"}
LANE = {"16": 0, "11": 1, "12": 2, "13": 3, "14": 4, "15": 5, "18": 6, "19": 7}
LN_LANE = {"56": 0, "51": 1, "52": 2, "53": 3, "54": 4, "55": 5, "58": 6, "59": 7}
MIRROR = np.array([(m & 1) | sum(1 << (8 - k) for k in range(1, 8) if (m >> k) & 1) for m in range(256)], dtype=np.int64)
POP = np.array([bin(m & 0xFE).count("1") for m in range(256)])


def read_text(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("shift_jis", "cp949", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def parse_rows(p: Path):
    """SP 7키 패턴 -> {(마디, 마디 안 위치 Fraction): 마스크}. DP/5키/9키는 None"""
    rows = collections.defaultdict(int)
    ln_ev = collections.defaultdict(list)          # LN 채널(51~59)은 시작/끝이 번갈아 나옴 -> 시작만 노트로
    dp = False
    lnobj = None
    for line in read_text(p).splitlines():
        line = line.strip()
        m = re.match(r"#LNOBJ\s+(\S\S)", line, re.I)
        if m:
            lnobj = m[1].upper()
        m = re.match(r"#(\d{3})([0-9A-Za-z]{2}):(\S+)", line)
        if not m:
            continue
        ch = m[2]
        if ch[0] == "2" or ch[0] == "6":
            if m[3].strip("0"):
                dp = True
            continue
        lane = LANE.get(ch, LN_LANE.get(ch))
        if lane is None:
            continue
        d = m[3]
        n = len(d) // 2
        ln = ch in LN_LANE
        for i in range(n):
            v = d[2 * i:2 * i + 2].upper()
            if v == "00" or (lnobj and v == lnobj):
                continue
            pos = (int(m[1]), Fraction(i, n))
            if ln:
                ln_ev[lane].append(pos)
            else:
                rows[pos] |= 1 << lane
    for lane, ev in ln_ev.items():
        for pos in sorted(ev)[0::2]:
            rows[pos] |= 1 << lane
    if dp or not rows:
        return None
    used = 0
    for v in rows.values():
        used |= v
    if not (used >> 6) & 1 and not (used >> 7) & 1:      # 6, 7번 건반을 안 쓰면 5키로 봄
        return None
    return dict(rows)


def compare(a: dict, b: dict):
    keys = [k for k in a if k in b]
    if not keys:
        return np.zeros(4, dtype=np.int64)
    x = np.array([a[k] for k in keys]); y = np.array([b[k] for k in keys])
    return np.array([(x == y).sum(), ((x == y) | (MIRROR[x] == y)).sum(), (POP[x] == POP[y]).sum(), len(keys)])


def notes(r: dict) -> int:
    return int(sum(POP[v] + (v & 1) for v in r.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bms-dir", nargs="+", required=True, help="곡 폴더들이 들어 있는 폴더(여러 개 가능) 또는 곡 폴더 하나")
    ap.add_argument("--min-notes", type=int, default=1000, help="노트 수가 이보다 적은 차분은 뺌 (쉬운 차분)")
    ap.add_argument("--similar", type=float, default=0.25, help="노트 수 차이가 이 비율 이하인 쌍만 따로 집계")
    ap.add_argument("--min-common", type=float, default=0.5, help="같은 박 위치 줄이 양쪽 줄 수의 이 비율 미만이면 뺌 (박자 구조가 다른 차분)")
    args = ap.parse_args()

    songs = collections.defaultdict(list)
    for root in args.bms_dir:
        root = Path(root)
        for f in root.rglob("*"):
            if f.suffix.lower() in EXTS:
                songs[f.parent.name].append(f)
    tot = np.zeros(4); tot_sim = np.zeros(4); pairs = n_sim = 0
    per_pair, used_songs, skipped = [], 0, 0
    for song, files in sorted(songs.items()):
        charts = []
        for f in files:
            try:
                r = parse_rows(f)
            except Exception:
                r = None
            if r is not None and notes(r) >= args.min_notes:
                charts.append((f.name, r))
        if len(charts) < 2:
            continue
        got = 0
        for (na, a), (nb, b) in itertools.combinations(charts, 2):
            c = compare(a, b)
            if c[3] < args.min_common * min(len(a), len(b)):
                skipped += 1
                continue
            tot += c; pairs += 1; got += 1
            per_pair.append(c[0] / c[3])
            if abs(notes(a) - notes(b)) / max(notes(a), notes(b)) <= args.similar:
                tot_sim += c; n_sim += 1
        if got:
            used_songs += 1
            print(f"  {song[:40]:40s} 차분 {len(charts)}개, 쌍 {got}개")
    if not pairs:
        raise SystemExit("비교할 쌍이 없어요. 곡 폴더 안에 같은 곡 차분(SP 7키, 노트 --min-notes 이상)이 2개 이상 있어야 해요")

    def line(name, t, n):
        print(f"  {name:28s} 쌍 {n:4d} | 같은 위치 줄 {int(t[3]):,}개 | 줄 모양 같음 {t[0] / t[3]:.1%} | "
              f"좌우 반전 허용 {t[1] / t[3]:.1%} | 건반 수 같음 {t[2] / t[3]:.1%}")

    print(f"\n사람 차분끼리 (곡 {used_songs}개, 박자 구조가 달라 뺀 쌍 {skipped}개):")
    line("전체", tot, pairs)
    if n_sim:
        line(f"노트 수 비슷한 쌍(±{args.similar:.0%})", tot_sim, n_sim)
    print(f"  쌍별 줄 모양 같음: 중앙값 {np.median(per_pair):.1%}, 상위 10% {np.percentile(per_pair, 90):.1%}")
    print("비교: 레인 모델 v5 (lane_eval, 정답 이력+계획 입력) 줄 모양 25.4%, 건반 수 56.1%")


if __name__ == "__main__":
    main()
