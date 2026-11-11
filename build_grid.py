"""
[학습 준비] 학습용 '반 마디 밀도' 정답을 만든다 (density_model.py 학습용)

사용:
    python build_grid.py --processed data/processed --out data/grid
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from bms_parser import parse_bms

FPS = 100


def chart_grid(c, T: int):
    """마디 단위로 반 마디를 나열한다. BPM 변경/STOP/마디 길이 변경도 BMS 타이밍 그대로 따른다."""
    from fractions import Fraction
    notes = [n for n in c.notes if 0 <= n.lane <= 7 and n.beat >= 0]
    if len(notes) < 10:
        return None
    rows = defaultdict(set)
    for n in notes:
        rows[round(n.beat * 48) / 48].add(n.lane)
    ms = c.measure_starts
    last = max(rows)
    starts_b, ends_b, halves = [], [], []
    for m in range(len(ms) - 1):
        a, e = ms[m], ms[m + 1]
        if a > last + 4:
            break
        mid = a + (e - a) / 2
        for j, (x, y) in enumerate(((a, mid), (mid, e))):
            if y - x > 1e-9:
                starts_b.append(x); ends_b.append(y); halves.append(j)
    sb, eb = np.array(starts_b), np.array(ends_b)
    to_f = lambda bb: int(round(c.timing.to_time(Fraction(bb).limit_denominator(4800)) * FPS))
    start = np.array([to_f(x) for x in sb], dtype=np.int64)
    end = np.array([to_f(x) for x in eb], dtype=np.int64)
    H = len(sb)
    r = np.zeros(H, np.int16); b = np.zeros(H, np.int16); nn = np.zeros(H, np.int16)
    for beat, lanes in rows.items():
        h = int(np.searchsorted(sb, beat + 1e-9, side="right") - 1)
        if not 0 <= h < H:
            continue
        r[h] += 1
        nn[h] += len(lanes)
        q = (beat - sb[h]) * 4                         # 반 마디 시작부터 16분 단위
        if abs(q - round(q)) < 1e-6 and int(round(q)) % 2 == 1:
            b[h] += 1
    ok = (start >= 0) & (end <= T) & (end - start >= 10)   # STOP 등으로 너무 짧은 구간 제외
    first = int(np.argmax(r > 0))
    sel = ok & (np.arange(H) >= max(first - 2, 0))
    return {"start": start[sel], "end": end[sel], "rows": r[sel], "btw": b[sel], "notes": nn[sel],
            "half": np.array(halves, np.int8)[sel]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/grid")
    args = ap.parse_args()
    proc, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(proc / "index.csv", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    ok, failed, tot = 0, [], defaultdict(int)
    for r in rows:
        try:
            g = chart_grid(parse_bms(Path(r["source"])), int(r["frames"]))
            if g is None:
                raise ValueError("노트가 너무 적음")
            np.savez(out / f"{r['chart_id']}.grid.npz", **g)
            ok += 1
            rr = g["rows"]
            tot["half"] += len(rr); tot["rest"] += int((rr <= 2).sum()); tot["b8"] += int(((rr >= 3) & (rr <= 5)).sum())
            tot["b16"] += int((rr >= 6).sum()); tot["zero"] += int((rr == 0).sum())
        except Exception as e:  # noqa: BLE001
            failed.append((r["chart_id"], repr(e)))
    print(f"완료: {ok}개 저장, 실패 {len(failed)}개 -> {out}")
    for cid, e in failed[:10]:
        print(f"  실패 {cid}: {e}")
    if tot["half"]:
        h = tot["half"]
        print(f"확인용: 반 마디 {h}개 | 쉼(0~2줄) {tot['rest'] / h:.0%} (그중 완전 빈 반 마디 {tot['zero'] / h:.0%}), "
              f"8비트(3~5줄) {tot['b8'] / h:.0%}, 16비트(6줄+) {tot['b16'] / h:.0%}")


if __name__ == "__main__":
    main()
