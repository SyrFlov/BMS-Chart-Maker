"""
[학습 준비] 학습 데이터에 '박자 위치'(마디 안 몇째 박, 박 안 16분 위치) 정보를 추가로 만든다

build_dataset.py가 만든 data/processed/index.csv의 원본 BMS 경로를 다시 읽어서,
패턴마다 {chart_id}.rhy.npy (프레임별 박자 코드, 노트 없는 프레임은 -1) 를 --out 폴더에 저장.
멜/라벨은 건드리지 않으므로 빠르고(몇 분), 결과 폴더만 Kaggle에 작은 데이터셋으로 올리면 된다.

사용 예 (bmsaiproject 폴더에서):
    python build_rhythm.py --processed data/processed --out data/rhythm
    -> data/rhythm 폴더를 zip으로 압축해 Kaggle 새 데이터셋(예: bms-rhythm)으로 업로드
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from bms_parser import parse_bms
from lane_v2 import rhythm_code

FPS = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/rhythm")
    args = ap.parse_args()

    proc, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(proc / "index.csv", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))

    ok, failed = 0, []
    sc_pos, all_pos = Counter(), Counter()
    for r in rows:
        try:
            c = parse_bms(Path(r["source"]))
            T = int(r["frames"])
            arr = np.full(T, -1, dtype=np.int8)
            for n in c.notes:
                f = int(round(n.time * FPS))
                if 0 <= f < T and arr[f] < 0:
                    code = rhythm_code(n.mbeat)
                    arr[f] = code
                    all_pos[code] += 1
                if n.lane == 0:
                    sc_pos[rhythm_code(n.mbeat)] += 1
            np.save(out / f"{r['chart_id']}.rhy.npy", arr)
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append((r["chart_id"], repr(e)))

    print(f"완료: {ok}개 저장, 실패 {len(failed)}개 -> {out}")
    for cid, e in failed[:10]:
        print(f"  실패 {cid}: {e}")
    tot = sum(sc_pos.values())
    if tot:
        down = sc_pos[0] / tot * 100
        onbeat = sum(sc_pos[k] for k in (0, 4, 8, 12)) / tot * 100
        print(f"확인용: 스크래치 중 마디 첫 박 {down:.0f}%, 정박 {onbeat:.0f}% (보유 차분 381개 기준 중앙값 41%, 보통 25~55%)")
    tn = sum(all_pos.values())
    if tn:
        print(f"확인용: 노트 시각 중 16분 격자 밖(셋잇단 등) {sum(all_pos[k] for k in range(16, 20)) / tn * 100:.1f}%")


if __name__ == "__main__":
    main()
