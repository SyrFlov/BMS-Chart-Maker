"""
[학습] 레인 모델 학습용 Dataset: 연속된 노트 L개를 한 샘플로 뽑는다.

build_dataset.py가 만든 결과(*.mel.npy, *.label.npy, index.csv)를 그대로 쓴다. 전처리를 다시 돌릴 필요 없음.

샘플 (노트 L개):
    patch  (L, n_mels, 2k+1)  각 노트 시각 주변의 멜 조각
    aux    (L, 11)            직전 마스크(8) + 직전/직후 간격(2) + 곡 밀도(1)   <- 입력
    target (L,)               정답 클래스 = 마스크 - 1 (0~254), 모자란 자리는 -100(무시)

학습할 때는 직전 노트로 "정답"을 넣고(teacher forcing) 다음 노트를 맞히게 한다.
같은 곡의 패턴은 index.csv의 split대로 같은 쪽에 들어간다 (곡 단위 분할).
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import csv
import json
from pathlib import Path

import numpy as np

from lane_utils import (AUX_DIM, aux_vector, cond_value, extract_patches, gap_arrays,
                        masks_from_labels)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:          # torch가 없어도 numpy로 동작 (테스트용)
    torch = None
    Dataset = object


def _t(a: np.ndarray):
    return torch.from_numpy(a) if torch is not None else a


class LaneSequences(Dataset):
    def __init__(self, root: str | Path, split: str = "train", seq_len: int = 64, random: bool = True,
                 epoch_size: int | None = None, chart_ids: list[str] | None = None, patch_k: int = 8):
        self.root = Path(root)
        self.cfg = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.L, self.k, self.random = seq_len, patch_k, random

        with open(self.root / "index.csv", newline="", encoding="utf-8-sig") as fp:
            rows = [r for r in csv.DictReader(fp) if r["split"] == split]
        if chart_ids is not None:
            keep = set(chart_ids)
            rows = [r for r in rows if r["chart_id"] in keep]

        self.charts: list[dict] = []
        for r in rows:
            cid = r["chart_id"]
            frames, masks = masks_from_labels(np.load(self.root / f"{cid}.label.npy"))
            if len(frames) == 0:
                continue
            dtp, dtn = gap_arrays(frames)
            self.charts.append({
                "chart_id": cid, "frames": frames, "masks": masks, "dtp": dtp, "dtn": dtn,
                "mel": np.load(self.root / f"{cid}.mel.npy", mmap_mode="r"),
                "cond": cond_value(float(r["density"])), "n": len(frames),
            })
        if not self.charts:
            raise ValueError(f"split='{split}'에 해당하는 차분이 없어요 (곡 수가 적으면 --val-ratio 조정)")

        sizes = np.array([c["n"] for c in self.charts])
        if random:
            self.p = sizes / sizes.sum()
            self.n = epoch_size or max(1, int(sizes.sum() // self.L))
        else:
            self.windows = [(i, s) for i, c in enumerate(self.charts) for s in range(0, c["n"], self.L)]
            self.n = len(self.windows)

    def __len__(self) -> int:
        return self.n

    def _make(self, ci: int, s: int) -> dict:
        c = self.charts[ci]
        n, L = c["n"], self.L
        idx = np.arange(s, s + L)
        valid = idx < n
        vidx = idx[valid]
        M = c["mel"].shape[1]

        patch = np.zeros((L, M, 2 * self.k + 1), dtype=np.float32)
        patch[valid] = extract_patches(c["mel"], c["frames"][vidx], self.k)
        aux = np.zeros((L, AUX_DIM), dtype=np.float32)
        for j, i in enumerate(vidx):
            prev = int(c["masks"][i - 1]) if i > 0 else 0
            aux[j] = aux_vector(prev, c["dtp"][i], c["dtn"][i], c["cond"])
        target = np.full(L, -100, dtype=np.int64)
        target[valid] = c["masks"][vidx] - 1
        return {"patch": _t(patch), "aux": _t(aux), "target": _t(target),
                "chart_id": c["chart_id"], "start": s}

    def __getitem__(self, idx: int) -> dict:
        if self.random:
            rng = np.random.default_rng()
            ci = int(rng.choice(len(self.charts), p=self.p))
            s = int(rng.integers(0, max(self.charts[ci]["n"] - self.L, 0) + 1))
        else:
            ci, s = self.windows[idx]
        return self._make(ci, s)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    ds = LaneSequences(root, split="train", random=True)
    x = ds[0]
    print(f"차분 {len(ds.charts)}개, 에폭당 샘플 {len(ds)}개")
    print(f"patch {tuple(x['patch'].shape)}  aux {tuple(x['aux'].shape)}  target {tuple(x['target'].shape)}")
    print(f"차분: {x['chart_id']}  시작 노트 번호: {x['start']}")
