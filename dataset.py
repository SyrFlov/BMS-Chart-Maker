"""
[학습] build_dataset.py가 만든 결과에서 20초 구간(윈도우)을 뽑아 주는 Dataset
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:          # torch가 없어도 numpy로 동작 (테스트용)
    torch = None
    Dataset = object


def _tensor(a: np.ndarray):
    return torch.from_numpy(a) if torch is not None else a


def crop_window(mel: np.ndarray, label: np.ndarray, start: int, n_frames: int):
    """mel[T, M], label[T, L] -> 길이 n_frames로 자르고 모자라면 0으로 채움. (M, W), (L, W) 반환."""
    m = np.asarray(mel[start:start + n_frames], dtype=np.float32)
    l = np.asarray(label[start:start + n_frames], dtype=np.float32)
    if len(m) < n_frames:
        pad = n_frames - len(m)
        m = np.pad(m, ((0, pad), (0, 0)))
        l = np.pad(l, ((0, pad), (0, 0)))
    return m.T.copy(), l.T.copy()


class ChartWindows(Dataset):
    def __init__(self, root: str | Path, split: str = "train", window_sec: float = 20.0,
                 random: bool = True, epoch_size: int | None = None,
                 chart_ids: list[str] | None = None, good_patterns: list[str] | None = None,
                 good_weight: float = 1.0):
        self.root = Path(root)
        self.cfg = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.fps = self.cfg["fps"]
        self.W = int(round(window_sec * self.fps))
        self.random = random

        with open(self.root / "index.csv", newline="", encoding="utf-8-sig") as fp:
            self.rows = [r for r in csv.DictReader(fp) if r["split"] == split]
        if chart_ids is not None:                       # 특정 패턴만 사용 (과적합 테스트용)
            self.rows = [r for r in self.rows if r["chart_id"] in set(chart_ids)]
        if not self.rows:
            raise ValueError(f"split='{split}'에 해당하는 차분이 없어요 (곡 수가 적으면 --val-ratio 조정)")

        self.frames = np.array([int(r["frames"]) for r in self.rows])
        self.density = np.array([float(r["density"]) for r in self.rows], dtype=np.float32)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # memmap 캐시

        if random:
            w = np.ones(len(self.rows))
            if good_patterns is not None and good_weight != 1.0:     # v3: 좋은 패턴을 더 자주 뽑기
                from good_charts import report
                good = report(self.rows, good_patterns, split)
                w = np.array([good_weight if r["chart_id"] in good else 1.0 for r in self.rows])
            self.p = self.frames * w / (self.frames * w).sum()
            # 에폭 크기 기본값: 전체 데이터를 대략 한 번 훑는 윈도우 수
            self.n = epoch_size or max(1, int(self.frames.sum() // self.W))
        else:
            # 곡 끝까지 빠짐없이 훑는다 (마지막 윈도우는 0으로 패딩)
            self.windows = [(i, s) for i, t in enumerate(self.frames)
                            for s in range(0, int(t), self.W)]
            self.n = len(self.windows)

    def __len__(self) -> int:
        return self.n

    def _load(self, i: int):
        if i not in self._cache:
            cid = self.rows[i]["chart_id"]
            self._cache[i] = (
                np.load(self.root / f"{cid}.mel.npy", mmap_mode="r"),
                np.load(self.root / f"{cid}.label.npy", mmap_mode="r"),
            )
        return self._cache[i]

    def __getitem__(self, idx: int) -> dict:
        if self.random:
            rng = np.random.default_rng()
            i = int(rng.choice(len(self.rows), p=self.p))
            T = int(self.frames[i])
            start = int(rng.integers(0, max(T - self.W, 0) + 1))
        else:
            i, start = self.windows[idx]

        mel, label = self._load(i)
        m, l = crop_window(mel, label, start, self.W)
        return {
            "mel": _tensor(m),                                    # (n_mels, W)
            "label": _tensor(l),                                  # (8, W)
            "density": _tensor(np.array([self.density[i]], dtype=np.float32)),
            "chart_id": self.rows[i]["chart_id"],
            "start_frame": start,
        }


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    ds = ChartWindows(root, split="train", window_sec=20, random=True)
    x = ds[0]
    print(f"윈도우 수(에폭): {len(ds)}  mel {tuple(x['mel'].shape)}  label {tuple(x['label'].shape)}")
    print(f"차분: {x['chart_id']}  시작 프레임: {x['start_frame']}  이 윈도우의 노트 수: {int(x['label'].sum())}")
