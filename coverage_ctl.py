"""
[규칙, STEP 5] 16분 스트림에서 7건반을 고루 쓰게 (--coverage)
"""
from __future__ import annotations

import numpy as np


def stream_flags(frames: np.ndarray, max_gap_frames: float) -> np.ndarray:
    """앞뒤 간격이 모두 max_gap_frames 이하인 노트 = 스트림 줄."""
    f = np.asarray(frames, dtype=np.float64)
    if len(f) < 3:
        return np.zeros(len(f), dtype=bool)
    d = np.diff(f)
    prev_ok = np.concatenate([[False], d <= max_gap_frames])
    next_ok = np.concatenate([d <= max_gap_frames, [False]])
    return prev_ok & next_ok


class Coverage:
    def __init__(self, strength: float, idle_lo: int = 4, idle_hi: int = 8):
        self.strength = float(strength)
        self.lo, self.hi = idle_lo, idle_hi
        self.since = np.full(8, idle_hi, dtype=np.int64)       # 레인별 마지막으로 쓴 뒤 지난 줄 수

    def logit_bonus(self) -> np.ndarray:
        out = np.zeros(8, dtype=np.float32)
        if self.strength <= 0:
            return out
        x = np.clip((self.since[1:] - self.lo) / max(1, self.hi - self.lo), 0.0, 1.0)
        out[1:] = self.strength * x
        return out

    def apply(self, logits: np.ndarray, temperature: float = 1.0, forbid: int = 0) -> np.ndarray:
        """보너스를 더하되, 칠 수 있는 건반(1~7, 금지 제외)의 기대 개수가 바뀌지 않게 공통값을 뺀다."""
        lg = np.asarray(logits, dtype=np.float64).copy()
        b = self.logit_bonus().astype(np.float64)
        if not b.any():
            return lg
        keys = np.array([k for k in range(1, 8) if not (int(forbid) >> k) & 1])
        if len(keys) == 0:
            return lg
        T = max(float(temperature), 1e-3)
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z / T, -50, 50)))
        target = sig(lg[keys]).sum()
        lo, hi = -30.0, 30.0
        for _ in range(50):                                    # 이분법: 기대 개수가 같아지는 c
            c = (lo + hi) / 2
            if sig(lg[keys] + b[keys] - c).sum() > target:
                lo = c
            else:
                hi = c
        c = (lo + hi) / 2
        lg[1:] = lg[1:] + b[1:] - c
        return lg

    def update(self, mask: int, in_stream: bool) -> None:
        if not in_stream:                                       # 스트림이 끊기면 처음부터
            self.since[:] = self.hi
            return
        self.since += 1
        for b in range(8):
            if (int(mask) >> b) & 1:
                self.since[b] = 0


def completes_repeat(prev: list[int], m: int, periods=(2, 3, 4)) -> bool:
    """m을 붙이면 '같은 p줄 묶음이 두 번 연속 그대로'가 완성되는가 (예: 1357 246 1357 246, 135 24 35 135 24 35)"""
    seq = list(prev) + [int(m)]
    for p in periods:
        if len(seq) < 2 * p:
            continue
        a, b = seq[-2 * p:-p], seq[-p:]
        if a == b and len(set(a)) > 1:
            return True
    return False


def coverage_stats(masks, flags) -> float | None:
    """확인용: 연속 스트림 8줄 창 중 7건반을 다 쓴 비율 (창이 16개 미만이면 None)."""
    m = [int(x) for x in masks]
    f = np.asarray(flags, dtype=bool)
    hits = []
    for i in range(len(m) - 7):
        if f[i:i + 8].all():
            u = 0
            for x in m[i:i + 8]:
                u |= x
            hits.append(((u >> 1) & 0x7F) == 0x7F)
    return float(np.mean(hits)) if len(hits) >= 16 else None


if __name__ == "__main__":
    cv = Coverage(2.0)
    for _ in range(10):
        cv.update((1 << 3) | (1 << 5), True)
    b = cv.logit_bonus()
    assert b[3] == 0 and b[5] == 0 and b[1] == 2.0 and b[7] == 2.0 and b[0] == 0, b
    cv.update(1 << 1, False)
    assert cv.logit_bonus()[3] == 2.0
    # apply: 기대 개수는 그대로, 안 쓴 건반(1,7) 쪽으로 확률이 옮겨 감
    lg = np.array([0, -1.0, -1.0, 0.5, -1.0, 0.5, -1.0, -1.0])
    sig = lambda z: 1 / (1 + np.exp(-z))
    out = cv.apply(lg, 1.0)
    assert abs(sig(out[1:]).sum() - sig(lg[1:]).sum()) < 1e-6
    cv2 = Coverage(2.0)
    for _ in range(10):
        cv2.update((1 << 3) | (1 << 5), True)
    out2 = cv2.apply(lg, 1.0)
    assert abs(sig(out2[1:]).sum() - sig(lg[1:]).sum()) < 1e-6
    assert sig(out2[1]) > sig(lg[1]) and sig(out2[3]) < sig(lg[3]) and out2[0] == lg[0]
    o, e = 0b10101010, 0b01010100
    assert completes_repeat([o, e, o], e) and not completes_repeat([o, e, o], 0b10)
    a, b, c = 0b00101010, 0b00010100, 0b00101000
    assert completes_repeat([a, b, c, a, b], c) and not completes_repeat([a, b, c, a, b], 0b01101000)
    assert not completes_repeat([a, a, a], a)
    assert list(stream_flags(np.array([0, 7, 14, 21, 60, 67]), 8)) == [False, True, True, False, False, False]
    full = [(1 << (1 + i % 7)) for i in range(40)]
    assert coverage_stats(full, [True] * 40) == 1.0
    print("OK")
