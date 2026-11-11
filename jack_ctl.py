"""
[규칙] 소리 기반 연타 조절 (--jack-penalty)
"""
from __future__ import annotations

import numpy as np


def onset_profiles(mel: np.ndarray, frames: np.ndarray, pre: int = 3, post: int = 4) -> np.ndarray:
    """노트마다 '새로 들어온 소리'의 스펙트럼 모양 (N, n_mels), 길이 1로 정규화.

    노트 직후 post프레임 평균 - 직전 pre프레임 평균 (양수만)."""
    m = np.asarray(mel, dtype=np.float32)
    T = len(m)
    out = np.zeros((len(frames), m.shape[1]), dtype=np.float32)
    for i, f in enumerate(np.asarray(frames, dtype=np.int64)):
        a = m[max(0, f - pre):max(1, f)].mean(0)
        b = m[min(f, T - 1):min(T, f + post)].mean(0)
        d = np.maximum(b - a, 0.0)
        if d.sum() < 1e-6:                         # 증가분이 없으면 소리 자체 모양으로
            d = b - b.mean()
        d = d - d.mean()
        n = np.linalg.norm(d)
        out[i] = d / n if n > 1e-8 else 0.0
    return out


class JackControl:
    """generate 루프에서 레인별 '연타 벌점'(로짓에서 뺄 값)을 준다."""

    def __init__(self, mel, frames, penalty: float, window_frames: float,
                 lo_pct: float = 50.0, hi_pct: float = 90.0):
        self.frames = np.asarray(frames, dtype=np.int64)
        self.P = onset_profiles(mel, self.frames)
        self.penalty = float(penalty)
        self.win = float(window_frames)
        if len(self.frames) > 2:
            s = (self.P[1:] * self.P[:-1]).sum(1)
            self.lo, self.hi = np.percentile(s, lo_pct), np.percentile(s, hi_pct)
        else:
            self.lo, self.hi = 0.0, 1.0
        if self.hi - self.lo < 1e-6:
            self.hi = self.lo + 1e-6
        self.last_idx = np.full(8, -1, dtype=np.int64)    # 레인별 마지막으로 쓴 노트 번호

    def sim(self, i: int, j: int) -> float:
        return float(self.P[i] @ self.P[j])

    def logit_penalty(self, i: int) -> np.ndarray:
        """노트 i에서 레인별로 로짓에서 뺄 값 (8,)."""
        out = np.zeros(8, dtype=np.float32)
        if self.penalty <= 0:
            return out
        f = self.frames[i]
        for lane in range(1, 8):                       # 스크래치(0)는 연타가 자연스러워서 제외
            j = self.last_idx[lane]
            # 바로 앞 줄에서 친 레인만 대상 (그 사이에 다른 노트가 있으면 연타가 아니라 흐름의 일부)
            if j != i - 1 or f - self.frames[j] > self.win:
                continue
            q = (self.sim(i, j) - self.lo) / (self.hi - self.lo)   # 0=다른 소리 .. 1=같은 소리
            out[lane] = self.penalty * (1.0 - float(np.clip(q, 0.0, 1.0)))
        return out

    def update(self, i: int, mask: int) -> None:
        for lane in range(8):
            if (int(mask) >> lane) & 1:
                self.last_idx[lane] = i


if __name__ == "__main__":
    # 자체 점검: 소리 A,B,A,B... 와 A,A,A,A... 구간. 같은 소리 반복 쌍은 벌점이 작고, 다른 소리 쌍은 커야 함
    rng = np.random.default_rng(0)
    n_mels, T = 96, 4000
    mel = np.full((T, n_mels), 0.1, dtype=np.float32) + rng.normal(0, 0.005, (T, n_mels)).astype(np.float32)
    A = np.zeros(n_mels, np.float32); A[5:20] = 0.6           # 저음(킥 같은 소리)
    B = np.zeros(n_mels, np.float32); B[50:80] = 0.5          # 고음(하이햇 같은 소리)
    frames = np.arange(20, 3900, 26)
    kinds = []
    for k, f in enumerate(frames):
        s = A if (k < len(frames) // 2 and k % 2 == 0) or k >= len(frames) // 2 else B
        kinds.append("A" if s is A else "B")
        mel[f:f + 10] += s
    jc = JackControl(mel, frames, penalty=3.0, window_frames=30)
    same, diff = [], []
    for i in range(1, len(frames)):
        jc.last_idx[:] = -1
        jc.update(i - 1, 0b10)                           # 직전 노트가 레인1
        p = jc.logit_penalty(i)[1]
        (same if kinds[i] == kinds[i - 1] else diff).append(p)
    print(f"같은 소리 반복 쌍 벌점 평균 {np.mean(same):.2f} / 다른 소리 쌍 {np.mean(diff):.2f}")
    assert np.mean(same) < 1.0 and np.mean(diff) > 2.5
    print("OK")
