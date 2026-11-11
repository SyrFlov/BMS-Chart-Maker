"""
[규칙] 같은 소리는 같은 자리에 (--same-sound)

사람은 4비트, 8비트 구간에서 같은 소리(같은 키음)가 다시 나오면 같은 자리(같은 레인 모양)에 찍는다.
mp3에는 키음 번호가 없으니 jack_ctl.py처럼 노트 시점에 '새로 들어온 소리'의 스펙트럼 모양을 비교한다.

- 대상: 스트림이 아닌 줄 (앞뒤 간격이 모두 16분보다 넓은 줄 = 4비트/8비트 구간의 줄)
  스트림까지 적용하면 같은 하이햇이 반복될 때 1357/246 같은 교대가 생기므로 뺀다.
- 최근 window 안에서 소리가 가장 비슷한 줄을 찾고, 그 유사도가 곡 안에서 높은 편이면
  그 줄의 레인 모양 쪽으로 확률을 민다 (그 레인은 올리고, 나머지는 절반만큼 내림).
- 16분 연타 금지(--no-jack-ms)는 그대로 먼저 지켜진다 (금지된 레인은 올려도 못 씀).
numpy만 사용.
"""
from __future__ import annotations

import numpy as np

from jack_ctl import onset_profiles


class SoundLock:
    def __init__(self, mel, frames, strength: float, window_frames: float, stream_frames: float,
                 lo_pct: float = 50.0, hi_pct: float = 85.0):
        self.frames = np.asarray(frames, dtype=np.int64)
        n = len(self.frames)
        self.P = onset_profiles(mel, self.frames)
        self.s = float(strength)
        self.win = float(window_frames)
        g = np.diff(self.frames)
        gp = np.r_[np.inf, g]
        gn = np.r_[g, np.inf]
        self.target = (gp > stream_frames) & (gn > stream_frames)     # 스트림이 아닌 줄
        self.min_gap = float(stream_frames)
        # 후보 쌍의 유사도 분포로 곡 안의 상대 기준을 잡는다
        sims = []
        self.cand: list[np.ndarray] = []
        for i in range(n):
            lo = np.searchsorted(self.frames, self.frames[i] - self.win, side="left")
            js = np.arange(lo, i)
            js = js[self.frames[i] - self.frames[js] >= self.min_gap]
            self.cand.append(js)
            if self.target[i] and len(js):
                sims.append(self.P[js] @ self.P[i])
        allv = np.concatenate(sims) if sims else np.array([0.0, 1.0])
        self.lo, self.hi = np.percentile(allv, lo_pct), np.percentile(allv, hi_pct)
        if self.hi - self.lo < 1e-6:
            self.hi = self.lo + 1e-6
        self.n_target = int(self.target.sum())
        self.n_locked = 0
        self.n_copied = 0
        self._last = None

    def bias(self, i: int, out: np.ndarray) -> np.ndarray:
        """노트 i에서 로짓에 더할 값 (8,). out = 지금까지 만든 마스크."""
        add = np.zeros(8, dtype=np.float32)
        self._last = None
        if self.s <= 0 or not self.target[i] or len(self.cand[i]) == 0:
            return add
        js = self.cand[i]
        sv = self.P[js] @ self.P[i]
        k = int(np.argmax(sv))
        q = float(np.clip((sv[k] - self.lo) / (self.hi - self.lo), 0.0, 1.0))
        if q <= 0:
            return add
        m = int(out[js[k]]) & 0xFE                     # 건반 모양만 따라 한다 (스크래치는 모델에 맡김)
        if m == 0:                                      # 스크래치만 있던 줄은 따라 하지 않음 (도입부 스크래치 연발 방지)
            return add
        on = np.array([(m >> b) & 1 for b in range(8)], dtype=bool)
        add[on] += self.s * q
        add[~on] -= 0.5 * self.s * q
        add[0] = 0.0
        if q >= 0.5:
            self.n_locked += 1
            self._last = m
        return add

    def update(self, m: int) -> None:
        if self._last is not None and (int(m) & 0xFE) == self._last:
            self.n_copied += 1

    def summary(self) -> str:
        return (f"같은 소리 같은 자리: 대상 줄(4·8비트 구간) {self.n_target}개 중 비슷한 소리를 찾은 줄 {self.n_locked}개, "
                f"그중 같은 모양으로 찍은 줄 {self.n_copied}개")


if __name__ == "__main__":
    # 자체 점검: 8분 간격으로 A,B,A,B... 소리. 강하게 걸면 A는 A끼리, B는 B끼리 같은 모양이 되어야 함
    rng = np.random.default_rng(0)
    n_mels, T = 96, 3000
    mel = np.full((T, n_mels), 0.1, np.float32) + rng.normal(0, 0.005, (T, n_mels)).astype(np.float32)
    A = np.zeros(n_mels, np.float32); A[5:20] = 0.6
    B = np.zeros(n_mels, np.float32); B[50:80] = 0.5
    frames = np.arange(20, 2900, 26)
    kinds = [k % 2 for k in range(len(frames))]
    for k, f in enumerate(frames):
        mel[f:f + 10] += A if kinds[k] == 0 else B
    sl = SoundLock(mel, frames, strength=6.0, window_frames=120, stream_frames=17)
    out = np.zeros(len(frames), np.int64)
    for i in range(len(frames)):
        lg = rng.normal(0, 1, 8).astype(np.float32) + sl.bias(i, out)
        m = 0
        for b in np.argsort(-lg)[:2]:
            m |= 1 << int(b)
        out[i] = m
        sl.update(m)
    same = np.mean([out[i] == out[i - 2] for i in range(4, len(frames))])
    print(sl.summary(), f"| 두 줄 전(같은 소리)과 같은 모양 {same:.0%}")
    assert same > 0.8
    print("OK")
