"""
[규칙, STEP 2] 곡의 셈여림(소리 크기)에 따라 노트 물량을 조절 (--dynamics)

소리가 큰 구간은 노트를 더 많이, 조용한 구간은 더 적게:
  1. 멜 스펙트로그램에서 프레임별 소리 크기를 구하고 2초 단위로 부드럽게 만든 뒤,
     곡 안에서 상대적인 크기(조용한 10% 지점=0, 큰 10% 지점=1)로 바꾼다
  2. 노트 임계값을 구간마다 다르게: 큰 구간은 낮추고(노트 증가) 조용한 구간은 올린다(노트 감소)
  3. 레인 모델에 주는 밀도 조건도 구간마다: 큰 구간은 동시치기를 조금 더, 조용한 구간은 덜

strength 0이면 꺼짐(지금과 같음), 1이면 기본 세기, 2면 두 배.
numpy만 사용.
"""
from __future__ import annotations

import numpy as np


def loudness_curve(mel: np.ndarray, fps: int = 100, smooth_sec: float = 2.0,
                   lo_pct: float = 10.0, hi_pct: float = 90.0) -> np.ndarray:
    """mel (T, n_mels, 0~1 스케일) -> (T,) 0~1 상대 소리 크기."""
    m = np.asarray(mel, dtype=np.float32)
    e = m.mean(axis=1)                                   # dB 스케일 평균 = 체감 크기에 가까움
    w = max(1, int(smooth_sec * fps))
    k = np.ones(w, dtype=np.float32) / w
    s = np.convolve(np.pad(e, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")
    lo, hi = np.percentile(s, lo_pct), np.percentile(s, hi_pct)
    if hi - lo < 1e-6:
        return np.full(len(s), 0.5, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def threshold_curve(base: float, loud: np.ndarray, strength: float, span: float = 0.3) -> np.ndarray:
    """큰 구간(loud=1)은 base - span/2*strength, 조용한 구간(loud=0)은 base + span/2*strength."""
    return np.clip(base + strength * span * (0.5 - loud), 0.05, 0.95).astype(np.float32)


def cond_scale(loud_at_notes: np.ndarray, strength: float, span: float = 0.5) -> np.ndarray:
    """레인 모델 밀도 조건의 배율: 큰 구간 1 + span/2*strength, 조용한 구간 1 - span/2*strength."""
    return np.clip(1.0 + strength * span * (np.asarray(loud_at_notes) - 0.5), 0.3, 2.0).astype(np.float32)


def density_loudness_corr(frames: np.ndarray, loud: np.ndarray, fps: int = 100, win_sec: float = 2.0) -> float:
    """생성된 노트의 구간 밀도와 소리 크기의 상관 (확인용). 1에 가까울수록 '크면 많고 작으면 적다'."""
    T = len(loud)
    w = int(win_sec * fps)
    if T < 2 * w or len(frames) < 4:
        return float("nan")
    counts = np.bincount(np.clip(np.asarray(frames) // w, 0, T // w), minlength=T // w + 1)[:T // w]
    lv = loud[:(T // w) * w].reshape(T // w, w).mean(1)
    if counts.std() < 1e-9 or lv.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(counts, lv)[0, 1])


if __name__ == "__main__":
    # 자체 점검: 앞 절반 조용, 뒤 절반 큼 -> 임계값은 앞이 높고, 같은 확률이면 뒤에서 노트가 더 많이 잡혀야 함
    import sys
    sys.path.insert(0, ".")
    from metrics import peak_pick
    rng = np.random.default_rng(0)
    T = 6000
    mel = np.full((T, 96), 0.3, dtype=np.float32)
    mel[T // 2:] = 0.7
    loud = loudness_curve(mel)
    assert loud[:T // 2 - 200].max() < 0.1 and loud[T // 2 + 200:].min() > 0.9
    probs = np.zeros(T, dtype=np.float32)
    probs[::12] = rng.uniform(0.3, 0.8, len(probs[::12]))
    flat = peak_pick(probs, 0.55)
    dyn = peak_pick(probs, threshold_curve(0.55, loud, 1.0))
    a, b = (flat < T // 2).sum(), (flat >= T // 2).sum()
    c, d = (dyn < T // 2).sum(), (dyn >= T // 2).sum()
    print(f"고정 임계값: 조용 {a} / 큼 {b}  ->  셈여림 반영: 조용 {c} / 큼 {d}")
    assert c < a and d > b
    print(f"상관: 고정 {density_loudness_corr(flat, loud):+.2f}, 셈여림 {density_loudness_corr(dyn, loud):+.2f}")
    print("OK")
