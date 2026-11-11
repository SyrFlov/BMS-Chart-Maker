"""
[점검] 두 오디오의 시간 차이(ms) 추정

같은 곡이라도 mp3와 BMS 키음 믹스는 곡 시작 위치가 수십 ms 다를 수 있다
(mp3 인코더가 앞에 붙이는 무음, 다른 마스터 등). 그대로 채점하면
모델 오차가 아니라 이 시작 위치 차이가 측정된다.
여기서는 두 오디오의 "소리 시작 세기" 곡선(1ms 단위)을 서로 밀어 보며
가장 잘 겹치는 시간 차이를 찾는다.

사용:
    python align.py input\\Chronomia.mp3 data\\audio\\Chrono\\_chronomia_L.wav
    -> "A가 B보다 +37.0ms 늦음" 같은 결과
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import numpy as np


def onset_envelope(mono: np.ndarray, sr: int) -> np.ndarray:
    """1ms 간격의 소리 시작 세기 곡선."""
    x = np.diff(mono.astype(np.float64), prepend=0.0)        # 고음 강조(타격음이 잘 드러남)
    hop = max(1, int(round(sr / 1000)))
    n = len(x) // hop
    e = (x[:n * hop] ** 2).reshape(n, hop).mean(axis=1)       # 1ms 에너지
    k = np.ones(4) / 4
    e = np.convolve(e, k, mode="same")                        # 4ms 평활
    le = np.log10(e + 1e-10)
    flux = np.maximum(np.diff(le, prepend=le[0]), 0.0)
    flux -= flux.mean()
    return flux


def estimate_lag_ms(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int,
                    max_ms: int = 1000) -> tuple[float, float]:
    """a가 b보다 몇 ms 늦은지(+면 a가 늦음)와 신뢰도(최고점/두번째 봉우리 비)."""
    fa, fb = onset_envelope(a, sr_a), onset_envelope(b, sr_b)
    n = len(fa) + len(fb)
    size = 1 << int(np.ceil(np.log2(n)))
    corr = np.fft.irfft(np.fft.rfft(fa, size) * np.conj(np.fft.rfft(fb, size)), size)
    lags = np.arange(-max_ms, max_ms + 1)
    c = corr[lags % size]
    i = int(np.argmax(c))
    lag = float(lags[i])
    if 0 < i < len(c) - 1:                                    # 포물선 보간으로 1ms 아래까지
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            lag += 0.5 * (y0 - y2) / den
    others = np.abs(lags - lags[i]) > 15                      # 주변 15ms 밖의 두 번째 봉우리
    second = c[others].max() if others.any() else 0.0
    conf = float(c[i] / second) if second > 0 else float("inf")
    return lag, conf


def to_mono(stereo: np.ndarray) -> np.ndarray:
    return stereo.mean(axis=1) if stereo.ndim == 2 else stereo


if __name__ == "__main__":
    if len(_sys.argv) >= 3:
        from pathlib import Path
        from render_audio import read_audio
        a, sra = read_audio(Path(_sys.argv[1]))
        b, srb = read_audio(Path(_sys.argv[2]))
        lag, conf = estimate_lag_ms(to_mono(a), sra, to_mono(b), srb)
        print(f"A가 B보다 {lag:+.1f}ms {'늦음' if lag >= 0 else '빠름'} (신뢰도 {conf:.2f}, 1.2 이상이면 믿을 만함)")
    else:                                                     # 자체 테스트
        rng = np.random.default_rng(0)
        sr = 44100
        t = np.arange(int(sr * 20)) / sr
        b = 0.01 * rng.standard_normal(len(t))
        hits = np.sort(rng.uniform(0.5, 19.5, 120))
        for h in hits:                                        # 감쇠하는 타격음
            i = int(h * sr)
            L = int(0.08 * sr)
            seg = np.sin(2 * np.pi * 900 * t[:L]) * np.exp(-t[:L] * 60)
            b[i:i + L] += seg[: len(b) - i]
        for true in (37.0, -12.5, 0.0, 250.0):
            sh = int(round(true / 1000 * sr))
            a = np.roll(b, sh) * 0.7 + 0.005 * rng.standard_normal(len(b))
            lag, conf = estimate_lag_ms(a, sr, b, sr)
            print(f"실제 {true:+.1f}ms -> 추정 {lag:+.1f}ms (신뢰도 {conf:.2f})")
            assert abs(lag - true) < 1.0
        print("OK")
