"""
[모델 보조] 레인 모델의 공용 함수 (numpy만 사용, torch 불필요)

레인 모델이 하는 일:
    노트 시각 모델이 찾은 노트 시각 t_1 < t_2 < ... 마다 "어느 레인들에 노트를 찍을까"를 순서대로 정한다.
    레인은 8개(0=스크래치, 1~7=건반)이고 한 시각에 여러 개(동시치기)가 가능하므로,
    한 시각의 선택을 8비트 마스크로 보고, 레인 8개 각각에 대해 칠지 말지를 예측한다.

노트 하나를 설명하는 입력:
    - 그 시각 주변의 멜 스펙트로그램 조각 (±k프레임)
    - 직전 노트의 마스크 (8비트), 직전/직후 노트와의 간격, 곡의 목표 노트 밀도
"""
from __future__ import annotations

import numpy as np

N_LANES = 8
N_CLASSES = 255                 # 마스크 1~255 (참고용. 모델은 레인 8개를 각각 이진 예측한다)
AUX_DIM = N_LANES + 2 + 1       # 직전 마스크 8 + 간격 2 + 밀도 1
POPCOUNT = np.array([bin(m).count("1") for m in range(256)], dtype=np.int64)

_LOG_LO, _LOG_HI = np.log(0.01), np.log(2.0)
NO_NEIGHBOR_FRAMES = 200        # 이웃 노트가 없을 때의 간격 (2초)


# ---------------------------------------------------------------------------
# 레이블 <-> 마스크
# ---------------------------------------------------------------------------
def mask_bits(mask: int) -> np.ndarray:
    return np.array([(int(mask) >> b) & 1 for b in range(N_LANES)], dtype=np.float32)


def masks_from_labels(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """label (T, 8) -> (노트가 있는 프레임 번호[n], 그 프레임의 마스크[n], 1~255)."""
    rows = np.flatnonzero(label.any(axis=1))
    weights = (1 << np.arange(N_LANES)).astype(np.int64)
    masks = (label[rows].astype(np.int64) * weights).sum(axis=1)
    return rows.astype(np.int64), masks


# ---------------------------------------------------------------------------
# 입력 특징
# ---------------------------------------------------------------------------
def gap_feature(dt_frames: float, fps: int = 100) -> float:
    """간격(프레임) -> 0~1. 10ms~2초를 로그 눈금으로 펼침."""
    dt = max(dt_frames / fps, 0.01)
    return float(np.clip((np.log(dt) - _LOG_LO) / (_LOG_HI - _LOG_LO), 0.0, 1.0))


def gap_arrays(frames: np.ndarray, default: int = NO_NEIGHBOR_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """각 노트의 (직전 노트와의 간격, 직후 노트와의 간격) [프레임]."""
    n = len(frames)
    dtp = np.full(n, default, dtype=np.int64)
    dtn = np.full(n, default, dtype=np.int64)
    if n > 1:
        d = np.diff(frames)
        dtp[1:] = d
        dtn[:-1] = d
    return dtp, dtn


def cond_value(density: float) -> float:
    """곡의 노트 밀도(개/초) -> 입력값. 30개/초를 1.0으로."""
    return float(np.clip(density / 30.0, 0.0, 2.0))


def aux_vector(prev_mask: int, dt_prev: float, dt_next: float, cond: float) -> np.ndarray:
    return np.concatenate([mask_bits(prev_mask),
                           [gap_feature(dt_prev), gap_feature(dt_next), cond]]).astype(np.float32)


def extract_patches(mel: np.ndarray, frames: np.ndarray, k: int = 8) -> np.ndarray:
    """mel (T, n_mels)에서 각 노트 프레임 ±k 구간을 잘라 (N, n_mels, 2k+1)로. 범위 밖은 0."""
    T, M = mel.shape
    P = 2 * k + 1
    out = np.zeros((len(frames), M, P), dtype=np.float32)
    for j, f in enumerate(frames):
        lo, hi = int(f) - k, int(f) + k + 1
        a, b = max(lo, 0), min(hi, T)
        if b > a:
            out[j, :, a - lo:b - lo] = np.asarray(mel[a:b], dtype=np.float32).T
    return out


# ---------------------------------------------------------------------------
# 생성 (샘플링)
# ---------------------------------------------------------------------------
def sample_mask(logits: np.ndarray, temperature: float, max_chord: int, rng: np.random.Generator,
                forbid: int = 0) -> int:
    """레인 8개의 로짓 -> 마스크 하나 (레인마다 칠 확률 = sigmoid(logit / temperature))"""
    z = np.asarray(logits, dtype=np.float64)
    allowed = np.array([not (forbid >> b) & 1 for b in range(N_LANES)])
    if not allowed.any():
        allowed[:] = True
    zz = np.where(allowed, z, -np.inf)
    if temperature <= 1e-6:
        on = zz > 0
    else:
        p = 1.0 / (1.0 + np.exp(-np.clip(z / temperature, -50, 50)))
        on = (rng.random(N_LANES) < p) & allowed
    if not on.any():
        if temperature <= 1e-6:
            on[int(np.argmax(zz))] = True
        else:
            w = np.exp((zz - zz.max()) / temperature)
            on[int(rng.choice(N_LANES, p=w / w.sum()))] = True
    if on[1:].sum() > max_chord:                    # 동시치기 상한은 건반(1~7)만 셈, 스크래치(레인 0)는 따로
        keys = np.argsort(-np.where(on[1:], zz[1:], -np.inf))[:max_chord] + 1
        sc = bool(on[0])
        on = np.zeros(N_LANES, dtype=bool)
        on[keys] = True
        on[0] = sc
    return int(sum(1 << b for b in range(N_LANES) if on[b]))


def generate_masks(step_fn, frames: np.ndarray, cond: float, temperature: float = 0.8,
                   max_chord: int = 6, no_jack_ms: float = 0.0, seed: int = 0) -> np.ndarray:
    """노트 시각마다 마스크를 순서대로 생성한다"""
    rng = np.random.default_rng(seed)
    n = len(frames)
    dtp, dtn = gap_arrays(frames)
    cond_arr = np.broadcast_to(np.asarray(cond, dtype=np.float32), (n,))   # 노트마다 다른 밀도 조건 허용
    out = np.zeros(n, dtype=np.int64)
    last_used = np.full(N_LANES, -10**9, dtype=np.int64)    # 레인별 마지막으로 친 프레임
    prev, state = 0, None
    for i in range(n):
        aux = aux_vector(prev, dtp[i], dtn[i], float(cond_arr[i]))
        logits, state = step_fn(i, aux, state)
        forbid = 0
        if no_jack_ms > 0:
            recent = (frames[i] - last_used) * 10.0 < no_jack_ms
            forbid = int(sum(1 << b for b in range(N_LANES) if recent[b]))
        prev = sample_mask(logits, temperature, max_chord, rng, forbid)
        out[i] = prev
        for b in range(N_LANES):
            if (prev >> b) & 1:
                last_used[b] = frames[i]
    return out


def chord_stats(masks: np.ndarray) -> dict:
    """마스크열의 요약 통계 (생성 결과가 실제 패턴과 비슷한 성질인지 볼 때)."""
    masks = np.asarray(masks, dtype=np.int64)
    n = len(masks)
    lane_use = np.array([((masks >> b) & 1).sum() for b in range(N_LANES)], dtype=np.float64)
    return {
        "n": n,
        "mean_chord": float(POPCOUNT[masks].mean()) if n else 0.0,
        "repeat_rate": float(((masks[1:] & masks[:-1]) != 0).mean()) if n > 1 else 0.0,
        "lane_share": (lane_use / max(lane_use.sum(), 1)).round(3).tolist(),
    }
