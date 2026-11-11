"""
[모델, STEP 4] 레인 모델 v2: 구간 단위로 패턴 흐름이 이어지게
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import csv
import json
from pathlib import Path

import numpy as np

from plan_feats import PLAN_DIM, alt_run_ratio, half_groups, is_alt, plan_features
from lane_utils import (N_LANES, NO_NEIGHBOR_FRAMES, POPCOUNT, cond_value, extract_patches, gap_arrays,
                        gap_feature, masks_from_labels, sample_mask)

K_PREV = 4                                  # 기억하는 직전 마스크 수
HIST_DIM = K_PREV * N_LANES + N_LANES       # 직전 마스크 4개(32) + 레인별 마지막 사용 후 경과(8)
NOTE_DIM = 4                                # 직전 간격, 직후 간격, 곡 밀도(v3: 좋은 패턴 표시), 구간 밀도
SEC_DIM = 4                                 # v3 구간 특징: 짧은/긴 범위 밀도, 쉼 뒤 경과, 다음 쉼까지
RHY_DIM = 9                                 # 박자 위치: 박 안 16분 위치(4) + 격자 밖 표시(1) + 마디 안 몇째 박(4)
ACC_DIM = 4                                 # v5 레인: 대역별(저/중저/중고/고) 어택 세기, 주변 0.35초 대비 상대값
NEVER = -10**9

_LOG_LO, _LOG_HI = np.log(0.01), np.log(2.0)


def _gap_vec(dt_frames: np.ndarray, fps: int = 100) -> np.ndarray:
    """gap_feature의 배열 버전 (값이 정확히 같아야 함)."""
    dt = np.maximum(np.asarray(dt_frames, dtype=np.float64) / fps, 0.01)
    return np.clip((np.log(dt) - _LOG_LO) / (_LOG_HI - _LOG_LO), 0.0, 1.0)


def _bits(masks: np.ndarray) -> np.ndarray:
    m = np.asarray(masks, dtype=np.int64)
    return ((m[..., None] >> np.arange(N_LANES)) & 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 노트 특징 (생성 결과와 무관, 미리 계산 가능)
# ---------------------------------------------------------------------------
def local_density(frames: np.ndarray, radius: int = 100) -> np.ndarray:
    """각 노트 앞뒤 radius 프레임(기본 ±1초) 안의 노트 시각 수 / 40, 0~1.5."""
    f = np.asarray(frames, dtype=np.int64)
    lo = np.searchsorted(f, f - radius, side="left")
    hi = np.searchsorted(f, f + radius, side="right")
    return np.clip((hi - lo) / 40.0, 0.0, 1.5).astype(np.float32)


REST_FRAMES = 60                            # 0.6초 이상 노트가 없으면 쉬는 구간으로 본다


def section_features(frames: np.ndarray) -> np.ndarray:
    """(n, SEC_DIM) 곡 구조 특징"""
    f = np.asarray(frames, dtype=np.int64)
    n = len(f)
    out = np.zeros((n, SEC_DIM), dtype=np.float32)
    if n == 0:
        return out
    for j, (r, d) in enumerate(((50, 20.0), (400, 160.0))):
        lo = np.searchsorted(f, f - r, side="left")
        hi = np.searchsorted(f, f + r, side="right")
        out[:, j] = np.clip((hi - lo) / d, 0.0, 1.5)
    gap_before = np.r_[True, np.diff(f) >= REST_FRAMES]      # 이 노트 앞이 쉼
    gap_after = np.r_[np.diff(f) >= REST_FRAMES, True]       # 이 노트 뒤가 쉼
    start = f[np.maximum.accumulate(np.where(gap_before, np.arange(n), 0))]
    end_idx = np.minimum.accumulate(np.where(gap_after, np.arange(n), n - 1)[::-1])[::-1]
    out[:, 2] = np.clip((f - start) / 800.0, 0.0, 1.0)
    out[:, 3] = np.clip((f[end_idx] - f) / 800.0, 0.0, 1.0)
    return out


def nf_layout(note_dim: int) -> tuple[bool, bool]:
    """입력 차원으로 모델 종류를 안다: 4 = v2, 13 = v2r(박자), 8 = 구간만, 17 = v3(박자+구간), 24 = v4(+계획),
    28 = v5(+대역별 어택)."""
    d = note_dim - NOTE_DIM
    full = (RHY_DIM + SEC_DIM + PLAN_DIM, RHY_DIM + SEC_DIM + PLAN_DIM + ACC_DIM)
    has_rhy = d in (RHY_DIM, RHY_DIM + SEC_DIM) + full
    has_sec = d in (SEC_DIM, RHY_DIM + SEC_DIM) + full
    return has_rhy, has_sec


def plan_in(note_dim: int) -> bool:
    """v4/v5: 반 마디 패턴 계획(plan_feats.py)을 입력으로 받는 모델인가"""
    return note_dim - NOTE_DIM in (RHY_DIM + SEC_DIM + PLAN_DIM, RHY_DIM + SEC_DIM + PLAN_DIM + ACC_DIM)


def acc_in(note_dim: int) -> bool:
    """v5: 대역별 어택 세기(accent_features)를 입력으로 받는 모델인가"""
    return note_dim - NOTE_DIM == RHY_DIM + SEC_DIM + PLAN_DIM + ACC_DIM


def _mel_centers(n_mels: int, fmin: float = 30.0, fmax: float = 16000.0) -> np.ndarray:
    h2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    m = np.linspace(h2m(fmin), h2m(fmax), n_mels + 2)[1:-1]
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


ACC_BANDS = (0.0, 200.0, 1000.0, 4000.0, 1e9)      # 저음(킥, 베이스) / 중저 / 중고 / 고음(하이햇)


def _moving(x: np.ndarray, r: int) -> np.ndarray:
    """(T, C) 열마다 ±r 이동 평균"""
    T = len(x)
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    lo = np.clip(np.arange(T) - r, 0, T)
    hi = np.clip(np.arange(T) + r + 1, 0, T)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)[:, None]


def accent_features(mel: np.ndarray, frames: np.ndarray, peak: int = 2, radius: int = 35) -> np.ndarray:
    """(n, ACC_DIM) 노트마다 대역별 어택 세기의 상대값 (-1~1)"""
    m = np.asarray(mel, dtype=np.float32)
    T = len(m)
    f = np.clip(np.asarray(frames, dtype=np.int64), 0, max(T - 1, 0))
    if T < 3 or len(f) == 0:
        return np.zeros((len(f), ACC_DIM), np.float32)
    hz = _mel_centers(m.shape[1])
    d = np.zeros_like(m)
    d[1:] = np.maximum(m[1:] - m[:-1], 0.0)
    fl = np.stack([d[:, (hz >= ACC_BANDS[b]) & (hz < ACC_BANDS[b + 1])].sum(1) for b in range(ACC_DIM)], 1)
    pk = fl.copy()
    for s_ in range(1, peak + 1):
        pk[s_:] = np.maximum(pk[s_:], fl[:-s_])
        pk[:-s_] = np.maximum(pk[:-s_], fl[s_:])
    mu = _moving(pk, radius)
    sd = np.sqrt(np.maximum(_moving(pk * pk, radius) - mu * mu, 0.0))
    z = (pk - mu) / (sd + 1e-3 + 0.05 * mu)
    return (np.clip(z[f], -3.0, 3.0) / 3.0).astype(np.float32)


def note_features(frames: np.ndarray, cond: float) -> np.ndarray:
    """(n, NOTE_DIM)"""
    dtp, dtn = gap_arrays(np.asarray(frames, dtype=np.int64))
    out = np.zeros((len(frames), NOTE_DIM), dtype=np.float32)
    out[:, 0] = _gap_vec(dtp)
    out[:, 1] = _gap_vec(dtn)
    out[:, 2] = cond
    out[:, 3] = local_density(frames)
    return out


# ---------------------------------------------------------------------------
# 박자 위치 특징 (v2 + 박자): 사람은 마디 첫 박에 스크래치/큰 동시치기를 두는 경향이 뚜렷함
#   사람 ☆패턴 11개: 스크래치의 34~41%가 마디 첫 박 (v2 생성은 13%)
# 코드: 0~15 = (마디 안 박 번호 0~3) * 4 + (박 안 16분 위치 0~3), 16~19 = 16분 격자 밖(셋잇단/32분 등) + 박 번호,
#       -1 = 모름
# ---------------------------------------------------------------------------
def rhythm_code(mbeat: float) -> int:
    """마디 안 박 위치(0 = 마디 첫 박) -> 코드."""
    if mbeat is None or mbeat < 0:
        return -1
    q = mbeat * 4.0
    qi = int(np.floor(q + 0.5))
    if abs(q - qi) < 0.15:
        return int(qi % 16)
    return 16 + int(np.floor(mbeat + 1e-6)) % 4


def rhythm_codes_from_times(t_chart: np.ndarray, bpm: float, downbeat: int = 0) -> np.ndarray:
    """생성용: 박이 시각 0의 배수에 오도록 맞춘 시각(t_chart)과 BPM, 마디 첫 박 위치(0~3) -> 코드."""
    beats = np.asarray(t_chart, dtype=np.float64) * bpm / 60.0
    mb = (beats - downbeat) % 4.0
    return np.array([rhythm_code(x) for x in mb], dtype=np.int64)


# ★20 패턴 13개: 앞 1박 안에 스크래치가 없을 때(처음 놓는 스크래치)와 있을 때(이어지는 스크래치)를 나눠 잼.
#   -> 예전 값(2, 4박 0.63)은 첫 박 스크래치 뒤에 이어 친 것이 섞인 값. 처음 놓는 스크래치는 마디 첫 박에 훨씬 몰림.
_SCR_LEAD = (1.0, 0.32, 0.24, 0.46, 0.04, 0.01, 0.06)     # 첫 박, 2박, 3박, 4박, 8분 뒷박, 16분, 그 밖
_SCR_FOLLOW = (1.0, 0.38, 0.70, 0.91, 0.28, 0.18, 0.21)


def scratch_weight(codes: np.ndarray, follow: bool = False) -> np.ndarray:
    """스크래치 자리 가중치 (--scratch-pos): '그 자리 줄 중 스크래치 비율'의 마디 첫 박 대비 값"""
    v = _SCR_FOLLOW if follow else _SCR_LEAD
    c = np.asarray(codes, dtype=np.int64)
    w = np.full(len(c), v[6])
    on = (c >= 0) & (c < 16)
    cc = np.where(on, c, 1)
    w = np.where(on & (cc % 2 == 1), v[5], w)
    w = np.where(on & (cc % 4 == 2), v[4], w)
    w = np.where(on & (cc == 4), v[1], w)
    w = np.where(on & (cc == 8), v[2], w)
    w = np.where(on & (cc == 12), v[3], w)
    w = np.where(on & (cc == 0), v[0], w)
    return np.where(c < 0, 1.0, w)

def rhythm_features(codes: np.ndarray) -> np.ndarray:
    """코드 (n,) -> (n, RHY_DIM)"""
    c = np.asarray(codes, dtype=np.int64)
    out = np.zeros((len(c), RHY_DIM), dtype=np.float32)
    on = (c >= 0) & (c < 16)
    off = c >= 16
    out[np.flatnonzero(on), c[on] % 4] = 1.0
    out[off, 4] = 1.0
    beat = np.where(on, c // 4, np.where(off, c - 16, -1))
    k = beat >= 0
    out[np.flatnonzero(k), 5 + beat[k]] = 1.0
    return out


# ---------------------------------------------------------------------------
# 이력 특징 (지금까지 만든 마스크에 따라 달라짐)
# ---------------------------------------------------------------------------
def history_features(masks: np.ndarray, frames: np.ndarray, init_prev: np.ndarray,
                     init_last: np.ndarray) -> np.ndarray:
    """구간 전체의 이력 특징을 한 번에 계산"""
    masks = np.asarray(masks, dtype=np.int64)
    frames = np.asarray(frames, dtype=np.int64)
    L = len(masks)
    out = np.zeros((L, HIST_DIM), dtype=np.float32)
    ext = np.concatenate([np.asarray(init_prev, dtype=np.int64), masks])
    for k in range(1, K_PREV + 1):
        out[:, (k - 1) * N_LANES:k * N_LANES] = _bits(ext[K_PREV - k:K_PREV - k + L])
    b = _bits(masks).astype(bool)                                   # (L, 8)
    used = np.where(b, frames[:, None], NEVER)
    cum = np.maximum.accumulate(used, axis=0)
    before = np.vstack([np.full((1, N_LANES), NEVER), cum[:-1]]) if L else cum
    last = np.maximum(before, np.asarray(init_last, dtype=np.int64)[None, :])
    dt = np.where(last <= NEVER // 2, NO_NEIGHBOR_FRAMES, frames[:, None] - last)
    out[:, K_PREV * N_LANES:] = _gap_vec(dt)
    return out


class HistoryState:
    """생성용: 노트를 하나씩 만들며 이력 특징을 갱신 (history_features와 같은 값)."""

    def __init__(self):
        self.prev = [0] * K_PREV
        self.last = np.full(N_LANES, NEVER, dtype=np.int64)

    def features(self, frame: int) -> np.ndarray:
        out = np.zeros(HIST_DIM, dtype=np.float32)
        for k in range(1, K_PREV + 1):
            out[(k - 1) * N_LANES:k * N_LANES] = _bits(self.prev[-k])
        dt = np.where(self.last <= NEVER // 2, NO_NEIGHBOR_FRAMES, int(frame) - self.last)
        out[K_PREV * N_LANES:] = _gap_vec(dt)
        return out

    def update(self, mask: int, frame: int) -> None:
        self.prev = self.prev[1:] + [int(mask)]
        for b in range(N_LANES):
            if (int(mask) >> b) & 1:
                self.last[b] = int(frame)


def init_state_from(masks: np.ndarray, frames: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """곡의 s번째 노트 앞까지를 보고 (init_prev, init_last)를 만든다."""
    init_prev = np.zeros(K_PREV, dtype=np.int64)
    p = masks[max(0, s - K_PREV):s]
    if len(p):
        init_prev[-len(p):] = p
    init_last = np.full(N_LANES, NEVER, dtype=np.int64)
    if s > 0:
        b = _bits(masks[:s]).astype(bool)
        for lane in range(N_LANES):
            nz = np.flatnonzero(b[:, lane])
            if len(nz):
                init_last[lane] = frames[nz[-1]]
    return init_prev, init_last


# ---------------------------------------------------------------------------
# 흐름 지표: 구간 스타일이 이어지는가 (사람 패턴 0.2~0.55, v1 생성 ~0.0)
# ---------------------------------------------------------------------------
def flow_consistency(masks: np.ndarray, block: int = 16) -> dict:
    """노트 block개씩 묶어 '손 이동량'과 '위치 흩어짐'을 재고, 이웃 묶음끼리의 상관을 본다."""
    m = np.asarray(masks, dtype=np.int64)
    cen, last = [], 4.0
    for x in m:
        keys = [b for b in range(1, N_LANES) if (x >> b) & 1]
        if keys:
            last = float(np.mean(keys))
        cen.append(last)
    cen = np.array(cen)
    mv = np.abs(np.diff(cen, prepend=cen[:1]))
    nb = len(m) // block
    if nb < 4:
        return {"move": float("nan"), "spread": float("nan")}
    mvb = mv[:nb * block].reshape(nb, block).mean(1)
    spb = cen[:nb * block].reshape(nb, block).std(1)

    def ac(x):
        if x.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return {"move": ac(mvb), "spread": ac(spb)}


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------
class LaneData:
    """build_dataset.py 결과에서 곡 단위 데이터를 읽어 둔다. 구간 샘플은 sample()/windows()로."""

    def __init__(self, root: str | Path, split: str, patch_k: int = 8, chart_ids=None,
                 rhythm_dir: str | Path | None = None, good_patterns: list[str] | None = None,
                 good_weight: float = 1.0, section: bool = False, plan: bool = False,
                 alt_max: float | None = None, alt_weight: float = 0.2, accent: bool = False):
        self.root = Path(root)
        self.plan = plan
        if accent and not plan:
            raise ValueError("v5 어택 입력(--accent)은 --plan과 함께 써야 해요")
        if plan and (rhythm_dir is None or not section):
            raise ValueError("v4 계획 입력(--plan)은 --rhythm-dir와 --section이 함께 있어야 해요")
        self.section = section
        self.quality = good_patterns is not None   # v3: 곡 밀도 대신 '좋은 패턴' 표시(1/0)를 넣는다
        self.rhythm = rhythm_dir is not None
        rdir = Path(rhythm_dir) if rhythm_dir else None
        n_missing = 0
        self.cfg = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.k = patch_k
        with open(self.root / "index.csv", newline="", encoding="utf-8-sig") as fp:
            rows = [r for r in csv.DictReader(fp) if r["split"] == split]
        if chart_ids is not None:
            rows = [r for r in rows if r["chart_id"] in set(chart_ids)]
        good = set()
        if self.quality:
            from good_charts import report
            good = report(rows, good_patterns, split)
        self.charts = []
        for r in rows:
            cid = r["chart_id"]
            frames, masks = masks_from_labels(np.load(self.root / f"{cid}.label.npy"))
            if len(frames) < 2:
                continue
            is_good = cid in good
            alt_r = alt_run_ratio(masks, frames)
            alt_heavy = alt_max is not None and alt_r > alt_max
            if alt_heavy:
                is_good = False                         # 교대가 많은 패턴은 좋은 패턴으로 치지 않음
            cond = (1.0 if is_good else 0.0) if self.quality else cond_value(float(r["density"]))
            nf = note_features(frames, cond)
            codes = None
            if self.rhythm:
                rp = rdir / f"{cid}.rhy.npy"
                if not rp.exists():
                    n_missing += 1
                    continue
                per_frame = np.load(rp)
                codes = per_frame[np.clip(frames, 0, len(per_frame) - 1)].astype(np.int64)
                nf = np.concatenate([nf, rhythm_features(codes)], axis=1)
            if section:
                nf = np.concatenate([nf, section_features(frames)], axis=1)
            nf_base, groups, plan_n, plan_g = nf, None, None, None
            if plan:
                groups = half_groups(frames, codes)
                plan_n, plan_g = plan_features(masks, groups)
                nf = np.concatenate([nf, plan_n], axis=1)
            mel_mm = np.load(self.root / f"{cid}.mel.npy", mmap_mode="r")
            if accent:                                     # v5: 대역별 어택 (계획 모델은 nf_base만 씀)
                nf = np.concatenate([nf, accent_features(np.asarray(mel_mm), frames)], axis=1)
            self.charts.append({"chart_id": cid, "frames": frames, "masks": masks, "cond": cond, "good": is_good, "alt": alt_r, "alt_heavy": alt_heavy,
                                "nf_base": nf_base, "groups": groups, "plan_n": plan_n, "plan_g": plan_g,
                                "nf": nf, "codes": codes,
                                "mel": mel_mm, "n": len(frames)})
        if n_missing:
            print(f"[{split}] 박자 파일(.rhy.npy)이 없는 차분 {n_missing}개는 뺐어요")
        if not self.charts:
            raise ValueError(f"split='{split}'에 해당하는 차분이 없어요")
        self.nf_dim = self.charts[0]["nf"].shape[1]
        sizes = np.array([c["n"] for c in self.charts], dtype=np.float64)
        w = np.array([(good_weight if c["good"] else 1.0) * (alt_weight if c["alt_heavy"] else 1.0)
                      for c in self.charts])
        if alt_max is not None:
            heavy = [c for c in self.charts if c["alt_heavy"]]
            print(f"[{split}] 묶음 교대가 많은 차분(교대 속 줄 > {alt_max:.0%}) {len(heavy)}개 -> 뽑는 빈도 {alt_weight:g}배"
                  + (f" | 예: {', '.join(c['chart_id'][:20] for c in heavy[:5])}" if heavy else ""))
        self.p = sizes * w / (sizes * w).sum()
        if self.quality:
            share = self.p[[c["good"] for c in self.charts]].sum()
            print(f"[{split}] 학습 샘플 중 좋은 차분 비율 {share:.0%} (가중치 {good_weight:g}배)")

    def make(self, ci: int, s: int, L: int) -> dict:
        c = self.charts[ci]
        n = c["n"]
        idx = np.arange(s, s + L)
        valid = idx < n
        vidx = idx[valid]
        M = c["mel"].shape[1]
        patch = np.zeros((L, M, 2 * self.k + 1), dtype=np.float32)
        patch[valid] = extract_patches(c["mel"], c["frames"][vidx], self.k)
        nf = np.zeros((L, c["nf"].shape[1]), dtype=np.float32)
        nf[valid] = c["nf"][vidx]
        masks = np.zeros(L, dtype=np.int64)
        masks[valid] = c["masks"][vidx]
        frames = np.full(L, c["frames"][vidx[-1]] if len(vidx) else 0, dtype=np.int64)
        frames[valid] = c["frames"][vidx]
        init_prev, init_last = init_state_from(c["masks"], c["frames"], s)
        return {"patch": patch, "nf": nf, "masks": masks, "frames": frames, "valid": valid,
                "init_prev": init_prev, "init_last": init_last}

    def sample(self, rng: np.random.Generator, L: int) -> dict:
        ci = int(rng.choice(len(self.charts), p=self.p))
        s = int(rng.integers(0, max(self.charts[ci]["n"] - L, 0) + 1))
        return self.make(ci, s, L)

    def windows(self, L: int):
        for ci, c in enumerate(self.charts):
            for s in range(0, c["n"], L):
                yield self.make(ci, s, L)


def collate(items: list[dict]) -> dict:
    return {k: np.stack([it[k] for it in items]) for k in items[0]}


def batch_history(masks: np.ndarray, frames: np.ndarray, init_prev: np.ndarray, init_last: np.ndarray) -> np.ndarray:
    """(B, L) -> (B, L, HIST_DIM)"""
    return np.stack([history_features(masks[i], frames[i], init_prev[i], init_last[i])
                     for i in range(len(masks))])


# ---------------------------------------------------------------------------
# 모델 (torch)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:      # numpy 부분만 쓸 때
    torch = None

if torch is not None:
    class LaneNetV2(nn.Module):
        def __init__(self, n_mels: int = 96, patch_w: int = 17, hidden: int = 192, layers: int = 2,
                     dropout: float = 0.2, note_dim: int = NOTE_DIM):
            super().__init__()
            self.note_dim = note_dim
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d((4, 1)),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d((4, 2)),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d((3, 2)),
            )
            flat = 64 * (n_mels // 4 // 4 // 3) * (patch_w // 2 // 2)
            self.enc_proj = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(inplace=True))
            self.note_proj = nn.Linear(note_dim, hidden)
            self.ctx = nn.GRU(hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.hist_proj = nn.Linear(HIST_DIM, hidden)
            self.drop = nn.Dropout(dropout)
            self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, N_LANES)

        def encode(self, patches):
            """(N, M, P) -> (N, H)"""
            return self.enc_proj(self.cnn(patches.unsqueeze(1)).flatten(1))

        def context(self, enc, nf):
            """(B, L, H), (B, L, NOTE_DIM) -> (B, L, H) 소리만 보는 앞뒤 문맥"""
            y, _ = self.ctx(F.relu(enc + self.note_proj(nf)))
            return y

        def decode(self, ctx, hist, state=None):
            x = self.drop(F.relu(ctx + self.hist_proj(hist)))
            y, state = self.rnn(x, state)
            return self.head(y), state

        def forward(self, patches, nf, hist):
            """학습용. patches (B, L, M, P), nf (B, L, NOTE_DIM), hist (B, L, HIST_DIM) -> 로짓 (B, L, 8)"""
            B, L = patches.shape[:2]
            enc = self.encode(patches.reshape(B * L, *patches.shape[2:])).view(B, L, -1)
            return self.decode(self.context(enc, nf), hist)[0]

    @torch.no_grad()
    def song_context(model, patches: np.ndarray, nf: np.ndarray, device: str, win: int = 256,
                     chunk: int = 512) -> "torch.Tensor":
        """곡 전체의 문맥 (n, H). 학습 때처럼 win개 구간 단위로 계산하고, 각 노트는 구간 가운데에 가까운 결과를 쓴다."""
        n = len(patches)
        encs = [model.encode(torch.from_numpy(patches[i:i + chunk]).to(device)) for i in range(0, n, chunk)]
        enc = torch.cat(encs, 0)
        nft = torch.from_numpy(nf).to(device)
        if n <= win:
            return model.context(enc[None], nft[None])[0]
        hop = win // 2
        starts = list(range(0, n - win + 1, hop))
        if starts[-1] != n - win:
            starts.append(n - win)
        out = torch.zeros(n, enc.shape[1], device=device)
        best = torch.full((n,), 1e9, device=device)
        for s in starts:
            y = model.context(enc[s:s + win][None], nft[s:s + win][None])[0]
            pos = torch.arange(s, s + win, device=device)
            d = (pos - (s + (win - 1) / 2)).abs()
            take = d < best[s:s + win]
            out[s:s + win][take] = y[take]
            best[s:s + win] = torch.where(take, d, best[s:s + win])
        return out

    @torch.no_grad()
    def generate_v2(model, mel: np.ndarray, frames: np.ndarray, cond: float, patch_k: int, device: str,
                    temperature: float = 1.0, max_chord: int = 6, no_jack_ms: float = 0.0, seed: int = 0,
                    win: int = 256, jack_penalty: float = 0.0, jack_window_ms: float = 0.0,
                    scratch_bias: float = 0.0, rhythm_codes: np.ndarray | None = None,
                    coverage: float = 0.0, stream_ms: float = 0.0, vary: float = 0.0,
                    same_sound: float = 0.0, same_window_ms: float = 0.0,
                    lone_scratch: bool = False, plan_feats: np.ndarray | None = None,
                    max_alt_rows: int = 0, scratch_pos: float = 0.0,
                    scratch_codes: np.ndarray | None = None,
                    scratch_beats: np.ndarray | None = None, row_seed: bool = False) -> np.ndarray:
        """row_seed: 줄마다 그 줄 프레임으로 정한 난수를 써서, 한 곳을 고쳐도 나머지 배치가 다시 뽑히지 않게 한다"""
        model.eval()
        frames = np.asarray(frames, dtype=np.int64)
        patches = extract_patches(mel, frames, patch_k)
        nf = note_features(frames, cond)
        has_rhy, has_sec = nf_layout(getattr(model, "note_dim", NOTE_DIM))
        if has_rhy:                                                   # 박자 위치를 쓰는 모델
            codes = rhythm_codes if rhythm_codes is not None else np.full(len(frames), -1)
            nf = np.concatenate([nf, rhythm_features(codes)], axis=1)
        if has_sec:                                                   # v3: 곡 구조(쉼/밀도) 특징
            nf = np.concatenate([nf, section_features(frames)], axis=1)
        if plan_in(getattr(model, "note_dim", NOTE_DIM)):             # v4: 반 마디 패턴 계획
            if plan_feats is None:
                print("[주의] v4 레인 모델인데 계획이 없어요 (--plan-ckpt). 계획 없이(0) 생성해요.")
                plan_feats = np.zeros((len(frames), PLAN_DIM), np.float32)
            nf = np.concatenate([nf, np.asarray(plan_feats, np.float32)], axis=1)
        if acc_in(getattr(model, "note_dim", NOTE_DIM)):              # v5: 대역별 어택 세기
            nf = np.concatenate([nf, accent_features(mel, frames)], axis=1)
        ctx = song_context(model, patches, nf, device, win)
        pos_lw = pos_fw = None
        if scratch_pos > 0 and scratch_codes is not None:     # 스크래치 자리 가중치: 확률(오즈) x w^scratch_pos
            pos_lw = scratch_pos * np.log(scratch_weight(scratch_codes)) * max(temperature, 1e-6)
            if scratch_beats is not None:                     # 앞 1박 안에 스크래치가 있으면 '이어지는 스크래치' 가중치
                pos_fw = scratch_pos * np.log(scratch_weight(scratch_codes, follow=True)) * max(temperature, 1e-6)
        last_scr = -1e9
        rng = np.random.default_rng(seed)
        hs = HistoryState()
        out = np.zeros(len(frames), dtype=np.int64)
        state = None
        jc = None
        if jack_penalty > 0 and jack_window_ms > 0:       # 소리 기반 연타 조절 (jack_ctl.py)
            from jack_ctl import JackControl
            jc = JackControl(mel, frames, jack_penalty, jack_window_ms / 10.0)
        cov, sflag = None, None
        if (coverage > 0 or vary > 0) and stream_ms > 0:  # 스트림 조절 (coverage_ctl.py)
            from coverage_ctl import Coverage, completes_repeat, stream_flags
            sflag = stream_flags(frames, stream_ms / 10.0)
            if coverage > 0:
                cov = Coverage(coverage)
        n_lone = 0
        n_alt_cut, alt_len = 0, 0
        sl = None
        if same_sound > 0 and same_window_ms > 0 and stream_ms > 0:   # 같은 소리는 같은 자리에 (sound_ctl.py)
            from sound_ctl import SoundLock
            sl = SoundLock(mel, frames, same_sound, same_window_ms / 10.0, stream_ms / 10.0)
        for i, f in enumerate(frames):
            rr = np.random.default_rng([int(seed) & 0xFFFFFFFF, 102, int(f)]) if row_seed else rng
            h = torch.from_numpy(hs.features(f)).to(device).view(1, 1, -1)
            logits, state = model.decode(ctx[i].view(1, 1, -1), h, state)
            forbid = 0
            if no_jack_ms > 0:
                recent = (f - hs.last) * 10.0 < no_jack_ms + 5.0   # 프레임 반올림(10ms) 여유: BPM 160 16분(93.75ms)이 10프레임으로 잡혀 빠지던 것 막음
                forbid = int(sum(1 << b for b in range(N_LANES) if recent[b]))
            lg = logits[0, 0].float().cpu().numpy()
            if jc is not None:
                lg = lg - jc.logit_penalty(i)
            if scratch_bias:
                lg[0] += scratch_bias                     # 스크래치(레인 0) 확률 올리기/내리기
            if pos_lw is not None:                        # 약한 박 자리일수록 스크래치 확률을 줄임
                near = pos_fw is not None and scratch_beats[i] - last_scr <= 1.0 + 1e-3
                lg[0] += pos_fw[i] if near else pos_lw[i]
            if sl is not None:
                lg = lg + sl.bias(i, out)                 # 비슷한 소리를 찍었던 레인 모양 쪽으로
            if cov is not None and sflag[i]:
                lg = cov.apply(lg, temperature, forbid)       # 건반 수는 그대로, 안 쓴 건반 쪽으로
            m = sample_mask(lg, temperature, max_chord, rr, forbid)
            if vary > 0 and sflag[i] and i >= 3:
                prev = [int(x) for x in out[max(0, i - 7):i]]
                # 반복이 완성될 때만 난수를 쓴다 (평소 난수 흐름을 바꾸지 않아서, 같은 시드면 나머지 결과가 그대로)
                if completes_repeat(prev, m) and rr.random() < vary:
                    for _ in range(4):                        # 같은 묶음 반복이 완성되면 다시 뽑아 섞기
                        m = sample_mask(lg, max(temperature, 1.0), max_chord, rr, forbid)
                        if not completes_repeat(prev, m):
                            break
            if max_alt_rows >= 2 and i >= 2 and frames[i] - frames[i - 2] <= 40:
                a, b = int(out[i - 2]) & 0xFE, int(out[i - 1]) & 0xFE
                if is_alt(a, b, m & 0xFE) and alt_len + 1 > max_alt_rows:
                    # 교대(A-B-A-B)가 제한보다 길어지면: 두 줄 전 모양의 레인을 눌러서 다시 뽑기
                    lg2 = lg.copy()
                    for x in range(1, 8):
                        if (a >> x) & 1:
                            lg2[x] -= 4.0
                    for _ in range(4):
                        m2 = sample_mask(lg2, temperature, max_chord, rr, forbid)
                        if not is_alt(a, b, m2 & 0xFE):
                            m = m2
                            n_alt_cut += 1
                            break
            if not lone_scratch and m == 1:
                # 스크래치만 있는 줄은 사람 패턴에서 3% 정도뿐 -> 건반 하나를 같이 붙인다 (금지 레인 제외, 확률 높은 순)
                for b in np.argsort(-lg[1:]) + 1:
                    if not (forbid >> int(b)) & 1:
                        m |= 1 << int(b)
                        n_lone += 1
                        break
            out[i] = m
            if (m & 1) and scratch_beats is not None:
                last_scr = float(scratch_beats[i])
            if i >= 2 and is_alt(int(out[i - 2]) & 0xFE, int(out[i - 1]) & 0xFE, m & 0xFE) \
                    and frames[i] - frames[i - 2] <= 40:
                alt_len = alt_len + 1 if alt_len else 3
            else:
                alt_len = 0
            hs.update(m, f)
            if jc is not None:
                jc.update(i, m)
            if sl is not None:
                sl.update(m)
            if cov is not None:
                cov.update(m, bool(sflag[i]))
        if sl is not None:
            print(sl.summary())
        if n_alt_cut:
            print(f"묶음 교대(1357/246 등)가 {max_alt_rows}줄을 넘으려던 곳 {n_alt_cut}곳을 다른 모양으로 바꿨어요")
        if n_lone:
            print(f"스크래치만 있던 줄 {n_lone}개에 건반 하나를 붙였어요 (끄려면 --lone-scratch)")
        return out


# ---------------------------------------------------------------------------
# 자체 점검 (numpy 부분)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    frames = np.cumsum(rng.integers(5, 30, n))
    masks = rng.integers(1, 256, n)
    # 1) 한꺼번에 계산 == 하나씩 계산 (구간 중간에서 시작해도)
    for s in (0, 7, 120):
        ip, il = init_state_from(masks, frames, s)
        L = 100
        full = history_features(masks[s:s + L], frames[s:s + L], ip, il)
        hs = HistoryState()
        for j in range(s):
            hs.update(masks[j], frames[j])
        for j in range(L):
            f = hs.features(frames[s + j])
            assert np.allclose(f, full[j], atol=1e-6), (s, j)
            hs.update(masks[s + j], frames[s + j])
    # 2) gap 벡터 == gap_feature
    for d in (1, 5, 37, 200, 1000):
        assert abs(_gap_vec(np.array([d]))[0] - gap_feature(d)) < 1e-6
    # 3) 흐름 지표: 무작위는 0 근처, 구간별로 스타일이 다른 시퀀스는 높게
    rand = flow_consistency(rng.integers(1, 256, 2000))
    styled = []
    for sec in range(40):
        if sec % 2 == 0:   # 계단 구간 (이동 1)
            styled += [1 << (1 + (i % 7)) for i in range(64)]
        else:              # 큰 도약 구간
            styled += [(1 << 1) if i % 2 else (1 << 7) for i in range(64)]
    st = flow_consistency(np.array(styled))
    print(f"흐름 지표: 무작위 move {rand['move']:+.2f} | 구간 스타일 있음 move {st['move']:+.2f}")
    assert abs(rand["move"]) < 0.15 and st["move"] > 0.5
    # 4) 박자 코드: 마디 첫 박=0, 둘째 박 8분 뒤=6, 셋잇단=격자 밖
    assert rhythm_code(0.0) == 0 and rhythm_code(1.5) == 6 and rhythm_code(3.75) == 15
    assert rhythm_code(1 / 3) == 16 and rhythm_code(2 + 2 / 3) == 18 and rhythm_code(-1) == -1
    B = 60 / 200
    tc = np.array([0, B, 2 * B + B / 4, 4 * B, 5 * B + B / 2])
    assert list(rhythm_codes_from_times(tc, 200, 0)) == [0, 4, 9, 0, 6]
    assert list(rhythm_codes_from_times(tc, 200, 1)) == [12, 0, 5, 12, 2]
    rf = rhythm_features(np.array([0, 6, 18, -1]))
    assert rf[0, 0] == 1 and rf[0, 5] == 1 and rf[1, 2] == 1 and rf[1, 6] == 1
    assert rf[2, 4] == 1 and rf[2, 7] == 1 and rf[3].sum() == 0
    print("numpy 부분 OK" + ("" if torch is None else " (torch 있음: python lane_v2.py --torch 로 모델 점검)"))
    if torch is not None and "--torch" in _sys.argv:
        sf = section_features(np.array([0, 10, 20, 200, 210]))
        assert sf.shape == (5, SEC_DIM) and sf[1, 2] > 0 and sf[3, 2] == 0 and sf[0, 3] > 0 and sf[2, 3] == 0
        assert nf_layout(4) == (False, False) and nf_layout(13) == (True, False) and nf_layout(17) == (True, True)
        assert nf_layout(24) == (True, True) and plan_in(24) and not plan_in(17)
        assert nf_layout(28) == (True, True) and plan_in(28) and acc_in(28) and not acc_in(24)
        af = accent_features(rng.random((500, 96)).astype(np.float32), np.array([0, 10, 250, 499]))
        assert af.shape == (4, ACC_DIM) and np.abs(af).max() <= 1.0
        for nd in (NOTE_DIM, NOTE_DIM + RHY_DIM, NOTE_DIM + RHY_DIM + SEC_DIM,
                   NOTE_DIM + RHY_DIM + SEC_DIM + PLAN_DIM,
                   NOTE_DIM + RHY_DIM + SEC_DIM + PLAN_DIM + ACC_DIM):             # v2 / v2r / v3 / v4 / v5
            m = LaneNetV2(note_dim=nd)
            B, L = 2, 40
            p = torch.rand(B, L, 96, 17)
            nf = torch.rand(B, L, nd)
            hm = rng.integers(1, 256, (B, L))
            fr = np.cumsum(rng.integers(5, 30, (B, L)), axis=1)
            h = torch.from_numpy(batch_history(hm, fr, np.zeros((B, K_PREV), np.int64),
                                               np.full((B, N_LANES), NEVER, np.int64)))
            y = m(p, nf, h)
            assert y.shape == (B, L, N_LANES)
            m.eval()
            # 한꺼번에(forward) == 한 노트씩(decode) : 같은 문맥, 같은 이력일 때
            with torch.no_grad():
                full = m(p[:1], nf[:1], h[:1])[0]
                enc = m.encode(p[0])
                ctx = m.context(enc[None], nf[:1])[0]
                st = None
                for i in range(L):
                    lg, st = m.decode(ctx[i].view(1, 1, -1), h[0, i].view(1, 1, -1), st)
                    assert torch.allclose(lg[0, 0], full[i], atol=1e-4), i
            fr1 = np.sort(rng.choice(3000, 300, replace=False))
            codes = rhythm_codes_from_times(fr1 / 100.0, 180) if nf_layout(nd)[0] else None
            g = generate_v2(m, rng.random((3000, 96)).astype(np.float32), fr1,
                            0.6, 8, "cpu", win=64, rhythm_codes=codes, jack_penalty=3.0, jack_window_ms=400,
                            scratch_bias=1.0, coverage=1.5, stream_ms=200, vary=0.7,
                            same_sound=2.0, same_window_ms=2000,
                            plan_feats=np.random.default_rng(1).random((300, PLAN_DIM)).astype(np.float32)
                            if plan_in(nd) else None)
            assert len(g) == 300 and (g > 0).all()
            print(f"모델 점검 OK (입력 {nd}차원) | 파라미터 {sum(q.numel() for q in m.parameters()):,}개")
