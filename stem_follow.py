"""
[규칙, STEP 3] 음원 분리(Demucs)로 드럼 / 악기 / 보컬 층을 나눠, 구간마다 고른 층을 따라 노트를 남긴다
(--follow, --follow-at, --follow-strength)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

FPS = 100
STEMS = ("drums", "bass", "other", "vocals")
_WORDS = {
    "drum": "drum", "drums": "drum", "d": "drum", "드럼": "drum", "타악": "drum",
    "inst": "inst", "instrument": "inst", "i": "inst", "other": "inst", "악기": "inst", "반주": "inst",
    "melody": "inst", "melo": "inst", "m": "inst", "멜로디": "inst", "멜": "inst", "선율": "inst",   # 예전 이름
    "vocal": "vocal", "vocals": "vocal", "v": "vocal", "voice": "vocal", "보컬": "vocal", "목소리": "vocal", "노래": "vocal",
    "off": "off", "none": "off", "auto": "off", "끔": "off", "자동": "off", "없음": "off", "-": "off",
}


def mode_word(w: str) -> str:
    k = str(w).strip().lower()
    if k in ("", "0", "false"):
        return "off"
    if k not in _WORDS:
        raise SystemExit(f"--follow 값 '{w}'을(를) 모르겠어요. drum(드럼) / inst(악기) / vocal(보컬) / off(끔) 중 하나로 써 주세요.")
    return _WORDS[k]


_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(초|s|sec)?\s*(?:([-~])\s*(?:(\d+(?:\.\d+)?)\s*(초|s|sec)?)?)?\s*[:=]?\s*(\S+)",
                    re.IGNORECASE)


def parse_follow_at(spec: str) -> list[tuple[float, float | None, str, str]]:
    """'1-16:drum, 17-32 악기, 33-:off, 1~3초 보컬' -> [(lo, hi, 모드, 'bar' 또는 'sec'), ...]
    hi가 None이면 곡 끝까지"""
    out = []
    if not spec or not spec.strip():
        return out
    for part in re.split(r"[,;/\n]+", spec):
        part = part.strip()
        if not part:
            continue
        m = _RANGE.fullmatch(part)
        if not m:
            raise SystemExit(f"--follow-at의 '{part}'을(를) 읽지 못했어요. 예: \"1-16:drum, 17-32 악기\" 또는 \"1~3초 보컬\"")
        a, ua, dash, b, ub, word = m.groups()
        sec = bool(ua or ub)
        lo = float(a)
        if dash:
            hi = float(b) if b else None
        else:
            hi = lo + 1.0 if sec else lo
        if hi is not None and hi < lo:
            lo, hi = hi, lo
        if not sec:
            if lo != int(lo) or (hi is not None and hi != int(hi)):
                raise SystemExit(f"--follow-at의 '{part}': 마디 번호는 정수로, 초로 쓰려면 숫자 뒤에 '초'를 붙여 주세요.")
            lo, hi = int(lo), (int(hi) if hi is not None else None)
        elif hi is not None and hi <= lo:
            raise SystemExit(f"--follow-at의 '{part}': 초 구간은 끝이 시작보다 커야 해요 (예 1~3초).")
        out.append((lo, hi, mode_word(word), "sec" if sec else "bar"))
    return out


def measure_modes(n_meas: int, default: str, ranges) -> list[str]:
    """마디 구간만 반영한 마디별 모드 (초 구간은 row_modes에서 줄마다 반영)"""
    modes = [default] * n_meas
    for r in ranges:                               # 뒤에 적은 것이 이김
        lo, hi, md = r[0], r[1], r[2]
        if len(r) > 3 and r[3] == "sec":
            continue
        for k in range(lo, (n_meas - 1 if hi is None else hi) + 1):
            if 0 <= k < n_meas:
                modes[k] = md
    return modes


def row_modes(meas: np.ndarray, times: np.ndarray, default: str, ranges) -> np.ndarray:
    """줄마다 모드: 기본값에서 시작해 적은 순서대로 마디/초 구간을 덮어씀 (뒤에 적은 것이 이김)"""
    out = np.full(len(meas), default, dtype=object)
    for r in ranges:
        lo, hi, md = r[0], r[1], r[2]
        if len(r) > 3 and r[3] == "sec":
            sel = (times >= lo - 1e-6) & ((times < hi - 1e-6) if hi is not None else True)
        else:
            sel = (meas >= lo) & ((meas <= hi) if hi is not None else True)
        out[sel] = md
    return out


# ---------------------------------------------------------------------------
# 어택 세기 (스펙트럼 플럭스, 10ms)
# ---------------------------------------------------------------------------
def spectral_flux(x: np.ndarray, sr: int, nfft: int = 2048) -> np.ndarray:
    """모노 신호 -> 10ms마다 양의 스펙트럼 변화량 합 (프레임 k = k*10ms 중심)"""
    from numpy.lib.stride_tricks import sliding_window_view
    hop = sr / FPS
    x = np.asarray(x, dtype=np.float32)
    n = int(len(x) / hop) + 1
    xp = np.pad(x, (nfft // 2, nfft // 2 + nfft))
    win = np.hanning(nfft).astype(np.float32)
    starts = np.round(np.arange(n) * hop).astype(np.int64)
    out = np.zeros(n, np.float32)
    prev = None
    view = sliding_window_view(xp, nfft)
    for a in range(0, n, 2048):
        fr = view[starts[a:a + 2048]] * win
        mag = np.log1p(10.0 * np.abs(np.fft.rfft(fr, axis=1))).astype(np.float32)
        d = np.diff(mag, axis=0, prepend=mag[:1] if prev is None else prev[None])
        out[a:a + len(mag)] = np.maximum(d, 0.0).sum(axis=1)
        prev = mag[-1]
    return out


def _cache_key(stereo: np.ndarray, sr: int, model_name: str) -> str:
    import hashlib
    a = np.ascontiguousarray(stereo[:: max(1, len(stereo) // 20000)], dtype=np.float32)
    return f"{model_name}:{sr}:{len(stereo)}:{hashlib.md5(a.tobytes()).hexdigest()}"


def stem_fluxes(stereo: np.ndarray, sr: int, model_name: str = "htdemucs", device: str = "cpu",
                cache: str | Path | None = None, log=print) -> dict[str, np.ndarray] | None:
    """음원(N,2) -> {'drums','bass','other','vocals': 10ms 어택 세기}. Demucs가 없으면 None."""
    key = _cache_key(stereo, sr, model_name)
    if cache is not None and Path(cache).exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                log(f"음원 분리: 저장해 둔 결과를 씀 ({Path(cache).name})")
                return {s: z[s].astype(np.float32) for s in STEMS}
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError:
        log("[주의] 음원 분리 모델(Demucs)이 설치돼 있지 않아 --follow를 건너뛰어요. "
            "bms-ai 환경에서 'pip install demucs'로 설치하면 쓸 수 있어요.")
        return None
    import time
    from render_audio import resample
    t0 = time.time()
    try:
        model = get_model(model_name)                # 처음엔 인터넷에서 모델(약 80MB)을 받음
    except Exception as e:  # noqa: BLE001
        log(f"[주의] 음원 분리 모델을 불러오지 못해 --follow를 건너뛰어요: {type(e).__name__}: {e}")
        return None
    model.eval()
    msr = int(model.samplerate)
    wav = resample(np.asarray(stereo, dtype=np.float32), sr, msr) if sr != msr else np.asarray(stereo, np.float32)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], 1)
    x = torch.from_numpy(np.ascontiguousarray(wav[:, :2].T))          # (2, T)
    ref = x.mean(0)
    mu, sd = float(ref.mean()), float(ref.std()) + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((x - mu) / sd)[None], device=device, split=True, overlap=0.25, progress=False)[0]
    out = out * sd + mu
    names = list(model.sources)
    res = {}
    for s in STEMS:
        mono = out[names.index(s)].mean(0).cpu().numpy() if s in names else np.zeros(x.shape[1], np.float32)
        res[s] = spectral_flux(mono, msr)
    log(f"음원 분리({model_name}, {device}): {time.time() - t0:.0f}초")
    if cache is not None:
        try:
            np.savez_compressed(cache, key=np.array(key), **{s: res[s].astype(np.float16) for s in STEMS})
        except Exception:  # noqa: BLE001
            pass
    return res


def onset_at(flux: np.ndarray, frames: np.ndarray, peak: int = 2, radius: int = 35,
             ratio: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """줄마다 (그 층 어택이 있나, 어택 세기 비율). 비율 = ±peak 프레임 최댓값 / 주변 ±radius 중앙값"""
    T = len(flux)
    f = np.clip(np.asarray(frames, dtype=np.int64), 0, max(T - 1, 0))
    if T == 0:
        return np.zeros(len(f), bool), np.zeros(len(f))
    scale = float(np.percentile(flux, 95)) + 1e-6
    pk = np.array([flux[max(0, i - peak):i + peak + 1].max() for i in f])
    med = np.array([np.median(flux[max(0, i - radius):i + radius + 1]) for i in f])
    r = pk / (med + 0.02 * scale)
    return (r >= ratio) & (pk >= 0.06 * scale), r


STEM_OF = {"drum": ("drums",), "inst": ("other",), "vocal": ("vocals",)}
KO = {"drum": "드럼", "inst": "악기", "vocal": "보컬"}


def follow_rows(t_chart: np.ndarray, times: np.ndarray, bpm: float, flux: dict[str, np.ndarray],
                default: str = "off", ranges=(), strength: float = 0.7, seed: int = 0,
                max_drop: float = 0.7, quiet: float = 0.1):
    """남길 줄(bool 배열)과 보고를 돌려준다. 보고: {모드: [대상 줄, 뺀 줄, 마디 수], 'skipped': [(마디, 모드)]}"""
    t_chart = np.asarray(t_chart, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n = len(t_chart)
    keep = np.ones(n, bool)
    if n == 0 or strength <= 0:
        return keep, {}
    bar = 4 * 60.0 / bpm
    meas = np.floor(t_chart / bar + 1e-6).astype(np.int64)
    modes = row_modes(meas, times, default, ranges)
    if all(m == "off" for m in modes):
        return keep, {}
    fr = np.rint(times * FPS).astype(np.int64)
    on_r = {}
    for md, stems in STEM_OF.items():
        ons, rs = zip(*[onset_at(flux[s], fr) for s in stems])
        on_r[md] = (np.logical_or.reduce(ons), np.maximum.reduce(rs))
    pad = float(t_chart[0] - times[0]) if n else 0.0

    def act(name, k):
        a = int(max(0, (k * bar - pad) * FPS))
        b = int(max(a + 1, ((k + 1) * bar - pad) * FPS))
        x = flux[name][a:b]
        return float(x.mean()) if len(x) else 0.0
    used = sorted(set(meas.tolist()))
    ref = {s: np.median([act(s, k) for k in used]) + 1e-6 for s in ("drums", "other", "vocals")}
    rng = np.random.default_rng(seed + 211)
    rep = {md: [0, 0, 0] for md in STEM_OF}
    rep["skipped"] = []
    for k in used:
        idx_k = np.flatnonzero(meas == k)
        for md in STEM_OF:
            idx = idx_k[modes[idx_k] == md]
            if len(idx) == 0:
                continue
            if max(act(s, k) / ref[s] for s in STEM_OF[md]) < quiet:     # 그 마디에 따라갈 층 소리가 거의 없음
                rep["skipped"].append((k, md))
                continue
            on, r = on_r[md][0][idx], on_r[md][1][idx]
            cand = idx[~on]
            drop = cand[rng.random(len(cand)) < strength]
            lim = int(max_drop * len(idx))
            if len(drop) > lim:                         # 어택이 가장 약한 줄부터 lim개만
                drop = drop[np.argsort(r[np.searchsorted(idx, drop)])[:lim]]
            keep[drop] = False
            rep[md][0] += len(idx)
            rep[md][1] += len(drop)
            rep[md][2] += 1
    return keep, rep
