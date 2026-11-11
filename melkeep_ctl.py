"""
[규칙, STEP 3, 5] --mel-keep: 밀도를 덜 때 멜로디, 보컬 지키기

(1) 분리 소리에서 멜로디 점수가 높은 줄은 '구간 밀도 분포'가 빼지 않는다.
(2) '폭타 구간 바꾸기'에서 보컬, 악기 비중이 높은 4마디는 8비트 대신 반쯤 빈 16비트(또는 사다리 폭타)로 둔다.
분리 소리가 없으면 둘 다 하지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STEMS = ("drums", "bass", "other", "vocals")


def load_stems_cache(path, stereo, sr: int, model_name: str = "htdemucs"):
    """저장해 둔 분리 결과만 읽음 (없거나 다른 음원이면 None - Demucs를 새로 돌리지 않음, 약 6분 걸려서)"""
    try:
        p = Path(path)
        if not p.exists():
            return None
        from stem_follow import _cache_key
        z = np.load(p, allow_pickle=False)
        if str(z["key"]) != _cache_key(stereo, sr, model_name):
            return None
        return {s: z[s].astype(np.float32) for s in STEMS}
    except Exception:  # noqa: BLE001
        return None


def _mel_hp(mel, kt: int = 17, kf: int = 9):
    from scipy.ndimage import median_filter
    S = np.power(10.0, np.asarray(mel, dtype=np.float32) * 9.0)          # 0~1 로그 멜 -> 대략 선형 (90dB 범위)
    H = median_filter(S, size=(kt, 1), mode="nearest")
    P = median_filter(S, size=(1, kf), mode="nearest")
    Mh = H ** 2 / (H ** 2 + P ** 2 + 1e-12)
    return S * Mh, S * (1.0 - Mh)


def row_melodic(frames, mel=None, stems=None, pre: int = 3, post: int = 4):
    """줄마다 멜로디 점수 (클수록 멜로디, 보컬, 베이스 쪽). 반환: (점수 배열 또는 None, 'stems' | 'mel' | '')"""
    fr = np.asarray(frames, dtype=np.int64)
    if len(fr) == 0:
        return None, ""
    if stems is not None:
        d = stems["drums"]
        mv = stems["vocals"] + stems["other"] + stems["bass"]
        T = len(d)

        def mx(x, f):
            a, b = max(0, f - 2), min(T, f + 3)
            return float(x[a:b].max()) if b > a else 0.0
        sd = np.percentile(d[d > 0], 90) if (d > 0).any() else 1.0
        sm = np.percentile(mv[mv > 0], 90) if (mv > 0).any() else 1.0
        sc = np.array([np.log(mx(mv, int(f)) / sm + 1e-3) - np.log(mx(d, int(f)) / sd + 1e-3) for f in fr])
        return sc, "stems"
    if mel is None:
        return None, ""
    try:
        Hs, Ps = _mel_hp(mel)
    except Exception:  # noqa: BLE001
        return None, ""
    T = len(Hs)

    def inc(X, f):
        a = X[max(0, f - pre):max(1, f)].mean(0)
        b = X[min(f, T - 1):min(T, f + post)].mean(0)
        return float(np.maximum(b - a, 0.0).sum())
    sc = np.array([np.log(inc(Hs, int(f)) + 1e-6) - np.log(inc(Ps, int(f)) + 1e-6) for f in fr])
    return sc, "mel"


def protect_rows(score, strength: float = 1.0) -> np.ndarray:
    """곡 안 순위가 1 - 0.4 x strength 이상인 줄"""
    if score is None or strength <= 0 or len(score) == 0:
        return np.zeros(0 if score is None else len(score), dtype=bool)
    r = np.empty(len(score))
    r[np.argsort(score, kind="mergesort")] = np.arange(len(score))
    r /= max(len(score) - 1, 1)
    return r >= 1.0 - 0.4 * min(float(strength), 2.5)


def meas_vocal_rank(stems, bpm: float, pad: float, n_meas: int):
    """마디마다 (보컬 어택 비중 순위, 악기 어택 비중 순위) - 곡 안 0~1. 분리 소리가 없으면 None"""
    if stems is None:
        return None
    B = 60.0 / bpm
    tot = sum(stems[s] for s in STEMS) + 1e-6
    sh = {"vocals": {}, "other": {}}
    for m in range(n_meas):
        a, b = int((m * 4 * B - pad) * 100), int(((m + 1) * 4 * B - pad) * 100)
        if a < 0 or b > len(tot) or tot[a:b].sum() < 1.0:
            continue
        for k in sh:
            sh[k][m] = float(stems[k][a:b].sum() / tot[a:b].sum())
    out = {}
    for k, d in sh.items():
        vs = sorted(d.values())
        out[k] = {m: float(np.searchsorted(vs, v) / max(len(vs) - 1, 1)) for m, v in d.items()}
    return out


def melodic_units(sec_kind: dict, ranks, strength: float = 1.0, kinds=("C1", "C"), to: str = "H"):
    """'폭타 구간 바꾸기' 4마디 단위 중 보컬, 악기 비중 순위가 높은 곳을 to 종류('H' 반쯤 빈 16비트, 'L' 사다리 폭타)로 바꾼다
    반환: (새 종류 dict, 바꾼 단위 [(첫 마디, 끝 마디, 원래 종류, 순위)])"""
    if ranks is None or strength <= 0 or not sec_kind:
        return dict(sec_kind), []
    thr = 1.0 - 0.4 * min(float(strength), 2.5)
    ms = sorted(sec_kind)
    units, cur = [], []
    for m in ms:
        if cur and (m != cur[-1] + 1 or len(cur) >= 4):
            units.append(cur); cur = []
        cur.append(m)
    if cur:
        units.append(cur)
    out, changed = dict(sec_kind), []
    for u in units:
        kd = sec_kind[u[0]]
        if kd not in kinds:
            continue
        rv = float(np.mean([ranks["vocals"].get(m, 0.0) for m in u]))
        ro = float(np.mean([ranks["other"].get(m, 0.0) for m in u]))
        if max(rv, ro) >= thr:
            for m in u:
                out[m] = to
            changed.append((u[0], u[-1], kd, max(rv, ro), "보컬" if rv >= ro else "악기"))
    return out, changed
