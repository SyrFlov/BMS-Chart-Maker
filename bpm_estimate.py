"""
[규칙, STEP 1] 노트 시각들로부터 곡의 BPM(고정)을 추정한다. numpy만 사용.

원리:
    BPM b가 맞으면 노트들이 16분음표 격자(간격 60/(4b)초)에 몰린다.
    격자 위상에 노트가 얼마나 몰렸는지(R, 0~1)를 BPM 후보마다 계산해 가장 잘 맞는 BPM을 고른다.

배수 문제:
    BPM 180이 맞으면 360(격자가 더 촘촘)도 똑같이 잘 맞고, 노트가 8분 위주면 90도 맞는다.
    그래서 "거의 최고 점수인 후보들 중 선호 범위 안의 가장 느린 BPM"을 고르고, 2배/절반 후보를 같이 보여 준다.

한계:
    고정 BPM 곡만. 변속곡은 R이 낮게 나오며(경고 출력) 추정값을 믿으면 안 된다.
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import numpy as np


REL_DEFAULT = 0.8         # 정답 BPM이 알려진 패턴 240개(고정 BPM)로 맞춘 값


def r_curve(times: np.ndarray, bpms: np.ndarray, sub: int = 4, chunk: int = 512) -> np.ndarray:
    """BPM 후보마다 노트가 1/sub박 격자에 맞는 정도 R (0~1)."""
    t = np.asarray(times, dtype=np.float64)
    out = np.empty(len(bpms), dtype=np.float64)
    for i in range(0, len(bpms), chunk):
        b = np.asarray(bpms[i:i + chunk], dtype=np.float64)[:, None]
        ph = 2.0 * np.pi * t[None, :] * b * sub / 60.0
        out[i:i + chunk] = np.abs(np.exp(1j * ph).mean(axis=1))
    return out


def _refine(times, b0: float, half_width: float, sub: int, n: int = 201) -> tuple[float, float]:
    bs = np.linspace(b0 - half_width, b0 + half_width, n)
    r = r_curve(times, bs, sub)
    i = int(np.argmax(r))
    return float(bs[i]), float(r[i])


def _snap(times, b: float, r: float, sub: int, keep: float = 0.98) -> tuple[float, float]:
    """사람이 쓰는 BPM 값(정수 > 0.5 > 0.1 단위)으로 맞춰도 점수가 거의 같으면 그 값을 쓴다."""
    for step in (1.0, 0.5, 0.1):
        cand = round(b / step) * step
        rc = float(r_curve(times, np.array([cand]), sub)[0])
        if rc >= keep * r:
            return float(cand), rc
    return b, r


def find_peaks(times, lo: float = 60.0, hi: float = 400.0, sub: int = 4) -> tuple[np.ndarray, list]:
    """정규화한 노트 시각과, R 곡선의 봉우리 [(bpm, R), ...] (점수 내림차순)."""
    t = np.unique(np.round(np.asarray(times, dtype=np.float64), 4))
    if len(t) < 30:
        return t, []
    span = float(t[-1] - t[0])
    t = t - t[0]
    # 곡 전체에서 위상이 0.2주기 이상 어긋나지 않을 만큼 촘촘하게 훑는다 (곡이 길수록 촘촘)
    step = min(0.05, 0.2 * 60.0 / (sub * max(span, 1.0)))
    bs = np.arange(lo, hi + step, step)
    r = r_curve(t, bs, sub)
    is_peak = np.r_[False, (r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:]), False]
    idx = np.flatnonzero(is_peak)
    idx = idx[np.argsort(-r[idx])]
    peaks: list[tuple[float, float]] = []
    for i in idx:                                   # 0.5 BPM 안의 봉우리는 하나로
        if all(abs(bs[i] - p) > 0.5 for p, _ in peaks):
            peaks.append((float(bs[i]), float(r[i])))
        if len(peaks) >= 40:
            break
    peaks = sorted((_refine(t, b, step * 2, sub) for b, _ in peaks), key=lambda p: -p[1])
    return t, peaks


def choose(peaks: list, rel: float, prefer: tuple[float, float]) -> tuple[float, float]:
    """최고 점수의 rel배 이상인 후보 중, 선호 범위 안에서 가장 느린 BPM."""
    rmax = peaks[0][1]
    good = [p for p in peaks if p[1] >= rel * rmax]
    in_pref = [p for p in good if prefer[0] <= p[0] <= prefer[1]]
    return min(in_pref or good, key=lambda p: p[0])


def estimate_bpm(times, lo: float = 60.0, hi: float = 400.0, prefer: tuple[float, float] = (120.0, 280.0),
                 rel: float = REL_DEFAULT, sub: int = 4) -> dict:
    """노트 시각(초) -> {"bpm", "R", "alternatives", "candidates", "ok", "reason"}"""
    t, peaks = find_peaks(times, lo, hi, sub)
    if not peaks:
        return {"bpm": None, "R": 0.0, "alternatives": [], "candidates": [], "ok": False,
                "reason": "노트가 너무 적음"}
    bpm, R = _snap(t, *choose(peaks, rel, prefer), sub)

    alts = []                                       # 2배/절반은 따로 계산해서 보여 준다
    for k in (0.5, 2.0):
        bk = bpm * k
        if 30 <= bk <= 999:
            alts.append((round(bk, 3), float(r_curve(t, np.array([bk]), sub)[0])))
    cands = [(round(b, 3), round(rv, 3)) for b, rv in peaks[:6]]
    return {"bpm": round(bpm, 3), "R": round(R, 3), "alternatives": alts, "candidates": cands,
            "ok": R >= 0.5, "reason": "" if R >= 0.5 else "격자 적합도가 낮음 (변속곡이거나 노트가 불규칙)"}


def format_result(est: dict) -> str:
    if est["bpm"] is None:
        return f"BPM 추정 실패: {est['reason']}"
    s = f"추정 BPM {est['bpm']:g} (격자 적합도 R={est['R']:.2f})"
    alts = ", ".join(f"{b:g}(R={r:.2f})" for b, r in est["alternatives"])
    s += f"\n  2배/절반 후보: {alts}  <- 박자가 두 배로 빠르거나 느리게 느껴지면 --bpm 으로 직접 지정"
    if not est["ok"]:
        s += f"\n  주의: {est['reason']}. --bpm 으로 직접 지정하는 걸 권장"
    return s


if __name__ == "__main__":
    import sys
    from bms_parser import parse_bms
    for f in sys.argv[1:]:
        c = parse_bms(f)
        est = estimate_bpm([n.time for n in c.notes])
        print(f"{f}: 헤더 BPM {c.init_bpm:g} -> {format_result(est)}")
