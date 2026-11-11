"""
[규칙, STEP 3, 5] --sound16: 16분이 실제로 들리는지 분리 소리(드럼, 악기 어택)로 확인

(1) 16분 폭타 마디인데 16분 뒷자리 어택이 거의 없고 8분 어택이 뚜렷하면 8분 동시치기 마디로 바꾼다.
(2) 16분이 덜 찬 마디인데 16분 뒷자리 어택이 많으면 어택이 뚜렷한 칸만 채운다.
분리 결과가 있을 때만 적용한다.
"""
from __future__ import annotations

import numpy as np

FPS = 100


def _onsets(flux, frames, ratio: float = 1.2):
    from stem_follow import onset_at
    return onset_at(flux, frames, ratio=ratio)


def sound16_pick(t_chart, bpm: float, pad: float, stems, skip=(), min_fill: int = 12, eighth_max_off: int = 2,
                 eighth_min_on: int = 4, fill_min_off: int = 5, strong: float = 1.5, strength: float = 1.0):
    """반환: (8분 동시치기로 바꿀 마디 [(마디, 뒷자리 어택 수, 8분 어택 수)], 더할 칸 번호[박당 48칸 기준], 채운 마디 [(마디, 전 칸, 후 칸, 어택 수)])"""
    t = np.asarray(t_chart, dtype=np.float64)
    if stems is None or strength <= 0 or len(t) < 32:
        return [], [], []
    B = 60.0 / bpm
    sl = np.rint(t / B * 48).astype(np.int64)
    occ = set(sl.tolist())
    skip = set(int(x) for x in skip)
    ms = sorted(set((sl // 192).tolist()))
    dr, ot, ba, vo = stems["drums"], stems["other"], stems["bass"], stems["vocals"]
    T = len(dr)
    to8, add, filled = [], [], []
    for m in ms:
        if m in skip:
            continue
        cells = [m * 192 + 12 * j for j in range(16)]
        fr = np.rint((np.array(cells, dtype=np.float64) * B / 48.0 - pad) * FPS).astype(np.int64)
        if fr[0] < 0 or fr[-1] >= T:
            continue
        on_d, r_d = _onsets(dr, fr)
        on_o, r_o = _onsets(ot, fr)
        on_b, r_b = _onsets(ba, fr)
        on_v, _ = _onsets(vo, fr)
        on_all = on_d | on_o | on_b | on_v          # 8분으로 바꿀 때: 어느 층에도 16분 뒷자리 어택이 거의 없어야 (EOS 폭타는 베이스가 16분)
        on_fill = on_d | on_o | on_b                # 채울 때: 드럼, 악기, 베이스 (보컬 엇박은 멜로디 지키기 몫)
        off_all = int(on_all[1::2].sum())
        e8_on = int(on_all[0::2].sum())
        off_on = int(on_fill[1::2].sum())
        f16 = sum(1 for c in cells if c in occ)
        if f16 >= min_fill:
            if off_all <= eighth_max_off and e8_on >= eighth_min_on:
                to8.append((m, off_all, e8_on))
        elif off_on >= fill_min_off:
            got = []
            for j in range(1, 16, 2):
                c = cells[j]
                if c in occ or any((c + d) in occ for d in range(-5, 6) if d):
                    continue
                if max(r_d[j] if on_d[j] else 0.0, r_o[j] if on_o[j] else 0.0, r_b[j] if on_b[j] else 0.0) >= strong:
                    got.append(c)
            if got:
                add += got
                filled.append((m, f16, f16 + len(got), off_on))
    return to8, add, filled


def meas_rank(mel, bpm: float, pad: float, n_meas: int) -> dict:
    """곡 전체 마디 소리 크기 순위 0~1 (thick_ctl._meas_loud와 같은 기준)"""
    if mel is None:
        return {}
    from thick_ctl import _meas_loud
    ld = _meas_loud(mel, bpm, pad, range(n_meas))
    vs = sorted(ld.values())
    return {m: float(np.searchsorted(vs, v) / max(len(vs) - 1, 1)) for m, v in ld.items()}


def rest_ratio(rank: float, rest_chord: float, base: float = 0.76, loud: float = 0.6) -> float:
    """--rest-chord: 폭타를 쉬게 하는 8분 동시치기 마디 건반 수 = 주변 폭타 마디 x 배수"""
    if rest_chord > 0 and rank >= loud:
        return float(rest_chord)
    return base


def eighth_pulse(stems, bpm: float, pad: float, n_meas: int, min_on: int = 6, ratio: float = 1.2) -> set:
    """--dens-pulse: 분리 소리 드럼, 베이스, 악기 중 어느 층에든 8분 칸(마디 8칸) 어택이 min_on칸 이상인 마디 집합"""
    if stems is None or n_meas <= 0:
        return set()
    B = 60.0 / bpm
    dr, ot, ba = stems["drums"], stems["other"], stems["bass"]
    T = len(dr)
    out = set()
    for m in range(n_meas):
        fr = np.rint((np.array([m * 4 * B + j * B / 2 for j in range(8)]) - pad) * FPS).astype(np.int64)
        if fr[0] < 0 or fr[-1] >= T:
            continue
        on = _onsets(dr, fr, ratio)[0] | _onsets(ot, fr, ratio)[0] | _onsets(ba, fr, ratio)[0]
        if int(on.sum()) >= min_on:
            out.add(m)
    return out
