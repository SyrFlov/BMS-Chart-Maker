"""
[규칙, STEP 5] 16분 연타 속 '따라랑 따라랑'(점8분 = 16분 3칸마다 강세) 묶음을 동시치기 두께로 드러낸다 (기본 켬, --no-accent3)
"""
from __future__ import annotations

import numpy as np


def _keys(m: int) -> list[int]:
    return [b for b in range(1, 8) if (int(m) >> b) & 1]


def attack_strength(mel: np.ndarray, frames: np.ndarray, win: int = 2) -> np.ndarray:
    """mel (T, n_mels, 0~1), frames (N,) -> (N,) 줄마다 어택 세기 (양의 스펙트럼 상승량 합, ±win 프레임 최댓값)"""
    m = np.asarray(mel, dtype=np.float32)
    flux = np.zeros(len(m), dtype=np.float32)
    flux[1:] = np.maximum(m[1:] - m[:-1], 0.0).sum(axis=1)
    f = np.asarray(frames, dtype=np.int64)
    return np.max([flux[np.clip(f + d, 0, len(flux) - 1)] for d in range(-win, win + 1)], axis=0)


def find_accent3(t_chart, strength, bpm: float, win: int = 12, min_contrast: float = 0.9, margin: float = 0.2,
                 min_support: float = 0.5, min_windows: int = 3):
    """점8분(16분 3칸 주기) 강세를 여러 창에서 같은 위상으로 확인될 때만 정한다
    반환: role (N,) 0 = 손대지 않음, 1 = 강세 줄, 2 = 사이 줄"""
    t = np.asarray(t_chart, dtype=np.float64)
    s = np.asarray(strength, dtype=np.float64)
    n = len(t)
    role = np.zeros(n, dtype=np.int8)
    if n < win:
        return role
    q = t / (60.0 / bpm) * 4.0
    k16 = np.rint(q).astype(np.int64)
    on16 = np.abs(q - k16) < 1e-3
    support = np.zeros((n, 3))                       # 줄마다, 강세 위상(k16 % 3)별로 인정한 창 수
    cover = np.zeros(n)                              # 줄마다, 덮은 16분 창 수
    for a in range(0, n - win + 1):
        idx = np.arange(a, a + win)
        if not on16[idx].all() or np.any(np.diff(k16[idx]) != 1):   # 끊김 없는 16분 연타 12칸만
            continue
        cover[idx] += 1
        v = s[idx]
        sd = v.std()
        if sd < 1e-6:
            continue
        kk = k16[idx]
        c3 = [(v[kk % 3 == r].mean() - v[kk % 3 != r].mean()) / sd for r in range(3)]
        c4 = max((v[kk % 4 == r].mean() - v[kk % 4 != r].mean()) / sd for r in range(4))
        r3 = int(np.argmax(c3))
        if c3[r3] >= min_contrast and c3[r3] >= c4 + margin:
            support[idx, r3] += 1
    best = support.argmax(1)
    ok = (support.max(1) >= min_windows) & (support.max(1) >= min_support * np.maximum(cover, 1))
    role[ok & (k16 % 3 == best)] = 1
    role[ok & (k16 % 3 != best)] = 2
    return role


def apply_accent3(t_chart, masks, role, seed: int = 0, acc_min: int = 2):
    """강세 줄은 acc_min개 이상 동시치기, 사이 줄은 단타로. 반환: (masks, 두껍게 한 줄, 얇게 한 줄)"""
    masks = [int(m) for m in masks]
    n = len(masks)
    rng = np.random.default_rng(seed + 23)
    thick = thin = 0
    for i in range(n):
        if role[i] == 2:
            ks = _keys(masks[i])
            if len(ks) >= 2:
                keep = int(rng.choice(ks))
                masks[i] = (masks[i] & 1) | (1 << keep)
                thin += 1
    for i in range(n):
        if role[i] != 1:
            continue
        m = masks[i]
        ks = _keys(m)
        if len(ks) >= acc_min:
            continue
        block = m | (masks[i - 1] if i > 0 else 0) | (masks[i + 1] if i < n - 1 else 0)
        while len(ks) < acc_min:
            cands = [c for c in range(1, 8) if not (block >> c) & 1]
            if not cands:
                break
            if ks:
                d = np.array([min(abs(c - k) for k in ks) for c in cands], dtype=np.float64)
                w = np.where(d >= 2, d, 0.1)
            else:
                w = np.ones(len(cands))
            c = int(rng.choice(cands, p=w / w.sum()))
            m |= 1 << c
            block |= 1 << c
            ks.append(c)
        if m != masks[i]:
            masks[i] = m
            thick += 1
    return masks, thick, thin


if __name__ == "__main__":
    # 자체 점검: 16분 32칸, 어택이 3칸마다 크면 강세/사이로 나뉘고, 4칸(박) 강세만 있으면 손대지 않아야 함
    bpm = 180.0
    t = np.arange(32) * (60 / bpm / 4)
    rng = np.random.default_rng(0)
    s3 = np.where(np.arange(32) % 3 == 1, 3.0, 1.0) + rng.normal(0, 0.1, 32)
    role = find_accent3(t, s3, bpm)
    print("3칸 강세:", "".join("-AF"[r] for r in role))
    assert (role[1::3] == 1).mean() > 0.8 and (role[np.arange(32) % 3 != 1] == 2).mean() > 0.8
    masks = [(1 << 2) | (1 << 5) if i % 2 else 1 << (1 + i % 7) for i in range(32)]
    new, th, tn = apply_accent3(t, masks, role)
    kc = [len(_keys(m)) for m in new]
    print("두께:", kc, f"(두껍게 {th}, 얇게 {tn})")
    assert all(kc[i] >= 2 for i in range(32) if role[i] == 1) and all(kc[i] == 1 for i in range(32) if role[i] == 2)
    assert all(not (new[i] & new[i + 1] & 0xFE) or (masks[i] & masks[i + 1] & 0xFE) for i in range(31))
    s4 = np.where(np.arange(32) % 4 == 0, 3.0, 1.0) + rng.normal(0, 0.1, 32)
    role4 = find_accent3(t, s4, bpm)
    print("4칸(박) 강세:", "".join("-AF"[r] for r in role4))
    assert (role4 == 0).all()
    print("OK")


def band_strength(mel: np.ndarray, frames: np.ndarray, lo_bin: int = 0, hi_bin: int = 40, win: int = 2) -> np.ndarray:
    """저, 중음 대역(기본 멜 0~39번, 약 1kHz 아래)만의 어택 세기. 하이햇(8분 강세)에 묻히지 않게."""
    m = np.asarray(mel, dtype=np.float32)[:, lo_bin:hi_bin]
    flux = np.zeros(len(m), dtype=np.float32)
    flux[1:] = np.maximum(m[1:] - m[:-1], 0.0).sum(axis=1)
    f = np.asarray(frames, dtype=np.int64)
    return np.max([flux[np.clip(f + d, 0, len(flux) - 1)] for d in range(-win, win + 1)], axis=0)


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """'39-41.5,100-102' -> [(39, 41.5), (100, 102)] (음원 기준 초)"""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return out


def manual_roles(t_chart, audio_t, strength, bpm: float, ranges, phase: int = -1):
    """사용자가 준 음원 구간(초)에서 16분 줄을 점8분 묶음으로 본다
    반환: role (0 손대지 않음 / 1 강세 / 2 사이), 고른 위상 목록"""
    t = np.asarray(t_chart, dtype=np.float64)
    at = np.asarray(audio_t, dtype=np.float64)
    s = np.asarray(strength, dtype=np.float64)
    q = t / (60.0 / bpm) * 4.0
    k16 = np.rint(q).astype(np.int64)
    on16 = np.abs(q - k16) < 1e-3
    role = np.zeros(len(t), dtype=np.int8)
    phases = []
    for a, b in ranges:
        sel = (at >= a) & (at <= b) & on16
        if sel.sum() < 6:
            phases.append(None)
            continue
        if phase in (0, 1, 2):
            ph = phase
        else:
            z = (s[sel] - s[sel].mean()) / (s[sel].std() + 1e-9)
            ph = int(np.argmax([z[k16[sel] % 3 == r].mean() for r in range(3)]))
        role[sel & (k16 % 3 == ph)] = 1
        role[sel & (k16 % 3 != ph)] = 2
        phases.append(ph)
    return role, phases
