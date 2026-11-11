"""
[학습] 학습용 멜 스펙트로그램 증강 (키음 믹스 -> "완성곡 mp3 같은 소리")
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import math

import torch
import torch.nn.functional as F

DB_MIN, DB_MAX = -80.0, 10.0
FMIN, FMAX, N_MELS_DEFAULT = 30.0, 16000.0, 96
FPS = 100


def _mel_centers_hz(n_mels: int, device) -> torch.Tensor:
    def hz2mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
    m = torch.linspace(hz2mel(FMIN), hz2mel(FMAX), n_mels + 2, device=device)[1:-1]
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _u(B, lo, hi, device):
    return torch.empty(B, device=device).uniform_(lo, hi)


def _bern(B, p, device):
    return torch.rand(B, device=device) < p


def _expand(v: torch.Tensor, ndim: int) -> torch.Tensor:
    """(B, ...) -> x와 같은 차원 수로 뒤에 1을 붙임."""
    return v.reshape(v.shape + (1,) * (ndim - v.dim()))


@torch.no_grad()
def augment_mel(x: torch.Tensor, p: float = 0.8, strength: float = 1.0) -> torch.Tensor:
    """x: (B, ..., M, T), 0~1. p: 샘플마다 증강할 확률. strength: 변형 세기 배율."""
    if p <= 0:
        return x
    dev, B, nd = x.device, x.shape[0], x.dim()
    M, T = x.shape[-2], x.shape[-1]
    s = strength
    db = x.float() * (DB_MAX - DB_MIN) + DB_MIN                     # dB로 되돌림
    on = _bern(B, p, dev)                                            # 이 샘플을 증강할지
    hz = _mel_centers_hz(M, dev)                                     # (M,)
    band = torch.linspace(0, 1, M, device=dev)                       # 0=저음 1=고음

    def gate(prob):                                                  # 증강 대상 & 이 변형을 쓸지
        return on & _bern(B, prob, dev)

    # 1) 볼륨
    g = gate(0.8)
    db = db + _expand(torch.where(g, _u(B, -6 * s, 6 * s, dev), torch.zeros(B, device=dev)), nd)

    # 2) EQ: 사인 곡선 3개를 합친 완만한 곡선 (M,)
    g = gate(0.7)
    if g.any():
        curve = torch.zeros(B, M, device=dev)
        for _ in range(3):
            freq = _u(B, 0.5, 3.0, dev)[:, None]
            phase = _u(B, 0, 2 * math.pi, dev)[:, None]
            amp = _u(B, 0, 3.0 * s, dev)[:, None]
            curve = curve + amp * torch.sin(2 * math.pi * freq * band[None] + phase)
        tilt = _u(B, -4 * s, 4 * s, dev)[:, None] * (band[None] - 0.5) * 2   # 밝게/어둡게
        curve = (curve + tilt) * g[:, None]
        db = db + curve.reshape((B,) + (1,) * (nd - 3) + (M, 1))

    # 3) mp3식 고음 잘림
    g = gate(0.5)
    if g.any():
        cut = _u(B, 11000, 16500, dev)[:, None]
        width = 800.0
        att = -40.0 * torch.sigmoid((hz[None] - cut) / (width / 4))    # (B, M)
        att = att * g[:, None]
        db = db + att.reshape((B,) + (1,) * (nd - 3) + (M, 1))

    # 4) 컴프레서: 프레임 크기(전 대역 평균 파워)가 임계값을 넘는 만큼 눌러 줌
    g = gate(0.6)
    if g.any():
        level = 10 * torch.log10(torch.mean(10 ** (db / 10), dim=-2, keepdim=True) + 1e-10)  # (..., 1, T)
        ref = level.flatten(1).quantile(0.9, dim=1)                  # 곡 구간에서 큰 소리 기준
        thr = _expand(ref - _u(B, 3, 15 * s, dev), nd)
        ratio = _expand(_u(B, 2, 8, dev), nd)
        red = -(1 - 1 / ratio) * F.relu(level - thr)
        # 어택/릴리즈 흉내: 시간축으로 살짝 번지게 (3프레임 평균)
        red = F.avg_pool1d(red.reshape(-1, 1, T), 3, 1, 1).reshape(red.shape)
        makeup = -red.mean(dim=-1, keepdim=True)                     # 줄어든 만큼 전체를 다시 키움
        db = db + (red + makeup) * _expand(g.float(), nd)

    # 5) 리버브: 파워 영역에서 지수 감쇠 꼬리를 시간축으로 더함
    g = gate(0.5)
    if g.any():
        K = min(120, T)                                              # 최대 1.2초 꼬리
        t = torch.arange(1, K + 1, device=dev, dtype=torch.float32)
        rt = _u(B, 0.15, 1.2, dev)[:, None] * FPS                    # 감쇠 시간(프레임)
        wet = 10 ** (_u(B, -22, -8 * min(s, 1.5), dev) / 10)[:, None]
        ker = wet * torch.exp(-6.9 * t[None] / rt)                   # rt에서 -60dB, (B, K)
        ker = ker * g[:, None]
        pw = 10 ** (db / 10)                                         # (B, ..., M, T)
        lead = pw.reshape(B, -1, T)                                  # (B, C, T)
        C = lead.shape[1]
        inp = F.pad(lead, (K, 0)).reshape(1, B * C, T + K)
        w = ker.flip(-1).repeat_interleave(C, dim=0)[:, None, :]     # (B*C, 1, K)
        tail = F.conv1d(inp, w, groups=B * C)[..., :T].reshape(pw.shape)
        db = 10 * torch.log10(pw + tail + 1e-10)

    # 6) 잡음 바닥
    g = gate(0.4)
    if g.any():
        floor = _u(B, -70, -45, dev)
        slope = _u(B, -10, 0, dev)[:, None] * band[None]             # 고음 쪽이 조금 작게
        n_db = (floor[:, None] + slope).reshape((B,) + (1,) * (nd - 3) + (M, 1))
        noise = n_db + torch.randn_like(db) * 3.0
        mix = 10 * torch.log10(10 ** (db / 10) + 10 ** (noise / 10))
        db = torch.where(_expand(g, nd), mix, db)

    # 7) 주파수 가리기 (밴드 0~2개, 각 최대 8밴드)
    out = ((db - DB_MIN) / (DB_MAX - DB_MIN)).clamp_(0, 1)
    g = gate(0.3)
    if g.any():
        for _ in range(2):
            w = torch.randint(0, 9, (B,), device=dev)
            f0 = (torch.rand(B, device=dev) * (M - w)).long()
            idx = torch.arange(M, device=dev)[None]
            m = (idx >= f0[:, None]) & (idx < (f0 + w)[:, None]) & g[:, None]   # (B, M)
            out = out.masked_fill(m.reshape((B,) + (1,) * (nd - 3) + (M, 1)), 0.0)

    # 증강 안 한 샘플은 원본 그대로
    keep = _expand(~on, nd)
    return torch.where(keep, x.float(), out).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(4, 96, 1000) * 0.3
    x[:, :, ::50] = 0.9                                              # 0.5초마다 강한 타격
    y = augment_mel(x, p=1.0)
    print("onset 형태", tuple(y.shape), "범위", float(y.min()), float(y.max()),
          "평균 변화", float((y - x).abs().mean()))
    z = augment_mel(torch.rand(3, 64, 96, 17), p=1.0)                # 레인 모델 패치 형태
    print("lane 형태", tuple(z.shape), "범위", float(z.min()), float(z.max()))
    assert torch.equal(augment_mel(x, p=0.0), x)
    print("OK")
