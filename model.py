"""
[모델, STEP 2] 노트 시각 모델: 멜 스펙트로그램 -> 프레임별 노트 확률 (CNN + BiLSTM)

입력  x : (B, n_mels, T)   멜 스펙트로그램 (값 0~1, 프레임당 10ms)
출력    : (B, n_out, T)    프레임별 로짓 (시그모이드를 씌우면 노트 확률)
    n_out = 1 : 타이밍 모드 - "어떤 레인이든 노트가 시작하는가"
    n_out = 8 : 레인 모드   - 레인별로 따로 예측

구조:
    Conv2d x3 (주파수 축만 풀링, 시간 해상도 10ms 유지)
      -> 프레임별 특징 벡터(채널 x 남은 주파수) -> Linear
      -> 양방향 LSTM (앞뒤 문맥)
      -> Linear -> 로짓
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OnsetNet(nn.Module):
    def __init__(self, n_mels: int = 96, n_out: int = 1,
                 channels: tuple[int, ...] = (16, 32, 64), pools: tuple[int, ...] = (3, 3, 2),
                 hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert len(channels) == len(pools)
        blocks: list[nn.Module] = []
        in_ch, f = 1, n_mels
        for ch, p in zip(channels, pools):
            blocks += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((p, 1)),        # 주파수 축만 줄이고 시간 축은 그대로
            ]
            in_ch, f = ch, f // p
        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(in_ch * f, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.rnn = nn.LSTM(hidden, hidden, num_layers=lstm_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, n_out)
        # 노트가 드문 데이터라(양성 ~3%) 시작 확률을 낮게 잡아 초반 학습을 안정시킨다
        nn.init.constant_(self.head.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t = x.shape
        z = self.cnn(x.unsqueeze(1))                       # (B, C, F, T)
        z = z.permute(0, 3, 1, 2).reshape(b, t, -1)        # (B, T, C*F)
        z = self.proj(z)                                   # (B, T, H)
        z, _ = self.rnn(z)                                 # (B, T, 2H)
        return self.head(z).transpose(1, 2)                # (B, n_out, T)


N_DENS = 13          # 문맥 가지의 보조 정답: 주변 1초 안의 노트 줄 수 0..12 (12 이상은 12)


class CtxOnsetNet(OnsetNet):
    """v6: OnsetNet + 긴 문맥 가지"""

    def __init__(self, n_mels: int = 96, n_out: int = 1, hidden: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.3, block: int = 25, **kw):
        super().__init__(n_mels=n_mels, n_out=n_out, hidden=hidden, lstm_layers=lstm_layers,
                         dropout=dropout, **kw)
        self.block = block
        self.ctx_rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.ctx_head = nn.Linear(2 * hidden, N_DENS)
        self.head = nn.Linear(4 * hidden, n_out)
        nn.init.constant_(self.head.bias, -3.0)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        b, _, t = x.shape
        z = self.cnn(x.unsqueeze(1))
        z = z.permute(0, 3, 1, 2).reshape(b, t, -1)
        z = self.proj(z)                                   # (B, T, H)
        fr, _ = self.rnn(z)                                # (B, T, 2H)
        nb = -(-t // self.block)
        pad = nb * self.block - t
        zp = nn.functional.pad(z, (0, 0, 0, pad)) if pad else z
        blk = zp.reshape(b, nb, self.block, -1).mean(2)    # (B, nb, H)
        cx, _ = self.ctx_rnn(blk)                          # (B, nb, 2H)
        up = cx.repeat_interleave(self.block, dim=1)[:, :t]
        out = self.head(torch.cat([fr, up], -1)).transpose(1, 2)
        if return_aux:
            return out, self.ctx_head(cx)                  # (B, n_out, T), (B, nb, N_DENS)
        return out


def dens_target(onset: torch.Tensor, block: int, radius: int = 50) -> torch.Tensor:
    """onset (B, T) 0/1 노트 프레임 -> 블록 가운데 기준 ±radius 프레임 안의 줄 수 (B, nb), 0..N_DENS-1"""
    b, t = onset.shape
    k = onset.new_ones(1, 1, 2 * radius + 1).float()
    cnt = nn.functional.conv1d(onset.unsqueeze(1).float(), k, padding=radius)[:, 0]    # (B, T)
    nb = -(-t // block)
    centers = (torch.arange(nb, device=onset.device) * block + block // 2).clamp(max=t - 1)
    return cnt[:, centers].round().long().clamp(0, N_DENS - 1)


def build_onset(a: dict, n_mels: int, n_out: int) -> nn.Module:
    """체크포인트 args로 맞는 모델을 만든다 (v6 문맥 모델이면 CtxOnsetNet)."""
    if a.get("ctx"):
        return CtxOnsetNet(n_mels=n_mels, n_out=n_out, hidden=a["hidden"], lstm_layers=a["lstm_layers"],
                           dropout=a["dropout"], block=a.get("ctx_block", 25))
    return OnsetNet(n_mels=n_mels, n_out=n_out, hidden=a["hidden"], lstm_layers=a["lstm_layers"],
                    dropout=a["dropout"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 모양 점검: 입력 (2, 96, 500) -> 출력 (2, n_out, 500)
    for n_out in (1, 8):
        m = OnsetNet(n_out=n_out)
        y = m(torch.rand(2, 96, 500))
        print(f"n_out={n_out}: 출력 {tuple(y.shape)}, 파라미터 {count_params(m):,}개")
        assert y.shape == (2, n_out, 500)
    m = CtxOnsetNet(n_out=1)
    y, aux = m(torch.rand(2, 96, 3010), return_aux=True)
    on = (torch.rand(2, 3010) < 0.05).float()
    tg = dens_target(on, 25)
    print(f"문맥 모델: 출력 {tuple(y.shape)}, 보조 {tuple(aux.shape)}, 정답 {tuple(tg.shape)} 최대 {int(tg.max())}, "
          f"파라미터 {count_params(m):,}개")
    assert y.shape == (2, 1, 3010) and aux.shape == (2, 121, N_DENS) and tg.shape == (2, 121)
    assert isinstance(build_onset({"ctx": True, "hidden": 64, "lstm_layers": 2, "dropout": 0.3}, 96, 1), CtxOnsetNet)
    print("OK")
