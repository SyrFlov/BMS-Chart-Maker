"""
[모델, 예전 버전] 레인 모델: 노트 시각마다 레인 8개를 칠지 말지(이진) 순서대로 예측

노트 하나의 처리:
    멜 조각 (n_mels x 2k+1) --CNN--> 소리 특징(hidden)
    + 보조 입력(직전 마스크 8, 간격 2, 곡 밀도 1) --Linear--> hidden
    -> 더해서 ReLU -> LSTM(앞 노트들의 흐름) -> Linear -> 레인 8개 로짓

학습: 직전 노트로 정답을 넣고(teacher forcing) 다음 노트를 맞힌다 -> forward()
생성: 방금 만든 마스크를 다음 입력으로 넣으며 한 노트씩 진행한다  -> step()
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lane_utils import AUX_DIM, N_LANES


class LaneNet(nn.Module):
    def __init__(self, n_mels: int = 96, patch_w: int = 17, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d((4, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d((4, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d((3, 2)),
        )
        flat = 64 * (n_mels // 4 // 4 // 3) * (patch_w // 2 // 2)    # 96x17 -> 64 x 2 x 4 = 512
        self.enc_proj = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(inplace=True))
        self.aux_proj = nn.Linear(AUX_DIM, hidden)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, N_LANES)      # 레인 8개 각각의 칠 확률(로짓)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """(N, n_mels, 2k+1) -> (N, hidden). 이전 노트와 무관하므로 한꺼번에 계산할 수 있다."""
        z = self.cnn(patches.unsqueeze(1))
        return self.enc_proj(z.flatten(1))

    def forward(self, patches: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """학습용. patches (B, L, n_mels, 2k+1), aux (B, L, 11) -> 로짓 (B, L, 8)."""
        B, L = patches.shape[:2]
        enc = self.encode(patches.reshape(B * L, *patches.shape[2:])).view(B, L, -1)
        x = self.drop(F.relu(enc + self.aux_proj(aux)))
        y, _ = self.rnn(x)
        return self.head(y)

    def step(self, enc_i: torch.Tensor, aux_i: torch.Tensor, state=None):
        """생성용. 노트 하나 진행. enc_i (1, hidden), aux_i (1, 11) -> 로짓 (1, 8), 새 상태."""
        x = F.relu(enc_i + self.aux_proj(aux_i)).unsqueeze(1)
        y, state = self.rnn(x, state)
        return self.head(y[:, 0]), state


def make_step_fn(model: LaneNet, patches: np.ndarray, device: str, chunk: int = 256):
    """곡 전체의 멜 조각을 미리 인코딩해 두고, lane_utils.generate_masks에 넘길 step_fn을 만든다.

    호출 전에 model.eval()을 해 두어야 한다.
    """
    with torch.no_grad():
        encs = [model.encode(torch.from_numpy(patches[i:i + chunk]).to(device))
                for i in range(0, len(patches), chunk)]
        enc = torch.cat(encs, dim=0) if encs else torch.zeros(0, 1, device=device)

    def step_fn(i: int, aux: np.ndarray, state):
        with torch.no_grad():
            a = torch.from_numpy(aux.astype(np.float32)).view(1, -1).to(device)
            logits, state = model.step(enc[i:i + 1], a, state)
        return logits[0].float().cpu().numpy(), state

    return step_fn


if __name__ == "__main__":
    # 모양 점검
    m = LaneNet()
    patches = torch.rand(2, 5, 96, 17)
    aux = torch.rand(2, 5, AUX_DIM)
    y = m(patches, aux)
    print(f"forward: 입력 patch {tuple(patches.shape)} aux {tuple(aux.shape)} -> 출력 {tuple(y.shape)}")
    assert y.shape == (2, 5, N_LANES)

    m.eval()
    fn = make_step_fn(m, patches[0].numpy(), "cpu")
    lg, st = fn(0, aux[0, 0].numpy(), None)
    lg2, st = fn(1, aux[0, 1].numpy(), st)
    print(f"step: 로짓 {lg.shape}, 상태 {tuple(st[0].shape)}")
    assert lg.shape == (N_LANES,)

    # 한 노트씩(step)과 한꺼번에(forward)가 같은 결과인지 확인
    with torch.no_grad():
        full = m(patches[:1], aux[:1])[0]
    st = None
    for i in range(5):
        lg, st = fn(i, aux[0, i].numpy(), st)
        assert np.allclose(lg, full[i].numpy(), atol=1e-4), f"step과 forward가 {i}번째에서 달라요"
    print(f"파라미터 {sum(p.numel() for p in m.parameters()):,}개 | step과 forward 결과 일치 | OK")
