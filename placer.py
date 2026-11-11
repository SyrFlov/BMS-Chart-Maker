"""
[모델, STEP 6] 자리 모델: 건반 수, 리듬은 그대로 두고, 앞 줄들을 보고 '어느 건반'인지만 다시 정한다

사용:
    python placer.py train --data <processed> --rhythm-dir <rhythm> --out placer_v1.pt [--steps 8000]
    python placer.py eval  --data <processed> --rhythm-dir <rhythm> --ckpt placer_v1.pt --out placer_eval.txt
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

FPS = 100
KEY_MASKS = np.arange(128, dtype=np.int64)                       # 건반 1~7 비트(1번 = 비트0)
KEY_POP = np.array([bin(m).count("1") for m in range(128)], dtype=np.int64)
RHY_DIM = 9
ACC_DIM = 4
COND_DIM = 7 + 1 + 2 + RHY_DIM + ACC_DIM + 1                       # k 원핫 7, 스크래치, 간격 2, 박자 9, 어택 4, 밀도 1
PREV_DIM = 14                                                    # 앞 두 줄 건반 7비트씩
_LOG_LO, _LOG_HI = np.log(0.01), np.log(2.0)


# ---------------------------------------------------------------- numpy 특징
def keys_of(mask8: np.ndarray) -> np.ndarray:
    """8비트 마스크(비트0 = 스크래치) -> 건반 7비트 (0~127)"""
    return (np.asarray(mask8, dtype=np.int64) >> 1) & 127


def mirror7(k7: np.ndarray) -> np.ndarray:
    k7 = np.asarray(k7, dtype=np.int64)
    out = np.zeros_like(k7)
    for b in range(7):
        out |= ((k7 >> b) & 1) << (6 - b)
    return out


def _gap(dt_frames):
    dt = np.maximum(np.asarray(dt_frames, dtype=np.float64) / FPS, 0.01)
    return np.clip((np.log(dt) - _LOG_LO) / (_LOG_HI - _LOG_LO), 0.0, 1.0)


def rhythm_feats(codes: np.ndarray) -> np.ndarray:
    from lane_v2 import rhythm_features
    return rhythm_features(np.asarray(codes, dtype=np.int64))


def cond_features(frames: np.ndarray, masks8: np.ndarray, codes: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """(n, COND_DIM) 생성 결과(자리)와 무관한 줄 조건"""
    f = np.asarray(frames, dtype=np.int64)
    n = len(f)
    k = np.array([bin(int(x)).count("1") for x in keys_of(masks8)])
    out = np.zeros((n, COND_DIM), np.float32)
    kk = np.clip(k, 1, 7) - 1
    out[np.arange(n), kk] = (k > 0).astype(np.float32)
    out[:, 7] = (np.asarray(masks8, dtype=np.int64) & 1).astype(np.float32)
    d = np.diff(f) if n > 1 else np.zeros(0, np.int64)
    dtp = np.concatenate([[200], d]) if n else np.zeros(0)
    dtn = np.concatenate([d, [200]]) if n else np.zeros(0)
    out[:, 8] = _gap(dtp)
    out[:, 9] = _gap(dtn)
    out[:, 10:10 + RHY_DIM] = rhythm_feats(codes)
    out[:, 10 + RHY_DIM:10 + RHY_DIM + ACC_DIM] = acc
    lo = np.searchsorted(f, f - 100, side="left")
    hi = np.searchsorted(f, f + 100, side="right")
    out[:, -1] = np.clip((hi - lo) / 40.0, 0.0, 1.5)
    return out


def prev_features(k7: np.ndarray) -> np.ndarray:
    """(n, 14) 앞 1줄, 2줄의 건반 비트"""
    n = len(k7)
    out = np.zeros((n, PREV_DIM), np.float32)
    bits = ((np.asarray(k7, dtype=np.int64)[:, None] >> np.arange(7)) & 1).astype(np.float32)
    if n > 1:
        out[1:, :7] = bits[:-1]
    if n > 2:
        out[2:, 7:] = bits[:-2]
    return out


def accent(mel, frames):
    from lane_v2 import accent_features
    return accent_features(np.asarray(mel), np.asarray(frames, dtype=np.int64))


# ---------------------------------------------------------------- 데이터
class PlaceData:
    def __init__(self, root, rhythm_dir, split: str, limit: int | None = None):
        from lane_utils import masks_from_labels
        root, rdir = Path(root), Path(rhythm_dir)
        with open(root / "index.csv", newline="", encoding="utf-8-sig") as fp:
            rows = [r for r in csv.DictReader(fp) if r["split"] == split]
        if limit:
            rows = rows[:limit]
        self.charts = []
        for r in rows:
            cid = r["chart_id"]
            rp = rdir / f"{cid}.rhy.npy"
            if not rp.exists():
                continue
            frames, masks = masks_from_labels(np.load(root / f"{cid}.label.npy"))
            if len(frames) < 64:
                continue
            per = np.load(rp)
            codes = per[np.clip(frames, 0, len(per) - 1)].astype(np.int64)
            mel = np.load(root / f"{cid}.mel.npy", mmap_mode="r")
            acc = accent(mel, frames)
            k7 = keys_of(masks)
            self.charts.append({"id": cid, "frames": frames, "masks": masks, "k7": k7, "codes": codes,
                                "cond": cond_features(frames, masks, codes, acc), "n": len(frames)})
        if not self.charts:
            raise ValueError(f"{split}: 차분 없음")
        sz = np.array([c["n"] for c in self.charts], float)
        self.p = sz / sz.sum()
        print(f"[{split}] 차분 {len(self.charts)}개, 줄 {int(sz.sum())}")

    def sample(self, rng, L: int, mirror: float):
        c = self.charts[int(rng.choice(len(self.charts), p=self.p))]
        s = int(rng.integers(0, max(c["n"] - L, 0) + 1))
        cond = c["cond"][s:s + L].copy()
        ext = c["k7"][max(0, s - 2):s + L].copy()              # 앞 두 줄까지 같이 (구간 첫 줄의 '앞 줄' 특징용)
        if rng.random() < mirror:
            ext = mirror7(ext)
            cond = cond.copy()                                 # 어택, 박자는 좌우와 무관
        n = len(cond)
        k7 = ext[-n:]
        pf = prev_features(ext)[-n:]
        x = np.zeros((L, COND_DIM + PREV_DIM), np.float32)
        x[:n, :COND_DIM] = cond
        x[:n, COND_DIM:] = pf
        y = np.full(L, -100, np.int64)
        kk = np.array([bin(int(v)).count("1") for v in k7])
        y[:n] = np.where(kk > 0, k7, -100)
        kpop = np.zeros(L, np.int64)
        kpop[:n] = kk
        return x, y, kpop


# ---------------------------------------------------------------- 모델
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None

if torch is not None:
    class PlacerNet(nn.Module):
        def __init__(self, d: int = 160, layers: int = 4, heads: int = 4, max_len: int = 128, dropout: float = 0.1):
            super().__init__()
            self.inp = nn.Linear(COND_DIM + PREV_DIM, d)
            self.pos = nn.Embedding(max_len, d)
            lay = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(lay, layers)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 128)
            self.max_len = max_len
            self.register_buffer("pop", torch.from_numpy(KEY_POP), persistent=False)

        def forward(self, x, kpop):
            """x (B, L, D), kpop (B, L) -> 로그확률 (B, L, 128), 건반 수가 다른 조합은 -inf"""
            B, L, _ = x.shape
            h = self.inp(x) + self.pos(torch.arange(L, device=x.device))[None]
            causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
            h = self.norm(self.tf(h, mask=causal))
            lg = self.head(h)
            ok = self.pop[None, None, :] == kpop[..., None]
            lg = lg.masked_fill(~ok, -1e4)
            return F.log_softmax(lg, dim=-1)


def _batch(data, rng, B, L, mirror):
    xs, ys, ks = zip(*[data.sample(rng, L, mirror) for _ in range(B)])
    return np.stack(xs), np.stack(ys), np.stack(ks)


def train(args):
    assert torch is not None, "torch 필요"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = PlaceData(args.data, args.rhythm_dir, "train", args.limit)
    va = PlaceData(args.data, args.rhythm_dir, "val", args.limit) if not args.no_val else None
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = PlacerNet(args.dim, args.layers, args.heads, args.len, args.dropout).to(dev)
    print(f"장치 {dev}, 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * max(0.05, 0.5 * (1 + np.cos(np.pi * min(s / args.steps, 1.0)))))
    best = 1e9
    vrng = np.random.default_rng(123)
    vb = [_batch(va, vrng, args.batch, args.len, 0.0) for _ in range(8)] if va else []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x, y, k = _batch(tr, rng, args.batch, args.len, args.mirror)
        x, y, k = (torch.from_numpy(a).to(dev) for a in (x, y, k))
        lp = model(x, k)
        loss = F.nll_loss(lp.reshape(-1, 128), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (step % args.eval_every == 0 or step == args.steps) and vb:
            model.eval()
            tl = tn = acc = 0.0
            with torch.no_grad():
                for x, y, k in vb:
                    x, y, k = (torch.from_numpy(a).to(dev) for a in (x, y, k))
                    lp = model(x, k)
                    m = y >= 0
                    tl += F.nll_loss(lp[m], y[m], reduction="sum").item()
                    acc += (lp.argmax(-1)[m] == y[m]).sum().item()
                    tn += m.sum().item()
            vl = tl / max(tn, 1)
            print(f"  [val] step {step} loss {vl:.3f} 정확도(같은 건반 수 안에서 1순위) {acc / max(tn, 1):.3f}", flush=True)
            if vl < best:
                best = vl
                torch.save({"state": model.state_dict(), "cfg": {"dim": args.dim, "layers": args.layers,
                            "heads": args.heads, "len": args.len}, "step": step, "val_loss": vl}, args.out)
    if not vb:
        torch.save({"state": model.state_dict(), "cfg": {"dim": args.dim, "layers": args.layers,
                    "heads": args.heads, "len": args.len}, "step": args.steps}, args.out)
    print(f"저장 {args.out} (val loss {best:.3f})")


def load(ckpt, device="cpu"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["cfg"]
    m = PlacerNet(c["dim"], c["layers"], c["heads"], c["len"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def _torch_score(model, x, kp, device):
    with torch.no_grad():
        return model(torch.from_numpy(x).float().to(device), torch.from_numpy(kp).to(device))[0, -1].cpu().numpy()


def place_masks(model, frames, masks8, codes, acc, jack_ms: float = 100.0, temperature: float = 0.8,
                top_p: float = 0.9, seed: int = 0, device: str = "cpu", keep=None, score_fn=None,
                rep_penalty: float = 0.0, rep_first: float = 0.0, beat_frames: float = 0.0, beat_pen: float = 0.0,
                beat_target=(0.145, 0.075), beat_win: int = 24, lag2_target=(0.066, 0.061), stair_target: float = 0.10,
                row_seed: bool = False):
    """건반 수, 스크래치는 그대로, 건반 자리만 앞에서부터 차례로 다시 뽑는다"""
    frames = np.asarray(frames, dtype=np.int64)
    masks8 = np.asarray(masks8, dtype=np.int64)
    n = len(frames)
    cond = cond_features(frames, masks8, codes, acc)
    k7 = keys_of(masks8).copy()
    kpop = KEY_POP[k7]
    L = getattr(model, "max_len", 128)
    score_fn = score_fn or _torch_score
    rng = np.random.default_rng(seed)
    out = k7.copy()
    jf = jack_ms / 1000.0 * FPS
    # beat_pen: 정확히 한 박 전 줄과 같은 조합 고르기 억제. 사람 277곡 폭타 마디: 한 박 전과 똑같은 줄
    # 얇은 폭타(줄당 1.6개 미만) 14.5% / 두꺼운 폭타 7.5%. 최근 beat_win줄(같은 두께 줄) 비율이
    # 목표보다 높으면 그 조합 로그확률에서 beat_pen을 뺌.
    copied = np.zeros(n, dtype=bool)
    has_b = np.zeros(n, dtype=bool)
    copied2 = np.zeros(n, dtype=bool)            # 그 줄이 두 줄 전과 같은가 (결정된 줄만)
    has2 = np.zeros(n, dtype=bool)
    # 16분 단타 연속에서 계단(1234, 7654처럼 한 칸씩 같은 방향 4줄+)이 이어지는가 (결정된 줄만)
    # 사람 277곡 단타 16분 연속 줄 중 계단 줄 곡별 중앙값 6%(상위 25% 24%), 생성 50% - 모델이 한번 계단을 타면 계속 탐(시드별 0~58%)
    on_stair = np.zeros(n, dtype=bool)
    is_single = np.zeros(n, dtype=bool)
    step_f = beat_frames * 0.3 if beat_frames > 0 else 12.0

    def _key1(m):
        return int(m).bit_length() - 1 if m and not (int(m) & (int(m) - 1)) else -1
    bj = np.full(n, -1, dtype=np.int64)
    if beat_frames > 0:
        for i in range(n):
            j = int(np.searchsorted(frames, frames[i] - beat_frames))
            for jj in (j - 1, j, j + 1):
                if 0 <= jj < i and abs(frames[i] - frames[jj] - beat_frames) <= 2.5:
                    bj[i] = jj
                    break
    for i in range(n):
        if kpop[i] == 0 or (keep is not None and keep[i]):
            continue
        s = max(0, i - L + 1)
        seg = out[s:i + 1]
        pf = prev_features(np.concatenate([out[max(0, s - 2):s], seg]))[-len(seg):]
        x = np.concatenate([cond[s:i + 1], pf], 1)[None]
        kp = kpop[s:i + 1][None]
        lp = score_fn(model, x, kp, device)
        # 연타 금지: jack_ms 안 앞뒤 줄(앞은 새 결과, 뒤는 아직 원래 값) 건반과 겹치는 조합 제외
        bl = 0
        j = i - 1
        while j >= 0 and frames[i] - frames[j] < jf:
            bl |= int(out[j])
            j -= 1
        j = i + 1
        while j < n and frames[j] - frames[i] < jf:
            if keep is not None and keep[j]:
                bl |= int(out[j])
            j += 1
        cand = np.flatnonzero((KEY_POP == kpop[i]) & ((KEY_MASKS & bl) == 0))
        if len(cand) == 0:
            continue                                   # 놓을 자리가 없으면 원래 그대로
        z = lp[cand] / max(temperature, 1e-3)
        if beat_pen > 0 and bj[i] >= 0 and out[bj[i]]:
            thin = kpop[i] == 1
            sel = [j for j in range(max(0, i - beat_win), i) if has_b[j] and (kpop[j] == 1) == thin]
            rate_ = float(np.mean(copied[sel])) if len(sel) >= 6 else 0.0
            if rate_ > (beat_target[0] if thin else beat_target[1]):
                z = z - beat_pen * (cand == out[bj[i]])
        if beat_pen > 0 and kpop[i] == 1 and i >= 3:
            # 바로 앞 세 줄이 16분 간격 단타로 한 칸씩 같은 방향(계단)이면, 최근 단타 중 계단 줄 비율이 사람보다 높을 때
            # 그 방향 다음 칸을 덜 고름
            ks_ = [_key1(out[i - j]) for j in (1, 2, 3)]
            if min(ks_) >= 0 and all(frames[i - j + 1] - frames[i - j] <= step_f for j in (1, 2, 3)):
                d1, d2 = ks_[0] - ks_[1], ks_[1] - ks_[2]
                if abs(d1) == 1 and d1 == d2 and 0 <= ks_[0] + d1 <= 6:
                    lo_ = max(0, i - 48)
                    sel = [j for j in range(lo_, i) if is_single[j]]
                    rate_ = float(np.mean(on_stair[sel])) if len(sel) >= 8 else 1.0
                    if rate_ > stair_target:
                        z = z - beat_pen * (cand == (1 << (ks_[0] + d1)))
        if beat_pen > 0 and i >= 2 and out[i - 2]:
            # 같은 두께 최근 줄 중 '두 줄 전과 똑같은 줄'(A B A B) 비율이 사람보다 높으면 덜 고름.
            # 폭타 줄당 건반을 2.3으로 올리자 자리 모델이 두꺼운 폭타에서 두 줄 전 그대로 8 -> 20%, 붙은 건반 트릴 1000줄당 44곳
            thin = kpop[i] == 1
            lo_ = max(0, i - beat_win)
            sel = [j for j in range(lo_, i) if has2[j] and (kpop[j] == 1) == thin]
            rate_ = float(np.mean(copied2[sel])) if len(sel) >= 6 else 0.0
            if rate_ > (lag2_target[0] if thin else lag2_target[1]):
                z = z - beat_pen * (cand == out[i - 2])
        if rep_penalty > 0 and i >= 2 and out[i - 2]:
            a = 1                                          # 두 줄 간격으로 같은 조합이 몇 번 이어졌나 (1 _ 1 _ 1, A B A B)
            while i - 2 * (a + 1) >= 0 and out[i - 2 * (a + 1)] == out[i - 2]:
                a += 1
            z = z - (rep_first + rep_penalty * (a - 1)) * (cand == out[i - 2])
        p = np.exp(z - z.max())
        p /= p.sum()
        o = np.argsort(-p)
        cp = np.cumsum(p[o])
        cut = o[: int(np.searchsorted(cp, top_p)) + 1]
        pp = p[cut] / p[cut].sum()
        if row_seed:
            # 줄마다 그 줄 프레임으로 정한 굼벨 잡음으로 뽑기 (확률이 조금 달라져도 같은 조합을 고르는 경우가 많아,
            # 앞에서 한 줄이 바뀌어도 뒤 배치가 통째로 다시 뽑히지 않음). 분포는 예전과 같음 (softmax(z) 위 상위 top_p)
            u_ = np.random.default_rng([int(seed) & 0xFFFFFFFF, 787, int(frames[i])]).random(len(KEY_MASKS))
            g_ = -np.log(-np.log(np.clip(u_, 1e-12, 1 - 1e-12)))
            out[i] = int(cand[cut[int(np.argmax(z[cut] + g_[cand[cut]]))]])
        else:
            out[i] = int(cand[cut[int(rng.choice(len(cut), p=pp))]])
        if kpop[i] == 1 and i >= 1 and frames[i] - frames[i - 1] <= step_f and _key1(out[i - 1]) >= 0:
            is_single[i] = True
            if i >= 3:
                ks_ = [_key1(out[i - j]) for j in (0, 1, 2, 3)]
                if min(ks_) >= 0 and all(frames[i - j + 1] - frames[i - j] <= step_f for j in (1, 2, 3)):
                    dd = [ks_[0] - ks_[1], ks_[1] - ks_[2], ks_[2] - ks_[3]]
                    if abs(dd[0]) == 1 and dd[0] == dd[1] == dd[2]:
                        on_stair[i - 3:i + 1] = True
        if i >= 2 and out[i - 2]:
            has2[i] = True
            copied2[i] = out[i] == out[i - 2]
        if bj[i] >= 0 and out[bj[i]]:
            has_b[i] = True
            copied[i] = out[i] == out[bj[i]]
    return (masks8 & 1) | (out << 1)


# ---------------------------------------------------------------- 평가
def flow_stats(frames, k7) -> dict:
    k7 = np.asarray(k7, dtype=np.int64)
    n = len(k7)
    norm = lambda m: tuple(b - min(bb for bb in range(7) if m >> bb & 1) for b in range(7) if m >> b & 1) if m else ()
    e2 = s1 = s2 = jack = adj = tot = 0
    for i in range(2, n):
        if k7[i] == 0:
            continue
        tot += 1
        e2 += k7[i] == k7[i - 2]
        s1 += norm(k7[i]) == norm(k7[i - 1]) and k7[i] != k7[i - 1]
        s2 += norm(k7[i]) == norm(k7[i - 2])
        jack += (frames[i] - frames[i - 1] < 10) and bool(k7[i] & k7[i - 1])
    edge = sum(1 for i in range(2, n) if (k7[i] & k7[i - 1] & k7[i - 2]) & 0b1000001)
    axis = sum(1 for i in range(2, n) if (k7[i] & k7[i - 1] & k7[i - 2]))
    return {"두 줄 전 그대로": e2 / max(tot, 1), "앞 줄 모양 옮김": s1 / max(tot, 1),
            "두 줄 전 같은 모양": s2 / max(tot, 1), "연타(100ms)": jack / max(tot, 1),
            "3줄 축 중 1·7": edge / max(axis, 1)}


def evaluate(args):
    assert torch is not None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    va = PlaceData(args.data, args.rhythm_dir, "val", args.limit)
    model = load(args.ckpt, dev)
    rng = np.random.default_rng(0)
    agg = {"사람": [], "자리 모델": [], "무작위(같은 건반 수)": []}
    exact = []
    for c in va.charts:
        fr, m8 = c["frames"], c["masks"]
        acc = c["cond"][:, 10 + RHY_DIM:10 + RHY_DIM + ACC_DIM]
        gen = place_masks(model, fr, m8, c["codes"], acc, temperature=args.temp, seed=1, device=dev)
        rnd = keys_of(m8).copy()
        for i in range(len(rnd)):
            k = KEY_POP[rnd[i]]
            if k:
                cand = np.flatnonzero(KEY_POP == k)
                rnd[i] = int(rng.choice(cand))
        agg["사람"].append(flow_stats(fr, keys_of(m8)))
        agg["자리 모델"].append(flow_stats(fr, keys_of(gen)))
        agg["무작위(같은 건반 수)"].append(flow_stats(fr, rnd))
        exact.append(np.mean(keys_of(gen) == keys_of(m8)))
    lines = [f"검증 차분 {len(va.charts)}개, 온도 {args.temp}", f"사람과 같은 줄(자리까지): {np.mean(exact):.1%}", ""]
    keys = list(agg["사람"][0])
    lines.append("지표 | " + " | ".join(agg))
    for k in keys:
        lines.append(f"{k} | " + " | ".join(f"{np.median([d[k] for d in v]):.1%}" for v in agg.values()))
    txt = "\n".join(lines)
    Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "smoke"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--rhythm-dir", required=True)
    ap.add_argument("--out", default="placer_v1.pt")
    ap.add_argument("--ckpt", default="placer_v1.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--len", type=int, default=128)
    ap.add_argument("--dim", type=int, default=160)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mirror", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "smoke":
        a.steps, a.batch, a.len, a.limit, a.eval_every, a.log_every = 30, 4, 64, 4, 15, 5
        a.out = a.ckpt = "placer_smoke.pt"
        train(a)
        a.temp = 0.8
        evaluate(a)
    elif a.cmd == "train":
        train(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
