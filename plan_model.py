"""
[모델, STEP 4] v4 패턴 계획 모델: 소리를 보고 반 마디마다 어떤 모양으로 찍을지(plan_feats.py의 7가지)를 정한다

사용:
    python plan_model.py --data <processed> --rhythm-dir <rhythm> --good --good-weight 3 --steps 3000 --aug 0.8 --out plan.pt
    python plan_model.py --selftest
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lane_utils import extract_patches
from lane_v2 import note_features, rhythm_features, section_features
from plan_feats import PLAN_DIM, PLAN_NAMES, half_groups

NB = 5                                                          # 수치마다 5칸
PLAN_MAX = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0], dtype=np.float32)   # move만 0~0.6


def to_bins(v: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(v) / PLAN_MAX * NB).astype(np.int64), 0, NB - 1)


def default_centers() -> np.ndarray:
    return ((np.arange(NB) + 0.5) / NB)[None, :] * PLAN_MAX[:, None]


class PlanNet(nn.Module):
    def __init__(self, n_mels: int = 96, nf_dim: int = 17, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.note = nn.Sequential(nn.Linear(2 * n_mels + nf_dim, hidden), nn.ReLU())
        self.grp = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, PLAN_DIM * NB)

    def forward(self, patch: torch.Tensor, nf: torch.Tensor, gidx: torch.Tensor, G: int) -> torch.Tensor:
        """patch (B, L, M, P), nf (B, L, D), gidx (B, L) 묶음 번호(없으면 -1) -> 로짓 (B, G, PLAN_DIM, NB)"""
        x = torch.cat([patch.mean(-1), patch.amax(-1), nf], -1)
        h = self.note(x)                                            # (B, L, H), ReLU라 0 이상
        B, L, H = h.shape
        valid = (gidx >= 0).unsqueeze(-1)
        idx = gidx.clamp(min=0).unsqueeze(-1)
        w = valid.to(h.dtype)
        s = torch.zeros(B, G, H, device=h.device, dtype=h.dtype).scatter_add(1, idx.expand(-1, -1, H), h * w)
        cnt = torch.zeros(B, G, 1, device=h.device, dtype=h.dtype).scatter_add(1, idx, w)
        mean = s / cnt.clamp(min=1.0)
        mx = torch.zeros(B, G, H, device=h.device, dtype=h.dtype).scatter_reduce(
            1, idx.expand(-1, -1, H), h * w, reduce="amax", include_self=True)
        g = self.grp(torch.cat([mean, mx, (cnt / 16.0).clamp(max=2.0), (cnt > 0).to(h.dtype)], -1))
        y, _ = self.gru(g)
        return self.head(y).view(B, G, PLAN_DIM, NB)


def plan_inputs(mel: np.ndarray, frames: np.ndarray, codes: np.ndarray, quality: float, patch_k: int = 8):
    """생성용 입력 (레인 모델 v3와 같은 노트 특징: 기본 4 + 박자 9 + 구간 4)"""
    frames = np.asarray(frames, dtype=np.int64)
    nf = np.concatenate([note_features(frames, quality), rhythm_features(codes), section_features(frames)], axis=1)
    return extract_patches(mel, frames, patch_k), nf.astype(np.float32)


@torch.no_grad()
def predict_plan(model: PlanNet, centers: np.ndarray, mel: np.ndarray, frames: np.ndarray, codes: np.ndarray,
                 quality: float = 1.0, temperature: float = 0.5, device: str = "cpu", patch_k: int = 8):
    """반환: (노트별 계획 (n, PLAN_DIM), 묶음별 계획 (G, PLAN_DIM), 묶음 번호 (n,))
    temperature가 낮을수록 가장 그럴듯한 칸 쪽으로 (0.5 권장)"""
    model.eval()
    groups = half_groups(frames, codes)
    G = int(groups.max()) + 1
    patch, nf = plan_inputs(mel, frames, codes, quality, patch_k)
    lg = model(torch.from_numpy(patch)[None].to(device), torch.from_numpy(nf)[None].to(device),
               torch.from_numpy(groups)[None].to(device), G)[0].float().cpu()
    pr = torch.softmax(lg / max(temperature, 1e-3), -1).numpy()           # (G, PLAN_DIM, NB)
    plan_g = (pr * centers[None]).sum(-1).astype(np.float32)
    return plan_g[groups], plan_g, groups


def load_plan(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    m = PlanNet(a["n_mels"], a["nf_dim"], a["hidden"], a["layers"], a["dropout"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, np.asarray(ck["centers"], dtype=np.float32), ck


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def make_sample(c: dict, s: int, L: int, k: int):
    n = c["n"]
    e = min(n, s + L)
    idx = np.arange(s, e)
    g = c["groups"][idx] - c["groups"][s]
    gvalid = np.ones(int(g.max()) + 1, dtype=bool)
    if e < n and c["groups"][e] == c["groups"][e - 1]:              # 창 끝에서 잘린 묶음은 손실에서 뺌
        gvalid[-1] = False
    tgt = to_bins(c["plan_g"][c["groups"][s]:c["groups"][s] + len(gvalid)])
    patch = extract_patches(c["mel"], c["frames"][idx], k)
    return patch, c["nf_base"][idx], g, tgt, gvalid


def collate_plan(items, L: int):
    B = len(items)
    M, P = items[0][0].shape[1:]
    D = items[0][1].shape[1]
    G = max(len(it[4]) for it in items)
    patch = np.zeros((B, L, M, P), np.float32)
    nf = np.zeros((B, L, D), np.float32)
    gidx = np.full((B, L), -1, np.int64)
    tgt = np.zeros((B, G, PLAN_DIM), np.int64)
    gv = np.zeros((B, G), bool)
    for b, (p, f, g, t, v) in enumerate(items):
        n = len(g)
        patch[b, :n], nf[b, :n], gidx[b, :n] = p, f, g
        tgt[b, :len(v)], gv[b, :len(v)] = t, v
    return patch, nf, gidx, tgt, gv, G


def sample_start(c: dict, rng) -> int:
    starts = np.flatnonzero(np.r_[True, np.diff(c["groups"]) != 0])     # 묶음 첫 노트에서 시작
    return int(starts[rng.integers(0, len(starts))])


def evaluate(model, data, device, L, k, temperature=0.5):
    model.eval()
    tot, n, corr = 0.0, 0, 0
    pred_all, true_all = [], []
    with torch.no_grad():
        for c in data.charts:
            s = 0
            while s < c["n"]:
                p, f, g, t, v = make_sample(c, s, L, k)
                patch, nf, gidx, tgt, gv, G = collate_plan([(p, f, g, t, v)], L)
                lg = model(torch.from_numpy(patch).to(device), torch.from_numpy(nf).to(device),
                           torch.from_numpy(gidx).to(device), G)
                m = torch.from_numpy(gv).to(device)
                tt = torch.from_numpy(tgt).to(device)
                tot += F.cross_entropy(lg[m].reshape(-1, NB), tt[m].reshape(-1), reduction="sum").item()
                corr += int((lg[m].argmax(-1) == tt[m]).sum().item())
                n += int(m.sum().item()) * PLAN_DIM
                pr = torch.softmax(lg[m].float() / temperature, -1).cpu().numpy()
                pred_all.append((pr * data.centers[None]).sum(-1))
                g0 = c["groups"][s]
                true_all.append(c["plan_g"][g0:g0 + len(v)][v])
                # 다음 창: 잘린 묶음의 첫 노트부터
                nxt = s + len(g)
                if nxt < c["n"] and not v[-1]:
                    nxt = int(np.searchsorted(c["groups"], c["groups"][s] + len(v) - 1))
                    if nxt <= s:
                        nxt = s + len(g)
                s = nxt
    P, T = np.concatenate(pred_all), np.concatenate(true_all)
    corr_f = [float(np.corrcoef(P[:, j], T[:, j])[0, 1]) if T[:, j].std() > 1e-6 and P[:, j].std() > 1e-6 else 0.0
              for j in range(PLAN_DIM)]
    return {"loss": tot / max(n, 1), "acc": corr / max(n, 1), "pred_mean": P.mean(0), "true_mean": T.mean(0),
            "corr": np.array(corr_f), "alt_hi_pred": float((P[:, 1] > 0.3).mean()),
            "alt_hi_true": float((T[:, 1] > 0.3).mean())}


def fmt_eval(r) -> str:
    s = f"손실 {r['loss']:.3f} | 칸 정확도 {r['acc']:.3f}\n"
    s += "    " + " ".join(f"{n:>7s}" for n in PLAN_NAMES) + "\n"
    s += "    예측" + " ".join(f"{x:7.2f}" for x in r["pred_mean"])[4:] + "  (평균)\n"
    s += "    정답" + " ".join(f"{x:7.2f}" for x in r["true_mean"])[4:] + "\n"
    s += "    상관" + " ".join(f"{x:7.2f}" for x in r["corr"])[4:] + "  (반 마디별로 맞히는 정도, 1에 가까울수록 좋음)\n"
    s += f"    묶음 교대(alt>0.3) 반 마디 비율: 예측 {r['alt_hi_pred']:.0%} / 정답 {r['alt_hi_true']:.0%}"
    return s


def main():
    from lane_v2 import LaneData
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--rhythm-dir", required=True)
    ap.add_argument("--patch-k", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=384, help="한 샘플의 노트 수 (약 50 반 마디)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--aug", type=float, default=0.0)
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--good", action="store_true")
    ap.add_argument("--good-weight", type=float, default=3.0)
    ap.add_argument("--good-list", default=None)
    ap.add_argument("--alt-max", type=float, default=None, help="train_lane2.py와 같음 (권장 0.05)")
    ap.add_argument("--alt-weight", type=float, default=0.2)
    ap.add_argument("--out", default="checkpoints/plan.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pats = None
    if args.good:
        from good_charts import load_patterns
        pats = load_patterns(args.good_list)
    kw = dict(rhythm_dir=args.rhythm_dir, good_patterns=pats, section=True, plan=True, alt_max=args.alt_max)
    train = LaneData(args.data, "train", args.patch_k, good_weight=args.good_weight if args.good else 1.0,
                     alt_weight=args.alt_weight, **kw)
    val = LaneData(args.data, "val", args.patch_k, good_weight=1.0, alt_weight=1.0, **kw)
    if not args.good:                                   # 표시 없이 학습하면 생성 때도 0이 맞지만, 기본은 --good 사용
        print("[주의] --good 없이 학습: 좋은 차분 표시 대신 곡 밀도가 들어가요")

    # 칸 대표값 = 학습 데이터에서 그 칸에 든 값들의 평균
    allg = np.concatenate([c["plan_g"] for c in train.charts])
    centers = default_centers().astype(np.float32)
    bins = to_bins(allg)
    for j in range(PLAN_DIM):
        for b in range(NB):
            sel = bins[:, j] == b
            if sel.sum() >= 5:
                centers[j, b] = allg[sel, j].mean()
    train.centers = val.centers = centers
    print("칸별 비율(학습 데이터): " + " | ".join(
        f"{n} " + "/".join(f"{(bins[:, j] == b).mean():.0%}" for b in range(NB)) for j, n in enumerate(PLAN_NAMES)))

    n_mels = train.charts[0]["mel"].shape[1]
    nf_dim = train.charts[0]["nf_base"].shape[1]
    model = PlanNet(n_mels, nf_dim, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    print(f"계획 모델 | 장치 {device} | 파라미터 {sum(p.numel() for p in model.parameters()):,}개 | "
          f"train {len(train.charts)} / val {len(val.charts)}")
    if args.aug > 0:
        from augment import augment_mel
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, run, t0 = float("inf"), 0.0, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        items = []
        for _ in range(args.batch):
            ci = int(rng.choice(len(train.charts), p=train.p))
            c = train.charts[ci]
            items.append(make_sample(c, sample_start(c, rng), args.seq_len, args.patch_k))
        patch, nf, gidx, tgt, gv, G = collate_plan(items, args.seq_len)
        patch = torch.from_numpy(patch).to(device)
        if args.aug > 0:
            patch = augment_mel(patch, args.aug, args.aug_strength)
        lg = model(patch, torch.from_numpy(nf).to(device), torch.from_numpy(gidx).to(device), G)
        m = torch.from_numpy(gv).to(device)
        if not bool(m.any()):
            continue
        loss = F.cross_entropy(lg[m].reshape(-1, NB), torch.from_numpy(tgt).to(device)[m].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        if step % args.log_every == 0:
            sec = (time.time() - t0) / step
            print(f"step {step:5d}/{args.steps} | loss {run / args.log_every:.4f} | {sec:.2f}초/스텝 | "
                  f"남은 시간 약 {sec * (args.steps - step) / 60:.1f}분", flush=True)
            run = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(model, val, device, args.seq_len, args.patch_k)
            print(f"  [평가 step {step}] {fmt_eval(r)}", flush=True)
            if r["loss"] < best:
                best = r["loss"]
                torch.save({"model": model.state_dict(), "centers": centers, "step": step,
                            "args": {**vars(args), "n_mels": n_mels, "nf_dim": nf_dim},
                            "metrics": {"loss": float(r["loss"]), "acc": float(r["acc"])}}, out)
                print(f"  -> 최고 기록 갱신, 저장: {out}", flush=True)
    print(f"\n완료: 최저 손실 {best:.3f} (총 {(time.time() - t0) / 60:.1f}분)")


def selftest():
    rng = np.random.default_rng(0)
    m = PlanNet(96, 17)
    B, L, G = 2, 40, 6
    patch = torch.rand(B, L, 96, 17)
    nf = torch.rand(B, L, 17)
    gidx = torch.from_numpy(np.sort(rng.integers(0, G, (B, L)), axis=1))
    gidx[1, 30:] = -1
    y = m(patch, nf, gidx, G)
    assert y.shape == (B, G, PLAN_DIM, NB)
    y.sum().backward()
    # 예측 (곡 하나)
    from lane_v2 import rhythm_codes_from_times
    fr = np.sort(rng.choice(3000, 300, replace=False))
    codes = rhythm_codes_from_times(fr / 100.0, 180)
    pn, pg, gr = predict_plan(m, default_centers().astype(np.float32), rng.random((3000, 96)).astype(np.float32),
                              fr, codes)
    assert pn.shape == (300, PLAN_DIM) and pg.shape[0] == gr.max() + 1 and np.isfinite(pn).all()
    # 학습 샘플 묶기
    c = {"n": 300, "frames": fr, "groups": gr, "plan_g": pg, "mel": rng.random((3000, 96)).astype(np.float32),
         "nf_base": rng.random((300, 17)).astype(np.float32)}
    items = [make_sample(c, sample_start(c, rng), 64, 8) for _ in range(3)]
    patch, nf, gidx, tgt, gv, G = collate_plan(items, 64)
    y = m(torch.from_numpy(patch), torch.from_numpy(nf), torch.from_numpy(gidx), G)
    assert y.shape[1] == G and tgt.shape == (3, G, PLAN_DIM)
    print(f"계획 모델 점검 OK | 파라미터 {sum(p.numel() for p in m.parameters()):,}개 | 반 마디 {G}개 예시")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest()
    else:
        main()
