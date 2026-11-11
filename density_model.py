"""
[모델, 선택 기능] 반 마디 밀도 모델: 곡 전체 소리를 보고 반 마디(2박)마다 사람이 몇 줄을 찍을지 정한다

사용:
    python density_model.py --data <processed> --grid-dir <grid> --good --steps 3000 --aug 0.8 --out density.pt
    python density_model.py --selftest
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NR, NB = 9, 5          # 줄 수 칸: 0..7, 8+ / 16분 사이 줄 칸: 0..3, 4+
K = 32                 # 반 마디를 32칸으로 줄여서 봄


def gather_halves(mel: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """mel (T, M), 반 마디 [start, end) -> (H, K, M)"""
    T = len(mel)
    s = np.clip(start, 0, T - 1)[:, None].astype(np.float64)
    e = np.clip(end, 1, T)[:, None].astype(np.float64)
    pos = s + (e - s) * (np.arange(K)[None, :] + 0.5) / K
    idx = np.clip(np.floor(pos).astype(np.int64), 0, T - 1)
    return np.asarray(mel[idx.reshape(-1)], dtype=np.float32).reshape(len(start), K, mel.shape[1])


class DensNet(nn.Module):
    def __init__(self, n_mels: int = 96, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(3 * n_mels + 3, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head_r = nn.Linear(2 * hidden, NR)
        self.head_b = nn.Linear(2 * hidden, NB)

    def forward(self, seg: torch.Tensor, extra: torch.Tensor):
        """seg (B, H, K, M), extra (B, H, 3) -> 로짓 (B, H, NR), (B, H, NB)"""
        flux = F.relu(seg[:, :, 1:] - seg[:, :, :-1]).mean(2)
        x = torch.cat([seg.mean(2), seg.amax(2), flux * 4.0, extra], -1)
        y, _ = self.gru(self.inp(x))
        return self.head_r(y), self.head_b(y)


def extra_feats(half: np.ndarray, start: np.ndarray, end: np.ndarray, quality: float) -> np.ndarray:
    return np.stack([half.astype(np.float32), np.full(len(half), quality, np.float32),
                     ((end - start) / 100.0).astype(np.float32)], 1)


def count_from_probs(p: np.ndarray, q: float) -> np.ndarray:
    """칸별 확률 (H, C) -> 누적 확률이 q를 넘는 가장 작은 칸 (중앙값이면 q=0.5)"""
    return np.argmax(np.cumsum(p, 1) >= q - 1e-9, axis=1)


@torch.no_grad()
def predict_density(model, mel, start, end, half, device="cpu", quality: float = 1.0):
    model.eval()
    seg = torch.from_numpy(gather_halves(mel, start, end))[None].to(device)
    ex = torch.from_numpy(extra_feats(half, start, end, quality))[None].to(device)
    lr, lb = model(seg, ex)
    return torch.softmax(lr[0].float(), -1).cpu().numpy(), torch.softmax(lb[0].float(), -1).cpu().numpy()


def load_density(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    m = DensNet(a["n_mels"], a["hidden"], a["layers"], a["dropout"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def load_split(data: Path, grid_dir: Path, split: str, good_patterns, good_weight: float):
    with open(data / "index.csv", newline="", encoding="utf-8-sig") as fp:
        rows = [r for r in csv.DictReader(fp) if r["split"] == split]
    good = set()
    if good_patterns is not None:
        from good_charts import report
        good = report(rows, good_patterns, split)
    charts, miss = [], 0
    for r in rows:
        gp = grid_dir / f"{r['chart_id']}.grid.npz"
        if not gp.exists():
            miss += 1
            continue
        g = dict(np.load(gp))
        if len(g["rows"]) < 8:
            continue
        g["mel"] = np.load(data / f"{r['chart_id']}.mel.npy", mmap_mode="r")
        g["good"] = r["chart_id"] in good
        g["id"] = r["chart_id"]
        charts.append(g)
    if miss:
        print(f"[{split}] grid 파일이 없는 차분 {miss}개는 뺐어요")
    w = np.array([len(c["rows"]) * (good_weight if c["good"] else 1.0) for c in charts], np.float64)
    return charts, w / w.sum()


def make_batch(charts, p, rng, B: int, L: int, quality_on: bool):
    segs, exs, tr, tb, mask = [], [], [], [], []
    for _ in range(B):
        c = charts[int(rng.choice(len(charts), p=p))]
        H = len(c["rows"])
        s = int(rng.integers(0, max(H - L, 0) + 1))
        sl = slice(s, min(H, s + L))
        n = sl.stop - sl.start
        seg = np.zeros((L, K, c["mel"].shape[1]), np.float32)
        seg[:n] = gather_halves(c["mel"], c["start"][sl], c["end"][sl])
        ex = np.zeros((L, 3), np.float32)
        ex[:n] = extra_feats(c["half"][sl], c["start"][sl], c["end"][sl], float(c["good"]) if quality_on else 1.0)
        r = np.zeros(L, np.int64); b = np.zeros(L, np.int64); m = np.zeros(L, bool)
        r[:n] = np.minimum(c["rows"][sl], NR - 1); b[:n] = np.minimum(c["btw"][sl], NB - 1); m[:n] = True
        segs.append(seg); exs.append(ex); tr.append(r); tb.append(b); mask.append(m)
    return (np.stack(segs), np.stack(exs), np.stack(tr), np.stack(tb), np.stack(mask))


def sect(r):
    return np.where(r <= 2, 0, np.where(r <= 5, 1, 2))


@torch.no_grad()
def evaluate(model, charts, device, quality_on: bool, q: float = 0.5):
    model.eval()
    allp, allt, corr, ce = [], [], [], []
    for c in charts:
        pr, pb = predict_density(model, c["mel"], c["start"], c["end"], c["half"], device,
                                 float(c["good"]) if quality_on else 1.0)
        t = np.minimum(c["rows"], NR - 1)
        ce.append(-np.log(pr[np.arange(len(t)), t] + 1e-9).mean())
        n = count_from_probs(pr, q)
        allp.append(n); allt.append(t)
        if t.std() > 0 and n.std() > 0:
            corr.append(np.corrcoef(n, t)[0, 1])
    P, T = np.concatenate(allp), np.concatenate(allt)
    st_p, st_t = sect(P), sect(T)
    conf = np.array([[np.mean(st_p[st_t == a] == b) if (st_t == a).any() else 0 for b in range(3)] for a in range(3)])
    return {"ce": float(np.mean(ce)), "mae": float(np.abs(P - T).mean()), "corr": float(np.mean(corr)),
            "conf": conf, "pred_rows": float(P.mean()), "true_rows": float(T.mean())}


def fmt(r) -> str:
    c = r["conf"]
    return (f"손실 {r['ce']:.3f} | 줄 수 오차 {r['mae']:.2f} | 곡별 상관 {r['corr']:.2f} | 반마디당 줄 예측 {r['pred_rows']:.2f} (정답 {r['true_rows']:.2f})\n"
            f"    사람 쉼 -> 예측 쉼/8비트/16비트 {c[0,0]:.0%}/{c[0,1]:.0%}/{c[0,2]:.0%} | "
            f"사람 8비트 -> {c[1,0]:.0%}/{c[1,1]:.0%}/{c[1,2]:.0%} | 사람 16비트 -> {c[2,0]:.0%}/{c[2,1]:.0%}/{c[2,2]:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-halves", type=int, default=64, help="한 샘플의 반 마디 수")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--aug", type=float, default=0.0)
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--good", action="store_true", help="좋은 차분 표시 입력 + 가중 (생성 때는 항상 1)")
    ap.add_argument("--good-weight", type=float, default=3.0)
    ap.add_argument("--good-list", default=None)
    ap.add_argument("--out", default="checkpoints/density.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pats = None
    if args.good:
        from good_charts import load_patterns
        pats = load_patterns(args.good_list)
    data, gd = Path(args.data), Path(args.grid_dir)
    train, p = load_split(data, gd, "train", pats, args.good_weight if args.good else 1.0)
    val, _ = load_split(data, gd, "val", pats, 1.0)
    allr = np.concatenate([c["rows"] for c in train])
    print(f"train {len(train)}곡 / val {len(val)}곡 | 반 마디 {len(allr)}개 | "
          f"쉼 {np.mean(allr <= 2):.0%}, 8비트 {np.mean((allr >= 3) & (allr <= 5)):.0%}, 16비트 {np.mean(allr >= 6):.0%}")
    n_mels = train[0]["mel"].shape[1]
    model = DensNet(n_mels, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    if args.aug > 0:
        from augment import augment_mel
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"밀도 모델 | 장치 {device} | 파라미터 {sum(q.numel() for q in model.parameters()):,}개")
    r0 = evaluate(model, val, device, args.good)
    print(f"  [학습 전] {fmt(r0)}")
    best, run, t0 = float("inf"), 0.0, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        seg, ex, tr, tb, m = make_batch(train, p, rng, args.batch, args.seq_halves, args.good)
        seg = torch.from_numpy(seg).to(device)
        if args.aug > 0:                               # (B, H, K, M) -> (B, H, M, K)로 바꿔 증강 후 되돌림
            seg = augment_mel(seg.transpose(2, 3), args.aug, args.aug_strength).transpose(2, 3)
        lr_, lb_ = model(seg, torch.from_numpy(ex).to(device))
        mm = torch.from_numpy(m).to(device)
        loss = (F.cross_entropy(lr_[mm], torch.from_numpy(tr).to(device)[mm])
                + 0.5 * F.cross_entropy(lb_[mm], torch.from_numpy(tb).to(device)[mm]))
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
            r = evaluate(model, val, device, args.good)
            print(f"  [평가 step {step}] {fmt(r)}", flush=True)
            if r["ce"] < best:
                best = r["ce"]
                torch.save({"model": model.state_dict(), "step": step,
                            "args": {**vars(args), "n_mels": n_mels},
                            "metrics": {"ce": r["ce"], "mae": r["mae"], "corr": r["corr"]}}, out)
                print(f"  -> 최고 기록 갱신, 저장: {out}", flush=True)
    print(f"\n완료: 최저 손실 {best:.3f} (총 {(time.time() - t0) / 60:.1f}분)")


def selftest():
    rng = np.random.default_rng(0)
    mel = rng.random((3000, 96)).astype(np.float32)
    st = np.arange(0, 2800, 50); en = st + 50
    seg = gather_halves(mel, st, en)
    assert seg.shape == (len(st), K, 96)
    m = DensNet(96)
    pr, pb = predict_density(m, mel, st, en, (np.arange(len(st)) % 2).astype(np.int8))
    assert pr.shape == (len(st), NR) and pb.shape == (len(st), NB) and np.allclose(pr.sum(1), 1, atol=1e-4)
    n = count_from_probs(np.array([[0.1, 0.3, 0.2, 0.4], [0.6, 0.2, 0.1, 0.1]]), 0.5)
    assert list(n) == [2, 0]
    charts = [{"mel": mel, "start": st, "end": en, "half": (np.arange(len(st)) % 2).astype(np.int8),
               "rows": rng.integers(0, 9, len(st)).astype(np.int16), "btw": rng.integers(0, 5, len(st)).astype(np.int16),
               "good": True}]
    b = make_batch(charts, np.array([1.0]), rng, 2, 16, True)
    lr_, lb_ = m(torch.from_numpy(b[0]), torch.from_numpy(b[1]))
    assert lr_.shape == (2, 16, NR)
    (lr_.sum() + lb_.sum()).backward()
    print(f"밀도 모델 점검 OK | 파라미터 {sum(q.numel() for q in m.parameters()):,}개")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest()
    else:
        main()
