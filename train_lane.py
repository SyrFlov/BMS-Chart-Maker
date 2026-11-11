"""
[학습] 레인 모델 학습

사용:
    python train_lane.py --overfit --steps 600
    python train_lane.py --steps 20000 --batch 16 --eval-every 1000
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment import augment_mel
from lane_dataset import LaneSequences
from lane_model import LaneNet, make_step_fn
from lane_utils import N_LANES, chord_stats, extract_patches, generate_masks, sample_mask


LANE_SHIFT = torch.arange(N_LANES)


def target_bits(target: torch.Tensor) -> torch.Tensor:
    """target (B, L) = 마스크 - 1 (패딩 -100) -> (B, L, 8) 0/1 float. 패딩 자리는 0 (valid로 가려서 쓴다)."""
    mask = (target.clamp(min=0) + 1)
    return ((mask.unsqueeze(-1) >> LANE_SHIFT.to(target.device)) & 1).float()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, ds: LaneSequences, device: str, batch_size: int, patch_k: int,
             max_chord: int, free_run: bool = True) -> dict:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_sum, n, correct = 0.0, 0, 0
    tp, fp, fn = np.zeros(N_LANES), np.zeros(N_LANES), np.zeros(N_LANES)
    rng = np.random.default_rng(0)
    for b in loader:
        logits = model(b["patch"].to(device), b["aux"].to(device))
        tgt = b["target"].to(device)
        valid = tgt >= 0
        bits = target_bits(tgt)
        loss_sum += F.binary_cross_entropy_with_logits(logits[valid], bits[valid], reduction="sum").item()
        lg = logits[valid].float().cpu().numpy()                    # (n, 8)
        tm = (tgt[valid] + 1).cpu().numpy()
        pm = np.array([sample_mask(row, 0.0, max_chord, rng) for row in lg])   # 생성 때와 같은 방식으로 마스크 결정
        correct += int((pm == tm).sum())
        n += len(tm)
        for lane in range(N_LANES):
            pb, tb = (pm >> lane) & 1, (tm >> lane) & 1
            tp[lane] += (pb & tb).sum()
            fp[lane] += (pb & (1 - tb)).sum()
            fn[lane] += ((1 - pb) & tb).sum()

    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    res = {"loss": loss_sum / max(n * N_LANES, 1), "acc": correct / max(n, 1), "f1_lane": f1, "n": n}

    if free_run:                                    # 정답을 안 보여 주고 스스로 이어 만들기 (가장 확률 높은 것만 선택)
        gen_all, true_all, exact = [], [], []
        for c in ds.charts:
            patches = extract_patches(c["mel"], c["frames"], patch_k)
            step_fn = make_step_fn(model, patches, device)
            g = generate_masks(step_fn, c["frames"], c["cond"], temperature=0.0, max_chord=max_chord)
            exact.append((g == c["masks"]).mean())
            gen_all.append(g)
            true_all.append(c["masks"])
        g, t = np.concatenate(gen_all), np.concatenate(true_all)
        res.update(free_exact=float(np.mean(exact)), gen=chord_stats(g), true=chord_stats(t))
    return res


def fmt(res: dict) -> str:
    s = (f"손실 {res['loss']:.3f} | 정확도 {res['acc']:.3f} | "
         f"레인별 F1 {' '.join(f'{x:.2f}' for x in res['f1_lane'])}")
    if "gen" in res:
        s += (f"\n    자유 생성: 정답과 같은 노트 {res['free_exact'] * 100:.0f}% | "
              f"평균 동시치기 {res['gen']['mean_chord']:.2f} (정답 {res['true']['mean_chord']:.2f}) | "
              f"같은 레인 반복률 {res['gen']['repeat_rate']:.2f} (정답 {res['true']['repeat_rate']:.2f})")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--overfit", action="store_true", help="1곡 과적합 테스트")
    ap.add_argument("--chart-id", default=None)
    ap.add_argument("--seq-len", type=int, default=64, help="한 샘플에 들어가는 연속 노트 수")
    ap.add_argument("--patch-k", type=int, default=8, help="노트 주변 멜 조각 반경(프레임)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-chord", type=int, default=6, help="평가용 자유 생성의 동시치기 상한")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/lane_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", type=float, default=0.0,
                    help="mp3 대비 증강 확률 (0=끔, 권장 0.8). 학습 배치에만 적용")
    ap.add_argument("--aug-strength", type=float, default=1.0, help="증강 세기 배율")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.overfit:
        probe = LaneSequences(args.data, split="train", seq_len=args.seq_len, random=False,
                              patch_k=args.patch_k)
        ids = [c["chart_id"] for c in probe.charts]
        cid = args.chart_id or ids[0]
        if cid not in ids:
            raise SystemExit(f"chart_id '{cid}'를 찾을 수 없어요. 사용 가능한 값: {ids}")
        train_ds = LaneSequences(args.data, split="train", seq_len=args.seq_len, random=True,
                                 epoch_size=args.batch * 50, chart_ids=[cid], patch_k=args.patch_k)
        eval_ds = LaneSequences(args.data, split="train", seq_len=args.seq_len, random=False,
                                chart_ids=[cid], patch_k=args.patch_k)
        print(f"[과적합 테스트] 차분: {cid} (노트 시각 {eval_ds.charts[0]['n']}개)")
    else:
        train_ds = LaneSequences(args.data, split="train", seq_len=args.seq_len, random=True,
                                 epoch_size=args.batch * 50, patch_k=args.patch_k)
        try:
            eval_ds = LaneSequences(args.data, split="val", seq_len=args.seq_len, random=False,
                                    patch_k=args.patch_k)
        except ValueError:
            raise SystemExit("val 곡이 없어요. 곡을 더 모으거나, 코드 점검이면 --overfit 옵션을 쓰세요.")
        print(f"train 차분 {len(train_ds.charts)}개 / val 차분 {len(eval_ds.charts)}개")

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        drop_last=True)
    batches = cycle(loader)

    n_mels = train_ds.charts[0]["mel"].shape[1]
    model = LaneNet(n_mels=n_mels, patch_w=2 * args.patch_k + 1, hidden=args.hidden, layers=args.layers,
                    dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    print(f"장치 {device} | 파라미터 {sum(p.numel() for p in model.parameters()):,}개 | "
          f"샘플 = 노트 {args.seq_len}개 x 배치 {args.batch}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_acc, run_loss, t0 = float("inf"), 0.0, 0.0, time.time()

    for step in range(1, args.steps + 1):
        model.train()
        b = next(batches)
        patch = b["patch"].to(device)
        if args.aug > 0:
            patch = augment_mel(patch, args.aug, args.aug_strength)
        logits = model(patch, b["aux"].to(device))
        tgt = b["target"].to(device)
        valid = tgt >= 0
        loss = F.binary_cross_entropy_with_logits(logits[valid], target_bits(tgt)[valid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        run_loss += loss.item()
        if step % args.log_every == 0:
            sec = (time.time() - t0) / step
            print(f"step {step:5d}/{args.steps} | loss {run_loss / args.log_every:.4f} | "
                  f"{sec:.2f}초/스텝 | 남은 시간 약 {sec * (args.steps - step) / 60:.1f}분")
            run_loss = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(model, eval_ds, device, args.batch, args.patch_k, args.max_chord)
            print(f"  [평가 step {step}] {fmt(res)}")
            best_acc = max(best_acc, res["acc"])
            if res["loss"] < best_loss:
                best_loss = res["loss"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "data_cfg": train_ds.cfg, "step": step,
                            "metrics": {"loss": res["loss"], "acc": res["acc"]}}, out_path)
                print(f"  -> 최고 기록 갱신, 저장: {out_path}")

    print(f"\n완료: 최고 정확도 {best_acc:.3f}, 최저 손실 {best_loss:.3f} (총 {(time.time() - t0) / 60:.1f}분)")
    if args.overfit:
        print("과적합 테스트: " + ("통과 - 모델이 이 곡의 레인 배치를 학습할 수 있어요" if best_acc >= 0.9
                              else "미달 - --steps를 늘려 보거나 결과를 알려 주세요"))


if __name__ == "__main__":
    main()
