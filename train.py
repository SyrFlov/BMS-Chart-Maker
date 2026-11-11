"""
[학습] 노트 시각 모델 학습

사용:
    python train.py --overfit --window-sec 8 --hidden 64 --steps 300 --batch 4
    python train.py --window-sec 20 --hidden 128 --steps 20000 --batch 16 --eval-every 1000
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
from dataset import ChartWindows
from metrics import best_by, evaluate_charts
from model import N_DENS, build_onset, count_params, dens_target

THRESHOLDS = np.arange(0.05, 0.96, 0.05)


# ---------------------------------------------------------------------------
# 타깃 / 손실
# ---------------------------------------------------------------------------
def to_target(label: torch.Tensor, mode: str) -> torch.Tensor:
    """label (B, 8, T) -> 타이밍 모드면 (B, 1, T) = 어느 레인이든 노트가 있으면 1"""
    return label.amax(dim=1, keepdim=True) if mode == "timing" else label


def widen(target: torch.Tensor) -> torch.Tensor:
    """정답 프레임의 양옆 1프레임(±10ms)을 0.5로 번지게 한다 (정확히 1프레임만 맞히라는 요구를 완화)."""
    c = target.shape[1]
    k = target.new_tensor([0.5, 1.0, 0.5]).view(1, 1, 3).repeat(c, 1, 1)
    return F.conv1d(target, k, padding=1, groups=c).clamp(max=1.0)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_charts(model, ds: ChartWindows, device: str, mode: str, batch_size: int):
    """패턴마다 윈도우 예측을 이어 붙여 (C, T) 확률/정답을 만든다 (윈도우 경계에서 노트가 잘리지 않게)."""
    model.eval()
    frames_of = {r["chart_id"]: int(r["frames"]) for r in ds.rows}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for b in loader:
        p = torch.sigmoid(model(b["mel"].to(device))).float().cpu().numpy()
        lab = b["label"].numpy()
        if mode == "timing":
            lab = lab.max(axis=1, keepdims=True)
        for k, cid in enumerate(b["chart_id"]):
            T = frames_of[cid]
            if cid not in probs:
                probs[cid] = np.zeros((p.shape[1], T), dtype=np.float32)
                labels[cid] = np.zeros((lab.shape[1], T), dtype=np.uint8)
            s = int(b["start_frame"][k])
            n = min(ds.W, T - s)                      # 마지막 윈도우의 0 패딩 구간은 버림
            probs[cid][:, s:s + n] = p[k, :, :n]
            labels[cid][:, s:s + n] = lab[k, :, :n]
    return probs, labels


def run_eval(model, ds, device, mode, batch_size) -> tuple[dict, list[dict]]:
    probs, labels = predict_charts(model, ds, device, mode, batch_size)
    sweep = evaluate_charts(probs, labels, THRESHOLDS, tols=(5, 2))
    return best_by(sweep, "union_f1_t5"), sweep


def fmt_eval(b: dict) -> str:
    ratio = b["union_pred"] / max(b["union_true"], 1)
    return (f"임계값 {b['thr']:.2f} | F1@50ms {b['union_f1_t5']:.3f} "
            f"(P {b['union_p_t5']:.3f} R {b['union_r_t5']:.3f}) | F1@20ms {b['union_f1_t2']:.3f} | "
            f"예측/정답 노트 수 {b['union_pred']}/{b['union_true']} (x{ratio:.2f})")


# ---------------------------------------------------------------------------
def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--target", choices=["timing", "lanes"], default="timing")
    ap.add_argument("--overfit", action="store_true", help="1곡 과적합 테스트 (train 차분 하나로 학습+평가)")
    ap.add_argument("--chart-id", default=None, help="--overfit에서 쓸 차분 chart_id (생략하면 첫 번째)")
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=4.0, help="노트(양성) 프레임 손실 가중치")
    ap.add_argument("--no-widen", action="store_true", help="정답 번지기(±1프레임) 끄기")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0, help="Windows에서는 0 권장")
    ap.add_argument("--no-amp", action="store_true", help="GPU에서 혼합정밀도 끄기")
    ap.add_argument("--out", default="checkpoints/onset_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", type=float, default=0.0,
                    help="mp3 대비 증강 확률 (0=끔, 권장 0.8). 학습 배치에만 적용, 평가는 원본")
    ap.add_argument("--aug-strength", type=float, default=1.0, help="증강 세기 배율")
    ap.add_argument("--good-weight", type=float, default=1.0,
                    help="v3: 좋은 차분(good_charts.py)을 이 배율만큼 더 자주 뽑기 (1 = 끔, 권장 3)")
    ap.add_argument("--ctx", action="store_true",
                    help="v6: 긴 문맥 가지를 붙인 모델(CtxOnsetNet). 주변 1초 노트 수를 보조로 배워 쉬는 구간을 덜 찍게 한다")
    ap.add_argument("--ctx-block", type=int, default=25, help="문맥 가지의 블록 길이(프레임, 25 = 0.25초)")
    ap.add_argument("--aux-weight", type=float, default=0.3, help="문맥 가지 보조 손실 가중치")
    ap.add_argument("--good-list", default=None, help="좋은 차분 이름을 덧붙일 텍스트 파일 (한 줄에 하나)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp
    n_out = 1 if args.target == "timing" else 8

    # ---- 데이터 ----
    if args.overfit:
        probe = ChartWindows(args.data, split="train", window_sec=args.window_sec, random=False)
        ids = [r["chart_id"] for r in probe.rows]
        cid = args.chart_id or ids[0]
        if cid not in ids:
            raise SystemExit(f"chart_id '{cid}'를 찾을 수 없어요. 사용 가능한 값: {ids}")
        train_ds = ChartWindows(args.data, split="train", window_sec=args.window_sec, random=True,
                                epoch_size=args.batch * 50, chart_ids=[cid])
        eval_ds = ChartWindows(args.data, split="train", window_sec=args.window_sec, random=False,
                               chart_ids=[cid])
        print(f"[과적합 테스트] 차분: {cid}")
    else:
        pats = None
        if args.good_weight != 1.0:
            from good_charts import load_patterns
            pats = load_patterns(args.good_list)
        train_ds = ChartWindows(args.data, split="train", window_sec=args.window_sec, random=True,
                                epoch_size=args.batch * 50, good_patterns=pats, good_weight=args.good_weight)
        try:
            eval_ds = ChartWindows(args.data, split="val", window_sec=args.window_sec, random=False)
        except ValueError:
            raise SystemExit("val 곡이 없어요. 곡을 더 모아서 build_dataset.py를 다시 돌리거나, "
                             "코드 점검이면 --overfit 옵션을 쓰세요.")
        print(f"train 차분 {len(train_ds.rows)}개 / val 차분 {len(eval_ds.rows)}개 "
              f"(곡 단위로 분할됨)")

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        drop_last=True)
    batches = cycle(loader)

    # ---- 모델 ----
    model = build_onset(vars(args), train_ds.cfg["n_mels"], n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:                      # torch 2.3 미만
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    pos_weight = torch.tensor(args.pos_weight, device=device)
    print(f"장치 {device}{' (AMP)' if use_amp else ''} | 목표 {args.target} | "
          f"파라미터 {count_params(model):,}개 | 윈도우 {args.window_sec:g}초 x 배치 {args.batch}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1, run_loss, run_aux, t0 = -1.0, 0.0, 0.0, time.time()

    for step in range(1, args.steps + 1):
        model.train()
        batch = next(batches)
        mel = batch["mel"].to(device)
        if args.aug > 0:
            mel = augment_mel(mel, args.aug, args.aug_strength)
        target = to_target(batch["label"].to(device), args.target)
        if not args.no_widen:
            target = widen(target)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            if args.ctx:
                logits, aux = model(mel, return_aux=True)
            else:
                logits = model(mel)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pos_weight)
        if args.ctx:
            onset = batch["label"].to(device).amax(dim=1)                    # (B, T) 레인 무관 노트 프레임
            dt = dens_target(onset, args.ctx_block)
            aux_loss = F.cross_entropy(aux.float().reshape(-1, N_DENS), dt.reshape(-1))
            run_aux += aux_loss.item()
            loss = loss + args.aux_weight * aux_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        run_loss += loss.item()
        if step % args.log_every == 0:
            sec = (time.time() - t0) / step
            print(f"step {step:5d}/{args.steps} | loss {run_loss / args.log_every:.4f} | "
                  + (f"문맥 {run_aux / args.log_every:.3f} | " if args.ctx else "") +
                  f"{sec:.2f}초/스텝 | 남은 시간 약 {sec * (args.steps - step) / 60:.1f}분")
            run_loss = run_aux = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            best, _ = run_eval(model, eval_ds, device, args.target, args.batch)
            print(f"  [평가 step {step}] {fmt_eval(best)}")
            if best["union_f1_t5"] > best_f1:
                best_f1 = best["union_f1_t5"]
                torch.save({"model": model.state_dict(), "args": vars(args), "threshold": best["thr"],
                            "data_cfg": train_ds.cfg, "metrics": best, "step": step}, out_path)
                print(f"  -> 최고 기록 갱신, 저장: {out_path}")

    print(f"\n완료: 최고 F1@50ms = {best_f1:.3f}  (총 {(time.time() - t0) / 60:.1f}분)")
    if args.overfit:
        verdict = "통과 - 모델이 이 곡을 학습할 수 있어요" if best_f1 >= 0.9 else \
            "미달 - 스텝을 늘리거나(--steps) 결과를 알려 주세요"
        print(f"과적합 테스트: {verdict}")


if __name__ == "__main__":
    main()
