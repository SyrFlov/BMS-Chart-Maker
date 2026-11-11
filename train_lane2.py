"""
[학습] 레인 모델 v2 학습 (lane_v2.py)

사용:
    python train_lane2.py --data /kaggle/working/processed --steps 6000 --aug 0.8 --out /kaggle/working/lane_v2.pt
    python train_lane2.py --data ... --rhythm-dir ... --good --good-weight 3 --section --steps 4000 --aug 0.8 --out .../lane_v3.pt
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
import torch.nn.functional as F

from augment import augment_mel
from lane_utils import N_LANES, POPCOUNT, sample_mask
from lane_v2 import (LaneData, LaneNetV2, batch_history, collate, flow_consistency, generate_v2)

SHIFT = torch.arange(N_LANES)
# 좌우 반전: 건반 1~7을 7~1로 (스크래치 0번은 그대로). 계획/구조/박자/어택 입력은 좌우와 무관해서 그대로 둔다
MIRROR = np.array([(m & 1) | sum(1 << (8 - k) for k in range(1, 8) if (m >> k) & 1) for m in range(256)], dtype=np.int64)
LANE_REV = np.array([0, 7, 6, 5, 4, 3, 2, 1])


def mirror_batch(b: dict, rng: np.random.Generator, p: float) -> int:
    """배치의 샘플마다 p 확률로 좌우 반전 (masks, 직전 마스크, 레인별 마지막 사용 시각). 반전한 샘플 수"""
    flip = rng.random(len(b["masks"])) < p
    for i in np.flatnonzero(flip):
        b["masks"][i] = MIRROR[b["masks"][i]]
        b["init_prev"][i] = MIRROR[b["init_prev"][i]]
        b["init_last"][i] = b["init_last"][i][LANE_REV]
    return int(flip.sum())


def bits_t(masks: torch.Tensor) -> torch.Tensor:
    return ((masks.unsqueeze(-1) >> SHIFT.to(masks.device)) & 1).float()


def to_t(b: dict, device: str):
    return (torch.from_numpy(b["patch"]).to(device), torch.from_numpy(b["nf"]).to(device),
            torch.from_numpy(b["masks"]).to(device), torch.from_numpy(b["valid"]).to(device))


@torch.no_grad()
def evaluate(model, data: LaneData, device: str, L: int, batch: int, free_run: bool, args) -> dict:
    model.eval()
    loss_sum, n, correct = 0.0, 0, 0
    tp, fp, fn = np.zeros(N_LANES), np.zeros(N_LANES), np.zeros(N_LANES)
    rng = np.random.default_rng(0)
    buf = []

    def flush(buf):
        nonlocal loss_sum, n, correct
        b = collate(buf)
        patch, nf, masks, valid = to_t(b, device)
        hist = torch.from_numpy(batch_history(b["masks"], b["frames"], b["init_prev"], b["init_last"])).to(device)
        logits = model(patch, nf, hist)
        tb = bits_t(masks)
        loss_sum += F.binary_cross_entropy_with_logits(logits[valid], tb[valid], reduction="sum").item()
        lg = logits[valid].float().cpu().numpy()
        tm = masks[valid].cpu().numpy()
        pm = np.array([sample_mask(r, 0.0, args.max_chord, rng) for r in lg])
        correct += int((pm == tm).sum())
        n += len(tm)
        for lane in range(N_LANES):
            pb, t_ = (pm >> lane) & 1, (tm >> lane) & 1
            tp[lane] += (pb & t_).sum(); fp[lane] += (pb & (1 - t_)).sum(); fn[lane] += ((1 - pb) & t_).sum()

    for w in data.windows(L):
        buf.append(w)
        if len(buf) == batch:
            flush(buf); buf = []
    if buf:
        flush(buf)
    res = {"loss": loss_sum / max(n * N_LANES, 1), "acc": correct / max(n, 1),
           "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1)}

    if free_run:
        g_all, t_all, fm, fs, tm_, ts_ = [], [], [], [], [], []
        sd_g, sd_t = [], []
        ss_g, ss_t = [], []
        ar_g, ar_t = [], []
        for c in data.charts:
            g = generate_v2(model, c["mel"], c["frames"], c["cond"], args.patch_k, device,
                            temperature=args.eval_temperature, max_chord=args.max_chord,
                            no_jack_ms=args.eval_no_jack_ms, seed=0, win=L, rhythm_codes=c.get("codes"),
                            lone_scratch=True, plan_feats=c.get("plan_n"))
            if c.get("codes") is not None:                       # 스크래치가 마디 첫 박에 오는 비율
                down = c["codes"] == 0
                sd_g.append(((g & 1).astype(bool) & down).sum() / max(1, (g & 1).sum()))
                sd_t.append(((c["masks"] & 1).astype(bool) & down).sum() / max(1, (c["masks"] & 1).sum()))
            fr = c["frames"]                                       # 스트림(0.2초 이내로 이어진 줄) 단타 비율
            dtf = np.diff(fr)
            st = np.r_[False, dtf <= 20] | np.r_[dtf <= 20, False]
            keys = 0xFE
            kg, kt = g[st] & keys, c["masks"][st] & keys
            if (kg > 0).any() and (kt > 0).any():
                ss_g.append(float((POPCOUNT[kg[kg > 0]] == 1).mean()))
                ss_t.append(float((POPCOUNT[kt[kt > 0]] == 1).mean()))
            ar_g.append(alt_run_rows(g, fr)); ar_t.append(alt_run_rows(c["masks"], fr))
            fg, ft = flow_consistency(g), flow_consistency(c["masks"])
            if not np.isnan(fg["move"]):
                fm.append(fg["move"]); fs.append(fg["spread"]); tm_.append(ft["move"]); ts_.append(ft["spread"])
            g_all.append(g); t_all.append(c["masks"])
        g, t = np.concatenate(g_all), np.concatenate(t_all)
        res.update(flow_gen=(np.mean(fm), np.mean(fs)), flow_true=(np.mean(tm_), np.mean(ts_)),
                   chord_gen=float(POPCOUNT[g].mean()), chord_true=float(POPCOUNT[t].mean()),
                   sc_gen=float((g & 1).mean()), sc_true=float((t & 1).mean()))
        res.update(alt_gen=float(np.mean(ar_g)), alt_true=float(np.mean(ar_t)))
        if ss_g:
            res.update(ss_gen=float(np.mean(ss_g)), ss_true=float(np.mean(ss_t)))
        if sd_g:
            res.update(scd_gen=float(np.mean(sd_g)), scd_true=float(np.mean(sd_t)))
    return res


def alt_run_rows(masks: np.ndarray, frames: np.ndarray) -> float:
    """묶음 교대(1357/246, 1245/367 등 A-B-A-B)가 4줄 이상 이어진 줄의 비율"""
    from plan_feats import alt_run_ratio
    return alt_run_ratio(masks, frames)


def fmt(r: dict) -> str:
    s = f"손실 {r['loss']:.3f} | 정확도 {r['acc']:.3f} | 레인별 F1 {' '.join(f'{x:.2f}' for x in r['f1'])}"
    if "flow_gen" in r:
        s += (f"\n    자유 생성(온도 {ARGS.eval_temperature:g}): 흐름 move {r['flow_gen'][0]:+.2f} spread {r['flow_gen'][1]:+.2f} "
              f"(정답 {r['flow_true'][0]:+.2f} / {r['flow_true'][1]:+.2f}) | "
              f"평균 동시치기 {r['chord_gen']:.2f} (정답 {r['chord_true']:.2f})"
              f"\n    스크래치 줄 비율 {r['sc_gen']:.1%} (정답 {r['sc_true']:.1%})")
        if "scd_gen" in r:
            s += f" | 스크래치 중 마디 첫 박 {r['scd_gen']:.0%} (정답 {r['scd_true']:.0%})"
        if "alt_gen" in r:
            s += f"\n    묶음 교대(1357/246, 1245/367 등)가 4줄 이상 이어진 줄 비율 {r['alt_gen']:.1%} (정답 {r['alt_true']:.1%})"
        if "ss_gen" in r:
            s += f"\n    스트림 단타 비율 {r['ss_gen']:.0%} (정답 {r['ss_true']:.0%}, 좋은 차분은 보통 10~40%)"
    return s


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--seq-len", type=int, default=256, help="한 샘플의 연속 노트 수 (v1은 64)")
    ap.add_argument("--patch-k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ss-max", type=float, default=0.25,
                    help="스케줄드 샘플링 최대 비율: 이전 노트 이력 중 이만큼을 모델이 뽑은 결과로 바꿔서 학습")
    ap.add_argument("--ss-ramp", type=float, default=0.5, help="전체 스텝 중 이 비율 동안 0에서 최대까지 늘림")
    ap.add_argument("--max-chord", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--free-every", type=int, default=1, help="평가 n번에 한 번 자유 생성 흐름 측정 (시간이 걸림)")
    ap.add_argument("--eval-temperature", type=float, default=1.0)
    ap.add_argument("--eval-no-jack-ms", type=float, default=100.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--aug", type=float, default=0.0)
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--rhythm-dir", default=None,
                    help="build_rhythm.py로 만든 .rhy.npy 폴더. 주면 박자 위치(마디 몇째 박, 16분 위치)를 입력으로 씀")
    ap.add_argument("--good", action="store_true",
                    help="v3: 곡 밀도 조건 대신 '좋은 차분' 표시를 넣고 좋은 차분을 더 자주 뽑는다 (good_charts.py)")
    ap.add_argument("--good-weight", type=float, default=3.0, help="좋은 차분을 뽑는 빈도 배율")
    ap.add_argument("--good-list", default=None, help="좋은 차분 이름을 덧붙일 텍스트 파일 (한 줄에 하나)")
    ap.add_argument("--section", action="store_true", help="v3: 곡 구조 특징(쉼/구간 밀도) 입력 추가")
    ap.add_argument("--plan", action="store_true",
                    help="v4: 반 마디 패턴 계획(plan_feats.py) 입력 추가. 학습은 사람 차분 값, 생성은 plan_model.py가 정함")
    ap.add_argument("--accent", action="store_true",
                    help="v5: 노트마다 대역별(저/중저/중고/고) 어택 세기 상대값 4개를 입력에 추가 (--plan 필요)")
    ap.add_argument("--alt-max", type=float, default=None,
                    help="묶음 교대(1357/246 등)가 4줄 이상 이어진 줄이 이 비율보다 많은 차분은 덜 뽑음 (권장 0.05)")
    ap.add_argument("--alt-weight", type=float, default=0.2, help="--alt-max를 넘는 차분을 뽑는 빈도 배율")
    ap.add_argument("--mirror", type=float, default=0.0,
                    help="학습 샘플을 이 확률로 좌우 반전(건반 1~7 -> 7~1). 데이터가 사실상 두 배. 권장 0.5. "
                         "단점: 높은 소리를 오른쪽에 두는 식의 소리-위치 습관은 약해짐")
    ap.add_argument("--out", default="checkpoints/lane_v2.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ARGS = args

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = args.seq_len

    pats = None
    if args.good:
        from good_charts import load_patterns
        pats = load_patterns(args.good_list)
    train = LaneData(args.data, "train", args.patch_k, rhythm_dir=args.rhythm_dir,
                     good_patterns=pats, good_weight=args.good_weight if args.good else 1.0, section=args.section,
                     plan=args.plan, alt_max=args.alt_max, alt_weight=args.alt_weight, accent=args.accent)
    val = LaneData(args.data, "val", args.patch_k, rhythm_dir=args.rhythm_dir,
                   good_patterns=pats, good_weight=1.0, section=args.section, plan=args.plan,
                   alt_max=args.alt_max, alt_weight=1.0, accent=args.accent)
    print(f"train 차분 {len(train.charts)}개 / val 차분 {len(val.charts)}개 | "
          f"박자 위치 입력 {'사용' if args.rhythm_dir else '안 씀'} | 좋은 차분 표시 {'사용' if args.good else '안 씀'} | "
          f"곡 구조 입력 {'사용' if args.section else '안 씀'} | 패턴 계획 입력 {'사용' if args.plan else '안 씀'} | "
          f"대역별 어택 입력 {'사용' if args.accent else '안 씀'}")
    n_mels = train.charts[0]["mel"].shape[1]
    model = LaneNetV2(n_mels, 2 * args.patch_k + 1, args.hidden, args.layers, args.dropout,
                      note_dim=train.nf_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    print(f"장치 {device} | 파라미터 {sum(p.numel() for p in model.parameters()):,}개 | "
          f"샘플 = 노트 {L}개 x 배치 {args.batch} | 스케줄드 샘플링 최대 {args.ss_max:.0%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, run, t0, n_eval = float("inf"), 0.0, time.time(), 0

    for step in range(1, args.steps + 1):
        model.train()
        b = collate([train.sample(rng, L) for _ in range(args.batch)])
        if args.mirror > 0:
            mirror_batch(b, rng, args.mirror)
        patch, nf, masks, valid = to_t(b, device)
        if args.aug > 0:
            patch = augment_mel(patch, args.aug, args.aug_strength)
        hist_np = batch_history(b["masks"], b["frames"], b["init_prev"], b["init_last"])

        p_ss = args.ss_max * min(1.0, step / max(1.0, args.ss_ramp * args.steps))
        if p_ss > 0:
            # 1차: 정답 이력으로 예측 -> 레인마다 확률대로 뽑은 마스크로 이력 일부를 바꾼다
            with torch.no_grad():
                lg = model(patch, nf, torch.from_numpy(hist_np).to(device))
                pr = torch.sigmoid(lg.float())
                on = (torch.rand_like(pr) < pr)
                none = ~on.any(-1)
                if none.any():
                    on[none] = F.one_hot(pr[none].argmax(-1), N_LANES).bool()
                sm = (on.long() << SHIFT.to(device)).sum(-1).cpu().numpy()
            swap = (rng.random(sm.shape) < p_ss) & b["valid"]
            mixed = np.where(swap, sm, b["masks"])
            hist_np = batch_history(mixed, b["frames"], b["init_prev"], b["init_last"])

        logits = model(patch, nf, torch.from_numpy(hist_np).to(device))
        loss = F.binary_cross_entropy_with_logits(logits[valid], bits_t(masks)[valid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        run += loss.item()
        if step % args.log_every == 0:
            sec = (time.time() - t0) / step
            print(f"step {step:5d}/{args.steps} | loss {run / args.log_every:.4f} | 샘플링 {p_ss:.0%} | "
                  f"{sec:.2f}초/스텝 | 남은 시간 약 {sec * (args.steps - step) / 60:.1f}분", flush=True)
            run = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            n_eval += 1
            free = (n_eval % args.free_every == 0) or step == args.steps
            r = evaluate(model, val, device, L, args.batch, free, args)
            print(f"  [평가 step {step}] {fmt(r)}", flush=True)
            if r["loss"] < best:
                best = r["loss"]
                torch.save({"model": model.state_dict(), "args": {**vars(args), "arch": "v2", "note_dim": train.nf_dim,
                                                                 "quality": bool(args.good), "plan": bool(args.plan)},
                            "data_cfg": train.cfg, "step": step,
                            "metrics": {"loss": float(r["loss"]), "acc": float(r["acc"]),
                                        **({"flow_gen": [float(x) for x in r["flow_gen"]]} if "flow_gen" in r else {})}}, out)
                print(f"  -> 최고 기록 갱신, 저장: {out}", flush=True)

    print(f"\n완료: 최저 손실 {best:.3f} (총 {(time.time() - t0) / 60:.1f}분)")


ARGS = None

if __name__ == "__main__":
    main()
