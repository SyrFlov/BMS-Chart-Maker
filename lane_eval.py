"""
[점검] 레인 모델이 멜(소리)을 보고 패턴을 고르는지 확인 (검증 패턴, 학습에 안 쓴 곡)

사용:
    python lane_eval.py --data data/processed --rhythm-dir data/rhythm --ckpt checkpoints/lane_v5.pt
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
import collections
import time

import numpy as np
import torch
import torch.nn.functional as F

from lane_utils import N_LANES, POPCOUNT, sample_mask
from lane_v2 import ACC_DIM, LaneData, LaneNetV2, accent_features, batch_history, collate

SHIFT = torch.arange(N_LANES)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def stream_flags(frames):
    d = np.diff(frames)
    return np.r_[False, d <= 20] | np.r_[d <= 20, False]


@torch.no_grad()
def score_model(model, data, device, L, batch, max_chord):
    """줄마다 (손실 nats, 맞힘, 건반 수 맞힘, 스트림 여부)"""
    model.eval()
    rng = np.random.default_rng(0)
    out = []
    buf = []

    def flush(buf):
        b = collate(buf)
        patch = torch.from_numpy(b["patch"]).to(device)
        nf = torch.from_numpy(b["nf"]).to(device)
        masks = torch.from_numpy(b["masks"]).to(device)
        valid = torch.from_numpy(b["valid"]).to(device)
        hist = torch.from_numpy(batch_history(b["masks"], b["frames"], b["init_prev"], b["init_last"])).to(device)
        lg = model(patch, nf, hist)
        tb = ((masks.unsqueeze(-1) >> SHIFT.to(device)) & 1).float()
        nll = F.binary_cross_entropy_with_logits(lg, tb, reduction="none").sum(-1)
        lgv = lg[valid].float().cpu().numpy()
        tm = masks[valid].cpu().numpy()
        pm = np.array([sample_mask(r, 0.0, max_chord, rng) for r in lgv])
        st = np.concatenate([w["stream"][w["valid"]] for w in buf])
        out.append(np.stack([nll[valid].cpu().numpy(), (pm == tm), POPCOUNT[pm] == POPCOUNT[tm], st], 1))

    for w in windows_with_stream(data, L):
        buf.append(w)
        if len(buf) == batch:
            flush(buf); buf = []
    if buf:
        flush(buf)
    return np.concatenate(out)


def windows_with_stream(data, L):
    for ci, c in enumerate(data.charts):
        st = stream_flags(c["frames"])
        for s in range(0, c["n"], L):
            w = data.make(ci, s, L)
            sv = np.zeros(L, dtype=bool)
            idx = np.arange(s, min(s + L, c["n"]))
            sv[: len(idx)] = st[idx]
            w["stream"] = sv
            yield w


def swap_mel(data, mode, shift_frames, accent):
    """멜을 바꾼 복사본 조건을 만든다. 원래 값은 반환해서 되돌린다."""
    saved = [(c["mel"], c["nf"]) for c in data.charts]
    n = len(data.charts)
    for i, c in enumerate(data.charts):
        need = int(c["frames"].max()) + 64
        if mode == "other":
            song = c["chart_id"].split("__")[0]
            j = next(((i + n // 2 + d) % n for d in range(n)
                      if data.charts[(i + n // 2 + d) % n]["chart_id"].split("__")[0] != song), i)   # 다른 곡
            src = np.asarray(saved[j][0])
            idx = np.arange(need) % len(src)
        else:                                               # "shift": 같은 곡을 시간만 민다
            src = np.asarray(saved[i][0])
            idx = (np.arange(need) + shift_frames) % len(src)
        new = np.ascontiguousarray(src[idx])
        c["mel"] = new
        if accent:
            nf = c["nf"].copy()
            nf[:, -ACC_DIM:] = accent_features(new, c["frames"])
            c["nf"] = nf
    return saved


def restore(data, saved):
    for c, (m, nf) in zip(data.charts, saved):
        c["mel"], c["nf"] = m, nf


def ngram_scores(train, val):
    """학습 패턴으로 빈도/n-gram을 세고 검증 패턴 줄마다 (손실, 맞힘, 건반 수 맞힘) 2종"""
    uni = collections.Counter()
    bi = collections.defaultdict(collections.Counter)
    tri = collections.defaultdict(collections.Counter)
    for c in train.charts:
        m = [int(x) for x in c["masks"]]
        p1 = p2 = 0
        for x in m:
            uni[x] += 1; bi[p1][x] += 1; tri[(p2, p1)][x] += 1
            p2, p1 = p1, x
    V = 256
    tot = sum(uni.values())
    pu = np.array([(uni[x] + 0.5) / (tot + 0.5 * V) for x in range(V)])
    best_u = int(np.argmax([uni[x] if x else -1 for x in range(V)]))

    def dist(p2, p1):
        b, t = bi.get(p1), tri.get((p2, p1))
        pb = pu.copy()
        if b:
            nb = sum(b.values()); lam = nb / (nb + 20.0)
            v = np.zeros(V); v[list(b)] = list(b.values()); pb = lam * v / nb + (1 - lam) * pu
        pt = pb
        if t:
            nt = sum(t.values()); lam = nt / (nt + 10.0)
            v = np.zeros(V); v[list(t)] = list(t.values()); pt = lam * v / nt + (1 - lam) * pb
        return pt

    ru, rn = [], []
    for c in val.charts:
        m = [int(x) for x in c["masks"]]
        st = stream_flags(c["frames"])
        p1 = p2 = 0
        for x, s in zip(m, st):
            ru.append((-np.log(pu[x]), x == best_u, POPCOUNT[best_u] == POPCOUNT[x], s))
            d = dist(p2, p1)
            d0 = d.copy(); d0[0] = 0
            g = int(np.argmax(d0))
            rn.append((-np.log(max(d[x], 1e-12)), g == x, POPCOUNT[g] == POPCOUNT[x], s))
            p2, p1 = p1, x
    return np.array(ru, dtype=np.float64), np.array(rn, dtype=np.float64)


def report(name, r):
    r = np.asarray(r, dtype=np.float64)
    s = r[:, 3] > 0
    say(f"  {name:26s} 손실 {r[:, 0].mean():.3f} | 정확도 {r[:, 1].mean():.1%} | 건반 수 맞힘 {r[:, 2].mean():.1%} || "
        f"연타 구간만: 손실 {r[s, 0].mean():.3f}, 정확도 {r[s, 1].mean():.1%}")
    return r[:, 0].mean(), r[s, 0].mean() if s.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--rhythm-dir", default="data/rhythm")
    ap.add_argument("--ckpt", default="checkpoints/lane_v5.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shift-sec", type=float, default=3.7)
    ap.add_argument("--out", default="lane_eval.txt")
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    pats = None
    if a.get("good"):
        from good_charts import load_patterns
        pats = load_patterns(a.get("good_list"))
    kw = dict(rhythm_dir=args.rhythm_dir if a.get("rhythm_dir") else None, good_patterns=pats, good_weight=1.0,
              section=a.get("section", False), plan=a.get("plan", False), alt_max=a.get("alt_max"),
              alt_weight=1.0, accent=a.get("accent", False))
    val = LaneData(args.data, args.split, a["patch_k"], **kw)
    train = LaneData(args.data, "train", a["patch_k"], **kw)
    n_mels = val.charts[0]["mel"].shape[1]
    model = LaneNetV2(n_mels, 2 * a["patch_k"] + 1, a["hidden"], a["layers"], a["dropout"],
                      note_dim=a["note_dim"]).to(device)
    model.load_state_dict(ck["model"])
    L = a.get("seq_len", 256)
    say(f"레인 모델 {args.ckpt} (step {ck.get('step')}) | {args.split} 차분 {len(val.charts)}개, "
        f"줄 {sum(c['n'] for c in val.charts):,}개 | 장치 {device}")
    say("조건별 결과 (손실 = 줄마다 평균, 낮을수록 좋음):")

    res = {}
    res["real"] = report("모델 (진짜 멜)", score_model(model, val, device, L, args.batch, a.get("max_chord", 6)))
    say(f"    ...{(time.time() - t0) / 60:.1f}분")
    sv = swap_mel(val, "other", 0, a.get("accent", False))
    res["other"] = report("모델 (다른 곡 멜)", score_model(model, val, device, L, args.batch, a.get("max_chord", 6)))
    restore(val, sv)
    say(f"    ...{(time.time() - t0) / 60:.1f}분")
    sv = swap_mel(val, "shift", int(args.shift_sec * 100), a.get("accent", False))
    res["shift"] = report(f"모델 (멜 {args.shift_sec:g}초 밀기)", score_model(model, val, device, L, args.batch,
                                                                       a.get("max_chord", 6)))
    restore(val, sv)
    ru, rn = ngram_scores(train, val)
    res["uni"] = report("단순 기준 1 (흔한 모양)", ru)
    res["ngram"] = report("단순 기준 2 (앞 1~2줄)", rn)

    say("\n판단:")
    d_other = res["other"][0] - res["real"][0]
    d_shift = res["shift"][0] - res["real"][0]
    d_ng = res["ngram"][0] - res["real"][0]
    say(f"  멜을 다른 곡으로 바꾸면 손실 +{d_other:.3f} ({d_other / res['real'][0]:+.1%}), "
        f"시간만 밀면 +{d_shift:.3f} ({d_shift / res['real'][0]:+.1%})")
    if d_other / res["real"][0] > 0.05:
        say("  -> 소리를 보고 패턴을 고른다. 멜-패턴 학습이 의미 있음")
    elif d_other / res["real"][0] > 0.015:
        say("  -> 소리를 조금 본다. 대부분은 이력/박자/계획 입력으로 찍음")
    else:
        say("  -> 소리를 거의 안 본다. 이력/박자/계획만으로 찍는 셈")
    say(f"  단순 기준 2(앞 1~2줄) 대비 모델 손실 {-d_ng:+.3f} 낮음 ({d_ng / res['ngram'][0]:.1%})"
        " (계획 입력이 정답 값이라 모델 쪽에 유리한 비교)")
    say(f"\n총 {(time.time() - t0) / 60:.1f}분")
    with open(args.out, "w", encoding="utf-8") as fp:
        fp.write("\n".join(LOG) + "\n")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
