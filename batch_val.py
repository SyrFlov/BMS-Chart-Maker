"""
[점검] val 곡으로 한 번에 생성하고 사람 패턴과 비교한다

사용:
    python batch_val.py --list                         # 목록만 보기
    python batch_val.py                                # 전부 (렌더링 -> 생성 -> 비교)
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import argparse
import csv
import os
import time
import shlex
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from bms_parser import parse_bms


def val_rows(data: Path):
    with open(data / "index.csv", newline="", encoding="utf-8-sig") as fp:
        return [r for r in csv.DictReader(fp) if r["split"] == "val"]


def chart_info(src: Path):
    c = parse_bms(src)
    return c, c.init_bpm, len(c.notes)


def alt_ratio(c) -> float:
    from plan_feats import alt_run_ratio
    d = defaultdict(int)
    for n in c.notes:
        if 0 <= n.lane <= 7:
            d[round(n.time, 4)] |= 1 << n.lane
    ts = sorted(d)
    return alt_run_ratio([d[t] for t in ts], np.rint(np.array(ts) * 100).astype(np.int64))


# ---------------------------------------------------------------------------
# 비교 (16분 칸 단위, 박 위치는 자동으로 맞춤)
# ---------------------------------------------------------------------------
def slots(path: Path):
    c = parse_bms(path)
    d = defaultdict(set)
    for n in c.notes:
        if 0 <= n.lane <= 7 and n.beat >= 0:
            d[int(round(n.beat * 4))].add(n.lane)
    return d


def compare(gen_path: Path, human_path: Path) -> dict:
    H, G0 = slots(human_path), slots(gen_path)
    hk = set(H)
    shift = max(range(-128, 129), key=lambda s: len(hk & {k + s for k in G0}))
    G = {k + shift: v for k, v in G0.items()}
    mx = max(max(H), max(G)) // 8 + 1
    hr = np.bincount([k // 8 for k in H if k >= 0], minlength=mx)[:mx]
    typ = np.where(hr <= 2, 0, np.where(hr <= 5, 1, 2))
    keys = lambda v: len([l for l in v if l > 0])
    r = {"gen_notes": sum(len(v) for v in G.values()), "human_notes": sum(len(v) for v in H.values()),
         "gen_rows": len(G), "human_rows": len(H),
         "overlap": len(set(G) & hk) / max(len(G), 1), "missed": len(hk - set(G)) / max(len(hk), 1)}
    for name, D in (("h", H), ("g", G)):
        for t in range(3):
            hs = set(np.flatnonzero(typ == t).tolist())
            n_h = max(len(hs), 1)
            rows = [v for k, v in D.items() if k // 8 in hs]
            r[f"{name}{t}_notes"] = sum(len(v) for v in rows) / n_h
            r[f"{name}{t}_rows"] = len(rows) / n_h
            r[f"{name}{t}_thick"] = np.mean([keys(v) >= 3 for v in rows]) if rows else 0.0
            r[f"{name}{t}_btw"] = sum(len(v) for k, v in D.items() if k // 8 in hs and k % 2 == 1) / n_h
        ks = sorted(D)
        runs, L = [], 1
        for a, b in zip(ks, ks[1:]):
            if b - a == 1:
                L += 1
            else:
                runs.append(L)
                L = 1
        runs.append(L)
        runs = np.array(runs)
        r[f"{name}_run"] = float(runs[runs >= 2].mean()) if (runs >= 2).any() else 0.0
        r[f"{name}_single"] = np.mean([keys(v) == 1 for v in D.values() if keys(v)])
    gr = np.bincount([k // 8 for k in G if 0 <= k // 8 < mx], minlength=mx)[:mx]
    gt = np.where(gr <= 2, 0, np.where(gr <= 5, 1, 2))
    for t in range(3):
        m = typ == t
        r[f"same{t}"] = float(np.mean(gt[m] == t)) if m.any() else float("nan")
    M = lambda D: np.bincount([k // 16 for k, v in D.items() if k >= 0 for _ in v], minlength=mx // 2 + 1)
    a, b = M(G), M(H)
    n = min(len(a), len(b))
    r["dens_corr"] = float(np.corrcoef(a[:n], b[:n])[0, 1])
    return r


def prob_analysis(rows_csv: Path, gen_path: Path, human_path: Path) -> dict | None:
    """생성 줄의 노트 시각 모델 확신도가 '사람도 찍은 줄'과 '생성에만 있는 줄'에서 얼마나 다른지, 문턱을 올리면 어떻게 되는지."""
    if not rows_csv.exists():
        return None
    arr = np.genfromtxt(rows_csv, delimiter=",", names=True)
    if arr.size == 0:
        return None
    H, G0 = slots(human_path), slots(gen_path)
    hk = set(H)
    shift = max(range(-128, 129), key=lambda s: len(hk & {k + s for k in G0}))
    sl = np.atleast_1d(arr["slot16"]).astype(np.int64) + shift
    pr = np.atleast_1d(arr["prob"]).astype(np.float64)
    mx = max(max(H), int(sl.max())) // 8 + 1
    hr = np.bincount([k // 8 for k in H if k >= 0], minlength=mx)[:mx]
    typ = np.where(hr <= 2, 0, np.where(hr <= 5, 1, 2))
    hit = np.array([k in hk for k in sl])
    st = typ[np.clip(sl // 8, 0, mx - 1)]
    out = {"p_hit": np.percentile(pr[hit], [25, 50, 75]) if hit.any() else [0, 0, 0],
           "p_extra": np.percentile(pr[~hit], [25, 50, 75]) if (~hit).any() else [0, 0, 0]}
    for t in range(3):
        m = st == t
        out[f"p_extra{t}"] = float(np.median(pr[m & ~hit])) if (m & ~hit).any() else float("nan")
        out[f"p_hit{t}"] = float(np.median(pr[m & hit])) if (m & hit).any() else float("nan")
    sweep = []
    n_rest = max(int((typ == 0).sum()), 1)
    for thr in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        k = pr >= thr
        sweep.append((thr, int(k.sum()), float(hit[k].mean()) if k.any() else 0.0,
                      1 - len(hk & set(sl[k].tolist())) / max(len(hk), 1),
                      float((k & (st == 0)).sum()) / n_rest))
    out["sweep"] = sweep
    return out


def prob_report(pa: dict) -> str:
    q = lambda v: "/".join(f"{x:.2f}" for x in v)
    L = [f"  1단계 확신도(25/50/75%): 사람도 찍은 줄 {q(pa['p_hit'])} | 생성에만 있는 줄 {q(pa['p_extra'])}",
         "  구간별 중간값(사람도 찍음 / 생성에만): " + ", ".join(
             f"{TN[t]} {pa[f'p_hit{t}']:.2f}/{pa[f'p_extra{t}']:.2f}" for t in range(3)),
         "  문턱별: " + " | ".join(f"{thr:.2f}: 줄 {n}, 겹침 {pr_:.0%}, 놓침 {ms:.0%}, 쉼 구간 줄/반마디 {rr:.1f}"
                                  for thr, n, pr_, ms, rr in pa["sweep"])]
    return "\n".join(L)


TN = ["쉼", "8비트", "16비트"]
KEY_LINES = ("밀도 계획", "예측한 노트", "구간 정리", "격자 정리", "패턴 계획", "스크래치:", "저장:", "오류", "Error", "Traceback")


def report(name: str, r: dict) -> str:
    L = [f"== {name}",
         f"  노트 생성 {r['gen_notes']} / 사람 {r['human_notes']} | 줄 {r['gen_rows']} / {r['human_rows']} | "
         f"생성 줄 중 사람과 겹침 {r['overlap']:.0%}, 사람 줄 중 놓침 {r['missed']:.0%} | 마디별 노트 수 상관 {r['dens_corr']:.2f}",
         f"  단타 줄 {r['g_single']:.0%} (사람 {r['h_single']:.0%}) | 16분 연속 평균 {r['g_run']:.1f}줄 (사람 {r['h_run']:.1f})"]
    for t in range(3):
        L.append(f"  사람 {TN[t]:4s} 구간: 생성도 {TN[t]}으로 찍음 {r[f'same{t}']:.0%} | 반마디당 노트 {r[f'g{t}_notes']:.1f} (사람 {r[f'h{t}_notes']:.1f})"
                 f" | 줄 {r[f'g{t}_rows']:.1f} ({r[f'h{t}_rows']:.1f}) | 3개+ 동시치기 {r[f'g{t}_thick']:.0%} ({r[f'h{t}_thick']:.0%})"
                 f" | 16분 사이 노트 {r[f'g{t}_btw']:.1f} ({r[f'h{t}_btw']:.1f})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--audio-dir", default="input/val")
    ap.add_argument("--out", default="out_val")
    ap.add_argument("--ckpt", default="checkpoints/onset_kaggle.pt")
    ap.add_argument("--lane-ckpt", default="checkpoints/lane_v4.pt")
    ap.add_argument("--plan-ckpt", default="checkpoints/plan.pt")
    ap.add_argument("--density-ckpt", default="checkpoints/density.pt", help="있으면 자동으로 사용 (없으면 생략)")
    ap.add_argument("--extra", default="--jack-penalty 3", help="infer.py에 덧붙일 옵션 (따옴표로 묶기)")
    ap.add_argument("--only", default=None, help="쉼표로 구분한 글자. chart_id/파일 이름에 들어간 곡만")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 이 개수만")
    ap.add_argument("--skip-alt", action="store_true", help="묶음 교대(1357/246 등)가 5%% 넘는 곡 빼기")
    ap.add_argument("--list", action="store_true", help="목록만 출력")
    ap.add_argument("--force-render", action="store_true", help="오디오가 이미 있어도 다시 만들기")
    ap.add_argument("--quiet", action="store_true", help="infer.py 출력은 중요한 줄만 보여 주기")
    ap.add_argument("--compare-only", action="store_true", help="렌더링/생성 없이 비교만")
    args = ap.parse_args()

    rows = val_rows(Path(args.data))
    if args.only:
        pats = [p.strip().lower() for p in args.only.split(",") if p.strip()]
        rows = [r for r in rows if any(p in (r["chart_id"] + r["source"]).lower() for p in pats)]
    items = []
    print(f"{'번호':>4s}  {'chart_id':40s} {'BPM':>6s} {'노트':>5s} {'교대':>5s}  원본")
    for i, r in enumerate(rows):
        src = Path(r["source"])
        if not src.exists():
            print(f"{i:4d}  {r['chart_id'][:40]:40s}  원본 없음: {src}")
            continue
        c, bpm, nn = chart_info(src)
        alt = alt_ratio(c)
        tm = getattr(c, "timing", None)
        varbpm = tm is not None and len({round(x, 2) for x in tm.bpms}) > 1
        skip = (args.skip_alt and alt > 0.05) or varbpm
        why = "  (건너뜀: 변속곡, 생성은 고정 BPM만)" if varbpm else "  (건너뜀: 교대 많음)" if skip else ""
        print(f"{i:4d}  {r['chart_id'][:40]:40s} {bpm:6.1f} {nn:5d} {alt:5.0%}  {src.name}{why}")
        if not skip:
            items.append((r["chart_id"], src, c, bpm))
    if args.limit:
        items = items[:args.limit]
    if args.list:
        return
    print(f"\n대상 {len(items)}곡")

    audio_dir, out_root = Path(args.audio_dir), Path(args.out)
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    from infer import safe_name
    if not args.compare_only:
        from render_audio import render_chart, write_wav
        for k, (cid, src, c, bpm) in enumerate(items, 1):
            wav = audio_dir / f"{cid}.wav"
            od = out_root / safe_name(cid)                 # infer.py가 만드는 폴더와 같게
            od.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, od / f"human{src.suffix}")      # 사람 패턴 (비교용)
            if args.force_render or not wav.exists():
                print(f"\n[{k}/{len(items)}] 오디오 렌더링: {cid}")
                audio, st = render_chart(c, src.parent)
                write_wav(wav, audio, 44100)
                print(f"  저장 {wav} (키음 누락 {st['missing']}, 읽기 실패 {st['failed']})")
            print(f"[{k}/{len(items)}] 생성: {cid} (BPM {bpm:g})")
            cmd = [_sys.executable, "infer.py", "--audio", str(wav), "--ckpt", args.ckpt,
                   "--lane-ckpt", args.lane_ckpt, "--bpm", f"{bpm:g}", "--snap", "beat", "--out", str(out_root)]
            if args.plan_ckpt and Path(args.plan_ckpt).exists():
                cmd += ["--plan-ckpt", args.plan_ckpt]
            if args.density_ckpt and Path(args.density_ckpt).exists():
                cmd += ["--density-ckpt", args.density_ckpt]
            cmd += shlex.split(args.extra, posix=False)
            t0 = time.time()
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", env=env)
            log = []
            for line in p.stdout:                       # infer.py 출력을 그대로 보여 주면서 저장
                log.append(line)
                if not args.quiet or any(w in line for w in KEY_LINES):
                    print("    " + line, end="")
            rc = p.wait()
            (od / "infer_log.txt").write_text("".join(log), encoding="utf-8")
            if rc != 0:
                print(f"  실패! {od / 'infer_log.txt'} 확인\n" + "".join(log[-8:]))
            else:
                print(f"  완료 ({time.time() - t0:.0f}초) | 전체 진행 {k}/{len(items)}")

    # 비교 (infer.py는 파일 이름의 특수문자를 바꿔서 폴더를 만든다)
    lines, allr = [], []
    for cid, src, c, bpm in items:
        od = out_root / safe_name(cid)
        gen, human = od / f"{safe_name(cid)}.bms", od / f"human{src.suffix}"
        if not gen.exists() or not human.exists():
            lines.append(f"== {cid}\n  결과 없음 ({gen})")
            continue
        r = compare(gen, human)
        allr.append(r)
        lines.append(report(f"{cid} (BPM {bpm:g})", r))
        pa = prob_analysis(od / f"{safe_name(cid)}_rows.csv", gen, human)
        if pa is not None:
            lines.append(prob_report(pa))
    if allr:
        avg = {k: float(np.nanmean([r[k] for r in allr])) for k in allr[0]}
        for k in ("gen_notes", "human_notes", "gen_rows", "human_rows"):
            avg[k] = int(round(avg[k]))
        lines.append(report(f"전체 평균 ({len(allr)}곡)", avg))
    text = "\n".join(lines)
    print("\n" + text)
    (out_root / "compare.txt").write_text(text, encoding="utf-8")
    print(f"\n저장: {out_root / 'compare.txt'}  <- 이 파일을 보내 주세요")


if __name__ == "__main__":
    main()
