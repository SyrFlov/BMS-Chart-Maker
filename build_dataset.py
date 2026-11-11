"""
[학습 준비, 생성에서는 멜 변환에 씀] BMS 곡 폴더들 -> 학습용 (멜 스펙트로그램, 노트 레이블) 쌍으로 변환

사용:
    python build_dataset.py --src data\\bms --out data\\processed
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from bms_parser import parse_bms
from render_audio import render_chart
from scan_bms import CHART_EXTS, analyze

# ---- 오디오/스펙트로그램 설정 ----
SR = 32000
HOP = 320                # 10ms
N_FFT = 2048             # 64ms
N_MELS = 96
FMIN, FMAX = 30.0, 16000.0
DB_MIN, DB_MAX = -80.0, 10.0   # 이 범위를 0~1로 스케일
FPS = SR // HOP          # 100
N_LANES = 8


# ---------------------------------------------------------------------------
# 멜 스펙트로그램 (numpy만 사용)
# ---------------------------------------------------------------------------
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    n_bins = N_FFT // 2 + 1
    freqs = np.linspace(0, SR / 2, n_bins)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))
    fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for i in range(N_MELS):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (ce - lo)
        down = (hi - freqs) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


_FB = mel_filterbank()
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)   # periodic hann


def log_mel(audio: np.ndarray) -> np.ndarray:
    """mono float32 -> (프레임, N_MELS) float32, 0~1 스케일. 프레임 t의 중심 = t*HOP 샘플."""
    pad = N_FFT // 2
    x = np.pad(audio.astype(np.float32), (pad, pad), mode="reflect")
    n_frames = 1 + len(audio) // HOP
    norm = _WIN.sum() / 2.0                      # 진폭 1 사인파 -> 크기 ~1
    out = np.empty((n_frames, N_MELS), dtype=np.float32)
    chunk = 1024                                 # 메모리 절약용 청크 처리
    offs = np.arange(N_FFT)[None, :]
    for s in range(0, n_frames, chunk):
        e = min(s + chunk, n_frames)
        frames = x[np.arange(s, e)[:, None] * HOP + offs] * _WIN
        mag = np.abs(np.fft.rfft(frames, axis=1)) / norm
        mel = (mag ** 2) @ _FB.T
        out[s:e] = 10.0 * np.log10(np.maximum(mel, 1e-10))
    return np.clip((out - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 레이블
# ---------------------------------------------------------------------------
def make_labels(notes, n_frames: int) -> np.ndarray:
    """노트 시작 시각 -> (프레임, 8) 이진 행렬. 롱노트도 시작점만 표시."""
    lab = np.zeros((n_frames, N_LANES), dtype=np.uint8)
    for n in notes:
        f = int(round(n.time * FPS))
        if 0 <= f < n_frames:
            lab[f, n.lane] = 1
    return lab


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------
def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")[:40] or "x"


def split_of(song_key: str, val_ratio: float, test_ratio: float) -> str:
    """곡(폴더) 단위로 고정된 분할. 같은 곡의 패턴은 항상 같은 쪽에 들어간다."""
    h = int(hashlib.md5(song_key.encode("utf-8")).hexdigest(), 16) % 10000 / 10000.0
    if h < test_ratio:
        return "test"
    if h < test_ratio + val_ratio:
        return "val"
    return "train"


def base_title(title: str) -> str:
    """패턴 이름 꼬리를 떼고 곡 이름만: 'Song -Sub-' / 'Song [SA]' / 'song (ANOTHER)' -> 'song'"""
    t = title or ""
    for _ in range(3):
        t = re.sub(r"\s*(\[[^\]]*\]|\([^)]*\)|<[^>]*>|-[^-]+-|～[^～]+～|~[^~]+~)\s*$", "", t.strip())
    return re.sub(r"\s+", " ", t).strip().lower() or (title or "").strip().lower()


def process_chart(f: Path) -> dict:
    chart = parse_bms(f)
    audio, stats = render_chart(chart, f.parent, sr=SR)
    mono = audio.mean(axis=1)
    mel = log_mel(mono)
    lab = make_labels(chart.notes, len(mel))
    return {"mel": mel, "lab": lab, "stats": stats}


def _one(f: Path, src: Path, out_dir: Path, group_by: str, overwrite: bool):
    """패턴 하나: 분석 -> (필요하면) 렌더링/멜/정답 저장. 여러 프로세스에서 돌 수 있게 바깥 상태를 안 건드림."""
    try:
        info = analyze(f)
    except Exception as e:  # noqa: BLE001
        return ("fail", str(f), f.name, repr(e))
    if not info["keep"]:
        return ("skip", str(f), f.name, info["reason"])
    rel_song = str(f.parent.relative_to(src))
    if group_by == "title":
        rel_song = "title:" + base_title(info["title"])
        song_id = f"{safe_name(base_title(info['title']))}_{hashlib.md5(rel_song.encode('utf-8')).hexdigest()[:6]}"
    else:
        song_id = f"{safe_name(f.parent.name)}_{hashlib.md5(rel_song.encode('utf-8')).hexdigest()[:6]}"
    chart_id = f"{song_id}__{safe_name(f.stem)}"
    if group_by == "title":           # 다른 폴더에 같은 파일 이름이 있어도 겹치지 않게
        chart_id += "_" + hashlib.md5(str(f.parent.relative_to(src)).encode("utf-8")).hexdigest()[:4]
    mel_path = out_dir / f"{chart_id}.mel.npy"
    lab_path = out_dir / f"{chart_id}.label.npy"
    try:
        if overwrite or not (mel_path.exists() and lab_path.exists()):
            r = process_chart(f)
            np.save(mel_path, r["mel"].astype(np.float16))
            np.save(lab_path, r["lab"])
            lab = r["lab"]
        else:
            lab = np.load(lab_path)
    except Exception as e:  # noqa: BLE001
        return ("fail", str(f), f.name, repr(e))
    row = {"chart_id": chart_id, "song_id": song_id, "split": "",
           "title": info["title"], "level": info["level"],
           "notes": info["notes"], "density": info["density"],
           "duration": info["duration"], "frames": len(lab), "source": str(f)}
    return ("ok", str(f), row, rel_song, len(lab), lab.sum(axis=0, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/bms", help="곡 폴더들이 들어 있는 폴더")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="동시에 처리할 프로세스 수 (예: 4~6, CPU 코어 수보다 적게). 키음 렌더링이 오래 걸려서 크게 빨라짐. "
                         "메모리가 모자라면 줄이세요")
    ap.add_argument("--group-by", choices=["folder", "title"], default="folder",
                    help="같은 곡으로 묶는 기준. 같은 곡 차분을 다른 폴더에 나눠 넣었으면 title "
                         "(곡 이름의 [SA] -ANOTHER- 같은 꼬리를 떼고 비교). 같은 곡이 train/val에 나뉘면 검증이 의미 없어짐")
    args = ap.parse_args()

    src, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "sr": SR, "hop": HOP, "n_fft": N_FFT, "n_mels": N_MELS, "fps": FPS,
        "fmin": FMIN, "fmax": FMAX, "db_min": DB_MIN, "db_max": DB_MAX,
        "n_lanes": N_LANES,
        "lanes": "0=scratch, 1..7=keys",
    }, indent=2), encoding="utf-8")

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in CHART_EXTS)
    rows, skipped, failed = [], [], []
    pos_count = np.zeros(N_LANES, dtype=np.int64)
    total_frames = 0

    jobs = [(f, src, out_dir, args.group_by, args.overwrite) for f in files]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"차분 {len(files)}개를 {args.workers}개 프로세스로 처리 (이미 만든 파일은 건너뜀, 중간에 멈춰도 다시 돌리면 이어서)")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, *j) for j in jobs]
            try:
                from tqdm import tqdm
                it = tqdm(as_completed(futs), total=len(futs))
            except ImportError:
                it = as_completed(futs)
            results = [fu.result() for fu in it]
    else:
        try:
            from tqdm import tqdm
            jobs_it = tqdm(jobs)
        except ImportError:
            jobs_it = jobs
        results = [_one(*j) for j in jobs_it]

    for res in sorted(results, key=lambda r: r[1]):
        kind = res[0]
        if kind == "skip":
            skipped.append((res[2], res[3])); continue
        if kind == "fail":
            failed.append((res[2], res[3])); continue
        _, _, row, rel_song, frames, psum = res
        total_frames += frames
        pos_count += psum
        row["split"] = split_of(rel_song, args.val_ratio, args.test_ratio)
        rows.append(row)

    if not rows:
        print("처리된 차분이 없어요.")
        for n, why in skipped[:10]:
            print(f"  제외 {n}: {why}")
        for n, why in failed[:10]:
            print(f"  실패 {n}: {why}")
        return

    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- 요약 ----
    songs = {r["song_id"] for r in rows}
    print(f"\n처리: 차분 {len(rows)}개 / 곡 {len(songs)}개 / 총 {total_frames / FPS / 3600:.2f}시간")
    for s in ("train", "val", "test"):
        sr_ = [r for r in rows if r["split"] == s]
        print(f"  {s:5s}: 곡 {len({r['song_id'] for r in sr_})}개, 차분 {len(sr_)}개")
    if skipped:
        print(f"제외 {len(skipped)}개: " + ", ".join(f"{n}({why})" for n, why in skipped[:5]))
    if failed:
        print(f"실패 {len(failed)}개:")
        for n, why in failed[:5]:
            print(f"  {n}: {why}")
    ratio = pos_count / max(total_frames, 1)
    print("레인별 노트 프레임 비율(양성 비율): " +
          " ".join(f"{i}:{p:.3f}" for i, p in enumerate(ratio)))
    print(f"-> {out_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
