"""
[학습 준비] zip 또는 폴더 안의 BMS 패턴을 전부 파싱해서 목록(CSV)으로 만든다.

사용 예:
    # zip 풀면서 스캔 (이미지/영상은 건너뛰고 패턴+오디오만 풀림)
    python scan_bms.py "곡.zip" --out data\\bms

    # 이미 풀려 있는 폴더 스캔 (곡 폴더가 여러 개 들어 있는 상위 폴더)
    python scan_bms.py data\\bms

결과: <폴더>/chart_index.csv  (keep 열이 True인 패턴만 학습에 사용)
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
import zipfile
from pathlib import Path

from bms_parser import parse_bms

CHART_EXTS = {".bms", ".bme", ".bml"}
AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}

# ---- 필터 기준 (필요하면 여기만 고치면 됨) ----
MIN_NOTES = 300
MAX_MISSING_RATIO = 0.02   # 키음 이벤트 중 파일이 없는 비율 상한
MAX_SILENT_RATIO = 0.15    # 정의 없는 키음 번호(소리 안 나는 자리표시 노트) 비율 상한.
                           # 이런 노트는 소리가 시작되는 위치와 상관이 없어서(실측: 무작위 위치와 같은 수준) 정답 레이블을 흐린다


def zip_member_name(info: zipfile.ZipInfo) -> str:
    """일본어 파일명이 깨지지 않게 cp932로 복원 (UTF-8 플래그 없는 zip)."""
    name = info.filename
    if not (info.flag_bits & 0x800):
        try:
            name = name.encode("cp437").decode("cp932")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = zip_member_name(info)
            if Path(name).suffix.lower() not in CHART_EXTS | AUDIO_EXTS:
                continue  # 이미지/영상/텍스트는 학습에 불필요
            out = (dest / name).resolve()
            if dest not in out.parents:      # 경로 탈출 방지
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
            n += 1
    print(f"{zip_path.name}: {n}개 파일 추출 -> {dest}")


def audio_index(folder: Path) -> dict[str, Path]:
    """폴더 안 오디오를 stem(소문자) 기준으로 색인. .wav를 .ogg로 바꿔 둔 곡도 찾기 위함."""
    idx: dict[str, Path] = {}
    # 하위 폴더(sound/ 등)에 키음을 모아 둔 곡도 있어서 재귀로 찾는다. 얕은 폴더의 파일을 우선.
    for p in sorted(folder.rglob("*"), key=lambda q: len(q.relative_to(folder).parts)):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            idx.setdefault(p.stem.lower(), p)
    return idx


def analyze(f: Path) -> dict:
    row = {
        "folder": f.parent.name, "file": f.name, "title": "", "level": "",
        "notes": 0, "long_notes": 0, "duration": 0.0, "density": 0.0,
        "keys": "", "dp": False, "missing_ratio": 0.0, "silent_ratio": 0.0, "keep": False, "reason": "",
    }
    try:
        c = parse_bms(f)
    except Exception as e:  # noqa: BLE001
        row["reason"] = f"파싱 실패: {e}"
        return row

    lanes = {n.lane for n in c.notes}
    idx = audio_index(f.parent)
    total = len(c.sounds)
    missing = 0      # #WAV로 정의돼 있는데 파일이 없는 이벤트 (진짜 누락)
    silent = 0       # #WAV 정의 자체가 없는 번호 = 플레이어에서도 소리가 안 나는 자리표시 노트
    for _, w in c.sounds:
        name = c.wav_files.get(w)
        if name is None:
            silent += 1
        elif Path(name).stem.lower() not in idx:
            missing += 1

    row.update(
        title=c.title, level=c.level, notes=len(c.notes),
        long_notes=sum(n.is_long for n in c.notes),
        duration=round(c.duration, 1),
        density=round(len(c.notes) / c.duration, 1) if c.duration else 0.0,
        keys="7K" if {6, 7} <= lanes else "5K",
        dp=c.is_dp,
        missing_ratio=round(missing / total, 4) if total else 1.0,
        silent_ratio=round(silent / total, 4) if total else 1.0,
    )

    reasons = []
    if c.is_dp:
        reasons.append("DP")
    if row["keys"] != "7K":
        reasons.append("7키 아님")
    if len(c.notes) < MIN_NOTES:
        reasons.append(f"노트<{MIN_NOTES}")
    if row["silent_ratio"] > MAX_SILENT_RATIO:
        reasons.append("무음 노트 과다")
    if row["missing_ratio"] > MAX_MISSING_RATIO:
        reasons.append("키음 파일 부족")
    row["keep"] = not reasons
    row["reason"] = ", ".join(reasons)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="zip 파일 또는 곡 폴더(들)의 상위 폴더")
    ap.add_argument("--out", default="data/bms", help="zip 추출 위치")
    args = ap.parse_args()

    src = Path(args.src)
    if src.suffix.lower() == ".zip":
        root = Path(args.out)
        extract_zip(src, root)
    else:
        root = src

    rows = [analyze(f) for f in sorted(root.rglob("*")) if f.suffix.lower() in CHART_EXTS]
    if not rows:
        print("차분 파일을 찾지 못했어요.")
        return

    csv_path = root / "chart_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'파일':20s} {'Lv':>3} {'키':>3} {'노트':>5} {'LN':>3} {'초':>6} {'노트/초':>6} {'누락':>6}  keep")
    for r in rows:
        print(f"{r['file'][:20]:20s} {str(r['level']):>3} {r['keys']:>3} {r['notes']:>5} "
              f"{r['long_notes']:>3} {r['duration']:>6} {r['density']:>6} {r['missing_ratio']:>6.1%}  "
              f"{'O' if r['keep'] else 'X ' + r['reason']}")
    kept = sum(r["keep"] for r in rows)
    print(f"\n총 {len(rows)}개 중 {kept}개 사용 가능 -> {csv_path}")


if __name__ == "__main__":
    main()
