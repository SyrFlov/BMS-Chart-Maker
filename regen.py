"""
[실행 보조] 결과 폴더의 생성_로그.txt 첫 줄 명령으로 소스 infer.py를 다시 돌린다 (exe 빌드, 창 없이)

사용:
    python regen.py --songs all --extra "--guard 1" --name GUARD     울타리 버전을 <곡>_GUARD.bms로 따로
    python regen.py --songs 곡1,곡2                           지금 설정 그대로 다시 (원래 패턴을 덮어씀)
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def show_cmd(cmd: list[str]) -> str:
    """bms_gui.show_cmd와 같은 규칙 (공백 있으면 따옴표)"""
    return " ".join(f'"{c}"' if (" " in c or not c) else c for c in cmd)


def read_cmd(log: Path) -> list[str]:
    """생성_로그.txt 첫 줄 '$ python infer.py ...' -> infer.py 옵션 토큰"""
    line = log.read_text(encoding="utf-8").splitlines()[0]
    if not line.startswith("$ "):
        raise ValueError(f"{log}: 첫 줄이 '$ python infer.py ...'가 아님")
    toks = shlex.split(line[2:], posix=False)
    toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t for t in toks]
    while toks and not toks[0].startswith("--"):
        toks.pop(0)                                   # python infer.py
    return toks


def drop_opt(toks: list[str], opt: str) -> list[str]:
    """'--opt 값' 쌍을 모두 뺌"""
    out, i = [], 0
    while i < len(toks):
        if toks[i] == opt:
            i += 2
            continue
        out.append(toks[i])
        i += 1
    return out


def tag_of(name: str) -> str:
    """bms_gui와 같은 차분명 -> 파일 이름 꼬리"""
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(name or "")).strip("._")


def main() -> int:
    ap = argparse.ArgumentParser(description="생성 로그 명령으로 소스 infer.py 다시 돌리기")
    ap.add_argument("--songs", default="all")
    ap.add_argument("--extra", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=str(ROOT / "dist" / "BMSChartAI" / "output"))
    ap.add_argument("--from", dest="from_log", default="생성_로그.txt")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")       # 윈도 콘솔에서 한글 깨짐 방지
    except Exception:
        pass

    out = Path(a.out)
    if a.songs.strip().lower() == "all":
        songs = sorted(p.name for p in out.iterdir() if (p / a.from_log).exists())
    else:
        songs = [s.strip() for s in a.songs.split(",") if s.strip()]
    tag = tag_of(a.name)
    extra = shlex.split(a.extra, posix=False) if a.extra.strip() else []
    extra = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t for t in extra]
    if not tag and not a.dry_run:
        print("[주의] --name이 비어 있어서 원래 차분(<곡>.bms, 생성_로그.txt)을 덮어써요.")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    results = []
    for s in songs:
        folder = out / s
        log = folder / a.from_log
        if not log.exists():
            results.append((s, "로그 없음", 0.0))
            print(f"\n=== {s}: {log} 없음, 건너뜀")
            continue
        if a.skip_done and tag and (folder / f"{s}_{tag}.bms").exists():
            results.append((s, "이미 있음", 0.0))
            print(f"\n=== {s}: {s}_{tag}.bms 이미 있음, 건너뜀")
            continue
        args = drop_opt(drop_opt(read_cmd(log), "--chart-name"), "--out")
        args += ["--out", str(out)]
        if tag:
            args += ["--chart-name", a.name]
        args += extra
        shown = "python infer.py " + show_cmd(args)
        print(f"\n=== {s}" + (f" [{a.name}]" if tag else "") + f"\n$ {shown}")
        if a.dry_run:
            results.append((s, "명령만", 0.0))
            continue
        t0 = time.time()
        p = subprocess.Popen([sys.executable, "-u", str(ROOT / "infer.py")] + args, cwd=str(ROOT), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace")
        lines = []
        for line in p.stdout:
            lines.append(line)
            sys.stdout.write(line)
        p.wait()
        sec = time.time() - t0
        folder.mkdir(parents=True, exist_ok=True)
        logname = f"생성_로그_{tag}.txt" if tag else "생성_로그.txt"
        (folder / logname).write_text(f"$ {shown}\n\n" + "".join(lines), encoding="utf-8")
        tot = next((l.strip() for l in lines if "총 노트" in l), "")
        status = (tot or "완료") if p.returncode == 0 else f"실패(코드 {p.returncode}): " + \
            next((l.strip() for l in reversed(lines) if l.strip()), "")[:100]
        results.append((s, status, sec))

    print("\n=== 정리")
    for s, st, sec in results:
        print(f"  {s:12s} {st}" + (f"  ({sec:.0f}초)" if sec else ""))
    return 0 if all(not st.startswith("실패") for _, st, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
