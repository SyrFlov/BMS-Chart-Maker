"""[학습 준비] 좋은 패턴 목록 (v3 학습에서 더 자주 뽑고, '좋은 패턴' 표시를 붙인다).

index.csv의 source(원본 BMS 경로) 파일 이름에 아래 글자가 들어 있으면 좋은 패턴으로 본다.
대소문자는 구분하지 않는다. 목록은 여기서 고치거나 --good-list 파일(한 줄에 하나)로 덧붙인다.
"""
from __future__ import annotations

from pathlib import Path

# 좋은 패턴으로 볼 파일 이름 조각. 공개 저장소에서는 비워 두고, --good-list 파일(한 줄에 하나)로 넣는다.
GOOD_PATTERNS: list[str] = []


def load_patterns(extra_file: str | None = None) -> list[str]:
    pats = list(GOOD_PATTERNS)
    if extra_file:
        for line in Path(extra_file).read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                pats.append(line)
    return pats


def is_good(source: str, patterns: list[str]) -> bool:
    name = Path(source.replace("\\", "/")).name.lower()
    return any(p in name for p in patterns)


def report(rows: list[dict], patterns: list[str], label: str) -> set[str]:
    """좋은 패턴 chart_id 집합을 돌려주고, 몇 개가 잡혔는지 / 안 잡힌 이름은 뭔지 출력한다."""
    good = {r["chart_id"] for r in rows if is_good(r.get("source", ""), patterns)}
    hit = {p for p in patterns for r in rows if p in Path(r.get("source", "").replace("\\", "/")).name.lower()}
    print(f"[{label}] 좋은 차분 {len(good)}개 / 전체 {len(rows)}개")
    miss = [p for p in patterns if p not in hit]
    if miss:
        if label == "train":
            print(f"[{label}] train에 없는 이름 (val에 있거나 데이터에 없음): {', '.join(miss)}")
    return good
