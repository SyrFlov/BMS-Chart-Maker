"""[학습 준비] 좋은 패턴 목록 (v3 학습에서 더 자주 뽑고, '좋은 패턴' 표시를 붙인다).

index.csv의 source(원본 BMS 경로) 파일 이름에 아래 글자가 들어 있으면 좋은 패턴으로 본다.
대소문자는 구분하지 않는다. 목록은 여기서 고치거나 --good-list 파일(한 줄에 하나)로 덧붙인다.
"""
from __future__ import annotations

from pathlib import Path

# 패턴 제작자 규칙: yumether 패턴은 전부 좋은 패턴 (7yum / 7yume / yumether)
GOOD_PATTERNS = [
    "yumether", "7yum",
    # 사용자가 고른 좋은 패턴
    "_chronomia_presea", "_chronomia_sa.",
    "yaku_7bt3maniaq", "03_light_of_white_magic_ml7", "air7god", "_morion_7yu",
    "_nnms_99_hardest", "crazybitsa", "crazybits_wc",
    "dive_air02galaxy", "we_are_the_hiiii", "xecus_x_z", "_f_remake",
    "_goodtek_sa__f", "_04_conflict.infelno", "_ele_butter_70",
    "fuxxing_your_brain_yuz", "_lemuria_lt",
    # 뺌: 묶음 교대(1357/246 등 반복)가 많은 곡
    #   whiteout_r, messier20, _aeuro_infelno, angeldust7m_
]


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
