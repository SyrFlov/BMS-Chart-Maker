"""
[빌드] 채보 생성기를 파이썬 없이 실행되는 exe 폴더로 묶기 (윈도우에서 한 번)

사용:
    python build_exe.py           ->  dist/BMSChartAI/  (BMSChartAI.exe + 모델 + input/output 폴더)
    python build_exe.py --zip     ->  위 폴더 + 다른 PC로 옮길 dist/BMSChartAI_날짜.zip
"""
from __future__ import annotations

import argparse
import ast
import datetime
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
BUILD_ENV = "bms-exe"
EXCLUDES = ["matplotlib", "pandas", "IPython", "notebook", "jupyter_client", "ipykernel", "tensorboard",
            "torchvision", "torchaudio", "sklearn", "PyQt5", "PyQt6", "PySide2", "PySide6", "cv2", "librosa"]

USAGE_TXT = """BMS AI 채보 생성기
=================

1. BMSChartAI.exe 실행 (처음엔 10~30초 걸릴 수 있어요)
   - "Windows의 PC 보호" 창이 뜨면 [추가 정보] -> [실행]
2. [곡 추가…]로 mp3/wav/ogg/flac 고르기 (여러 곡 가능)
3. [채보 만들기] -> output\\<곡 이름>\\ 에 .bms와 음원이 생겨요
   - 결과 목록을 두 번 누르면 그 폴더가 열려요
   - 한 곡에 1~3분 정도

문제가 생기면 결과 폴더의 생성_로그.txt를 보내 주세요.


창 설정 (첫 화면)
================
  빠른 설정             기본 / 밀도 줄이기 / 밀도 늘리기
  생성 로그에서 불러오기   결과 폴더의 생성_로그*.txt에 적힌 설정을 창에 채움
  노트 확률             클수록 노트가 적어요 (기본 0.45)
  최대 동시치기 수        한 번에 칠 수 있는 최대 건반 수, 스크래치는 따로 (기본 6)
  건반 배율             동시치기 크기만 이 배수로 (기본 1)
  떨림(드르릉) 살리기     드르릉·떨림 구간을 24·32비트로 채우기 (기본 0 = 끔)
  16비트 폭타 줄이기      약한 16비트 뒷자리 노트를 빼기 (기본 0 = 끔)
  16비트 이상으로 채우기   약한 소리를 32·24비트로, 작을수록 많이 (기본 0.25)
  폭타 쌍 합 평균        폭타 8비트+16비트 건반 합의 평균 목표 (기본 4)
  폭타 쌍 합 편차 ±      소리가 센 폭타는 두껍게, 약한 폭타는 얇게 (기본 1.5)
  자리 모델 온도         낮으면 흔한 모양, 높으면 다양하게 (기본 1)
  스크래치 성향 / 최소량   스크래치 고르는 정도 / 마디당 최소 수 (1.5 / 0.9)
  슬라이더 이름이나 설명에 마우스를 올리면 자세한 설명이 떠요.

세부 규칙 (고급 설정 보기 안, 보통 그대로)
=========================================
  구간 밀도 분포         조용한 구간이 사람 차분보다 빽빽하면 덜기 (기본 1)
  폭타 두께 맞추기        폭타 구간이 사람 차분보다 얇으면 늘리기 (기본 1)
  최고 밀도 상한         초당 노트 수 상한 (기본 0 = 없음)
  멜로디 지키기          밀도를 덜 때 멜로디·보컬 줄은 남기기 (기본 1)
  낮은 BPM 2·3연타 허용   BPM 150 아래에서 2·3연타 일부 허용 (기본 0.3)
  24·32비트 허용        1 = 자동, 0 = 16비트만, 2 = 다 허용
  폭타 사다리           가벼운 구간을 두께만 줄인 폭타로 (기본 켬)
  구간 강약             조용한 곳은 얇게, 큰 곳은 두껍게 (기본 1)
  그 밖               아래 추가 옵션 목록에서 [창: 고급] 표시

곡 정보
=======
  판정(JUDGE)          EASY / NORMAL / HARD / VERY HARD
  TOTAL                게이지 회복량 (기본 300)
  BPM / 제목           한 곡일 때만. BPM이 2배·절반으로 잡히면 직접 입력
  차분명               있으면 <곡>_<차분명>.bms로 따로 저장 (안 덮어씀)
  작곡가               #ARTIST. #GENRE는 항상 AI Generated
  곡 목록              차분명·작곡가 칸을 두 번 눌러 곡마다 따로
  따라갈 악기          끔 / 드럼 / 악기 / 보컬: 그 소리가 없는 줄을 '세기' 확률로 뺌
  구간별               예: 1-16 드럼, 17-32 악기 / 1~3초 보컬 (뒤에 적은 것이 이김)
                       처음 한 번 음원 분리로 몇 분 더 걸려요


추가 옵션 (고급 설정 > 추가 옵션 칸)
===================================
infer.py 옵션을 적으면 창 설정 뒤에 붙어요. 같은 옵션이면 이 칸이 이겨요.
예) --max-chord 5 --scratch-bias -0.5

표시: [창] 창에서 정함  [고정] 창이 늘 넣는 값  [숨김] 창에 없음, 여기서만
      (기본 …) = 창 기본 설정으로 만들 때 쓰이는 값

■ 입력 · 출력 · 모델
  --audio 파일          곡 오디오 [창]
  --out 폴더            저장 폴더 [창]
  --folder 이름         결과 폴더 이름 [창: 곡 이름]
  --title 글자          제목 (기본 파일 이름) [창]
  --chart-name 글자     차분명, 따로 저장 [창]
  --artist 글자         작곡가 [창]
  --judge 판정          easy / normal(기본) / hard / veryhard [창]
  --total 숫자          TOTAL (기본 300) [창]
  --ckpt 파일           1단계(노트 타이밍) 모델 [창: 고급]
  --lane-ckpt 파일      레인 모델 [창: 고급]
  --plan-ckpt 파일      계획 모델 [창: 고급]
  --density-ckpt 파일   반 마디 밀도 모델 (지금은 안 씀)
  --key-sound 종류      silence(기본) / click [창: 클릭음]
  --no-clicks           결과 음원에 확인용 클릭을 안 섞음
  --seed 숫자           무작위 시드, 바꾸면 다른 배치 [창: 고급]
  --dump-probs 파일     확인용: 노트 확률을 .npy로 저장

■ BPM · 박자 격자
  --bpm 숫자            곡 BPM, 비우면 자동 [창]
  --bpm-range 최소 최대  자동 BPM 범위 (기본 120 280)
  --offset 초           격자 위상, 비우면 자동
  --downbeat 0~3        마디 첫 박 위치 (기본 자동)
  --snap 방식           격자 붙이기 [고정 beat]
  --beat-divs 목록      쓸 분할 (기본 4,3,6,8)
  --offgrid-keep 0~100  32분·셋잇단 노트를 남길 기준 [고정 0]
  --tol-ms 숫자         격자 허용 오차 ms (기본 10)
  --snap-min-ms 숫자    가장 촘촘한 칸 간격 ms (기본 25)
  --shift-ms 숫자       모든 노트를 늦춤(음수면 당김)
  --no-section-fix      구간 정리 끔 [고정]
  --snap-fix ms         멀리 밀린 노트를 맞는 칸에 다시 놓기 (기본 25) [창: 고급]
  --fine-grid 0~2       24·32비트 허용 (1 = 자동) [창: 고급]
  --fine-bpm 0~1        박마다 소리가 맞을 때만 24·32비트 (기본 1) [창: 고급]

■ 노트 양 · 밀도
  --threshold 숫자      노트 확률, 클수록 적음 (기본 0.45) [창]
  --thin-weak 0~100     약한 16비트 하위 N% 빼기 (기본 0) [창]
  --thin-local N        위 N%를 앞뒤 N마디 기준으로 (기본 2) [창: 고급]
  --thin-boost 0~1      뺀 자리 옆 동시치기 +1 확률 (기본 0.7) [창: 고급]
  --stream-hole N       폭타 속 한 칸 빈칸 막기 (기본 8) [창: 고급]
  --fill32 숫자         약한 소리를 32·24비트로 채움, 작을수록 많이 (기본 0.25) [창]
  --fill32-sound 0~3    소리가 있는 자리만 채우기 (기본 0, 권장 1.5) [창: 고급]
  --chord32 0~1         약한 32비트를 옆 줄과 합치기 (기본 0, 권장 0.45) [창: 고급]
  --trem 0~1.5          떨림(드르릉) 살리기 (기본 0) [창]
  --dens-shape 0~1.5    조용한 구간이 사람보다 빽빽하면 덜기 (기본 1) [창: 고급]
  --dens-cap 초당       최고 밀도 상한 (기본 0 = 없음) [창: 고급]
  --dens-flat 0/1       강약 적은 곡은 덜 비우기 (기본 1) [창: 고급]
  --dens-dev 0/1        큰 소리 대비 편차로 밀도 정하기 (기본 1) [창: 고급]
  --dens-pulse 0/1      8분 박이 들리면 8비트 유지 (기본 1) [창: 고급]
  --mel-keep 0~2        멜로디·보컬 줄 지키기 (기본 1) [창: 고급]
  --bass-sound 0~1      소리 크기에 베이스 넣기 (기본 0.4) [창: 고급]
  --key-scale 0.5~1.5   동시치기 크기 배율 (기본 1) [창]
  --dynamics 숫자       셈여림 반영 (기본 0)
  --keep-sure 0~1       확실한 노트는 지우지 않음 (기본 0.93)
  --density-q 0~1       밀도 모델 기준 (기본 0.5)
  --density-conf 0~1    밀도 모델 보호 기준 (기본 0.25)
  --window-sec 숫자     모델이 한 번에 보는 길이 (기본 20초)
  --overlap-sec 숫자    겹치는 길이 (기본 3초)
  --note-log 0/1        노트 후보 기록 _notes_debug.csv (기본 1) [숨김]

■ 폭타 구조
  --stream-fill N       거의 찬 폭타 마디 빈칸 채우기 (기본 11) [창: 고급]
  --fill-grade 0~1      폭타 잇기를 소리 크기별 단계로 (기본 1) [숨김]
  --stream-target 0~1   폭타 마디 비율이 넘으면 가벼운 구간 (기본 0.64) [창: 고급]
  --ladder 0/1          가벼운 구간을 두께만 줄인 폭타로 (기본 1) [창: 고급]
  --stream-break N      폭타가 N마디 넘게 이어지면 끊기 (기본 16) [창: 고급]
  --rest-chord 0~1.2    소리 큰 곳 쉬는 동시치기 두께 (기본 1) [창: 고급]
  --half-fill 0~0.45    소리 큰 곳 빈 16비트 채우기 (기본 0.30) [창: 고급]
  --accent-rest 0~2     큰 소리 뒤 치고 쉬기 (기본 1) [숨김]
  --accent-rest-len 박  쉬는 길이 (기본 1박) [숨김]
  --stream-lift 0~1.5   얇은 폭타를 사람 두께로 (기본 1) [창: 고급]
  --stream-kpr 0~3      폭타 줄당 건반 (기본 0 = 끔) [창: 고급]
  --pair-avg 2~7        폭타 쌍 합 평균 (기본 4) [창]
  --pair-dev 0~3        폭타 쌍 합 편차 ± (기본 1.5) [창]
  --pair-spread 숫자    옛 이름 (편차 = 값 / 2)
  --pair-even 0~1       쌍 합을 마디 안에서 고르게 (기본 1) [창: 고급]
  --pair-keep 0~1       쌍 합 높낮이를 남기기 (기본 1) [숨김]
  --pair-min N          쌍 합 최소 (기본 5) [숨김]
  --four-one 0~1        4+2 폭타 일부를 4+1로 (기본 1) [창: 고급]
  --section-dyn 0~1.5   구간 강약 (기본 1) [창: 고급]
  --end-shape 0~1       폭타 끝 줄 크기·모양 (기본 1) [창: 고급]
  --end-fat 0~1         폭타 끝 줄을 앞보다 두껍게 (기본 1) [숨김]
  --end-over N          폭타 끝 줄만 상한 + N (기본 1) [창: 고급]
  --stream-end-chord N  폭타 끝 줄을 앞 줄보다 N개 더 [고정 1]
  --restart-shift 0~1   폭타 다시 시작 줄 겹침을 사람처럼 (기본 1) [숨김]
  --guard 0/1           울타리 모드(실험) (기본 0) [숨김]

■ 레인 배치 (레인 모델)
  --temperature 숫자    레인 뽑기 온도 (기본 0.8)
  --density 숫자        레인 모델 목표 밀도 (기본 20)
  --max-chord 숫자      한 줄 최대 건반 수, 스크래치 제외 (기본 6) [창]
  --no-jack-ms 숫자     이 ms 안 같은 건반 금지 (기본 100)
  --jack-penalty 숫자   다른 소리 같은 건반 억제 [고정 3]
  --jack-beats 숫자     위 억제 범위, 박 (기본 1)
  --jack16 0~1          16분 연타를 옆 건반으로 (기본 0.5) [숨김]
  --jack8 0~1           8분 간격 줄연타 줄이기 (기본 1) [창: 고급]
  --jack-allow 0~1      BPM 150 아래 16비트 2·3연타 허용 (기본 0.3) [창: 고급]
  --scratch-bias 숫자   스크래치 성향 (기본 1.5) [창]
  --scratch-add 숫자    마디당 최소 스크래치 (기본 0.9) [창]
  --scratch-pos 숫자    스크래치 자리 가중치 (기본 1) [숨김]
  --scratch-model 파일  스크래치 자리 모델 (기본 scratch_pos.npz)
  --scratch-place 0~1   스크래치를 자리 모델 순서로 (기본 1) [숨김]
  --lone-scratch        스크래치만 있는 줄을 그대로 둠
  --plan-temp 숫자      계획 모델 온도 (기본 0.5)
  --max-alt-rows 숫자   1357 246 같은 교대 최대 줄 수 (기본 3)
  --alt-scale 숫자      교대 계획 배율 (기본 1)
  --lane-mode 방식      레인 모델 없을 때 cycle / random
  --coverage 숫자       안 쓴 건반 쓰게 (기본 0)
  --vary 0~1            같은 묶음 반복 다시 뽑기 (기본 0)
  --same-sound 숫자     비슷한 소리는 같은 모양 쪽으로 (기본 0)
  --same-beats 숫자     위 찾는 범위, 박 (기본 8)
  --row-seed 0/1        줄마다 고정 난수 (기본 1) [숨김]

■ 자리 모델
  --placer-ckpt 경로    자리 모델 파일 (기본 placer_v1.pt)
  --placer 0~1          자리를 다시 뽑는 줄 비율 (기본 1) [숨김]
  --placer-temp 숫자    온도 (기본 1.0) [창]
  --placer-rep 0~2      A B A B 반복 억제 (기본 0.45) [창: 고급]
  --placer-beat 0~6     한 박 전 복사·단타 계단 억제 (기본 3) [창: 고급]

■ 후처리 (모양 다듬기)
  --on8-match 0~1       폭타 8비트 자리 두께를 ★20 기준으로 (기본 1) [창: 고급]
  --off-match 0~1       폭타 16비트 뒷자리 두께를 ★20 기준으로 (기본 1) [숨김]
  --stream-cap 0~1      폭타 속 4개+ 동시치기 덜기 (기본 0.25) [창: 고급]
  --count-smooth 0~1    8비트 동시치기 개수 튐 줄이기 (기본 0.8) [창: 고급]
  --count-keep 0~1      반 마디 동안 개수 이어 가기 (기본 0.5)
  --trip-match 0~1      셋잇단 두께를 사람처럼 (기본 1) [창: 고급]
  --skel-swap 0~1       8비트 자리가 뒷자리보다 얇으면 맞바꾸기 (기본 1) [숨김]
  --beat-pair 0~1       박 안 두 8비트 두께 맞추기 (기본 1) [숨김]
  --eighth-audit 0~1    8비트 두께 비율 마지막 점검 (기본 1) [창: 고급]
  --bar-repeat 0~1      비슷한 마디는 앞 마디처럼 (기본 0.6) [창: 고급]
  --bar-shape 0~1       위에서 모양까지 따라가기 (기본 0.35) [숨김]
  --beat-motif 0~1      앞 박과 같은 두께 패턴 (기본 1) [창: 고급]
  --eighth-shape 0~1    8비트 동치 A B A B·옆으로 옮기기 (기본 1) [창: 고급]
  --eighth-trans 0~1    8비트 동치 이어지는 방식 (기본 1) [창: 고급]
  --eighth-overlap 0~1  8비트 구간 겹친 건반 옮기기 (기본 0.7) [숨김]
  --eighth-same 0~1     같은 소리 겹침도 옮기기 (기본 0.6) [숨김]
  --eighth-big 0~1      8비트 동치 건반 더하기 (기본 0) [숨김]
  --diag 0~1            폭타 대각선 모양 (기본 0.25) [창: 고급]
  --shape-run 0~1       같은 모양 이어짐 끊기 (기본 1) [창: 고급]
  --edge4 0~1           가운데 몰린 4건반을 1·7번으로 (기본 1) [창: 고급]
  --edge-axis 0~1       1·7번 축을 안쪽으로 (기본 1) [창: 고급]
  --pillar-fix 0~1      8비트 자리 기둥 옮기기 (기본 0.3) [숨김]
  --quarter-spread 0~1  4비트 한쪽 손 쏠림 풀기 (기본 1) [숨김]
  --hand-balance 0~1    폭타 밖 손 쏠림 풀기 (기본 1) [창: 고급]
  --hand32 0~1          32비트 한쪽 손 쏠림 풀기 (기본 1) [숨김]
  --trill-break 0~1     한 손 트릴 끊기 (기본 1) [숨김]
  --superset-fix 0~1    45 -> 457 같은 모양 풀기 (기본 1) [숨김]
  --fine-single 0~1     채운 32·24비트를 단타로 (기본 1) [창: 고급]
  --fine-shape 0~1      32·24비트 연속을 계단 등으로 (기본 1) [창: 고급]
  --fine-density 0~1    폭타 속 32·24비트는 단타 위주로 (기본 1) [창: 고급]
  --fill-axis 0~1       채운 단타가 같은 건반이면 옮기기 (기본 1) [숨김]
  --flow 0~1            동시치기 방향 이어 가기 (기본 0.35) [숨김]
  --denim-max 숫자      2건반 16분 연속을 이 줄에서 끊기 (기본 8)
  --spread-chords 0~1   붙은 건반(23) 벌리기 [고정 0.2]
  --thick 0~1           곡 세기에 따른 두께 [고정 0]
  --thick-range 최소,최대  위 목표 두께 (기본 1.0,3.2)
  --chord-style 0~1     반 마디 동치 모양 유지 [고정 0]
  --beat-contrast 0~1   뒷자리를 8비트보다 얇게 [고정 0]
  --no-sparse-spread    성긴 구간 4번 쏠림 풀기 끔
  --accent3-at 구간     점8분 묶음으로 찍을 구간(초), 예: 39-41.5
  --accent3-phase 0~2   위 강세 위치 (기본 자동)
  --accent3-auto 숫자   점8분 자동 찾기 (기본 0)

■ 악기 따라가기 (Demucs 음원 분리)
  --follow 층           drum / inst / vocal / off(기본) [창]
  --follow-at "범위"    구간별: "1-16:drum, 17-32:inst" 또는 "1~3초 보컬" [창]
  --follow-strength 0~1 따라가지 않는 층 줄을 뺄 확률 (기본 0.7) [창: 세기]
  --stems-model 이름    분리 모델 (기본 htdemucs)
  --stems-device 장치   cpu(기본) / cuda

■ 확인용
  --raw-lanes           레인 후처리를 모두 끔
  --random-lanes        건반 수는 두고 자리만 무작위
  --ref 파일            정답 채보와 노트 시각 비교
  --ref-audio 파일      정답 채보의 오디오
  --ref-shift-ms 숫자   정답 시각을 늦춰 비교
"""


def sh(cmd, **kw):
    print("$", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def local_modules(entries=("bms_gui", "infer")) -> list[str]:
    """프로젝트 .py 중 실제로 불러오는 것들 (PyInstaller에 확실히 넣으려고)"""
    local = {p.stem for p in ROOT.glob("*.py")}
    seen, stack = set(), [e for e in entries if e in local]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        tree = ast.parse((ROOT / f"{m}.py").read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module.split(".")[0]]
            stack += [x for x in names if x in local and x not in seen]
    return sorted(seen)


def check_project():
    import bms_gui
    need = [ROOT / "bms_gui.py", ROOT / "infer.py"]
    need += [ROOT / bms_gui.DEFAULTS[k] for k in ("ckpt", "lane_ckpt", "plan_ckpt")]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        raise SystemExit("필요한 파일이 없어요 (프로젝트 폴더에서 실행했는지 확인):\n  " + "\n  ".join(miss))
    for mod in ("numpy", "scipy", "soundfile", "torch"):
        if importlib.util.find_spec(mod) is None:
            raise SystemExit(f"{mod}가 없는 환경이에요. conda activate bms-ai 후 다시 실행하세요.")


def pip_version(mod: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return None


def ensure_build_env(args) -> bool:
    """True면 이 프로세스에서 바로 빌드. False면 빌드 전용 환경에서 다시 실행했으니 끝."""
    import torch
    cuda = getattr(torch.version, "cuda", None)
    if args.inner or not cuda or args.keep_cuda:
        if importlib.util.find_spec("PyInstaller") is None:
            print("PyInstaller 설치 중…")
            sh([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True
    conda = os.environ.get("CONDA_EXE")
    if not conda:
        raise SystemExit(f"지금 torch가 GPU(CUDA {cuda})용이에요. 그대로 묶으려면 --keep-cuda (3GB 이상), "
                         f"작게 만들려면 Anaconda Prompt에서 conda activate bms-ai 후 다시 실행하세요.")
    print(f"지금 torch는 GPU(CUDA {cuda})용이라 그대로 묶으면 3GB를 넘어요 -> CPU용 torch로 빌드 전용 환경 '{BUILD_ENV}'을 씁니다")
    envs = subprocess.run([conda, "env", "list"], capture_output=True, text=True).stdout
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if not any(line.split() and line.split()[0] == BUILD_ENV for line in envs.splitlines()):
        sh([conda, "create", "-y", "-n", BUILD_ENV, f"python={pyver}", "tk"])
    run = [conda, "run", "--no-capture-output", "-n", BUILD_ENV]
    base = torch.__version__.split("+")[0]
    try:
        sh(run + ["python", "-m", "pip", "install", f"torch=={base}", "--index-url",
                  "https://download.pytorch.org/whl/cpu"])
    except subprocess.CalledProcessError:
        sh(run + ["python", "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    pins = [f"{m}=={v}" if (v := pip_version(m)) else m for m in ("numpy", "scipy", "soundfile")]
    sh(run + ["python", "-m", "pip", "install", *pins, "tqdm", "pyinstaller"])
    argv = [a for a in sys.argv[1:] if a != "--inner"]
    r = subprocess.run(run + ["python", str(Path(__file__).resolve()), "--inner", *argv], cwd=str(ROOT))
    sys.exit(r.returncode)


def run_pyinstaller(name: str, dist: Path):
    import PyInstaller.__main__
    mods = local_modules()
    print(f"묶을 프로젝트 모듈 {len(mods)}개: {', '.join(mods)}")
    a = [str(ROOT / "bms_gui.py"), "--name", name, "--onedir", "--windowed", "--noconfirm", "--clean",
         "--distpath", str(dist), "--workpath", str(ROOT / "build" / "pyi"), "--specpath", str(ROOT / "build"),
         "--paths", str(ROOT), "--hidden-import", "_cffi_backend"]
    for m in mods:
        a += ["--hidden-import", m]
    if importlib.util.find_spec("demucs") is not None:               # 음원 분리(따라갈 악기): 모델 목록 파일(remote/*.txt, *.yaml)까지
        a += ["--collect-all", "demucs"]
        # Demucs 모델 파일(pickle)이 numpy.core.multiarray를 찾는데, numpy 2에서는 호환용 껍데기 모듈이라
        #          PyInstaller가 빼먹음 -> "No module named 'numpy.core.multiarray'" 방지
        for m in ("numpy.core", "numpy.core.multiarray", "numpy.core._multiarray_umath", "numpy.core.numeric",
                  "numpy._core.multiarray", "numpy._core.numeric"):
            if importlib.util.find_spec(m.split(".")[0]) is not None:
                a += ["--hidden-import", m]
    if importlib.util.find_spec("_soundfile_data") is not None:      # libsndfile DLL (mp3 읽기)
        a += ["--collect-all", "_soundfile_data"]
    for x in EXCLUDES:
        a += ["--exclude-module", x]
    ico = ROOT / "icon.ico"
    if ico.exists():
        a += ["--icon", str(ico)]
    t0 = time.time()
    PyInstaller.__main__.run(a)
    print(f"PyInstaller 끝 ({time.time() - t0:.0f}초)")


KEEP = ("input", "output", "gui_settings.json")      # 다시 빌드해도 남길 것 (사용자가 만든 패턴, 넣어 둔 곡, 창 설정)


def keep_user_files(app: Path, dist: Path) -> Path | None:
    have = [n for n in KEEP if (app / n).exists()]
    if not have:
        return None
    tmp = dist / f"_keep_{app.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for n in have:
        shutil.move(str(app / n), str(tmp / n))
    print(f"기존 exe 폴더에서 {', '.join(have)} 를 잠시 옮겨 둠 (빌드 뒤 되돌림)")
    return tmp


def restore_user_files(app: Path, tmp: Path | None):
    if tmp is None:
        return
    app.mkdir(parents=True, exist_ok=True)
    for p in tmp.iterdir():
        dst = app / p.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(str(p), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    print("input / output / 창 설정을 되돌려 놓음")


def finish_folder(app: Path):
    import bms_gui
    (app / "checkpoints").mkdir(exist_ok=True)
    for k in ("ckpt", "lane_ckpt", "plan_ckpt"):
        src = ROOT / bms_gui.DEFAULTS[k]
        dst = app / bms_gui.DEFAULTS[k]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"모델 복사: {dst.relative_to(app)} ({src.stat().st_size / 1e6:.1f}MB)")
    sp = ROOT / bms_gui.DEFAULTS["scratch_model"]                 # 스크래치 자리 모델 (있으면)
    if sp.exists():
        shutil.copy2(sp, app / bms_gui.DEFAULTS["scratch_model"])
        print(f"모델 복사: {bms_gui.DEFAULTS['scratch_model']} ({sp.stat().st_size / 1e3:.0f}KB)")
    pp = ROOT / bms_gui.DEFAULTS["placer_ckpt"]                   # 자리 모델 (빠져 있어서 exe에서 늘 꺼져 있었음)
    if pp.exists():
        shutil.copy2(pp, app / bms_gui.DEFAULTS["placer_ckpt"])
        print(f"모델 복사: {bms_gui.DEFAULTS['placer_ckpt']} ({pp.stat().st_size / 1e6:.1f}MB)")
    else:
        print(f"[주의] 자리 모델 {pp} 이 없어 exe에서 자리 모델이 꺼져요")
    (app / "input").mkdir(exist_ok=True)
    (app / "output").mkdir(exist_ok=True)
    (app / "사용법.txt").write_text(USAGE_TXT.replace("\n", "\r\n"), encoding="utf-8-sig")


def make_test_audio(path: Path, bpm: float = 150.0, sec: float = 24.0, sr: int = 44100):
    """합성 드럼+베이스: 킥 8분, 스네어 2, 4박, 하이햇 16분, 베이스 음 변화 (실제 패턴 생성이 끝까지 도는지 보는 용도)"""
    import numpy as np
    import soundfile as sf
    n = int(sec * sr)
    y = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    s16 = 60.0 / bpm / 4
    t = np.arange(int(0.25 * sr)) / sr
    kick = (np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 18)).astype(np.float32)
    snare = (rng.standard_normal(len(t)) * np.exp(-t * 25) * 0.6).astype(np.float32)
    hat = (rng.standard_normal(len(t)) * np.exp(-t * 90) * 0.25).astype(np.float32)
    notes = [55, 55, 65.4, 49]
    for k in range(int(sec / s16)):
        i = int(k * s16 * sr)
        seg = slice(i, min(n, i + len(t)))
        L = seg.stop - seg.start
        if k % 2 == 0:
            y[seg] += kick[:L]
        if k % 8 == 4:
            y[seg] += snare[:L]
        y[seg] += hat[:L] * (1.0 if k % 2 else 0.6)
        if k % 4 == 0:
            f = notes[(k // 16) % 4]
            y[seg] += (0.3 * np.sin(2 * np.pi * f * t[:L]) * np.exp(-t[:L] * 6)).astype(np.float32)
    y /= max(1e-6, float(np.abs(y).max())) * 1.1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([y, y], 1), sr)


def self_test(app: Path, name: str) -> bool:
    import bms_gui
    exe = app / (f"{name}.exe" if os.name == "nt" else name)
    print("\n[점검 1] 라이브러리/모델/mp3 읽기 (exe --selftest)")
    try:
        subprocess.run([str(exe), "--selftest"], timeout=900)
    except subprocess.TimeoutExpired:
        print("  시간 초과")
    rep = app / "selftest.txt"
    text = rep.read_text(encoding="utf-8") if rep.exists() else "selftest.txt가 안 생겼어요 (exe가 바로 꺼짐)"
    print("  " + text.replace("\n", "\n  "))
    ok1 = text.startswith("모두 정상")
    rep.unlink(missing_ok=True)

    print("[점검 2] 합성 드럼 24초로 실제 채보 만들기 (exe --run-infer)")
    work = ROOT / "build" / "selftest"
    shutil.rmtree(work, ignore_errors=True)
    wav = work / "selftest_150bpm.wav"
    make_test_audio(wav)
    s = dict(bms_gui.DEFAULTS)
    for k in ("ckpt", "lane_ckpt", "plan_ckpt"):
        s[k] = str(app / bms_gui.DEFAULTS[k])
    s.update(out=str(work / "out"), bpm="150", title="selftest")
    log = work / "infer_log.txt"
    cmd = [str(exe), "--run-infer"] + bms_gui.infer_args(str(wav), s, single=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(app), timeout=1800, env={**os.environ, "BMS_CHILD_LOG": str(log)})
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = -1
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    bms = list((work / "out").rglob("*.bms"))
    ok2 = code == 0 and bool(bms)
    tail = [l for l in lines if l.startswith(("예측한 노트", "레인 배치", "오디오", "모델"))] or lines[-8:]
    print("  " + "\n  ".join(tail))
    print(f"  -> {'성공' if ok2 else '실패'} (종료 코드 {code}, {time.time() - t0:.0f}초, bms {len(bms)}개, 로그 {log})")
    if not ok2 and lines:
        print("  마지막 로그:\n  " + "\n  ".join(lines[-15:]))
    return ok1 and ok2


def folder_size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def main():
    ap = argparse.ArgumentParser(description="채보 생성기 exe 만들기")
    ap.add_argument("--name", default="BMSChartAI", help="exe/폴더 이름 (영문 권장)")
    ap.add_argument("--zip", action="store_true", help="다른 PC로 옮길 zip도 만듦")
    ap.add_argument("--keep-cuda", action="store_true", help="GPU용 torch 그대로 묶기 (3GB 이상)")
    ap.add_argument("--skip-test", action="store_true", help="빌드 뒤 점검 생략")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    check_project()
    if not ensure_build_env(args):
        return
    import torch
    print(f"빌드 환경: Python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"numpy {pip_version('numpy')}, soundfile {pip_version('soundfile')}")
    dist = ROOT / "dist"
    app = dist / args.name
    if app.exists():                             # 실행 중인 exe가 폴더를 잡고 있으면 PyInstaller가 '액세스가 거부되었습니다'로 끝남
        probe = app.with_name(app.name + "_lockcheck")
        try:
            app.rename(probe)
            probe.rename(app)
        except OSError as e:
            raise SystemExit(f"\n{app} 폴더를 쓸 수 없어요 ({e}).\n"
                             f"BMSChartAI.exe(채보 생성기 창)가 켜져 있거나, 그 폴더 안 파일을 다른 프로그램이 열고 있어요.\n"
                             f"창을 닫고 작업 관리자에서 BMSChartAI.exe가 남아 있으면 끝낸 뒤 다시 실행하세요.")
    kept = keep_user_files(app, dist)          # 다시 빌드하면 PyInstaller가 dist/이름 폴더를 통째로 지움
    try:
        run_pyinstaller(args.name, dist)
    finally:
        restore_user_files(app, kept)            # 빌드가 실패해도 되돌림
    finish_folder(app)
    print(f"\n완성: {app}  ({folder_size(app):.2f}GB)")
    ok = True
    if not args.skip_test:
        ok = self_test(app, args.name)
        print("\n점검 결과:", "모두 통과 - 시연에 써도 돼요" if ok else "문제 있음 - 위 로그를 보내 주세요")
    if args.zip:
        stamp = datetime.date.today().strftime("%Y%m%d")
        print("압축 중… (몇 분 걸려요)")
        z = shutil.make_archive(str(dist / f"{args.name}_{stamp}"), "zip", root_dir=dist, base_dir=args.name)
        print(f"zip: {z} ({Path(z).stat().st_size / 1e9:.2f}GB)")
    print(f"\n실행: {app / (args.name + '.exe')}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
