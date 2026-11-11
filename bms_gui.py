"""
[창] BMS AI 채보 생성기 (창 프로그램)

사용:
    python bms_gui.py                 또는  BMS_GUI.bat 더블클릭
    python bms_gui.py 곡1.mp3 곡2.mp3  (곡을 미리 넣은 채로 열기, .bat에 파일을 끌어다 놓아도 됨)
"""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import tempfile
import time
import traceback
from pathlib import Path

# tkinter는 창을 띄울 때만 불러옴 (--run-infer / --selftest 자식 프로세스는 창 없이)
tk = ttk = filedialog = messagebox = ScrolledText = None


def _load_tk():
    global tk, ttk, filedialog, messagebox, ScrolledText
    import tkinter as _tk
    from tkinter import filedialog as _fd, messagebox as _mb, ttk as _ttk
    from tkinter.scrolledtext import ScrolledText as _st
    tk, ttk, filedialog, messagebox, ScrolledText = _tk, _ttk, _fd, _mb, _st


FROZEN = bool(getattr(sys, "frozen", False))          # PyInstaller exe 안에서 실행 중
HERE = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
SETTINGS = HERE / "gui_settings.json"
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".flac")
AUDIO_TYPES = [("오디오", " ".join("*" + e for e in AUDIO_EXT)), ("모든 파일", "*.*")]

DEFAULTS = {
    "out": str(HERE / "output"),
    "ckpt": "checkpoints/onset_ctx2.pt",
    "lane_ckpt": "checkpoints/lane_v7.pt",
    "plan_ckpt": "checkpoints/plan.pt",
    "threshold": 0.45,
    "max_chord": 6,
    "thin_weak": 0.0,
    "thin_local": 2,            # 16비트 폭타 줄이기 기준 범위: 앞뒤 N마디 (0 = 곡 전체)
    "snap_fix": 25.0,
    "stream_hole": 8.0,         # 폭타 속 한 칸 빈자리 보호: 곡 하위 N%만 뺌
    "end_shape": 1.0,           # 폭타 끝 모양, 두께를 사람처럼
    "superset_fix": 1.0,
    "edge_axis": 1.0,           # 1, 7번 축 풀기
    "quarter_spread": 1.0,      # 4비트 퍼짐
    "trill_break": 1.0,         # 한 손 트릴 끊기 (창에서는 숨김)
    "eighth_audit": 1.0,        # 8비트 비율 마지막 점검
    "hand_balance": 1.0,        # 폭타 밖 손 쏠림 풀기
    "fine_density": 1.0,        # 폭타 속 32, 24비트 밀도
    "pair_even": 1.0,           # 폭타 8분+16분 쌍 합 고르게
    "pair_min": 5,              # 쌍 합 목표 최소 (쌍 합 평균이 0일 때만)
    "pair_avg": 4.0,            # 폭타 쌍 합 평균
    "pair_dev": 1.5,            # 쌍 합 구간 편차 ±n
    "stream_fill": 11,          # 폭타 잇기: 16분 칸 N/16 이상 찬 마디는 빈칸 채움
    "beat_motif": 1.0,          # 박 반복
    "eighth_shape": 1.0,        # 8비트 모양 반복        # 16분 '앞 줄 품고 더하기'(45 -> 457) 풀기 (창에서는 숨김)           # 격자 보정: 이 ms 넘게 밀린 노트를 32, 24분 칸으로
    "note_log": 1,              # 노트 진단표 <이름>_notes_debug.csv (창에서는 숨김)
    "thin_boost": 0.7,
    "count_smooth": 0.8,
    "flow": 0.35,
    "scratch_add": 0.9,
    "stream_cap": 0.25,
    "on8_match": 1.0,
    "off_match": 1.0,           # 폭타 간격 두께: 창에서는 숨김, 기능은 켬
    "hand32": 1.0,              # 32분 한 손 몰림 풀기: 창에서는 숨김
    "jack16": 0.5,              # 16비트 연타 줄이기 (기본 0.5)
    "trip_match": 1.0,          # 셋잇단 두께 맞추기
    "skel_swap": 1.0,           # 폭타 뼈대
    "bar_repeat": 0.6,          # 마디 반복
    "bar_shape": 0.35,          # 마디 반복 중 모양까지 따라갈 확률 (창에서는 숨김)
    "diag": 0.25,               # 대각선
    "fine_single": 1.0,         # 32, 24분 채운 줄 단타로
    "fine_shape": 1.0,          # 24, 32분 연속 모양
    "section_dyn": 1.0,         # 구간 강약
    "eighth_trans": 1.0,        # 8비트 동시치기 이어짐 방식
    "end_fat": 1.0,             # 폭타 끝 묵직하게 (창에서는 숨김)
    "end_over": 1,              # 폭타 끝 줄만 상한 +N
    "fill_axis": 1.0,           # 채운 단타 축 풀기 (창에서는 숨김)
    "restart_shift": 1.0,       # 쉼 뒤 폭타 재시작 겹침 줄이기 (창에서는 숨김)
    "beat_pair": 1.0,           # 박 안 두 8분 자리 맞추기
    "edge4": 1.0,               # 가운데 몰린 4건반 풀기 (사람보다 많은 만큼만)
    "scratch_place": 1.0,       # 스크래치 자리 모델로 다시 놓기 (창에서는 숨김)
    "scratch_model": "checkpoints/scratch_pos.npz",
    "placer_ckpt": "checkpoints/placer_v1.pt",   # 자리 모델 (파일이 있으면 씀)
    "placer": 1.0,              # 자리 모델로 다시 뽑을 줄 비율
    "placer_temp": 1.0,         # 자리 모델 온도
    "placer_rep": 0.45,         # 자리 모델 반복 억제
    "stream_break": 16,         # 폭타 끊기
    "placer_beat": 3.0,         # 자리 모델 한 박 복사 억제
    "stream_target": 0.64,      # 폭타 마디 비율 목표
    "stream_kpr": 0.0,          # 폭타 줄당 건반 (기본 끔 - 얇은, 두꺼운 폭타 차이가 지워짐)
    "fill_grade": 1.0,          # 폭타 잇기 단계식 (창에서는 숨김)
    "accent_rest": 1.0,         # 치고 쉬기 (창에서는 숨김)
    "pair_keep": 1.0,           # 쌍 합 고르게가 마디 안 높낮이 남기기 (창에서는 숨김)
    "accent_rest_len": 1.0,     # 치고 쉬기 쉬는 길이(박) (0.75 -> 1, 창에서는 숨김)
    "dens_shape": 1.0,          # 구간 밀도 분포
    "dens_cap": 0.0,            # 최고 밀도 상한, 초당 노트 (기본 0 = 없음 - 사용자: 최고 밀도는 건들지 마)
    "stream_lift": 1.0,         # 폭타 두께 사람 분포로
    "guard": 0.0,               # 울타리 모드 (실험용, 창에서는 숨김 - 추가 옵션 --guard 1로 켬)
    "jack8": 1.0,               # 8분 간격 겹침 줄이기 (사람 BPM대 울타리)
    "trem": 0.0,                # 떨림(드르릉) 살리기 (실험 중이라 기본 끔)
    "shape_run": 1.0,           # 같은 모양 이어짐 끊기 (사람 기준)
    "row_seed": 1.0,            # 줄마다 고정 난수 (창에서는 숨김 - 한 곳을 고쳐도 나머지 배치가 그대로)
    "mel_keep": 1.0,            # 멜로디 지키기 (밀도 덜 때 멜로디 줄, 보컬 구간 지킴)
    "jack_allow": 0.3,          # 낮은 BPM 2연타 허용 (기본 0.3)
    "ladder": 1.0,              # 폭타 사다리 (폭타 구간 바꾸기, 폭타 끊기를 16분 줄 그대로 두께만 줄이는 쪽으로)
    "half_fill": 0.30,          # 소리 큰 곳 반쯤 빈 마디 채우기 (아깝게 빠진 16분 뒷자리, 모델 확률 기준)
    "four_one": 1.0,            # 4+1 섞기 (4 단계 폭타 중 4+1 사람 비율 35%까지)
    "dens_flat": 1.0,           # 소리 폭 맞춤 (소리 편차 적은 곡은 덜 비우고 조금 쉬는 폭타로)
    "bass_sound": 0.4,          # 소리에 베이스 넣기 (마디 소리 크기 = 멜 전체 0.6 + 저음 0.4)
    "dens_dev": 1.0,            # 편차 기준 (구간 밀도 목표를 곡 큰 소리 대비 편차로)
    "dens_pulse": 1.0,          # 8분 박 지키기 (분리 소리에 8분 박이 있으면 8분 뒷자리 줄은 안 뺌)
    "chord32": 0.0,             # 약한 32비트 동시치기로 (모델 확률 이 값 미만 32분 줄을 옆 16분 줄 동시치기로, 기본 끔 - 권장 0.45)
    "fill32_sound": 0.0,        # 채움 소리 확인 (16비트 이상 채우기는 분리 소리 어택이 있는 자리만, 기본 끔 - 권장 1.5)
    "rest_chord": 1.0,          # 쉬는 동시치기 두께 (소리 큰 곳 폭타 끊기, 바꾸기 동시치기 = 줄마다 주변 폭타 쌍 합 x 값)
    "fine_bpm": 1.0,            # 24, 32비트 BPM대 문턱 (곡 전체 4% 대신 박마다, 느린 곡 32, 24 쉽게, 200+ 8분 셋잇단 쉽게)
    "fine_grid": 1.0,           # 24, 32비트 격자 허용 (0 = 16분만, 1 = 자동, 2 = 다 허용)
    "key_scale": 1.0,           # 건반 배율 (줄은 그대로, 동시치기 크기만)
    "pillar_fix": 0.3,
    "eighth_overlap": 0.7,
    "eighth_same": 0.6,
    "eighth_big": 0.0,
    "scratch_bias": 1.5,
    "scratch_pos": 1.0,
    "fill32": 0.25,             # --fill32 그대로 (0 = 끔, 작을수록 많이 채움)
    "end_chord": True,          # 연타 끝 묵직하게: 항상 켬 (화면에는 안 보임)
    "click": False,
    "bpm": "",
    "title": "",
    "chart_name": "",           # 차분명: <이름>_<차분명>.bms로 따로 저장
    "artist": "",               # 작곡가 (#ARTIST, 없으면 빈칸)
    "judge": "NORMAL",          # 판정 (#RANK)
    "total": "300",             # #TOTAL
    "follow": "끔",             # 따라갈 악기 (곡 전체): 끔 / 드럼 / 악기 / 보컬
    "follow_at": "",            # 구간별 따라갈 악기 (한 곡만 만들 때), 예: 1-16 드럼, 17-32 악기 / 1~3초 보컬, 6~8초 드럼
    "follow_strength": 0.7,     # --follow-strength
    "seed": 0,
    "extra": "",
}

# (키, 이름, 최소, 최대, 간격, 요약 - 창에 한 줄로. 자세한 설명은 아래 DETAILS)
SLIDERS = [
    ("threshold", "노트 확률", 0.1, 0.95, 0.01, "클수록 노트가 적어요"),
    ("max_chord", "최대 동시치기 수", 1, 7, 1, "한 번에 칠 수 있는 최대 건반 수에요"),
    ("key_scale", "건반 배율", 0.5, 1.5, 0.05, "동시치기 크기만 이 배수로 덜어내요"),
    ("dens_shape", "구간 밀도 분포", 0, 1.5, 0.05, "조용한 구간이 사람 차분보다 빽빽하면 덜어내요"),
    ("stream_lift", "폭타 두께 맞추기", 0, 1.5, 0.05, "폭타 구간이 사람 차분보다 얇으면 늘려요"),
    ("dens_cap", "최고 밀도 상한(초당)", 0, 50, 1, "초당 노트 수 상한이에요. 이 값을 넘어가지 않아요. (0 = 없음)"),
    ("trem", "떨림(드르릉) 살리기", 0, 1.5, 0.05, "드르릉·떨림 구간을 24·32비트로 채워요"),
    ("mel_keep", "멜로디 지키기", 0, 2, 0.1, "밀도를 덜 때 멜로디·보컬 줄은 남겨요"),
    ("jack_allow", "낮은 BPM 2·3연타 허용", 0, 1, 0.05, "BPM 150 아래에서 2·3연타 일부를 허용해요"),
    ("fine_grid", "24·32비트 허용", 0, 2, 0.1, "셋잇단·32비트 그리드 사용 여부예요 (1 = 자동, 0 = 16비트만)"),
    ("fine_bpm", "24·32비트 BPM대 문턱", 0, 1, 0.05, "박마다 소리가 맞을 때만 24·32비트"),
    ("thin_weak", "16비트 폭타 줄이기", 0, 40, 5, "약한 16비트 뒷자리 노트를 빼요"),
    ("thin_local", "폭타 줄이기 기준 범위", 0, 8, 1, "'16비트 폭타 줄이기'를 앞뒤 N마디 기준으로"),
    ("stream_hole", "폭타 속 빈칸 지키기", 0, 20, 1, "폭타 중간에 한 칸만 비는 것 막기"),
    ("pair_even", "폭타 쌍 합 고르게", 0, 1, 0.05, "폭타 쌍 합을 마디 안에서 고르게"),
    ("pair_avg", "폭타 쌍 합 평균", 2, 7, 0.5, "폭타 8비트+16비트 건반 합의 평균 목표예요"),
    ("pair_dev", "폭타 쌍 합 편차 ±", 0, 3, 0.5, "소리가 센 폭타는 두껍게, 약한 폭타는 얇게 해요 ('폭타 쌍 합 평균' ± n)"),
    ("placer_temp", "자리 모델 온도", 0.3, 1.2, 0.05, "낮으면 흔한 모양, 높으면 다양하게 찍어요"),
    ("placer_rep", "자리 모델 반복 억제", 0, 2, 0.05, "A B A B 같은 반복을 덜 고르게"),
    ("placer_beat", "자리 모델 한 박 복사 억제", 0, 6, 0.5, "한 박 전·두 줄 전 복사와 단타 계단 줄이기"),
    ("edge_axis", "1·7번 축 풀기", 0, 1, 0.05, "1·7번에 몰린 축을 안쪽으로"),
    ("hand_balance", "폭타 밖 손 쏠림 풀기", 0, 1, 0.05, "폭타 밖 오른손 쏠림을 좌우 반전"),
    ("jack8", "8분 간격 겹침 줄이기", 0, 1, 0.05, "8분 간격 줄연타를 사람 수준으로"),
    ("shape_run", "같은 모양 이어짐 끊기", 0, 1, 0.05, "같은 모양이 미끄러지듯 이어지면 끊기"),
    ("fine_density", "폭타 속 32·24비트 밀도", 0, 1, 0.05, "폭타 속 32·24비트는 단타 위주로"),
    ("eighth_audit", "8비트 비율 마지막 점검", 0, 1, 0.05, "8비트 두께 비율이 넘치면 되돌리기"),
    ("stream_target", "폭타 마디 비율", 0, 1, 0.02, "폭타가 이 비율을 넘으면 가벼운 구간 넣기"),
    ("half_fill", "소리 큰 곳 빈 16비트 채우기", 0, 0.45, 0.01, "소리 큰 곳의 빈 16비트 채우기"),
    ("four_one", "4+1 섞기", 0, 1, 0.05, "4+2 폭타 일부를 4+1로"),
    ("ladder", "폭타 사다리", 0, 1, 1, "가벼운 구간을 두께만 줄인 폭타로 변환해요. ON/OFF 옵션이에요."),
    ("stream_kpr", "폭타 줄당 건반", 0, 3, 0.02, "폭타 줄당 건반을 이 값까지 (0 = 끔)"),
    ("dens_flat", "소리 폭 맞춤", 0, 1, 1, "강약이 적은 곡은 덜 비우기"),
    ("bass_sound", "소리에 베이스 넣기", 0, 1, 0.05, "소리 크기를 잴 때 베이스도 넣기"),
    ("dens_dev", "편차 기준", 0, 1, 1, "큰 소리 대비 얼마나 작은지로 밀도 정하기"),
    ("dens_pulse", "8분 박 지키기", 0, 1, 1, "8분 박이 들리는 마디는 8비트 유지"),
    ("fill32_sound", "채움 소리 확인", 0, 3, 0.1, "소리가 실제로 있는 자리만 32·24비트 채우기"),
    ("chord32", "약한 32비트 동시치기로", 0, 1, 0.05, "약하게 들린 32비트를 옆 줄과 합치기"),
    ("rest_chord", "쉬는 동시치기 두께", 0, 1.2, 0.05, "소리 큰 곳 쉬는 동시치기 두께"),
    ("stream_break", "폭타 끊기(마디)", 0, 40, 1, "폭타가 이 마디보다 길면 한 번 끊기"),
    ("stream_fill", "폭타 잇기", 0, 16, 1, "거의 찬 폭타 마디의 빈칸 채우기"),
    ("beat_motif", "박 반복", 0, 1, 0.05, "앞 박과 같은 두께 패턴 반복"),
    ("eighth_shape", "8비트 모양 반복", 0, 1, 0.05, "8비트 동치를 A B A B·옆으로 옮기기로"),
    ("end_shape", "폭타 끝 모양", 0, 1, 0.05, "폭타 끝 줄 크기·모양을 소리대로"),
    ("snap_fix", "격자 보정(ms)", 0, 40, 5, "멀리 밀린 노트를 맞는 칸에 다시 놓기"),
    ("thin_boost", "16비트 줄인 자리 메우기", 0, 1, 0.1, "뺀 16비트 옆 동시치기를 두껍게"),
    ("count_smooth", "동시치기 개수 고르게", 0, 1, 0.1, "8비트 동시치기 개수 튐 줄이기"),
    ("on8_match", "폭타 두께", 0, 1, 0.05, "폭타 8비트 자리 두께를 ★20 기준으로"),
    ("stream_cap", "16비트 큰 동시치기 줄이기", 0, 1, 0.05, "폭타 속 4개+ 동시치기 덜기"),
    ("trip_match", "셋잇단 두께 맞추기", 0, 1, 0.05, "셋잇단 두께를 사람처럼"),
    ("bar_repeat", "마디 반복", 0, 1, 0.05, "비슷한 마디는 앞 마디처럼"),
    ("diag", "대각선", 0, 0.5, 0.05, "폭타에 대각선 모양 만들기"),
    ("fine_single", "32·24비트 단타로", 0, 1, 0.05, "채운 32·24비트 줄을 단타로"),
    ("eighth_trans", "8비트 동치 이어짐", 0, 1, 0.05, "8비트 동치가 이어지는 방식 정리"),
    ("end_over", "폭타 끝 상한 +", 0, 2, 1, "폭타 마지막 줄만 상한 + N"),
    ("section_dyn", "구간 강약", 0, 1.5, 0.05, "조용한 줄은 얇게, 큰 줄은 두껍게 해요"),
    ("fine_shape", "32·24비트 연속 모양", 0, 1, 0.05, "32·24비트 연속을 계단 등 모양으로"),
    ("edge4", "가운데 4건반 풀기", 0, 1, 0.05, "가운데 몰린 4건반을 1·7번으로"),
    ("fill32", "16비트 이상으로 채우기", 0, 0.45, 0.01, "약한 소리를 32·24비트로 채우는 기준이에요"),
    ("scratch_bias", "스크래치 성향", -1, 2.5, 0.1, "스크래치를 고르는 정도예요"),
    ("scratch_add", "스크래치 최소량", 0, 1.5, 0.1, "마디당 최소 스크래치 수에요"),
]
# 슬라이더 자세한 설명 - 창에는 요약(SLIDERS 마지막 칸)만 한 줄로, 이름, 요약에 마우스를 올리면 이 글이 뜸
DETAILS = {
    "threshold": "이 확률 이하의 노트는 찍지 않아요 (기본 0.45)",
    "max_chord": "한 줄에 놓는 최대 건반 수예요. 스크래치는 최댓값에 포함되지 않아요. (기본 6)",
    "key_scale": "줄 개수와 관계 없이 동시치기 크기만 덜어요. 0.8 = 큰 동시치기부터 20% 감소, 1.2 = 단타 스트림의 20%가 동시치기로 전환",
    "dens_shape": "2마디마다 소리가 곡의 큰 소리보다 얼마나 작은지(편차)를 재서, 사람 차분의 밀도와 맞춰요. 지금 밀도가 그보다 높으면 덜어요. 조용하면 노트 찍는 줄을 빼고, 세면 건반을 빼요. 폭타는 건드리지 않아서 '폭타 두께 맞추기' 옵션과 겹치지 않아요. 값이 높을수록 밀도가 낮아져요. 1 = 사람 기준, 0 = 끔.",
    "stream_lift": "폭타 마디마다 줄당 건반 수를 재서, 소리가 큰 폭타일수록 두껍게 찍어요. 같은 BPM대 사람 차분 두께에 맞춰요. 건반만 더하는 옵션이라 노트 찍는 줄이 빠지는 경우는 없어요. 가장 두꺼운 폭타와 최고 밀도는 그대로예요. 값이 높을수록 밀도가 높아져요. 1 = 사람 기준, 0 = 끔.",
    "dens_cap": "넘는 구간은 동시치기부터 덜어요. (★16~21 기준 중앙값 31, 상위 25% 33) 기본 0 = 상한 없음. '구간 밀도 분포'가 0이면 안 써요.",
    "trem": "드르릉·고음 떨림처럼 소리는 촘촘한데 모델 확신이 낮아 빠지던 박을 24·32비트로 채워요. 사이 칸은 대부분 단타, 박 머리는 사람 떨림 구간처럼 동시치기도 섞어요. 값이 높을수록 덜 촘촘한 곳까지 떨림으로 봐요. 0 = 끔.",
    "mel_keep": "'구간 밀도 분포'나 '폭타 마디 비율' 옵션으로부터 멜로디·보컬·베이스를 지켜요. 음원 분리 결과가 있을 때만 적용 가능해요. '곡 정보'의 '따라갈 악기' 옵션을 한 번 사용하면 생겨요. 값이 높을수록 지키는 줄이 많아요. 1 = 기본, 0 = 끔.",
    "jack_allow": "16비트 간격이 100ms 이상인 곡의 빽빽한 폭타 마디에서, 앞 줄과 같은 소리가 반복되는 곳에만 16비트 2·3연타를 일부 허용해요. 건반 수는 변하지 않아요. 값이 높을수록 연타가 많아져요. 0.3 = 기본, 0 = 끔.",
    "fine_grid": "모델이 노트를 박자 칸에 붙일 때, 소리가 셋잇단·32비트에 더 어울리다고 판단하면 그 격자에 찍어요. 값을 낮추면 더 엄격하게 보고 높일수록 느슨하게 봐요. 즈레·셋잇단이 과하면 낮추세요",
    "fine_bpm": "셋잇단·32비트를 박마다 소리에 확실히 맞을 때만 써요. BPM대별 사람 쓰임에 맞춰 느린 곡은 32비트를 쉽게, 빠른 곡은 어렵게. 1 = 사람 기준, 0 = 곡 전체 문턱(셋잇단·32비트 박이 이어진 구간이 곡의 4% 이상일 때만 그 격자를 써요).",
    "thin_weak": "16분 뒷자리 노트 중 모델 확률이 앞뒤 2마디 노트의 하위 N%보다 낮은 것을 빼요. 폭타 속에 한 칸만 비는 곳은 남겨요. 값이 높을수록 많이 빠져요. 0 = 끔(기본).",
    "thin_local": "하위 N%를 앞뒤 이만큼의 마디 안에서 정해, 조용한 구간이 통째로 깎이지 않게 해요. 0 = 곡 전체.",
    "stream_hole": "'16비트 폭타 줄이기'가 폭타 속 16분 한 칸만 비게 만들 줄은 확률이 곡 하위 N%일 때만 빼요. 작을수록 덜 비어요. 사람 기준 8, 0 = 보호 끔.",
    "pair_even": "폭타 8비트 자리 줄 + 바로 뒤 16비트 줄 건반 합을 마디 안에서 비슷하게(5, 5, 6, 5…) 맞춰요. 1 = 사람 기준, 0 = 끔(쌍 합 평균도 꺼짐).",
    "pair_avg": "폭타에서 8비트 자리 줄과 바로 뒤 16비트 줄의 건반 합을 이 값에 맞추려고 해요. 클수록 폭타가 두꺼워요. 사람 곡 평균 2.7~4.6, 기본 4. 끄려면 '폭타 쌍 합 고르게'를 0으로.",
    "pair_dev": "4마디마다 소리가 가장 센 곳은 평균 + n, 가장 약한 곳은 평균 - n. 클수록 강약이 커져요. 사람 ±1.5 근처, 0 = 곡 전체 같은 값.",
    "placer_temp": "'자리 모델'의 적용 정도에요. 낮을 수록 사람이 자주 찍는 특정 패턴 위주로 찍어요. (기본 1.0)",
    "placer_rep": "두 줄 간격으로 같은 조합(1 _ 1 _ 1, A B A B)이 이어질수록 자리 모델이 덜 골라요. 기본 0.45, 0 = 모델 그대로.",
    "placer_beat": "한 박 전·두 줄 전과 똑같은 줄, 단타 계단(1234)이 사람보다 많아지면 자리 모델이 덜 골라요. 기본 3, 0 = 끔.",
    "edge_axis": "8분 자리 3줄 연속 같은 건반(축)이 1·7번에 몰리면 가운데 줄을 안쪽 건반으로 옮겨 사람 비율(약 40%)로 맞춰요. 0 = 끔.",
    "hand_balance": "폭타가 아닌 곳에서 오른손만 친 줄이 왼손만 친 줄의 1.42배(사람)를 넘으면 넘친 만큼 좌우 반전해요. 0 = 끔.",
    "jack8": "바로 앞 8분 줄과 같은 건반(줄연타)이 같은 BPM대 사람 범위를 넘으면 겹친 건반만 옆으로 옮겨요. 16분 같은 건반 3줄 연속도. 건반 수 그대로. 0 = 끔.",
    "shape_run": "폭타에서 같은 모양 동시치기가 한 칸씩 미끄러지며(36 57 46 35) 3줄 넘게 이어지는 곳이 사람보다 많으면 모양을 바꿔요. 건반 수 그대로. 0 = 끔.",
    "fine_density": "폭타 속 32·24비트 줄을 단타 위주(사람 84%)로, 그 박이 너무 빽빽하면 앞뒤 16비트 뒷자리에서 덜어요. 0 = 끔.",
    "eighth_audit": "다른 규칙이 겹쳐 '박 안 두 8비트 같음'·'8비트 자리가 더 두꺼움' 비율이 사람보다 넘치면 넘친 만큼만 되돌려요. 0 = 끔.",
    "stream_target": "곡 전체 폭타 마디 비율이 이 값을 넘으면 4~8마디를 가벼운 구간으로 바꿔요. 조용한 곳부터 드문드문 / 8비트 단타 / 사다리 폭타. 사람 폭타 위주 곡 64%, 0 = 끔.",
    "half_fill": "소리 큰 마디인데 16비트가 반쯤 비고 앞뒤가 폭타면, 빈 뒷자리를 모델 확률이 이 값 이상일 때 채워요. 일부러 비운 구간은 그대로. 낮을수록 많이, 기본 0.30, 0 = 끔.",
    "four_one": "4개 동시치기 폭타 중 4+1 비율을 사람(35%) × 값까지 올려요. 소리 작은 마디부터. 1 = 사람 비율, 0 = 끔.",
    "ladder": "폭타가 너무 많을 때 넣는 가벼운 구간을, 16비트 줄은 그대로 두고 두께만 줄인 폭타(1+1 / 2+1 / 2+2)로 만들어요. 폭타 끊기도 같은 방식. 1 = 켬(기본), 0 = (8비트 동시치기).",
    "stream_kpr": "남은 폭타 마디의 줄당 건반을 이 값까지 올려요. 소리 큰 마디일수록 두껍게(사람 2.28). 켜면 얇은·두꺼운 폭타 차이가 줄어요. 기본 0 = 끔.",
    "dens_flat": "곡 강약 폭이 비슷한 사람 곡에 맞춰 밀도를 덜어요. 강약이 적은 곡은 가벼운 구간도 덜 비우고, 줄보다 건반부터 덜어요. 1 = 켬(기본), 0 = 강약 폭과 상관없이 모든 곡에 같은 기준(가장 가벼운 구간 = 최고의 0.22배).",
    "bass_sound": "마디 소리 크기를 잴 때 저음(베이스)을 이 비율로 넣어요. 베이스가 큰 곳이 얇은 폭타가 되지 않게. 기본 0.4, 0 = 전체 소리(멜)만 써요.",
    "dens_dev": "구간 밀도 목표를 곡 안 순위 대신 '곡에서 큰 소리보다 얼마나 작은가'로 정해요. 강약이 적은 곡은 조용한 구간도 덜 비워요. 1 = 켬(기본), 0 = 곡 안 순위(소리 크기·밀도 순위 반반)로 목표를 정해요.",
    "dens_pulse": "분리 소리에 8분 박이 또렷한 마디는 밀도를 덜 때 8분 줄은 남기고 건반만 덜어요(4분 단타가 되지 않게). 분리 소리가 있을 때만. 1 = 켬(기본), 0 = 끔.",
    "fill32_sound": "16비트 이상 채우기가 더할 자리에 분리 소리 어택이 주변 대비 이 배 이상 있을 때만 더해요. 높을수록 엄격, 권장 1.5, 0 = 끔(기본). 분리 소리가 있을 때만.",
    "chord32": "모델 확률이 이 값 미만인 32비트 줄을 바로 옆 16비트 줄에 합쳐 동시치기로 만들어요. 또렷한 롤은 그대로, 건반 수도 그대로. 높을수록 많이 합침, 권장 0.45, 0 = 끔(기본).",
    "rest_chord": "소리 큰 곳에서 폭타를 끊고 넣는 8비트 동시치기의 건반 수 = 주변 폭타 쌍 합 × 값(쌍 합 5.5면 5~6개). 최대 동시치기 수는 안 넘어요. 기본 1, 0 = 소리 크기와 상관없이 주변 폭타의 0.76배.",
    "stream_break": "폭타가 이 마디 수보다 길게 이어지면 16비트가 약한 마디를 가볍게 바꿔 끊어요. 사람 폭타 최장 연속 17마디, 기본 16, 0 = 끔.",
    "stream_fill": "16분 칸이 이 수(16칸 중) 이상 찬 마디는 빈 칸을 채워 폭타가 끊기지 않게 해요. 작을수록 많이 채워요. 기본 11, 0 = 끔.",
    "beat_motif": "리듬이 같은 앞 박과 줄별 동시치기 수를 같게(일부는 모양까지) 해서 박 단위 반복을 만들어요. 0 = 끔.",
    "eighth_shape": "8비트 동시치기를 두 줄 전 그대로(A B A B) 쓰거나 같은 모양을 옆으로 옮겨 정돈해요. 0 = 끔.",
    "end_shape": "폭타 끝 줄을 소리가 크면 크게(4~5개), 작으면 가볍게(1~2개), 모양은 사람 비율대로 놓아요. 0 = 끔.",
    "snap_fix": "박 격자에 붙일 때 이 ms보다 멀리 밀린 노트를 32·24비트·셋잇단 칸에 다시 놓고, 합쳐져 사라질 노트를 살려요. 0 = 끔.",
    "thin_boost": "'16비트 폭타 줄이기'로 뺀 자리 옆 8비트 줄에 건반을 하나 더할 확률이에요.",
    "count_smooth": "8비트 동시치기 개수가 튀면(4 1 3) ±1 안으로 맞출 확률이에요.",
    "on8_match": "폭타 8비트 자리 동시치기 수를 ★20 사람 분포에 맞춰요. 클수록 더 맞춰요(대개 줄어듦). 0 = 끔.",
    "stream_cap": "폭타 8비트 자리의 4개 이상 동시치기에서 건반마다 이 확률로 덜어요. 최소 1개는 남겨요.",
    "trip_match": "4분 셋잇단(6비트) 반복은 동시치기 수를 고르게, 16분 셋잇단(24비트) 연타는 단타 위주로. 0 = 끔.",
    "bar_repeat": "리듬·소리가 앞 마디와 비슷하면 같은 자리 동시치기 수(일부는 모양까지)를 앞 마디처럼 할 확률이에요. 0 = 끔.",
    "diag": "폭타에서 두 줄 앞 모양을 한 칸 옮긴 대각선 모양을 만들 확률이에요. 0 = 끔.",
    "fine_single": "16비트 사이를 채운 32·24비트 줄을 단타로 만드는 세기예요. 1 = 단타 90%, 0 = 끔.",
    "eighth_trans": "8비트 동시치기가 이어질 때 그대로 / 하나만 이동 / 평행이동 / 새 자리 중 하나로 사람 비율대로 바꿔요. 0 = 끔.",
    "end_over": "폭타 마지막 줄만 최대 동시치기 수보다 이만큼 더 허용해요. 몸통은 그대로. 0 = 상한 그대로.",
    "section_dyn": "구간 소리 크기에 따라 동시치기 두께를 바꿔요. 줄이면 강약이 약해져 어려워져요. 1 = 사람 기준, 0 = 끔.",
    "fine_shape": "32·24비트 3줄 이상 연속을 계단 / 좌우 교대 / 지그재그 / 좁은 반복 중 하나로 놓을 확률이에요. 0 = 끔.",
    "edge4": "2356·2346처럼 1·7 없이 가운데 몰린 4건반이 사람(7%)보다 많으면 넘는 만큼 끝 건반을 1·7번으로 옮겨요. 0 = 끔.",
    "fill32": "'노트 확률'보다 낮아 빠진 소리 중, 확실한 두 노트 사이에 있는 약한 소리를 32비트(16비트 사이)·24비트(셋잇단 사이)로 채워요. 값이 작을수록 많이 바뀌어요.  0.1 = 많이, 0.25 = 기본, 0 = 끔. ‘24·32비트 허용'이 0이면 안 써요.",
    "scratch_bias": "레인 모델이 스크래치를 고르는 정도예요. 클수록 많아요. 기본 1.5, 0 = 모델 그대로.",
    "scratch_add": "스크래치가 마디당 이 수보다 적으면 어울리는 자리에 채워요. (기본 0.9)",
}
# 화면에는 안 보이지만 명령에는 들어가는 값 (DEFAULTS / 설정 파일 값 그대로): flow, eighth_overlap, eighth_same, scratch_pos
HIDDEN = ("flow", "eighth_overlap", "eighth_same", "scratch_pos", "off_match", "hand32", "scratch_place", "bar_shape",
          "fill_axis", "restart_shift", "end_fat", "note_log", "superset_fix", "trill_break",
          # 창에서 뺌 (효과 없음 / 다른 설정과 겹침 / 기본 0 / Demucs 뒤 정리할 과보정 규칙) - 기본값 그대로 명령에 들어감
          "pair_min", "pillar_fix", "eighth_big", "jack16", "skel_swap", "beat_pair",
          # 폭타 잇기 단계식 (기본 1), 치고 쉬기, 쌍 합 높낮이 남기기 (기본 1)
          "fill_grade", "accent_rest", "pair_keep", "accent_rest_len", "guard", "row_seed",
          # 사용자가 창에서 숨김 (자리 모델, 4비트 퍼짐) - 기본값 그대로 명령에 들어감
          "placer", "quarter_spread")
# 첫 화면에 보이는 슬라이더 (곡마다 바꿀 만한 것). 나머지 SLIDERS는 '고급 설정 보기' 안 '세부 규칙'에
# 구간 밀도 분포, 폭타 두께 맞추기, 최고 밀도 상한, 멜로디 지키기, 2, 3연타 허용, 24, 32비트 허용, 폭타 사다리, 구간 강약은 세부 규칙으로
MAIN_SLIDERS = ("threshold", "max_chord", "key_scale", "trem", "thin_weak", "fill32", "pair_avg", "pair_dev",
                "placer_temp", "scratch_bias", "scratch_add")

_BASE = {"threshold": 0.45, "max_chord": 6, "thin_weak": 0.0, "thin_boost": 0.7, "count_smooth": 0.8, "flow": 0.35,
         "fill32": 0.25, "scratch_bias": 1.5, "scratch_pos": 1.0, "scratch_add": 0.9,
         "stream_cap": 0.25, "pillar_fix": 0.3, "eighth_overlap": 0.7, "eighth_big": 0.0, "on8_match": 1.0,
         "off_match": 1.0, "eighth_same": 0.6, "hand32": 1.0, "jack16": 0.5, "trip_match": 1.0,
         "skel_swap": 1.0, "bar_repeat": 0.6, "bar_shape": 0.35, "diag": 0.25,
         "fine_single": 1.0, "fine_shape": 1.0, "beat_pair": 1.0, "edge4": 1.0, "section_dyn": 1.0,
         "eighth_trans": 1.0, "end_over": 1, "thin_local": 2, "snap_fix": 25.0, "stream_hole": 8.0, "end_shape": 1.0, "beat_motif": 1.0, "eighth_shape": 1.0, "stream_fill": 11, "edge_axis": 1.0, "pair_even": 1.0, "pair_min": 5, "placer": 1.0, "placer_temp": 1.0, "pair_avg": 4.0, "pair_dev": 1.5, "quarter_spread": 1.0, "trill_break": 1.0, "eighth_audit": 1.0, "hand_balance": 1.0, "fine_density": 1.0, "placer_rep": 0.45, "stream_break": 16, "placer_beat": 3.0, "stream_target": 0.64, "stream_kpr": 0.0, "fill_grade": 1.0, "accent_rest": 1.0, "pair_keep": 1.0, "fine_grid": 1.0, "key_scale": 1.0, "accent_rest_len": 1.0, "dens_shape": 1.0, "dens_cap": 0.0, "stream_lift": 1.0, "guard": 0.0, "jack8": 1.0, "trem": 0.0, "shape_run": 1.0, "row_seed": 1.0, "jack_allow": 0.3, "mel_keep": 1.0, "ladder": 1.0, "half_fill": 0.30, "four_one": 1.0, "fine_bpm": 1.0, "rest_chord": 1.0, "dens_flat": 1.0, "bass_sound": 0.4, "dens_dev": 1.0, "dens_pulse": 1.0, "chord32": 0.0, "fill32_sound": 0.0}
PRESETS = {
    "기본": dict(_BASE),
    "밀도 줄이기": {**_BASE, "threshold": 0.50, "thin_weak": 20.0, "key_scale": 0.85},
    "밀도 늘리기": {**_BASE, "threshold": 0.40, "scratch_add": 1.0, "key_scale": 1.15},
}


JUDGES = {"EASY": "easy", "NORMAL": "normal", "HARD": "hard", "VERY HARD": "veryhard"}   # 화면 이름 -> infer --judge
FOLLOW = {"끔": "off", "드럼": "drum", "악기": "inst", "보컬": "vocal"}                                    # 화면 이름 -> infer --follow


# 로그 줄 앞부분 -> 진행 단계 표시
STAGES = [
    ("오디오:", "소리 읽는 중", 0.10),
    ("모델:", "노트 타이밍 찾는 중", 0.20),
    ("추정 BPM", "BPM 찾음", 0.45),
    ("레인 모델", "건반 배치 준비", 0.55),
    ("격자 정리", "박자에 맞추는 중", 0.65),
    ("스크래치:", "건반 배치 중", 0.75),
    ("예측한 노트 시각", "마무리 중", 0.95),
]


def ascii_name(text: str) -> str:
    """infer.py safe_name과 같은 규칙 (결과 폴더 이름)"""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_") or "song"


def cp932_ok(text: str) -> bool:
    try:
        text.encode("cp932")
        return True
    except UnicodeEncodeError:
        return False


def load_settings() -> dict:
    s = dict(DEFAULTS)
    raw = {}
    try:
        raw = json.loads(SETTINGS.read_text(encoding="utf-8"))
        s.update({k: v for k, v in raw.items() if k in DEFAULTS})
    except Exception:
        pass
    if isinstance(s.get("fill32"), bool):    # 아주 예전 설정 파일(체크박스 시절)
        s["fill32"] = 0.25 if s["fill32"] else 0.0
    s["end_chord"] = True                   # 연타 끝 묵직하게는 항상 켬 (예전 설정 파일에 꺼져 있어도)
    for k in HIDDEN:                        # 창에 없는 값은 항상 기본값 (예전 설정 파일의 off_match 0 등 무시)
        s[k] = DEFAULTS[k]
    if not raw.get("_v93"):                 # 폭타 줄당 건반 기본이 2.28 -> 0(끔)으로 바뀜 - 예전 설정 파일 값은 무시
        s["stream_kpr"] = DEFAULTS["stream_kpr"]
    try:                                    # stream_cap 뜻이 바뀜(건반마다 덜 확률 0~1) -> 예전 값은 기본값으로
        if not raw.get("_v46") or not 0 <= float(s["stream_cap"]) <= 1:
            s["stream_cap"] = DEFAULTS["stream_cap"]
    except Exception:
        s["stream_cap"] = DEFAULTS["stream_cap"]
    return s


def abs_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else HERE / q


def folder_name(text: str) -> str:
    """윈도우 폴더 이름에 못 쓰는 글자만 바꿈 (한글/일본어는 그대로)"""
    t = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).strip().rstrip(". ")
    return t or "song"


def unique_folder(out: str, audio: str) -> str:
    """결과 폴더 이름: <저장 폴더>/<곡 이름>. 같은 곡을 다시 만들면 같은 폴더에 덮어씀 (패턴, 클릭 음원, 로그)"""
    return folder_name(Path(audio).stem)


def infer_args(audio: str, s: dict, single: bool, folder: str | None = None) -> list[str]:
    """infer.py에 넘길 옵션 (명령 복사/로그에는 'python infer.py ' + 이것)"""
    cmd = ["--audio", str(audio),
           "--ckpt", str(abs_path(s["ckpt"])), "--lane-ckpt", str(abs_path(s["lane_ckpt"])),
           "--plan-ckpt", str(abs_path(s["plan_ckpt"])),
           # 고정 권장값 (지금까지 맞춰 온 설정)
           "--snap", "beat", "--jack-penalty", "3", "--no-section-fix",
           "--thick", "0", "--chord-style", "0", "--beat-contrast", "0",
           "--spread-chords", "0.2", "--offgrid-keep", "0",
           # 슬라이더/체크
           "--threshold", f"{float(s['threshold']):.2f}",
           "--max-chord", str(int(round(float(s.get('max_chord', 6))))),
           "--thin-weak", f"{float(s['thin_weak']):g}",
           "--thin-local", f"{float(s.get('thin_local', 2)):g}",
           "--snap-fix", f"{float(s.get('snap_fix', 25.0)):g}",
           "--note-log", str(int(round(float(s.get('note_log', 1))))),
           "--stream-hole", f"{float(s.get('stream_hole', 8.0)):g}",
           "--end-shape", f"{float(s.get('end_shape', 1.0)):g}",
           "--superset-fix", f"{float(s.get('superset_fix', 1.0)):g}",
           "--pair-even", f"{float(s.get('pair_even', 1.0)):g}",
           "--placer-ckpt", (str(abs_path(s.get("placer_ckpt") or DEFAULTS["placer_ckpt"]))
                             if abs_path(s.get("placer_ckpt") or DEFAULTS["placer_ckpt"]).exists() else "none"),
           "--placer", f"{float(s.get('placer', 1.0)):g}",
           "--placer-temp", f"{float(s.get('placer_temp', 1.0)):g}",
           "--placer-rep", f"{float(s.get('placer_rep', 0.45)):g}",
           "--stream-break", str(int(round(float(s.get('stream_break', 16))))),
           "--placer-beat", f"{float(s.get('placer_beat', 3.0)):g}",
           "--stream-target", f"{float(s.get('stream_target', 0.64)):g}",
           "--stream-kpr", f"{float(s.get('stream_kpr', 0.0)):g}",
           "--fill-grade", f"{float(s.get('fill_grade', 1.0)):g}",
           "--accent-rest", f"{float(s.get('accent_rest', 1.0)):g}",
           "--pair-keep", f"{float(s.get('pair_keep', 1.0)):g}",
           "--accent-rest-len", f"{float(s.get('accent_rest_len', 1.0)):g}",
           "--fine-grid", f"{float(s.get('fine_grid', 1.0)):g}",
           "--dens-shape", f"{float(s.get('dens_shape', 1.0)):g}",
           "--dens-cap", f"{float(s.get('dens_cap', 0.0)):g}",
           "--stream-lift", f"{float(s.get('stream_lift', 1.0)):g}",
           "--guard", f"{float(s.get('guard', 0.0)):g}",
           "--jack8", f"{float(s.get('jack8', 1.0)):g}",
           "--trem", f"{float(s.get('trem', 0.0)):g}",
           "--shape-run", f"{float(s.get('shape_run', 1.0)):g}",
           "--row-seed", f"{float(s.get('row_seed', 1.0)):g}",
           "--jack-allow", f"{float(s.get('jack_allow', 0.3)):g}",
           "--mel-keep", f"{float(s.get('mel_keep', 1.0)):g}",
           "--ladder", f"{float(s.get('ladder', 1.0)):g}",
           "--half-fill", f"{float(s.get('half_fill', 0.30)):g}",
           "--four-one", f"{float(s.get('four_one', 1.0)):g}",
           "--fine-bpm", f"{float(s.get('fine_bpm', 1.0)):g}",
           "--rest-chord", f"{float(s.get('rest_chord', 1.0)):g}",
           "--dens-flat", f"{float(s.get('dens_flat', 1.0)):g}",
           "--bass-sound", f"{float(s.get('bass_sound', 0.4)):g}",
           "--dens-dev", f"{float(s.get('dens_dev', 1.0)):g}",
           "--dens-pulse", f"{float(s.get('dens_pulse', 1.0)):g}",
           "--chord32", f"{float(s.get('chord32', 0.0)):g}",
           "--fill32-sound", f"{float(s.get('fill32_sound', 0.0)):g}",
           "--key-scale", f"{float(s.get('key_scale', 1.0)):g}",
           "--pair-avg", f"{float(s.get('pair_avg', 4.0)):g}",
           "--pair-dev", f"{float(s.get('pair_dev', 1.5)):g}",
           "--pair-min", str(int(round(float(s.get('pair_min', 5))))),
           "--edge-axis", f"{float(s.get('edge_axis', 1.0)):g}",
           "--quarter-spread", f"{float(s.get('quarter_spread', 1.0)):g}",
           "--trill-break", f"{float(s.get('trill_break', 1.0)):g}",
           "--eighth-audit", f"{float(s.get('eighth_audit', 1.0)):g}",
           "--hand-balance", f"{float(s.get('hand_balance', 1.0)):g}",
           "--fine-density", f"{float(s.get('fine_density', 1.0)):g}",
           "--stream-fill", f"{float(s.get('stream_fill', 11)):g}",
           "--beat-motif", f"{float(s.get('beat_motif', 1.0)):g}",
           "--eighth-shape", f"{float(s.get('eighth_shape', 1.0)):g}",
           "--thin-boost", f"{float(s['thin_boost']):g}",
           "--count-smooth", f"{float(s['count_smooth']):g}",
           "--flow", f"{float(s['flow']):g}",
           "--scratch-add", f"{float(s['scratch_add']):g}",
           "--scratch-bias", f"{float(s['scratch_bias']):g}",
           "--scratch-pos", f"{float(s['scratch_pos']):g}",
           "--stream-cap", f"{float(s['stream_cap']):g}",
           "--on8-match", f"{float(s['on8_match']):g}",
           "--off-match", f"{float(s['off_match']):g}",
           "--hand32", f"{float(s['hand32']):g}",
           "--jack16", f"{float(s['jack16']):g}",
           "--trip-match", f"{float(s.get('trip_match', 1.0)):g}",
           "--skel-swap", f"{float(s.get('skel_swap', 1.0)):g}",
           "--bar-repeat", f"{float(s.get('bar_repeat', 0.6)):g}",
           "--bar-shape", f"{float(s.get('bar_shape', 0.35)):g}",
           "--diag", f"{float(s.get('diag', 0.25)):g}",
           "--fine-single", f"{float(s.get('fine_single', 1.0)):g}",
           "--fine-shape", f"{float(s.get('fine_shape', 1.0)):g}",
           "--fill-axis", f"{float(s.get('fill_axis', 1.0)):g}",
           "--section-dyn", f"{float(s.get('section_dyn', 1.0)):g}",
           "--eighth-trans", f"{float(s.get('eighth_trans', 1.0)):g}",
           "--end-fat", f"{float(s.get('end_fat', 1.0)):g}",
           "--end-over", str(int(round(float(s.get('end_over', 1))))),
           "--restart-shift", f"{float(s.get('restart_shift', 1.0)):g}",
           "--beat-pair", f"{float(s.get('beat_pair', 1.0)):g}",
           "--edge4", f"{float(s.get('edge4', 0.6)):g}",
           "--scratch-place", f"{float(s.get('scratch_place', 1.0)):g}",
           "--scratch-model", (str(abs_path(s.get("scratch_model") or DEFAULTS["scratch_model"]))
                               if abs_path(s.get("scratch_model") or DEFAULTS["scratch_model"]).exists() else "none"),
           "--eighth-same", f"{float(s['eighth_same']):g}",
           "--pillar-fix", f"{float(s['pillar_fix']):g}",
           "--eighth-overlap", f"{float(s['eighth_overlap']):g}",
           "--eighth-big", f"{float(s['eighth_big']):g}",
           "--fill32", f"{float(s['fill32']):g}",
           "--stream-end-chord", "1" if s["end_chord"] else "0",
           "--seed", str(int(s["seed"])),
           "--judge", JUDGES.get(str(s.get("judge", "NORMAL")), "normal"),
           "--total", f"{float(s.get('total') or 300):g}",
           "--follow", FOLLOW.get(str(s.get("follow", "끔")), "off"),
           "--follow-strength", f"{float(s.get('follow_strength', 0.7)):g}",
           "--out", str(s["out"])]
    if folder:
        cmd += ["--folder", folder]
    if s["click"]:
        cmd += ["--key-sound", "click"]
    stem = Path(audio).stem
    if single and str(s["bpm"]).strip():
        cmd += ["--bpm", str(s["bpm"]).strip()]
    if single and str(s.get("follow_at", "")).strip():
        cmd += ["--follow-at", str(s["follow_at"]).strip()]
    if single and str(s["title"]).strip():
        cmd += ["--title", str(s["title"]).strip()]
    elif cp932_ok(stem):                        # 일본어/영어 곡 이름은 그대로 제목으로 (BMS는 Shift-JIS라 한글은 깨짐)
        cmd += ["--title", stem]
    if str(s.get("chart_name", "")).strip():
        cmd += ["--chart-name", str(s["chart_name"]).strip()]
    if str(s.get("artist", "")).strip():
        cmd += ["--artist", str(s["artist"]).strip()]
    if str(s["extra"]).strip():
        toks = shlex.split(s["extra"], posix=(os.name != "nt"))
        cmd += [t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t for t in toks]
    return cmd


def build_cmd(audio: str, s: dict, single: bool, folder: str | None = None) -> list[str]:
    """실제로 실행할 명령: 이 프로그램 자신을 --run-infer 모드로 (exe든 파이썬이든 같은 방식)"""
    me = [sys.executable] if FROZEN else [sys.executable, "-u", str(Path(__file__).resolve())]
    return me + ["--run-infer"] + infer_args(audio, s, single, folder)


def show_cmd(cmd: list[str]) -> str:
    out = []
    for c in cmd:
        out.append(f'"{c}"' if (" " in c or not c) else c)
    return " ".join(out)


# ---------- 생성 로그에서 설정 불러오기 ----------
LOG_FLOAT_KEYS = ("threshold", "thin_weak", "thin_boost", "count_smooth", "flow", "scratch_add", "scratch_bias", "scratch_pos",
                  "stream_cap", "on8_match", "off_match", "hand32", "jack16", "trip_match", "skel_swap", "bar_repeat",
                  "bar_shape", "diag", "fine_single", "fine_shape", "fill_axis", "section_dyn", "eighth_trans", "end_fat",
                  "restart_shift", "beat_pair", "edge4", "scratch_place", "eighth_same", "pillar_fix", "eighth_overlap",
                  "eighth_big", "fill32", "thin_local", "snap_fix", "stream_hole", "end_shape", "superset_fix", "beat_motif", "eighth_shape", "stream_fill", "edge_axis", "pair_even", "placer", "placer_temp", "pair_avg", "pair_dev", "quarter_spread", "trill_break", "eighth_audit", "hand_balance", "fine_density", "placer_rep", "placer_beat", "stream_target", "stream_kpr", "fill_grade", "accent_rest", "pair_keep", "fine_grid", "key_scale", "accent_rest_len", "dens_shape", "dens_cap", "stream_lift", "guard", "jack8", "trem", "shape_run", "row_seed", "jack_allow", "mel_keep", "ladder", "half_fill", "four_one", "fine_bpm", "rest_chord", "dens_flat", "bass_sound", "dens_dev", "dens_pulse", "chord32", "fill32_sound")
LOG_INT_KEYS = ("max_chord", "end_over", "note_log", "pair_min", "stream_break")
# 창이 늘 고정값으로 넣는 옵션, 경로 (불러오지 않음)
LOG_SKIP = {"--audio": 1, "--ckpt": 1, "--lane-ckpt": 1, "--plan-ckpt": 1, "--scratch-model": 1, "--placer-ckpt": 1, "--out": 1, "--folder": 1,
            "--snap": 1, "--jack-penalty": 1, "--no-section-fix": 0, "--thick": 1, "--chord-style": 1,
            "--beat-contrast": 1, "--spread-chords": 1, "--offgrid-keep": 1}


def settings_from_log(text: str) -> tuple[dict, list[str], int]:
    """생성_로그.txt 첫 줄('$ python infer.py ...') -> (창 설정 값, 창에 없는 옵션 토큰(추가 옵션 칸으로), 읽은 옵션 수)"""
    line = next((l for l in text.splitlines() if l.startswith("$ ")), "")
    if not line:
        raise ValueError("로그 첫 줄에 '$ python infer.py ...' 명령이 없어요. 생성_로그.txt 파일을 골라 주세요.")
    toks = shlex.split(line[2:], posix=False)
    toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t for t in toks]
    while toks and not toks[0].startswith("--"):
        toks.pop(0)                                  # python infer.py
    args, order, i = {}, [], 0
    while i < len(toks):
        f = toks[i]
        if f.startswith("--") and i + 1 < len(toks) and not (toks[i + 1].startswith("--") and not _is_num(toks[i + 1])):
            args[f] = toks[i + 1]
            i += 2
        else:
            args[f] = None
            i += 1
        order.append(f)
    upd, extra, n = {}, [], 0
    audio_stem = re.split(r"[\\/]", args.get("--audio") or "")[-1].rsplit(".", 1)[0]   # 윈도 경로도
    for f in order:
        v = args[f]
        key = f[2:].replace("-", "_")
        try:
            if f in LOG_SKIP:
                continue
            if key in LOG_FLOAT_KEYS:
                upd[key] = float(v)
            elif key in LOG_INT_KEYS:
                upd[key] = int(round(float(v)))
            elif f == "--stream-end-chord":
                upd["end_chord"] = float(v) > 0
            elif f == "--seed":
                upd["seed"] = int(float(v))
            elif f == "--judge":
                inv = {b: a for a, b in JUDGES.items()}
                upd["judge"] = inv.get(str(v).lower().replace("_", ""), "NORMAL")
            elif f == "--total":
                upd["total"] = f"{float(v):g}"
            elif f == "--follow":
                from stem_follow import mode_word
                inv = {b: a for a, b in FOLLOW.items()}
                upd["follow"] = inv.get(mode_word(v), "끔")
            elif f == "--follow-strength":
                upd["follow_strength"] = float(v)
            elif f == "--follow-at":
                upd["follow_at"] = v or ""
            elif f == "--bpm":
                upd["bpm"] = v or ""
            elif f == "--title":
                if v and v != audio_stem:              # 곡 파일 이름으로 자동으로 들어간 제목은 빼고
                    upd["title"] = v
            elif f == "--chart-name":
                upd["chart_name"] = v or ""
            elif f == "--artist":
                upd["artist"] = v or ""
            elif f == "--key-sound":
                upd["click"] = v == "click"
            else:
                extra += [f] + ([v] if v is not None else [])
                continue
            n += 1
        except (TypeError, ValueError):
            extra += [f] + ([v] if v is not None else [])
    upd.setdefault("title", "")
    upd.setdefault("follow_at", "")
    upd.setdefault("bpm", "")
    upd.setdefault("click", False)
    return upd, extra, n


def _is_num(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def open_folder(path: Path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        if os.name == "nt":
            os.startfile(str(path))                       # noqa: S606 (윈도우 탐색기)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:                                 # 탐색기를 못 열어도 프로그램은 계속
        messagebox.showinfo("폴더", f"{path}\n\n({e})")


class _Tip:
    """마우스를 올리면 자세한 설명을 작은 창으로 (0.4초 뒤, 줄바꿈해서)"""

    def __init__(self, widgets, text, wrap=440):
        self.text, self.wrap = text, wrap
        self.win = None
        self.job = None
        for w in widgets:
            w.bind("<Enter>", self._enter, add="+")
            w.bind("<Leave>", self._leave, add="+")
            w.bind("<ButtonPress>", self._leave, add="+")
        self.w0 = widgets[0]

    def _enter(self, e):
        self._cancel()
        self.job = self.w0.after(400, lambda x=e.x_root, y=e.y_root: self._show(x, y))

    def _leave(self, _e=None):
        self._cancel()
        if self.win is not None:
            self.win.destroy()
            self.win = None

    def _cancel(self):
        if self.job is not None:
            self.w0.after_cancel(self.job)
            self.job = None

    def _show(self, x, y):
        self.job = None
        if self.win is not None or not self.text:
            return
        win = tk.Toplevel(self.w0)
        win.wm_overrideredirect(True)
        tk.Label(win, text=self.text, justify="left", wraplength=self.wrap, background="#ffffe8",
                 relief="solid", borderwidth=1, padx=6, pady=4).pack()
        win.update_idletasks()
        sw = win.winfo_screenwidth()
        x = min(x + 14, sw - win.winfo_reqwidth() - 8)
        win.wm_geometry(f"+{max(0, x)}+{y + 18}")
        self.win = win


class App:
    def __init__(self, root: tk.Tk, files: list[str]):
        self.root = root
        self.s = load_settings()
        self.q: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.stop_flag = threading.Event()
        self.running = False
        self.results: list[tuple[str, Path]] = []        # (표시 글, 결과 폴더)
        self.last_dir = str(Path(files[0]).parent) if files else ""

        root.title("BMS AI 채보 생성기")
        root.minsize(760, 420)
        style = ttk.Style()
        for th in ("vista", "clam"):
            if th in style.theme_names():
                style.theme_use(th)
                break
        style.configure("Hint.TLabel", foreground="#777777")
        style.configure("Big.TButton", padding=(18, 6))

        # 화면이 작으면 아래가 잘려서 창 전체를 스크롤되게 (캔버스 안에 내용 프레임)
        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        main = ttk.Frame(self.canvas, padding=10)
        self._main = main
        self._win = self.canvas.create_window((0, 0), window=main, anchor="nw")
        main.columnconfigure(0, weight=1)
        main.bind("<Configure>", lambda e: self._fit_scroll())
        self.canvas.bind("<Configure>", lambda e: self._fit_scroll())
        root.bind_all("<MouseWheel>", self._on_wheel)                    # Windows / macOS
        root.bind_all("<Button-4>", lambda e: self._on_wheel(e, -1))     # Linux
        root.bind_all("<Button-5>", lambda e: self._on_wheel(e, 1))

        self._build_files(main)
        self._build_options(main)
        self._build_song_info(main)
        self._build_advanced(main)
        self._build_run(main)
        self._build_log(main)

        for f in files:
            self._add_file(f)
        root.update_idletasks()                     # 처음 창 크기: 내용 전부, 단 화면보다 크면 화면에 맞춤
        w = max(760, main.winfo_reqwidth() + 20)
        h = min(main.winfo_reqheight() + 4, root.winfo_screenheight() - 90)
        root.geometry(f"{w}x{max(420, h)}")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll)

    # ---------- 화면 ----------
    def _build_files(self, main):
        box = ttk.LabelFrame(main, text=" 1. 곡 고르기 ", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        # 곡마다 차분명, 작곡가를 따로 (칸을 두 번 누르면 고침, 비우면 '곡 정보'의 기본값)
        self.files = ttk.Treeview(box, columns=("song", "chart", "artist"), show="headings", height=4,
                                  selectmode="extended")
        self.files.heading("song", text="곡")
        self.files.heading("chart", text="차분명")
        self.files.heading("artist", text="작곡가")
        self.files.column("song", width=330, stretch=True)
        self.files.column("chart", width=110, stretch=False)
        self.files.column("artist", width=140, stretch=False)
        self.files.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.files.bind("<Double-1>", self._edit_cell)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.files.yview)
        sb.grid(row=0, column=1, rowspan=3, sticky="ns")
        self.files.configure(yscrollcommand=sb.set)
        ttk.Button(box, text="곡 추가…", command=self.pick_files).grid(row=0, column=2, padx=(8, 0), sticky="ew")
        ttk.Button(box, text="선택 빼기", command=self.remove_files).grid(row=1, column=2, padx=(8, 0), sticky="ew")
        ttk.Button(box, text="모두 비우기", command=lambda: self.files.delete(*self.files.get_children())).grid(
            row=2, column=2, padx=(8, 0), sticky="ew")
        ttk.Label(box, text="칸을 두 번 누르면 곡마다 차분명·작곡가를 따로 적어요 (비우면 '3. 곡 정보' 값)",
                  style="Hint.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))
        # '같은 곡 하나 더' 버튼 없앰 - 일괄 생성은 모든 줄이 같은 설정이라 똑같은 패턴만 나왔음

        row = ttk.Frame(box)
        row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="저장 폴더").grid(row=0, column=0, padx=(0, 8))
        self.v_out = tk.StringVar(value=self.s["out"])
        ttk.Entry(row, textvariable=self.v_out).grid(row=0, column=1, sticky="ew")
        ttk.Button(row, text="찾기…", command=self.pick_out).grid(row=0, column=2, padx=(8, 0))

    def _build_options(self, main):
        box = ttk.LabelFrame(main, text=" 2. 설정 ", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        box.columnconfigure(1, weight=1)

        top = ttk.Frame(box)
        top.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(top, text="빠른 설정").pack(side="left")
        self.v_preset = tk.StringVar(value="")
        cb = ttk.Combobox(top, textvariable=self.v_preset, values=list(PRESETS), state="readonly", width=14)
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(self.v_preset.get()))
        ttk.Label(top, text="고른 뒤 슬라이더로 조금씩 바꿔도 돼요", style="Hint.TLabel").pack(side="left")
        ttk.Button(top, text="생성 로그에서 불러오기…", command=self.load_log).pack(side="left", padx=(12, 0))

        self.vars: dict[str, tk.DoubleVar] = {}
        self.val_labels: dict[str, ttk.Label] = {}
        main_specs = [sp for sp in SLIDERS if sp[0] in MAIN_SLIDERS]
        for r, spec in enumerate(main_specs, start=1):
            self._slider_row(box, r, spec)

        chk = ttk.Frame(box)
        chk.grid(row=len(main_specs) + 1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.v_click = tk.BooleanVar(value=bool(self.s["click"]))
        ttk.Checkbutton(chk, text="노트마다 클릭음 (싱크 확인용)", variable=self.v_click).pack(side="left")

    def _slider_row(self, box, r, spec):
        key, name, lo, hi, step, hint = spec
        v = tk.DoubleVar(value=float(self.s[key]))
        self.vars[key] = v
        nl = ttk.Label(box, text=name, width=22)
        nl.grid(row=r, column=0, sticky="w")
        sc = ttk.Scale(box, from_=lo, to=hi, variable=v, orient="horizontal",
                       command=lambda val, k=key, st=step: self._snap(k, st))
        sc.grid(row=r, column=1, sticky="ew", padx=6)
        lab = ttk.Label(box, width=5, anchor="e")
        lab.grid(row=r, column=2)
        self.val_labels[key] = lab
        hl = ttk.Label(box, text=hint, style="Hint.TLabel")             # 요약 한 줄
        hl.grid(row=r, column=3, sticky="w", padx=(8, 0))
        if DETAILS.get(key):
            _Tip([nl, hl], DETAILS[key])                                 # 자세한 설명은 마우스를 올리면
        self._snap(key, step)

    def _build_song_info(self, main):
        box = ttk.LabelFrame(main, text=" 3. 곡 정보 ", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        r1 = ttk.Frame(box)
        r1.pack(fill="x")
        ttk.Label(r1, text="판정(JUDGE)").pack(side="left")
        self.v_judge = tk.StringVar(value=str(self.s["judge"]) if self.s["judge"] in JUDGES else "NORMAL")
        ttk.Combobox(r1, textvariable=self.v_judge, values=list(JUDGES), state="readonly", width=11).pack(
            side="left", padx=(6, 18))
        ttk.Label(r1, text="TOTAL").pack(side="left")
        self.v_total = tk.StringVar(value=str(self.s["total"]))
        ttk.Entry(r1, textvariable=self.v_total, width=8).pack(side="left", padx=6)
        ttk.Label(r1, text="게이지 회복량 (기본 300)", style="Hint.TLabel").pack(
            side="left", padx=6)
        r2 = ttk.Frame(box)
        r2.pack(fill="x", pady=(6, 0))
        ttk.Label(r2, text="BPM").pack(side="left")
        self.v_bpm = tk.StringVar(value=str(self.s["bpm"]))
        ttk.Entry(r2, textvariable=self.v_bpm, width=8).pack(side="left", padx=(6, 18))
        ttk.Label(r2, text="제목").pack(side="left")
        self.v_title = tk.StringVar(value=str(self.s["title"]))
        ttk.Entry(r2, textvariable=self.v_title, width=30).pack(side="left", padx=6)
        ttk.Label(r2, text="한 곡일 때만, 비우면 자동 (BPM이 2배·절반이면 직접 입력)",
                  style="Hint.TLabel").pack(side="left", padx=6)
        r2b = ttk.Frame(box)
        r2b.pack(fill="x", pady=(6, 0))
        ttk.Label(r2b, text="차분명 기본").pack(side="left")
        self.v_chart = tk.StringVar(value=str(self.s.get("chart_name", "")))
        ttk.Entry(r2b, textvariable=self.v_chart, width=14).pack(side="left", padx=(6, 18))
        ttk.Label(r2b, text="작곡가 기본").pack(side="left")
        self.v_artist = tk.StringVar(value=str(self.s.get("artist", "")))
        ttk.Entry(r2b, textvariable=self.v_artist, width=22).pack(side="left", padx=6)
        ttk.Label(r2b, text="곡별 칸이 비면 이 값. 차분명이 있으면 <곡>_<차분명>.bms로 따로 저장",
                  style="Hint.TLabel").pack(side="left", padx=6)
        r3 = ttk.Frame(box)
        r3.pack(fill="x", pady=(6, 0))
        ttk.Label(r3, text="따라갈 악기").pack(side="left")
        fv = "악기" if self.s["follow"] == "멜로디" else str(self.s["follow"])     # 예전 설정 파일
        self.v_follow = tk.StringVar(value=fv if fv in FOLLOW else "끔")
        ttk.Combobox(r3, textvariable=self.v_follow, values=list(FOLLOW), state="readonly", width=7).pack(
            side="left", padx=(6, 12))
        ttk.Label(r3, text="세기").pack(side="left")
        self.v_fstr = tk.StringVar(value=f"{float(self.s['follow_strength']):g}")
        ttk.Spinbox(r3, from_=0, to=1, increment=0.1, textvariable=self.v_fstr, width=5).pack(side="left", padx=(6, 12))
        ttk.Label(r3, text="구간별").pack(side="left")
        self.v_fat = tk.StringVar(value=str(self.s["follow_at"]))
        ttk.Entry(r3, textvariable=self.v_fat, width=40).pack(side="left", padx=6)
        ttk.Label(box, text="예: 1-16 드럼, 17-32 악기 / 1~3초 보컬 (마디 또는 초, 한 곡일 때만)",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(box, text="악기 = 드럼·보컬·베이스 뺀 나머지. 보컬은 세기 0.5부터. 곡마다 처음 한 번 음원 분리로 몇 분 더 걸려요",
                  style="Hint.TLabel").pack(anchor="w")

    def _build_advanced(self, main):
        self.adv_open = tk.BooleanVar(value=False)
        hdr = ttk.Checkbutton(main, text="고급 설정 보기", variable=self.adv_open, command=self._toggle_adv)
        hdr.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.adv = ttk.Frame(main, padding=(8, 4))
        self.adv.columnconfigure(1, weight=1)
        self.v_ck = {}
        for r, (key, name) in enumerate([("ckpt", "1단계(타이밍) 모델"), ("lane_ckpt", "레인 모델"),
                                         ("plan_ckpt", "계획 모델")]):
            ttk.Label(self.adv, text=name, width=18).grid(row=r, column=0, sticky="w")
            v = tk.StringVar(value=self.s[key])
            self.v_ck[key] = v
            ttk.Entry(self.adv, textvariable=v).grid(row=r, column=1, sticky="ew", pady=1)
            ttk.Button(self.adv, text="찾기…", command=lambda k=key: self.pick_ckpt(k)).grid(row=r, column=2, padx=(6, 0))
        r = 3
        ttk.Label(self.adv, text="시드").grid(row=r, column=0, sticky="w")
        self.v_seed = tk.StringVar(value=str(self.s["seed"]))
        sd = ttk.Frame(self.adv)
        sd.grid(row=r, column=1, sticky="w")
        ttk.Spinbox(sd, from_=0, to=9999, textvariable=self.v_seed, width=6).pack(side="left")
        ttk.Label(sd, text="  같은 곡을 다른 배치로 다시 뽑고 싶을 때 바꿔요", style="Hint.TLabel").pack(side="left")
        ttk.Label(self.adv, text="추가 옵션").grid(row=r + 1, column=0, sticky="w")
        self.v_extra = tk.StringVar(value=self.s["extra"])
        ttk.Entry(self.adv, textvariable=self.v_extra).grid(row=r + 1, column=1, sticky="ew", pady=1)
        ttk.Button(self.adv, text="명령 복사", command=self.copy_cmd).grid(row=r + 1, column=2, padx=(6, 0))
        ttk.Label(self.adv, text="예: --max-chord 5 (infer.py 옵션을 그대로)",
                  style="Hint.TLabel").grid(row=r + 2, column=1, sticky="w")
        det = ttk.LabelFrame(self.adv, text=" 세부 규칙 (보통 그대로 두세요) ", padding=6)
        det.grid(row=r + 3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        det.columnconfigure(1, weight=1)
        for i, spec in enumerate([sp for sp in SLIDERS if sp[0] not in MAIN_SLIDERS]):
            self._slider_row(det, i, spec)

    def _fit_scroll(self):
        """내용 프레임 폭 = 창 폭, 높이 = max(내용, 창) (창이 크면 기록 칸이 늘어나고, 작으면 스크롤)"""
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        rh = self._main.winfo_reqheight()
        self.canvas.itemconfigure(self._win, width=cw, height=max(rh, ch))
        self.canvas.configure(scrollregion=(0, 0, cw, max(rh, ch)))

    def _on_wheel(self, e, step=None):
        w = e.widget
        try:                                        # 기록/목록 칸 위에서는 그 칸이 스크롤 (단, 끝에 닿았으면 창이)
            cls = w.winfo_class()
        except Exception:  # noqa: BLE001
            cls = ""
        if cls in ("Text", "Listbox", "TCombobox", "TSpinbox"):
            return
        if self._main.winfo_reqheight() <= self.canvas.winfo_height():
            return
        if step is None:
            step = -1 if e.delta > 0 else 1
            step *= max(1, abs(int(e.delta)) // 120) * 3
        self.canvas.yview_scroll(step, "units")

    def _toggle_adv(self):
        if self.adv_open.get():
            self.adv.grid(row=4, column=0, sticky="ew")
        else:
            self.adv.grid_remove()

    def _build_run(self, main):
        bar = ttk.Frame(main)
        bar.grid(row=5, column=0, sticky="ew", pady=(10, 4))
        bar.columnconfigure(3, weight=1)
        self.b_run = ttk.Button(bar, text="채보 만들기", style="Big.TButton", command=self.start)
        self.b_run.grid(row=0, column=0)
        self.b_stop = ttk.Button(bar, text="중지", command=self.stop, state="disabled")
        self.b_stop.grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="결과 폴더 열기", command=self.open_result).grid(row=0, column=2)
        self.v_status = tk.StringVar(value="곡을 추가하고 '채보 만들기'를 누르세요")
        ttk.Label(bar, textvariable=self.v_status, anchor="e").grid(row=0, column=3, sticky="e")
        self.prog = ttk.Progressbar(main, mode="determinate", maximum=1.0)
        self.prog.grid(row=6, column=0, sticky="ew")

    def _build_log(self, main):
        nb = ttk.Notebook(main)
        nb.grid(row=7, column=0, sticky="nsew", pady=(8, 0))
        main.rowconfigure(7, weight=1)
        res = ttk.Frame(nb, padding=4)
        res.columnconfigure(0, weight=1)
        res.rowconfigure(0, weight=1)
        self.res_list = tk.Listbox(res, height=6, activestyle="none")
        self.res_list.grid(row=0, column=0, sticky="nsew")
        self.res_list.bind("<Double-Button-1>", lambda e: self.open_result())
        ttk.Label(res, text="두 번 누르면 그 곡의 결과 폴더가 열려요 (.bms + 음원 + 생성_로그.txt)",
                  style="Hint.TLabel").grid(row=1, column=0, sticky="w")
        logf = ttk.Frame(nb, padding=4)
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.log = ScrolledText(logf, height=10, wrap="none", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        nb.add(res, text="  결과  ")
        nb.add(logf, text="  자세한 기록  ")
        self.nb = nb

    # ---------- 입력 ----------
    def _snap(self, key, step):
        v = self.vars[key]
        x = round(round(float(v.get()) / step) * step, 4)
        if abs(x - float(v.get())) > 1e-9:
            v.set(x)
        txt = f"{x:.2f}" if step < 0.1 else f"{x:g}"
        if key in ("fill32", "thin_weak", "snap_fix", "stream_hole", "end_shape", "beat_motif", "eighth_shape", "stream_fill", "edge_axis", "pair_even", "pair_min", "placer", "pair_avg", "quarter_spread", "eighth_audit", "hand_balance", "fine_density", "stream_break", "stream_target", "stream_kpr", "dens_shape", "dens_cap", "stream_lift", "jack8", "trem", "shape_run", "jack_allow", "mel_keep", "ladder", "half_fill", "four_one", "fine_bpm", "rest_chord", "dens_flat", "bass_sound", "dens_dev", "dens_pulse", "chord32", "fill32_sound") and x <= 0:
            txt = "끔"                                   # 16비트 이상 채우기는 작을수록 많이 채우지만 0은 꺼짐
        if key in ("ladder", "dens_flat", "dens_dev", "dens_pulse") and x > 0:     # 켬/끔만
            txt = "켬"
        if key == "fine_grid":
            txt = "16비트만" if x <= 0 else ("다 허용" if x >= 2 else ("자동" if abs(x - 1) < 1e-9 else f"{x:g}"))
        self.val_labels[key].configure(text=txt)

    def apply_preset(self, name):
        p = PRESETS.get(name)
        if not p:
            return
        for k, v in p.items():
            if k in self.vars:
                self.vars[k].set(v)
                self._snap(k, next(s[4] for s in SLIDERS if s[0] == k))
            elif k in HIDDEN:
                self.s[k] = v                                  # 화면에 없는 값은 설정에만

    def load_log(self):
        """결과 폴더의 생성_로그*.txt를 골라 거기 적힌 설정을 창에 채움"""
        f = filedialog.askopenfilename(title="생성 로그 고르기", filetypes=[("생성 로그", "*.txt"), ("모든 파일", "*.*")],
                                       initialdir=str(self.v_out.get() or HERE))
        if not f:
            return
        try:
            upd, extra, n = settings_from_log(Path(f).read_text(encoding="utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            messagebox.showwarning("불러오기 실패", str(e))
            return
        for k, v in upd.items():
            if k in self.vars:
                self.vars[k].set(v)
                self._snap(k, next(sl[4] for sl in SLIDERS if sl[0] == k))
            elif k in DEFAULTS:
                self.s[k] = v                                  # 화면에 없는 값(숨김)은 이번 실행 설정에
        simple = {"seed": self.v_seed, "total": self.v_total, "bpm": self.v_bpm, "title": self.v_title,
                  "follow_at": self.v_fat, "chart_name": self.v_chart, "artist": self.v_artist}
        for k, var in simple.items():
            if k in upd:
                var.set(str(upd[k]))
        if "judge" in upd:
            self.v_judge.set(upd["judge"])
        if "follow" in upd:
            self.v_follow.set(upd["follow"])
        if "follow_strength" in upd:
            self.v_fstr.set(f"{float(upd['follow_strength']):g}")
        if "click" in upd:
            self.v_click.set(bool(upd["click"]))
        self.v_extra.set(show_cmd(extra) if extra else "")
        self.v_preset.set("")
        msg = f"'{Path(f).parent.name}/{Path(f).name}' 설정 {n}개를 불러왔어요"
        if extra:
            msg += f" (창에 없는 옵션 {len([x for x in extra if x.startswith('--')])}개는 고급 설정의 추가 옵션 칸에)"
        self.v_status.set(msg)

    def _add_file(self, f, chart: str = "", artist: str = ""):
        f = str(Path(f))
        if not hasattr(self, "_paths"):
            self._paths = {}                              # 목록 줄 id -> 곡 경로 (같은 곡을 여러 줄로 넣을 수 있게)
        if not chart and not artist and f in [self._paths.get(i) for i in self.files.get_children()]:
            return                                        # 그냥 다시 고른 건 무시
        iid = self.files.insert("", "end", values=(Path(f).name, chart, artist))
        self._paths[iid] = f

    def file_items(self) -> list[tuple[str, str, str]]:
        """(곡 경로, 차분명, 작곡가) 목록 - 칸이 비면 '곡 정보'의 기본값"""
        out = []
        for iid in self.files.get_children():
            _, ch, ar = (list(self.files.item(iid, "values")) + ["", "", ""])[:3]
            out.append((self._paths[iid], str(ch).strip(), str(ar).strip()))
        return out

    def _edit_cell(self, e):
        iid = self.files.identify_row(e.y)
        col = self.files.identify_column(e.x)              # '#1' 곡 / '#2' 차분명 / '#3' 작곡가
        if not iid or col not in ("#2", "#3"):
            return
        box = self.files.bbox(iid, col)
        if not box:
            return
        x, y, w, h = box
        name = "chart" if col == "#2" else "artist"
        var = tk.StringVar(value=self.files.set(iid, name))
        ent = ttk.Entry(self.files, textvariable=var)
        ent.place(x=x, y=y, width=w, height=h)
        ent.focus_set()
        ent.select_range(0, "end")

        closed = []

        def done(_=None, keep=True):
            if closed:
                return
            closed.append(1)
            if keep and self.files.exists(iid):
                self.files.set(iid, name, var.get().strip())
            ent.destroy()
        ent.bind("<Return>", done)
        ent.bind("<FocusOut>", done)
        ent.bind("<Escape>", lambda _e: done(keep=False))

    def pick_files(self):
        fs = filedialog.askopenfilenames(title="곡 고르기", filetypes=AUDIO_TYPES,
                                         initialdir=self.last_dir or str(HERE / "input"))
        for f in fs:
            self._add_file(f)
        if fs:
            self.last_dir = str(Path(fs[0]).parent)

    def remove_files(self):
        for iid in self.files.selection():
            self.files.delete(iid)

    def pick_out(self):
        d = filedialog.askdirectory(title="저장 폴더", initialdir=self.v_out.get() or str(HERE))
        if d:
            self.v_out.set(d)

    def pick_ckpt(self, key):
        f = filedialog.askopenfilename(title="모델 파일", filetypes=[("모델", "*.pt"), ("모든 파일", "*.*")],
                                       initialdir=str(HERE / "checkpoints"))
        if f:
            try:
                f = str(Path(f).resolve().relative_to(HERE))
            except ValueError:
                pass
            self.v_ck[key].set(f)

    def collect(self) -> dict:
        s = dict(self.s)
        s.update({k: round(float(v.get()), 4) for k, v in self.vars.items()})
        s["max_chord"] = int(round(float(s.get("max_chord", 6))))
        s.update({k: v.get() for k, v in self.v_ck.items()})
        s.update(out=self.v_out.get().strip() or DEFAULTS["out"], end_chord=True,
                 click=self.v_click.get(), bpm=self.v_bpm.get().strip(),
                 title=self.v_title.get().strip(), extra=self.v_extra.get().strip(),
                 chart_name=self.v_chart.get().strip(), artist=self.v_artist.get().strip(),
                 judge=self.v_judge.get() or "NORMAL", total=self.v_total.get().strip() or "300",
                 follow=self.v_follow.get() or "끔", follow_at=self.v_fat.get().strip())
        try:
            s["follow_strength"] = min(1.0, max(0.0, float(self.v_fstr.get())))
        except ValueError:
            s["follow_strength"] = 0.7
        try:
            s["seed"] = int(self.v_seed.get())
        except ValueError:
            s["seed"] = 0
        return s

    def save(self):
        self.s = self.collect()
        try:
            SETTINGS.write_text(json.dumps({**self.s, "_v46": True, "_v93": True}, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def copy_cmd(self):
        items = self.file_items() or [("input/곡.mp3", "", "")]
        s = self.collect()
        f0, ch0, ar0 = items[0]
        s.update(chart_name=ch0 or s.get("chart_name", ""), artist=ar0 or s.get("artist", ""))
        cmd = "python infer.py " + show_cmd(infer_args(f0, s, single=len({f for f, _, _ in items}) == 1))
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.v_status.set("명령을 복사했어요 (터미널에 붙여 넣으면 같은 결과)")

    def check(self, s, files) -> str | None:
        if not files:
            return "곡을 먼저 추가하세요."
        miss = [f for f in files if not Path(f).exists()]
        if miss:
            return "없는 파일이 있어요:\n" + "\n".join(miss[:5])
        if not FROZEN and not (HERE / "infer.py").exists():
            return f"infer.py가 이 프로그램과 같은 폴더에 있어야 해요.\n({HERE})"
        for k, name in (("ckpt", "1단계 모델"), ("lane_ckpt", "레인 모델"), ("plan_ckpt", "계획 모델")):
            if not abs_path(s[k]).exists():
                return f"{name} 파일이 없어요: {abs_path(s[k])}\n고급 설정에서 위치를 바꿔 주세요."
        try:
            assert float(s["total"]) > 0
        except Exception:
            return "TOTAL은 0보다 큰 숫자로 넣어 주세요 (예: 300)."
        if s["bpm"] and len(files) == 1:
            try:
                b = float(s["bpm"])
                assert 30 <= b <= 400
            except Exception:
                return "BPM은 30~400 사이 숫자로 넣거나 비워 두세요."
        if s.get("follow_at", "").strip():
            from stem_follow import parse_follow_at
            try:
                parse_follow_at(s["follow_at"])
            except SystemExit as e:
                return str(e)
        try:
            shlex.split(s["extra"], posix=(os.name != "nt"))
        except ValueError:
            return "추가 옵션의 따옴표가 맞지 않아요."
        return None

    # ---------- 실행 ----------
    def start(self):
        if self.running:
            return
        s = self.collect()
        items = self.file_items()
        files = [f for f, _, _ in items]
        err = self.check(s, files)
        if err:
            messagebox.showwarning("확인해 주세요", err)
            return
        if len(set(files)) > 1 and (s["bpm"] or s["title"]):
            self.v_status.set("곡이 여러 개라 BPM/제목 칸은 무시해요")
        self.save()
        self.running = True
        self.stop_flag.clear()
        self.b_run.configure(state="disabled")
        self.b_stop.configure(state="normal")
        self.prog.configure(value=0.0)
        self.log.delete("1.0", "end")
        threading.Thread(target=self.worker, args=(items, s), daemon=True).start()

    def stop(self):
        self.stop_flag.set()
        p = self.proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
        self.v_status.set("멈추는 중…")

    def worker(self, items, s0):
        n = len(items)
        single = len({f for f, _, _ in items}) == 1          # 같은 곡을 차분명만 달리 여러 번 넣은 것도 한 곡으로
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        flags = 0x08000000 if os.name == "nt" else 0           # CREATE_NO_WINDOW: 검은 창 안 띄움
        for i, (f, ch, ar) in enumerate(items):
            if self.stop_flag.is_set():
                break
            s = dict(s0)
            if ch:
                s["chart_name"] = ch                          # 곡별 칸이 있으면 그것, 없으면 '곡 정보'의 기본값
            if ar:
                s["artist"] = ar
            name = Path(f).name + (f" [{s['chart_name']}]" if str(s.get("chart_name", "")).strip() else "")
            fname = unique_folder(s["out"], f)                # 같은 곡이면 같은 폴더에 덮어씀
            cmd = build_cmd(f, s, single=single, folder=fname)
            shown = "python infer.py " + show_cmd(infer_args(f, s, single=single, folder=fname))
            self.q.put(("file", i, n, name))
            self.q.put(("log", f"\n===== [{i + 1}/{n}] {name} =====\n$ {shown}\n"))
            st = {"lines": [], "summary": ""}
            t0, ok = time.time(), False
            fd, logp = tempfile.mkstemp(prefix="bms_ai_", suffix=".log")
            os.close(fd)
            try:
                self.proc = subprocess.Popen(cmd, cwd=str(HERE), stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             env={**env, "BMS_CHILD_LOG": logp}, creationflags=flags)
                with open(logp, encoding="utf-8", errors="replace") as fp:
                    buf = ""
                    while True:
                        chunk = fp.read()
                        if chunk:
                            buf += chunk
                            *full, buf = buf.split("\n")
                            for line in full:
                                self._line(line + "\n", i, n, st)
                        elif self.proc.poll() is not None:
                            break
                        else:
                            time.sleep(0.1)
                    buf += fp.read()
                    for line in buf.splitlines(True):
                        self._line(line, i, n, st)
                ok = self.proc.returncode == 0
            except Exception as e:
                self._line(f"\n실행 오류: {e}\n", i, n, st)
            finally:
                try:
                    os.remove(logp)
                except OSError:
                    pass
            lines, summary = st["lines"], st["summary"]
            folder = Path(s["out"]) / fname
            try:
                folder.mkdir(parents=True, exist_ok=True)
                tag = re.sub(r'[\\/:*?"<>|\s]+', "_", str(s.get("chart_name", ""))).strip("._")
                logname = f"생성_로그_{tag}.txt" if tag else "생성_로그.txt"
                (folder / logname).write_text(f"$ {shown}\n\n" + "".join(lines), encoding="utf-8")
            except Exception:
                pass
            sec = time.time() - t0
            if self.stop_flag.is_set():
                status = "중지됨"
            elif ok:
                status = f"완료 · {summary} · {sec:.0f}초" if summary else f"완료 · {sec:.0f}초"
            else:
                err = next((l.strip() for l in reversed(lines) if l.strip()), "알 수 없는 오류")
                status = f"실패 · {err[:80]}"
            self.q.put(("done", i, n, name, status, folder, ok))
        self.q.put(("all", None))

    def _line(self, line, i, n, st):
        st["lines"].append(line)
        self.q.put(("log", line))
        for key, label, frac in STAGES:
            if line.startswith(key):
                self.q.put(("stage", i, n, label, frac))
        if line.startswith("예측한 노트 시각"):
            m = re.search(r"총 노트 (\d+)개 \(([\d.]+)개/초\)", line)
            if m:
                st["summary"] = f"노트 {m[1]}개 ({m[2]}/초)"

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self.log.insert("end", msg[1])
                    self.log.see("end")
                elif kind == "file":
                    _, i, n, name = msg
                    self.v_status.set(f"[{i + 1}/{n}] {name} 시작")
                    self.prog.configure(value=i / n)
                elif kind == "stage":
                    _, i, n, label, frac = msg
                    self.v_status.set(f"[{i + 1}/{n}] {label}")
                    self.prog.configure(value=(i + frac) / n)
                elif kind == "done":
                    _, i, n, name, status, folder, ok = msg
                    self.results.append((f"{name}  —  {status}", folder))
                    self.res_list.insert("end", f"{name}  —  {status}")
                    self.res_list.itemconfigure("end", foreground=("#1a7f37" if ok else "#c62828"))
                    self.res_list.see("end")
                    self.prog.configure(value=(i + 1) / n)
                    if not ok and not self.stop_flag.is_set():
                        self.nb.select(1)
                elif kind == "all":
                    self.running = False
                    self.proc = None
                    self.b_run.configure(state="normal")
                    self.b_stop.configure(state="disabled")
                    self.v_status.set("중지했어요" if self.stop_flag.is_set() else "다 만들었어요")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def open_result(self):
        sel = self.res_list.curselection()
        if sel:
            open_folder(self.results[sel[0]][1])
        elif self.results:
            open_folder(self.results[-1][1])
        else:
            open_folder(Path(self.v_out.get() or DEFAULTS["out"]))

    def on_close(self):
        if self.running and not messagebox.askyesno("닫기", "채보를 만드는 중이에요. 멈추고 닫을까요?"):
            return
        self.stop()
        self.save()
        self.root.destroy()


def run_infer(argv: list[str]) -> None:
    """자식 프로세스: infer.py main()을 이 프로세스 안에서 실행. 출력은 BMS_CHILD_LOG 파일로 (창이 그 파일을 읽어 보여 줌)."""
    log = os.environ.get("BMS_CHILD_LOG")
    fp = None
    if log:
        fp = open(log, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = fp
    elif sys.stdout is None:                              # exe 창 모드에서 직접 실행된 경우
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    sys.argv = ["infer.py"] + list(argv)
    code = 0
    try:
        import infer
        infer.main()
    except SystemExit as e:
        if isinstance(e.code, int):
            code = e.code
        elif e.code is not None:
            print(e.code)
            code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)


def selftest() -> int:
    """exe가 제대로 묶였는지 점검: 라이브러리, mp3 읽기, 모델 파일 읽기, 창 라이브러리. 결과는 selftest.txt"""
    out, ok = [], True

    def chk(name, fn):
        nonlocal ok
        try:
            out.append(f"OK    {name}: {fn()}")
        except BaseException as e:                       # noqa: BLE001
            ok = False
            out.append(f"FAIL  {name}: {type(e).__name__}: {e}")

    def _torch():
        import torch
        return f"{torch.__version__}, GPU {'있음' if torch.cuda.is_available() else '없음(CPU)'}, 스레드 {torch.get_num_threads()}"

    def _scipy():
        import scipy
        import scipy.signal                               # noqa: F401
        from scipy.io import wavfile                      # noqa: F401
        return scipy.__version__

    def _sf():
        import soundfile as sf
        fm = sf.available_formats()
        assert "MP3" in fm, "MP3를 읽을 수 없는 libsndfile (soundfile 0.12 이상 필요)"
        return f"{sf.__version__}, libsndfile {sf.__libsndfile_version__}, MP3/FLAC/OGG 읽기 가능"

    def _infer():
        import infer                                     # noqa: F401
        import thick_ctl                                 # noqa: F401
        import fill32                                    # noqa: F401
        import stem_follow                               # noqa: F401
        return "불러옴"

    def _demucs():
        try:
            import demucs                                # noqa: F401
        except ImportError:
            return "없음 (따라갈 악기 기능만 못 씀, 나머지는 정상)"
        return f"있음 {getattr(demucs, '__version__', '')}"

    def _tk():
        import tkinter
        return f"Tk {tkinter.TkVersion}"

    chk("numpy", lambda: __import__("numpy").__version__)
    chk("scipy", _scipy)
    chk("torch", _torch)
    chk("soundfile", _sf)
    chk("채보 코드(infer 등)", _infer)
    chk("창(tkinter)", _tk)
    chk("음원 분리(Demucs, 선택)", _demucs)

    def _scr():
        p = abs_path(s.get("scratch_model") or DEFAULTS["scratch_model"])
        if not p.exists():
            return f"없음 ({p.name}, 스크래치는 박 위치 가중치로 채움)"
        from scratch_model import TreeModel
        m = TreeModel(p)
        return f"{p.name} (나무 {len(m.start) - 1}개)"
    s = load_settings()
    chk("스크래치 자리 모델(선택)", _scr)
    s = load_settings()
    for key in ("ckpt", "lane_ckpt", "plan_ckpt"):
        path = abs_path(s[key])

        def _load(path=path):
            import torch
            ck = torch.load(path, map_location="cpu", weights_only=False)
            return f"{path.name} ({path.stat().st_size / 1e6:.1f}MB, step {ck.get('step') if isinstance(ck, dict) else '?'})"
        chk(f"모델 {key}", _load)
    text = ("모두 정상\n" if ok else "문제 있음 (FAIL 줄 확인)\n") + "\n".join(out) + "\n"
    try:
        (HERE / "selftest.txt").write_text(text, encoding="utf-8")
    except Exception:
        pass
    if sys.stdout is not None:
        try:
            print(text)
        except Exception:
            pass
    return 0 if ok else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--run-infer":
        run_infer(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    _load_tk()
    files = [a for a in sys.argv[1:] if Path(a).suffix.lower() in AUDIO_EXT]
    try:
        from ctypes import windll                          # 윈도우 고해상도 화면에서 글자 흐림 방지 (창 만들기 전에)
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root, files)
    root.mainloop()


if __name__ == "__main__":
    main()
