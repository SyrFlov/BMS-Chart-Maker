# 리듬게임 패턴 생성기 (BMSChartAI)

음원(mp3, wav, flac)을 넣으면 7건반 리듬게임 패턴(BMS 파일)을 만들어 주는 Windows 프로그램입니다.
서울과학기술대학교 전자공학과 2026년 졸업작품입니다.

- 시연 영상: https://youtu.be/3HTdRS2ZZd4
- 학습 모델 5개와, 사람이 만든 패턴의 통계로 정한 규칙 약 60개로 구성
- 사람이 만든 패턴과 노트 시각 84.7% 일치 (사람끼리 89.0%, 기존 생성기 AUTORHYTHM 66.6%)
- 노트북 CPU에서 곡당 약 50초

## 생성 순서

| STEP | 하는 일 | 주요 파일 |
|---|---|---|
| 1 곡 변환 | 음원을 멜 스펙트로그램으로 바꾸고 BPM을 추정 | build_dataset.py, bpm_estimate.py |
| 2 노트 시각 모델 | 10ms마다 노트가 시작될 확률을 계산 | model.py, dynamics.py |
| 3 박자와 줄 규칙 | 노트 시각을 박자 격자에 맞추고 줄을 정리 | beat_snap.py, fill32.py, stem_follow.py |
| 4 계획 모델과 레인 모델 | 줄마다 건반 초안을 만듦 | plan_model.py, lane_v2.py |
| 5 두께와 밀도 규칙 | 줄마다 건반 수와 구간 밀도를 사람 패턴 분포에 맞춤 | thick_ctl.py, melkeep_ctl.py, sound16_ctl.py, trem_ctl.py |
| 6 자리 모델과 배치 규칙 | 건반 수는 두고 자리를 다시 정함, 스크래치를 놓음 | placer.py, scratch_model.py, shape_ctl.py, lowjack_ctl.py |
| 7 저장 | BMS 파일로 쓰고 다시 읽어 검증 | bms_writer.py, bms_parser.py |

전체 흐름은 infer.py에 있고, 주석의 STEP 번호가 위 표와 같습니다.
파일마다 첫 줄에 [모델], [규칙], [학습], [평가] 같은 역할을 적어 두었습니다.

## 실행

```
pip install -r requirements.txt
python infer.py --audio input\mysong.mp3 --ckpt checkpoints\onset_best.pt --lane-ckpt checkpoints\lane_best.pt --out output
```

창으로 쓰려면 `python bms_gui.py`를 실행합니다.
학습된 모델 파일(checkpoints)은 저장소에 넣지 않았습니다.

## 저장소에 없는 것

- 학습 데이터: 사람이 만든 BMS 패턴 277개와 그 음원
- 시험과 시연에 쓴 음원, 생성 결과의 소리 파일
- 학습된 모델 파일

## 알려진 한계

- BPM이 고정된 4/4 박자 곡만 다룹니다. BPM 자동 추정은 2배나 절반으로 틀릴 수 있습니다.
- 분리 소리(Demucs)를 미리 만들어 두지 않은 곡은 멜로디 지키기 등 규칙 4개가 돌지 않습니다.
- 같은 건반을 100ms 안에 다시 치는 노트는 만들지 않지만, 스크래치에는 이 검사가 없습니다.
- 규칙 하나가 오류로 멈추면 그 규칙만 건너뛰고 생성은 끝까지 갑니다. 생성 로그의 [참고] 줄에 남습니다.

## 사용한 공개 소프트웨어

PyTorch, NumPy, SciPy, soundfile, Demucs(htdemucs), scikit-learn, PyInstaller

## 개발 방식

코드 작성에 LLM(Claude)을 활용했습니다.
