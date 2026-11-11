# [학습]
# ===== v9 학습 셀: 늘린 데이터로 레인 모델(v7) + 노트 시각 모델(onset_ctx2) 다시 학습 (셀 하나에 통째로) =====
# Input: processed 데이터셋(새 버전), 박자 데이터셋(새 버전), 코드 데이터셋(새 버전)
# 순서: 레인 모델 lane_v7.pt (좌우 반전, 약 30~60분) -> lane_eval -> 노트 시각 모델 onset_ctx2.pt (오래 걸림)
# 레인 결과는 먼저 result_lane.zip으로 묶어 두므로, 노트 시각 모델이 중간에 끊겨도 레인 모델은 남는다.
import glob, os, shutil, zipfile, subprocess, pandas as pd, torch

W = "/kaggle/working"; DATA = f"{W}/processed"; CODE = f"{W}/code"; RHY = f"{W}/rhythm"
for d in (DATA, CODE, RHY):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs(CODE)

def find_zip(suffix):
    return [p for p in glob.glob("/kaggle/input/**/*.zip", recursive=True)
            if any(n.endswith(suffix) for n in zipfile.ZipFile(p).namelist())]

idx = glob.glob("/kaggle/input/**/index.csv", recursive=True)
if idx:
    shutil.copytree(os.path.dirname(idx[0]), DATA)
else:
    z = find_zip("index.csv"); assert z, "processed 데이터 없음"
    tmp = f"{W}/_d"; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z[0]).extractall(tmp)
    shutil.move(os.path.dirname(glob.glob(f"{tmp}/**/index.csv", recursive=True)[0]), DATA)

py = glob.glob("/kaggle/input/**/train_lane2.py", recursive=True)
if py:
    src = os.path.dirname(py[0])
else:
    z = find_zip("train_lane2.py"); assert z, "코드 없음"
    tmp = f"{W}/_c"; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z[0]).extractall(tmp)
    src = os.path.dirname(glob.glob(f"{tmp}/**/train_lane2.py", recursive=True)[0])
for f in glob.glob(f"{src}/*.py"): shutil.copy(f, CODE)

rh = glob.glob("/kaggle/input/**/*.rhy.npy", recursive=True)
os.makedirs(RHY)
if rh:
    for f in rh: shutil.copy(f, RHY)
else:
    z = find_zip(".rhy.npy"); assert z, "박자 파일 없음"
    tmp = f"{W}/_r"; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z[0]).extractall(tmp)
    for f in glob.glob(f"{tmp}/**/*.rhy.npy", recursive=True): shutil.copy(f, RHY)

df = pd.read_csv(f"{DATA}/index.csv")
print("차분 수:", len(df), df["split"].value_counts().to_dict(), "| 곡 수:", df["song_id"].nunique(),
      "| 박자 파일:", len(os.listdir(RHY)))
miss = set(df["chart_id"]) - {os.path.basename(p)[:-8] for p in glob.glob(f"{RHY}/*.rhy.npy")}
assert not miss, f"박자 파일이 없는 차분 {len(miss)}개 → build_rhythm.py를 새 데이터로 다시 돌려 올리세요 (예: {list(miss)[:3]})"
assert "--mirror" in open(f"{CODE}/train_lane2.py", encoding="utf-8").read() and os.path.exists(f"{CODE}/lane_eval.py"), \
    "코드 데이터셋이 예전 버전이에요 → Input을 최신으로"
print("GPU:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU가 꺼져 있어요"

def run(cmd, log):
    print("\n$", cmd)
    with open(log, "w", encoding="utf-8") as fp:
        p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            print(line, end=""); fp.write(line)
    assert p.wait() == 0, f"실패: {cmd}"

def pack(name, files):
    with zipfile.ZipFile(f"{W}/{name}", "w") as zf:
        for f in files:
            if os.path.exists(f"{W}/{f}"): zf.write(f"{W}/{f}", f)

# 1) 레인 모델 v7 (데이터가 늘어서 스텝도 늘림)
run(f"python train_lane2.py --data {DATA} --rhythm-dir {RHY} --good --good-weight 3 --section --plan --accent "
    f"--alt-max 0.05 --alt-weight 0.2 --mirror 0.5 "
    f"--steps 8000 --batch 8 --eval-every 500 --aug 0.8 --out {W}/lane_v7.pt", f"{W}/lane_v7_log.txt")
run(f"python lane_eval.py --data {DATA} --rhythm-dir {RHY} --ckpt {W}/lane_v7.pt --out {W}/lane_eval_v7.txt",
    f"{W}/lane_eval_v7_log.txt")
pack("result_lane.zip", ["lane_v7.pt", "lane_v7_log.txt", "lane_eval_v7.txt"])
print("\n레인 모델 저장 완료: result_lane.zip")

# 2) 노트 시각 모델(문맥) (예전 설정 그대로, 데이터만 늘어남)
run(f"python train.py --data {DATA} --ctx --window-sec 30 --hidden 128 --batch 8 --steps 10000 "
    f"--lr 1e-3 --pos-weight 2 --aux-weight 0.3 --aug 0.8 --eval-every 500 --log-every 50 --workers 2 "
    f"--out {W}/onset_ctx2.pt", f"{W}/onset2_log.txt")
pack("result_onset.zip", ["onset_ctx2.pt", "onset2_log.txt"])

for d in (DATA, CODE, RHY, f"{W}/_d", f"{W}/_c", f"{W}/_r"):
    shutil.rmtree(d, ignore_errors=True)
print("\n완료:", sorted(os.listdir(W)))
