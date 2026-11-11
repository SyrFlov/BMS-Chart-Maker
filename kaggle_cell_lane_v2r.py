# [학습]
# ===== 레인 모델 v2 + 박자 위치 학습 셀 (노트북 셀 하나에 통째로 붙여 넣기) =====
# Input에 붙일 것: processed 데이터셋, 코드 데이터셋(최신 버전), 박자 데이터셋(data/rhythm 압축본)

# ===== 1) 준비: 데이터와 코드 복사 =====
import glob, os, shutil, zipfile, subprocess, pandas as pd, torch

W = "/kaggle/working"; DATA = f"{W}/processed"; CODE = f"{W}/code"; RHY = f"{W}/rhythm"
for d in (DATA, CODE, RHY):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs(CODE)

idx = glob.glob("/kaggle/input/**/index.csv", recursive=True)
if idx:
    shutil.copytree(os.path.dirname(idx[0]), DATA)
else:
    z = [p for p in glob.glob("/kaggle/input/**/*.zip", recursive=True)
         if any(n.endswith("index.csv") for n in zipfile.ZipFile(p).namelist())]
    assert z, "processed 데이터 없음 → Add Input으로 데이터셋을 붙이세요"
    tmp = f"{W}/_d"; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z[0]).extractall(tmp)
    shutil.move(os.path.dirname(glob.glob(f"{tmp}/**/index.csv", recursive=True)[0]), DATA)

py = glob.glob("/kaggle/input/**/train.py", recursive=True)
if py:
    src = os.path.dirname(py[0])
else:
    z = [p for p in glob.glob("/kaggle/input/**/*.zip", recursive=True)
         if any(n.endswith("train.py") for n in zipfile.ZipFile(p).namelist())]
    assert z, "코드(train.py) 없음 → 코드 데이터셋을 Add Input 하세요"
    tmp = f"{W}/_c"; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z[0]).extractall(tmp)
    src = os.path.dirname(glob.glob(f"{tmp}/**/train.py", recursive=True)[0])
for f in glob.glob(f"{src}/*.py"): shutil.copy(f, CODE)

# 박자 파일(.rhy.npy): 폴더 그대로 또는 zip
rh = glob.glob("/kaggle/input/**/*.rhy.npy", recursive=True)
if rh:
    os.makedirs(RHY)
    for f in rh: shutil.copy(f, RHY)
else:
    z = [p for p in glob.glob("/kaggle/input/**/*.zip", recursive=True)
         if any(n.endswith(".rhy.npy") for n in zipfile.ZipFile(p).namelist())]
    assert z, "박자 파일(.rhy.npy) 없음 → data/rhythm 압축본을 데이터셋으로 올리고 Add Input 하세요"
    tmp = f"{W}/_r"; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z[0]).extractall(tmp)
    os.makedirs(RHY)
    for f in glob.glob(f"{tmp}/**/*.rhy.npy", recursive=True): shutil.copy(f, RHY)

df = pd.read_csv(f"{DATA}/index.csv")
print("곡 수:", len(df), df["split"].value_counts().to_dict(), "| 박자 파일:", len(os.listdir(RHY)))
for need in ["lane_v2.py", "train_lane2.py", "augment.py", "bms_parser.py"]:
    assert os.path.exists(f"{CODE}/{need}"), f"{need}가 없어요 → 코드 데이터셋을 새 버전으로 올리고 Input을 최신으로 바꾸세요"
assert "RHY_DIM" in open(f"{CODE}/lane_v2.py", encoding="utf-8").read(), \
    "lane_v2.py가 예전 버전이에요 → 새 파일로 코드 데이터셋을 다시 올리고 Input을 최신으로"
print("GPU:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU가 꺼져 있어요 → Accelerator를 GPU로 바꾸세요"

# ===== 2) 학습: 레인 모델 v2 + 박자 위치 =====
def run(cmd, log):
    print("\n$", cmd)
    with open(log, "w", encoding="utf-8") as fp:
        p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            print(line, end=""); fp.write(line)
    assert p.wait() == 0, f"실패: {cmd}"

run("python lane_v2.py --torch", f"{W}/lane_v2r_check.txt")
run(f"python train_lane2.py --data {DATA} --rhythm-dir {RHY} --steps 3000 --batch 8 --eval-every 500 "
    f"--aug 0.8 --out {W}/lane_v2r.pt", f"{W}/lane_v2r_log.txt")

# ===== 3) 결과 묶기 =====
for d in (DATA, CODE, RHY, f"{W}/_d", f"{W}/_c", f"{W}/_r"):
    shutil.rmtree(d, ignore_errors=True)
with zipfile.ZipFile(f"{W}/result.zip", "w") as zf:
    for f in ["lane_v2r.pt", "lane_v2r_log.txt", "lane_v2r_check.txt"]:
        if os.path.exists(f"{W}/{f}"): zf.write(f"{W}/{f}", f)
print("\n완료:", sorted(os.listdir(W)))
