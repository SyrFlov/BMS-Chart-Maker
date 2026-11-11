# [학습]
# ===== 자리 모델 학습 셀: placer_v1.pt =====
# Input: processed 데이터셋, 박자 데이터셋(*.rhy.npy), 코드 데이터셋(placer.py 포함 최신) - v9 셀과 같은 것
# GPU 필요 (T4 한 장 기준 약 20~40분). 결과: /kaggle/working/result_placer.zip (placer_v1.pt, 로그, 평가)
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

py = glob.glob("/kaggle/input/**/placer.py", recursive=True)
if py:
    src = os.path.dirname(py[0])
else:
    z = find_zip("placer.py"); assert z, "코드 데이터셋에 placer.py가 없어요 -> 최신 코드로 다시 올리세요"
    tmp = f"{W}/_c"; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z[0]).extractall(tmp)
    src = os.path.dirname(glob.glob(f"{tmp}/**/placer.py", recursive=True)[0])
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
print("차분 수:", len(df), df["split"].value_counts().to_dict(), "| 박자 파일:", len(os.listdir(RHY)))
print("GPU:", torch.cuda.is_available()); assert torch.cuda.is_available(), "GPU가 꺼져 있어요"

def run(cmd, log):
    print("\n$", cmd)
    with open(log, "w", encoding="utf-8") as fp:
        p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            print(line, end=""); fp.write(line)
    assert p.wait() == 0, f"실패: {cmd}"

run(f"python placer.py train --data {DATA} --rhythm-dir {RHY} --out {W}/placer_v1.pt "
    f"--steps 12000 --batch 48 --len 128 --dim 160 --layers 4 --heads 4 --mirror 0.5 --eval-every 500",
    f"{W}/placer_log.txt")
for t in (0.6, 0.8, 1.0):
    run(f"python placer.py eval --data {DATA} --rhythm-dir {RHY} --ckpt {W}/placer_v1.pt --temp {t} "
        f"--out {W}/placer_eval_t{t}.txt", f"{W}/placer_eval_log_t{t}.txt")
with zipfile.ZipFile(f"{W}/result_placer.zip", "w") as zf:
    for f in ["placer_v1.pt", "placer_log.txt"] + [f"placer_eval_t{t}.txt" for t in (0.6, 0.8, 1.0)]:
        if os.path.exists(f"{W}/{f}"): zf.write(f"{W}/{f}", f)
print("\n저장 완료: result_placer.zip")
