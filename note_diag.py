"""
[진단] 노트 후보 진단 (--note-log)

모델 확률 곡선의 모든 봉우리(후보)와 음원 어택(소리 시작점)을 한 표로 남긴다.
후보마다 어느 단계에서 빠졌는지(확률 미달 / 격자 합침 / 약한 16분 정리 / 악기 따라가기 ...) 또는
최종 노트로 남았는지를 적고, 모델 후보가 없는 강한 어택도 "모델 후보 없음"으로 적는다.
엑셀에서 바로 열 수 있게 utf-8-sig CSV로 쓴다.
"""
from __future__ import annotations

import numpy as np

FPS = 100


def audio_onsets(stereo: np.ndarray, sr: int):
    """음원 어택 세기 곡선(10ms 간격)과 봉우리. 반환 (곡선 0~1+, 봉우리 프레임, 봉우리 세기)"""
    x = np.asarray(stereo, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    hop = int(round(sr / FPS))
    n = 2048
    if len(x) < n + hop:
        return np.zeros(1), np.zeros(0, int), np.zeros(0)
    nfr = 1 + (len(x) - n) // hop
    win = np.hanning(n).astype(np.float32)
    prev = None
    flux = np.zeros(nfr, dtype=np.float32)
    for s in range(0, nfr, 2048):                       # 메모리 아끼려고 나눠서
        e = min(nfr, s + 2048)
        idx = (np.arange(s, e) * hop)[:, None] + np.arange(n)[None, :]
        S = np.log1p(100.0 * np.abs(np.fft.rfft(x[idx] * win, axis=1)))
        if prev is not None:
            S_all = np.vstack([prev[None, :], S])
        else:
            S_all = np.vstack([S[:1], S])
        flux[s:e] = np.maximum(0.0, np.diff(S_all, axis=0)).sum(axis=1)
        prev = S[-1]
    # 프레임 i = 시각 (i*hop + n/2)/sr  -> 10ms 단위로 맞춤
    off = int(round((n / 2) / hop))
    curve = np.zeros(nfr + off, dtype=np.float32)
    curve[off:] = flux
    ref = float(np.percentile(curve, 99)) or 1.0
    curve = curve / ref
    pk = [i for i in range(1, len(curve) - 1)
          if curve[i] >= curve[i - 1] and curve[i] > curve[i + 1] and curve[i] >= 0.15]
    # 60ms 안에서는 가장 센 것만
    pk = sorted(pk, key=lambda i: -curve[i])
    taken = np.zeros(len(curve), bool)
    out = []
    for i in pk:
        if taken[max(0, i - 6):i + 7].any():
            continue
        taken[i] = True
        out.append(i)
    out = np.array(sorted(out), dtype=int)
    return curve, out, curve[out] if len(out) else np.zeros(0)


class NoteLog:
    """단계별로 후보 프레임이 빠지는 것을 기록한다."""

    def __init__(self, probs: np.ndarray, threshold: float, floor: float = 0.15):
        from metrics import peak_pick
        self.probs = probs
        cand = peak_pick(probs, min(floor, threshold), width=1)
        self.fate = {int(f): None for f in cand}
        self.note = {}                 # 프레임 -> 덧붙임 설명
        self.chart_t = {}              # 프레임 -> 격자에 붙은 시각(차트)
        self.threshold = threshold

    def start(self, frames):
        fs = set(int(f) for f in frames)
        for f in list(self.fate):
            if f not in fs:
                self.fate[f] = "확률 미달"
        for f in fs:
            if f not in self.fate:
                self.fate[f] = None
                self.note[f] = "32·24분 채움으로 추가"

    def drop(self, before, after, why: str):
        a = set(int(f) for f in after)
        for f in before:
            f = int(f)
            if f not in a and self.fate.get(f) is None:
                self.fate[f] = why

    def merged(self, groups, keep, frames_before, t_snapped):
        for gi, gr in enumerate(groups):
            k = int(frames_before[keep[gi]])
            for m in gr:
                f = int(frames_before[m])
                if f != k and self.fate.get(f) is None:
                    self.fate[f] = "격자 합침"
                    self.note[f] = f"{t_snapped[gi]:.3f}초 노트와 같은 칸"

    def placed(self, frames, t_chart):
        for f, t in zip(frames, t_chart):
            self.chart_t[int(f)] = float(t)

    def write(self, path, stereo, sr, pad: float, bpm: float, final_t_chart):
        curve, apk, ast = audio_onsets(stereo, sr)
        fin = np.sort(np.asarray(final_t_chart, dtype=float))
        B = 60.0 / bpm

        def strength(fr):
            lo, hi = max(0, fr - 3), min(len(curve), fr + 4)
            return float(curve[lo:hi].max()) if hi > lo else 0.0

        def beatpos(tc):
            b = tc / B
            return f"{int(b // 4) + 1:03d}마디 {b % 4 + 1:.3f}박"

        rows = []
        for f, why in self.fate.items():
            tc = self.chart_t.get(f)
            res = why
            if res is None:
                if tc is not None and len(fin) and np.min(np.abs(fin - tc)) < 1e-4:
                    res = "노트"
                else:
                    res = "후처리에서 빠짐"
            t_audio = f / FPS
            mv = "" if tc is None else f"{(tc - pad - t_audio) * 1000:+.0f}"
            rows.append((t_audio, float(self.probs[min(f, len(self.probs) - 1)]), strength(f), res,
                         "" if tc is None else f"{tc - pad:.3f}", mv,
                         beatpos(tc if tc is not None else t_audio + pad), self.note.get(f, "")))
        cand_fr = np.array(sorted(self.fate), dtype=int)
        for fr, st in zip(apk, ast):
            if st < 0.3:
                continue
            if len(cand_fr) and np.min(np.abs(cand_fr - fr)) <= 4:
                continue
            rows.append((fr / FPS, float(self.probs[min(fr, len(self.probs) - 1)]), float(st), "모델 후보 없음",
                         "", "", beatpos(fr / FPS + pad), ""))
        rows.sort(key=lambda r: r[0])
        with open(path, "w", encoding="utf-8-sig") as fp:
            fp.write("시각(초),모델확률,소리세기,결과,노트시각(초),옮긴ms,박위치,메모\n")
            for r in rows:
                fp.write(f"{r[0]:.3f},{r[1]:.3f},{r[2]:.2f},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]}\n")
        # 요약: 센 소리(세기 0.4 이상)가 어떻게 됐나
        strong = [r for r in rows if r[2] >= 0.4]
        cnt = {}
        for r in strong:
            cnt[r[3]] = cnt.get(r[3], 0) + 1
        far = sum(1 for r in rows if r[3] == "노트" and r[5] and abs(int(r[5])) > 25)
        return len(rows), len(strong), cnt, far
