"""
[생성 본체] 곡 오디오 -> (학습한 모델이 노트 시각 예측) -> BMS 파일

사용:
    python infer.py --audio input\\mysong.mp3 --ckpt checkpoints\\onset_best.pt ^
                    --lane-ckpt checkpoints\\lane_best.pt --out output

결과는 output\\<이름>\\ 아래에 <이름>.bms, <이름>.wav, <이름>_clicks.wav(클릭 미리듣기)로 저장된다.
--bpm을 생략하면 BPM을 자동 추정한다 (고정 BPM 곡만).
"""
# 생성 순서는 보고서 Fig.2-2의 7단계를 따른다 (아래 주석의 STEP 번호).
# STEP 1 곡 변환 / 2 노트 시각 모델 / 3 박자와 줄 규칙 / 4 계획 모델과 레인 모델 / 5 두께와 밀도 규칙 / 6 자리 모델과 배치 규칙 / 7 저장
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # 출력이 파이프/파일로 갈 때 Windows 로캘(cp932 등) 때문에 한글에서 멈추지 않게
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import argparse
import re
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bms_parser import Note
from bms_writer import verify_written, write_bms
from build_dataset import HOP, N_MELS, SR, log_mel
from lane_utils import cond_value, extract_patches, generate_masks
from metrics import peak_pick
from render_audio import add_clicks, read_audio, resample, write_wav

FPS = SR // HOP   # 100프레임/초 (10ms)


# ---------------------------------------------------------------------------
# 1) 오디오 -> 프레임별 노트 확률
# ---------------------------------------------------------------------------
def stitch_windows(mel: np.ndarray, predict_fn, window_frames: int, overlap_frames: int) -> np.ndarray:
    """mel[T, n_mels]을 겹치는 윈도우로 나눠 예측하고, 각 윈도우의 가운데 부분만 이어 붙인다"""
    T = len(mel)
    W, ov = window_frames, overlap_frames
    hop = W - 2 * ov
    if hop <= 0:
        raise ValueError("overlap이 너무 커요 (window > 2 x overlap 이어야 해요)")
    probs = np.zeros(T, dtype=np.float32)
    s = 0
    while True:
        seg = mel[s:s + W]
        n = len(seg)
        x = np.zeros((mel.shape[1], W), dtype=np.float32)    # 학습 때처럼 모자라면 0으로 채움
        x[:, :n] = seg.T
        p = np.asarray(predict_fn(x))[:n]
        last = s + W >= T
        a = 0 if s == 0 else ov
        b = n if last else W - ov
        probs[s + a:s + b] = p[a:b]
        if last:
            break
        s += hop
    return probs


def mel_from_audio(stereo: np.ndarray, sr0: int, normalize: bool = True) -> np.ndarray:
    """스테레오 오디오 -> (T, n_mels) 멜. 학습 때와 같은 전처리(32kHz 모노, 로그 멜)를 쓴다."""
    mono = resample(stereo, sr0, SR).mean(axis=1).astype(np.float32)
    peak = float(np.abs(mono).max())
    if normalize and peak > 0:              # 학습용 믹스는 피크가 0.98로 맞춰져 있었음
        mono = mono * (0.98 / peak)
    return log_mel(mono)


def probs_from_audio(stereo: np.ndarray, sr0: int, predict_fn, window_sec: float = 20.0,
                     overlap_sec: float = 3.0, normalize: bool = True) -> np.ndarray:
    """스테레오 오디오 -> (T,) 노트 확률."""
    mel = mel_from_audio(stereo, sr0, normalize)
    return stitch_windows(mel, predict_fn, int(window_sec * FPS), int(overlap_sec * FPS))


# ---------------------------------------------------------------------------
# 2) 박자 격자 맞추기
# ---------------------------------------------------------------------------
def grid_fit(times, bpm: float, sub: int = 4) -> tuple[float, float]:
    """노트 시각들이 (BPM 기준 1/sub 박) 격자의 어느 위상에 몰려 있는지 구한다
    반환: (phase[초], R) R은 0~1이고 클수록 격자에 잘 맞음"""
    g = 60.0 / (bpm * sub)
    th = 2.0 * np.pi * (np.asarray(times, dtype=np.float64) % g) / g
    z = np.exp(1j * th).mean()
    return float((np.angle(z) / (2.0 * np.pi) * g) % g), float(abs(z))


def assign_lanes(n: int, mode: str = "cycle", seed: int = 0) -> list[int]:
    """임시 레인 배치 (레인 모델이 생기기 전까지). 1~7번 건반만 사용."""
    if mode == "cycle":
        return [1 + (i % 7) for i in range(n)]
    rng = np.random.default_rng(seed)
    out: list[int] = []
    prev = None
    for _ in range(n):
        prev = int(rng.choice([l for l in range(1, 8) if l != prev]))
        out.append(prev)
    return out


def match_offsets(pred, true, tol: float) -> np.ndarray:
    """정렬된 두 시각 배열(초)을 일대일로 맞춰서, 맞은 쌍의 (예측 - 정답) 차이[초]를 돌려준다."""
    i = j = 0
    diffs: list[float] = []
    while i < len(pred) and j < len(true):
        d = float(pred[i]) - float(true[j])
        if abs(d) <= tol:
            diffs.append(d)
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return np.asarray(diffs)


def offset_report(pred, ref, label: str, tol: float = 0.05) -> dict:
    """예측 시각을 정답 시각과 비교한 요약. 양수 = 예측이 정답보다 늦음."""
    pred, ref = np.sort(np.asarray(pred, dtype=float)), np.sort(np.asarray(ref, dtype=float))
    d = match_offsets(pred, ref, tol) * 1000.0
    out = {"label": label, "matched": len(d), "n_pred": len(pred), "n_ref": len(ref)}
    if len(d):
        out.update(median=float(np.median(d)), mean=float(d.mean()), std=float(d.std()),
                   p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)),
                   within10=float((np.abs(d) <= 10).mean()), within20=float((np.abs(d) <= 20).mean()))
    return out


def print_offset_report(r: dict) -> None:
    if not r["matched"]:
        print(f"  [{r['label']}] 맞는 노트가 없어요")
        return
    print(f"  [{r['label']}] 정답 {r['n_ref']}개 중 {r['matched']}개 대응 | 중앙값 {r['median']:+.1f}ms, "
          f"평균 {r['mean']:+.1f}ms, 표준편차 {r['std']:.1f}ms | 하위10%~상위10% {r['p10']:+.1f}~{r['p90']:+.1f}ms | "
          f"±10ms 안 {r['within10'] * 100:.0f}%, ±20ms 안 {r['within20'] * 100:.0f}%")
    if abs(r["median"]) >= 5.0:
        way = "늦게" if r["median"] > 0 else "빠르게"
        print(f"    -> 전체적으로 약 {abs(r['median']):.0f}ms {way} 찍는 치우침이 있어요")
    elif r["std"] >= 12.0:
        print("    -> 치우침은 없지만 노트마다 흩어짐이 커요")
    else:
        print("    -> 치우침도 흩어짐도 작아요 (정답과 잘 맞음)")


def write_silence(path: Path, sr: int = 44100, sec: float = 0.1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * sec))


def write_click(path: Path, sr: int = 44100, ms: float = 30.0, freq: float = 1500.0) -> None:
    """노트 키음용 짧은 클릭 WAV (모노 16비트)."""
    t = np.arange(int(sr * ms / 1000)) / sr
    x = np.sin(2 * np.pi * freq * t) * np.exp(-t * 100) * 0.8
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_") or "song"


def snap_levels(bpm: float, min_step_ms: float = 25.0, divs=(16, 32, 64)) -> tuple[int, ...]:
    """단계별 맞춤에 쓸 분할들"""
    meas_ms = 240000.0 / bpm
    lv = tuple(N for N in divs if meas_ms / N >= min_step_ms)
    return lv or (divs[0],)


def snap_hier(t_chart: np.ndarray, bpm: float, levels: tuple[int, ...] = (16, 32),
              bias: float = 0.35, first_bias: float = 0.5) -> tuple[np.ndarray, list[np.ndarray]]:
    """단계별 격자 맞춤: 기본은 가장 성긴 격자(16분)에 붙이고, 필요한 노트만 더 촘촘한 격자로 보낸다"""
    meas = 240.0 / bpm
    t = np.asarray(t_chart, dtype=np.float64)
    pos = np.full(len(t), np.nan)
    occupied: set[float] = set()
    remaining = np.argsort(t)
    for li, N in enumerate(levels):
        step = meas / N
        b = first_bias if li == 0 else bias
        q = np.round(t[remaining] / step) * step
        err = np.abs(t[remaining] - q)
        nxt = []
        # 칸마다 가장 가까운 노트만 그 칸을 차지한다
        order = np.argsort(err)
        taken_here: set[float] = set()
        for k in order:
            i, qk = remaining[k], round(float(q[k]), 6)
            if err[k] <= b * step + 1e-9 and qk not in occupied and qk not in taken_here:
                pos[i] = qk
                taken_here.add(qk)
            else:
                nxt.append(i)
        occupied |= taken_here
        remaining = np.array(sorted(nxt, key=lambda i: t[i]), dtype=int)
        if len(remaining) == 0:
            break
    if len(remaining):                                # 남은 노트: 가장 촘촘한 격자의 가장 가까운 칸 (겹치면 합침)
        step = meas / levels[-1]
        pos[remaining] = np.round(np.round(t[remaining] / step) * step, 6)
    uniq, inv = np.unique(pos, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    return uniq, groups


def snap_times(t_chart: np.ndarray, bpm: float, divs: tuple[int, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    """노트 시각을 음악 격자(마디를 N등분한 점, N은 divs 중 하나)의 가장 가까운 점으로 옮긴다
    반환: (옮긴 뒤의 고유 시각[k], 각 고유 시각에 합쳐진 원래 노트 번호들[k])"""
    meas = 240.0 / bpm
    best = None
    for N in divs:                                   # 거리가 같으면 앞의 분할(보통 16분)을 우선
        step = meas / N
        cand = np.round(t_chart / step) * step
        if best is None:
            best = cand
        else:
            closer = np.abs(cand - t_chart) < np.abs(best - t_chart) - 1e-9
            best = np.where(closer, cand, best)
    key = np.round(best, 6)
    uniq, inv = np.unique(key, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    return uniq, groups


# ---------------------------------------------------------------------------
# 3) 확률 -> 노트 -> BMS 파일
# ---------------------------------------------------------------------------
def estimate_downbeat(mel: np.ndarray | None, bpm: float, pad: float, dur: float) -> tuple[int, list[float]]:
    """마디 첫 박 추정: 박 위치마다 '새로 들어온 소리 크기'(멜 증가분)를 재서, 4박 주기 중 가장 큰 쪽.
    (마디 첫 박에는 크래시/베이스/킥이 몰리는 경향) 반환: (0~3, 후보별 점수)"""
    if mel is None:
        return 0, [0.0] * 4
    m = np.asarray(mel, dtype=np.float32)
    flux = np.maximum(np.diff(m, axis=0, prepend=m[:1]), 0).sum(1)
    lowf = np.maximum(np.diff(m[:, :m.shape[1] // 4], axis=0, prepend=m[:1, :m.shape[1] // 4]), 0).sum(1)
    B = 60.0 / bpm
    n_beats = int((dur + pad) / B)
    sc = np.zeros(4)
    cnt = np.zeros(4)
    for k in range(n_beats):
        t = k * B - pad                                # 원본 오디오 기준 박 시각
        f = int(round(t * FPS))
        if f < 2 or f >= len(flux) - 2:
            continue
        v = flux[f - 2:f + 3].max() / (flux.mean() + 1e-6) + lowf[f - 2:f + 3].max() / (lowf.mean() + 1e-6)
        sc[k % 4] += v
        cnt[k % 4] += 1
    sc = sc / np.maximum(cnt, 1)
    sc = sc / (sc.max() + 1e-9)
    return int(np.argmax(sc)), [float(x) for x in sc]


def apply_density(density_fn, mel, frames, t_chart, probs, bpm, pad, conf=0.25, sure=0.93):
    """밀도 모델이 쉼/8비트로 본 반 마디에서만 정해진 줄 수만 남긴다 (density_model.py)"""
    from collections import defaultdict
    B = 60.0 / bpm
    t_chart = np.asarray(t_chart, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    Hn = int(t_chart.max() // (2 * B)) + 2
    k = np.arange(Hn)
    start = np.rint((2 * k * B - pad) * FPS).astype(np.int64)
    end = np.rint((2 * (k + 1) * B - pad) * FPS).astype(np.int64)
    valid = (end > 0) & (start < len(mel))
    out = density_fn(mel, np.clip(start, 0, len(mel) - 1), np.clip(end, 1, len(mel)), (k % 2).astype(np.int8))
    n_rows, n_btw = out[0], out[1]
    p16 = out[2] if len(out) > 2 else np.zeros(Hn)
    q16 = t_chart / B * 4.0
    k16 = np.rint(q16).astype(np.int64)
    on16 = np.abs(q16 - k16) < 1e-3
    half_of = np.floor(t_chart / (2 * B) + 1e-6).astype(np.int64)   # 경계 반올림 오차 방지
    pk = np.max([probs[np.clip(frames + d, 0, len(probs) - 1)] for d in range(-3, 4)], axis=0)
    score = pk + 0.15 * (on16 & (k16 % 4 == 0)) + 0.08 * (on16 & (k16 % 2 == 0))
    by_half = defaultdict(list)
    for i, h in enumerate(half_of):
        by_half[int(h)].append(i)
    keep = np.ones(len(t_chart), dtype=bool)
    sect_pred = np.zeros(3, int)
    skipped = 0
    for h, ids in by_half.items():
        if h >= Hn or not valid[h]:
            continue
        nr, nb = int(n_rows[h]), int(n_btw[h])
        sect_pred[0 if nr <= 2 else 1 if nr <= 5 else 2] += 1
        if nr >= 6 or p16[h] >= conf:
            # 16비트 구간으로 보거나 16비트일 가능성이 꽤 있는 곳은 손대지 않는다.
            skipped += nr < 6
            continue
        if nr >= 3:
            nb = min(nb, 2)                             # 8비트 구간: 16분 사이 줄은 많아야 2개
        btw = [i for i in ids if on16[i] and k16[i] % 2 == 1 and pk[i] < sure]   # 확실한 소리는 지우지 않음
        if nb < 4 and len(btw) > nb:
            for i in sorted(btw, key=lambda i: score[i])[:len(btw) - nb]:
                keep[i] = False
        alive = [i for i in ids if keep[i]]
        if len(alive) > nr:
            weak = [i for i in alive if pk[i] < sure]
            for i in sorted(weak, key=lambda i: score[i])[:len(alive) - nr]:
                keep[i] = False
    tot = max(sect_pred.sum(), 1)
    print(f"밀도 계획: 반 마디 {tot}개 중 쉼 {sect_pred[0] / tot:.0%}, 8비트 {sect_pred[1] / tot:.0%}, 16비트 {sect_pred[2] / tot:.0%} | "
          f"후보 줄 {len(keep)}개 중 {int((~keep).sum())}개 뺌, 애매해서 둔 반 마디 {skipped}개 (--density-conf로 조절), "
          f"확신도 {sure:g} 이상은 지우지 않음 (--keep-sure)")
    return frames[keep], t_chart[keep]


def fix_sections(t_chart, masks, frames, probs, bpm, max_keys: int = 5, sure: float = 0.93):
    """반 마디(2박) 단위로 사람 패턴처럼 노트를 정리한다"""
    from collections import defaultdict
    B = 60.0 / bpm
    t_chart = np.asarray(t_chart, dtype=np.float64)
    masks = [int(m) for m in masks]
    q = t_chart / B * 4.0
    k16 = np.rint(q).astype(np.int64)
    on16 = np.abs(q - k16) < 1e-3
    row_at = {int(k16[i]): i for i in range(len(k16)) if on16[i]}
    by_half = defaultdict(list)
    for i in range(len(k16)):
        by_half[int(k16[i]) // 8].append(i)
    remove = set()
    n_fold = n_rest = n_keys_moved = n_keys_lost = 0
    pk = probs[np.clip(np.asarray(frames, dtype=np.int64), 0, len(probs) - 1)]
    pk_med = float(np.median(pk))
    fr_ = np.asarray(frames, dtype=np.int64)            # 격자에 붙이며 몇 프레임 움직였을 수 있어 ±30ms 최댓값
    pk_w = np.max([probs[np.clip(fr_ + d, 0, len(probs) - 1)] for d in range(-3, 4)], axis=0)
    for h, ids in by_half.items():
        if any(not on16[i] for i in ids):          # 32분/셋잇단이 섞인 반 마디는 그대로
            continue
        btw_all = [i for i in ids if k16[i] % 2 == 1]       # 구간 판단은 전체 줄로
        btw = [i for i in btw_all if pk_w[i] < sure]       # 지우거나 합치는 건 확실하지 않은 줄만 (--keep-sure)
        evn = [i for i in ids if k16[i] % 2 == 0]
        if not btw:
            continue
        if len(ids) <= 3:                           # 쉬는 구간: 사이 줄은 빼고, 8분 칸 줄도 가장 센 것 외에 약한 건 뺀다
            remove.update(btw)
            n_rest += len(btw)
            if len(evn) >= 2:
                top = max(evn, key=lambda c: (pk[c], int(k16[c]) % 4 == 0))
                for i in evn:
                    if i != top and pk[i] < pk_med and pk_w[i] < sure:
                        remove.add(i)
                        n_rest += 1
            continue
        if len(btw_all) <= 2 and len(evn) >= 2:     # 8비트 구간: 사이 줄을 이웃 8분 줄에 합치기
            for i in btw:
                cands = [row_at[k] for k in (int(k16[i]) - 1, int(k16[i]) + 1)
                         if k in row_at and row_at[k] not in remove and k // 8 == h]
                if not cands:
                    continue
                j = max(cands, key=lambda c: (int(k16[c]) % 4 == 0, pk[c]))   # 정박 쪽, 그다음 확신 높은 쪽
                nb = [row_at[k] for k in (int(k16[j]) - 1, int(k16[j]) + 1)
                      if k in row_at and row_at[k] != i and row_at[k] not in remove]
                busy = 0
                for x in nb:
                    busy |= masks[x]
                add = masks[i] & ~masks[j]
                for b in range(8):
                    if not (add >> b) & 1:
                        continue
                    if b > 0 and ((busy >> b) & 1 or bin(masks[j] >> 1).count("1") >= max_keys):
                        n_keys_lost += 1                # 16분 연타가 되거나 너무 두꺼우면 버림
                        continue
                    masks[j] |= 1 << b
                    n_keys_moved += 1
                remove.add(i)
                n_fold += 1
    if remove:
        keep = np.array([i not in remove for i in range(len(masks))])
        print(f"구간 정리: 8비트 구간의 16분 사이 줄 {n_fold}개를 이웃 8분 줄에 합침(건반 {n_keys_moved}개 옮김, "
              f"{n_keys_lost}개는 연타/두께 때문에 버림), 쉬는 구간의 약한 줄 {n_rest}개 뺌 (끄려면 --no-section-fix)")
        return t_chart[keep], [m for m, k in zip(masks, keep) if k], np.asarray(frames)[keep]
    return t_chart, masks, frames


def chart_from_probs(probs: np.ndarray, stereo: np.ndarray, sr0: int, out_dir, name: str,
                     bpm: float, threshold: float, offset: float | None = None,
                     lane_mode: str = "cycle", tol_ms: float = 10.0, title: str | None = None,
                     clicks: bool = True, seed: int = 0, key_sound: str = "silence",
                     lane_fn=None, shift_ms: float = 0.0, snap: tuple[int, ...] | None = None,
                     snap_min_ms: float = 25.0, beat_divs: tuple[int, ...] | None = None,
                     mel: np.ndarray | None = None, downbeat: int = -1, offgrid_keep: float = 75.0,
                     thin_weak: float = 0.0, section_fix: bool = True, density_fn=None,
                     sparse_spread: bool = True, keep_sure: float = 0.93, end_chord: int = 1,
                     accent3: float = 0.0, accent3_at: str = "", accent3_phase: int = -1,
                     thick: float = 0.7, thick_range: tuple = (1.0, 3.2), spread: float = 0.3,
                     chord_style: float = 0.5, beat_contrast: float = 0.7,
                     denim_max: int = 8, alt_rows: int = 3, random_lanes: bool = False,
                     count_keep: float = 0.5, scratch_add: float = 0.7,
                     flow: float = 0.0, no_jack_ms: float = 100.0, fill32_low: float = 0.0,
                     count_smooth: float = 0.0, thin_boost: float = 0.0, rank: int = 2,
                     total: float = 300.0, scratch_pos: float = 0.0, stream_cap: float = 0.0,
                     pillar_fix: float = 0.0, eighth_overlap: float = 0.0, eighth_big: float = 0.0,
                     max_chord: int = 6, on8_match: float = 0.0, eighth_same: float = 0.0,
                     off_match: float = 0.0, hand32: float = 0.0, jack16: float = 0.0, trip_match: float = 0.0,
                     skel_swap: float = 0.0, bar_repeat: float = 0.0, bar_shape: float = 0.35, diag: float = 0.0,
                     fine_single: float = 0.0, fine_shape: float = 0.0, beat_pair: float = 0.0, edge4: float = 0.0,
                     fill_axis: float = 0.0, restart_shift: float = 0.0, section_dyn: float = 0.0,
                     eighth_trans: float = 0.0, end_fat: float = 0.0, end_over: int = 1,
                     follow: str = "off", follow_at: str = "", follow_strength: float = 0.7,
                     stems_model: str = "htdemucs", stems_device: str = "cpu",
                     scratch_model: str = "", scratch_place: float = 0.0,
                     chart_name: str = "", artist: str = "",
                     snap_fix_ms: float = 25.0, thin_local: float = 2.0, note_log: bool = True,
                     stream_hole: float = 8.0, end_shape: float = 1.0, superset_fix: float = 1.0,
                     beat_motif: float = 1.0, eighth_shape: float = 1.0, stream_fill: float = 11.0,
                     edge_axis: float = 1.0, pair_even: float = 1.0,
                     pair_min: int = 5, placer_ckpt: str = "", placer: float = 1.0,
                     placer_temp: float = 1.0, pair_avg: float = 4.0, pair_dev: float = 1.5,
                     quarter_spread: float = 1.0, trill_break: float = 1.0, eighth_audit: float = 1.0,
                     hand_balance: float = 1.0, fine_density: float = 1.0, placer_rep: float = 0.45,
                     stream_break: int = 16, placer_beat: float = 3.0, stream_target: float = 0.64,
                     stream_kpr: float = 0.0, fill_grade: float = 1.0, accent_rest: float = 1.0,
                     pair_keep: float = 1.0, fine_grid: float = 1.0, key_scale: float = 1.0,
                     accent_rest_len: float = 1.0, dens_shape: float = 1.0, dens_cap: float = 0.0,
                     stream_lift: float = 1.0, guard: float = 0.0, jack8: float = 1.0, trem: float = 0.0,
                     shape_run: float = 1.0, row_seed: float = 1.0, jack_allow: float = 0.3,
                     mel_keep: float = 1.0, ladder: float = 1.0, half_fill: float = 0.30, four_one: float = 1.0,
                     fine_bpm: float = 1.0, rest_chord: float = 1.0, dens_flat: float = 1.0,
                     bass_sound: float = 0.4, dens_dev: float = 1.0, dens_pulse: float = 1.0,
                     chord32: float = 0.0, fill32_sound: float = 0.0) -> dict:
    """lane_fn(frames) -> 노트 시각마다의 8비트 레인 마스크(동시치기 가능)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = peak_pick(probs, threshold, width=1)
    if fill32_low > 0 and snap and snap[0] == "beat" and fine_grid <= 0:
        print("16비트 이상 채우기(--fill32): 24·32비트 격자를 끈 상태(--fine-grid 0)라 건너뜀")
    elif fill32_low > 0:
        import fill32 as _f32m
        n0 = len(frames)
        st_f32 = None
        if fill32_sound > 0:
            # --fill32-sound: 놓을 자리에 분리 소리 어택이 있을 때만 (저장해 둔 것만, 없으면 예전대로)
            try:
                from melkeep_ctl import load_stems_cache
                st_f32 = load_stems_cache(Path(out_dir) / f"{name}_stems.npz", stereo, sr0, stems_model)
            except Exception:  # noqa: BLE001
                st_f32 = None
        frames, n_f32 = _f32m.fill32(probs, frames, bpm, fill32_low, stems=st_f32, ratio=fill32_sound if st_f32 is not None else 0.0)
        print(f"16비트 이상 채우기(--fill32 {fill32_low:g}): 16분·8분 셋잇단 두 노트 사이 약한 봉우리 {n_f32}개를 32분·24분으로 더함 ({n0} -> {len(frames)})"
              + (f" | 채움 소리 확인(--fill32-sound {fill32_sound:g}): 그 자리에 분리 소리(드럼·베이스·악기·보컬) 어택이 없는 {_f32m.LAST_SKIP}개는 안 더함"
                 f" (M109 채움 노트 중 어택 있는 비율 NEON 29%·DIAVOLO 46%, 일반 노트 67~98%)" if st_f32 is not None else
                 (" | 채움 소리 확인: 저장해 둔 분리 소리가 없어 확인 없이 채움" if fill32_sound > 0 else "")))
    nlog = None
    if note_log:
        try:                                   # 노트 후보 진단 (--note-log)
            from note_diag import NoteLog
            nlog = NoteLog(probs, threshold)
            nlog.start(frames)
        except Exception as e:  # noqa: BLE001
            print(f"[참고] 노트 진단을 못 켰어요: {e}")
            nlog = None
    times = np.maximum(frames / FPS + shift_ms / 1000.0, 0.0)
    if len(times) < 2:
        raise ValueError(f"노트가 {len(times)}개뿐이에요. --threshold를 낮춰 보세요.")

    # 박자 격자 위상: 16분음표 격자가 시각 0에 오도록 오디오 앞에 무음을 붙인다
    g = 60.0 / (bpm * 4)
    phase_est, R = grid_fit(times, bpm)
    phase = phase_est if offset is None else offset % g
    # 격자 대비 노트 위치(중앙값)를 곡 앞/중간/뒤로 나눠 본다. 점점 변하면 BPM이 미세하게 틀린 것
    resid = ((times - phase + g / 2) % g - g / 2) * 1000.0
    thirds = np.array_split(np.argsort(times), 3)
    drift_ms = [float(np.median(resid[ix])) if len(ix) else 0.0 for ix in thirds]
    beat_mode = bool(snap) and snap[0] == "beat"
    thin_dropped = np.zeros(0)            # --thin-weak로 뺀 줄 시각 (--thin-boost용)
    sec_meas = []                         # 악절째 8비트 동시치기로 바꾼 마디 (--stream-target)
    sec_kind, sec_rank = {}, {}           # 그 마디 종류(C/C1/H/P/L)와 소리 크기 순위
    lad_plan = {}                         # 사다리 폭타 마디 {마디: (단계 상한 또는 None, 주변보다 몇 단계 아래)} (--ladder)
    lad_rung = {}                         # 레인 뒤 정한 단계 {마디: 0~5}
    pair_tg = {}                          # 폭타 쌍 합 평균(--pair-avg)이 정한 마디별 목표 쌍 합 (--four-one이 씀)
    m_rank = {}                           # 곡 전체 마디 소리 크기 순위 (--rest-chord)
    accent_t = []                         # 치고 쉬기 큰 소리 줄 시각 (--accent-rest)
    trem_slots = set()                    # 떨림 박 격자 칸 (박당 96칸 번호, --trem)
    stems_mk = None                       # 분리 소리 어택 (--follow로 만든 것 또는 저장해 둔 것, --mel-keep)
    stems_mk_tried = False

    def _stems_mk():
        # 분리 소리는 저장해 둔 것만 씀 (Demucs를 새로 돌리면 약 6분)
        nonlocal stems_mk, stems_mk_tried
        if not stems_mk_tried and (mel_keep > 0 or dens_pulse > 0):       # --dens-pulse도 씀
            stems_mk_tried = True
            from melkeep_ctl import load_stems_cache
            stems_mk = load_stems_cache(out_dir / f"{name}_stems.npz", stereo, sr0, stems_model)
        return stems_mk
    cb_meas = []                          # 폭타를 끊고 8비트 동시치기로 바꾼 마디 (--stream-break)
    import thick_ctl as _tc_bs            # --bass-sound: 마디 소리 크기 순위에 저음(베이스)도 넣음
    _tc_bs.SOUND_LOW_W = float(np.clip(bass_sound, 0.0, 1.0)) if mel is not None else 0.0
    if beat_mode:                              # 박(4분음표) 시작이 시각 0에 오도록 (16분 칸 4개 중 박 위치를 고름)
        from beat_snap import choose_beat_phase
        B = 60.0 / bpm
        from beat_snap import DEFAULT_PATTERNS
        beat_pats = tuple(p for p in DEFAULT_PATTERNS if beat_divs is None or p[0] in beat_divs or p[0] == 4)
        pad = (-choose_beat_phase(times, bpm, phase, min_step_ms=snap_min_ms, patterns=beat_pats)) % B
        if pad < 0.002 or B - pad < 0.002:
            pad = 0.0
        # 마디 첫 박: 박 4개 중 어느 것인가. 무음을 박 단위로 더 붙여 BMS 마디선도 맞춘다
        if downbeat >= 0:
            db, db_scores = downbeat % 4, None
        else:
            db, db_scores = estimate_downbeat(mel, bpm, pad, len(stereo) / sr0)
        pad += ((4 - db) % 4) * B
        print(f"마디 첫 박: 박 4개 중 {db}번째"
              + (" (직접 지정)" if db_scores is None else
                 f" (자동 추정, 점수 {' / '.join(f'{x:.2f}' for x in db_scores)}; 틀리면 --downbeat 0~3)"))
    else:
        pad = (-phase) % g
        if pad < 0.002 or g - pad < 0.002:    # 이미 맞아 있으면 패딩 안 함
            pad = 0.0
    lead = np.zeros((int(round(pad * sr0)), 2), dtype=np.float32)
    padded = np.concatenate([lead, stereo.astype(np.float32)], axis=0)
    t_chart = times + pad

    # ---- STEP 3 박자와 줄 규칙: 노트 시각을 박자 격자에 맞추고 줄을 정리 ----
    snap_info = None
    if snap:
        beat_info = None
        beat_div_fg = None
        if beat_mode:
            from beat_snap import snap_beats
            use_fb = fine_bpm > 0 and 0 < fine_grid < 2
            if use_fb:
                # --fine-bpm: 박 분할 후보 벌점을 BPM대 사람 쓰임에 맞춤
                from beat_snap import bpm_patterns, fine_ratio
                beat_pats_s = bpm_patterns(beat_pats, bpm, fine_bpm)
            else:
                beat_pats_s = beat_pats
            t_snapped, groups, beat_info = snap_beats(t_chart, bpm, min_step_ms=snap_min_ms, patterns=beat_pats_s)
            if fine_grid < 2:
                # 셋잇단, 32분 격자를 곡 전체로 쓸지 정하고(옆 박도 같은 계열인 박이 곡 박의 4% 이상일 때만),
                # 아니면 그 박은 16분으로 다시 맞춤 -> 느린 곡의 드문드문 24, 32비트가 16비트 폭타로
                from beat_snap import fine_gate, TRI_DIVS, B32_DIVS
                if use_fb:
                    # 곡 전체 4% 문턱 대신 박마다 - 16분 대비 이득이 BPM대 문턱 이상인 박만 (노래를 따라가고 억지로 채우지 않음)
                    from beat_snap import fine_gate_beat
                    g0_fb = 10.0 / fine_grid if fine_grid < 1 else 10.0 * max(2.0 - fine_grid, 0.25)
                    ok_fg, allow_fg, rep_fg = fine_gate_beat(beat_info["beat_div"], beat_info["beat_gain"], bpm, fine_bpm, g0=g0_fb)
                else:
                    ok_fg, allow_fg, rep_fg = fine_gate(beat_info["beat_div"], fine_grid)
                # 악절 안 같은 박 자리에서 되풀이되는 셋잇단16분은 옆 박과 상관없이 셋잇단으로
                # 32분도 같은 방식이지만 오판이 많아 --fill32를 기본(0.25)보다 낮췄을 때만
                per_div = {}
                divs_p = tuple(d_ for d_ in (6, 8) if any(p[0] == d_ for p in beat_pats_s)
                               and 60.0 / bpm / d_ * 1000.0 >= snap_min_ms
                               and (d_ == 6 or (0 < fill32_low < 0.25)))
                if divs_p:
                    from beat_snap import periodic_fine
                    per_div, _ = periodic_fine(t_chart, bpm, divs=divs_p)
                    for b_, d_ in per_div.items():
                        allow_fg[int(b_)] = {int(d_)}
                per_tri = {b_ for b_, d_ in per_div.items() if d_ == 6}
                per_32 = {b_ for b_, d_ in per_div.items() if d_ == 8}
                pats_fg = tuple(p for p in beat_pats_s if p[0] == 4 or (p[0] == 6 and per_tri) or (p[0] == 8 and per_32)
                                or (p[0] in TRI_DIVS and "tri" in ok_fg) or (p[0] in B32_DIVS and "32" in ok_fg))
                if not pats_fg:
                    pats_fg = tuple(p for p in beat_pats_s if p[0] == 4) or beat_pats_s[:1]
                before_fg = beat_info["beats"]
                t_snapped, groups, beat_info = snap_beats(t_chart, bpm, min_step_ms=snap_min_ms, patterns=pats_fg,
                                                          beat_allow=allow_fg)
                beat_div_fg = beat_info["beat_div"]
                _nm = {"tri": "셋잇단", "32": "32분"}
                if use_fb:
                    _dn = {3: "셋잇단8분", 6: "셋잇단16분", 8: "32분", 12: "셋잇단32분", 16: "64분"}
                    print(f"24·32비트 박마다(--fine-bpm {fine_bpm:g}, BPM {bpm:g}): 사람 BPM대 쓰임 배수 "
                          + ", ".join(f"{_dn[d_]} {fine_ratio(d_, bpm):.2f}x" for d_ in (8, 6, 3))
                          + " -> 박 이득 문턱 " + ", ".join(f"{_dn.get(d_, d_)} {g_:.1f}" for d_, g_ in sorted(rep_fg["thr"].items()))
                          + f" (옆 박도 같은 계열이면 절반) | 셋잇단 박 {rep_fg['all']['tri']}개 중 {rep_fg['kept']['tri']}개, "
                          f"32분 박 {rep_fg['all']['32']}개 중 {rep_fg['kept']['32']}개 남김 (곡 전체 4% 문턱 대신, 사람 277: 32비트 느린 곡 1.1~1.5x / "
                          "200+ 0.1~0.2x, 8분 셋잇단 200+ 1.8~3.7x)")
                print(f"24·32비트 격자(--fine-grid {fine_grid:g}): "
                      + ", ".join(f"{_nm[f]} 박 {rep_fg['all'][f]}개(이어진 박 {rep_fg['run'][f]}개 = "
                                  f"{rep_fg['run'][f] / rep_fg['beats']:.1%})" for f in ("tri", "32"))
                      + (" | 기준 박마다(위 줄)" if use_fb else
                         (f" | 기준 {rep_fg['thr']:.1%}" if np.isfinite(rep_fg['thr']) else " | 16분만"))
                      + " -> " + (", ".join(f"{_nm[f]} 씀(남긴 박 {rep_fg['kept'][f]}개)" for f in ("tri", "32") if f in ok_fg)
                                  or "둘 다 16분으로")
                      + f" | 박별 분할 {', '.join(f'{k} {v}' for k, v in before_fg.items())} -> "
                      + ", ".join(f"{k} {v}" for k, v in beat_info["beats"].items())
                      + " (0 = 16분만, 1 = 자동, 2 = 다 허용)"
                      + "".join(f" | 악절 안 같은 박 자리에 되풀이되는 {nm_} {len(bs_)}박(마디 "
                                + ", ".join(f"{m_:03d}" for m_ in sorted(set(b_ // 4 + 1 for b_ in bs_))[:16]) + f")은 {nm_}으로"
                                for nm_, bs_ in (("셋잇단16분", per_tri), ("32분", per_32)) if bs_))
        elif snap[0] == "hier":
            t_snapped, groups = snap_hier(t_chart, bpm, tuple(snap[1:]))
        else:
            t_snapped, groups = snap_times(t_chart, bpm, tuple(snap))
        if beat_mode and snap_fix_ms > 0:
            # 25ms 넘게 밀린 노트는 32분, 24분 칸에 다시 놓고, 합쳐져 사라질 노트는 떼어 살림
            from beat_snap import refine_far
            n_far = int(sum((np.abs(t_snapped[i] - t_chart[gr]) * 1000.0 > snap_fix_ms).sum() for i, gr in enumerate(groups)))
            t_snapped, groups, n_mv, n_sp = refine_far(t_chart, t_snapped, groups, bpm, snap_fix_ms, snap_min_ms,
                                                       beat_div=beat_div_fg)
            print(f"격자 보정(--snap-fix {snap_fix_ms:g}ms): 박 격자에 붙이며 {snap_fix_ms:g}ms 넘게 밀린 노트 {n_far}개 중 "
                  f"{n_mv}개를 " + ("그 박 격자(16분 박은 16분)의 다른 칸으로" if beat_div_fg is not None else "32분·24분·셋잇단 칸으로")
                  + f" 다시 놓고, 옆 노트와 합쳐질 뻔한 {n_sp}개를 따로 살림")
        moved = np.concatenate([np.abs(t_snapped[i] - t_chart[gr]) for i, gr in enumerate(groups)]) * 1000.0
        keep = np.array([gr[np.argmin(np.abs(t_chart[gr] - t_snapped[i]))] for i, gr in enumerate(groups)])
        if nlog is not None:
            nlog.merged(groups, keep, frames, t_snapped - pad)
        snap_info = {"divs": tuple(snap), "before": len(t_chart), "after": len(t_snapped), "beat": beat_info,
                     "mean_ms": float(moved.mean()), "max_ms": float(moved.max())}
        frames, times, t_chart = frames[keep], t_snapped - pad, t_snapped
        if beat_mode and offgrid_keep > 0:
            # 16분이 아닌 칸(32분, 셋잇단)에 붙은 노트 중 확신이 약한 것은 16분으로 옮기거나(빈 칸) 합친다(찬 칸).
            # 소리가 확실한(노트 시각 모델 확률이 곡 안에서 상위권인) 노트만 32분/셋잇단으로 남긴다.
            B_ = 60.0 / bpm
            slot = np.rint(t_chart / B_ * 48).astype(np.int64)            # 박당 48칸 (16분 = 12칸마다)
            off = slot % 12 != 0
            if off.any():
                pk = probs[np.clip(frames, 0, len(probs) - 1)]
                thr_k = np.percentile(pk, offgrid_keep) if offgrid_keep < 100 else np.inf
                weak = off & (pk < thr_k)
                if weak.any():
                    s16 = np.rint(slot / 12).astype(np.int64) * 12
                    occupied = set(slot[~weak].tolist())
                    keep_m = np.ones(len(slot), dtype=bool)
                    new_slot = slot.copy()
                    for i in np.flatnonzero(weak):
                        if s16[i] in occupied:
                            keep_m[i] = False
                        else:
                            new_slot[i] = s16[i]
                            occupied.add(int(s16[i]))
                    order = np.argsort(new_slot[keep_m], kind="stable")
                    if nlog is not None:
                        nlog.drop(frames, frames[keep_m], "격자 정리(--offgrid-keep)")
                    frames = frames[keep_m][order]
                    t_chart = new_slot[keep_m][order] * B_ / 48.0
                    times = t_chart - pad
                    print(f"격자 정리: 16분이 아닌 칸의 노트 {int(off.sum())}개 중 확신이 약한 {int(weak.sum())}개 -> "
                          f"16분으로 옮김 {int((weak & keep_m).sum())}개, 이미 있는 노트와 합침 {int((~keep_m).sum())}개 "
                          f"(남긴 32분/셋잇단 {int((off & ~weak).sum())}개, --offgrid-keep {offgrid_keep:g})")
        if beat_mode and density_fn is not None and mel is not None:
            _fr0 = frames
            frames, t_chart = apply_density(density_fn, mel, frames, t_chart, probs, bpm, pad,
                                            getattr(density_fn, "conf", 0.25), keep_sure)
            times = t_chart - pad
            if nlog is not None:
                nlog.drop(_fr0, frames, "밀도 모델")
        if beat_mode and thin_weak > 0:
            # 16분 사이 칸(박의 2번째/4번째 16분)에 있는 약한 노트를 뺀다.
            # 사람 패턴과 비교하면 생성에만 있는 줄의 63%가 이 칸이고, 대부분 스트림 빈틈을 메운 약한 소리였다.
            # 사람은 이런 소리를 건너뛰고 강한 소리에 노트를 둬서 리듬이 살아난다.
            B_ = 60.0 / bpm
            slot = np.rint(t_chart / B_ * 48).astype(np.int64)
            between = (slot % 24) == 12
            pk = probs[np.clip(frames, 0, len(probs) - 1)]
            if thin_local > 0 and len(pk) > 8:
                # 곡 전체가 아니라 앞뒤 thin_local마디 안의 줄끼리 하위 N%를 정한다
                # (혼자 나오는 악기 구간처럼 확신도가 전체적으로 낮은 곳이 통째로 깎이지 않게)
                W = thin_local * 4 * B_
                o_ = np.argsort(t_chart, kind="stable")
                ts_, ps_ = t_chart[o_], pk[o_]
                lo_ = np.searchsorted(ts_, t_chart - W, side="left")
                hi_ = np.searchsorted(ts_, t_chart + W, side="right")
                thr_w = np.array([np.percentile(ps_[a:b], thin_weak) if b - a >= 8 else np.percentile(pk, thin_weak)
                                  for a, b in zip(lo_, hi_)])
                # 곡 전체 기준보다 높아지지 않게 (꽉 찬 폭타에서 확실한 16분까지 빠져 한 칸씩 비던 문제)
                thr_w = np.minimum(thr_w, np.percentile(pk, thin_weak))
            else:
                thr_w = np.percentile(pk, thin_weak)
            drop = between & (pk < thr_w)
            n_guard = 0
            if stream_hole > 0 and drop.any():
                # 앞뒤 16분, 8분 자리가 다 있는 폭타 속에서 한 칸만 비게 되는 줄은 곡 하위 stream_hole%만 뺌
                ss_ = set(slot.tolist())
                inner = np.array([all((int(x) + d) in ss_ for d in (-24, -12, 12, 24)) for x in slot])
                hole_keep = drop & inner & (pk >= np.percentile(pk, stream_hole))   # 'guard' 이름이 --guard 인자를 덮어써서 바꿈
                n_guard = int(hole_keep.sum())
                drop &= ~hole_keep
            thin_dropped = t_chart[drop].copy()
            if drop.any():
                keep_w = ~drop
                if nlog is not None:
                    nlog.drop(frames, frames[keep_w], "약한 16분 정리(--thin-weak)")
                frames, t_chart = frames[keep_w], t_chart[keep_w]
                times = t_chart - pad
                print(f"약한 16분 사이 노트 정리: {int(between.sum())}줄 중 {int(drop.sum())}줄 뺌 "
                      + (f"(확신도가 앞뒤 {thin_local:g}마디 줄 중 하위 {thin_weak:g}%보다 낮은 것, --thin-weak, --thin-local)"
                         if thin_local > 0 else f"(확신도가 곡 전체 하위 {thin_weak:g}%보다 낮은 것, --thin-weak)")
                      + (f" | 폭타 속 한 칸 빈자리가 될 {n_guard}줄은 남김(--stream-hole {stream_hole:g})" if n_guard else ""))
        if follow_strength > 0 and (str(follow).strip().lower() not in ("", "off") or str(follow_at).strip()):
            # --follow / --follow-at: 음원 분리로 드럼/악기/보컬 층을 나눠, 구간(마디 또는 초)마다 고른 층의 어택이 없는 줄을 뺀다 (stem_follow.py)
            from stem_follow import follow_rows, mode_word, parse_follow_at, stem_fluxes
            f_def, f_rng = mode_word(follow), parse_follow_at(follow_at)
            if f_def != "off" or any(r[2] != "off" for r in f_rng):
                fl = stem_fluxes(stereo, sr0, stems_model, stems_device, cache=out_dir / f"{name}_stems.npz")
                if fl is not None:
                    stems_mk, stems_mk_tried = fl, True
                    keep_f, rep = follow_rows(t_chart, times, bpm, fl, f_def, f_rng, follow_strength, seed)
                    if (~keep_f).any():
                        if nlog is not None:
                            nlog.drop(frames, frames[keep_f], "악기 따라가기(--follow)")
                        frames, t_chart = frames[keep_f], t_chart[keep_f]
                        times = t_chart - pad
                    from stem_follow import KO as ko
                    parts = [f"{ko[k]} 따라가기 마디 {rep[k][2]}개에서 {rep[k][0]}줄 중 {rep[k][1]}줄 뺌"
                             for k in ("drum", "inst", "vocal") if rep.get(k) and rep[k][2]]
                    sk = [f"{k:03d}({ko[md]})" for k, md in rep.get("skipped", [])]
                    print(f"악기 따라가기(--follow {f_def}, --follow-at \"{follow_at}\", 세기 {follow_strength:g}): "
                          + (", ".join(parts) or "해당 마디 없음")
                          + (f" | 따라갈 층 소리가 거의 없어 그대로 둔 마디 {len(sk)}개: "
                             + ", ".join(sk[:20]) + (" …" if len(sk) > 20 else "") if sk else ""))
        if beat_mode and stream_fill > 0 and len(t_chart) > 8:
            # 16분 칸이 stream_fill개 이상 찬 마디는 빈 16분 칸을 채워 폭타를 잇는다.
            B_ = 60.0 / bpm
            sl_ = np.rint(t_chart / B_ * 48).astype(np.int64)
            occ_ = set(sl_.tolist())
            add_ = []
            n_meas_f = 0
            tg_f = {}
            if fill_grade > 0:
                # 단계식 - 소리 큰 마디만 16칸까지, 나머지는 14~15 / 12~13칸까지
                from thick_ctl import fill_targets
                tg_f = fill_targets(t_chart, bpm, stream_fill, mel=mel, pad=pad, seed=seed)
            for mi in sorted(set((sl_ // 192).tolist())):
                cells = [mi * 192 + 12 * j for j in range(16)]
                o_ = sum(1 for c in cells if c in occ_)
                if o_ >= stream_fill and o_ < 16:
                    miss = [c for c in cells if c not in occ_ and not any((c + d) in occ_ for d in range(-5, 6) if d)]
                    if tg_f:
                        need_ = max(0, tg_f.get(mi, 16) - o_)
                        pr_c = lambda c: probs[min(max(int(round((c * B_ / 48.0 - pad) * FPS)), 0), len(probs) - 1)]
                        miss = sorted(miss, key=lambda c: -pr_c(c))[:need_]
                    if miss:
                        n_meas_f += 1
                        add_ += miss
            if add_:
                t_add = np.array(add_, dtype=np.float64) * B_ / 48.0
                f_add = np.maximum(np.rint((t_add - pad) * FPS).astype(np.int64), 0)
                # 가까운(30ms 안) 확률 봉우리가 있으면 그 프레임을 씀
                pk_all = np.flatnonzero((probs[1:-1] >= probs[:-2]) & (probs[1:-1] > probs[2:])) + 1
                for a_ in range(len(f_add)):
                    j_ = np.searchsorted(pk_all, f_add[a_])
                    near = [pk_all[k_] for k_ in (j_ - 1, j_) if 0 <= k_ < len(pk_all) and abs(pk_all[k_] - f_add[a_]) <= 3]
                    if near:
                        f_add[a_] = max(near, key=lambda f_: probs[min(f_, len(probs) - 1)])
                frames = np.concatenate([frames, f_add])
                t_chart = np.concatenate([t_chart, t_add])
                o_ = np.argsort(t_chart, kind="stable")
                frames, t_chart = frames[o_], t_chart[o_]
                times = t_chart - pad
                if len(thin_dropped):
                    thin_dropped = np.array([x for x in thin_dropped if np.min(np.abs(t_add - x)) > 1e-4])
                if tg_f:
                    occ2_ = set(np.rint(t_chart / B_ * 48).astype(np.int64).tolist())
                    full_ = [sum(1 for j in range(16) if mi * 192 + 12 * j in occ2_) for mi in tg_f]
                    n16_, n14_, n12_ = (sum(1 for x in full_ if x == 16), sum(1 for x in full_ if 14 <= x <= 15),
                                        sum(1 for x in full_ if 12 <= x <= 13))
                    tot_ = max(n16_ + n14_ + n12_, 1)
                    grade_ = (f" | 단계식(--fill-grade): 12칸+ 마디 16칸 {n16_ / tot_:.0%} / 14~15칸 {n14_ / tot_:.0%} / "
                              f"12~13칸 {n12_ / tot_:.0%} (사람 64 / 17 / 19%, 소리 큰 마디일수록 꽉)")
                else:
                    grade_ = ""
                print(f"폭타 잇기(--stream-fill {stream_fill:g}): 16분 칸이 {stream_fill:g}/16 이상 찬 마디 {n_meas_f}개의 빈 16분 칸 "
                      f"{len(add_)}개를 채움 (사람 lv20 바쁜 마디 16칸 다 참 48% / 생성 p71 3%)" + grade_)
        def _row_peak_p():
            # 줄마다 노트 시각 모델 확률 = 원래 봉우리 프레임 앞뒤 3프레임 중 최대 (_rows.csv와 같은 격자 이동 보정).
            # 예전에는 격자에 붙인 시각에서 읽어서, 늦게 찍히는 곡은 확실한 소리(0.7~0.97)도
            # 낮게 읽혀 '16분 뒷자리가 약한 악절'로 뽑힘 -> 8비트로 잘못 바뀜 (--stream-target, --stream-break)
            fr_ = np.asarray(frames, dtype=np.int64)
            return np.max([probs[np.clip(fr_ + d_, 0, len(probs) - 1)] for d_ in range(-3, 4)], axis=0)
        def _fine_cnt():
            # 4마디 안 같은 박 자리(1~4박째)에 16분 칸 밖(24, 32, 셋잇단) 리듬이 되풀이되는 마디 - 리듬이 핵심이라 8비트로 안 바꿈
            # 흩어진 한두 박(약한 32분 채움 등)은 안 셈
            B_ = 60.0 / bpm
            q48 = np.rint(np.asarray(t_chart) / B_ * 48).astype(np.int64)
            fb_ = set((q48[q48 % 12 != 0] // 48).tolist())
            out_ = {}
            for b_ in fb_:
                m_, k_ = b_ // 4, b_ % 4
                if any(((m_ + d_) * 4 + k_) in fb_ for d_ in (-3, -2, -1, 1, 2, 3)):
                    out_[m_] = out_.get(m_, 0) + 1
            return out_
        def _to_eighth(ms_, why_):
            # 고른 마디의 8분 자리가 아닌 줄을 빼고 빈 8분 칸을 채움 (폭타 끊기 / 구간 바꾸기 공통)
            nonlocal frames, t_chart, times
            B_ = 60.0 / bpm
            q48 = np.rint(t_chart / B_ * 48).astype(np.int64)
            drop_ = np.isin(q48 // 192, ms_) & (q48 % 24 != 0)
            if nlog is not None:
                nlog.drop(frames, frames[~drop_], why_)
            frames, t_chart = frames[~drop_], t_chart[~drop_]
            occ_ = set(np.rint(t_chart / B_ * 48).astype(np.int64).tolist())
            add_ = [m_ * 192 + 24 * j for m_ in ms_ for j in range(8) if m_ * 192 + 24 * j not in occ_]
            if add_:
                t_add = np.array(add_, dtype=np.float64) * B_ / 48.0
                f_add = np.maximum(np.rint((t_add - pad) * FPS).astype(np.int64), 0)
                frames = np.concatenate([frames, f_add])
                t_chart = np.concatenate([t_chart, t_add])
                o_ = np.argsort(t_chart, kind="stable")
                frames, t_chart = frames[o_], t_chart[o_]
            times = t_chart - pad
            return int(drop_.sum())
        def _reshape(ms_, keep_off16, max_on8, fill8, why_, extra=None):
            # 고른 마디를 8분 자리 줄만(+ 확률 높은 16분 뒷자리 keep_off16줄) 남기고, 8분 자리 줄도 max_on8줄까지(확률 순),
            # fill8이면 빈 8분 칸을 채움
            nonlocal frames, t_chart, times
            if not ms_:
                return 0
            B_ = 60.0 / bpm
            q48 = np.rint(t_chart / B_ * 48).astype(np.int64)
            pr_ = _row_peak_p()                # 격자에 붙이며 옮긴 줄도 봉우리 확률로
            ex_ = extra() if extra is not None else None
            if ex_ is not None and len(ex_) != len(pr_):
                ex_ = None
            inm = np.isin(q48 // 192, ms_)
            keep_ = ~inm | (q48 % 24 == 0)
            for m_ in ms_:
                ix_ = np.flatnonzero((q48 // 192 == m_) & (q48 % 24 == 12))
                ko_ = int(keep_off16.get(m_, 3)) if isinstance(keep_off16, dict) else int(keep_off16)   # 마디마다 1~3
                if ko_ > 0 and len(ix_):
                    sc_ = pr_[ix_] if ex_ is None else pr_[ix_] + ex_[ix_]   # 멜로디 점수(곡 안 순위 0~1) 더함
                    keep_[ix_[np.argsort(-sc_)[:ko_]]] = True
                if max_on8 < 8:
                    ix8 = np.flatnonzero((q48 // 192 == m_) & (q48 % 24 == 0))
                    if len(ix8) > max_on8:
                        keep_[ix8[np.argsort(-pr_[ix8])[max_on8:]]] = False
            drop_ = ~keep_
            if nlog is not None:
                nlog.drop(frames, frames[keep_], why_)
            frames, t_chart = frames[keep_], t_chart[keep_]
            if fill8:
                occ_ = set(np.rint(t_chart / B_ * 48).astype(np.int64).tolist())
                add_ = [m_ * 192 + 24 * j for m_ in ms_ for j in range(8) if m_ * 192 + 24 * j not in occ_]
                if add_:
                    t_add = np.array(add_, dtype=np.float64) * B_ / 48.0
                    f_add = np.maximum(np.rint((t_add - pad) * FPS).astype(np.int64), 0)
                    frames = np.concatenate([frames, f_add])
                    t_chart = np.concatenate([t_chart, t_add])
                    o_ = np.argsort(t_chart, kind="stable")
                    frames, t_chart = frames[o_], t_chart[o_]
            times = t_chart - pad
            return int(drop_.sum())
        if beat_mode and stream_target > 0 and len(t_chart) > 64:
            # 곡 전체 16분 폭타 마디 비율이 사람(폭타 위주 곡 중앙값 64%)보다 높으면 4~8마디 악절을 통째로 바꿈
            # 한 가지(8비트 동시치기 0.5배)로 바꾸지 않고 4마디 단위로 소리 크기 순서대로
            #          드문드문 / 8비트 단타 / 8비트 동시치기 / 반쯤 빈 16비트
            from thick_ctl import section_break_pick, section_kinds, SECTION_KIND_KO, rank_rung, half_off_counts, LADDER_KO
            rp_ = _row_peak_p()                # 격자에 붙이며 옮긴 줄도 봉우리 확률로
            fc_ = _fine_cnt()
            if bass_sound > 0 and mel is not None:
                from thick_ctl import bass_gap_meas
                bg_ = bass_gap_meas(mel, bpm, pad, int(np.max(np.floor(t_chart / (4 * 60.0 / bpm) + 1e-9))) + 2)
                print(f"소리에 베이스 넣기(--bass-sound {bass_sound:g}): 마디 소리 크기 = 멜 전체 {1 - min(bass_sound, 1.0):.1f} + 저음(멜 0~10번 대역) "
                      f"{min(bass_sound, 1.0):.1f} (곡 안 z값) - 폭타 구간 고르기·악절 종류·사다리 단계·반쯤 빈 마디 채우기·쉬는 동시치기 크기에 씀 | "
                      f"저음 순위가 멜 순위보다 0.3+ 높은 마디 {len(bg_)}개"
                      + (f"({', '.join(f'{m_ + 1:03d}' for m_ in bg_[:12])}{' …' if len(bg_) > 12 else ''})" if bg_ else "")
                      + " (사람 277개: 멜 하위 1/3 마디 중 저음 하위 1/3 폭타 22% / 저음 상위 1/3 폭타 53%)")
            sec_meas = section_break_pick(t_chart, rp_, bpm, float(stream_target), mel=mel, pad=pad, fine_cnt=fc_, fine_min=1)
            if sec_meas:
                n_meas_ = int(np.max(np.floor(t_chart / (4 * 60.0 / bpm) + 1e-9))) + 2
                sk_info = {}
                sec_kind, sec_rank = section_kinds(sec_meas, bpm, n_meas_, mel=mel, pad=pad, t_chart=t_chart, row_prob=rp_,
                                                   ladder=ladder > 0, info=sk_info)
                if ladder > 0 and dens_flat > 0:
                    # --dens-flat: 소리 편차가 적은 곡은 드문드문 대신 사다리 폭타(조금 쉬는 폭타)
                    from thick_ctl import loud_range as _lrng
                    lr_st = _lrng(mel, bpm, pad, n_meas_)
                    if np.isfinite(lr_st) and lr_st < 0.065:
                        n_p = sum(1 for m_ in sec_meas if sec_kind.get(m_) == "P")
                        for m_ in sec_meas:
                            if sec_kind.get(m_) == "P":
                                sec_kind[m_] = "L"
                        if n_p:
                            print(f"소리 폭 맞춤(--dens-flat): 소리 폭 {lr_st:.3f}(0.065 미만)이라 폭타 구간 바꾸기의 드문드문 {n_p}마디를 사다리 폭타로")
                mk_extra = None

                def _mk_rank():
                    from melkeep_ctl import row_melodic
                    sc_mk, _ = row_melodic(frames, None, _stems_mk())
                    if sc_mk is None:
                        return None
                    rk_ = np.empty(len(sc_mk))
                    rk_[np.argsort(sc_mk, kind="mergesort")] = np.arange(len(sc_mk)) / max(len(sc_mk) - 1, 1)
                    return rk_
                if mel_keep > 0:
                    # 보컬, 악기가 센 4마디(8비트 단타, 8비트 동시치기로 정해진 곳)는 반쯤 빈 16비트로 (melkeep_ctl.py)
                    from melkeep_ctl import meas_vocal_rank, melodic_units
                    st_ = _stems_mk()
                    if st_ is None:
                        print(f"멜로디 지키기(--mel-keep {mel_keep:g}): 폭타 구간 바꾸기 종류는 그대로 - 저장해 둔 분리 소리(<이름>_stems.npz)가 "
                              "없음 (악기 따라가기 --follow를 한 번 쓰면 생김)")
                    else:
                        to_mk = "L" if ladder > 0 else "H"     # 사다리면 16분 줄을 다 남기는 사다리 폭타로
                        sec_kind, ch_mk = melodic_units(sec_kind, meas_vocal_rank(st_, bpm, pad, n_meas_), mel_keep,
                                                        kinds=("C1", "C", "H") if ladder > 0 else ("C1", "C"), to=to_mk)
                        if ch_mk or ladder > 0:
                            mk_extra = _mk_rank
                        print(f"멜로디 지키기(--mel-keep {mel_keep:g}): 폭타 구간 바꾸기에서 보컬·악기 어택 비중이 곡 안 위쪽 "
                              f"{0.4 * min(mel_keep, 2.5):.0%}인 4마디 {len(ch_mk)}곳을 {SECTION_KIND_KO[to_mk]}로"
                              + (" - " + ", ".join(f"{a_ + 1:03d}-{b_ + 1:03d} {SECTION_KIND_KO[k_]} -> {SECTION_KIND_KO[to_mk]}({w_} 순위 {r_:.2f})"
                                                   for a_, b_, k_, r_, w_ in ch_mk) if ch_mk else "")
                              + (" (16분 줄은 그대로, 건반만 줄임)" if ladder > 0 else " (16분 뒷자리 3줄은 모델 확률 + 멜로디 점수 순)"))
                by_ = {k_: [m_ for m_ in sec_meas if sec_kind.get(m_) == k_] for k_ in ("C", "C1", "H", "P", "L")}
                why_ = "폭타 구간 바꾸기(--stream-target)"
                units_, cur_ = [], []
                for m_ in sec_meas:
                    if cur_ and (m_ != cur_[-1] + 1 or len(cur_) >= 4):
                        units_.append(cur_); cur_ = []
                    cur_.append(m_)
                if cur_:
                    units_.append(cur_)
                # 반쯤 빈 16비트 뒷자리 수는 마디마다 1~3 (그 마디 16분 소리 세기 순위 -> 사람 분포), 사다리는 단위 소리 순위로 단계 상한
                ho_ = half_off_counts(t_chart, rp_, bpm, by_["H"], sk_info.get("ref16") or 0.5) if (ladder > 0 and by_["H"]) else 3
                u_r = {}
                for u_ in units_:
                    r_u = float(np.mean([sec_rank.get(m_, 0.5) for m_ in u_]))
                    for m_ in u_:
                        u_r[m_] = r_u
                        if sec_kind.get(m_) == "L":
                            lad_plan[m_] = (rank_rung(r_u), 1)
                lad_acc = {}
                if by_["L"]:
                    # 사다리 마디가 1111 1111처럼 강세 없이 평평해짐 -
                    # (1) 모델 소리가 없는 16분 뒷자리 줄(확률 0.30 미만, 폭타 잇기가 채운 칸)은 뺌,
                    # (2) 8분 자리 소리가 16분 뒷자리보다 뚜렷하면(확률 평균 차 0.15+) 최소 2+1 (박마다 8분 자리에 2개)
                    q48_l = np.rint(t_chart / (60.0 / bpm) * 48).astype(np.int64)
                    lm_ = set(by_["L"])
                    for m_ in by_["L"]:
                        ix8_ = np.flatnonzero((q48_l // 192 == m_) & (q48_l % 24 == 0))
                        ixo_ = np.flatnonzero((q48_l // 192 == m_) & (q48_l % 24 == 12))
                        if len(ix8_) and len(ixo_):
                            lad_acc[m_] = float(np.mean(rp_[ix8_]) - np.mean(rp_[ixo_]))
                        if lad_acc.get(m_, 0.0) >= 0.15:
                            c_, d_ = lad_plan[m_][:2]
                            lad_plan[m_] = (max(1, c_ if c_ is not None else 1), d_, 1)
                    drop_l = np.isin(q48_l // 192, list(lm_)) & (q48_l % 24 == 12) & (rp_ < 0.30)
                    if drop_l.any():
                        if nlog is not None:
                            nlog.drop(frames, frames[~drop_l], "사다리 폭타 소리 없는 16분 뒷자리(--ladder)")
                        frames, t_chart = frames[~drop_l], t_chart[~drop_l]
                        times = t_chart - pad
                        rp_ = rp_[~drop_l]
                    print(f"사다리 폭타 소리 따라(--ladder): 소리 없는 16분 뒷자리 {int(drop_l.sum())}줄 뺌(모델 확률 0.30 미만), "
                          f"8분 자리 소리가 뚜렷한(16분 뒷자리보다 확률 0.15+ 큼) {sum(1 for v_ in lad_acc.values() if v_ >= 0.15)}/{len(lm_)}마디는 최소 2+1(박 강세)")
                nd_ = _reshape(by_["C"] + by_["C1"], 0, 8, True, why_)
                nd_ += _reshape(by_["H"], ho_, 8, True, why_, extra=mk_extra)
                nd_ += _reshape(by_["P"], 0, 4, False, why_)
                cb_meas = list(sec_meas)

                def _ulab(u_):
                    k_ = sec_kind[u_[0]]
                    if k_ == "L":
                        return f"사다리 폭타(상한 {LADDER_KO[lad_plan[u_[0]][0]]})"
                    if k_ == "H" and isinstance(ho_, dict):
                        return "반쯤 빈 16비트(뒷자리 " + "·".join(str(ho_.get(m_, 3)) for m_ in u_) + ")"
                    return SECTION_KIND_KO[k_]
                if ladder > 0:
                    rule_ = ("종류는 곡 전체 소리 크기 순위로 - 하위 20% 드문드문 / 20~30% 8비트 단타 / 그 위 사다리 폭타(16분 줄 그대로, 두께만 - 상한 "
                             "0.30~0.45 1+1 / 0.45~0.65 2+1 / 그 위 2+2, 주변 폭타보다 한 단계+ 아래), 단 16분 소리가 곡 폭타 중 하위 25%면 반쯤 빈 16비트"
                             "(뒷자리 1~3, 사람 27/32/41%) 또는 곡 상위 30%면 8비트 동시치기 / 사람 277개 소리 순위 0~10 / 10~20 / 20~30%: "
                             "드문드문 49 / 35 / 23%, 8비트 단타 12 / 11 / 8%, 얇은 폭타 12 / 12 / 15%")
                else:
                    rule_ = ("종류는 곡 전체 소리 크기 순위로 - 하위 25% 드문드문 / 25~40% 8비트 단타 / 그 위 8비트 동시치기, 16분 소리가 있는 큰 곳은 반쯤 빈 16비트 / "
                             "사람 189곡 폭타 사이 덩어리: 드문드문 35% 0.16배 / 8비트 단타 16% 0.28배 / 8비트 동시치기 36% 0.67배 / 반쯤 빈 16비트 14% 0.65배")
                orank_ = sk_info.get("orank", {})
                print(f"폭타 구간 바꾸기(--stream-target {stream_target:g}{', --ladder ' + format(ladder, 'g') if ladder > 0 else ''}): "
                      f"16분 폭타 마디가 너무 많아 {len(sec_meas)}마디를 4마디씩 "
                      + ", ".join(f"{u_[0] + 1:03d}-{u_[-1] + 1:03d} {_ulab(u_)}"
                                  + (f"[소리 {u_r[u_[0]]:.2f}, 16분 {np.mean([orank_.get(m_, 0.5) for m_ in u_]):.2f}]" if ladder > 0 else "")
                                  for u_ in units_)
                      + f"으로, 줄 {nd_}개 뺌 (4마디 안 같은 박 자리에 24·32·셋잇단 리듬이 되풀이되는 악절은 안 고름 - 그런 마디 {len(fc_)}개, "
                        f"약하고 조용한 악절·소리가 바뀌는 곳 우선, {rule_})")
        if beat_mode and stream_break > 0 and len(t_chart) > 32:
            # 폭타가 stream_break마디 넘게 쉬지 않으면 16분 뒷자리가 약한 마디를 8비트 동시치기 마디로 (사람 최장 17마디)
            from thick_ctl import stream_break_pick
            rp_ = _row_peak_p()                # 격자에 붙이며 옮긴 줄도 봉우리 확률로
            sb_meas = stream_break_pick(t_chart, rp_, bpm, int(stream_break), mel=mel, pad=pad, fine_cnt=_fine_cnt(),
                                        skip=set(lad_plan))   # 사다리로 얇게 한 마디는 이미 끊긴 것으로
            # 폭타 끊기는 다시 8비트 동시치기로 (사다리로 끊으면 박 강세가 사라짐)
            if sb_meas:
                if not m_rank and mel is not None:
                    from sound16_ctl import meas_rank
                    m_rank = meas_rank(mel, bpm, pad, int(np.max(np.floor(t_chart / (240.0 / bpm) + 1e-9))) + 2)
                nd_ = _to_eighth(sb_meas, "폭타 끊기(--stream-break)")
                cb_meas = sorted(set(cb_meas) | set(sb_meas))
                print(f"폭타 끊기(--stream-break {int(stream_break)}): 쉬지 않는 폭타 안 16분 뒷자리가 약한 마디 {len(sb_meas)}개"
                      f"({', '.join(f'{m_ + 1:03d}' for m_ in sb_meas[:12])}{' …' if len(sb_meas) > 12 else ''})를 8비트 동시치기로, "
                      f"16분 뒷자리 {nd_}줄 뺌 (4마디 악절 처음·끝 1~2마디, 소리가 바뀌는 곳 우선 / 사람 277곡 폭타 최장 연속 중앙값 17마디)")
        if beat_mode and half_fill > 0 and len(t_chart) > 64:
            # --half-fill: 곡 소리 상위 50% 마디 중 반쯤 빈 16비트이면서 앞이나 뒤가
            # 폭타인 곳은 빈 16분 뒷자리를 모델 확률 half_fill 이상이면 채움.
            # 폭타 구간 바꾸기, 폭타 끊기가 고른 마디는 안 건드림 (일부러 비운 곳)
            from thick_ctl import half_fill_pick
            add_hf, rep_hf = half_fill_pick(t_chart, probs, bpm, pad, FPS, mel, min_p=float(half_fill), skip=set(cb_meas))
            if add_hf:
                B_ = 60.0 / bpm
                t_add = np.array(add_hf, dtype=np.float64) * B_ / 48.0
                f_add = np.maximum(np.rint((t_add - pad) * FPS).astype(np.int64), 0)
                pk_all = np.flatnonzero((probs[1:-1] >= probs[:-2]) & (probs[1:-1] > probs[2:])) + 1
                for a_ in range(len(f_add)):
                    j_ = np.searchsorted(pk_all, f_add[a_])
                    near = [pk_all[k_] for k_ in (j_ - 1, j_) if 0 <= k_ < len(pk_all) and abs(pk_all[k_] - f_add[a_]) <= 3]
                    if near:
                        f_add[a_] = max(near, key=lambda f_: probs[min(f_, len(probs) - 1)])
                frames = np.concatenate([frames, f_add])
                t_chart = np.concatenate([t_chart, t_add])
                o_ = np.argsort(t_chart, kind="stable")
                frames, t_chart = frames[o_], t_chart[o_]
                times = t_chart - pad
                n12_ = sum(1 for _, a_, b_, _ in rep_hf if b_ >= 12)
                print(f"소리 큰 곳 반쯤 빈 마디 채우기(--half-fill {half_fill:g}): 곡 소리 상위 50% 반쯤 빈 16비트 마디 중 앞뒤가 폭타인 "
                      f"{len(rep_hf)}마디에 16분 뒷자리 {len(add_hf)}개 더함 (모델 확률 {half_fill:g}+, 그중 16분 칸 12+ 된 마디 {n12_}개) | "
                      + ", ".join(f"{m_ + 1:03d} {a_}->{b_}칸" for m_, a_, b_, _ in rep_hf[:10]) + (" …" if len(rep_hf) > 10 else "")
                      + " (사람 소리 상위 반쯤 빈 16비트 10~12% / 생성 M105 15~22%)")
        if beat_mode and accent_rest > 0 and len(t_chart) > 32:
            # 폭타 중 '따당' 하고 큰 소리가 난 뒤 소리가 잦아드는 곳 -> 치고 쉼
            try:
                from note_diag import audio_onsets as _aon
                from thick_ctl import accent_rest_pick
                cv_ar, _, _ = _aon(stereo, sr0)
                drop_ar, hits_ar, info_ar = accent_rest_pick(t_chart, bpm, cv_ar, pad, ratio=2.0 / max(accent_rest, 1e-3),
                                                             rest=float(accent_rest_len))
                if len(info_ar):
                    accent_t = [float(t_chart[h_]) for h_ in hits_ar]
                    keep_ar = np.ones(len(t_chart), dtype=bool)
                    keep_ar[drop_ar] = False
                    if nlog is not None:
                        nlog.drop(frames, frames[keep_ar], "치고 쉬기(--accent-rest)")
                    frames, t_chart = frames[keep_ar], t_chart[keep_ar]
                    times = t_chart - pad
                    B_ = 60.0 / bpm
                    print(f"치고 쉬기(--accent-rest {accent_rest:g}): 폭타 중 큰 소리 뒤 소리가 잦아드는 {len(info_ar)}곳("
                          + ", ".join(f"{int(x_ / B_ // 4) + 1:03d}마디 {x_ / B_ % 4 + 1:.2f}박" for x_, _, _, _ in info_ar[:10])
                          + (" …" if len(info_ar) > 10 else "")
                          + f")에서 뒤 {accent_rest_len:g}박 줄 {len(drop_ar)}개 뺌, 그 소리는 동시치기로 (조건: 앞 2박 보통 세기의 {2.0 / max(accent_rest, 1e-3):g}배 이상, "
                            "뒤 1박이 그 소리의 0.6배 이하이면서 앞 폭타 보통 세기 이하)")
            except Exception as e_:  # noqa: BLE001
                print(f"치고 쉬기 건너뜀: {e_}")
        if beat_mode and trem > 0 and len(t_chart) > 32:
            # 떨림(드르릉) 살리기 - 모델 봉우리(0.15+)가 초당 16개 이상 이어지는 박을 그 속도(24, 32분) 격자로 다 채움,
            # 마지막에 단타로 (trem_ctl.py). 치고 쉬기로 비운 박은 건너뜀
            try:
                from trem_ctl import trem_beats, skip_rest, trem_fill, TREM_RATE
                from metrics import peak_pick as _pp_tr
                tb_, _ = trem_beats(probs, bpm, pad, shift_ms / 1000.0, FPS, trem)
                tb_ = skip_rest(tb_, accent_t, bpm, float(accent_rest_len) if accent_rest > 0 else 0.0)
                if tb_:
                    f_b4 = np.asarray(frames, dtype=np.int64)
                    t_chart, frames, keep_tr, n_add_tr, n_drop_tr, trem_slots = trem_fill(
                        t_chart, frames, bpm, pad, tb_, FPS, _pp_tr(probs, 0.15, width=1))
                    times = t_chart - pad
                    if nlog is not None:
                        nlog.drop(f_b4, f_b4[keep_tr], "떨림 격자로(--trem)")
                        new_f = set(int(x) for x in frames) - set(int(x) for x in f_b4[keep_tr])
                        for f_ in new_f:
                            if nlog.fate.get(f_, "확률 미달") == "확률 미달":
                                nlog.fate[f_] = None
                                nlog.note[f_] = "떨림 채우기(--trem)"
                    n24_ = sum(1 for v in tb_.values() if v == 6)
                    ms_tr = sorted(set(b_ // 4 + 1 for b_ in tb_))
                    print(f"떨림 살리기(--trem {trem:g}): 모델 봉우리(확률 0.15+)가 초당 {TREM_RATE / trem:.0f}개 이상이고 16분 칸 밖이 2개 이상인 박이 "
                          f"2박 넘게 이어진 {len(tb_)}박(24분 {n24_} / 32분 {len(tb_) - n24_}, 마디 "
                          + ", ".join(f"{m_:03d}" for m_ in ms_tr[:24]) + (" …" if len(ms_tr) > 24 else "")
                          + f")을 떨림 속도 격자로 다 채움: 줄 {n_add_tr}개 더하고 격자 밖 {n_drop_tr}개 뺌 "
                            "(구간 밀도 분포가 안 빼고, 마지막에 단타로 - 박 머리만 최대 2개)")
            except Exception as e_:  # noqa: BLE001
                print(f"떨림 살리기 건너뜀: {type(e_).__name__}: {e_}")
                trem_slots = set()
        if nlog is not None:
            nlog.placed(frames, t_chart)
        # 레인 배치(16분 연타 금지 등)는 격자에 붙인 뒤의 시각으로 판단해야 한다.
        # 예전에는 붙이기 전 시각을 써서, 105ms 떨어진 두 노트가 16분(66ms) 이웃 칸으로 붙으면 16분 연타가 생겼다.
        frames = np.maximum(np.rint(times * FPS).astype(np.int64), 0)

    # ---- STEP 4 계획 모델과 레인 모델: 줄마다 건반 초안 ----
    # 이어서 같은 블록 안에서 STEP 5 두께와 밀도 규칙(thick_ctl.py 등)을 차례로 적용
    if lane_fn is not None:
        masks = [int(m) for m in lane_fn(frames, t_chart if beat_mode else None, bpm)]
        if random_lanes:
            # 확인용 (--random-lanes): 줄마다 건반 수는 모델 그대로, 자리만 무작위 (100ms 안 바로 앞 줄 건반은 피함)
            rr = np.random.default_rng(seed + 101)
            for i, m in enumerate(masks):
                k = bin(m & 0xFE).count("1")
                block = masks[i - 1] if i > 0 and (t_chart[i] - t_chart[i - 1]) < 0.1 else 0
                free = [b for b in range(1, 8) if not (block >> b) & 1]
                if len(free) < k:
                    free = list(range(1, 8))
                masks[i] = (m & 1) | sum(1 << int(b) for b in rr.choice(free, size=k, replace=False))
            print("확인용: 레인 자리를 무작위로 바꿈 (건반 수는 모델 그대로) --random-lanes")
        raw_lane = {round(float(t), 4): int(m) for t, m in zip(t_chart, masks)}
        if beat_mode:
            from lane_v2 import rhythm_codes_from_times
            codes = rhythm_codes_from_times(t_chart, bpm, 0)
            sc = np.array([(m & 1) == 1 for m in masks])
            if sc.any():
                print(f"스크래치: {int(sc.sum())}줄 ({sc.mean():.1%}) | 그중 마디 첫 박 {np.mean(codes[sc] == 0):.0%}, "
                      f"정박 {np.mean(np.isin(codes[sc], (0, 4, 8, 12))):.0%} (사람 차분 보통 25~55% / 55~90%)")
        from coverage_ctl import completes_repeat, coverage_stats, stream_flags
        sf = stream_flags(np.round(t_chart * FPS).astype(np.int64), 60.0 / bpm / 4 * 1.3 * FPS)
        cs = coverage_stats(masks, sf)
        n_rep = sum(1 for i in range(3, len(masks)) if sf[i] and completes_repeat(masks[max(0, i - 7):i], masks[i]))
        if sf.sum() >= 16:
            print(f"스트림에서 같은 2~4줄 묶음이 그대로 두 번 반복된 곳: {n_rep}곳 (스트림 {int(sf.sum())}줄)")
        if cs is not None:
            kc = np.array([bin(int(m) >> 1).count("1") for m in masks])[sf]
            print(f"스트림 8줄마다 7건반을 다 쓰는 비율 {cs:.0%} (사람 크로노미아 94~98%, 지난 생성 약 80%) | "
                  f"스트림 평균 동시치기 {kc.mean():.2f}, 3개 이상 {np.mean(kc >= 3):.0%} (사람 약 1.85 / 19%)")
    else:
        masks = [1 << l for l in assign_lanes(len(t_chart), lane_mode, seed)]
    if beat_mode and section_fix and lane_fn is not None:
        t_chart, masks, frames = fix_sections(t_chart, masks, frames, probs, bpm, sure=keep_sure)
        times = t_chart - pad
    if beat_mode and sparse_spread and lane_fn is not None:
        from sparse_ctl import key_share, spread_sparse_singles
        before = key_share(masks, t_chart, bpm)
        masks, n_mv, n_sp = spread_sparse_singles(t_chart, masks, bpm, seed)
        if n_sp:
            print(f"성긴 구간 단타 {n_sp}개 중 {n_mv}개 건반 옮김 (한 건반 쏠림 방지) | 가장 많이 쓴 건반 비율 "
                  f"{before.max():.0%} -> {key_share(masks, t_chart, bpm).max():.0%}")
    if beat_mode and thick > 0 and lane_fn is not None and mel is not None and len(frames) == len(masks):
        from thick_ctl import apply_thickness
        masks, n_tc, n_tt, tst = apply_thickness(t_chart, frames, masks, mel, bpm, strength=thick,
                                                 lo=thick_range[0], hi=thick_range[1], seed=seed)
        if tst is not None:
            print(f"곡 세기에 따른 두께(--thick {thick:g}): 연타 {n_tt}줄 중 {n_tc}줄 바꿈 | 가장 빈 25% 평균 두께 "
                  f"{tst[0]:.2f} -> {tst[1]:.2f}, 가장 꽉 찬 25% {tst[2]:.2f} -> {tst[3]:.2f}")
    if beat_mode and lane_fn is not None and mel is not None and len(frames) == len(masks) and (accent3 > 0 or accent3_at):
        from accent3_ctl import (apply_accent3, attack_strength, band_strength, find_accent3, manual_roles,
                                 parse_ranges)
        if accent3_at:
            ranges = parse_ranges(accent3_at)
            role, phs = manual_roles(t_chart, np.asarray(frames) / FPS, band_strength(mel, frames), bpm, ranges,
                                     accent3_phase)
            how = "지정 구간 " + ", ".join(f"{a:g}~{b:g}초(강세 위상 {p})" for (a, b), p in zip(ranges, phs))
        else:
            role = find_accent3(t_chart, attack_strength(mel, frames), bpm, min_support=accent3)
            how = "자동 찾기"
        if (role > 0).any():
            masks, n_th, n_tn = apply_accent3(t_chart, masks, role, seed)
            print(f"점8분 묶음({how}): 강세 줄 {int((role == 1).sum())}개 (두껍게 {n_th}), "
                  f"사이 줄 {int((role == 2).sum())}개 (단타로 {n_tn})")
        else:
            print(f"점8분 묶음({how}): 해당하는 16분 줄 없음")
    if beat_mode and spread > 0 and lane_fn is not None:
        from thick_ctl import adjacent_ratio, spread_chords
        a0 = adjacent_ratio(masks)
        masks, n_sp = spread_chords(t_chart, masks, spread, seed)
        print(f"동시치기 벌리기(--spread-chords {spread:g}): {n_sp}줄 | 붙은 건반이 있는 동시치기 {a0:.0%} -> "
              f"{adjacent_ratio(masks):.0%} (사람 차분 약 30~50%)")
    if beat_mode and count_keep > 0 and lane_fn is not None:
        from thick_ctl import _keys, chord_count_keep, style_persistence
        kp = lambda L: style_persistence(t_chart, L, bpm, key=lambda x: len(_keys(x)))
        p0 = kp(masks)
        masks, n_ck = chord_count_keep(t_chart, masks, bpm, count_keep, seed)
        print(f"반 마디 동시치기 개수 유지(--count-keep {count_keep:g}): {n_ck}줄 (건반 하나씩만 더하거나 뺌) | "
              f"반 마디 건반 수 유지 {p0:.0%} -> {kp(masks):.0%} (사람 61~77%)")
    if beat_mode and thin_boost > 0 and lane_fn is not None and len(thin_dropped):
        from thick_ctl import thin_boost as _tb
        masks, n_tb = _tb(t_chart, masks, bpm, thin_dropped, thin_boost, seed, jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0,
                          max_notes=max_chord)
        print(f"덜어낸 16분 옆 8분 동시치기 +1건반(--thin-boost {thin_boost:g}): {n_tb}줄 (--thin-weak로 뺀 {len(thin_dropped)}줄 자리)")
    if beat_mode and count_smooth > 0 and lane_fn is not None:
        from thick_ctl import count_smooth as _csm
        masks, n_sm = _csm(t_chart, masks, bpm, count_smooth, seed, jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0)
        print(f"8분 자리 동시치기 수 고르게(--count-smooth {count_smooth:g}): {n_sm}줄 (8분 자리가 이어지는 구간에서 4 1 3 같은 단타 떨어짐·2개 넘는 점프를 ±1로, --thin-weak와 같이 써도 됨)")
    if beat_mode and flow > 0 and lane_fn is not None and mel is not None and len(frames) == len(masks):
        from thick_ctl import chord_flow
        masks, n_fl = chord_flow(t_chart, frames, masks, mel, bpm, flow, seed)
        print(f"동시치기 방향 이어 가기(--flow {flow:g}): {n_fl}줄 (75 64 다음 75 대신 53처럼, 소리가 다를 때만)")
    if beat_mode and lane_fn is not None and (stream_cap > 0 or pillar_fix > 0 or eighth_overlap > 0 or eighth_big > 0
                                              or on8_match > 0 or off_match > 0):
        from thick_ctl import _keys, eighth_chords, pillar_fix as _pf, row_kind, stream_cap as _sc
        kind = row_kind(t_chart, bpm)

        def big_share(k, lo):
            a = np.array([len(_keys(m)) for m, kk in zip(masks, kind) if kk == k])
            return float((a >= lo).mean()) if len(a) else 0.0
        b16, b8 = big_share(16, 4), big_share(8, 4)
        masks, n_cap = _sc(t_chart, masks, bpm, stream_cap, seed=seed) if stream_cap > 0 else (masks, 0)  # 건반마다 확률로 덜기
        if on8_match > 0:
            from thick_ctl import on8_match as _om
            tg_om = None
            if guard > 0:
                from thick_ctl import guard_on8_target
                tg_om = guard_on8_target(t_chart, masks, bpm)
            masks, n_om = _om(t_chart, masks, bpm, on8_match, seed, max_chord=max_chord,
                              jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0, target=tg_om)
            print(f"폭타 속 8분 자리 두께를 사람 분포로(--on8-match {on8_match:g}): {n_om}줄에서 건반 하나씩 빼거나 더함 "
                  + (("(울타리 --guard: 사람 곡별 10~90% 범위 밖만, 목표 " + " / ".join(f"{k}{'+' if k == 5 else ''}개 {v:.0%}" for k, v in tg_om.items()) + ")")
                     if tg_om else "(lv20 기준 1개 4% / 2개 24% / 3개 44% / 4개 24% / 5+ 4%)"))
        if off_match > 0:
            from thick_ctl import off_match as _ofm
            tg_of = None
            if guard > 0:
                from thick_ctl import guard_off_target
                tg_of = guard_off_target(t_chart, masks, bpm)
            masks, n_ofm = _ofm(t_chart, masks, bpm, off_match, seed, jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0,
                                target=tg_of)
            print(f"폭타 뒷자리 두께를 lv20 분포로(--off-match {off_match:g}): {n_ofm}줄에 건반 하나씩 더함 "
                  + (("(울타리 --guard: 사람 곡별 10~90% 범위 밖만, 목표 " + " / ".join(f"{k}개 {v:.0%}" for k, v in tg_of.items()) + ")")
                     if tg_of else "(1개 36% / 2개 47% / 3개 15%, 큰동치-단타 교대 줄임)"))
        masks, n_pf = _pf(t_chart, masks, bpm, pillar_fix, seed, jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0) \
            if pillar_fix > 0 else (masks, 0)
        diff = None
        if mel is not None and len(frames) == len(masks):
            from thick_ctl import sound_diff
            diff = sound_diff(mel, frames)                # 앞 줄과 다른 소리일 때만 겹침을 옮김
        masks, n_mv, n_gr = eighth_chords(t_chart, masks, bpm, eighth_overlap, eighth_big, max_chord, seed,
                                          jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0, diff=diff,
                                          same_prob=eighth_same)
        print(f"구간별 동시치기: 16비트 8분 자리 큰 동시치기 확률로 덜기(--stream-cap {stream_cap:g}) {n_cap}줄, "
              f"8분 자리 기둥 풀기(--pillar-fix {pillar_fix:g}) {n_pf}줄, 8비트 겹침 옮기기(--eighth-overlap {eighth_overlap:g}) {n_mv}줄, "
              f"8비트 키우기(--eighth-big {eighth_big:g}) {n_gr}줄 | 4건반 이상: 16비트 {b16:.0%} -> {big_share(16, 4):.0%} "
              f"(사람 9%), 8비트 {b8:.0%} -> {big_share(8, 4):.0%} (사람 14%)")
    if beat_mode and chord_style > 0 and lane_fn is not None:
        from thick_ctl import _keys, adjacent_ratio, chord_style as _cs, style_persistence
        kp = lambda L: style_persistence(t_chart, L, bpm, key=lambda x: len(_keys(x)))
        p0, s0 = kp(masks), style_persistence(t_chart, masks, bpm)
        masks, n_cs = _cs(t_chart, masks, bpm, chord_style, seed)
        print(f"동시치기 성격 유지(--chord-style {chord_style:g}): {n_cs}줄 | 반 마디 건반 수 유지 {p0:.0%} -> {kp(masks):.0%} "
              f"(사람 약 50~85%), 건반 수+간격 {s0:.0%} -> {style_persistence(t_chart, masks, bpm):.0%} (사람 약 17~60%), "
              f"붙은 동시치기 {adjacent_ratio(masks):.0%}")
    if beat_mode and beat_contrast > 0 and lane_fn is not None:
        from thick_ctl import beat_contrast as _bc, beat_sizes
        e0, o0 = beat_sizes(t_chart, masks, bpm)
        masks, n_bc = _bc(t_chart, masks, bpm, beat_contrast, seed)
        e1, o1 = beat_sizes(t_chart, masks, bpm)
        print(f"8분 자리 두껍게/뒷자리 얇게(--beat-contrast {beat_contrast:g}): {n_bc}줄 | 16분 연타 평균 건반 8분 자리 {e0:.2f}/뒷자리 {o0:.2f}"
              f" -> {e1:.2f}/{o1:.2f} (사람 중앙값 2.69/1.61, Chronomia presea 2.40/1.44)")
    if beat_mode and denim_max > 0 and lane_fn is not None:
        from thick_ctl import denim_cap, denim_runs
        d0 = denim_runs(t_chart, masks, bpm)
        masks, n_dc = denim_cap(t_chart, masks, bpm, lo=min(4, denim_max), hi=denim_max, seed=seed)
        d1 = denim_runs(t_chart, masks, bpm)
        print(f"데님(2개 동시치기 16분 연속) 끊기(--denim-max {denim_max}): {n_dc}곳 | 가장 긴 데님 {max(d0, default=0)}줄 -> "
              f"{max(d1, default=0)}줄 (사람 곡별 최장 중앙값 7.5줄, presea 8줄)")
    if beat_mode and section_dyn > 0 and lane_fn is not None and mel is not None and len(frames) == len(masks):
        # 구간 강약 - 조용한 구간은 얇게, 큰 구간은 두껍게 (16, 8, 4, 2비트 모두)
        from thick_ctl import section_dyn as _sdy, section_loudness
        from scratch_model import mel_row_features
        row_loud = mel_row_features(mel, np.asarray(frames, dtype=np.int64))[:, :4].mean(axis=1)   # 노트 순간 대역 크기
        loud_ = section_loudness(t_chart, row_loud, bpm)
        masks, n_rm, n_ad, b0, b1 = _sdy(t_chart, masks, bpm, loud_, section_dyn, seed, max_chord=max_chord,
                                          jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0)
        if b0:
            print(f"구간 강약(--section-dyn {section_dyn:g}): 건반 {n_rm}개 빼고 {n_ad}개 더함 | 줄당 건반 조용한 1/3 구간 "
                  f"{b0[0]:.2f} -> {b1[0]:.2f}, 큰 1/3 구간 {b0[1]:.2f} -> {b1[1]:.2f} (사람 1.65 / 2.41)")
    if beat_mode and end_chord > 0 and lane_fn is not None:
        from sparse_ctl import stream_end_chords
        masks, n_ch, n_end = stream_end_chords(t_chart, masks, bpm, end_chord, seed, max_notes=max_chord,
                                                     jack_ms=100.0)   # --no-jack-ms 기본값과 같은 기준
        if n_end:
            print(f"16분 연타 끝 {n_end}곳 중 {n_ch}곳을 바로 앞 줄보다 {end_chord}개 이상 많은 동시치기로 (--stream-end-chord, 0이면 끔)")
    if beat_mode and alt_rows >= 2 and lane_fn is not None:
        from thick_ctl import alt_cap
        masks, n_ac = alt_cap(t_chart, masks, alt_rows, seed)
        if n_ac:
            print(f"후처리 뒤 묶음 교대(1357 246 1357, 13 24 13)가 {alt_rows}줄을 넘은 곳 {n_ac}곳을 다시 끊음 (--max-alt-rows)")
    if beat_mode and trip_match > 0 and lane_fn is not None:
        from thick_ctl import trip_match as _tm
        masks, n_t6, n_t24, r6, r24 = _tm(t_chart, masks, bpm, trip_match, seed, max_chord=max_chord,
                                          jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0)
        if r6 or r24:
            print(f"셋잇단 두께 맞추기(--trip-match {trip_match:g}): 6비트(4분 셋잇단) 연속 {r6}곳 {n_t6}줄 건반 수 맞춤, "
                  f"24비트(16분 셋잇단) 연속 {r24}곳 {n_t24}줄 건반 하나 뺌 (사람: 6비트 같은 개수 반복 74%, 24비트 단타 61%)")
        from thick_ctl import trip12_accent as _t12          # 12비트 3연 강-약-중
        masks, n_12, g12 = _t12(t_chart, masks, bpm, trip_match, seed)
        if g12:
            print(f"12비트 3연 강약(--trip-match {trip_match:g}): 박 안 셋잇단 3연 {g12}곳에서 {n_12}줄 덜어냄 "
                  f"(사람: 첫째 > 둘째 58~67%, 둘째가 더 큼 3%, 셋 다 같음 22~32%, 평균 강-약-중)")
    jk = no_jack_ms if no_jack_ms > 0 else 100.0
    if beat_mode and bar_repeat > 0 and lane_fn is not None and mel is not None and len(frames) == len(masks):
        from thick_ctl import bar_repeat as _br
        masks, n_bb, n_bc, n_bs = _br(t_chart, frames, masks, mel, bpm, bar_repeat, bar_shape, seed, jack_ms=jk,
                                      max_chord=max_chord)
        print(f"마디 반복(--bar-repeat {bar_repeat:g}, --bar-shape {bar_shape:g}): 앞 마디와 리듬·소리가 비슷한 {n_bb}마디에서 "
              f"{n_bc}줄 건반 수, {n_bs}줄 모양을 앞 마디처럼 (사람: 한 마디 전 건반 수 같음 62%, 모양 같음 13%)")
    if beat_mode and diag > 0 and lane_fn is not None:
        from thick_ctl import diag as _dg
        masks, n_dg = _dg(t_chart, masks, bpm, diag, seed, jack_ms=jk)
        print(f"대각선(--diag {diag:g}): {n_dg}줄을 두 줄 앞 모양의 한 칸 평행이동으로")
    if beat_mode and skel_swap > 0 and lane_fn is not None:      # 반복, 대각선이 건반 수를 바꾼 뒤에 뼈대를 맞춤
        from thick_ctl import skel_swap as _ss
        masks, n_ss = _ss(t_chart, masks, bpm, skel_swap, seed, jack_ms=jk)
        print(f"폭타 뼈대(--skel-swap {skel_swap:g}): 8분 자리가 뒷자리보다 얇은 {n_ss}쌍의 건반을 맞바꿈 (사람: 8분 자리가 더 두꺼움 69%)")
    if beat_mode and beat_pair > 0 and lane_fn is not None:
        from thick_ctl import beat_pair as _bp
        masks, n_bp, n_bpp = _bp(t_chart, masks, bpm, beat_pair, seed, jack_ms=jk, max_chord=max_chord)
        print(f"박 안 두 8분 자리 맞추기(--beat-pair {beat_pair:g}): {n_bpp}쌍 중 {n_bp}쌍의 둘째 줄 건반 수를 첫째와 같게 "
              f"(사람 277곡 중앙값: 같음 41% / 첫째가 큼 40% / 둘째가 큼 15%)")
    if beat_mode and edge4 > 0 and lane_fn is not None:
        from thick_ctl import edge4 as _e4
        masks, n_e4 = _e4(t_chart, masks, edge4, seed, jack_ms=jk)
        print(f"가운데 몰린 4건반 풀기(--edge4 {edge4:g}): {n_e4}줄 (사람보다 많을 때만, 1·7 없는 붙은 4건반 사람 4건반 중 7%)")
    if beat_mode and hand32 > 0 and lane_fn is not None:
        from thick_ctl import hand32 as _h32
        masks, n_h32 = _h32(t_chart, masks, bpm, hand32, seed, jack_ms=no_jack_ms if no_jack_ms > 0 else 100.0)
        if n_h32:
            print(f"32분 한 손 몰림 풀기(--hand32 {hand32:g}): {n_h32}줄을 반대 손으로 옮김")
    if beat_mode and fine_single > 0 and lane_fn is not None:     # hand32 뒤: 연속 모양을 hand32가 다시 흩지 않게
        from thick_ctl import fine_single as _fs
        masks, n_fs, n_fr = _fs(t_chart, masks, bpm, 0.9, fine_single, seed, jack_ms=jk)
        if n_fr:
            print(f"32분·24분 채운 줄 단타로(--fine-single {fine_single:g}): 앞뒤 같은 간격 줄 {n_fr}개 중 {n_fs}줄 (목표 단타 90%)")
    if beat_mode and fine_shape > 0 and lane_fn is not None:
        from thick_ctl import fine_shape as _fsh
        masks, n_fh, n_runs, used = _fsh(t_chart, masks, bpm, fine_shape, seed, jack_ms=jk)
        if n_runs:
            names = {"stair_line": "한 줄 계단", "stair_wrap": "감아 도는 계단", "stair_double": "겹계단",
                     "stair_bent": "꺾이는 계단", "alt": "좌우 교대", "zig": "지그재그", "narrow": "좁은 자리 반복",
                     "none": "그대로"}
            print(f"24·32분 연속 모양(--fine-shape {fine_shape:g}): {n_runs}곳 중 {n_fh}곳을 단타 모양으로 ("
                  + ", ".join(f"{names[k]} {v}" for k, v in used.items()) + ")")
    if beat_mode and fill_axis > 0 and lane_fn is not None:
        from thick_ctl import fill_axis as _fax
        masks, n_fx, n_fl = _fax(t_chart, masks, bpm, fill_axis, seed, jack_ms=jk)
        if n_fl:
            print(f"채운 단타 축 풀기(--fill-axis {fill_axis:g}): 32·24분 채운 단타 {n_fl}줄 중 {n_fx}줄을 다른 건반으로 "
                  f"(1박 안 다른 채운 단타와 같은 건반, 사람 반 박 안 같은 건반 6%)")
    if beat_mode and restart_shift > 0 and lane_fn is not None:
        from thick_ctl import restart_shift as _rsh
        masks, n_rs, n_sp = _rsh(t_chart, masks, bpm, restart_shift, seed, jack_ms=jk)
        if n_sp:
            print(f"쉼 뒤 폭타 재시작(--restart-shift {restart_shift:g}): 1/2~1박 쉬고 다시 시작하는 {n_sp}곳 중 {n_rs}곳의 "
                  f"시작 줄을 끝 줄과 안 겹치게 (사람: 겹침 49%)")
    end_allow = {}
    if beat_mode and eighth_trans > 0 and lane_fn is not None:
        # 8비트 동시치기가 이어질 때 바뀌는 방식을 사람처럼 (그대로 / 하나만 이동 / 평행이동 / 새로 / 여럿)
        from thick_ctl import eighth_trans as _etr, sound_diff as _sdf
        dif_ = _sdf(mel, frames) if (mel is not None and len(frames) == len(masks)) else None
        masks, n_et, n_ep, used_et = _etr(t_chart, masks, bpm, eighth_trans, seed, jack_ms=jk, diff=dif_)
        if n_ep:
            ko_ = {"same": "그대로", "one": "하나만 이동", "par": "평행이동", "new": "새로", "sev": "여럿 바뀜"}
            print(f"8비트 동치 이어짐(--eighth-trans {eighth_trans:g}): 같은 개수로 이어진 {n_ep}쌍 중 {n_et}줄 바꿈 ("
                  + ", ".join(f"{ko_[k]} {v}" for k, v in used_et.items())
                  + ") (사람: 그대로 5~10 / 하나만 이동 9~16 / 평행 16~21 / 새로 30~42 / 여럿 24~29%)")
    if beat_mode and beat_motif > 0 and lane_fn is not None:
        # 리듬이 같은 앞 박의 줄별 건반 수를 따라가게 (사람 1박 전 수패턴 같음 27~28%, 생성 8%)
        from thick_ctl import beat_motif as _bmo
        masks, n_bm, n_bx, n_bp, (rb0, rb1) = _bmo(t_chart, masks, bpm, beat_motif, seed, max_chord=max_chord, jack_ms=jk)
        if n_bp:
            print(f"박 반복(--beat-motif {beat_motif:g}): 리듬이 같은 앞 박 {n_bp}쌍 중 {n_bm}박 건반 수 맞춤, {n_bx}박 그대로 옮김 | "
                  f"1박 전과 줄별 건반 수 같음 {rb0:.0%} -> {rb1:.0%} (사람 27~32%)")
    if beat_mode and eighth_shape > 0 and lane_fn is not None:
        # 8분 자리 줄을 반복, 교대 구조로
        from thick_ctl import eighth_shape as _esp
        r_s0 = None
        u_s = {}
        for ps_ in range(2):
            masks, u_, n_tr, r_a, r_b = _esp(t_chart, masks, bpm, eighth_shape, seed + ps_, jack_ms=jk)
            r_s0 = r_a if r_s0 is None else r_s0
            for k_, v_ in u_.items():
                u_s[k_] = u_s.get(k_, 0) + v_
        if n_tr:
            print(f"8비트 모양 반복(--eighth-shape {eighth_shape:g}): 8분 자리 {n_tr}줄 중 두 줄 전 그대로 {u_s.get('e2', 0)}, "
                  f"두 줄 전 모양 옮김 {u_s.get('s2', 0)}, 앞 줄 모양 옮김 {u_s.get('s1', 0)} | 두 줄 전과 같음 {r_s0[0]:.0%} -> {r_b[0]:.0%}, "
                  f"같은 모양 앞 {r_s0[2]:.0%} -> {r_b[2]:.0%} / 두 줄 전 {r_s0[1]:.0%} -> {r_b[1]:.0%} (사람 12 / 15~21 / 20~28%)")
    if beat_mode and pair_even > 0 and lane_fn is not None:
        # 바쁜 마디의 '8분 자리 줄 + 바로 뒤 16분 줄' 건반 합을 마디 안에서 일정하게 (사람 폭타는 5, 6처럼 이어짐)
        from thick_ctl import pair_even as _pev
        tg_pe = None
        if pair_avg > 0 and mel is not None:
            # 4마디 묶음 저, 중음(lowmid) 크기 순위로 목표 쌍 합 (평균 = --pair-avg, 폭 = 0.8 x --pair-spread)
            from thick_ctl import block_lowmid, pair_targets
            n_meas_pe = int(np.max(np.asarray(t_chart)) / (240.0 / bpm)) + 1
            bl_pe = block_lowmid(mel, bpm, pad, n_meas_pe)
            pa_pe = pair_avg
            if guard > 0 and abs(pair_avg - 4.0) < 1e-6:
                # 울타리: 쌍 합 평균을 4로 맞추지 않고 지금 곡 그대로, 사람 곡별 10~90%(3.49~5.29) 밖이면 그 끝으로
                from thick_ctl import pair_natural, PAIR_LO, PAIR_HI
                nat_pe = pair_natural(t_chart, masks, bpm)
                if np.isfinite(nat_pe):
                    pa_pe = float(np.clip(nat_pe, PAIR_LO, PAIR_HI))
                    print(f"울타리(--guard): 쌍 합 평균 지금 {nat_pe:.2f} -> 목표 {pa_pe:.2f} (사람 곡별 10~90% {PAIR_LO}~{PAIR_HI}, 범위 안이면 그대로)")
            tg_pe = pair_targets(t_chart, masks, bpm, bl_pe, pa_pe, 2.0 * pair_dev, seed=seed)
        masks, n_pe, (pe0, pe1) = _pev(t_chart, masks, bpm, pair_even, seed, max_chord=max_chord, jack_ms=jk,
                                    pair_min=pair_min, targets=tg_pe, keep_var=pair_keep > 0)
        pair_tg = dict(tg_pe) if tg_pe else {}
        if tg_pe:
            v_ = np.array(list(tg_pe.values()), float)
            print(f"폭타 쌍 합 평균(--pair-avg {pa_pe:g}, 편차 ±{pair_dev:g}): 바쁜 마디 {len(v_)}개를 4마디 묶음 "
                  f"저·중음 크기 순위대로 목표 {v_.min():.0f}~{v_.max():.0f}, 평균 {v_.mean():.2f} "
                  f"(사람 lv16~20 곡 평균 2.65~4.63, 곡 안 폭 약 2.6, 묶음 안 표준편차 0.4)")
        if n_pe:
            print(f"폭타 쌍 합 고르게(--pair-even {pair_even:g}): {n_pe}줄 | 바쁜 마디 안 '8분+16분' 건반 합 표준편차 "
                  f"{pe0:.2f} -> {pe1:.2f} (사람 lv20 0.71, Air -GOD- 0.60)"
                  + (f" | 목표 최소 {pair_min} (마디 중앙값 +1까지, --pair-min)" if pair_min > 0 else ""))
    if beat_mode and end_fat > 0 and lane_fn is not None:
        # 폭타 끝 줄을 앞 1박 8분 자리 최대 이상으로, 끝 줄만 상한 + end_over 허용
        from thick_ctl import stream_end_fat as _sef
        masks, n_ef, n_en, end_allow = _sef(t_chart, masks, bpm, end_fat, seed, max_chord=max_chord, over=end_over,
                                            jack_ms=jk)
        if n_en:
            print(f"폭타 끝 묵직하게(--end-fat {end_fat:g}, --end-over {end_over}): 끝 {n_en}곳 중 {n_ef}곳을 앞 1박 8분 자리 "
                  f"최대 이상으로 (끝 줄만 상한 {min(7, max_chord + end_over)}까지, 사람: 끝 >= 앞 최대 66%, > 21%)")
    if beat_mode and end_shape > 0 and lane_fn is not None:
        # 폭타 끝 줄 건반 수는 소리 크기 순위로 사람 분포에, 모양은 이동/비슷/벌어짐/모임/그대로를 사람 비율로
        from thick_ctl import end_shape as _esh, SHAPE_KO
        lo_e = None
        if mel is not None and len(frames) == len(masks):
            from scratch_model import mel_row_features
            lo_e = mel_row_features(mel, np.asarray(frames, dtype=np.int64))[:, :4].mean(axis=1)
        masks, n_es, n_ee, al_e, used_e, (k0_e, k1_e) = _esh(t_chart, masks, bpm, lo_e, end_shape, seed,
                                                             max_chord=max_chord, over=end_over, jack_ms=jk)
        for k_, v_ in al_e.items():
            end_allow[k_] = max(end_allow.get(k_, 0), v_)
        if n_ee:
            tot_e = max(sum(used_e.values()), 1)
            print(f"폭타 끝 모양(--end-shape {end_shape:g}): 끝 {n_ee}곳 중 {n_es}줄 바꿈 | 평균 건반 {k0_e:.2f} -> {k1_e:.2f} "
                  f"(센 소리는 크게, 약한 소리는 가볍게; 사람 1박 이하 쉼 1~2개 36% / 4개+ 30%) | 모양 "
                  + ", ".join(f"{SHAPE_KO[k_]} {v_ / tot_e:.0%}" for k_, v_ in sorted(used_e.items(), key=lambda z: -z[1]))
                  + " (사람 이동 33 / 비슷 29 / 벌어짐 23 / 모임 10 / 그대로 5%)")
    if beat_mode and stream_kpr > 0 and lane_fn is not None:
        # 남은 16분 폭타 마디 줄당 건반을 사람(2.28)까지 (소리 큰 마디일수록 두껍게)
        from thick_ctl import stream_thicken
        masks, n_th, (k0_th, k1_th) = stream_thicken(t_chart, masks, bpm, float(stream_kpr), max_chord=max_chord,
                                                      exclude=cb_meas, mel=mel, pad=pad, seed=seed)
        if n_th:
            print(f"폭타 두께 맞추기(--stream-kpr {stream_kpr:g}): 16분 폭타 마디에 건반 {n_th}개 더함 | 줄당 건반 {k0_th:.2f} -> {k1_th:.2f} "
                  "(사람 폭타 마디 중앙값 2.28, 두꺼운 마디 34% / 얇은 마디 12%, 소리 큰 마디일수록 두껍게)")
    if beat_mode and cb_meas and lane_fn is not None:
        # 8비트 동시치기로 바꾼 마디 건반 수 = 주변 폭타 마디의 0.76배 (사람), 악절째 바꾼 구간은 0.5배
        from thick_ctl import chord_fill
        from thick_ctl import section_ratio
        from sound16_ctl import rest_ratio
        # --rest-chord: 소리 큰 곳을 동시치기로 메울 때는 그에 맞는 밀도로.
        # 소리 큰 곳(곡 순위 0.6+)의 폭타 끊기, 폭타 구간 바꾸기 8비트 동시치기는 줄마다 주변 폭타 쌍 합 x rest_chord, 그 아래는 예전대로
        rmap_ = {m_: rest_ratio(m_rank.get(m_, 0.0), rest_chord) for m_ in cb_meas if m_ not in sec_meas}
        for m_ in sec_meas:
            k_ = sec_kind.get(m_, "C")
            rmap_[m_] = (rest_ratio(sec_rank.get(m_, 0.5), rest_chord, base=section_ratio(k_, sec_rank.get(m_, 0.5)))
                         if k_ == "C" else section_ratio(k_, sec_rank.get(m_, 0.5)))
        masks, n_cf, rep_cf = chord_fill(t_chart, masks, bpm, [m_ for m_ in cb_meas if m_ not in lad_plan], seed,
                                         max_chord=max_chord, ratio_map=rmap_)
        if rep_cf:
            print(f"폭타 끊은 마디 동시치기 채우기: 건반 {n_cf}개 조절 | 마디별 건반 수 "
                  + ", ".join(f"{m_ + 1:03d} {a_}->{b_}(목표 {g_})" for m_, a_, b_, g_ in rep_cf[:8])
                  + (" …" if len(rep_cf) > 8 else "")
                  + f" (1마디 끊기는 주변 폭타 마디의 0.76배, 소리 큰 곳(순위 0.6+)의 8비트 동시치기는 {rest_chord:g}배 = 줄마다 주변 폭타 쌍 합"
                    "(--rest-chord, 사용자 기준 / 사람 소리 큰 곳 폭타 옆 동시치기 쌍 합 4.5~5.5일 때 줄당 중앙 4.1, 상위 25% 4.9), "
                    "악절째 바꾼 구간은 종류별 0.16~0.88배)")
    if beat_mode and lad_plan and lane_fn is not None:
        # 사다리 폭타 - 16분 줄은 그대로, 건반 수만 단계에 맞춤
        from thick_ctl import ladder_plan, ladder_fill, meas_rungs, LADDER_KO
        lad_rung = ladder_plan(lad_plan, t_chart, masks, bpm)
        rg0_ = meas_rungs(t_chart, masks, bpm)
        masks, n_lf, rep_lf = ladder_fill(t_chart, masks, bpm, lad_rung, seed, max_chord=max_chord, jack_ms=jk)
        if rep_lf:
            from collections import Counter as _Ctr
            c_lf = _Ctr(LADDER_KO[v_] for v_ in lad_rung.values())
            print(f"사다리 폭타(--ladder {ladder:g}): {len(rep_lf)}마디 건반 {n_lf}개 조절 (16분 줄은 그대로) | "
                  + ", ".join(f"{k_} {v_}마디" for k_, v_ in sorted(c_lf.items(), key=lambda z: LADDER_KO.index(z[0]))) + " | "
                  + ", ".join(f"{m_ + 1:03d} {LADDER_KO[rg0_[m_]] if m_ in rg0_ else '-'}->{LADDER_KO[lv_]}(줄당 {a_:.1f}->{b_:.1f})"
                              for m_, a_, b_, lv_ in rep_lf[:10])
                  + (" …" if len(rep_lf) > 10 else "")
                  + " (사람 277개 폭타 마디 단계 소리 하위 20%: 1+1 30 / 2+1 27 / 2+2 14%, 20~40%: 11 / 24 / 12%)")
    if beat_mode and accent_t and lane_fn is not None:
        # 치고 쉬기 큰 소리 줄은 3개 이상 동시치기로
        from thick_ctl import accent_chord
        hit_ix_ = [int(np.argmin(np.abs(np.asarray(t_chart) - x_))) for x_ in accent_t]
        masks, n_ac = accent_chord(masks, hit_ix_, max_chord=max_chord, seed=seed, t_chart=t_chart, jack_ms=jk)
        if n_ac:
            print(f"치고 쉬기 큰 소리 {len(hit_ix_)}줄을 동시치기로: 건반 {n_ac}개 더함 (최소 3개)")
    if (beat_mode and stream_lift > 0 and lane_fn is not None and mel is not None and len(frames) == len(masks)
            and len(masks) > 64):
        # 얇은, 중간 폭타를 같은 BPM대 사람 두께 분포까지 올림 (가장 진한 폭타, 최고 밀도는 그대로)
        try:
            from thick_ctl import stream_lift as _sl
            from scratch_model import mel_row_features
            loud_sl = mel_row_features(mel, np.asarray(frames, dtype=np.int64))[:, :4].mean(axis=1)
            keep_sl = [int(np.argmin(np.abs(np.asarray(t_chart) - x_))) for x_ in accent_t] if accent_t else []
            if lad_rung:                    # 사다리 폭타 마디는 다시 두껍게 하지 않음
                ms_sl = np.floor(np.asarray(t_chart) / (240.0 / bpm) + 1e-6).astype(np.int64)
                keep_sl = keep_sl + [int(i_) for i_ in np.flatnonzero(np.isin(ms_sl, list(lad_rung)))]
            sc_sl = min(1.0, pair_avg / 4.0) if (pair_even > 0 and pair_avg > 0) else 1.0
            masks, n_sl, rep_sl = _sl(t_chart, masks, bpm, loud_sl, stream_lift, sc_sl, seed=seed, max_chord=max_chord,
                                      jack_ms=jk, keep_idx=keep_sl, guard=guard > 0)
            if rep_sl:
                b_, a_, h_ = rep_sl["before"], rep_sl["after"], rep_sl["human"]
                print(f"폭타 두께 사람 분포로(--stream-lift {stream_lift:g}): 폭타 마디 {rep_sl['meas']}개에 건반 {n_sl}개 더함 | "
                      f"줄당 노트 얇은(하위 10%)/중간/두꺼운(상위 10%) {b_[0]:.2f}/{b_[1]:.2f}/{b_[2]:.2f} -> {a_[0]:.2f}/{a_[1]:.2f}/{a_[2]:.2f} "
                      f"(사람 lv15~22 같은 BPM대 {'하위 10% 곡 ' if guard > 0 else ''}{h_[0]:.2f}/{h_[1]:.2f}/{h_[2]:.2f}"
                      + (f", 쌍 합 평균 {pair_avg:g}이라 x{sc_sl:.2f}" if sc_sl < 1 else "")
                      + ", 가장 두꺼운 폭타·최고 밀도는 그대로, 빼지는 않음)")
        except Exception as e_:  # noqa: BLE001
            print(f"[참고] 폭타 두께 맞추기를 건너뜀: {type(e_).__name__}: {e_}")
    if beat_mode and four_one > 0 and lane_fn is not None and len(masks) > 64:
        # --four-one: 4 단계 폭타 중 4+1을 사람 비율까지 - 4+2 마디 중 소리 작은 것부터
        # 16분 뒷자리 줄을 1개씩으로 (폭타 두께 사람 분포 뒤: 그 규칙이 뒷자리부터 더해서 4+1이 거의 다 4+2가 됨).
        # 폭타 쌍 합 평균(--pair-avg)이 그 마디에 정한 목표 쌍 합이 5 이하일 때만 (목표 6+인 소리 큰 묶음의 4+2는 그대로)
        try:
            from thick_ctl import four_one as _f41
            lo_41 = None
            if mel is not None and len(frames) == len(masks):
                from scratch_model import mel_row_features
                lo_41 = mel_row_features(mel, np.asarray(frames, dtype=np.int64))[:, :4].mean(axis=1)
            keep_41 = [int(np.argmin(np.abs(np.asarray(t_chart) - x_))) for x_ in accent_t] if accent_t else []
            masks, n_41, r_41 = _f41(t_chart, masks, bpm, lo_41, four_one, seed=seed, keep_idx=keep_41, skip=set(lad_rung),
                                     targets=pair_tg)
            if r_41.get("blocked"):
                print(f"4+1 섞기(--four-one): 폭타 쌍 합 평균(--pair-avg)이 정한 목표가 6+인 4+2 {r_41['blocked']}마디는 그대로 둠 "
                      "(소리 큰 묶음, 쌍 합 평균을 올리면 늘어남)")
            if r_41.get("meas"):
                b_, a_ = r_41["before"], r_41["after"]
                print(f"4+1 섞기(--four-one {four_one:g}): 4 단계 폭타 4+1 / 4+2 / 4+3+ {b_[0]} / {b_[1]} / {b_[2]} -> {a_[0]} / {a_[1]} / {a_[2]}마디, "
                      f"건반 {n_41}개 뺌 (4+2 중 소리 작은 {len(r_41['meas'])}마디: "
                      + ", ".join(f"{m_ + 1:03d}" for m_ in sorted(r_41["meas"])[:12]) + (" …" if len(r_41["meas"]) > 12 else "")
                      + ") (사람 277: 4+1 30 / 4+2 55 / 4+3+ 15%, 2마디 묶음 밀도 순위 4+1 0.77 = 3+2와 같음, 4+2 0.82)")
        except Exception as e_:  # noqa: BLE001
            print(f"[참고] 4+1 섞기를 건너뜀: {type(e_).__name__}: {e_}")
    if (beat_mode and dens_shape > 0 and lane_fn is not None and mel is not None and len(frames) == len(masks)
            and len(masks) > 64):
        # 구간 밀도 분포를 사람처럼 - 조용한 구간은 더 가볍게, 최고 밀도는 상한까지, 최고가 너무 길게 이어지면 끊음
        try:
            from thick_ctl import dens_shape as _dsh, stream_rows as _srw
            from trem_ctl import trem_rows as _trw
            from scratch_model import mel_row_features
            fr_ = np.asarray(frames, dtype=np.int64)
            loud_ds = mel_row_features(mel, fr_)[:, :4].mean(axis=1)
            prob_ds = np.max([probs[np.clip(fr_ + d_, 0, len(probs) - 1)] for d_ in range(-3, 4)], axis=0)
            keep_ds = [int(np.argmin(np.abs(np.asarray(t_chart) - x_))) for x_ in accent_t] if accent_t else []
            n0_ds = sum(bin(int(m) & 0xFF).count("1") for m in masks)
            trw_ds = _trw(t_chart, bpm, trem_slots)
            prot_ds = _srw(t_chart, bpm) | trw_ds
            mk_ds = None
            if mel_keep > 0:
                # 멜로디, 보컬, 베이스 어택이 센 줄(곡 안 위쪽 40%)은 구간 밀도 분포가 안 뺌
                from melkeep_ctl import protect_rows, row_melodic
                sc_ds, src_ds = row_melodic(fr_, None, _stems_mk())
                if sc_ds is not None:
                    mk_ds = protect_rows(sc_ds, mel_keep)
                    prot_ds = prot_ds | mk_ds
                else:
                    print(f"멜로디 지키기(--mel-keep {mel_keep:g}): 구간 밀도 분포에서 멜로디 줄을 보호하지 않음 - 저장해 둔 분리 소리(<이름>_stems.npz)가 없음 "
                          "(악기 따라가기 --follow를 한 번 쓰면 생김, 약 6분)")
            dl_ds, pl_ds = None, None
            n_ms_ds = int(np.max(np.floor(np.asarray(t_chart) / (240.0 / bpm) + 1e-9))) + 2
            if dens_dev > 0 and mel is not None:
                # --dens-dev: 분포 대신 편차 기준 - 곡 큰 소리보다 얼마나 작은가로 목표
                from thick_ctl import meas_dev_loud as _mdl
                dl_ds = _mdl(mel, bpm, pad, n_ms_ds, w=0.3)     # 사람 편차 곡선과 같은 기준 (0.7 멜 전체 + 0.3 저음)
            if dens_pulse > 0:
                # --dens-pulse: 8분 박 소리(드럼, 베이스, 악기)가 있는 마디는 8분 뒷자리 줄을 통째로 안 뺌
                st_p = _stems_mk()
                if st_p is not None:
                    from sound16_ctl import eighth_pulse as _e8p
                    pl_ds = _e8p(st_p, bpm, pad, n_ms_ds)
            lr_ds = None
            if dens_flat > 0:
                # --dens-flat: 곡 소리 폭에 맞는 사람 분포 - 편차 적은 곡은 덜 비우고 줄보다 건반부터
                from thick_ctl import loud_range as _lrng
                lr_ds = _lrng(mel, bpm, pad, int(np.max(np.floor(np.asarray(t_chart) / (240.0 / bpm) + 1e-9))) + 2)
            kp_ds, masks, rep_ds = _dsh(t_chart, masks, bpm, loud_ds, prob_ds, dens_shape, dens_cap, keep_idx=keep_ds, seed=seed,
                                        protect=prot_ds, guard=guard > 0,
                                        brk_keys_ok=prot_ds & ~trw_ds,   # 최고 근처 긴 연속 끊는 묶음은 폭타, 멜로디 줄도 건반만 덜기
                                        loud_range=lr_ds, dev_loud=dl_ds, pulse_meas=pl_ds)
            if mk_ds is not None:
                print(f"멜로디 지키기(--mel-keep {mel_keep:g}): 구간 밀도 분포가 안 뺄 멜로디 줄 {int(mk_ds.sum())}개 "
                      f"(곡 안 멜로디 점수 위쪽 {0.4 * min(mel_keep, 2.5):.0%}, 점수 = "
                      + "분리 소리 보컬·악기·베이스 어택 / 드럼 어택) - 빼야 할 양은 드럼 엇박 줄·큰 동시치기 건반에서 먼저")
            if not kp_ds.all():
                if nlog is not None:
                    nlog.drop(frames, np.asarray(frames)[kp_ds], "구간 밀도 분포(--dens-shape)")
                frames = np.asarray(frames)[kp_ds]
                t_chart = np.asarray(t_chart)[kp_ds]
                masks = [m for m, k_ in zip(masks, kp_ds) if k_]
                times = t_chart - pad
            n1_ds = sum(bin(int(m) & 0xFF).count("1") for m in masks)
            if "P" in rep_ds:
                print(f"구간 밀도 분포(--dens-shape {dens_shape:g}, --dens-cap {dens_cap:g}): 폭타 아닌 곳만, 2마디 묶음 {rep_ds['blocks']}개에서 "
                      f"줄 {rep_ds['rows']}개(박 단위 16분·8분 뒷자리) + 건반 {rep_ds['keys']}개 뺌, 노트 {n0_ds} -> {n1_ds} | "
                      f"최고(상위 5%) {rep_ds['P']:.1f} -> {rep_ds['P_after']:.1f}/초, 가장 가벼운 10% 묶음 최고 대비 "
                      f"{rep_ds['q10_before']:.2f} -> {rep_ds['q10_after']:.2f}, 최고 근처 긴 연속 끊음 {rep_ds['breaks']}곳"
                      f"({rep_ds.get('max_run', 16)}마디 넘게 이어지면 - patch105 사람 BPM대 가장 긴 유지 중앙값) "
                      f"(사람 lv15~22: 최고 31/초(상위 25% 33), 2마디 묶음 하위 10% 0.22 / 중앙 0.72 / 상위 25% 0.89, 최고 근처 최장 11마디)"
                      + (f" | 소리 폭 {rep_ds['loud_range']:.3f}(--dens-flat): 사람 같은 소리 폭 곡 분포로 하위 10% 묶음 목표 "
                         f"{float(np.interp(0.1, np.linspace(0, 1, 21), __import__('thick_ctl').dens_curve(rep_ds['loud_range']))):.2f}, "
                         f"줄 빼기 비중 x{rep_ds['flat_k']:.2f} (사람 소리 폭 0.065 미만 곡 하위 10% 0.38 / 0.12 이상 0.18)"
                         if "loud_range" in rep_ds else "")
                      + (f" | 편차 기준(--dens-dev): 묶음 목표 = 사람 같은 편차(곡 큰 소리 대비, 하위 10% {rep_ds['dev_q10']:+.3f}) 밀도 중앙값 - "
                         f"순위 기준 목표보다 낮아진 묶음 {rep_ds['dev_lo']}개 / 높아진 {rep_ds['dev_hi']}개 (큰 소리 근처는 순위 기준 목표보다 안 낮춤)"
                         if "dev_q10" in rep_ds else "")
                      + (f" | 8분 박 지키기(--dens-pulse): 분리 소리 8분 박 마디 {len(pl_ds)}개 - 8분 뒷자리 줄 {rep_ds['pulse_rows']}개는 안 빼고 건반만"
                         if pl_ds is not None else ""))
        except Exception as e_:  # noqa: BLE001
            print(f"[참고] 구간 밀도 분포를 건너뜀: {type(e_).__name__}: {e_}")
    if beat_mode and chord32 > 0 and lane_fn is not None and len(masks) > 0 and len(frames) == len(masks):
        # --chord32: 모델이 약하게 들은(확률 < 값) 32분 뒷자리 줄을 옆 16분 줄 동시치기로.
        # 잘 안 들리는 소리에 붙은 32분 노트만 어색해서 합침. 또렷한 롤과 떨림 줄은 그대로, 줄만 합치고 건반은 그대로라 세기와 완급은 남음.
        # 자리 모델 앞이라 자리는 모델이 다시 고름
        try:
            from thick_ctl import chord32 as _c32
            from trem_ctl import trem_rows as _trw3
            fr_c0 = np.asarray(frames, dtype=np.int64)
            rp_c = np.max([probs[np.clip(fr_c0 + d_, 0, len(probs) - 1)] for d_ in range(-3, 4)], axis=0)
            kp_c, masks_c, t_c, r_c = _c32(t_chart, masks, bpm, max_chord=max_chord,
                                           protect=_trw3(t_chart, bpm, trem_slots), strength=chord32, row_prob=rp_c)
            if r_c["merged"] or r_c["snapped"]:
                if nlog is not None and not kp_c.all():
                    nlog.drop(frames, np.asarray(frames)[kp_c], "32비트 동시치기로(--chord32)")
                fr_c = np.asarray(frames)[kp_c]
                tc_c = np.asarray(t_c)[kp_c]
                mk_c = [m_ for m_, k_ in zip(masks_c, kp_c) if k_]
                o_c = np.argsort(tc_c, kind="stable")
                frames = fr_c[o_c]
                t_chart = tc_c[o_c]
                masks = [mk_c[i_] for i_ in o_c]
                times = t_chart - pad
            ms_c = sorted(r_c["meas"])
            print(f"약한 32비트 동시치기로(--chord32 {chord32:g}): 32분 뒷자리 줄 {r_c['n32']}개 중 모델 확률 {chord32:g} 미만 "
                  f"{r_c['merged'] + r_c['snapped']}개 - {r_c['merged']}개는 옆 16분 줄(8분 자리 우선) 동시치기로 합치고 {r_c['snapped']}개는 16분 칸으로 옮김, "
                  f"또렷한 {r_c['strong']}개는 그대로"
                  + (f", 떨림 줄 {r_c['kept']}개는 그대로" if r_c["kept"] else "")
                  + (f", 동시치기 상한으로 건반 {r_c['lost']}개 뺌" if r_c["lost"] else "")
                  + (f" | 마디 {', '.join(f'{m_ + 1:03d}' for m_ in ms_c[:14])}{' …' if len(ms_c) > 14 else ''}" if ms_c else "")
                  + " (사람 277곡 32분 줄 1000줄당 중앙 150BPM 미만 19.7 / 150~180 9.8 / 180+ 7.7, 0인 곡 20~33%)")
        except Exception as e_:  # noqa: BLE001
            print(f"[참고] 32비트 동시치기를 건너뜀: {type(e_).__name__}: {e_}")
    if beat_mode and lane_fn is not None and abs(key_scale - 1.0) > 1e-6 and len(masks) > 0:
        # 건반 배율 - 줄은 그대로, 동시치기 크기만 (자리 모델 앞이라 자리는 모델이 다시 고름)
        from thick_ctl import key_scale as _ks
        keep_ks = [int(np.argmin(np.abs(np.asarray(t_chart) - x_))) for x_ in accent_t] if accent_t else []
        k0_ = sum(bin(int(m) & 0xFE).count("1") for m in masks)
        masks, k1_ = _ks(masks, key_scale, seed=seed, t_chart=t_chart, bpm=bpm, jack_ms=jk, max_chord=max_chord,
                         keep_idx=keep_ks)
        n_r_ = sum(1 for m in masks if int(m) & 0xFE)
        print(f"건반 배율(--key-scale {key_scale:g}): 건반 {k0_} -> {k1_}개 (줄 {len(masks)}개 그대로, "
              f"줄당 {k0_ / max(n_r_, 1):.2f} -> {k1_ / max(n_r_, 1):.2f}, 단타는 그대로 두고 동시치기 크기만)")
    # ---- STEP 6 자리 모델과 배치 규칙: 건반 수는 그대로 두고 자리만 다시 정함 ----
    if beat_mode and placer > 0 and placer_ckpt and lane_fn is not None and mel is not None and len(frames) == len(masks):
        # 자리 모델 - 건반 수, 스크래치는 지금까지 정한 그대로, 앞 줄들을 보고 건반 자리만 다시 뽑음
        try:
            import placer as _pl
            from lane_v2 import rhythm_codes_from_times as _rcf
            pm_ = _pl.load(placer_ckpt, "cpu")
            codes_ = _rcf(np.asarray(t_chart), bpm, 0)
            acc_ = _pl.accent(mel, frames)
            rr_ = np.random.default_rng(seed + 787)
            keep_ = rr_.random(len(masks)) >= placer if placer < 1 else None
            old_ = list(masks)
            masks = [int(m) for m in _pl.place_masks(pm_, np.asarray(frames), np.asarray(masks), codes_, acc_,
                                                      jack_ms=jk, temperature=placer_temp, seed=seed, keep=keep_,
                                                      rep_penalty=placer_rep, rep_first=0.28 * placer_rep,
                                                      beat_frames=60.0 / bpm * FPS, beat_pen=placer_beat,
                                                      row_seed=row_seed > 0)]
            ch_ = sum(1 for a, b in zip(old_, masks) if a != b)
            fs0 = _pl.flow_stats(np.asarray(frames), _pl.keys_of(np.asarray(old_)))
            fs1 = _pl.flow_stats(np.asarray(frames), _pl.keys_of(np.asarray(masks)))
            print(f"자리 모델(--placer {placer:g}, 온도 {placer_temp:g}, {Path(placer_ckpt).name}): {len(masks)}줄 중 {ch_}줄 자리 바꿈 "
                  f"(건반 수 그대로) | 두 줄 전 그대로 {fs0['두 줄 전 그대로']:.0%} -> {fs1['두 줄 전 그대로']:.0%}, "
                  f"두 줄 전 같은 모양 {fs0['두 줄 전 같은 모양']:.0%} -> {fs1['두 줄 전 같은 모양']:.0%}")
        except Exception as e:  # noqa: BLE001
            print(f"[참고] 자리 모델을 못 써서 건너뛰어요: {type(e).__name__}: {e}")
    if beat_mode and edge_axis > 0 and lane_fn is not None:
        # 1, 7번 축(8분 자리 3줄 연속)이 축 전체의 40%를 넘으면 가운데 줄 1, 7을 가까운 안쪽(2, 3 / 6, 5)으로
        from thick_ctl import edge_axis as _eax
        masks, n_ea, (ea0, ea1) = _eax(t_chart, masks, bpm, edge_axis, seed, jack_ms=jk)
        if n_ea:
            print(f"1·7번 축 풀기(--edge-axis {edge_axis:g}): 건반 {n_ea}개를 안쪽으로 | 축 중 1·7번 {ea0:.0%} -> {ea1:.0%} "
                  f"(사람 277 39% / lv20 37%)")
    if beat_mode and quarter_spread > 0 and lane_fn is not None:
        # 4비트 세 줄이 한쪽 손 안에서 가운데 -> 끝으로 밀려가는 모양(5 4 47)을 반대쪽 손으로 (사람 1.3%)
        from thick_ctl import quarter_drift as _qd
        masks, n_qd, (qd0, qd1) = _qd(t_chart, masks, bpm, quarter_spread, seed, jack_ms=jk)
        if n_qd:
            print(f"4비트 퍼짐(--quarter-spread {quarter_spread:g}): 한쪽 손 안에서 끝으로 밀려가던 4비트 {n_qd}줄을 반대쪽으로 | "
                  f"4비트 세 줄 중 이런 모양 {qd0:.1%} -> {qd1:.1%} (사람 277곡 1.4%)")
    if beat_mode and hand_balance > 0 and lane_fn is not None:
        # 폭타 밖 한 손 줄 좌우 쏠림 (사람 오른손만 27% / 왼손만 19% = 1.42배)
        from thick_ctl import hand_balance as _hb
        masks, n_hb, (hb0, hb1) = _hb(t_chart, masks, bpm, hand_balance, seed, jack_ms=jk)
        if n_hb:
            print(f"폭타 밖 손 쏠림 풀기(--hand-balance {hand_balance:g}): {n_hb}줄 좌우 반전 | 오른손만 줄 / 왼손만 줄 "
                  f"{hb0:.2f} -> {hb1:.2f}배 (사람 277곡 1.42배, 4번은 오른손)")
    if beat_mode and trill_break > 0 and lane_fn is not None:
        # 한 손 붙은 두 건반 트릴(67 67, 12 12 ...) 4줄 이상 끊기 (사람 1000줄당 약 1곳, 곡별 중앙값 0)
        from thick_ctl import trill_break as _tb2
        masks, n_tr, (tr0, tr1) = _tb2(t_chart, masks, bpm, trill_break, seed, jack_ms=jk)
        if n_tr:
            print(f"한 손 트릴 끊기(--trill-break {trill_break:g}): {n_tr}줄 | 붙은 두 건반 트릴 4줄+ 1000줄당 {tr0:.1f} -> {tr1:.1f}곳 "
                  f"(사람 277곡 67 0.9 / 12 0.9 / 56 1.7 / 23 0.9, 곡별 중앙값 0)")
    if beat_mode and shape_run > 0 and lane_fn is not None:
        # 폭타 안 이웃 16분 동시치기가 같은 모양(평행이동)으로 이어지는 것을 사람 범위로 (shape_ctl.py)
        from shape_ctl import shape_run_fix as _srf, SAME_P75, CHAIN_P50
        masks, r_sr = _srf(t_chart, masks, bpm, shape_run, seed, jack_ms=jk)
        if r_sr["chain_rows"] or r_sr["pair_rows"]:
            print(f"같은 모양 이어짐 끊기(--shape-run {shape_run:g}): {r_sr['chain_rows'] + r_sr['pair_rows']}줄 모양 바꿈 (건반 수 그대로) | "
                  f"폭타 안 이웃 16분 동시치기가 앞 줄과 같은 모양 {r_sr['before'][0]:.0%} -> {r_sr['after'][0]:.0%}, 같은 모양 3줄+ 사슬 "
                  f"폭타 1000줄당 {r_sr['before'][1]:.1f} -> {r_sr['after'][1]:.1f} (사람 158곡: 같은 모양 중앙값 11% / 75% {SAME_P75:.0%}, "
                  f"사슬 중앙값 {CHAIN_P50:g} / 75% 18.4)")
    if beat_mode and fine_density > 0 and lane_fn is not None:
        # 폭타 속 32, 24분이 낀 박 밀도 (사람: 16분만 있는 박의 1.13배, 32, 24분 줄 단타 84%)
        from thick_ctl import fine_density as _fd
        masks, n_fd, r_fd = _fd(t_chart, masks, bpm, fine_density, seed, jack_ms=jk)
        if n_fd:
            print(f"폭타 속 32·24비트 밀도(--fine-density {fine_density:g}): 건반 {n_fd}개 뺌 | 32·24분 줄 단타 "
                  f"{r_fd['single'][0]:.0%} -> {r_fd['single'][1]:.0%} (사람 84%), 낀 박 건반 수 / 16분만 박 "
                  f"{r_fd['ratio'][0]:.2f} -> {r_fd['ratio'][1]:.2f}배 (사람 1.13배)")
    if beat_mode and eighth_audit > 0 and lane_fn is not None:
        # 마지막 점검 - 앞 규칙들이 겹쳐 민 '박 안 두 8분 같음', '뼈대 8분이 큼'을 사람 비율보다 넘친 만큼만 되돌림
        from thick_ctl import eighth_audit as _ea8
        masks, n_ea8, r_ea8 = _ea8(t_chart, masks, bpm, eighth_audit, seed, jack_ms=jk, max_chord=max_chord)
        if n_ea8:
            print(f"8비트 비율 마지막 점검(--eighth-audit {eighth_audit:g}): {n_ea8}줄 | 박 안 두 8분 같음 "
                  f"{r_ea8['bp'][0]:.0%} -> {r_ea8['bp'][1]:.0%} (사람 41%), 뼈대 8분이 큼 {r_ea8['sk'][0]:.0%} -> {r_ea8['sk'][1]:.0%} (사람 64%)")
    if lane_fn is not None and superset_fix > 0:
        from thick_ctl import superset_fix as _ssf
        masks, n_ss, n_sf = _ssf(t_chart, masks, bpm, seed, jack_ms=jk)
        if n_sf:
            print(f"16분 '앞 줄 품고 더하기'(45 -> 457) 풀기(--superset-fix): {n_sf}곳 중 {n_ss}곳 (사람 16분 간격 0.1%)")
    if lane_fn is not None and jack16 > 0:
        from thick_ctl import jack16 as _j16
        masks, n_j16 = _j16(t_chart, masks, bpm, jack16, seed)
        print(f"16분 간격 연타 줄이기(--jack16 {jack16:g}): 건반 {n_j16}개를 옆 건반으로 옮김 (16분 = {60000.0 / bpm / 4:.0f}ms)")
    if lane_fn is not None and no_jack_ms > 0:
        from thick_ctl import jack_fix
        masks, n_jf = jack_fix(t_chart, masks, no_jack_ms, seed)
        if n_jf:
            print(f"연타 마지막 점검: {no_jack_ms:g}ms 안 같은 건반 {n_jf}줄을 옆 건반으로 옮김 (--no-jack-ms)")
    if beat_mode and lane_fn is not None and jack8 > 0:
        # 8분 간격 겹침, 16분 3줄 연속을 BPM대 사람 울타리까지 (자리 모델, 자리 규칙 뒤라 되돌려지지 않음)
        from thick_ctl import jack8_fix as _j8
        masks, r_j8 = _j8(t_chart, masks, bpm, jack8, seed, jack_ms=jk)
        (e8a, r8a, k8a, _, t3a), (_, r8b, k8b, _, t3b) = r_j8["before"], r_j8["after"]
        fr_, fk_, ft_ = r_j8["fence"]
        if "chance" in r_j8:                         # 줄이 두꺼우면 '우연히 겹칠 양 대비' 사람 중앙값이 울타리
            fk_, fr_ = r_j8["chance"][2], r_j8["chance"][3]
        print(f"8분 간격 겹침 줄이기(--jack8 {jack8:g}): {r_j8['rows']}줄에서 겹친 건반 {r_j8['keys'] - r_j8['tri']}개 + 16분 3줄 연속 "
              f"{r_j8['tri']}곳을 안 겹치는 자리로 (건반 수 그대로) | 8분 간격 줄 {e8a}개 중 겹침 {r8a:.0%} -> {r8b:.0%}, 줄당 "
              f"{k8a:.2f} -> {k8b:.2f}, 16분 3줄 연속 {t3a} -> {t3b}곳 (울타리 = 사람 같은 BPM대·같은 두께 {fr_:.0%} / {fk_:.2f} / "
              f"16분 줄 1000개당 {ft_:.1f}곳)")
    if beat_mode and lane_fn is not None and jack_allow > 0:
        # BPM 150 아래 빽빽한 폭타 마디에서 이웃 16분 줄 같은 건반(2연타)을 BPM, 두께 비슷한 사람 곡 중앙값 쪽으로,
        # 같은 소리가 반복되는 쌍에만 (lowjack_ctl.py). 연타 정리(--jack16 / --no-jack-ms / --jack8) 뒤라 다시 지워지지 않음
        from lowjack_ctl import jack_allow as _jal
        from trem_ctl import trem_rows as _trw2
        prof_ja = None
        if mel is not None and len(frames) == len(masks):
            from jack_ctl import onset_profiles
            prof_ja = onset_profiles(mel, np.asarray(frames, dtype=np.int64))
        masks, r_ja = _jal(t_chart, masks, bpm, jack_allow, seed, jack_ms=jk, protect=_trw2(t_chart, bpm, trem_slots),
                           prof=prof_ja)
        if r_ja["skip"]:
            print(f"낮은 BPM 따닥 허용(--jack-allow {jack_allow:g}): 안 함 - {r_ja['skip']}")
        elif "ref" not in r_ja:
            print(f"낮은 BPM 따닥 허용(--jack-allow {jack_allow:g}): 안 함 - 빽빽한 폭타 마디 없음")
        else:
            (np_, ov0, hu_, rn0, _, _), (_, ov1, _, rn1, _, _) = r_ja["before"], r_ja["after"]
            e1, e2, e3 = r_ja["edits"]
            rr_, nref_, db_, dk_, kpr_ = r_ja["ref"]
            snd_ = r_ja["sound"]
            print(f"낮은 BPM 따닥 허용(--jack-allow {jack_allow:g}): 빽빽한 폭타 마디 {r_ja['meas']}개(줄당 건반 {kpr_:.2f}) 이웃 16분 줄 "
                  f"{np_}쌍 중 같은 건반 있음 {ov0:.0%} -> {ov1:.0%} (사람 기준 {hu_:.0%} = BPM ±{db_:g}·줄당 건반 ±{dk_:g} 비슷한 "
                  f"{nref_}곡의 무작위 대비 겹침 중앙값 {rr_:.2f}배) | "
                  + (f"같은 소리 반복 쌍(곡 안 유사도 위쪽 절반) {snd_[1]}/{snd_[2]}쌍에만, 비슷한 쌍부터 | " if snd_ else "소리 못 봄(무작위 자리) | ")
                  + f"{e1 + e2 + e3}줄 건반 하나 바꿈 (건반 수 그대로: 한 건반 이어짐 {e1}, 둘째 {e2}, 셋째 {e3}) | "
                  f"같은 건반 16분 2번/3번/4번+ 연속 {rn0[0]}/{rn0[1]}/{rn0[2]} -> {rn1[0]}/{rn1[1]}/{rn1[2]}곳")
    # 스크래치 자리 모델 (STEP 6)
    sc_done = False
    if (beat_mode and scratch_place > 0 and scratch_model and Path(scratch_model).exists() and lane_fn is not None
            and mel is not None and len(frames) == len(masks)):
        # --scratch-model: 사람 패턴 277개로 학습한 '스크래치 자리' 확률 순서로 스크래치를 놓음 (scratch_model.py)
        from scratch_model import TreeModel, place_scratch
        bar = 4 * 60.0 / bpm
        n_meas = len(set(np.floor(np.asarray(t_chart) / bar).astype(np.int64).tolist()))
        cur = sum(m & 1 for m in masks)
        target = max(cur, int(round(scratch_add * n_meas))) if scratch_add > 0 else cur
        masks, _, rep = place_scratch(t_chart, frames, masks, mel, bpm, TreeModel(scratch_model), target,
                                      scratch_place, seed)
        if rep:
            pz = rep["pos"]
            print(f"스크래치 자리 모델(--scratch-place {scratch_place:g}): {rep['before']}줄 -> {rep['after']}줄"
                  f" (그대로 둔 자리 {rep['kept']}) | 마디 첫 박 {pz[0]:.0%}, 3박 {pz[1]:.0%}, 2·4박 {pz[2]:.0%}, "
                  f"8분 {pz[3]:.0%}, 16분 {pz[4]:.0%} (사람 27/12/22/20/9%, 심벌·쉼 뒤·큰 소리에 몰림)")
        sc_done = True
    if beat_mode and scratch_add > 0 and not sc_done and lane_fn is not None and mel is not None and len(frames) == len(masks):
        from thick_ctl import add_scratch
        n_sc0 = sum(m & 1 for m in masks)
        masks, n_sc = add_scratch(t_chart, frames, masks, mel, bpm, scratch_add, seed, pos_weight=scratch_pos)
        if n_sc:
            print(f"스크래치 채우기(--scratch-add {scratch_add:g}/마디): {n_sc0}줄 -> {n_sc0 + n_sc}줄 "
                  f"(lv20 마디당 0.96, 저·중음이 센 8분 격자 줄에)")
    if lane_fn is not None and max_chord < 7:
        from thick_ctl import chord_cap
        masks, n_cc = chord_cap(t_chart, masks, max_chord, seed, allow=end_allow)
        if n_cc:
            print(f"동시치기 상한 마지막 점검(--max-chord {max_chord}): 후처리로 넘친 {n_cc}줄에서 건반을 뺌 (스크래치는 안 셈)")
    if beat_mode and lane_fn is not None and trem > 0 and trem_slots:
        # 떨림 박 줄은 단타로 (구간 밀도 줄이기) - 박마다 사람 24, 32분 단타 연속 모양, 박 머리만 원래 동시치기면 2개
        from trem_ctl import trem_single
        masks, r_ts = trem_single(t_chart, masks, bpm, trem_slots, jack_ms=jk, seed=seed, max_chord=max_chord)   # 사람 두께
        kd_ = r_ts["kinds"]
        print(f"떨림 줄 두께(--trem): {r_ts['rows']}줄 건반 {r_ts['keys_before']} -> 단타 {r_ts['keys_single']} -> 사람 두께 {r_ts['keys_after']}개 "
              f"(patch105: 박 머리 단타/2/3+ 34/35/31%, 16분 칸 60/30/10%, 사이 88/11/1% - 사람 124곡 떨림 박 3083개) | 박 {r_ts['beats']}개 중 "
              f"사람 24·32분 연속 모양 {r_ts['shaped']}박(" + ", ".join(f"{k_} {v_}" for k_, v_ in sorted(kd_.items(), key=lambda x: -x[1]))
              + f"), 빈 건반 따라 {r_ts['greedy']}박 | {jk:g}ms 안 같은 건반 {r_ts['jack']}줄")
    if beat_mode and lane_fn is not None and lad_rung:
        # 사다리 폭타 마디가 뒤 후처리로 단계보다 두꺼워졌으면 덜기만 함
        from thick_ctl import ladder_fill as _lf2
        keep_lc = [int(np.argmin(np.abs(np.asarray(t_chart) - x_))) for x_ in accent_t] if accent_t else []
        masks, n_lc, _ = _lf2(t_chart, masks, bpm, lad_rung, seed + 1, max_chord=max_chord, jack_ms=jk, add=False,
                              keep_idx=keep_lc)
        if n_lc:
            print(f"사다리 폭타 마지막 점검(--ladder): 뒤 후처리로 단계보다 두꺼워진 건반 {n_lc}개 뺌")
    if lane_fn is not None:
        # 레인 모델이 처음 찍은 건반이 후처리 뒤에 얼마나 남았나
        same = kept = raw_n = fin_n = 0
        for t, m in zip(t_chart, masks):
            r = raw_lane.get(round(float(t), 4))
            if r is None:
                continue
            same += int(r == m)
            kept += bin(r & m).count("1"); raw_n += bin(r).count("1"); fin_n += bin(m).count("1")
        print(f"레인 모델 결정이 남은 비율: 줄 그대로 {same / max(len(masks), 1):.0%} | 모델이 찍은 건반 중 남은 것 "
              f"{kept / max(raw_n, 1):.0%} | 최종 건반 중 모델이 찍은 것 {kept / max(fin_n, 1):.0%}")
    note_t = [float(t) for t, m in zip(t_chart, masks) for b in range(8) if (m >> b) & 1]   # 동시치기는 같은 시각이 반복됨
    lanes = [b for m in masks for b in range(8) if (m >> b) & 1]
    write_wav(out_dir / f"{name}.wav", padded, sr0)
    key_wav = "silence.wav" if key_sound == "silence" else "click.wav"
    write_silence(out_dir / "silence.wav") if key_sound == "silence" else write_click(out_dir / "click.wav")
    # 차분명이 있으면 <이름>_<차분명>.bms / _rows.csv / _clicks.wav 로 따로 저장 (음원, 분리 결과는 같이 씀)
    tag = re.sub(r'[\\/:*?"<>|\s]+', "_", str(chart_name or "")).strip("._")
    stem = f"{name}_{tag}" if tag else name
    bms_path = out_dir / f"{stem}.bms"
    try:                                       # 분석용: 줄마다 박 위치(16분 칸), 노트 시각 모델 확신도, 레인
        fr_ = np.asarray(frames, dtype=np.int64)
        pk_rows = np.max([probs[np.clip(fr_ + d, 0, len(probs) - 1)] for d in range(-3, 4)], axis=0)  # 격자 이동 보정
        with open(out_dir / f"{stem}_rows.csv", "w", encoding="utf-8") as fp:
            fp.write("slot16,beat,prob,mask\n")
            for tt, pp, mm in zip(t_chart, pk_rows, masks):
                b = float(tt) / (60.0 / bpm)
                fp.write(f"{int(round(b * 4))},{b:.4f},{float(pp):.4f},{int(mm)}\n")
    except Exception as e:  # noqa: BLE001
        print(f"[참고] 분석용 줄 목록을 못 썼어요: {e}")
    if nlog is not None:
        try:                                   # 노트 후보 진단표
            n_r, n_s, cnt, n_far2 = nlog.write(out_dir / f"{stem}_notes_debug.csv", stereo, sr0, pad, bpm, t_chart)
            order_ = ["노트", "확률 미달", "격자 합침", "약한 16분 정리(--thin-weak)", "악기 따라가기(--follow)",
                      "격자 정리(--offgrid-keep)", "밀도 모델", "치고 쉬기(--accent-rest)", "구간 밀도 분포(--dens-shape)",
                      "후처리에서 빠짐", "모델 후보 없음"]
            parts = [f"{k} {cnt[k]}" for k in order_ if cnt.get(k)]
            print(f"노트 진단(--note-log): {stem}_notes_debug.csv ({n_r}줄) | 센 소리(세기 0.4+) {n_s}개 -> "
                  + ", ".join(parts) + f" | 25ms 넘게 옮겨진 노트 {n_far2}개")
        except Exception as e:  # noqa: BLE001
            print(f"[참고] 노트 진단표를 못 썼어요: {e}")
    # ---- STEP 7 저장: BMS 파일로 쓰고, 다시 읽어 노트 수와 시각을 재검증 ----
    quant, qstats = write_bms(bms_path, note_t, lanes, bpm, title=title or name,
                              bgm_wav=f"{name}.wav", key_wav=key_wav,
                              tol_ms=min(tol_ms, 1.0) if snap else tol_ms,   # 이미 격자에 붙였으면 근사 분할 금지
                              rank=rank, total=total, artist=artist or "", subtitle=chart_name or "")
    check = verify_written(bms_path, quant)

    clicks_path = None
    if clicks:
        chart_like = SimpleNamespace(notes=[Note(time=float(t), lane=int(l))
                                            for t, l in zip(note_t, lanes)])
        clicks_path = out_dir / f"{stem}_clicks.wav"
        write_wav(clicks_path, add_clicks(padded, chart_like, sr0), sr0)

    return {
        "bms": bms_path, "clicks": clicks_path, "n_notes": len(times), "n_total": len(note_t),
        "duration": len(stereo) / sr0, "grid_R": R, "phase_ms": phase * 1000.0,
        "pad_ms": pad * 1000.0, "quant": qstats, "verify": check, "drift_ms": drift_ms, "snap": snap_info,
        "times": times, "t_chart": t_chart, "lanes": lanes, "masks": masks,
        # BMS에 실제로 쓴 시각(양자화 후)을 원본 오디오 기준으로 되돌린 값 (동시치기는 한 시각으로 묶음)
        "written_times": np.unique(np.round(np.array([qt for qt, _, _ in quant]) - pad, 6)),
    }


# ---------------------------------------------------------------------------
# 모델 불러오기 (torch는 여기서만 import)
# ---------------------------------------------------------------------------
def load_model(ckpt_path):
    import torch
    from model import build_onset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)   # 직접 학습한 체크포인트
    a, cfg = ck["args"], ck["data_cfg"]
    if (cfg["sr"], cfg["hop"], cfg["n_mels"]) != (SR, HOP, N_MELS):
        raise SystemExit("체크포인트의 전처리 설정이 build_dataset.py와 달라요. 같은 설정으로 다시 학습하세요.")
    n_out = 1 if a["target"] == "timing" else 8
    model = build_onset(a, cfg["n_mels"], n_out).to(device)     # v6 문맥 모델도 자동 인식
    model.load_state_dict(ck["model"])
    model.eval()

    def predict_fn(x: np.ndarray) -> np.ndarray:           # x: (n_mels, W) -> (W,)
        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0).to(device)
            p = torch.sigmoid(model(t))[0].float().cpu().numpy()   # (C, W)
        return p.max(axis=0)

    return predict_fn, ck, device


def load_lane_fn(ckpt_path, mel: np.ndarray, density: float, temperature: float = 0.8,
                 max_chord: int = 6, no_jack_ms: float = 0.0, seed: int = 0,
                 loud: np.ndarray | None = None, dynamics: float = 0.0,
                 jack_penalty: float = 0.0, jack_window_ms: float = 0.0, scratch_bias: float = 0.0,
                 coverage: float = 0.0, stream_ms: float = 0.0, vary: float = 0.0,
                 same_sound: float = 0.0, same_window_ms: float = 0.0, lone_scratch: bool = False,
                 plan_ckpt: str | None = None, plan_temp: float = 0.5, alt_scale: float = 1.0,
                 max_alt_rows: int = 0, scratch_pos: float = 0.0, row_seed: bool = False):
    """레인 모델 체크포인트를 불러와 lane_fn(frames) -> 마스크 배열을 만든다. torch는 여기서만 import."""
    import torch
    from lane_model import LaneNet, make_step_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)   # 직접 학습한 체크포인트
    a, cfg = ck["args"], ck["data_cfg"]
    if (cfg["sr"], cfg["hop"], cfg["n_mels"]) != (SR, HOP, N_MELS):
        raise SystemExit("레인 체크포인트의 전처리 설정이 build_dataset.py와 달라요.")
    def cond_at(frames: np.ndarray):
        """노트마다의 밀도 조건. --dynamics면 소리 큰 구간은 높게, 조용한 구간은 낮게."""
        base = cond_value(density)
        if loud is None or dynamics <= 0:
            return base
        from dynamics import cond_scale
        f = np.clip(np.asarray(frames, dtype=np.int64), 0, len(loud) - 1)
        return (base * cond_scale(loud[f], dynamics)).astype(np.float32)

    if a.get("arch") == "v2":                     # 레인 모델 v2 (구간 흐름 유지)
        from lane_v2 import NOTE_DIM, LaneNetV2, generate_v2, nf_layout, plan_in, rhythm_codes_from_times
        note_dim = int(a.get("note_dim", NOTE_DIM))
        has_rhy, has_sec = nf_layout(note_dim)
        planner = None
        if plan_in(note_dim):
            if not plan_ckpt:
                raise SystemExit("v4 레인 모델은 계획 모델이 필요해요 → --plan-ckpt checkpoints\\plan.pt 를 붙이세요")
            from plan_model import load_plan, predict_plan
            planner = load_plan(plan_ckpt, device)
            print(f"계획 모델: {plan_ckpt} (학습 step {planner[2].get('step')}, 온도 {plan_temp:g})")
        quality = bool(a.get("quality", False))
        model = LaneNetV2(cfg["n_mels"], 2 * a["patch_k"] + 1, a["hidden"], a["layers"], a["dropout"],
                          note_dim=note_dim).to(device)
        if has_rhy:
            print("레인 모델: 박자 위치(마디 첫 박 등) 입력 사용")
        if has_sec:
            print("레인 모델: 곡 구조(쉼/구간 밀도) 입력 사용")
        if quality:
            print("레인 모델 v3: 좋은 차분 모드로 생성 (--density, --dynamics는 레인 모델에 쓰이지 않아요)")
        model.load_state_dict(ck["model"])
        model.eval()
        cond = cond_value(density)

        def lane_fn_v2(frames: np.ndarray, t_chart=None, bpm=None) -> np.ndarray:
            c = 1.0 if quality else cond_at(frames)
            codes = None
            if has_rhy:
                if t_chart is not None and bpm:
                    codes = rhythm_codes_from_times(t_chart, bpm, 0)
                else:
                    print("[주의] 박자 위치 모델은 --snap beat와 함께 써야 해요. 지금은 박자 정보 없이 생성해요.")
            plan_n = None
            if planner is not None:
                from plan_feats import PLAN_NAMES
                pc = codes if codes is not None else np.full(len(frames), -1)
                plan_n, plan_g, _ = predict_plan(planner[0], planner[1], mel, frames, pc, 1.0, plan_temp, device,
                                                 a["patch_k"])
                if alt_scale != 1.0:                        # 묶음 교대 계획 줄이기/늘리기
                    plan_n[:, 1] *= alt_scale
                    plan_g[:, 1] *= alt_scale
                print(f"패턴 계획: 반 마디 {len(plan_g)}개 | 평균 " +
                      ", ".join(f"{n} {v:.2f}" for n, v in zip(PLAN_NAMES, plan_g.mean(0))) +
                      f" | 묶음 교대(1357/246, 1245/367 등)가 많은(>0.3) 반 마디 {(plan_g[:, 1] > 0.3).mean():.0%}")
            return generate_v2(model, mel, frames, c, a["patch_k"], device, temperature, max_chord,
                               no_jack_ms, seed, win=a.get("seq_len", 256),
                               jack_penalty=jack_penalty, jack_window_ms=jack_window_ms,
                               scratch_bias=scratch_bias, rhythm_codes=codes,
                               coverage=coverage, stream_ms=stream_ms, vary=vary,
                               same_sound=same_sound, same_window_ms=same_window_ms,
                               lone_scratch=lone_scratch, plan_feats=plan_n, max_alt_rows=max_alt_rows,
                               scratch_pos=scratch_pos,
                               scratch_codes=rhythm_codes_from_times(t_chart, bpm, 0)
                               if (t_chart is not None and bpm) else None,
                               scratch_beats=np.asarray(t_chart, dtype=np.float64) * bpm / 60.0
                               if (t_chart is not None and bpm) else None, row_seed=row_seed)
        return lane_fn_v2, ck
    model = LaneNet(n_mels=cfg["n_mels"], patch_w=2 * a["patch_k"] + 1, hidden=a["hidden"],
                    layers=a["layers"], dropout=a["dropout"]).to(device)
    if ck["model"]["head.weight"].shape[0] != 8:
        raise SystemExit("이 레인 체크포인트는 출력이 255클래스라 이 코드(8레인 출력)와 맞지 않아요. train_lane.py로 다시 학습하세요.")
    model.load_state_dict(ck["model"])
    model.eval()
    cond = cond_value(density)

    def lane_fn(frames: np.ndarray, t_chart=None, bpm=None) -> np.ndarray:
        step_fn = make_step_fn(model, extract_patches(mel, frames, a["patch_k"]), device)
        return generate_masks(step_fn, frames, cond_at(frames), temperature, max_chord, no_jack_ms, seed)

    return lane_fn, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="곡 오디오 (wav/ogg/flac 권장)")
    ap.add_argument("--ckpt", default="checkpoints/onset_best.pt")
    ap.add_argument("--bpm", type=float, default=None,
                    help="곡의 BPM (고정 BPM만 지원). 생략하면 예측한 노트에서 자동 추정")
    ap.add_argument("--bpm-range", type=float, nargs=2, default=[120.0, 280.0], metavar=("MIN", "MAX"),
                    help="자동 추정 때 우선할 BPM 범위 (2배/절반 중 이 범위 안의 값을 고름)")
    ap.add_argument("--offset", type=float, default=None,
                    help="박자 격자 위상(초). 생략하면 예측한 노트에서 자동 추정")
    ap.add_argument("--threshold", type=float, default=None, help="노트 확률 임계값 (생략하면 체크포인트의 값)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--folder", default=None,
                    help="--out 안에 만들 결과 폴더 이름 (생략하면 곡 파일 이름을 영문으로 바꾼 것). 창 프로그램은 곡 이름 그대로, "
                         "같은 곡을 다시 만들면 덮어씀")
    ap.add_argument("--title", default=None)
    ap.add_argument("--chart-name", default="",
                    help="차분명. 주면 <이름>_<차분명>.bms 등으로 따로 저장(같은 곡 여러 개 비교, 덮어쓰지 않음)하고 #SUBTITLE에 씀. "
                         "BMS는 Shift-JIS라 게임 안 표시는 영어·숫자·일본어만 (한글은 ?로)")
    ap.add_argument("--artist", default="", help="작곡가 (#ARTIST). 없으면 빈칸. #GENRE는 항상 AI Generated")
    ap.add_argument("--judge", default="normal", type=str.lower,
                    choices=["easy", "normal", "hard", "veryhard", "very_hard", "0", "1", "2", "3"],
                    help="판정 난이도(#RANK): easy / normal(기본) / hard / veryhard (숫자 3/2/1/0도 됨)")
    ap.add_argument("--total", type=float, default=300.0,
                    help="#TOTAL (게이지 회복량, 기본 300). 노트가 많으면 보통 더 크게")
    ap.add_argument("--lane-mode", choices=["cycle", "random"], default="cycle")
    ap.add_argument("--tol-ms", type=float, default=10.0,
                    help="마디 격자 맞춤 허용 오차(ms). 크게 하면 정돈된 격자(16분 등)로 더 많이 맞춰짐")
    ap.add_argument("--window-sec", type=float, default=20.0)
    ap.add_argument("--overlap-sec", type=float, default=3.0)
    ap.add_argument("--no-clicks", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-sound", choices=["silence", "click"], default="silence",
                    help="노트를 칠 때 나는 소리. click이면 BMS 플레이어에서 노트마다 클릭음이 나서 정렬을 귀로 확인할 수 있음")
    ap.add_argument("--lane-ckpt", default=None,
                    help="2단계(레인 배치) 체크포인트. 생략하면 레인을 규칙(cycle/random)으로 배정")
    ap.add_argument("--density", type=float, default=20.0, help="레인 모델에 주는 목표 노트 밀도(개/초 기준 곡 전체 평균)")
    ap.add_argument("--temperature", type=float, default=0.8, help="레인 샘플링 온도 (0이면 항상 최고 확률, 높을수록 다양)")
    ap.add_argument("--max-chord", type=int, default=6, help="동시치기 상한")
    ap.add_argument("--no-jack-ms", type=float, default=100.0,
                    help="이 간격(ms)보다 가까운 노트는 직전과 같은 레인을 쓰지 않음 (기본 100: 16분 연타 금지, 0이면 끔)")
    ap.add_argument("--jack-penalty", type=float, default=0.0,
                    help="소리 기반 연타 조절 (레인 모델 v2). 직전과 다른 소리인데 같은 레인을 치려 하면 그만큼 억제. "
                         "같은 소리 반복(강조)은 그대로 둠. 추천 3, 0이면 끔")
    ap.add_argument("--scratch-bias", type=float, default=0.0,
                    help="스크래치 빈도 조절 (레인 모델 v2). 숫자가 클수록 스크래치가 많아짐. "
                         "1이면 약 2.5배, 0.5면 약 1.6배, 음수면 줄어듦. 0이면 그대로")
    ap.add_argument("--scratch-pos", type=float, default=1.0,
                    help="스크래치 자리 가중치 세기 (lv20 기준). 앞 1박 안에 스크래치가 없으면(처음 놓는) 첫 박 1 / 4박 0.46 / 2박 0.32 / "
                         "3박 0.24 / 8분 뒷박 0.04 / 16분 0.01, 있으면(이어 치는) 첫 박 1 / 4박 0.91 / 3박 0.7 / 2박 0.38 / 8분 0.28 / 16분 0.18을 "
                         "스크래치 확률(오즈)에 곱함 (1 = 그대로, 2 = 제곱이라 더 강하게, 0 = 끔). 레인 모델과 --scratch-add 둘 다")
    ap.add_argument("--coverage", type=float, default=0.0,
                    help="16분 스트림에서 7건반을 고루 쓰게: 4줄 넘게 안 쓴 건반일수록 확률을 올림 "
                         "(레인 모델 v2). 0이면 끔, 추천 1~2")
    ap.add_argument("--vary", type=float, default=0.0,
                    help="16분 스트림에서 같은 2~4줄 묶음이 두 번 연속 그대로 반복되려 하면 이 확률로 다시 뽑아 섞음 "
                         "(1357 246 1357 246 -> 중간에 변화). 0이면 끔, 추천 0.7")
    ap.add_argument("--downbeat", type=int, default=-1,
                    help="--snap beat에서 마디 첫 박 위치. -1(기본)이면 소리로 자동 추정, 0~3이면 직접 지정. "
                         "구동기에서 마디선이 한 박씩 어긋나 보이면 바꿔 보세요")
    ap.add_argument("--same-sound", type=float, default=0.0,
                    help="4·8비트 구간에서 비슷한 소리가 다시 나오면 그때 찍은 레인 모양 쪽으로 미는 세기 (0=끔, 권장 2~3)")
    ap.add_argument("--plan-ckpt", default=None, help="v4 계획 모델 (plan.pt). v4 레인 모델이면 필수")
    ap.add_argument("--plan-temp", type=float, default=0.5,
                    help="계획 온도: 낮을수록 계획 모델이 가장 그럴듯하다고 본 모양으로, 1이면 평균적으로")
    ap.add_argument("--max-alt-rows", type=int, default=3,
                    help="묶음 교대(1357/246, 1245/367, 13/24 등 A-B-A-B)를 최대 몇 줄까지 허용할지. "
                         "presea는 최장 3줄(A-B-A), 2면 A-B-A도 막음. 0이면 제한 없음")
    ap.add_argument("--alt-scale", type=float, default=1.0,
                    help="v4: 계획 모델이 정한 교대 비율에 곱하는 값 (0이면 교대 계획 없음)")
    ap.add_argument("--lone-scratch", action="store_true",
                    help="스크래치만 있는 줄을 그대로 둔다 (기본은 건반 하나를 같이 붙임, 사람 차분은 이런 줄이 3%% 정도)")
    ap.add_argument("--same-beats", type=float, default=8.0,
                    help="--same-sound가 비슷한 소리를 찾는 범위(박). 기본 8박 = 2마디")
    ap.add_argument("--jack-beats", type=float, default=1.0,
                    help="--jack-penalty를 적용할 간격(박 단위). 1이면 4분까지(8분/4분 연타 포함), 0.5면 8분까지만")
    ap.add_argument("--snap", default="off",
                    help="노트를 붙일 격자: off(기본, 예측 시각을 --tol-ms 안에서 그대로 둠) / "
                         "beat(추천: 박마다 16분/셋잇단/32분 중 맞는 것을 골라 붙임) / 16(전부 16분) / "
                         "auto(16분 기본, 같은 칸에 두 노트가 모이면 32분/64분으로 분리. 모델 오차까지 32분이 될 수 있음) / "
                         "32 / 16,12(16분+셋잇단)")
    ap.add_argument("--beat-divs", default="4,3,6,8",
                    help="--snap beat에서 박당 쓸 수 있는 분할 (4=16분, 3=셋잇단8분, 6=셋잇단16분, 8=32분, "
                         "느린 곡용 12, 16). 예: 32분을 아예 안 쓰려면 4,3,6 / 셋잇단을 안 쓰려면 4,8")
    ap.add_argument("--offgrid-keep", type=float, default=75.0,
                    help="--snap beat에서 32분/셋잇단 칸 노트를 남길 기준(곡 안 노트 확신도 백분위). "
                         "75면 상위 25%%만 남기고 나머지는 16분으로. 100이면 전부 16분으로, 0이면 정리 안 함")
    ap.add_argument("--thin-weak", type=float, default=0.0,
                    help="--snap beat에서 16분 사이 칸(박의 2·4번째 16분)의 노트 중 확신도가 곡 전체 하위 N%%인 것을 뺀다. "
                         "스트림이 약한 소리까지 다 메워 강세가 흐려질 때 20~40 권장. 0이면 끔")
    ap.add_argument("--density-ckpt", default=None,
                    help="반 마디 밀도 모델 (density.pt). 주면 반 마디마다 사람처럼 찍을 줄 수만 남긴다 (--snap beat 필요)")
    ap.add_argument("--density-q", type=float, default=0.5,
                    help="밀도 모델 확률에서 줄 수를 고르는 기준 (0.5 = 중앙값, 높이면 노트가 많아짐, 0.3~0.7)")
    ap.add_argument("--density-conf", type=float, default=0.25,
                    help="밀도 모델이 16비트일 확률을 이 값 이상으로 보면 그 반 마디는 손대지 않는다 "
                         "(낮을수록 확실한 곳만 깎음, 1이면 16비트 확률과 상관없이 중앙값만 보고 깎음)")
    ap.add_argument("--keep-sure", type=float, default=0.93,
                    help="1단계 확신도가 이 값 이상인 노트는 밀도 모델/구간 정리가 지우지 않는다 (1이면 확신도와 상관없이 지울 수 있음)")
    ap.add_argument("--stream-end-chord", type=int, default=1,
                    help="16분 연타가 끊기는 마지막 줄을 바로 앞 줄보다 이 개수 이상 많은 동시치기로 (0이면 끔, 2면 더 묵직하게). "
                         "앞뒤 줄 건반은 피해서(연타 금지) 자리가 모자라면 넣을 수 있는 만큼만")
    ap.add_argument("--thick", type=float, default=0.7,
                    help="곡 전체에서 꽉 찬 구간은 두꺼운 동시치기, 빈 구간은 얇게 (0 = 끔, 1 = 목표 그대로). 노트 위치는 그대로")
    ap.add_argument("--thick-range", default="1.0,3.2",
                    help="가장 빈 연타 / 가장 꽉 찬 연타의 목표 평균 두께 (예: 1.2,2.8이면 차이가 작게)")
    ap.add_argument("--fill32", type=float, default=0.0,
                    help="16비트 이상 채우기: 확실한 두 노트 한가운데의 약한 소리(1단계 확신도가 이 값 이상)를 더함. "
                         "16분 사이 -> 32분, 8분 셋잇단 사이 -> 24분. 작을수록 많이. 0이면 끔"),
    ap.add_argument("--dump-probs", default="",
                    help="확인용: 1단계 노트 확신도(10ms마다)를 .npy로 저장 (예: out/probs.npy). 32분 등을 놓칠 때 원인 확인")
    ap.add_argument("--raw-lanes", action="store_true",
                    help="확인용: 레인 후처리 규칙을 모두 끔 (두께, 벌리기, 성격 유지, 대비, 데님, 교대 재제한, 연타 끝, 성긴 구간). "
                         "레인 모델이 뽑을 때의 연타 금지와 --max-alt-rows는 그대로")
    ap.add_argument("--random-lanes", action="store_true",
                    help="확인용: 레인 모델이 정한 건반 수는 두고 자리만 무작위로 (모델이 자리를 의미 있게 고르는지 비교)")
    ap.add_argument("--denim-max", type=int, default=8,
                    help="데님(13 24 13 / 13 24 35 46 같은 2개 동시치기 16분 연속)을 4~이 값 줄에서 끊음. 0이면 끔")
    ap.add_argument("--beat-contrast", type=float, default=0.7,
                    help="16분 연타에서 뒷자리 줄이 양옆 8분 줄보다 얇지 않으면 이 확률로 얇게 (사람처럼 센 줄-약한 줄 대비). 0이면 끔")
    ap.add_argument("--flow", type=float, default=0.0,
                    help="16분 연타 속 8분 자리 동시치기가 방향을 바꾸려 할 때 이 확률로 같은 방향 계단으로 (75 64 -> 53). "
                         "소리가 같으면 안 바꿈. 0.35면 같은 방향 비율 19%%->27%% (presea 20%%, SA 28%%). 0이면 끔")
    ap.add_argument("--thin-boost", type=float, default=0.7,
                    help="--thin-weak로 16분 뒷자리를 뺀 자리 바로 옆 8분 자리 동시치기에 이 확률로 건반 하나를 더함 "
                         "(16분이 8분이 된 곳을 묵직하게, 연타 금지 지킴). --thin-weak를 안 쓰면 아무 일 없음. 0이면 끔")
    ap.add_argument("--follow", default="off",
                    help="곡 전체에서 따라갈 층: drum(드럼) / inst(악기, 드럼·보컬·베이스 뺀 나머지) / vocal(보컬) / off(끔, 기본). 음원 분리(Demucs)로 층을 나눠, "
                         "그 층의 어택이 없는 줄을 --follow-strength 확률로 뺌. 'pip install demucs' 필요, --snap beat일 때만")
    ap.add_argument("--follow-at", default="",
                    help="구간별 따라갈 층 (--follow보다 우선, 뒤에 적은 것이 이김): 마디로 \"1-16:drum, 17-32:inst, 33-:off\" / "
                         "초로 \"1~3초 보컬, 6~8초 드럼, 30~35초 악기\" (원래 음원 시각, a~b초는 a초부터 b초 직전까지). "
                         "마디 번호는 만든 BMS의 #마디 번호(000부터), 끝을 비우면 곡 끝까지. 섞어 써도 됨")
    ap.add_argument("--follow-strength", type=float, default=0.7,
                    help="따라가지 않는 층에만 있는 줄을 뺄 확률 (0~1, 기본 0.7). 한 마디에서 70%%까지만 뺌")
    ap.add_argument("--stems-model", default="htdemucs", help="음원 분리 모델 이름 (Demucs, 기본 htdemucs)")
    ap.add_argument("--stems-device", default="cpu", help="음원 분리 장치 (cpu / cuda)")
    ap.add_argument("--jack16", type=float, default=0.5,
                    help="16분 간격 연타(같은 건반이 16분 뒤에 또)를 이 확률로 옆 건반으로 옮김. --no-jack-ms처럼 전부 막지 않고 일부만 남김 "
                         "(BPM 120처럼 16분이 100ms보다 길어 연타가 많이 나올 때). 기본 0.5(patch111, M109), 0이면 끔")
    ap.add_argument("--trip-match", type=float, default=1.0,
                    help="셋잇단 연속 두께를 사람 쪽으로: 6비트(4분 셋잇단) 연속은 건반 수를 고르게(같은 개수 반복), "
                         "24비트(16분 셋잇단) 연속은 단타 위주(1개 61%%)로 덜어냄. 0이면 끔")
    ap.add_argument("--skel-swap", type=float, default=1.0,
                    help="폭타 뼈대: 16분 연타 속 8분 자리 줄이 바로 뒤 뒷자리 줄보다 얇으면 이 확률로 두 줄 건반을 맞바꿈 "
                         "(사람은 8분 자리가 더 두꺼움 69%%, 뒷자리가 더 두꺼움 3%%). 0이면 끔")
    ap.add_argument("--bar-repeat", type=float, default=0.6,
                    help="마디 반복: 리듬·소리가 한 마디 전과 비슷한 마디에서 같은 칸 줄의 건반 수를 이 확률로 앞 마디처럼 "
                         "(사람 한 마디 전 건반 수 같음 62%%). 0이면 끔")
    ap.add_argument("--bar-shape", type=float, default=0.35,
                    help="--bar-repeat으로 따라간 줄 중 모양까지(같은 모양 / 좌우 반전 / 한 칸 평행이동) 따라갈 확률 (기본 0.35)")
    ap.add_argument("--diag", type=float, default=0.25,
                    help="대각선: 16분 연타에서 두 줄 앞과 건반 수가 같은 줄을 이 확률로 두 줄 앞 모양의 한 칸 평행이동으로 "
                         "(사람 7%%). 0이면 끔")
    ap.add_argument("--fine-single", type=float, default=1.0,
                    help="32분·24분 채운 줄(16분 격자 밖, 앞뒤 같은 간격) 단타 비율을 90%%까지 올리는 세기 "
                         "(사람 32분 89~93%%, 24분 83%%). 연타 금지 지킴. 0이면 끔")
    ap.add_argument("--fine-shape", type=float, default=1.0,
                    help="24·32분 3줄 이상 연속마다 이 확률로 모양 하나(계단 21 / 좌우 교대 23 / 지그재그 19 / 좁은 자리 반복 28 / "
                         "그대로 8%%)를 골라 단타로 놓음. 0이면 끔")
    ap.add_argument("--section-dyn", type=float, default=1.0,
                    help="구간 강약: 마디마다 구간 소리 크기 순위대로 줄당 건반 수를 조절 (조용한 구간은 얇게, 큰 구간은 두껍게, "
                         "16·8·4·2비트 모두). 1 = 사람 기준(조용한 1/3 줄당 약 1.65, 큰 1/3 약 2.41), 줄이면 강약이 약해져 어려워짐, "
                         "1.5까지. 0이면 끔")
    ap.add_argument("--eighth-trans", type=float, default=1.0,
                    help="8분~1박 간격으로 같은 개수 동치가 이어질 때 다음 줄을 사람 비율(그대로 8 / 하나만 이동 13 / 통째 평행이동 18 / "
                         "전부 새로 36 / 여럿 바뀜 25%%)로 바꿀 확률. 같은 소리면 그대로·하나만 이동을 더 고름. 0이면 끔")
    ap.add_argument("--end-fat", type=float, default=1.0,
                    help="폭타 끝 줄을 앞 1박 8분 자리 최대 동시치기 이상으로 (사람 66%%, 그중 더 큼 21%%). 0이면 끔")
    ap.add_argument("--end-over", type=int, default=1,
                    help="폭타 끝 줄만 --max-chord보다 이만큼 더 허용 (몸통은 그대로, 기본 1, 0이면 상한 그대로)")
    ap.add_argument("--snap-fix", type=float, default=25.0,
                    help="박 격자에 붙이며 이 ms보다 멀리 밀린 노트를 32분·24분·셋잇단 칸에 다시 놓고, 옆 노트와 합쳐져 "
                         "사라질 노트를 따로 살림 (박에 딱 안 맞는 연주 구간용, 기본 25, 0이면 끔)")
    ap.add_argument("--thin-local", type=float, default=2.0,
                    help="--thin-weak의 하위 N%%를 곡 전체가 아니라 앞뒤 이만큼의 마디 안에서 정함 "
                         "(조용한 구간이 통째로 깎이지 않게, 기본 2, 0이면 곡 전체 줄끼리)")
    ap.add_argument("--stream-hole", type=float, default=8.0,
                    help="--thin-weak가 폭타 속(앞뒤 16분·8분 자리가 다 있는 곳)에서 한 칸만 비게 만들 줄은 확신도가 곡 하위 이 %%일 때만 뺌 "
                         "(사람 폭타 1000줄당 한 칸 빈 곳 중앙값 36, 기본 8, 0이면 이 보호를 끔)")
    ap.add_argument("--end-shape", type=float, default=1.0,
                    help="폭타 끝 줄의 건반 수를 소리 크기 순위대로 사람 분포에(센 소리 4~5개, 약한 소리 1~2개), "
                         "모양을 사람 비율(이동 33 / 비슷 29 / 벌어짐 23 / 모임 10 / 그대로 5%%)로 바꿀 확률. 0이면 끔")
    ap.add_argument("--superset-fix", type=float, default=1.0,
                    help="16분 간격에서 뒷줄이 앞 줄(2개+)을 그대로 품고 건반을 더한 모양(45 -> 457)을 풂 (사람 0.1%%). 0이면 끔")
    ap.add_argument("--pair-even", type=float, default=1.0,
                    help="바쁜 마디(16분 칸 8+/16)에서 '8분 자리 줄 + 바로 뒤 16분 줄' 건반 합을 마디 중앙값으로 고르게: 2 이상 벗어나면 "
                         "1 안으로, 1 벗어나면 절반 확률로 맞춤 (사람 마디 안 표준편차 lv20 0.71 / Air -GOD- 0.60, 생성 0.93). 0이면 끔")
    ap.add_argument("--placer-ckpt", default="",
                    help="자리 모델(patch78) 체크포인트. 기본: infer.py 옆 checkpoints/placer_v1.pt가 있으면 그것, 'none'이면 안 씀")
    ap.add_argument("--placer", type=float, default=1.0,
                    help="자리 모델로 건반 자리를 다시 뽑을 줄 비율 (건반 수·스크래치는 그대로, 1 = 전부, 0 = 끔)")
    ap.add_argument("--placer-temp", type=float, default=1.0,
                    help="자리 모델 온도 (낮을수록 사람 차분에 흔한 모양만, 높을수록 다양하게, 기본 1.0 - Kaggle 평가에서 사람과 가장 가까움)")
    ap.add_argument("--pair-avg", type=float, default=4.0,
                    help="폭타 쌍 합(8분 자리 줄 + 바로 뒤 16분 줄 건반 합) 평균. 4마디 묶음마다 저·중음 크기 순위대로 "
                         "약한 구간은 낮게, 강한 구간은 높게 목표를 정함 (사람 lv16~20 곡 평균 10/50/90%% 2.65/3.7/4.63, 기본 4, "
                         "0이면 목표 = 그 마디 쌍 합 중앙값, --pair-min보다 작으면 중앙값 +1까지 올림)")
    ap.add_argument("--pair-dev", type=float, default=1.5,
                    help="--pair-avg 구간 편차 ±n: 가장 센 4마디 묶음은 평균 + n, 가장 약한 묶음은 평균 - n (클수록 센 곳은 더 살리고 약한 곳은 더 얌전하게, "
                         "사람 lv16~20 하위~상위 10%% 폭 약 2.6 = ±1.5 근처, 기본 1.5, 0이면 곡 전체 같은 값)")
    ap.add_argument("--pair-spread", type=float, default=None, help=argparse.SUPPRESS)   # 옛 이름 (폭 = 2 x 편차)
    ap.add_argument("--pair-min", type=int, default=5,
                    help="--pair-even 목표 쌍 합의 최소값: 마디 중앙값이 이보다 작으면 중앙값 +1까지 올림 "
                         "(Air -GOD- 5~6, 사람 lv20 평균 4.6, 기본 5, 0이면 중앙값 그대로)")
    ap.add_argument("--edge-axis", type=float, default=1.0,
                    help="1·7번 축(8분 자리 3줄 연속 같은 건반)이 축 전체의 40%%(사람 37~39%%)를 넘으면 가운데 줄의 1·7을 "
                         "가까운 안쪽 빈 건반(2·3 / 6·5)으로 옮기는 확률. 0이면 끔")
    ap.add_argument("--quarter-spread", type=float, default=1.0,
                    help="4비트(1박 이상 간격) 세 줄이 한쪽 손 범위 안에서 가운데 -> 끝으로 밀려가면(5 4 47) 마지막 줄을 반대쪽 손으로 옮기는 확률 "
                         "(사람 277곡 1.4%%, 같은 건반 반복·제자리 왕복은 그대로, 기본 1, 0이면 끔)")
    ap.add_argument("--trill-break", type=float, default=1.0,
                    help="한 손 안 붙은 두 건반(12·23·45·56·67)이 16분 이하 간격으로 번갈아 4줄 이상(그 손 몫이 6 7 6 7)이면 "
                         "셋째 줄마다 그 손의 다른 빈 건반으로 바꿔 끊는 확률 (사람 1000줄당 약 1곳, 기본 1, 0이면 끔)")
    ap.add_argument("--eighth-audit", type=float, default=1.0,
                    help="마지막 점검: 앞 규칙들이 겹쳐 민 결과를 재서 '박 안 두 8분 자리 건반 수 같음'(사람 41%%)과 "
                         "'뼈대 8분 줄이 뒤 16분 줄보다 큼'(사람 64%%)이 넘친 만큼만 되돌림 (건반 수 거의 그대로, 기본 1, 0이면 끔)")
    ap.add_argument("--hand-balance", type=float, default=1.0,
                    help="폭타 밖에서 오른손만 친 줄이 왼손만 친 줄의 1.42배(사람)를 넘으면 넘친 만큼 좌우 반전해 왼손으로 (4번은 오른손, 기본 1, 0이면 끔)")
    ap.add_argument("--fine-density", type=float, default=1.0,
                    help="폭타 속 32·24분 줄 단타를 사람(84%%)까지, 그 박 건반 수가 16분만 있는 박의 1.13배(사람)를 넘으면 "
                         "32·24분 앞뒤 16분 뒷자리 줄에서 덜어냄 (기본 1, 0이면 끔)")
    ap.add_argument("--placer-rep", type=float, default=0.45,
                    help="자리 모델 반복 억제: 두 줄 간격으로 같은 조합(1 _ 1 _ 1 기둥, A B A B 교대)이 이어질수록 덜 고르게 하는 세기 "
                         "(사람 277곡: 두 줄 전 그대로 8%%, 최장 6번 / 억제 없이 theHole 18~39%%, 최장 32번, 기본 0.45, 0이면 모델 그대로)")
    ap.add_argument("--stream-break", type=int, default=16,
                    help="16분 폭타 마디가 이 마디 수보다 길게 쉬지 않고 이어지면, 그 안에서 16분 뒷자리 소리가 약한 마디를 골라 "
                         "8비트 동시치기 마디(건반 수 = 주변 폭타의 0.76배)로 바꿈 (사람 277곡 폭타 최장 연속 중앙값 17마디, 기본 16, 0이면 끔)")
    ap.add_argument("--stream-target", type=float, default=0.64,
                    help="곡 전체에서 16분 폭타 마디 비율이 이 값을 넘으면 4~8마디 악절을 통째로 8비트 동시치기 구간(건반 수 = 주변 폭타의 0.5배)으로 바꿈 "
                         "(사람 256곡: 폭타 위주 곡 중앙값 64%%, 폭타 아닌 마디의 81%%가 4마디+ 덩어리 / 생성 EOS 79%%, 기본 0.64, 0이면 끔)")
    ap.add_argument("--stream-kpr", type=float, default=0.0,
                    help="16분 폭타 마디 줄당 건반을 이 값까지 올림, 소리 큰 마디일수록 두껍게 (사람 중앙값 2.28 / 생성 EOS 1.98). "
                         "patch93: 기본 0(끔) - 켜면 얇은·두꺼운 폭타 차이가 지워짐(마디별 줄당 편차 사람 0.43 / 켬 0.19 / 끔 0.48)")
    ap.add_argument("--accent-rest", type=float, default=1.0,
                    help="치고 쉬기: 폭타 중 큰 소리(앞 2박 보통 세기의 2배+) 뒤 소리가 잦아들면(뒤 1박이 그 소리 0.6배 이하이면서 앞 폭타 보통 세기 이하) "
                         "그 소리를 동시치기로 치고 뒤 --accent-rest-len박을 쉼 (flicknote 075·083마디, 기본 1, 클수록 더 자주, 0이면 끔)")
    ap.add_argument("--accent-rest-len", type=float, default=1.0,
                    help="치고 쉬기에서 큰 소리 뒤 쉬는 길이(박). 다음 줄은 이만큼 뒤에 옴 (patch95: 0.75 -> 1, 사용자 '1박 쉬는 게 나음')")
    ap.add_argument("--pair-keep", type=float, default=1.0,
                    help="폭타 쌍 합 고르게(--pair-even)가 마디 안 쌍 합 높낮이를 남기고 목표에서 2 이상 벗어난 쌍만 고침 "
                         "(사람 마디 안 표준편차 0.71 / 0일 때 0.33 -> 꽉 찬 박 건반 수 패턴이 2-1-2-1로 몰림, 기본 1, 0이면 모든 쌍을 목표 쪽으로 맞춤)")
    ap.add_argument("--fine-grid", type=float, default=1.0,
                    help="24·32비트(셋잇단·32분) 격자 허용 (patch107: --fine-bpm이 켜져 있으면 자동은 박마다 BPM대 문턱으로, 아래는 --fine-bpm 0일 때): 곡 전체에서 옆 박도 같은 계열인 셋잇단·32분 박이 곡 박의 4%% 이상일 때만 그 계열을 쓰고 "
                         "혼자 떨어진 박은 16분으로 (barijung 140: 셋잇단 2.8%% / 32분 1.2%% -> 16분 폭타로, DIAVOLO·REDRED 셋잇단 12~16%% -> 남김). "
                         "격자 보정(--snap-fix)도 그 박 격자 안에서만. 0 = 16분만(--fill32도 끔), 1 = 자동(기본), 0~1은 더 엄하게, 1~2는 느슨하게, 2 = 다 허용(박마다 고른 분할을 거르지 않음)")
    ap.add_argument("--dens-flat", type=float, default=1.0,
                    help="소리 폭 맞춤(patch108): 구간 밀도 분포(--dens-shape)의 목표를 곡 소리 폭(마디 소리 크기 10~90%% 폭)이 비슷한 사람 곡 분포로 - "
                         "소리 편차가 적은 곡은 가벼운 구간도 덜 가볍게(사람 소리 폭 0.065 미만 곡: 2마디 묶음 하위 10%% 최고의 0.38배, 0.12 이상 곡 0.18배) "
                         "+ 줄보다 건반부터 덜어 듬성듬성 대신 조금 쉬는 폭타로 (REDRED 소리 폭 0.057). 1 = 켬(기본), 0 = 소리 폭과 상관없이 모든 곡에 같은 분포(하위 10%% 묶음 0.22배), 줄 빼기 비중도 그대로")
    ap.add_argument("--bass-sound", type=float, default=0.4,
                    help="소리에 베이스 넣기(patch109): 마디 소리 크기 순위 = 멜 전체 (1-값) + 저음(멜 0~10번 대역) 값 (곡 안 z값). "
                         "폭타 구간 고르기·악절 종류·사다리 단계·반쯤 빈 마디 채우기·쉬는 동시치기 크기에 씀 "
                         "(EOS 029~036 멜 순위 0.36 / 저음 0.85 - 베이스 16분이 큰데 1+1 얇은 폭타였음 -> 0.49~0.50 = 2+1, "
                         "사람 277개 0.3~0.4가 가장 잘 맞음: 폭타 마디 가르기 AUC 0.767 -> 0.775). 0 = 멜 전체만")
    ap.add_argument("--dens-dev", type=float, default=1.0,
                    help="편차 기준(patch109): 구간 밀도 분포(--dens-shape) 목표를 곡 안 순위 대신 '곡 큰 소리보다 얼마나 작은가'로 "
                         "(사람 277개 묶음 밀도 예측 오차 순위 0.176 / 소리 폭 0.169 / 편차 0.156, 큰 소리 근처는 순위 기준 목표보다 안 낮춤). 1 = 켬(기본), 0 = 끔(곡 안 순위로 목표)")
    ap.add_argument("--dens-pulse", type=float, default=1.0,
                    help="8분 박 지키기(patch109): 분리 소리(저장해 둔 <이름>_stems.npz) 드럼·베이스·악기에 8분 박 어택이 6/8칸+인 마디는 "
                         "구간 밀도 분포가 8분 뒷자리 줄을 통째로 빼지 않고 건반만 덞 (EOS M106 시작 4분 단타 -> 8분). 1 = 켬(기본), 0 = 끔")
    ap.add_argument("--fill32-sound", type=float, default=0.0,
                    help="채움 소리 확인(patch110): 16비트 이상 채우기(--fill32)가 더할 자리에 분리 소리(저장해 둔 <이름>_stems.npz) 드럼·베이스·악기·보컬 중 "
                         "어느 층에든 어택(주변 대비 이 값 배 이상)이 있을 때만 더함 - M109 채움 노트 중 어택 있는 비율 NEON 29%%·DIAVOLO 46%% "
                         "(일반 노트 67~98%%, 큰 소리 바로 뒤 잔향 같은 봉우리). 높을수록 엄격, 권장 1.5, 0 = 끔(기본, 확인 안 함 - 사용자: 선택해서 쓰게)")
    ap.add_argument("--chord32", type=float, default=0.0,
                    help="약한 32비트 동시치기로(patch110): 32분 뒷자리(박의 1/8·3/8·5/8·7/8) 줄 중 모델 확률이 이 값 미만(약하게 들린 즈레)인 것을 "
                         "바로 옆 16분 줄(8분 자리 우선)에 합쳐 동시치기로 - 또렷한 롤은 그대로, 줄만 합치고 건반 수는 그대로, 떨림(--trem) 줄은 그대로 "
                         "(사용자·프로게이머 자문: 일부 즈레가 어색, 그렇게 안 들리는 소리에 즈레). 권장 0.45(= 노트 기준), 1 = 32분 전부, 0 = 끔(기본 - 사용자: 선택해서 쓰게)")
    ap.add_argument("--rest-chord", type=float, default=1.0,
                    help="쉬는 동시치기 두께(patch108): 소리 큰 곳(곡 소리 순위 0.6+)에서 폭타를 끊거나 바꾼 8비트 동시치기 마디의 건반 수를 "
                         "주변 폭타 마디 x 이 값으로 = 줄마다 주변 폭타 쌍 합(8분+16분 건반 합) x 값 (사용자: 쌍 합 5.5 폭타면 동시치기 최소 4, 평균 5~6 / "
                         "사람 소리 큰 곳 폭타 옆 동시치기 쌍 합 대비 중앙 0.86, 상위 25%% 약 1). 동시치기 상한(--max-chord)은 그대로. "
                         "그 아래 소리는 주변 폭타의 0.76배. 기본 1, 0 = 소리 크기와 상관없이 0.76배")
    ap.add_argument("--fine-bpm", type=float, default=1.0,
                    help="24·32비트 BPM대 문턱(patch107): --fine-grid가 자동(0~2 사이)일 때, 곡 전체 4%% 문턱 대신 박마다 16분 대비 이득이 "
                         "BPM대 문턱 이상인 박만 셋잇단·32분으로 (노래를 따라가고 억지로 채우지 않음). 분할 후보 벌점과 문턱을 사람 277개 BPM대 쓰임 "
                         "(150~180 대비 32비트 느린 곡 1.3x / 200+ 0.12~0.25x, 셋잇단16분 느린 곡 2x, 셋잇단8분 200+ 1.8~3x)에 맞춤. "
                         "1 = 사람 기준(기본), 0 = 곡 전체 4%% 문턱(--fine-grid 설명: 이어진 셋잇단·32분 박이 곡 박의 4%% 이상인 계열만)")
    ap.add_argument("--dens-shape", type=float, default=1.0,
                    help="구간 밀도 분포: 2마디 묶음 초당 노트를 곡 최고 대비 사람 분포(lv15~22 179곡 중앙값: 하위 10%% 0.22 / 중앙 0.72 / 상위 25%% 0.89)로, "
                         "넘친 묶음만 줄임 - 폭타 마디는 안 건드리고(두께는 --stream-lift), 폭타 아닌 곳에서 조용한 묶음은 줄(박 단위 16분 뒷자리), 센 묶음은 건반 위주. "
                         "1 = 사람 기준(기본), 0 = 끔, 0~1은 넘친 만큼의 일부만")
    ap.add_argument("--dens-cap", type=float, default=0.0,
                    help="구간 밀도 분포의 최고 밀도 상한(초당 노트, 스크래치 포함). 이보다 진한 묶음은 여기까지 (사람 최고 중앙값 31, 상위 25%% 33, "
                         "p94 Chronomia·Cthugha 40). 기본 0 = 상한 없음 (사용자: 최고 밀도는 건들지 마)")
    ap.add_argument("--stream-lift", type=float, default=1.0,
                    help="폭타 두께 사람 분포로: 폭타 마디(16분 칸 12/16+)를 곡 안 순위대로 같은 BPM대 사람 줄당 노트 분포(150 이하 얇은 1.81 / 중간 2.73, "
                         "150~180 1.57 / 2.44, 180~200 1.67 / 2.25, 200+ 1.49 / 2.00)까지 올림, 건반만 더하고 빼지 않음. 상한 = 그 곡 지금 상위 10%% 두께·마디 노트 수 "
                         "(가장 진한 폭타·최고 밀도는 그대로). 쌍 합 평균이 4보다 낮으면 그 비율만큼. 1 = 사람 기준(기본), 0 = 끔")
    ap.add_argument("--guard", type=float, default=0.0,
                    help="울타리 모드(patch97 실험): 밀도·두께 규칙(--on8-match, --off-match, --pair-avg 4, --stream-lift, --dens-shape)이 "
                         "사람 중앙값을 목표로 하지 않고 사람 lv15~22 곡별 값의 10~90%% 범위를 벗어난 만큼만 고침 (범위 안이면 그 곡 개성으로 둠). 1 = 켬, 0 = 끔(기본)")
    ap.add_argument("--jack8", type=float, default=1.0,
                    help="8분 간격 겹침 줄이기(patch98): 바로 앞 8분 줄과 같은 건반이 같은 BPM대 사람 울타리(곡별 75%% 지점: 겹침 줄 ~150 53%%, "
                         "150~180 46%%, 180~200 48%%, 200+ 42%%, 줄당 겹친 건반 0.95/0.84/0.73/0.63)를 넘으면 넘친 만큼 겹친 건반만 안 겹치는 자리로 "
                         "(건반 수 그대로), 16분 간격 같은 건반 3줄 연속도 사람 중앙값(16분 줄 1000개당 ~150 7.7곳, 165+ 0)까지 풂. "
                         "자리 모델 뒤 마지막에 함. 1 = 사람 울타리(기본), 0 = 끔")
    ap.add_argument("--trem", type=float, default=0.0,
                    help="떨림(드르릉) 살리기(patch99): 1단계 모델 확률 봉우리(0.15+)가 초당 16/값 개 이상이고 16분 칸 밖 봉우리가 2개 이상인 박이 "
                         "2박 넘게 이어지면 그 박을 떨림 속도 격자(봉우리 위치가 더 잘 맞는 24분 또는 32분, 간격 40ms 미만이면 24분)로 다 채우고 "
                         "격자 밖 줄은 뺌, 구간 밀도 분포(--dens-shape)는 안 뺌, 마지막에 사람 24·32분 연속 모양으로 놓고 줄 두께는 사람 떨림 구간처럼 "
                         "(patch105: 박 머리 단타/2/3+ 34/35/31%%, 16분 칸 60/30/10%%, 사이 88/11/1%%). "
                         "치고 쉬기로 비운 박은 건너뜀. DIAVOLO 드르릉·고음 떨림이 임계값 아래(0.2~0.4)라 빠지던 것. 0 = 끔(기본), 1 = 기준")
    ap.add_argument("--shape-run", type=float, default=1.0,
                    help="같은 모양 이어짐 끊기(patch101): 16분 폭타 안 이웃 동시치기가 같은 모양(평행이동 무시, 예: 36 57 46 35)으로 3줄 넘게 이어지는 "
                         "사슬이 폭타 1000줄당 사람 중앙값(9.6곳)보다 많으면 셋째 줄부터, 앞 줄과 같은 모양 비율이 사람 75%% 곡(21%%)보다 높으면 "
                         "넘친 만큼 모양을 바꿈 (건반 수 그대로, 연타 없이, 원래 건반을 될 수 있으면 남김, 자리 모델 뒤). 1 = 사람 기준(기본), 0 = 끔")
    ap.add_argument("--row-seed", type=float, default=1.0,
                    help="줄마다 고정 난수(patch102): 레인 모델·자리 모델이 줄마다 그 줄 시각으로 정한 난수를 씀 -> 한 곳을 고쳐도 "
                         "다른 줄의 뽑기는 그대로라 나머지 배치가 통째로 다시 뽑히지 않음. 1 = 켬(기본), 0 = 곡 전체에 난수 하나를 이어 씀(줄 하나가 바뀌면 뒤 배치가 다시 뽑힘)")
    ap.add_argument("--jack-allow", type=float, default=0.3,
                    help="낮은 BPM 따닥 허용(patch103/103b): BPM 150 아래, 16분이 --no-jack-ms보다 길 때 빽빽한 폭타 마디(건반 수 상위 25%%)의 "
                         "이웃 16분 줄 같은 건반 비율을 BPM ±15·줄당 건반 ±0.4 비슷한 사람 곡의 '무작위 대비 겹침' 중앙값 쪽으로 이 세기만큼 올림. "
                         "같은 소리가 반복되는 쌍(곡 안 소리 유사도 위쪽 절반)에만, 비슷한 쌍부터 (뒷 줄 건반 하나를 앞 줄 건반으로, 건반 수 그대로, "
                         "4번 연속 금지, 3번 연속은 사람 비율까지). 1 = 비슷한 사람 곡 중앙값, 0.3 = 기본(patch103c, 사용자가 영상 보고 0.5보다 약하게), 0 = 끔")
    ap.add_argument("--mel-keep", type=float, default=1.0,
                    help="멜로디 지키기(patch104): (1) 구간 밀도 분포(--dens-shape)가 멜로디·보컬·베이스 어택이 센 줄(곡 안 위쪽 40%% x 값)을 안 빼고 "
                         "드럼 엇박 줄·큰 동시치기 건반에서 먼저 덜어냄, (2) 폭타 구간 바꾸기(--stream-target)에서 보컬·악기 비중이 곡 안 위쪽 40%% x 값인 "
                         "4마디는 8비트 단타·동시치기 대신 반쯤 빈 16비트로(--ladder 켜면 16분 줄 그대로인 사다리 폭타로). 둘 다 저장해 둔 음원 분리 결과 <이름>_stems.npz가 있을 때만 "
                         "(악기 따라가기 --follow를 한 번 쓰면 생김). 1 = 기본, 0 = 끔")
    ap.add_argument("--ladder", type=float, default=1.0,
                    help="폭타 사다리(patch106): 폭타 구간 바꾸기(--stream-target)가 소리 순위 30%% 위 4마디를 8비트 동시치기·반쯤 빈 16비트 대신 "
                         "16분 줄은 그대로 두고 두께만 줄인 사다리 폭타(1+1 / 2+1 / 2+2, 소리 순위 0.30~0.45 / 0.45~0.65 / 그 위, 주변 폭타보다 한 단계+ 아래)로, "
                         "20~30%%는 8비트 단타, 하위 20%%는 드문드문. 16분 소리가 곡 폭타 중 하위 25%%인 4마디만 반쯤 빈 16비트(뒷자리 1~3개, 사람 분포) "
                         "또는 곡 상위 30%%면 8비트 동시치기. 폭타 끊기(--stream-break)도 8비트 동시치기 대신 주변보다 두 단계 아래 사다리로 (줄은 그대로). "
                         "1 = 켬(기본), 0 = 사다리 안 씀: 폭타 구간 바꾸기는 소리 순위 하위 25%% 드문드문 / 25~40%% 8비트 단타 / 그 위 8비트 동시치기"
                         "(상위 40%%이면서 16분 소리가 곡 폭타 중앙값 이상이면 반쯤 빈 16비트), 폭타 끊기는 8비트 동시치기")
    ap.add_argument("--half-fill", type=float, default=0.30,
                    help="소리 큰 곳 반쯤 빈 마디 채우기(patch107): 곡 소리 상위 50%% 마디 중 반쯤 빈 16비트(8분 자리 6줄+, 16분 뒷자리 1~5)이면서 "
                         "앞이나 뒤가 폭타인 곳은 빈 16분 뒷자리를 모델 확률이 이 값 이상이면 채움 (노트 기준 0.45 아래로 아깝게 빠진 소리, "
                         "M105 소리 상위 반쯤 빈 마디 116개 빠진 뒷자리 확률 중앙 0.30, 0.30이면 37마디가 폭타로). 폭타 구간 바꾸기·끊기가 고른 마디는 그대로. "
                         "기본 0.30, 0 = 끔")
    ap.add_argument("--four-one", type=float, default=1.0,
                    help="4+1 섞기(patch107): 4 단계 폭타(8분 자리 평균 3.5+) 중 4+1 비율을 사람(4+1 / (4+1 + 4+2) 35%%) x 값까지 - 4+2 마디 중 "
                         "소리 작은 것부터 16분 뒷자리 줄을 1개씩으로, 폭타 쌍 합 평균(--pair-avg)이 그 마디에 정한 목표 쌍 합이 5 이하일 때만 - 쌍 합 평균을 올리면 소리 큰 묶음의 4+2는 그대로 남음 (사람 277: 4+1 30 / 4+2 55 / 4+3+ 15%%, 생성 M105 4 / 93 / 3%%). 1 = 사람 비율(기본), 0 = 끔")
    ap.add_argument("--key-scale", type=float, default=1.0,
                    help="건반 배율: 줄(리듬)은 그대로 두고 동시치기 크기만 이 배로 (1 = 그대로, 0.8 = 건반 20%% 덜, 1.2 = 20%% 더). "
                         "줄일 때 단타는 그대로, 동시치기의 '1개 넘는 몫'을 같은 비율로 덜고, 늘릴 때는 연타를 피해 더함. 치고 쉬기 큰 소리 줄은 그대로. 자리 모델 앞에서 함")
    ap.add_argument("--fill-grade", type=float, default=1.0,
                    help="폭타 잇기(--stream-fill)를 단계식으로: 소리 큰 마디만 16칸까지, 나머지는 14~15 / 12~13칸까지 "
                         "(사람 12칸+ 마디 16칸 64 / 14~15 17 / 12~13 19%%, 기본 1, 0이면 전부 16칸까지 채움)")
    ap.add_argument("--placer-beat", type=float, default=3.0,
                    help="자리 모델이 정확히 한 박 전 줄과 같은 조합을 고르는 것 억제 세기: 최근 같은 두께 줄 중 그 비율이 사람"
                         "(얇은 폭타 14.5%%, 두꺼운 폭타 7.5%%)보다 높으면 그 조합을 덜 고름, patch92부터 두 줄 전과 같은 조합도(사람 6%%), patch94부터 16분 단타 계단(1234, 사람 6%% / 생성 EOS p93 50%%)도 (생성 EOS 40%% / 16%%, 기본 3, 0이면 끔)")
    ap.add_argument("--stream-fill", type=float, default=11.0,
                    help="16분 칸이 이 수(16칸 중) 이상 찬 마디는 빈 16분 칸을 채워 폭타를 끊기지 않게 "
                         "(사람 lv20 바쁜 마디 16칸 다 참 48%%, 생성 p71 3%%, 기본 11, 0이면 끔)")
    ap.add_argument("--beat-motif", type=float, default=1.0,
                    help="리듬이 같은 앞 박의 줄별 건반 수를 따라가 박 단위 반복을 만드는 세기 (사람 1박 전 같음 27~32%%, 일부는 줄 모양까지). 0이면 끔")
    ap.add_argument("--eighth-shape", type=float, default=1.0,
                    help="8분 자리 줄을 두 줄 전 그대로(A B A B)나 앞 줄·두 줄 전 모양을 옆으로 옮겨 쓰는 세기 "
                         "(사람 두 줄 전 같음 12%%, 같은 모양 15~28%%, 건반 수는 그대로). 0이면 끔")
    ap.add_argument("--note-log", type=int, default=1,
                    help="노트 후보마다 확률·음원 소리세기·어느 단계에서 빠졌는지를 <이름>_notes_debug.csv로 남김 (기본 1, 0이면 끔)")
    ap.add_argument("--fill-axis", type=float, default=1.0,
                    help="32·24분 채운 단타가 1박 안 다른 채운 단타와 같은 건반이면(축) 이 확률로 가장 가까운 다른 건반으로. 0이면 끔")
    ap.add_argument("--restart-shift", type=float, default=1.0,
                    help="16분 연타가 1/2~1박 쉬고 다시 시작할 때 시작 줄이 끝 줄과 건반을 나눠 쓰는 비율을 사람(49%%)으로 낮추는 세기. 0이면 끔")
    ap.add_argument("--beat-pair", type=float, default=1.0,
                    help="박 안 두 8분 자리(16분 연타 속) 건반 수를 사람 비율(같음 56 / 첫째가 큼 29 / 둘째가 큼 15%%)로: "
                         "넘치는 쌍의 둘째 줄을 첫째와 같게 하는 세기. 0이면 끔")
    ap.add_argument("--edge4", type=float, default=1.0,
                    help="1·7번이 없고 붙은 쌍이 둘 이상인 4건반(2356·2346·2456·2345·3456)이 4건반 줄 중 사람(7%%)보다 많으면 "
                         "넘는 만큼 x 이 값의 줄에서 끝 건반을 1번이나 7번으로 옮김 (2356 -> 1356 / 2357). 0이면 끔")
    ap.add_argument("--hand32", type=float, default=1.0,
                    help="32분 사이 칸 줄이 앞뒤 32분 이웃 줄이 더 많이 쓰는 손(1~3번 왼손 / 5~7번 오른손) 쪽에만 있으면 "
                         "이 확률로 반대 손 건반으로 옮김 (4번은 가운데로 안 셈, 연타 금지 지킴). 0이면 끔")
    ap.add_argument("--off-match", type=float, default=1.0,
                    help="16분 연타 속 뒷자리(16분 사이 칸) 줄 건반 수를 lv20 분포(1개 36%% / 2개 47%% / 3개 15%%)로 올림. "
                         "양옆이 큰 동시치기인 단타부터 (큰동치-단타 교대 줄임). 0이면 끔")
    ap.add_argument("--on8-match", type=float, default=1.0,
                    help="16분 연타 속 8분 자리 줄 건반 수를 lv20 분포(1개 4%% / 2개 24%% / 3개 44%% / 4개 24%% / 5+ 4%%)로 내림. "
                         "많은 쪽에서만 하나씩 뺌 (1 = 분포 그대로, 0.5 = 절반만, 0 = 끔). 큰동치-단타 교대가 몰리는 것 방지")
    ap.add_argument("--stream-cap", type=float, default=0.25,
                    help="16분 연타 속 8분 자리(박·8분 뒷박)의 4건반 이상 동시치기에서 건반마다 이 확률로 덜어냄 "
                         "(덜 개수는 정하지 않음, 최소 1건반 남김, 축 건반부터). 0이면 끔")
    ap.add_argument("--eighth-same", type=float, default=0.6,
                    help="8비트 동시치기 구간에서 앞뒤 줄과 같은 소리인데 겹친 건반도 이 확률로 옮김. "
                         "lv20 기준: 같은 소리여도 겹침 30%%. 0이면 같은 소리 반복은 그대로")
    ap.add_argument("--pillar-fix", type=float, default=0.3,
                    help="16분 연타 속 8분 자리 줄이 8분 전/1박 전 줄과 같은 건반(1 _ 1 _ 기둥)이면 이 확률로 빈 건반으로 옮김. "
                         "사람: 8분 전 줄과 겹침 56%%, 생성 flicknote 88%%. 0이면 끔")
    ap.add_argument("--eighth-overlap", type=float, default=0.7,
                    help="8비트 동시치기 구간(16분 없는 8분)에서 앞뒤 줄과 겹치는 건반을 이 확률로 겹치지 않는 자리로. "
                         "사람: 겹침 24%%, 생성 flicknote 53%%. 0이면 끔")
    ap.add_argument("--eighth-big", type=float, default=0.0,
                    help="8비트 동시치기 구간에서 이 확률로 건반을 하나 더 (또 이 확률이면 하나 더, --max-chord까지). "
                         "앞뒤 줄에 없는 건반만 써서 1357 / 246처럼. 0이면 끔")
    ap.add_argument("--count-smooth", type=float, default=0.8,
                    help="16분 연타 속 8분 자리 동시치기 건반 수가 양옆보다 2개 이상 튀거나(4 1 3) 바로 앞과 2개 넘게 차이 나면 "
                         "이 확률로 ±1 안으로 (건반 하나씩 더하거나 뺌). 사람: ±2 이상 점프 14%%, 사이 단타 3%%. 0이면 끔")
    ap.add_argument("--count-keep", type=float, default=0.5,
                    help="반 마디 동안 8분 자리 동시치기 개수를 이 확률로 이어 감 (건반 하나씩만 더하거나 빼서 모델 자리는 유지). 0이면 끔")
    ap.add_argument("--scratch-model", default="",
                    help="스크래치 자리 모델 (기본: infer.py 옆 checkpoints/scratch_pos.npz가 있으면 그것). 'none'이면 안 씀")
    ap.add_argument("--scratch-place", type=float, default=1.0,
                    help="스크래치 자리 모델로 레인 모델의 스크래치를 다시 놓을 정도 (0~1, 기본 1 = 전부 다시). 개수는 그대로 두되 "
                         "--scratch-add가 더 많으면 그 수까지. 0이면 레인 모델이 고른 스크래치 자리 그대로, 모자라면 박 위치 가중치로 채움)")
    ap.add_argument("--scratch-add", type=float, default=0.9,
                    help="스크래치가 마디당 이 줄 수보다 적으면 저·중음이 센 줄에 채움 (lv20 평균 0.96, 기본 0.9). 0이면 끔")
    ap.add_argument("--chord-style", type=float, default=0.5,
                    help="반 마디마다 8분 자리 동시치기 성격(건반 수+간격)을 정하고 이 확률로 다음 반 마디에도 유지 (사람처럼 같은 모양 이어가다 전환). 0이면 끔")
    ap.add_argument("--spread-chords", type=float, default=0.3,
                    help="붙은 건반 동시치기(23, 56 등)를 이 확률로 벌림(23 -> 24). 연타 금지 지킴, 0이면 끔")
    ap.add_argument("--accent3-at", default="",
                    help="점8분(16분 3칸) 묶음으로 찍을 음원 구간(초), 예: \"39-41.5,100-102\". 강세 줄은 동시치기, 사이 줄은 단타")
    ap.add_argument("--accent3-phase", type=int, default=-1,
                    help="--accent3-at의 강세 위치(0~2). -1이면 저·중음 어택으로 자동. 묶음이 한 칸 밀려 보이면 다른 값으로")
    ap.add_argument("--accent3-auto", type=float, default=0.0,
                    help="점8분 묶음 자동 찾기 (0 = 끔, 0.5 = 기준). 실제 곡(Chronomia)에서는 하이햇 때문에 잘 못 찾아 기본 끔")
    ap.add_argument("--no-sparse-spread", action="store_true",
                    help="성긴 구간 단타가 한 건반(주로 4번)에 몰리는 것을 푸는 보정을 끈다")
    ap.add_argument("--no-section-fix", action="store_true",
                    help="--snap beat에서 반 마디 구간 정리(8비트 구간의 16분 사이 줄 합치기, 쉬는 구간 정리)를 끈다")
    ap.add_argument("--snap-min-ms", type=float, default=25.0,
                    help="--snap auto/beat에서 쓸 가장 촘촘한 격자의 최소 칸 간격(ms). BPM 180이면 32분(42ms)까지, 120이면 64분(31ms)까지")
    ap.add_argument("--dynamics", type=float, default=0.0,
                    help="셈여림 반영 세기: 소리 큰 구간은 노트/동시치기를 더 많이, 조용한 구간은 더 적게 "
                         "(0=끔, 1=기본, 2=강하게)")
    ap.add_argument("--shift-ms", type=float, default=0.0,
                    help="모든 노트를 이만큼 늦춤(ms, 음수면 당김). 싱크가 일정하게 밀릴 때 보정")
    ap.add_argument("--ref", default=None,
                    help="정답 채보(.bms/.bme). 같은 오디오일 때만 의미 있음. 예측 시각이 정답과 얼마나 어긋나는지 출력")
    ap.add_argument("--ref-audio", default=None,
                    help="정답 채보의 오디오(키음 믹스 wav). 주면 입력 오디오와의 시작 위치 차이를 자동으로 찾아 "
                         "정답 시각을 맞춘 뒤 비교 (mp3를 키음 차분으로 채점할 때)")
    ap.add_argument("--ref-shift-ms", type=float, default=None,
                    help="정답 시각을 이만큼 늦춰서 비교(ms). --ref-audio 대신 직접 지정")
    args = ap.parse_args()
    if str(args.follow).strip().lower() not in ("", "off") or str(args.follow_at).strip():
        from stem_follow import mode_word, parse_follow_at
        mode_word(args.follow)
        parse_follow_at(args.follow_at)                  # 오타는 오래 걸리는 생성 전에 바로 알림
        if not str(args.snap).startswith("beat"):
            print("[주의] --follow / --follow-at은 --snap beat일 때만 동작해요 (마디 번호를 맞추려고)")
    if str(args.placer_ckpt).lower() == "none":
        args.placer_ckpt = ""
    elif not args.placer_ckpt:
        _pd = Path(__file__).resolve().parent / "checkpoints" / "placer_v1.pt"
        args.placer_ckpt = str(_pd) if _pd.exists() else ""
    if str(args.scratch_model).lower() == "none":
        args.scratch_model = ""
    elif not args.scratch_model:
        dflt = Path(__file__).resolve().parent / "checkpoints" / "scratch_pos.npz"
        args.scratch_model = str(dflt) if dflt.exists() else ""
    if args.raw_lanes:
        args.thick = args.spread_chords = args.chord_style = args.beat_contrast = 0.0
        args.denim_max = args.stream_end_chord = 0
        args.count_keep = args.scratch_add = args.flow = args.count_smooth = args.thin_boost = args.scratch_pos = 0.0
        args.stream_cap = args.pillar_fix = args.eighth_overlap = args.eighth_big = args.on8_match = args.hand32 = args.trip_match = 0.0
        args.skel_swap = args.bar_repeat = args.diag = 0.0
        args.shape_run = args.jack_allow = args.mel_keep = args.ladder = args.half_fill = args.four_one = 0.0
        args.fine_single = args.fine_shape = args.beat_pair = args.edge4 = 0.0
        args.fill_axis = args.restart_shift = args.section_dyn = args.eighth_trans = args.end_fat = args.end_shape = args.superset_fix = args.beat_motif = args.eighth_shape = args.edge_axis = args.pair_even = args.placer = args.quarter_spread = args.trill_break = args.eighth_audit = args.hand_balance = args.fine_density = 0.0
        args.stream_break = 0
        args.stream_target = args.stream_kpr = args.fill_grade = args.accent_rest = 0.0
        args.key_scale = 1.0
        args.dens_shape = args.stream_lift = 0.0
        args.scratch_place = 0.0
        args.eighth_same = args.off_match = 0.0
        args.no_sparse_spread = True
        args.accent3_auto, args.accent3_at = 0.0, ""
        print("확인용: 레인 후처리 규칙을 모두 껐어요 (--raw-lanes)")

    snap_mode = args.snap.lower()
    if snap_mode == "beat":
        snap = ("beat",)
    else:
        snap = None if snap_mode in ("off", "auto") else tuple(int(x) for x in args.snap.split(","))
    if args.total <= 0:
        raise SystemExit("--total은 0보다 큰 숫자로 입력하세요.")
    if args.bpm is not None and not 30 <= args.bpm <= 400:
        raise SystemExit("BPM은 30~400 사이로 입력하세요.")

    audio_path = Path(args.audio)
    name = safe_name(audio_path.stem)
    stereo, sr0 = read_audio(audio_path)
    print(f"오디오: {audio_path.name}  {len(stereo) / sr0:.1f}초, {sr0}Hz")

    predict_fn, ck, device = load_model(args.ckpt)
    thr = args.threshold if args.threshold is not None else float(ck["threshold"])
    m = ck.get("metrics", {})
    print(f"모델: 장치 {device}, 학습 step {ck.get('step')}, 임계값 {thr:.2f}"
          + (f", 학습 때 F1@50ms {m['union_f1_t5']:.3f}" if "union_f1_t5" in m else ""))

    if ck["args"].get("ctx") and args.window_sec < ck["args"]["window_sec"]:
        args.window_sec = float(ck["args"]["window_sec"])   # 문맥 모델은 학습 때 윈도우 길이 이상으로 본다
        print(f"문맥 모델: 예측 윈도우 {args.window_sec:g}초")
    # ---- STEP 1 곡 변환: 음원 -> 멜 스펙트로그램 ----
    mel = mel_from_audio(stereo, sr0)
    # ---- STEP 2 노트 시각 모델: 10ms마다 노트 확률 ----
    probs = stitch_windows(mel, predict_fn, int(args.window_sec * FPS), int(args.overlap_sec * FPS))
    if args.dump_probs:
        Path(args.dump_probs).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_probs, probs.astype(np.float32))
        print(f"확인용: 1단계 확신도(프레임 100fps) 저장 -> {args.dump_probs}")
    # BPM을 주지 않으면 노트 확률의 봉우리 간격으로 추정 (STEP 1에 해당)
    bpm = args.bpm
    if bpm is None:
        from bpm_estimate import estimate_bpm, format_result
        est = estimate_bpm(peak_pick(probs, thr, width=1) / FPS, prefer=tuple(args.bpm_range))
        print(format_result(est))
        if est["bpm"] is None:
            raise SystemExit("BPM을 추정하지 못했어요. --bpm 으로 직접 지정하세요.")
        bpm = float(est["bpm"])
    if snap_mode == "auto":
        snap = ("hier",) + snap_levels(bpm, args.snap_min_ms)
    from dynamics import density_loudness_corr, loudness_curve, threshold_curve
    loud = loudness_curve(mel)
    thr_used = thr
    if args.dynamics > 0:
        thr_used = threshold_curve(thr, loud, args.dynamics)
        print(f"셈여림 반영: 세기 {args.dynamics:g} | 임계값 {float(thr_used.min()):.2f}(큰 구간) ~ "
              f"{float(thr_used.max()):.2f}(조용한 구간), 기준 {thr:.2f}")
    # ---- STEP 4 준비: 패턴 계획 모델과 레인 모델 불러오기 ----
    lane_fn = None
    if args.lane_ckpt:
        lane_fn, lck = load_lane_fn(args.lane_ckpt, mel, args.density, args.temperature,
                                    args.max_chord, args.no_jack_ms, args.seed,
                                    loud=loud, dynamics=args.dynamics,
                                    jack_penalty=args.jack_penalty,
                                    jack_window_ms=args.jack_beats * 60000.0 / bpm * 1.05,
                                    scratch_bias=args.scratch_bias,
                                    coverage=args.coverage, stream_ms=60000.0 / bpm / 4 * 1.3,
                                    same_sound=args.same_sound, same_window_ms=args.same_beats * 60000.0 / bpm,
                                    lone_scratch=args.lone_scratch,
                                    plan_ckpt=args.plan_ckpt, plan_temp=args.plan_temp,
                                    alt_scale=args.alt_scale, max_alt_rows=args.max_alt_rows,
                                    vary=args.vary, scratch_pos=args.scratch_pos, row_seed=args.row_seed > 0)
        lm = lck.get("metrics", {})
        print(f"레인 모델{' v2' if lck['args'].get('arch') == 'v2' else ''}: 학습 step {lck.get('step')}"
              + (f", 정확도 {lm['acc']:.3f}" if "acc" in lm else "")
              + ("" if lck["args"].get("quality") else f" | 밀도 {args.density:g}")
              + f" | 온도 {args.temperature:g}, 동시치기 상한 {args.max_chord}"
              + (f" | 같은 소리 같은 자리 {args.same_sound:g} (범위 {args.same_beats:g}박)" if args.same_sound > 0 else "")
              + (" | 줄마다 고정 난수 켬(--row-seed: 레인·자리 모델 뽑기를 줄 시각으로 정함)" if args.row_seed > 0 else ""))
    density_fn = None
    if args.density_ckpt:
        import torch
        from density_model import count_from_probs, load_density, predict_density
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dmodel, dck = load_density(args.density_ckpt, dev)
        print(f"밀도 모델: {args.density_ckpt} (학습 step {dck.get('step')}, 기준 {args.density_q:g})")

        def density_fn(mel_, st, en, half):
            pr, pb = predict_density(dmodel, mel_, st, en, half, dev, 1.0)
            p16 = np.asarray(pr)[:, 6:].sum(axis=1)          # 16비트(6줄 이상)일 확률
            return count_from_probs(pr, args.density_q), count_from_probs(pb, args.density_q), p16
        density_fn.conf = args.density_conf
    # ---- STEP 3 ~ 7: chart_from_probs 안에서 차례로 진행 ----
    info = chart_from_probs(probs, stereo, sr0, Path(args.out) / (args.folder or name), name, bpm, thr_used,
                            offset=args.offset, lane_mode=args.lane_mode, tol_ms=args.tol_ms,
                            title=args.title, clicks=not args.no_clicks, seed=args.seed,
                            key_sound=args.key_sound, lane_fn=lane_fn, shift_ms=args.shift_ms,
                            snap=snap, snap_min_ms=args.snap_min_ms,
                            beat_divs=tuple(int(x) for x in args.beat_divs.split(",")),
                            offgrid_keep=args.offgrid_keep, thin_weak=args.thin_weak,
                            section_fix=not args.no_section_fix, density_fn=density_fn,
                            sparse_spread=not args.no_sparse_spread, keep_sure=args.keep_sure,
                            end_chord=args.stream_end_chord,
                            accent3=args.accent3_auto, accent3_at=args.accent3_at, accent3_phase=args.accent3_phase,
                            thick=args.thick, thick_range=tuple(float(x) for x in args.thick_range.split(",")),
                            spread=args.spread_chords, chord_style=args.chord_style, beat_contrast=args.beat_contrast, denim_max=args.denim_max, alt_rows=0 if args.raw_lanes else args.max_alt_rows, random_lanes=args.random_lanes, count_keep=args.count_keep, scratch_add=args.scratch_add, flow=args.flow, no_jack_ms=args.no_jack_ms, fill32_low=args.fill32, count_smooth=args.count_smooth, thin_boost=args.thin_boost,
                            scratch_pos=args.scratch_pos, stream_cap=args.stream_cap, pillar_fix=args.pillar_fix,
                            eighth_overlap=args.eighth_overlap, eighth_big=args.eighth_big, max_chord=args.max_chord,
                            on8_match=args.on8_match, eighth_same=args.eighth_same, off_match=args.off_match, hand32=args.hand32, jack16=args.jack16, trip_match=args.trip_match,
                            skel_swap=args.skel_swap, bar_repeat=args.bar_repeat, bar_shape=args.bar_shape, diag=args.diag,
                            fine_single=args.fine_single, fine_shape=args.fine_shape, beat_pair=args.beat_pair, edge4=args.edge4,
                            fill_axis=args.fill_axis, restart_shift=args.restart_shift, section_dyn=args.section_dyn,
                            eighth_trans=args.eighth_trans, end_fat=args.end_fat, end_over=args.end_over,
                            follow=args.follow, follow_at=args.follow_at, follow_strength=args.follow_strength,
                            stems_model=args.stems_model, stems_device=args.stems_device,
                            scratch_model=args.scratch_model, scratch_place=args.scratch_place,
                            chart_name=args.chart_name, artist=args.artist,
                            snap_fix_ms=args.snap_fix, thin_local=args.thin_local, note_log=bool(args.note_log),
                            stream_hole=args.stream_hole, end_shape=args.end_shape, superset_fix=args.superset_fix,
                            beat_motif=args.beat_motif, eighth_shape=args.eighth_shape, stream_fill=args.stream_fill,
                            edge_axis=args.edge_axis, pair_even=args.pair_even, pair_min=args.pair_min,
                            placer_ckpt=args.placer_ckpt, placer=args.placer, placer_temp=args.placer_temp,
                            pair_avg=args.pair_avg,
                            pair_dev=(args.pair_spread / 2.0 if args.pair_spread is not None else args.pair_dev),
                            quarter_spread=args.quarter_spread,
                            trill_break=args.trill_break,
                            eighth_audit=args.eighth_audit,
                            hand_balance=args.hand_balance, fine_density=args.fine_density,
                            placer_rep=args.placer_rep,
                            stream_break=args.stream_break,
                            placer_beat=args.placer_beat,
                            stream_target=args.stream_target, stream_kpr=args.stream_kpr,
                            fill_grade=args.fill_grade, accent_rest=args.accent_rest, pair_keep=args.pair_keep,
                            fine_grid=args.fine_grid, key_scale=args.key_scale, accent_rest_len=args.accent_rest_len,
                            dens_shape=args.dens_shape, dens_cap=args.dens_cap, stream_lift=args.stream_lift, guard=args.guard,
                            jack8=args.jack8, trem=args.trem, shape_run=args.shape_run, row_seed=args.row_seed,
                            jack_allow=args.jack_allow, mel_keep=args.mel_keep, ladder=args.ladder,
                            half_fill=args.half_fill, four_one=args.four_one, fine_bpm=args.fine_bpm,
                            rest_chord=args.rest_chord, dens_flat=args.dens_flat,
                            bass_sound=args.bass_sound, dens_dev=args.dens_dev, dens_pulse=args.dens_pulse,
                            chord32=args.chord32, fill32_sound=args.fill32_sound,
                            rank={"easy": 3, "normal": 2, "hard": 1, "veryhard": 0, "very_hard": 0}.get(args.judge, None)
                            if not args.judge.isdigit() else int(args.judge), total=args.total,
                            mel=mel, downbeat=args.downbeat)

    q, v = info["quant"], info["verify"]
    print(f"\n예측한 노트 시각 {info['n_notes']}개 -> 총 노트 {info['n_total']}개 "
          f"({info['n_total'] / info['duration']:.1f}개/초)")
    if lane_fn is not None:
        from lane_utils import chord_stats
        cs = chord_stats(np.array(info["masks"]))
        print(f"레인 배치: 평균 동시치기 {cs['mean_chord']:.2f}, 같은 레인 반복률 {cs['repeat_rate']:.2f}, "
              f"레인별 비율 {cs['lane_share']}")
    flag = "" if info["grid_R"] >= 0.7 else "  <- 낮아요. BPM이 맞는지 확인하세요"
    print(f"박자 격자 적합도 R = {info['grid_R']:.2f} (1에 가까울수록 BPM이 맞음){flag}")
    print(f"격자 위상 {info['phase_ms']:.1f}ms -> 오디오 앞에 무음 {info['pad_ms']:.1f}ms 추가")
    d = info["drift_ms"]
    trend = "  <- 앞뒤 차이가 커요. BPM이 미세하게 틀렸을 수 있어요" if abs(d[2] - d[0]) >= 8 else ""
    print(f"격자 대비 노트 위치(중앙값): 앞 {d[0]:+.1f}ms / 중간 {d[1]:+.1f}ms / 뒤 {d[2]:+.1f}ms{trend}")
    if args.shift_ms:
        print(f"노트 전체를 {args.shift_ms:+.0f}ms 옮겼어요 (--shift-ms)")
    sn = info["snap"]
    if sn:
        if sn["divs"][0] == "beat":
            label = "박 단위"
        else:
            dv = [d for d in sn["divs"] if d != "hier"]
            label = ("단계별 " if sn["divs"][0] == "hier" else "") + "/".join(f"{d}분" for d in dv)
        print(f"격자 맞춤({label}): 노트 시각 {sn['before']}개 -> {sn['after']}개 "
              f"(같은 칸으로 모여 합쳐진 것 {sn['before'] - sn['after']}개), "
              f"옮긴 거리 평균 {sn['mean_ms']:.1f}ms / 최대 {sn['max_ms']:.1f}ms")
        if sn.get("beat"):
            bi = sn["beat"]
            print(f"  박별 분할: {', '.join(f'{k} {v}박' for k, v in bi['beats'].items())} "
                  f"(후보: {'/'.join(bi['candidates'])})")
    print(f"양자화: 마디 {q['n_measures']}개, 분할 분포 {q['div_hist']}, "
          f"오차 평균 {q['mean_err_ms']:.1f}ms / 최대 {q['max_err_ms']:.1f}ms, 겹쳐서 제외 {q['dropped_dup']}개")
    print(f"재검증: BMS를 다시 읽은 결과 {'일치' if v['ok'] else '불일치!'} "
          f"({v['n_parsed']}/{v['n_expected']}개, 최대 차이 {v['max_diff_ms']:.4f}ms)")
    pc = [bin(int(m)).count("1") for m in info["masks"]]
    gen_frames = np.repeat(np.round(np.asarray(info["times"]) * FPS).astype(np.int64), pc)
    print(f"셈여림-물량 상관(2초 구간, 1에 가까울수록 소리 큰 곳에 노트가 많음): "
          f"{density_loudness_corr(gen_frames, loud):+.2f}")
    if args.ref:
        from bms_parser import parse_bms
        ref_times = np.unique(np.round([n.time for n in parse_bms(args.ref).notes], 3))
        ref_shift = 0.0
        if args.ref_shift_ms is not None:
            ref_shift = args.ref_shift_ms
            print(f"\n정답 시각을 {ref_shift:+.1f}ms 옮겨서 비교 (직접 지정)")
        elif args.ref_audio:
            from align import estimate_lag_ms, to_mono
            ra, rsr = read_audio(Path(args.ref_audio))
            ref_shift, conf = estimate_lag_ms(to_mono(stereo), sr0, to_mono(ra), rsr)
            print(f"\n오디오 정렬: 입력 오디오가 정답 오디오보다 {ref_shift:+.1f}ms "
                  f"{'늦게' if ref_shift >= 0 else '일찍'} 시작함 (신뢰도 {conf:.2f}"
                  f"{', 낮음 - 결과 주의' if conf < 1.2 else ''}) -> 정답 시각을 그만큼 옮겨서 비교")
        ref_times = ref_times + ref_shift / 1000.0
        ref_all = np.array([n.time for n in parse_bms(args.ref).notes]) + ref_shift / 1000.0
        print(f"정답 채보의 셈여림-물량 상관: "
              f"{density_loudness_corr(np.round(ref_all[ref_all >= 0] * FPS).astype(np.int64), loud):+.2f}")
        print("\n정답 채보와 비교 (+는 예측이 정답보다 늦음):")
        print_offset_report(offset_report(info["times"], ref_times, "모델 예측"))
        print_offset_report(offset_report(info["written_times"], ref_times, "BMS에 쓴 시각(양자화 후)"))
    print(f"\n저장: {info['bms']}")
    if info["clicks"]:
        print(f"미리듣기: {info['clicks']}")


if __name__ == "__main__":
    main()
