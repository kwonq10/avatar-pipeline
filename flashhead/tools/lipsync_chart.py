# リップシンクの確認用の資料：完成 MP4 の各フレームの「口の開き」と、音声の波形・音量を同じ時間軸に並べる（CPU だけ・モデルは使わない）
# 口の開き＝MediaPipe Face Mesh（mediapipe 同梱、ダウンロード不要）の唇の内側の上下（13・14）の距離 ÷ 口の横幅（61・291）
# 注意：これは目安。音量との相関だけでリップシンクが正確だとは言えない。最終判断は動画を再生して人が行う
# 使い方：python lipsync_chart.py <mp4> <出力フォルダ> [区間 "開始-終了:ラベル" ...]
import sys, os, json, subprocess
import numpy as np, cv2, librosa
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mp4, out = sys.argv[1], sys.argv[2]
segs = []
for s in sys.argv[3:]:
    rng, lab = s.split(":", 1); a, b = rng.split("-"); segs.append((float(a), float(b), lab))
os.makedirs(out, exist_ok=True)

cap = cv2.VideoCapture(mp4); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
mar, t = [], []
with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3) as fm:
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        r = fm.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if r.multi_face_landmarks:
            L = r.multi_face_landmarks[0].landmark; h, w = fr.shape[:2]
            P = lambda k: np.array([L[k].x * w, L[k].y * h])
            mar.append(float(np.linalg.norm(P(13) - P(14)) / max(1e-6, np.linalg.norm(P(61) - P(291)))))
        else:
            mar.append(np.nan)
        t.append(i / fps); i += 1
cap.release()
mar, t = np.array(mar), np.array(t)

# 音声：完成品の音声トラックを読む（映像と同じファイル＝同じ時間軸）
wav = os.path.join(out, "_audio_from_mp4.wav")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
y, sr = librosa.load(wav, sr=16000, mono=True)
hop = int(sr / fps)
rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop, center=True)[0]
rt = np.arange(len(rms)) / fps
rms_db = 20 * np.log10(np.maximum(rms, 1e-5))

# 区間ごとの集計（発話／無音で口の開きを比べる）
rows = []
for a, b, lab in segs:
    m = (t >= a) & (t < b)
    v = mar[m]; v = v[~np.isnan(v)]
    rows.append({"start": a, "end": b, "label": lab, "frames": int(m.sum()),
                 "mouth_open_mean": round(float(v.mean()), 4) if len(v) else None,
                 "mouth_open_max": round(float(v.max()), 4) if len(v) else None})
n = min(len(mar), len(rms_db)); ok = ~np.isnan(mar[:n])
corr = float(np.corrcoef(mar[:n][ok], rms_db[:n][ok])[0, 1]) if ok.sum() > 5 else None
summary = {"mp4": mp4, "fps": fps, "frames": int(len(mar)), "face_detected_frames": int((~np.isnan(mar)).sum()),
           "corr_mouth_open_vs_rms_db": None if corr is None else round(corr, 3), "segments": rows,
           "note": "目安。相関や静止画だけでリップシンクの正確さは判断しない。最終判断は再生して人が行う"}
json.dump(summary, open(os.path.join(out, "lipsync_metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(out, "mouth_open_per_frame.csv"), "w", encoding="utf-8") as f:
    f.write("frame,t_s,mouth_open,rms_db\n")
    for k in range(len(mar)):
        f.write(f"{k},{t[k]:.3f},{'' if np.isnan(mar[k]) else f'{mar[k]:.4f}'},{rms_db[k]:.2f}\n" if k < len(rms_db) else f"{k},{t[k]:.3f},{mar[k]:.4f},\n")

# 図：上＝波形、中＝音量（dB）、下＝口の開き。無音の区間は灰色
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
ax[0].plot(np.arange(len(y)) / sr, y, lw=0.4, color="#333"); ax[0].set_ylabel("波形")
ax[1].plot(rt, rms_db, color="#1e6bff"); ax[1].set_ylabel("音量 dB")
ax[2].plot(t, mar, color="#d92d20", marker=".", ms=3, lw=1); ax[2].set_ylabel("口の開き（比）"); ax[2].set_xlabel("時間（秒）")
for a, b, lab in segs:
    for x in ax:
        if "無音" in lab: x.axvspan(a, b, color="#bbb", alpha=0.35)
    ax[0].text((a + b) / 2, ax[0].get_ylim()[1] * 0.85, lab, ha="center", fontsize=8)
fig.suptitle(f"口の開きと音声（{os.path.basename(mp4)}）※目安。最終判断は再生して確認", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(out, "lipsync_chart.png"), dpi=110)
print(json.dumps(summary, ensure_ascii=False, indent=1))
