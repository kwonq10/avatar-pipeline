# 比較用の「軽い余韻・ノイズ調整」WAV を作り、元の音声と比べる（CPU だけ。元の音声ファイルは変更しない）
# 方針：
#   - 長さ（サンプル数）・発話の開始と終了・声の高さ・速さは変えない（ゲインを掛けるだけ。時間方向の処理はしない）
#   - 発話の保護：-45dBFS を超える 10ms 窓、または有声（pyin）の窓を発話とし、前 100ms・後 150ms を保護する。保護の中は元のサンプルのまま（ビット単位で同じ）
#   - 保護の外で -48dBFS より小さい音だけ、最大 MAX_ATT_DB 弱める（完全な無音にはしない）。境目は 30ms の滑らかな傾きでつなぐ
# 使い方：python audio_tail_adjust.py <元の WAV（44.1kHz・16bit）> <出力フォルダ>
import sys, os, json
import numpy as np, soundfile as sf, librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
MAX_ATT_DB, SPEECH_DB, QUIET_DB = 12.0, -45.0, -48.0
PRE_MS, POST_MS, RAMP_MS, WIN_MS = 100, 150, 30, 10

x, sr = sf.read(SRC, dtype="int16")
xf = x.astype(np.float64) / 32768.0
hop = int(sr * WIN_MS / 1000); n = int(np.ceil(len(x) / hop))
db = np.array([20 * np.log10(np.sqrt(np.mean(xf[i * hop:(i + 1) * hop] ** 2)) + 1e-9) for i in range(n)])
# 有声の判定（16kHz にした信号で pyin、20ms ごと → 10ms の窓に写す）
y16 = librosa.resample(xf, orig_sr=sr, target_sr=16000)
_, vf, _ = librosa.pyin(y16, fmin=70, fmax=500, sr=16000, frame_length=1024, hop_length=320)
voiced = np.zeros(n, bool)
for k, v in enumerate(vf):
    if v: voiced[min(n - 1, 2 * k):min(n, 2 * k + 2)] = True
speech = (db > SPEECH_DB) | voiced
protect = speech.copy()
pre, post = PRE_MS // WIN_MS, POST_MS // WIN_MS
for i in np.where(speech)[0]:
    protect[max(0, i - pre):min(n, i + post + 1)] = True
att = ~(protect | (db >= QUIET_DB))                                         # 窓ごとに「弱める」か
# 傾き（30ms の移動平均）が保護の中に入らないよう、弱める範囲を両側から（傾きの半分＋1窓）縮める
er = int(np.ceil(RAMP_MS / 2 / WIN_MS)) + 1
att_e = att.copy()
for i in range(n):
    if att[i] and not att[max(0, i - er):min(n, i + er + 1)].all(): att_e[i] = False
target_db = np.where(att_e, -MAX_ATT_DB, 0.0)
# サンプル単位のゲイン：窓の目標を伸ばし、30ms の移動平均で滑らかにする（保護の中は必ず 1）
g_db = np.repeat(target_db, hop)[:len(x)]
k = int(sr * RAMP_MS / 1000); g_db = np.convolve(g_db, np.ones(k) / k, mode="same")
prot_s = np.repeat(protect, hop)[:len(x)]
g = np.where(prot_s, 1.0, 10 ** (g_db / 20))
yout = np.where(g >= 1.0, x, np.round(x * g).astype(np.int16)).astype(np.int16)

os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "adjusted_light_tail_-12dB_44k.wav")
sf.write(dst, yout, sr, subtype="PCM_16")

# ---- 検査 ----
yo = yout.astype(np.float64) / 32768.0
db2 = np.array([20 * np.log10(np.sqrt(np.mean(yo[i * hop:(i + 1) * hop] ** 2)) + 1e-9) for i in range(n)])
def onsets(d, thr=-40.0):
    on, off, prev = [], [], False
    for i, v in enumerate(d):
        c = v > thr
        if c and not prev: on.append(round(i * WIN_MS / 1000, 2))
        if prev and not c: off.append(round(i * WIN_MS / 1000, 2))
        prev = c
    return on, off
lags = range(-200, 201)
cc = [np.dot(xf[max(0, l):len(xf) + min(0, l)], yo[max(0, -l):len(yo) - max(0, l)]) for l in lags]
changed = np.where(yout != x)[0]
segs = []                                                  # 変更した区間（秒）
if len(changed):
    br = np.where(np.diff(changed) > hop)[0]
    starts = np.r_[changed[0], changed[br + 1]]; ends = np.r_[changed[br], changed[-1]]
    segs = [(round(s / sr, 3), round(e / sr, 3)) for s, e in zip(starts, ends) if (e - s) / sr > 0.005]
chk = {
    "samples_equal": len(yout) == len(x), "samples": len(x), "seconds": round(len(x) / sr, 6),
    "protected_region_bit_identical": bool(np.all(yout[prot_s] == x[prot_s])),
    "speech_windows": int(speech.sum()), "protected_windows": int(protect.sum()),
    "best_xcorr_lag_samples": list(lags)[int(np.argmax(cc))],
    "onsets_offsets_-40dB_original": onsets(db), "onsets_offsets_-40dB_adjusted": onsets(db2),
    "changed_segments_s": segs,
    "max_gain_step_per_sample_db": round(float(np.max(np.abs(np.diff(20 * np.log10(np.maximum(g, 1e-6)))))), 4),
    "peak_dbfs_original": round(20 * np.log10(np.max(np.abs(xf)) + 1e-12), 2), "peak_dbfs_adjusted": round(20 * np.log10(np.max(np.abs(yo)) + 1e-12), 2),
}
regions = [(0, 0.16, "冒頭"), (0.16, 0.82, "こんにちは"), (0.82, 1.38, "小さな音（息）"), (1.38, 1.72, "短い無音"), (1.72, 4.20, "発話2"),
           (4.20, 5.26, "長い無音"), (5.26, 6.57, "発話3"), (6.57, 6.95, "短い無音2"), (6.95, 8.02, "発話4"), (8.02, 8.333, "末尾")]
chk["regions_db"] = [{"region": f"{a}-{b}", "label": l, "orig_mean_db": round(float(db[int(a * 100):int(b * 100)].mean()), 1),
                      "adj_mean_db": round(float(db2[int(a * 100):int(b * 100)].mean()), 1)} for a, b, l in regions]
json.dump(chk, open(os.path.join(OUT, "audio_adjust_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- 比較の図 ----
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
t = np.arange(len(x)) / sr; tw = np.arange(n) * WIN_MS / 1000
fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
ax[0].plot(t, xf, lw=0.3, color="#555"); ax[0].set_ylabel("元の波形")
ax[1].plot(t, yo, lw=0.3, color="#1e6bff"); ax[1].set_ylabel("調整後の波形")
ax[2].plot(tw, db, color="#555", label="元"); ax[2].plot(tw, db2, color="#1e6bff", label="調整後"); ax[2].set_ylabel("音量 dBFS")
ax[2].axhline(SPEECH_DB, ls=":", color="#d92d20", lw=0.8); ax[2].set_ylim(-100, 0); ax[2].legend(loc="lower right")
ax[3].plot(t, 20 * np.log10(np.maximum(g, 1e-6)), color="#a94fe0"); ax[3].set_ylabel("掛けたゲイン dB"); ax[3].set_xlabel("時間（秒）")
for i in range(n):
    if protect[i]: ax[3].axvspan(i * WIN_MS / 1000, (i + 1) * WIN_MS / 1000, color="#34c759", alpha=0.08, lw=0)
for a, b, l in regions:
    if "無音" in l or "息" in l:
        for xx in ax[:3]: xx.axvspan(a, b, color="#bbb", alpha=0.25)
    ax[0].text((a + b) / 2, 0.55, l, ha="center", fontsize=8)
fig.suptitle("元の音声と軽い調整（保護の外の小さな音だけ最大 -12dB。緑＝保護して変更しない範囲）", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "audio_comparison_chart.png"), dpi=110)
# 拡大（0.6〜1.9 秒）
fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
m = (t >= 0.6) & (t <= 1.9)
ax[0].plot(t[m], xf[m], lw=0.4, color="#555", label="元"); ax[0].plot(t[m], yo[m], lw=0.4, color="#1e6bff", alpha=0.8, label="調整後"); ax[0].legend()
mw = (tw >= 0.6) & (tw <= 1.9)
ax[1].plot(tw[mw], db[mw], color="#555", marker=".", label="元"); ax[1].plot(tw[mw], db2[mw], color="#1e6bff", marker=".", label="調整後")
ax[1].set_ylabel("dBFS"); ax[1].set_xlabel("時間（秒）"); ax[1].legend()
fig.suptitle("拡大：0.6〜1.9 秒（「こんにちは」の後の息・短い無音・「きょう」の k）", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "audio_comparison_zoom_0.6-1.9s.png"), dpi=110)
print(json.dumps(chk, ensure_ascii=False, indent=1))
