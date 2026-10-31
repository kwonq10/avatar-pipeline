# A1（先読み）の要否を CPU で確かめる：wav2vec2 の特徴量を、公式の stream 方式（ワーカーと同じ）と once 方式で計算して比べる
# GPU・FlashHead の生成モデルは使わない（音声エンコーダ wav2vec2 だけを CPU で動かす）。本体のコードは読み込むだけで変更しない
# 使い方：python a1_context_check.py <出力フォルダ> <WAV:ラベル> [<WAV:ラベル> ...]
import sys, os, json
import numpy as np, torch, librosa
REPO = r"D:\Avatar-Pipeline\flashhead\SoulX-FlashHead"; W2V = r"D:\Avatar-Pipeline\flashhead\models\wav2vec2-base-960h"
sys.path.insert(0, REPO)
from transformers import Wav2Vec2FeatureExtractor
from flash_head.audio_analysis.wav2vec2 import Wav2Vec2Model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]; INPUTS = [a.rsplit(":", 1) for a in sys.argv[2:]]
SR, FPS, FRAME_NUM, MOTION, CACHED = 16000, 25, 33, 5, 8
SLICE = FRAME_NUM - MOTION; slice_samples = SLICE * SR // FPS
torch.set_num_threads(max(1, os.cpu_count() // 2))
fe = Wav2Vec2FeatureExtractor.from_pretrained(W2V, local_files_only=True)
enc = Wav2Vec2Model.from_pretrained(W2V, local_files_only=True).eval()
enc.feature_extractor._freeze_parameters()

def emb(arr):   # pipeline.preprocess_audio と同じ（seq_len = 長さ×25/16000）→ (フレーム, 12 層, 768)
    v = torch.from_numpy(np.squeeze(fe(arr, sampling_rate=SR).input_values)).float().unsqueeze(0)
    with torch.no_grad():
        h = enc(v, seq_len=int(len(arr) * FPS / SR), output_hidden_states=True)
    return torch.stack(h.hidden_states[1:], dim=1).squeeze(0).permute(1, 0, 2).numpy()

def ctx(e, start, end):   # get_audio_embedding と同じ ±2 フレームの参照（範囲の端で切り詰め）。中心と、切り詰めの有無を返す
    idx = np.clip(np.arange(start, end)[:, None] + np.arange(-2, 3)[None, :], 0, end - 1)
    return e[idx], (np.arange(start, end) + 2 > end - 1)

def stream(y):  # generate_video.py の stream（ワーカーと同じ）：新しい 28 フレームぶんの特徴量と、±2 の切り詰めの有無
    rem = len(y) % slice_samples
    if rem: y = np.concatenate([y, np.zeros(slice_samples - rem, y.dtype)])
    win = np.zeros(SR * CACHED, np.float32); a_end = CACHED * FPS; a_start = a_end - FRAME_NUM
    feats, clamped = [], []
    for s in y.reshape(-1, slice_samples):
        win = np.concatenate([win[len(s):], s]).astype(np.float32)
        c, cl = ctx(emb(win), a_start, a_end)
        feats.append(c[MOTION:]); clamped.append(cl[MOTION:])
    return np.concatenate(feats), np.concatenate(clamped)

def once(y):    # generate_video.py の once：全体を一度にエンコード（後ろの文脈がある）
    fn = FRAME_NUM * SR // FPS; rem = (len(y) - fn) % slice_samples
    if rem: y = np.concatenate([y, np.zeros(slice_samples - rem, y.dtype)])
    e = emb(y.astype(np.float32)); c, cl = ctx(e, 0, e.shape[0])
    return c, cl

def cos(a, b):  # (フレーム, 5, 12, 768) どうしの、フレームごとのコサイン類似度（5×12 の平均）
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9); b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a * b).sum(-1).mean(axis=(1, 2))

res = {}; series = {}
for path, lab in INPUTS:
    y, _ = librosa.load(path, sr=SR, mono=True)
    nf = round(len(y) / SR * FPS)
    fs, cls = stream(y); fo, _ = once(y)
    n = min(nf, len(fs), len(fo))
    sim = cos(fs[:n], fo[:n])                                     # stream と once の一致度（1 に近いほど同じ）
    pos = np.arange(n) % SLICE                                    # チャンクの中の位置（0〜27）
    by_pos = [round(float(sim[pos == p].mean()), 4) for p in range(SLICE)]
    sil = fs[int(4.40 * FPS):int(5.10 * FPS)].mean(axis=0)        # 本当の無音（4.40〜5.10 秒）の平均の特徴量（stream）
    silsim = cos(fs[:n], np.repeat(sil[None], n, axis=0))         # 各フレームが「無音らしい」度合い
    def rng(a, b): return slice(int(round(a * FPS)), int(round(b * FPS)))
    res[lab] = {"frames": int(n), "stream_vs_once_mean": round(float(sim.mean()), 4),
                "stream_vs_once_by_chunk_position": by_pos,
                "clamped_frames_in_stream(pos 26,27)": int(cls[:n].sum()),
                "stream_vs_once_at": {k: round(float(sim[rng(a, b)].mean()), 4) for k, (a, b) in
                                      {"0.20-0.28 開き閉じ": (0.20, 0.28), "0.82-1.38 息": (0.82, 1.38), "1.38-1.72 短い無音": (1.38, 1.72),
                                       "onset 1.72±0.12": (1.60, 1.84), "onset 5.26±0.12": (5.14, 5.38), "onset 6.95±0.12": (6.83, 7.07),
                                       "チャンク末尾 pos26-27 全体": (0, 0)}.items() if b > a},
                "silence_likeness": {k: round(float(silsim[rng(a, b)].mean()), 4) for k, (a, b) in
                                     {"0.82-0.98 は の息(保護)": (0.82, 0.98), "0.98-1.38 息": (0.98, 1.38), "1.38-1.72 短い無音": (1.38, 1.72),
                                      "4.20-5.26 長い無音": (4.20, 5.26), "6.57-6.95 短い無音2": (6.57, 6.95), "発話2 1.80-4.10": (1.80, 4.10)}.items()}}
    res[lab]["stream_vs_once_at"]["チャンク末尾 pos26-27 全体"] = round(float(sim[pos >= 26].mean()), 4)
    res[lab]["stream_vs_once_at"]["チャンク前半 pos0-20 全体"] = round(float(sim[pos <= 20].mean()), 4)
    series[lab] = (sim, silsim)

os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "a1_context_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for lab, (sim, silsim) in series.items():
    t = np.arange(len(sim)) / FPS
    ax[0].plot(t, sim, marker=".", ms=3, label=lab); ax[1].plot(t, silsim, marker=".", ms=3, label=lab)
for x in ax:
    for b in np.arange(SLICE, 8.4 * FPS, SLICE) / FPS: x.axvline(b, color="#d92d20", ls=":", lw=0.8)
    for a, b in [(1.38, 1.72), (4.20, 5.26), (6.57, 6.95)]: x.axvspan(a, b, color="#bbb", alpha=0.3)
    x.axvspan(0.82, 1.38, color="#ffcc00", alpha=0.15); x.legend(loc="lower right")
ax[0].set_ylabel("stream と once の一致度"); ax[1].set_ylabel("無音らしさ（stream）"); ax[1].set_xlabel("時間（秒）　赤の点線＝チャンクの境目")
fig.suptitle("A1 の確認：先読みの無い stream 方式の特徴量と、全体を一度に読む once 方式の違い（wav2vec2・CPU）", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "a1_context_check.png"), dpi=110)
print(json.dumps(res, ensure_ascii=False, indent=1))
