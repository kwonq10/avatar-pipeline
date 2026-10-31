# FlashHead Pro のワーカー（FlashHead 専用 venv で、run_stage.py の監視の下で GUI から別プロセスとして起動される）。
#
# 入力：<job_dir>/job.json  {"image", "audio", "output_dir", "seed", "use_face_crop", "mode": "dry_run"|"gpu",
#                            "allow_unverified_length"(任意), "dry_sec_per_chunk"(任意)}
# 出力：<job_dir>/status.json   進捗（GUI が1秒ごとに読む）
#       <job_dir>/settings.json 生成設定（モデル・SHA256・パラメータ・計画したフレーム数）
#       <job_dir>/result.json   実行結果（ok=true のときだけ完成品。失敗時は ok=false と不完全なファイルの一覧）
#       <job_dir>/worker_mem.csv チャンクごとのワーカーと ffmpeg のメモリ（private・working set）
#       <output_dir>/<job_id>.mp4  音声付き MP4（H.264 + AAC）。検査に合格したときだけ、この名前になる
#   途中のファイル（完成品ではない）：
#       <job_dir>/_video_noaudio.partial.mp4      逐次書き出し中の映像（音声なし）
#       <output_dir>/<job_id>.incomplete.mp4      音声との結合中・検査前
#   RAM・VRAM・コミットの記録は run_stage.py が <job_dir>/job_memory.csv と job_result.json に書く
#
# mode:
#   dry_run … torch もモデルも読み込まない。チャンクごとに「番号を埋め込んだダミーフレーム」を作り、
#             GPU 生成と同じ逐次書き出し・切り方・結合・検査を通す（GPU を使わない）
#   gpu     … FlashHead Pro で実際に生成する。--allow-gpu が無いと拒否する
#
# 長さ：公式の stream モードと同じく、音声を 28 フレーム（=1.12秒）単位まで無音でつめ物をしてチャンクごとに生成する。
#   フレームはチャンクごとに ffmpeg（frame_sink.FrameSink）へ送り、目標のフレーム数（round(音声の秒数 × 25)）で打ち切る
#   （最終チャンクの余分なフレームは書かない）。最後に元の音声（切らない）と結合する（映像の再エンコードはしない）
# 上限：fh_limits.json の検証済みのフレーム数（gpu_verified_max_frames）を超える GPU 生成は、allow_unverified_length=true が無いと拒否する（fh_length.py）
import argparse, json, os, shutil, subprocess, sys, time, wave, hashlib, datetime, ctypes

FPS = 25
SR = 16000
FRAME_NUM = 33                       # infer_params.yaml
MOTION_FRAMES = (2 - 1) * 4 + 1      # motion_frames_latent_num=2、vae_stride[0]=4 → 5
SLICE = FRAME_NUM - MOTION_FRAMES    # 28 フレーム / chunk（Pro）
SIZE = 512
TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)                                          # D:\Avatar-Pipeline\flashhead
REPO = os.path.join(ROOT, "SoulX-FlashHead")
CKPT = os.path.join(ROOT, "models", "SoulX-FlashHead-1_3B")
W2V = os.path.join(ROOT, "models", "wav2vec2-base-960h")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
LIMITS = json.load(open(os.path.join(TOOLS, "fh_limits.json"), encoding="utf-8"))
sys.path.insert(0, TOOLS)
from frame_sink import FrameSink, EncoderError, proc_mem_mib
import fh_length                                                       # 長さと検証済みの判定（GUI・job_manager と共通）

ap = argparse.ArgumentParser()
ap.add_argument("--job-dir", required=True)
ap.add_argument("--allow-gpu", action="store_true", help="実際の GPU 生成を許可する")
ap.add_argument("--dry-sec-per-chunk", type=float, default=0.7, help="dry_run で1チャンクの処理に見立てる秒数")
a = ap.parse_args()
JOB = json.load(open(os.path.join(a.job_dir, "job.json"), encoding="utf-8"))
T0 = time.time()

def put(**kw):
    p = os.path.join(a.job_dir, "status.json"); tmp = p + ".tmp"
    json.dump({"updated": datetime.datetime.now().isoformat(timespec="seconds"), **kw}, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, p)

def log(stage, **kw):
    print(json.dumps({"t": round(time.time() - T0, 2), "stage": stage, **kw}, ensure_ascii=False), flush=True)

INCOMPLETE = []                      # 失敗したときに残る不完全なファイル（完成品と区別して result.json に書く）

def fail(msg):
    files = [p for p in INCOMPLETE if os.path.exists(p)]
    json.dump({"ok": False, "mode": JOB.get("mode"), "error": msg, "incomplete_files": files, "elapsed_s": round(time.time() - T0, 1)},
              open(os.path.join(a.job_dir, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    put(state="failed", progress=1.0, message=msg, incomplete_files=files); log("failed", message=msg, incomplete_files=files); sys.exit(2)

def mem_gate():
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tf", ctypes.c_ulonglong), ("af", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(m); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"avail_ram_gib": round(m.ap / 2**30, 2), "commit_headroom_gib": round(m.af / 2**30, 2)}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration,sample_rate,channels",
                          "-of", "json", path], capture_output=True, text=True).stdout
    return json.loads(out or "{}").get("streams", [])

# ---- 1. 入力の検査 ----
put(state="running", progress=0.01, message="入力を確認しています")
mode = JOB.get("mode")
img, wav_path = JOB.get("image"), JOB.get("audio")
if not img or not os.path.isfile(img): fail("キャラクター画像が見つかりません")
if not wav_path or not os.path.isfile(wav_path): fail("音声（WAV）が見つかりません")
if not wav_path.lower().endswith(".wav"): fail("音声は WAV にしてください")
try:
    with wave.open(wav_path, "rb") as w:
        wav_sr, wav_ch, wav_n, wav_sw = w.getframerate(), w.getnchannels(), w.getnframes(), w.getsampwidth()
except Exception as e:
    fail(f"WAV を読めません（PCM の WAV が必要です）：{e}")
audio_sec = wav_n / wav_sr
if audio_sec < 0.5: fail(f"音声が短すぎます（{audio_sec:.2f}秒）")
verified = float(LIMITS["gpu_verified_max_seconds"])
unverified = fh_length.is_unverified(audio_sec, LIMITS)               # フレーム数で判定（表示の丸めは使わない）
if mode == "gpu" and unverified and not JOB.get("allow_unverified_length"):
    fail(f"GPU 生成で未検証の長さです：{fh_length.describe(audio_sec, LIMITS)}。"
         "実行するには「未検証の長さで実行する」の確認が必要です")
if mode == "dry_run" and audio_sec > LIMITS["dry_run_max_seconds"]:
    fail(f"接続テストは {LIMITS['dry_run_max_seconds']}秒までです（{audio_sec:.1f}秒）")
try:
    from PIL import Image
    with Image.open(img) as im:
        img_w, img_h = im.size
except Exception as e:
    fail(f"画像を読めません：{e}")

# ---- 2. 計画（公式 stream モードと同じ切り方） ----
target_frames = round(audio_sec * FPS)                      # 最終的な映像のフレーム数＝音声の長さ
samples16 = round(audio_sec * SR)
slice_samples = SLICE * SR // FPS                            # 17,920 サンプル（1.12秒）
n_chunks = -(-samples16 // slice_samples)                    # 切り上げ（末尾は無音でつめ物）
gen_frames = n_chunks * SLICE
out_dir = JOB.get("output_dir") or a.job_dir
os.makedirs(out_dir, exist_ok=True)
out_mp4 = os.path.join(out_dir, f"{JOB['job_id']}.mp4")
out_incomplete = os.path.join(out_dir, f"{JOB['job_id']}.incomplete.mp4")
part_video = os.path.join(a.job_dir, "_video_noaudio.partial.mp4")
for p in (out_mp4, out_incomplete, part_video):
    if os.path.exists(p): fail(f"出力先に同じ名前のファイルがあります（上書きしません）：{p}")
settings = {
    "job_id": JOB["job_id"], "mode": mode, "model": "SoulX-FlashHead Pro（Lite は使わない）",
    "repo_commit": "9bc03de06bb0de82cd6bc477804512ae06144bf2", "patches": ["FH-01", "FH-02", "FH-03"],
    "lowmem_load": os.environ.get("FLASHHEAD_LOWMEM_LOAD", "1"),
    "weights": {"Model_Pro": "e47e61b9…af250", "VAE_Wan": "38071ab5…fd981", "wav2vec2": "8aa76ab2…d74a"},
    "infer": {"fps": FPS, "sample_rate": SR, "frame_num": FRAME_NUM, "motion_frames": MOTION_FRAMES, "slice_frames": SLICE,
              "sample_steps": 4, "audio_encode_mode": "stream", "seed": int(JOB.get("seed", 42)), "use_face_crop": bool(JOB.get("use_face_crop", False)),
              "size": [SIZE, SIZE]},
    "writer": "stream（チャンクごとに ffmpeg の stdin へ rawvideo rgb24 → libx264 CRF16・bf0。目標フレーム数で打ち切り）",
    "input": {"image": img, "image_size": [img_w, img_h], "image_sha256": sha256(img),
              "audio": wav_path, "audio_sha256": sha256(wav_path), "audio_sr": wav_sr, "audio_channels": wav_ch,
              "audio_sample_width": wav_sw, "audio_seconds": round(audio_sec, 3)},
    "length": {"gpu_verified_max_seconds": verified, "gpu_verified_max_frames": fh_length.verified_frames(LIMITS),
               "audio_seconds_exact": audio_sec, "target_frames": fh_length.target_frames(audio_sec), "unverified": mode == "gpu" and unverified,
               "allow_unverified_length": bool(JOB.get("allow_unverified_length"))},
    "plan": {"chunks": n_chunks, "generated_frames": gen_frames, "target_frames": target_frames,
             "trimmed_frames": gen_frames - target_frames, "output": out_mp4},
    "character": JOB.get("character"),          # 名前（任意）・元のファイル名・SHA256・画像チェックの結果
    "audio_source": JOB.get("audio_source"),    # 元の音声ファイル名・SHA256・形式（作成元は問わない）
}
if (JOB.get("character") or {}).get("image_check", {}).get("level") == "ng":
    fail("画像チェックで使えないと判定された画像です")
json.dump(settings, open(os.path.join(a.job_dir, "settings.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("plan", **settings["plan"], audio_seconds=round(audio_sec, 3))

# ---- 3. 映像を作る（チャンクごとに逐次書き出し） ----
def dummy_chunk(i):
    """dry_run 用：チャンク i の 28 フレーム（uint8）。上端に生成フレームの通し番号を 16bit の白黒ブロックで埋め込む
    （verify_frames.py で読み戻し、重複・欠落がないかを確かめる）"""
    import numpy as np
    fr = np.empty((SLICE, SIZE, SIZE, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for j in range(SLICE):
        g = i * SLICE + j
        fr[j, :, :, 0] = (xx + 4 * g) % 256
        fr[j, :, :, 1] = (yy + 2 * g) % 256
        fr[j, :, :, 2] = 96 + 64 * (i % 2)                    # チャンクごとに色を変える（目視で境目が分かる）
        fr[j, :64, :, :] = 0
        for b in range(16):
            if (g >> b) & 1:
                fr[j, :64, b * 32:(b + 1) * 32, :] = 255
    return fr

mem_csv = open(os.path.join(a.job_dir, "worker_mem.csv"), "w", encoding="utf-8")
mem_csv.write("t_s,chunk,frames_written,gen_s,write_s,worker_private_mib,worker_ws_mib,ffmpeg_private_mib,ffmpeg_ws_mib,"
              "vram_alloc_mib,vram_reserved_mib,vram_used_mib,gpu_temp_c,avail_ram_gib,commit_headroom_gib,job_commit_peak_gib\n")

class _JBLI(ctypes.Structure):          # JOBOBJECT_BASIC_LIMIT_INFORMATION（run_stage.py と同じ定義）
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_ulong), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_ulong),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_ulong), ("SchedulingClass", ctypes.c_ulong)]

class _JELI(ctypes.Structure):          # JOBOBJECT_EXTENDED_LIMIT_INFORMATION（このプロセスが入っている run_stage のジョブを読む）
    _fields_ = [("Basic", _JBLI), ("IoInfo", ctypes.c_ulonglong * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

def job_peak_gib():
    """自分が入っているジョブ（run_stage）のコミットの最大。取れなければ空"""
    try:
        info = _JELI()
        ok = ctypes.windll.kernel32.QueryInformationJobObject(None, 9, ctypes.byref(info), ctypes.sizeof(info), None)
        return round(info.PeakJobMemoryUsed / 2**30, 3) if ok else ""
    except Exception:
        return ""

def gpu_smi():
    """nvidia-smi：VRAM 使用量（MiB）と温度（℃）。読めなければ空"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0].split(",")
        return int(out[0]), float(out[1])
    except Exception:
        return "", ""

def mem_row(i, sink, gen_s, write_s, torch=None):
    w = proc_mem_mib() or {}; f = proc_mem_mib(sink.proc.pid) or {}
    va = vr = ""
    if torch is not None:
        va = round(torch.cuda.memory_allocated() / 2**20); vr = round(torch.cuda.memory_reserved() / 2**20)
    vu, tc = gpu_smi(); g = mem_gate()
    mem_csv.write(f"{time.time() - T0:.1f},{i},{sink.written},{gen_s:.2f},{write_s:.3f},{w.get('private_mib', '')},{w.get('ws_mib', '')},"
                  f"{f.get('private_mib', '')},{f.get('ws_mib', '')},{va},{vr},{vu},{tc},{g['avail_ram_gib']},{g['commit_headroom_gib']},{job_peak_gib()}\n")
    mem_csv.flush()

def run_chunks(make_chunk, label, torch=None):
    """make_chunk(i) → numpy uint8 (28, 512, 512, 3)。チャンクごとに sink へ送り、参照をすぐ捨てる"""
    sink = FrameSink(part_video, target_frames, SIZE, SIZE, FPS, stderr_path=os.path.join(a.job_dir, "ffmpeg_stream.log"))
    INCOMPLETE.append(part_video)
    t_loop = time.time()
    try:
        for i in range(n_chunks):
            done_s = time.time() - t_loop
            avg = done_s / i if i else None
            put(state="running", progress=0.1 + 0.8 * i / n_chunks, message=f"{label}：チャンク {i + 1}/{n_chunks}",
                chunk=i + 1, chunks=n_chunks, frames_written=sink.written, target_frames=target_frames,
                sec_per_chunk=round(avg, 2) if avg else None, eta_s=round(avg * (n_chunks - i)) if avg else None)
            t_g = time.time()
            fr = make_chunk(i)
            gen_s = time.time() - t_g
            if fr.shape[0] != SLICE:
                raise EncoderError(f"チャンク {i} のフレーム数が {fr.shape[0]}（{SLICE} のはず）")
            t_w = time.time()
            sink.write(fr)
            write_s = time.time() - t_w
            del fr
            mem_row(i, sink, gen_s, write_s, torch)
        sink.close()
    except BaseException:
        sink.abort()
        raise
    finally:
        mem_csv.flush()
    return sink.written, round(time.time() - t_loop, 1)

try:
    if mode == "dry_run":
        import numpy as np                                    # torch・モデルは読み込まない
        dry = float(JOB.get("dry_sec_per_chunk", a.dry_sec_per_chunk))
        fail_at = JOB.get("dry_fail_at_chunk")                # テスト用（dry_run だけ）：このチャンクで故障させ、失敗時の扱いを確かめる
        def make(i):
            if fail_at is not None and i == int(fail_at):
                raise RuntimeError(f"テスト用の故障（チャンク {i + 1}）")
            t = time.time(); fr = dummy_chunk(i)
            rest = dry - (time.time() - t)
            if rest > 0: time.sleep(rest)
            return fr
        written, loop_s = run_chunks(make, "[接続テスト] 推論なし・GPU 未使用")
    elif mode == "gpu":
        if not a.allow_gpu:
            fail("GPU での生成は未承認です（--allow-gpu が必要）")
        before = mem_gate()
        log("preflight_gate", **before)
        if before["avail_ram_gib"] < 3 or before["commit_headroom_gib"] < 12:
            fail(f"開始条件を満たしません（空きRAM {before['avail_ram_gib']}GiB・コミット残量 {before['commit_headroom_gib']}GiB。必要：3GiB・12GiB）")
        sys.path.insert(0, REPO); os.chdir(REPO)
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("FLASHHEAD_LOWMEM_LOAD", "1")
        t_imp = time.time()
        import numpy as np, torch, librosa
        from collections import deque
        from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
        log("timing", step="import", sec=round(time.time() - t_imp, 1))
        put(state="running", progress=0.03, message="モデルを読み込んでいます（Pro・省メモリ）")
        t_ld = time.time()
        pipe = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type="pro", wav2vec_dir=W2V)
        log("timing", step="get_pipeline", sec=round(time.time() - t_ld, 1)); t_bd = time.time()
        get_base_data(pipe, cond_image_path_or_dir=img, base_seed=int(JOB.get("seed", 42)), use_face_crop=bool(JOB.get("use_face_crop", False)))
        log("timing", step="get_base_data", sec=round(time.time() - t_bd, 1))
        ip = get_infer_params()
        assert ip["frame_num"] == FRAME_NUM and ip["motion_frames_num"] == MOTION_FRAMES and ip["tgt_fps"] == FPS and ip["sample_rate"] == SR
        after = mem_gate(); after["free_vram_gib"] = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
        log("post_load_gate", **after)
        if after["avail_ram_gib"] < 3 or after["commit_headroom_gib"] < 8 or after["free_vram_gib"] < 2:
            fail(f"読み込み後の条件を満たしません：{after}")
        t_au = time.time()
        speech, _ = librosa.load(wav_path, sr=SR, mono=True)  # 音声全体（3分で約 11.5MB。長さに比例する唯一の配列）
        rem = len(speech) % slice_samples
        if rem: speech = np.concatenate([speech, np.zeros(slice_samples - rem, dtype=speech.dtype)])
        slices = speech.reshape(-1, slice_samples)
        assert len(slices) == n_chunks, (len(slices), n_chunks)
        log("timing", step="audio_load", sec=round(time.time() - t_au, 1))
        cached = SR * ip["cached_audio_duration"]; a_end = ip["cached_audio_duration"] * FPS; a_start = a_end - FRAME_NUM
        dq = deque([0.0] * cached, maxlen=cached)            # 最新 8 秒の固定窓（長さによらず一定）
        def make(i):
            dq.extend(slices[i].tolist())
            emb = get_audio_embedding(pipe, np.array(dq), a_start, a_end)
            with torch.no_grad():
                v = run_pipeline(pipe, emb)[MOTION_FRAMES:]   # 33 フレームのうち、つなぎの 5 を除く 28 フレーム
            out = v.to(torch.uint8).cpu().numpy()             # T1 と同じ変換（float 0〜255 → uint8）
            del v, emb
            torch.cuda.synchronize()
            return out
        written, loop_s = run_chunks(make, "生成中", torch)
        del pipe
        torch.cuda.empty_cache()
    else:
        fail(f"未知の mode：{mode}")
except EncoderError as e:
    fail(f"映像の書き出しに失敗：{e}")
except Exception as e:
    fail(f"生成に失敗：{type(e).__name__}: {e}")
log("stream_done", frames_written=written, loop_s=loop_s)

# ---- 4. 元の音声と結合（映像はコピー、音声は切らずに AAC）→ 検査 → 合格したら完成品の名前にする ----
put(state="running", progress=0.93, message="音声と結合しています")
INCOMPLETE.append(out_incomplete)
r = subprocess.run([FFMPEG, "-v", "error", "-n", "-i", part_video, "-i", wav_path, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-f", "mp4", out_incomplete],
                   capture_output=True, text=True)
if r.returncode != 0: fail("ffmpeg（結合）に失敗：" + r.stderr[-300:])
streams = probe(out_incomplete)
v = next((s for s in streams if s.get("codec_type") == "video"), {}); au = next((s for s in streams if s.get("codec_type") == "audio"), {})
vd, ad = float(v.get("duration", 0)), float(au.get("duration", 0))
vf = int(v.get("nb_frames", 0))
checks = {"frames_equal_target": vf == target_frames, "size_512": [v.get("width"), v.get("height")] == [SIZE, SIZE],
          "fps_25": v.get("r_frame_rate") == f"{FPS}/1", "h264": v.get("codec_name") == "h264", "aac": au.get("codec_name") == "aac",
          "av_diff_lt_1_frame": abs(vd - ad) < 1 / FPS, "audio_not_cut": ad >= audio_sec - 1 / FPS}
result = {"ok": all(checks.values()), "mode": mode, "output": out_mp4, "video_frames": vf, "target_frames": target_frames,
          "frames_written": written, "video_seconds": round(vd, 3), "audio_seconds_out": round(ad, 3), "audio_seconds_in": round(audio_sec, 3),
          "av_diff_seconds": round(abs(vd - ad), 3), "av_within_1_frame": abs(vd - ad) < 1 / FPS, "checks": checks,
          "size": [v.get("width"), v.get("height")], "loop_s": loop_s, "elapsed_s": round(time.time() - T0, 1)}
if not result["ok"]:
    fail("出力の検査に不合格：" + ", ".join(k for k, ok in checks.items() if not ok) + f"（映像 {vf}/{target_frames}・差 {abs(vd - ad):.3f}秒）")
try:
    os.rename(out_incomplete, out_mp4)                        # Windows では行き先があると失敗する（上書きしない）
except OSError as e:
    fail(f"完成品の名前に変えられません（上書きしません）：{e}")
INCOMPLETE.remove(out_incomplete)
os.remove(part_video); INCOMPLETE.remove(part_video)
json.dump(result, open(os.path.join(a.job_dir, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", **result)
label = "接続テスト完了（番号入りのダミー映像＋入力の音声。推論はしていません）" if mode == "dry_run" else "生成完了"
put(state="done", progress=1.0, message=label, output=out_mp4, chunks=n_chunks, chunk=n_chunks, frames_written=written, target_frames=target_frames)
