# GUI と生成処理をつなぐ部分（GUI のプロセスでは torch もモデルも読み込まない）。
#
# 1ジョブ = gui/jobs/<job_id>/ フォルダ
#   job.json      … 入力（画像・音声・台本・設定）
#   status.json   … 進捗（ワーカーが書く：state / progress / message / output）
#   <name>_result.json, _memory.csv, _stdout.txt … run_stage.py が書く（RAM・VRAM・コミットの記録）
#   STOP          … GUI の停止ボタンで作る → run_stage.py がプロセスツリーごと止める
#
# GPU を使う処理は同時に1つだけ（gui/jobs/.gpu.lock に実行中のジョブ ID と PID を書く）。
# ワーカーは run_stage.py の下で別プロセスとして動く（ジョブの上限・空きRAM・コミット残量・VRAM を監視し、異常なら止める）。
import json, os, subprocess, sys, time, datetime, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # D:\Avatar-Pipeline
GUI = os.path.join(ROOT, "gui")
JOBS = os.path.join(GUI, "jobs")
LOCK = os.path.join(JOBS, ".gpu.lock")
RUN_STAGE = os.path.join(ROOT, "tools", "run_stage.py")
GUI_PY = os.path.join(GUI, ".venv", "Scripts", "python.exe")                # run_stage.py は標準ライブラリだけなので GUI の Python で動かす
FH_PY = os.path.join(ROOT, "flashhead", ".venv", "Scripts", "python.exe")   # 将来の本番ワーカー用（今回は使わない）
LIMITS = ["--min-avail-gib", "2.0", "--job-mem-gib", "12", "--min-commit-headroom-gib", "5", "--max-vram-mib", "11800"]
FH_LIMITS_FILE = os.path.join(ROOT, "flashhead", "tools", "fh_limits.json")
sys.path.insert(0, os.path.join(ROOT, "flashhead", "tools"))
import fh_length                                                               # 長さと検証済みの判定（ワーカーと共通・標準ライブラリだけ）
# GPU 温度：83℃ 以上が 10 秒続いたら停止。温度を 15 秒以上読めなくても停止（FlashHead のジョブだけ）
TEMP_LIMITS = ["--max-gpu-temp-c", "83", "--gpu-temp-hold-s", "10", "--require-gpu-temp", "--gpu-temp-missing-s", "15"]

def gpu_temp_c():
    """nvidia-smi で今の GPU 温度（℃）。読めなければ None"""
    try:
        out = subprocess.run([os.environ.get("AVATAR_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return None
CHUNK_SEC = 28 / 25                                                            # 1チャンク＝28フレーム＝1.12秒

def fh_limits():
    return json.load(open(FH_LIMITS_FILE, encoding="utf-8"))

def plan_for(seconds, kind="fh_gpu", dry_sec_per_chunk=0.7):
    """音声の長さから、目標フレーム数・チャンク数・見込みの処理時間・run_stage の時間切れを決める（固定の 7,200秒をやめた）"""
    import math
    L = fh_limits()
    chunks = max(1, math.ceil(round(seconds * 16000) / 17920))
    if kind == "fh_gpu":
        est = L["gpu_startup_sec_measured"] + chunks * L["gpu_sec_per_chunk_measured"]
        timeout = int(300 + chunks * L["gpu_sec_per_chunk_measured"] * L["timeout_margin"])
    else:
        est = 10 + chunks * (dry_sec_per_chunk + 0.3)
        timeout = int(120 + chunks * (dry_sec_per_chunk + 1.0) * 2)
    return {"seconds": round(seconds, 3), "target_frames": round(seconds * 25), "chunks": chunks, "est_sec": round(est),
            "timeout_s": timeout, "verified_max_seconds": L["gpu_verified_max_seconds"], "verified_max_frames": fh_length.verified_frames(L),
            "unverified": kind == "fh_gpu" and fh_length.is_unverified(seconds, L),   # フレーム数で判定（ワーカーと同じ）
            "est_mp4_mib": round(seconds * 0.35, 1)}
os.makedirs(JOBS, exist_ok=True)

def _pid_alive(pid):
    """Windows API で直接確かめる（tasklist の出力は cp932 で、UTF-8 モードでは読めないため使わない）"""
    if not pid:
        return False
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
    k32.CloseHandle(ctypes.c_void_p(h))
    return bool(ok) and code.value == 259                  # STILL_ACTIVE

def current_job():
    """実行中のジョブ ID（無ければ None）。ロックが残っていても PID が死んでいれば解除する"""
    if not os.path.exists(LOCK):
        return None
    try:
        d = json.load(open(LOCK, encoding="utf-8"))
    except Exception:
        d = {}
    if _pid_alive(d.get("pid")):
        return d.get("job_id")
    os.remove(LOCK)
    return None

FH_WORKER = os.path.join(ROOT, "flashhead", "tools", "fh_worker.py")
FH_IMAGE_CHECK = os.path.join(ROOT, "flashhead", "tools", "image_check.py")
DEFAULT_OUT = os.path.join(ROOT, "outputs")
CHECK_DIR = os.path.join(GUI, "tmp", "image_check")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")

def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def check_image(path, out_dir=None):
    """キャラクター画像の事前チェック（FlashHead 専用 venv の image_check.py を CPU で実行。GPU・torch は使わない）"""
    out_dir = out_dir or os.path.join(CHECK_DIR, _sha256(path)[:16])
    r = subprocess.run([FH_PY, FH_IMAGE_CHECK, path, out_dir], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONUTF8="1"), creationflags=subprocess.CREATE_NO_WINDOW, timeout=120)
    try:
        return json.load(open(os.path.join(out_dir, "image_check.json"), encoding="utf-8"))
    except Exception:
        return {"level": "ng", "reasons": ["画像のチェックに失敗しました：" + (r.stderr or "")[-300:]]}

def audio_info(path):
    """音声の作成元に依存しない：ffprobe で長さ・Hz・ch を読む（WAV 以外も可）"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
                          "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace",
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout
    d = json.loads(out or "{}"); st = (d.get("streams") or [{}])[0]
    return {"codec": st.get("codec_name"), "sample_rate": int(st.get("sample_rate") or 0), "channels": st.get("channels"),
            "seconds": float((d.get("format") or {}).get("duration") or 0)}

# ---------------------------------------------------------------------------------------------
# GPU 生成の許可（3段階）
#   1段目：次のどちらか
#          (a) このプロセスの環境変数 AVATAR_ALLOW_GPU=1（Claude Code・コマンドラインからの1回の実行。恒久的には設定しない）
#          (b) GUI を GPU 生成モード（app.py --gpu-enabled）で起動し、そのバックエンドが発行した「1回限りの許可（permit）」
#              permit はメモリの中だけにある乱数の ID。入力（画像・音声の SHA256・長さ・seed・顔の切り出し・出力先・未検証の確認）に
#              結び付き、180秒で期限切れ、1回使うと消える。ボタンの表示や API の呼び出しだけでは作れない（下の検査に合格したときだけ発行）
#   2段目：ワーカーへ --allow-gpu を渡す（従来どおり。ワーカーも確認する）
#   3段目：検証済みの長さ（fh_limits.json）を超える音声は allow_unverified_length=True が必要（GUI では未検証の確認をした permit だけ）
#   さらに、開始前の検査（gpu_precheck）を permit の発行時と開始時の両方でバックエンドが強制する
# ---------------------------------------------------------------------------------------------
GPU_SESSION = {"enabled": False, "rehearsal": False}   # app.py が起動引数から設定する（環境変数は使わない）
PERMIT_TTL_S = 180
_PERMITS = {}
PRE_MIN_HEADROOM_GIB = 19.0      # 開始前のシステムのコミット残量
PRE_MIN_AVAIL_GIB = 3.0          # 開始前の空きRAM
PRE_MIN_FREE_VRAM_MIB = 8192     # 開始前の空き VRAM（T1c の実測：モデル分 約 7.6GiB）
PRE_MAX_START_TEMP_C = 80.0      # 開始時の GPU 温度（実行中の停止基準 83℃・10秒より手前で始めない）
FH_REPO = os.path.join(ROOT, "flashhead", "SoulX-FlashHead")
FH_PATCH = os.path.join(ROOT, "patches", "flashhead", "fh_01-03_windows_lowmem.patch")
FH_MODELS = os.path.join(ROOT, "flashhead", "models")
FH_SHA_LIST = os.path.join(ROOT, "licenses", "flashhead_weights_sha256.txt")
SHA_CACHE = os.path.join(JOBS, ".weights_sha256_ok.json")

def sys_mem():
    """空きRAM・システムのコミット残量（GiB）。run_stage.py の mem() と同じ定義"""
    import ctypes
    class S(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]
    m = S(); m.dwLength = ctypes.sizeof(m); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"avail_gib": m.ap / 2**30, "headroom_gib": m.apf / 2**30, "commit_limit_gib": m.tpf / 2**30}

def gpu_status():
    """nvidia-smi：温度（℃）・VRAM 使用・合計（MiB）。読めない値は None"""
    try:
        out = subprocess.run([os.environ.get("AVATAR_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=temperature.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip().splitlines()
        t, u, tot = (float(x) for x in out[0].split(","))
        return {"temp_c": t, "vram_used_mib": u, "vram_total_mib": tot}
    except Exception:
        return {"temp_c": None, "vram_used_mib": None, "vram_total_mib": None}

def weights_ok():
    """重みの SHA256 を一覧と照合する。全部を読むと時間がかかるので、合格したときのファイルの大きさと更新時刻を記録し、変わっていなければ再計算しない"""
    try: cache = json.load(open(SHA_CACHE, encoding="utf-8"))
    except Exception: cache = {}
    bad, new = [], {}
    for line in open(FH_SHA_LIST, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        h, p = line.split(" *", 1); f = os.path.normpath(os.path.join(FH_MODELS, p))
        if not os.path.isfile(f): bad.append(f"無い：{p}"); continue
        st = os.stat(f); key = f"{f}|{st.st_size}|{st.st_mtime_ns}"
        if cache.get(key) == h: new[key] = h; continue
        if _sha256(f) == h: new[key] = h
        else: bad.append(f"不一致：{p}")
    if not bad:
        json.dump(new, open(SHA_CACHE, "w", encoding="utf-8"), indent=0)
    return (not bad), bad

def patch_ok():
    """FH-01〜03 が SoulX-FlashHead に入っているか（逆向きに当てられる＝適用済み）"""
    r = subprocess.run(["git", "apply", "--check", "-R", "--ignore-whitespace", FH_PATCH], cwd=FH_REPO, capture_output=True, text=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    return r.returncode == 0

def gpu_precheck(check_weights=True):
    """GPU 生成の開始前の検査。items は (名前, 合格か, 値の説明)。ok は全部合格のとき True"""
    m = sys_mem(); g = gpu_status(); running = current_job(); patched = patch_ok()
    free_vram = (g["vram_total_mib"] - g["vram_used_mib"]) if g["vram_total_mib"] is not None else None
    items = [
        ("実行中のジョブが無い", running is None, running or "なし"),
        (f"コミット残量 {PRE_MIN_HEADROOM_GIB:.0f}GiB 以上", m["headroom_gib"] >= PRE_MIN_HEADROOM_GIB, f"{m['headroom_gib']:.2f}GiB"),
        (f"空きRAM {PRE_MIN_AVAIL_GIB:.0f}GiB 以上", m["avail_gib"] >= PRE_MIN_AVAIL_GIB, f"{m['avail_gib']:.2f}GiB"),
        ("GPU 温度を読める", g["temp_c"] is not None, "読めない" if g["temp_c"] is None else f"{g['temp_c']:.0f}℃"),
        (f"GPU 温度 {PRE_MAX_START_TEMP_C:.0f}℃ 未満", g["temp_c"] is not None and g["temp_c"] < PRE_MAX_START_TEMP_C,
         "—" if g["temp_c"] is None else f"{g['temp_c']:.0f}℃"),
        (f"空き VRAM {PRE_MIN_FREE_VRAM_MIB:,}MiB 以上", free_vram is not None and free_vram >= PRE_MIN_FREE_VRAM_MIB,
         "—" if free_vram is None else f"{free_vram:,.0f}MiB"),
        ("Windows 対応の修正 FH-01〜03", patched, "適用済み" if patched else "見つからない"),
    ]
    if check_weights:
        ok, bad = weights_ok()
        items.append(("モデルの重みの SHA256", ok, "一致" if ok else "／".join(bad)[:120]))
    return {"ok": all(x[1] for x in items), "items": items, "mem": m, "gpu": g}

def _fingerprint(payload, seconds):
    return {"image_sha256": _sha256(payload["image"]), "audio_sha256": _sha256(payload["audio"]), "seconds": round(float(seconds), 3),
            "seed": int(payload.get("seed", 42)), "use_face_crop": bool(payload.get("use_face_crop")),
            "output_dir": os.path.abspath(payload.get("output_dir") or DEFAULT_OUT),
            # 未検証の確認は、検証済みを超える長さのときだけ意味を持つ（検証済みの長さでは常に False として扱う）
            "allow_unverified_length": bool(payload.get("allow_unverified_length")) and plan_for(seconds, "fh_gpu")["unverified"]}

def issue_gpu_permit(payload):
    """GUI（GPU 生成モード）の「動画を生成」→ バックエンドで入力と開始前の条件を検査し、合格したら1回限りの許可を発行する。
       戻り値：(permit_id, 計画, 検査)。不合格なら RuntimeError"""
    if not GPU_SESSION["enabled"]:
        raise RuntimeError("この GUI は GPU 生成モードで起動していません（start_gui_gpu.cmd で起動してください）")
    for k, label in (("image", "キャラクター画像"), ("audio", "音声ファイル")):
        if not payload.get(k) or not os.path.isfile(payload[k]):
            raise RuntimeError(f"{label}を選んでください（必須）")
    if os.path.splitext(payload["audio"])[1].lower() not in AUDIO_EXT:
        raise RuntimeError("対応していない音声形式です（WAV・MP3・FLAC・M4A・AAC・OGG・Opus）")
    sec = audio_seconds(payload["audio"])
    if sec < 0.5:
        raise RuntimeError(f"音声が短すぎます（{sec:.2f}秒）")
    pl = plan_for(sec, "fh_gpu")
    if pl["unverified"] and not payload.get("allow_unverified_length"):
        raise RuntimeError(f"GPU 生成で未検証の長さです：{fh_length.describe(sec, fh_limits())}。"
                           "「未検証の長さで実行する」にチェックしてから、もう一度押してください")
    chk = check_image(payload["image"])
    if chk.get("level") == "ng":
        raise RuntimeError("この画像は使えません：" + " / ".join(chk.get("reasons", [])))
    pre = gpu_precheck()
    if not pre["ok"]:
        raise RuntimeError("開始前の条件を満たしません：" + "、".join(f"{n}（{v}）" for n, ok, v in pre["items"] if not ok))
    for k in [k for k, v in _PERMITS.items() if v["expires"] < time.time()]:
        _PERMITS.pop(k, None)
    import secrets
    pid = secrets.token_hex(16)
    _PERMITS[pid] = {"fingerprint": _fingerprint(payload, sec), "expires": time.time() + PERMIT_TTL_S, "issued": time.time()}
    return pid, pl, pre

def revoke_gpu_permit(pid):
    _PERMITS.pop(pid or "", None)

def _consume_permit(pid, payload, seconds):
    p = _PERMITS.pop(pid or "", None)                                    # 1回使ったら消える（不合格でも消す）
    if not p:
        raise RuntimeError("生成の許可がありません、または使用済みです。「動画を生成」からやり直してください")
    if p["expires"] < time.time():
        raise RuntimeError("生成の許可の期限が切れました（180秒）。「動画を生成」からやり直してください")
    if p["fingerprint"] != _fingerprint(payload, seconds):
        raise RuntimeError("確認したあとに入力（画像・音声・設定）が変わりました。「動画を生成」からやり直してください")

LEN_DIR = os.path.join(GUI, "tmp", "length_check")

def audio_seconds(path):
    """判定に使う音声の長さ（秒）：start_job と同じ ffmpeg の変換で PCM の WAV にし、そのサンプル数から測る（ワーカーが処理する値と同じ）。
       元のファイルは変更しない。変換した WAV は SHA256 ごとに gui の tmp フォルダ（length_check）に置いて使い回す"""
    os.makedirs(LEN_DIR, exist_ok=True)
    dst = os.path.join(LEN_DIR, _sha256(path)[:24] + ".wav")
    if not os.path.isfile(dst):
        tmp = dst + ".tmp.wav"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-c:a", "pcm_s16le", tmp], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError("音声を WAV に変換できません：" + r.stderr[-200:])
        os.replace(tmp, dst)
    return fh_length.wav_seconds(dst)

def start_job(kind, payload, permit=None):
    """kind:
       'dummy'      … GUI だけの確認用（GUI の Python で dummy_job.py）
       'fh_dryrun'  … FlashHead 専用 venv の fh_worker.py を、推論なし（torch・モデルを読まない）で実行する接続テスト
       'fh_gpu'     … FlashHead Pro で実際に生成する。1段目の許可（環境変数 AVATAR_ALLOW_GPU=1、または GUI の permit）と
                      開始前の検査（gpu_precheck）に合格したときだけ。GPU_SESSION["rehearsal"] のときは、同じ許可と検査を通したうえで
                      ワーカーを推論なし（dry_run）で動かす（GUI のダミーテスト用。GPU は使わない）"""
    if kind not in ("dummy", "fh_dryrun", "fh_gpu"):
        raise RuntimeError(f"未知の種類：{kind}")
    rehearsal = False
    if kind == "fh_gpu":
        if permit is not None:
            if not GPU_SESSION["enabled"]:
                raise RuntimeError("この GUI は GPU 生成モードで起動していません")
            if not payload.get("audio") or not os.path.isfile(payload["audio"]) or not payload.get("image") or not os.path.isfile(payload["image"]):
                raise RuntimeError("キャラクター画像と音声ファイルを選んでください（必須）")
            _consume_permit(permit, payload, audio_seconds(payload["audio"]))
            rehearsal = bool(GPU_SESSION["rehearsal"])
        elif os.environ.get("AVATAR_ALLOW_GPU") != "1":
            raise RuntimeError("FlashHead Pro での実際の生成は未承認です")
        pre = gpu_precheck()                                             # 開始時にもう一度（発行から時間がたっていても安全に）
        if not pre["ok"]:
            raise RuntimeError("開始前の条件を満たしません：" + "、".join(f"{n}（{v}）" for n, ok, v in pre["items"] if not ok))
    if current_job():
        raise RuntimeError(f"実行中のジョブがあります: {current_job()}（GPU 処理は同時に1つまで）")
    job_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + kind + ("_rehearsal" if rehearsal else "")
    d = os.path.join(JOBS, job_id); os.makedirs(d)
    job = {"job_id": job_id, "kind": kind, "created": datetime.datetime.now().isoformat(timespec="seconds"), **payload}
    if kind in ("fh_dryrun", "fh_gpu"):
        # ジョブごとに入力を分ける：キャラクター画像と音声をジョブのフォルダにコピーして、それを使う
        inp = os.path.join(d, "inputs"); os.makedirs(inp)
        img_src, aud_src = payload.get("image"), payload.get("audio")
        if not img_src or not os.path.isfile(img_src):
            shutil.rmtree(d); raise RuntimeError("キャラクター画像を選んでください（必須）")
        if not aud_src or not os.path.isfile(aud_src):
            shutil.rmtree(d); raise RuntimeError("音声ファイルを選んでください（必須）")
        if os.path.splitext(aud_src)[1].lower() not in AUDIO_EXT:
            shutil.rmtree(d); raise RuntimeError("対応していない音声形式です（WAV・MP3・FLAC・M4A・AAC・OGG・Opus）")
        img_dst = os.path.join(inp, "character" + os.path.splitext(img_src)[1].lower())
        shutil.copy2(img_src, img_dst)
        chk = check_image(img_dst, os.path.join(inp, "image_check"))
        if chk.get("level") == "ng":
            shutil.rmtree(d); raise RuntimeError("この画像は使えません：" + " / ".join(chk.get("reasons", [])))
        # 音声：元のファイルを残し、ワーカーには PCM の WAV（サンプルレート・ch はそのまま）を渡す
        aud_orig = os.path.join(inp, "audio_original" + os.path.splitext(aud_src)[1].lower())
        shutil.copy2(aud_src, aud_orig)
        aud_wav = os.path.join(inp, "audio.wav")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", aud_orig, "-vn", "-c:a", "pcm_s16le", aud_wav],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode != 0:
            shutil.rmtree(d); raise RuntimeError("音声を WAV に変換できません：" + r.stderr[-200:])
        job["image"], job["audio"] = img_dst, aud_wav
        wav_sec = fh_length.wav_seconds(aud_wav)                            # 判定に使う長さ（ワーカーと同じ測り方）
        job["character"] = {"name": (payload.get("character_name") or "").strip() or None,
                            "image_original_name": os.path.basename(img_src), "image_sha256": _sha256(img_dst),
                            "image_check": {k: chk.get(k) for k in ("level", "reasons", "recommend_use_face_crop", "info")}}
        job["audio_source"] = {"original_name": os.path.basename(aud_src), "original_sha256": _sha256(aud_orig),
                               "info": audio_info(aud_orig), "converted_to": "audio.wav (PCM 16bit)", "wav_seconds": wav_sec}
        job.pop("character_name", None)
        out = payload.get("output_dir") or DEFAULT_OUT
        if not os.path.isabs(out):
            shutil.rmtree(d); raise RuntimeError("出力フォルダは絶対パスで指定してください")
        os.makedirs(out, exist_ok=True); job["output_dir"] = out
        job["mode"] = "dry_run" if (kind == "fh_dryrun" or rehearsal) else "gpu"
        if kind == "fh_gpu":
            job["gpu_permission"] = {"source": "gui_permit" if permit is not None else "env AVATAR_ALLOW_GPU (one process)",
                                     "rehearsal_no_gpu": rehearsal, "precheck": [list(x) for x in pre["items"]]}
        job["model_type"] = "pro"                            # Lite は選べない
    json.dump(job, open(os.path.join(d, "job.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"state": "queued", "progress": 0.0, "message": "起動中"}, open(os.path.join(d, "status.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if kind == "dummy":
        worker = [GUI_PY, os.path.join(GUI, "tools", "dummy_job.py"), "--job-dir", d]
    else:
        worker = [FH_PY, FH_WORKER, "--job-dir", d] + (["--allow-gpu"] if (kind == "fh_gpu" and not rehearsal) else [])
    timeout = 1800
    if kind in ("fh_dryrun", "fh_gpu"):
        pl = plan_for(job["audio_source"]["wav_seconds"], kind, float(payload.get("dry_sec_per_chunk", 0.7)))
        if pl["unverified"] and not payload.get("allow_unverified_length"):
            shutil.rmtree(d); raise RuntimeError(f"GPU 生成で未検証の長さです：{fh_length.describe(job['audio_source']['wav_seconds'], fh_limits())}。"
                                                 "「未検証の長さで実行する」を確認してください")
        job["plan_estimate"] = pl
        timeout = pl["timeout_s"]
        json.dump(job, open(os.path.join(d, "job.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    temp = []
    if kind in ("fh_dryrun", "fh_gpu"):
        temp = list(TEMP_LIMITS)
        if kind == "fh_dryrun" and payload.get("test_max_gpu_temp_c") is not None:   # テスト用（接続テストだけ）：しきい値を下げて停止を確かめる
            temp[1] = str(float(payload["test_max_gpu_temp_c"]))
    cmd = [GUI_PY, RUN_STAGE, "--name", "job", "--out", d, "--stop-file", os.path.join(d, "STOP"), *LIMITS, "--timeout-s", str(timeout),
           *temp, "--cwd", d, "--", *worker]
    log = open(os.path.join(d, "run_stage_console.txt"), "w", encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", FLASHHEAD_LOWMEM_LOAD="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=os.path.join(ROOT, "cache", "hf"), TORCH_HOME=os.path.join(ROOT, "cache", "torch"),
               TMP=os.path.join(ROOT, "cache", "tmp"), TEMP=os.path.join(ROOT, "cache", "tmp"))
    env.pop("GEMINI_API_KEY", None)                          # ワーカーには API キーを渡さない
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=d,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW, env=env)
    json.dump({"job_id": job_id, "pid": p.pid, "started": time.time()}, open(LOCK, "w", encoding="utf-8"))
    return job_id

def stop_job():
    j = current_job()
    if not j:
        return None
    open(os.path.join(JOBS, j, "STOP"), "w").close()
    return j

def read_status(job_id):
    d = os.path.join(JOBS, job_id)
    st = {}
    try:
        st = json.load(open(os.path.join(d, "status.json"), encoding="utf-8"))
    except Exception:
        pass
    for name, key in (("result.json", "result"), ("settings.json", "settings")):
        fp = os.path.join(d, name)
        if os.path.exists(fp):
            try: st[key] = json.load(open(fp, encoding="utf-8"))
            except Exception: pass
    rp = os.path.join(d, "job_result.json")
    if os.path.exists(rp):
        try:
            r = json.load(open(rp, encoding="utf-8"))
            st["monitor"] = {"exit_code": r.get("exit_code"), "aborted": r.get("aborted"), "elapsed_s": r.get("elapsed_s"), "peak": r.get("peak")}
            if r.get("aborted"):
                st["state"] = "stopped" if "ユーザー" in r["aborted"] else "aborted"
                st["message"] = f"停止：{r['aborted']}"
            elif r.get("exit_code") not in (0, None) and st.get("state") != "done":
                st["state"] = "failed"
        except Exception:
            pass
    # 完成品は「state=done かつ result.ok=true かつ ファイルがある」ときだけ。それ以外の出力は不完全として扱う
    res = st.get("result") or {}
    if st.get("output") and not (st.get("state") == "done" and res.get("ok") and os.path.isfile(st["output"])):
        st.pop("output", None)
    jid = os.path.basename(d)
    out_dir = ((st.get("settings") or {}).get("plan") or {}).get("output")
    cands = [os.path.join(d, "_video_noaudio.partial.mp4")] + ([os.path.join(os.path.dirname(out_dir), jid + ".incomplete.mp4")] if out_dir else [])
    inc = [p for p in cands if os.path.isfile(p)]
    if inc and st.get("state") in ("failed", "stopped", "aborted"):
        st["incomplete_files"] = inc
    return st

def history(limit=30):
    rows = []
    for j in sorted((x for x in os.listdir(JOBS) if not x.startswith(".")), reverse=True)[:limit]:
        st = read_status(j)
        try:
            job = json.load(open(os.path.join(JOBS, j, "job.json"), encoding="utf-8"))
        except Exception:
            job = {}
        mon = st.get("monitor") or {}
        peak = mon.get("peak") or {}
        chn = ((job.get("character") or {}).get("name") or (job.get("character") or {}).get("image_original_name") or "")
        rows.append([j, job.get("kind", ""), chn, st.get("state", ""), f"{100 * float(st.get('progress', 0)):.0f}%",
                     mon.get("elapsed_s", ""), peak.get("vram_max_mib", ""), peak.get("avail_min_gib", ""),
                     st.get("output", "") or ("（不完全なファイルあり）" if st.get("incomplete_files") else "")])
    return rows
