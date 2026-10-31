# 長さの判定が GUI（job_manager）とワーカーで一致するかの CPU テスト。GPU は使わない
# ワーカーは mode=gpu・--allow-gpu なしで直接起動する → 長さの判定に合格すると「--allow-gpu が必要」で、torch を読み込む前に止まる
import os, sys, json, glob, subprocess, shutil, hashlib
os.environ.pop("AVATAR_ALLOW_GPU", None)
sys.path.insert(0, r"D:\Avatar-Pipeline\gui")
import job_manager as jm, fh_length
SP = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(SP, "lentest")
MP3 = glob.glob(r"D:\Avatar-Pipeline\inputs\落ち着いた女性-2026-10-31-13-36-*.mp3")[0]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
mp3_before = sha(MP3)
IMG = r"D:\Avatar-Pipeline\inputs\avatar_bob.png"
cases = [("T1c の元の MP3", MP3, False), ("T1c のジョブの WAV", r"D:\Avatar-Pipeline\gui\jobs\20261031-134405_fh_gpu\inputs\audio.wav", False),
         ("8.339秒（208 フレーム）", os.path.join(T, "tone_8.339.wav"), False), ("8.36秒（209 フレーム）", os.path.join(T, "tone_8.36.wav"), False),
         ("9秒", os.path.join(T, "tone_9.wav"), False), ("9秒・未検証を確認", os.path.join(T, "tone_9.wav"), True),
         ("12秒", r"D:\Avatar-Pipeline\gui\tmp\ui_test\dummy_tone_12s.wav", False)]
L = jm.fh_limits(); print("limits", L["gpu_verified_max_seconds"], L["gpu_verified_max_frames"])
rows = []
for name, path, allow in cases:
    sec = jm.audio_seconds(path); pl = jm.plan_for(sec, "fh_gpu")
    gui_ok = (not pl["unverified"]) or allow                       # GUI・バックエンドが受け付けるか
    # ワーカー：job_manager と同じ変換の WAV を渡す
    d = os.path.join(T, "job_" + str(len(rows))); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    wav = os.path.join(d, "audio.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-c:a", "pcm_s16le", wav], check=True)
    json.dump({"job_id": "lentest_" + str(len(rows)), "mode": "gpu", "image": IMG, "audio": wav, "output_dir": os.path.join(d, "out"),
               "seed": 42, "use_face_crop": True, "allow_unverified_length": allow}, open(os.path.join(d, "job.json"), "w", encoding="utf-8"))
    subprocess.run([jm.FH_PY, jm.FH_WORKER, "--job-dir", d], capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))
    st = json.load(open(os.path.join(d, "status.json"), encoding="utf-8"))
    worker_ok = "--allow-gpu が必要" in st.get("message", "")          # 長さに合格して、GPU の許可の確認まで進んだ
    rows.append((name, f"{sec:.6f}", fh_length.target_frames(sec), pl["unverified"], allow, gui_ok, worker_ok, gui_ok == worker_ok, st.get("message", "")[:90]))
print("ケース | 秒 | フレーム | 未検証 | 確認 | GUI受付 | ワーカー受付 | 一致 | ワーカーの文")
for r in rows: print(" | ".join(str(x) for x in r))
print("all_agree", all(r[7] for r in rows))
print("mp3_unchanged", sha(MP3) == mp3_before, mp3_before[:16])
