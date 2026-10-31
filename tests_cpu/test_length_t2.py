# 検証済み 1,500 フレーム（T2）に更新したあとの、長さの判定の CPU テスト（GPU は使わない）
# - job_manager（GUI と同じ）とワーカーの判定が一致するか（ワーカーは mode=gpu・--allow-gpu なしで直接起動 → 長さに合格すると torch を読む前に「--allow-gpu が必要」で止まる）
# - 1,500 フレームを超えると、未検証の確認（allow_unverified_length）が必要か。permit は発行しても使わずに取り消す（ジョブは作らない）
import os, sys, json, glob, subprocess, shutil, hashlib
os.environ.pop("AVATAR_ALLOW_GPU", None)
sys.path.insert(0, r"D:\Avatar-Pipeline\gui")
import job_manager as jm, fh_length
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, "lentest_t2"); os.makedirs(T, exist_ok=True)
IMG = r"D:\Avatar-Pipeline\inputs\avatar_bob.png"
T2 = r"D:\Avatar-Pipeline\inputs\t2_ja_60s.wav"
T1C = glob.glob(r"D:\Avatar-Pipeline\inputs\落ち着いた女性-2026-10-31-13-36-*.mp3")[0]
def tone(sec):
    p = os.path.join(T, f"tone_{sec}.wav")
    if not os.path.exists(p):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=300:duration={sec}", "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", p], check=True)
    return p
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
before = {p: sha(p) for p in (T2, T1C)}
L = jm.fh_limits(); print("limits", L["gpu_verified_max_seconds"], L["gpu_verified_max_frames"])
cases = [("T1c の MP3（208）", T1C, False), ("T2 の WAV（60.000秒）", T2, False), ("60.019秒（1,500）", tone(60.019), False),
         ("60.04秒（1,501）", tone(60.04), False), ("60.04秒・確認あり", tone(60.04), True), ("65秒", tone(65), False), ("120秒", tone(120), False),
         ("120秒・確認あり", tone(120), True)]
rows = []
for name, path, allow in cases:
    sec = jm.audio_seconds(path); pl = jm.plan_for(sec, "fh_gpu"); gui_ok = (not pl["unverified"]) or allow
    d = os.path.join(T, "job_" + str(len(rows))); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    wav = os.path.join(d, "audio.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-c:a", "pcm_s16le", wav], check=True)
    json.dump({"job_id": "lentest_t2_" + str(len(rows)), "mode": "gpu", "image": IMG, "audio": wav, "output_dir": os.path.join(d, "out"),
               "seed": 42, "use_face_crop": True, "allow_unverified_length": allow}, open(os.path.join(d, "job.json"), "w", encoding="utf-8"))
    subprocess.run([jm.FH_PY, jm.FH_WORKER, "--job-dir", d], capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))
    st = json.load(open(os.path.join(d, "status.json"), encoding="utf-8"))
    worker_ok = "--allow-gpu が必要" in st.get("message", "")
    rows.append((name, f"{sec:.6f}", fh_length.target_frames(sec), pl["unverified"], allow, gui_ok, worker_ok, gui_ok == worker_ok, st.get("message", "")[:70]))
print("ケース | 秒 | フレーム | 未検証 | 確認 | GUI受付 | ワーカー受付 | 一致 | ワーカーの文")
for r in rows: print(" | ".join(str(x) for x in r))
print("all_agree", all(r[7] for r in rows))
# permit（GUI の GPU 生成モードと同じ。ジョブは作らない）
jobs_before = set(os.listdir(jm.JOBS)); jm.GPU_SESSION.update(enabled=True, rehearsal=True)
base = {"image": IMG, "character_name": "test", "output_dir": os.path.join(T, "out"), "seed": 42, "use_face_crop": True}
for name, path, allow in [("T2 60秒・確認なし", T2, False), ("65秒・確認なし", tone(65), False), ("65秒・確認あり", tone(65), True)]:
    try:
        pid, pl, _ = jm.issue_gpu_permit(dict(base, audio=path, allow_unverified_length=allow)); jm.revoke_gpu_permit(pid)
        print(f"permit {name}: 発行（取り消し）unverified={pl['unverified']}")
    except Exception as e:
        print(f"permit {name}: 拒否 {str(e)[:90]}")
print("新しいジョブ:", set(os.listdir(jm.JOBS)) - jobs_before or "なし", "| permits:", len(jm._PERMITS), "| lock:", os.path.exists(jm.LOCK))
print("入力の音声は変更なし:", all(sha(p) == h for p, h in before.items()))
