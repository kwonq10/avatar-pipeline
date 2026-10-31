# T1c：FlashHead Pro の GPU 生成を1回だけ実行して、終わるまで待つ（自動の再実行はしない）
# AVATAR_ALLOW_GPU=1 はこのプロセスの環境変数だけで渡す（恒久的には設定しない）
import os, sys, time, json, glob
sys.path.insert(0, r"D:\Avatar-Pipeline\gui")
import job_manager as jm

img = r"D:\Avatar-Pipeline\inputs\avatar_bob.png"
aud = glob.glob(r"D:\Avatar-Pipeline\inputs\落ち着いた女性-2026-10-31-13-36-*.mp3")
assert len(aud) == 1, aud
aud = aud[0]
chk = jm.check_image(img)
face = bool(chk.get("recommend_use_face_crop"))
print("image_check", chk.get("level"), "recommend_use_face_crop", face, flush=True)
payload = {"image": img, "audio": aud, "character_name": "T1c 比較用（avatar_bob）", "output_dir": r"D:\Avatar-Pipeline\outputs\t1c",
           "seed": 42, "use_face_crop": face, "allow_unverified_length": True}
t0 = time.time()
jid = jm.start_job("fh_gpu", payload)
print("job", jid, flush=True)
last = None
while jm.current_job() == jid:
    st = jm.read_status(jid)
    cur = (st.get("state"), st.get("chunk"), st.get("frames_written"))
    if cur != last:
        print(f"{time.time() - t0:6.1f}s", cur, st.get("message"), flush=True); last = cur
    time.sleep(2)
st = jm.read_status(jid)
print("wall_s", round(time.time() - t0, 1))
print("state", st.get("state"), "|", st.get("message"))
print("output", st.get("output"), "| incomplete", st.get("incomplete_files"))
print("monitor", json.dumps(st.get("monitor"), ensure_ascii=False))
res = st.get("result") or {}
print("result", json.dumps(res, ensure_ascii=False))
