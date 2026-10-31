# GUI の GPU 許可（permit）のバックエンドのテスト。リハーサルモードだけで行い、GPU 推論はしない
# 一部のテストは、条件を満たさない状態を作るために sys_mem / gpu_status をこのプロセスの中だけで差し替える（コードは変更しない）
import os, sys, time, json, glob
os.environ.pop("AVATAR_ALLOW_GPU", None)
sys.path.insert(0, r"D:\Avatar-Pipeline\gui")
import job_manager as jm

IMG = r"D:\Avatar-Pipeline\inputs\avatar_bob.png"
JA = glob.glob(r"D:\Avatar-Pipeline\inputs\落ち着いた女性-2026-10-31-13-36-*.mp3")[0]
T12 = r"D:\Avatar-Pipeline\gui\tmp\ui_test\dummy_tone_12s.wav"
OUT = r"D:\Avatar-Pipeline\outputs\gui_gpu_rehearsal"
base = {"image": IMG, "audio": JA, "character_name": "rehearsal", "output_dir": OUT, "seed": 42, "use_face_crop": True, "allow_unverified_length": False}
results = []
def expect_fail(name, fn, must=""):
    try:
        r = fn(); results.append((name, "NG（拒否されなかった）", str(r)[:80]))
    except Exception as e:
        ok = must in str(e)
        results.append((name, "OK（拒否）" if ok else "要確認（別の理由で拒否）", str(e)[:150]))
def expect_ok(name, fn):
    try:
        r = fn(); results.append((name, "OK", str(r)[:120])); return r
    except Exception as e:
        results.append((name, "NG（失敗）", str(e)[:200])); return None
real_mem, real_gpu = jm.sys_mem, jm.gpu_status
good_mem = lambda: {"avail_gib": 8.0, "headroom_gib": 22.0, "commit_limit_gib": 31.9}

# 0. GPU 生成モードでない
expect_fail("通常モード：permit の発行", lambda: jm.issue_gpu_permit(dict(base)), "GPU 生成モード")
expect_fail("通常モード：permit 付きの開始", lambda: jm.start_job("fh_gpu", dict(base), permit="0" * 32), "GPU 生成モード")
expect_fail("環境変数なし・permit なしの開始", lambda: jm.start_job("fh_gpu", dict(base)), "未承認")

jm.GPU_SESSION.update(enabled=True, rehearsal=True)
t = time.time(); pre = jm.gpu_precheck(); print("precheck(実際の値)", round(time.time() - t, 1), "s", json.dumps(pre["items"], ensure_ascii=False))
real_ok = pre["ok"]
if not real_ok:
    print("※ 今の PC の値では開始前の条件を満たさない → 成功の経路のテストでは sys_mem だけを合格の値に差し替える（リハーサル＝GPU 未使用）")

# 1. 入力の検査
expect_fail("音声なし", lambda: jm.issue_gpu_permit(dict(base, audio=None)), "音声ファイル")
expect_fail("未検証の長さ（12秒）で確認なし", lambda: jm.issue_gpu_permit(dict(base, audio=T12)), "未検証")
# 2. 開始前の条件の不合格（差し替え）
jm.sys_mem = lambda: {"avail_gib": 8.0, "headroom_gib": 15.0, "commit_limit_gib": 31.9}
expect_fail("コミット残量 15GiB", lambda: jm.issue_gpu_permit(dict(base)), "コミット残量")
jm.sys_mem = lambda: {"avail_gib": 2.5, "headroom_gib": 22.0, "commit_limit_gib": 31.9}
expect_fail("空きRAM 2.5GiB", lambda: jm.issue_gpu_permit(dict(base)), "空きRAM")
jm.sys_mem = good_mem
jm.gpu_status = lambda: {"temp_c": None, "vram_used_mib": None, "vram_total_mib": None}
expect_fail("GPU 温度を読めない", lambda: jm.issue_gpu_permit(dict(base)), "GPU 温度を読める")
jm.gpu_status = lambda: {"temp_c": 81.0, "vram_used_mib": 1000.0, "vram_total_mib": 12288.0}
expect_fail("GPU 温度 81℃", lambda: jm.issue_gpu_permit(dict(base)), "80℃ 未満")
jm.gpu_status = lambda: {"temp_c": 55.0, "vram_used_mib": 5000.0, "vram_total_mib": 12288.0}
expect_fail("空き VRAM 7,288MiB", lambda: jm.issue_gpu_permit(dict(base)), "空き VRAM")
jm.gpu_status = real_gpu
# 3. permit の不正な使い方
expect_fail("偽の permit", lambda: jm.start_job("fh_gpu", dict(base), permit="f" * 32), "使用済み")
pid = expect_ok("permit の発行（8.333秒・検証済み）", lambda: jm.issue_gpu_permit(dict(base))[0])
expect_fail("確認後に seed を変えた", lambda: jm.start_job("fh_gpu", dict(base, seed=7), permit=pid), "入力")
expect_fail("同じ permit をもう一度（不合格でも消える）", lambda: jm.start_job("fh_gpu", dict(base), permit=pid), "使用済み")
pid = expect_ok("permit の発行（期限切れの確認用）", lambda: jm.issue_gpu_permit(dict(base))[0])
if pid: jm._PERMITS[pid]["expires"] = time.time() - 1
expect_fail("期限切れの permit", lambda: jm.start_job("fh_gpu", dict(base), permit=pid), "期限")
pid = expect_ok("permit の発行（開始時の再検査の確認用）", lambda: jm.issue_gpu_permit(dict(base))[0])
jm.sys_mem = lambda: {"avail_gib": 8.0, "headroom_gib": 16.0, "commit_limit_gib": 31.9}
expect_fail("発行後に条件が悪化 → 開始時に拒否", lambda: jm.start_job("fh_gpu", dict(base), permit=pid), "コミット残量")
jm.sys_mem = real_mem if real_ok else good_mem
# 4. 成功（リハーサル：8.333秒の日本語音声）→ 完了
pid = expect_ok("permit の発行（成功の経路）", lambda: jm.issue_gpu_permit(dict(base))[0])
jid = expect_ok("permit で開始（リハーサル）", lambda: jm.start_job("fh_gpu", dict(base), permit=pid))
expect_fail("使った permit の再利用", lambda: jm.start_job("fh_gpu", dict(base), permit=pid), "使用済み")
if jid:
    expect_fail("実行中に次の permit", lambda: jm.issue_gpu_permit(dict(base)), "実行中のジョブ")
    while jm.current_job() == jid: time.sleep(0.5)
    st = jm.read_status(jid); job = json.load(open(os.path.join(jm.JOBS, jid, "job.json"), encoding="utf-8"))
    cmd = json.load(open(os.path.join(jm.JOBS, jid, "job_result.json"), encoding="utf-8")).get("cmd", [])
    results.append(("リハーサルの完了", "OK" if st.get("state") == "done" and (st.get("result") or {}).get("ok") else "NG",
                    f"state={st.get('state')} frames={(st.get('result') or {}).get('video_frames')} mode={job.get('mode')} "
                    f"--allow-gpu={'--allow-gpu' in cmd} source={job['gpu_permission']['source']} out={st.get('output')}"))
# 5. 未検証の長さ（12秒）を確認したうえで → 開始 → 停止
pid = expect_ok("未検証（12秒）を確認して permit", lambda: jm.issue_gpu_permit(dict(base, audio=T12, allow_unverified_length=True))[0])
jid2 = expect_ok("未検証の permit で開始（リハーサル）", lambda: jm.start_job("fh_gpu", dict(base, audio=T12, allow_unverified_length=True), permit=pid))
if jid2:
    time.sleep(4); jm.stop_job()
    while jm.current_job() == jid2: time.sleep(0.3)
    st = jm.read_status(jid2); job = json.load(open(os.path.join(jm.JOBS, jid2, "job.json"), encoding="utf-8"))
    results.append(("リハーサルの停止", "OK" if st.get("state") == "stopped" and not st.get("output") else "NG",
                    f"state={st.get('state')} msg={st.get('message')} allow_unverified={job.get('allow_unverified_length')} incomplete={bool(st.get('incomplete_files'))}"))
jm.sys_mem, jm.gpu_status = real_mem, real_gpu
print("\n".join(f"{r[1]:22s} | {r[0]} | {r[2]}" for r in results))
print("lock_left", os.path.exists(jm.LOCK), "permits_left", len(jm._PERMITS))
