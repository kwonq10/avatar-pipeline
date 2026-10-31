# 逐次書き出しの接続テスト（推論なし・GPU 未使用）を job_manager 経由で1本実行し、結果をまとめて表示する。
# 使い方：python stream_dryrun_test.py <音声> [--stop-after-sec N] [--dry-sec-per-chunk 0.7]
import argparse, csv, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import job_manager as jm

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--image", default=os.path.join(jm.ROOT, "inputs", "avatar_bob.png"))
ap.add_argument("--stop-after-sec", type=float, default=None, help="この秒数の後に GUI の停止と同じ操作（STOP ファイル）をする")
ap.add_argument("--dry-sec-per-chunk", type=float, default=0.7)
ap.add_argument("--fail-at-chunk", type=int, default=None, help="テスト用：このチャンクで故障させる（dry_run だけ）")
ap.add_argument("--test-max-temp", type=float, default=None, help="テスト用：温度の停止しきい値を下げる（接続テストだけ）")
ap.add_argument("--out", default=os.path.join(jm.ROOT, "outputs", "stream_test"))
a = ap.parse_args()

jid = jm.start_job("fh_dryrun", {"image": a.image, "audio": a.audio, "character_name": "接続テスト（逐次書き出し）",
                                  "output_dir": a.out, "seed": 42, "use_face_crop": True, "dry_sec_per_chunk": a.dry_sec_per_chunk,
                                  **({"dry_fail_at_chunk": a.fail_at_chunk} if a.fail_at_chunk is not None else {}),
                                  **({"test_max_gpu_temp_c": a.test_max_temp} if a.test_max_temp is not None else {})})
d = os.path.join(jm.JOBS, jid)
print("job", jid, "timeout", json.load(open(os.path.join(d, "job.json"), encoding="utf-8"))["plan_estimate"], flush=True)
t0 = time.time(); stopped = False
while jm.current_job() == jid:
    if a.stop_after_sec and not stopped and time.time() - t0 > a.stop_after_sec:
        jm.stop_job(); stopped = True; ts = time.time(); print("stop requested", flush=True)
    time.sleep(0.2)
if stopped:
    print("stop latency (s)", round(time.time() - ts, 2))
st = jm.read_status(jid)
print("state", st.get("state"), "|", st.get("message"))
print("output", st.get("output"), "| incomplete", st.get("incomplete_files"))
print("monitor", json.dumps(st.get("monitor"), ensure_ascii=False))
res = st.get("result") or {}
print("result", json.dumps({k: res.get(k) for k in ("ok", "video_frames", "target_frames", "frames_written", "video_seconds",
                                                   "audio_seconds_out", "audio_seconds_in", "av_diff_seconds", "checks", "loop_s", "error")}, ensure_ascii=False))
mp = os.path.join(d, "worker_mem.csv")
if os.path.exists(mp):
    rows = list(csv.DictReader(open(mp, encoding="utf-8")))
    if rows:
        def col(k): return [float(r[k]) for r in rows if r[k]]
        n = len(rows); q = max(1, n // 10)
        for k in ("worker_private_mib", "worker_ws_mib", "ffmpeg_private_mib", "ffmpeg_ws_mib"):
            v = col(k)
            if v:
                print(f"{k}: first10% avg {sum(v[:q]) / q:.1f} / last10% avg {sum(v[-q:]) / q:.1f} / min {min(v):.1f} / max {max(v):.1f} (rows {n})")
# 埋め込んだ通し番号を完成品から読み戻す（重複・欠落の検査）
if st.get("output"):
    r = subprocess.run([jm.FH_PY, os.path.join(jm.ROOT, "flashhead", "tools", "verify_frames.py"), st["output"], str(res.get("target_frames"))],
                       capture_output=True, text=True)
    print("verify", r.stdout.strip()[:600])
# 既存の履歴に載るか
print("history_row", [row for row in jm.history(50) if row[0] == jid])
