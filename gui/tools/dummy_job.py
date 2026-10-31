# ダミーのワーカー（GPU・モデル・ネットワークを使わない）。GUI ⇄ run_stage.py ⇄ ワーカー のつながりと、進捗・停止・履歴を確かめるためのもの。
# 本番のワーカー（FlashHead Pro）と同じ約束で status.json を書く：
#   {"state": "running|done|failed", "progress": 0〜1, "message": str, "chunk": i, "chunks": n, "output": mp4 のパス}
# 最後に ffmpeg のテストパターンで 3秒の「DUMMY」動画（512x512・25fps・無音ではなく 440Hz のテスト音）を作る。
import argparse, json, os, subprocess, time, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--job-dir", required=True)
ap.add_argument("--chunks", type=int, default=8)
ap.add_argument("--sec-per-chunk", type=float, default=1.0)
a = ap.parse_args()
st_path = os.path.join(a.job_dir, "status.json")

def put(**kw):
    tmp = st_path + ".tmp"
    json.dump(kw, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, st_path)

job = json.load(open(os.path.join(a.job_dir, "job.json"), encoding="utf-8"))
for i in range(a.chunks):
    put(state="running", progress=i / (a.chunks + 1), message=f"ダミーのチャンク {i + 1}/{a.chunks}（GPU 未使用）", chunk=i + 1, chunks=a.chunks)
    time.sleep(a.sec_per_chunk)

put(state="running", progress=a.chunks / (a.chunks + 1), message="ダミー動画を書き出し中", chunk=a.chunks, chunks=a.chunks)
out = os.path.join(a.job_dir, "dummy_output.mp4")
ff = shutil.which("ffmpeg") or "ffmpeg"
cmd = [ff, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=512x512:rate=25:duration=3",
       "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=24000",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    put(state="failed", progress=1.0, message="ffmpeg に失敗: " + r.stderr[-300:])
    raise SystemExit(1)
put(state="done", progress=1.0, message="完了（ダミー。実際の生成ではありません）", chunk=a.chunks, chunks=a.chunks, output=out)
