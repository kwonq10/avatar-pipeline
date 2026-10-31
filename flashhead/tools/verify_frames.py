# dry_run の出力 MP4 から、各フレームに埋め込んだ通し番号（上端 64px・16bit の白黒ブロック）を読み戻し、
# 0, 1, 2, … , N-1 と1つずつ並んでいるか（重複・欠落・順番の入れ替わりがないか）を確かめる。
# ffmpeg の stdout から1フレームずつ読むので、長い動画でもメモリは一定。
# 使い方：python verify_frames.py <mp4> [expected_frames]
import subprocess, sys, json, shutil
import numpy as np

SIZE = 512
path = sys.argv[1]
expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
ff = shutil.which("ffmpeg") or "ffmpeg"
p = subprocess.Popen([ff, "-v", "error", "-i", path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
                     stdout=subprocess.PIPE, bufsize=SIZE * SIZE * 4)
n = 0; bad = []
while True:
    buf = p.stdout.read(SIZE * SIZE)
    if not buf:
        break
    if len(buf) < SIZE * SIZE:
        bad.append({"frame": n, "error": "途中で切れたフレーム"}); break
    fr = np.frombuffer(buf, dtype=np.uint8).reshape(SIZE, SIZE)
    val = 0
    for b in range(16):
        if fr[16:48, b * 32 + 8:b * 32 + 24].mean() > 128:
            val |= 1 << b
    if val != n and len(bad) < 20:
        bad.append({"frame": n, "decoded": val})
    n += 1
rc = p.wait()
res = {"file": path, "frames": n, "expected": expected, "sequence_ok": not bad and rc == 0 and (expected is None or n == expected),
       "mismatches": bad, "ffmpeg_rc": rc}
print(json.dumps(res, ensure_ascii=False))
sys.exit(0 if res["sequence_ok"] else 1)
