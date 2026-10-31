# frame_sink.FrameSink の異常系テスト（GPU・torch を使わない。ffmpeg は自分で起動した子プロセスだけを扱う）
# 使い方：python test_frame_sink.py <一時フォルダ>
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_sink import FrameSink, EncoderError

tmp = sys.argv[1]; os.makedirs(tmp, exist_ok=True)
res = {}
fr = np.zeros((28, 512, 512, 3), dtype=np.uint8)

def expect_error(name, fn):
    try:
        fn(); res[name] = "NG（エラーにならなかった）"
    except EncoderError as e:
        res[name] = "OK：" + str(e)[:120]

# 1. エンコーダーが途中で異常終了 → 次の write で検知する（このテストで起動した ffmpeg を、そのハンドルで止める）
p1 = os.path.join(tmp, "crash.mp4")
s = FrameSink(p1, 1500)
s.write(fr); s.write(fr)
s.proc.kill(); s.proc.wait()
def w():
    for _ in range(5): s.write(fr)
expect_error("encoder_crash_detected_on_write", w)
s.abort()
res["encoder_crash_written_frames"] = s.written

# 2. 書いたフレーム数が目標に足りないまま閉じる → close がエラー（完成品として扱わない）
p2 = os.path.join(tmp, "short.mp4")
s = FrameSink(p2, 100); s.write(fr)
expect_error("short_close_rejected", s.close)

# 3. 既にあるファイルには書かない（上書きしない）
open(os.path.join(tmp, "exists.mp4"), "wb").write(b"keep")
expect_error("no_overwrite", lambda: FrameSink(os.path.join(tmp, "exists.mp4"), 10))
res["existing_file_untouched"] = open(os.path.join(tmp, "exists.mp4"), "rb").read() == b"keep"

# 4. 形式の違うフレーム（float や大きさ違い）は拒否
s = FrameSink(os.path.join(tmp, "badshape.mp4"), 28)
expect_error("bad_dtype_rejected", lambda: s.write(fr.astype(np.float32)))
expect_error("bad_size_rejected", lambda: s.write(np.zeros((28, 256, 256, 3), np.uint8)))
s.abort()

# 5. 目標を超える分は書かない（最終チャンクの余り）：28×4=112 を送って 100 で打ち切る
p5 = os.path.join(tmp, "trim.mp4")
s = FrameSink(p5, 100)
n = sum(s.write(fr) for _ in range(4))
s.close()
res["trim_written"] = n
print(json.dumps(res, ensure_ascii=False, indent=1))
