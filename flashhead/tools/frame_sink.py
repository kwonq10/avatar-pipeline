# フレームを ffmpeg の標準入力へ逐次送り、H.264 の映像だけの MP4 を作る（全フレームを RAM にためない）。
#
# - rawvideo（rgb24・W×H・fps）を stdin に流す。パイプが詰まったら write がブロックする（= ffmpeg が追いつくまで安全に待つ）
# - 書くフレーム数は target_frames で打ち切る（最終チャンクの余分なフレームは書かない）
# - 書く前に ffmpeg が生きているかを確かめ、終わっていれば（異常終了）EncoderError
# - close() で stdin を閉じて終了を待ち、戻り値と書いたフレーム数を確かめる。どちらかがおかしければ EncoderError
# - 出力先が既にあれば開始しない（上書きしない。ffmpeg にも -n を付ける）
# - ffmpeg の stderr はファイルに書く（PIPE にして読まないと詰まるため）
# numpy だけを使う（torch には依存しない。dry_run でも同じコードを通す）
import os, subprocess, shutil, ctypes
from ctypes import wintypes as wt

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class EncoderError(RuntimeError):
    pass


class FrameSink:
    def __init__(self, path, target_frames, width=512, height=512, fps=25, crf=16, stderr_path=None):
        if os.path.exists(path):
            raise EncoderError(f"出力先が既にあります（上書きしません）：{path}")
        self.path, self.target, self.w, self.h = path, int(target_frames), width, height
        self.written = 0
        self.frame_bytes = width * height * 3
        self.stderr_path = stderr_path or (path + ".ffmpeg.log")
        self._err = open(self.stderr_path, "w", encoding="utf-8", errors="replace")
        cmd = [FFMPEG, "-hide_banner", "-v", "error", "-n",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-bf", "0", "-pix_fmt", "yuv420p",
               "-r", str(fps), path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err, bufsize=0,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _err_tail(self):
        try:
            self._err.flush()
            with open(self.stderr_path, encoding="utf-8", errors="replace") as f:
                return f.read()[-400:]
        except Exception:
            return ""

    @property
    def remaining(self):
        return self.target - self.written

    def write(self, frames):
        """frames: numpy uint8 (N, H, W, 3)。target を超える分は捨てる。書いた枚数を返す"""
        import numpy as np
        if self.proc.poll() is not None:
            raise EncoderError(f"ffmpeg が途中で終了しました（code {self.proc.returncode}）：{self._err_tail()}")
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (self.h, self.w, 3):
            raise EncoderError(f"フレームの形式が違います：{frames.dtype} {frames.shape}（必要：uint8 (N,{self.h},{self.w},3)）")
        n = min(len(frames), self.remaining)
        if n <= 0:
            return 0
        buf = np.ascontiguousarray(frames[:n])
        try:
            self.proc.stdin.write(memoryview(buf).cast("B"))     # パイプが詰まればここで待つ
        except (BrokenPipeError, OSError) as e:
            raise EncoderError(f"ffmpeg へ書けません（{e}）：{self._err_tail()}")
        self.written += n
        return n

    def close(self, timeout=600):
        """正常終了：stdin を閉じて待つ。戻り値 0 かつ target ちょうどでなければ EncoderError"""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.abort()
            raise EncoderError(f"ffmpeg が {timeout}秒以内に終わりません")
        self._err.close()
        if rc != 0:
            raise EncoderError(f"ffmpeg の異常終了（code {rc}）：{self._err_tail()}")
        if self.written != self.target:
            raise EncoderError(f"書いたフレーム数が足りません：{self.written}/{self.target}")
        return self.written

    def abort(self):
        """異常時：ffmpeg を止める（出力は不完全のまま残る。呼ぶ側で完成品と区別する）"""
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=30)
        except Exception:
            pass
        try:
            self._err.close()
        except Exception:
            pass


# ---- メモリの計測（Windows API。psutil を使わない）----
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def proc_mem_mib(pid=None):
    """そのプロセスの今の private（コミット）と working set（MiB）。pid=None なら自分"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.OpenProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
    own = pid is None
    k32.GetCurrentProcess.restype = wt.HANDLE
    h = k32.GetCurrentProcess() if own else k32.OpenProcess(0x1000 | 0x0010, False, int(pid))   # QUERY_LIMITED | VM_READ
    if not h:
        return None
    try:
        c = _PMC(); c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return None
        return {"private_mib": round(c.PrivateUsage / 2**20, 1), "ws_mib": round(c.WorkingSetSize / 2**20, 1)}
    finally:
        if not own:
            k32.CloseHandle(h)
