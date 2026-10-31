# 1つの工程（LivePortrait / MuseTalk / 音声 など）を子プロセスとして実行し、メモリを記録しながら異常時に確実に止める。
#
# EchoMimic では ComfyUI の /interrupt（協調的な中断）が Sampler 実行中に効かず、空きRAMが 0.28GiB まで下がった。
# ここでは「お願いして止める」のではなく、OS の仕組みで止める：
#   1. Windows Job Object に子プロセスを入れ、ジョブ全体のコミット上限（JobMemoryLimit）を掛ける
#      → 上限を超える確保は OS が失敗させる（ページファイルを食い尽くさない）
#   2. 空きRAM・VRAM・経過時間を監視し、しきい値を割ったら TerminateJobObject でプロセスツリーごと強制終了
#   3. 監視側が落ちても KILL_ON_JOB_CLOSE で子プロセスは道連れに終了する
#   4. （任意）GPU 温度：--max-gpu-temp-c を付けたときだけ、その温度以上が --gpu-temp-hold-s 秒続いたら強制終了する
#      --require-gpu-temp を付けると、温度を読めない状態が --gpu-temp-missing-s 秒続いた場合も強制終了する
#      （付けなければ従来どおり。他の工程の動作は変わらない）
# 標準ライブラリだけで動く（どの Python でも可）。VRAM と温度は nvidia-smi を 1 秒ごとに読む（パスは環境変数 AVATAR_NVIDIA_SMI で変えられる。テスト用）。
#
# 使い方:
#   python run_stage.py --name lp_test --out <ログフォルダ> [--min-avail-gib 2.0] [--job-mem-gib 9]
#                       [--max-vram-mib 11000] [--timeout-s 1800] -- <実行するコマンド...>
import argparse, csv, ctypes, json, os, subprocess, sys, threading, time
from ctypes import wintypes as wt

k32 = ctypes.WinDLL("kernel32", use_last_error=True)

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                   "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9
PROCESS_ALL_ACCESS = 0x1F0FFF
CREATE_SUSPENDED = 0x4
GiB = 1024 ** 3

k32.CreateJobObjectW.restype = wt.HANDLE
k32.OpenProcess.restype = wt.HANDLE

def mem():
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / GiB, (m.ullTotalPageFile - m.ullAvailPageFile) / GiB, m.ullAvailPageFile / GiB

def job_info(job):
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info), None)
    return info

class Vram(threading.Thread):
    """nvidia-smi を 1 秒ごとに読む：VRAM 使用量と GPU 温度。読めなかったときは temp_ok_at が更新されない"""
    def __init__(self):
        super().__init__(daemon=True); self.used = None; self.temp = None; self.temp_ok_at = None; self.stop = False
    def run(self):
        while not self.stop:
            try:
                out = subprocess.run([os.environ.get("AVATAR_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                f = [x.strip() for x in out[0].split(",")]
                self.used = int(f[0])
                try:
                    self.temp = float(f[1]); self.temp_ok_at = time.time()
                except (IndexError, ValueError):
                    self.temp = None
            except Exception:
                self.temp = None
            time.sleep(1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-avail-gib", type=float, default=2.0, help="空き物理RAMがこれを下回ったら強制終了")
    ap.add_argument("--job-mem-gib", type=float, default=9.0, help="ジョブ全体のコミット上限（OSが強制）")
    ap.add_argument("--max-vram-mib", type=int, default=11264, help="VRAM使用量がこれを超えたら強制終了")
    ap.add_argument("--stop-file", default=None, help="このファイルが現れたら、プロセスツリーごと停止する（GUI の停止ボタン用）")
    ap.add_argument("--min-commit-headroom-gib", type=float, default=0.0, help="システムのコミット残量（上限−使用中）がこれを下回ったら強制終了")
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--max-gpu-temp-c", type=float, default=None, help="GPU 温度がこれ以上の状態が --gpu-temp-hold-s 秒続いたら強制終了（省略時は温度で止めない）")
    ap.add_argument("--gpu-temp-hold-s", type=float, default=10.0)
    ap.add_argument("--require-gpu-temp", action="store_true", help="温度を読めない状態が --gpu-temp-missing-s 秒続いたら強制終了")
    ap.add_argument("--gpu-temp-missing-s", type=float, default=15.0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    os.makedirs(a.out, exist_ok=True)

    job = k32.CreateJobObjectW(None, None)
    lim = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    lim.JobMemoryLimit = int(a.job_mem_gib * GiB)
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(lim), ctypes.sizeof(lim)):
        sys.exit(f"SetInformationJobObject failed: {ctypes.get_last_error()}")

    avail0, commit0, _ = mem()
    t0 = time.time()
    stdout = open(os.path.join(a.out, f"{a.name}_stdout.txt"), "w", encoding="utf-8", errors="replace")
    # 一時停止状態で起動 → ジョブに入れてから再開（ジョブ外で子プロセスが生まれる隙をなくす）
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")   # 子の標準出力をファイルに書くとき cp932 で落ちないように
    p = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdout=stdout, stderr=subprocess.STDOUT, creationflags=CREATE_SUSPENDED)
    hproc = k32.OpenProcess(PROCESS_ALL_ACCESS, False, p.pid)
    if not k32.AssignProcessToJobObject(job, hproc):
        p.kill(); sys.exit(f"AssignProcessToJobObject failed: {ctypes.get_last_error()}")
    # メインスレッドを再開
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess(wt.HANDLE(hproc))

    vr = Vram(); vr.start()
    peak = {"avail_min_gib": avail0, "commit_max_gib": commit0, "job_commit_peak_gib": 0.0, "vram_max_mib": 0, "gpu_temp_max_c": None}
    hot_since = None
    abort = None
    with open(os.path.join(a.out, f"{a.name}_memory.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "avail_gib", "sys_commit_gib", "job_commit_peak_gib", "vram_used_mib", "gpu_temp_c"])
        while p.poll() is None:
            avail, commit, headroom = mem()
            peak["commit_headroom_min_gib"] = min(peak.get("commit_headroom_min_gib", headroom), headroom)
            jpk = job_info(job).PeakJobMemoryUsed / GiB
            v = vr.used
            t = time.time() - t0
            tc = vr.temp
            w.writerow([f"{t:.1f}", f"{avail:.3f}", f"{commit:.3f}", f"{jpk:.3f}", v if v is not None else "", tc if tc is not None else ""])
            f.flush()                            # GUI が実行中の値（空きRAM・VRAM・温度）を読めるようにする（判定には影響しない）
            if tc is not None: peak["gpu_temp_max_c"] = max(peak["gpu_temp_max_c"] or tc, tc)
            if a.max_gpu_temp_c is not None and tc is not None and tc >= a.max_gpu_temp_c:
                hot_since = hot_since or time.time()
            else:
                hot_since = None
            peak["avail_min_gib"] = min(peak["avail_min_gib"], avail)
            peak["commit_max_gib"] = max(peak["commit_max_gib"], commit)
            peak["job_commit_peak_gib"] = max(peak["job_commit_peak_gib"], jpk)
            if v is not None: peak["vram_max_mib"] = max(peak["vram_max_mib"], v)
            if a.stop_file and os.path.exists(a.stop_file): abort = "ユーザーの停止"
            elif avail < a.min_avail_gib: abort = f"空きRAM {avail:.2f}GiB < {a.min_avail_gib}"
            elif headroom < a.min_commit_headroom_gib: abort = f"コミット残量 {headroom:.2f}GiB < {a.min_commit_headroom_gib}"
            elif v is not None and v > a.max_vram_mib: abort = f"VRAM {v}MiB > {a.max_vram_mib}"
            elif t > a.timeout_s: abort = f"時間切れ {t:.0f}s > {a.timeout_s}"
            elif hot_since and time.time() - hot_since >= a.gpu_temp_hold_s:
                abort = f"GPU 温度 {tc:.0f}℃ ≥ {a.max_gpu_temp_c:.0f}℃ が {a.gpu_temp_hold_s:.0f}秒続いた"
            elif a.require_gpu_temp and time.time() - (vr.temp_ok_at or t0) > a.gpu_temp_missing_s:
                abort = f"GPU 温度を {a.gpu_temp_missing_s:.0f}秒以上読めない"
            if abort:
                k32.TerminateJobObject(job, 1)   # プロセスツリーごと即時終了（協調的な中断ではない）
                p.wait(timeout=30)
                break
            time.sleep(a.interval)
    vr.stop = True
    stdout.close()
    peak["job_commit_peak_gib"] = max(peak["job_commit_peak_gib"], job_info(job).PeakJobMemoryUsed / GiB)
    res = {"name": a.name, "cmd": cmd, "exit_code": p.returncode, "aborted": abort, "elapsed_s": round(time.time() - t0, 1),
           "before": {"avail_gib": round(avail0, 2), "sys_commit_gib": round(commit0, 2)},
           "peak": {k: (round(x, 2) if isinstance(x, float) else x) for k, x in peak.items()},
           "limits": {"min_commit_headroom_gib": a.min_commit_headroom_gib, "min_avail_gib": a.min_avail_gib, "job_mem_gib": a.job_mem_gib, "max_vram_mib": a.max_vram_mib, "timeout_s": a.timeout_s,
                      "max_gpu_temp_c": a.max_gpu_temp_c, "gpu_temp_hold_s": a.gpu_temp_hold_s, "require_gpu_temp": a.require_gpu_temp}}
    json.dump(res, open(os.path.join(a.out, f"{a.name}_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    k32.CloseHandle(job)
    sys.exit(0 if (p.returncode == 0 and not abort) else 1)

if __name__ == "__main__":
    main()
