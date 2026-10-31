# FlashHead Pro の読み込みだけのテスト（推論・生成はしない）。作業フォルダは SoulX-FlashHead（設定ファイルを相対パスで開くため）。
# 実行例（承認後）:
#   python ../../tools/run_stage.py --name fh_load_test --out <runs\...> --min-avail-gib 2 --job-mem-gib 12 \
#     --min-commit-headroom-gib 5 --max-vram-mib 11800 --timeout-s 900 --cwd D:\Avatar-Pipeline\flashhead\SoulX-FlashHead \
#     -- D:\Avatar-Pipeline\flashhead\.venv\Scripts\python.exe ..\tools\load_test.py
import os, sys, json, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("FLASHHEAD_LOWMEM_LOAD", "1")
T0 = time.time()
def log(stage, **kw): print(json.dumps({"t": round(time.time() - T0, 2), "stage": stage, **kw}, ensure_ascii=False), flush=True)
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "SoulX-FlashHead-1_3B")
W2V = os.path.join(ROOT, "models", "wav2vec2-base-960h")
torch.cuda.init(); log("cuda_ready", device=torch.cuda.get_device_name(0), torch=torch.__version__)
# 公式のコードは変えずに、3つのモデルの読み込み箇所を包んで、それぞれの時間と VRAM の増加を記録する
import flash_head.utils.lowmem_load as lml
import flash_head.wan.modules as wanmods
import flash_head.audio_analysis.wav2vec2 as w2v
PARTS = {}
def timed(name, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated(); t0 = time.time()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        PARTS[name] = {"seconds": round(time.time() - t0, 2), "vram_delta_gib": round((torch.cuda.memory_allocated() - m0) / 2**30, 3)}
        log(name + "_loaded", **PARTS[name])
        return out
    return wrap
lml.load_model_lowmem = timed("pro_model(lowmem)", lml.load_model_lowmem)
_WanVAE = wanmods.WanVAE
class TimedWanVAE(_WanVAE):
    def __init__(self, *a, **k):
        timed("wan_vae", lambda: _WanVAE.__init__(self, *a, **k))()
wanmods.WanVAE = TimedWanVAE
w2v.Wav2Vec2Model.from_pretrained = timed("wav2vec2", w2v.Wav2Vec2Model.from_pretrained)
from flash_head.inference import get_pipeline
t = time.time()
pipe = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type="pro", wav2vec_dir=W2V)
torch.cuda.synchronize()
def info(m):
    ps = list(m.parameters())
    return {"devices": sorted({str(p.device) for p in ps}), "dtypes": sorted({str(p.dtype) for p in ps}), "params": sum(p.numel() for p in ps)}
log("pipeline_loaded", seconds=round(time.time() - t, 2), model=info(pipe.model), vae=info(pipe.vae.model),
    audio_encoder=info(pipe.audio_encoder), vae_class=type(pipe.vae).__name__, use_ltx=pipe.use_ltx,
    vram_alloc_gib=round(torch.cuda.memory_allocated() / 2**30, 2), vram_peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
# 生成を始める条件（今回は判定するだけで、生成はしない）
import ctypes
class MS(ctypes.Structure):
    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                ("tf", ctypes.c_ulonglong), ("af", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
ms = MS(); ms.l = ctypes.sizeof(ms); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
free_vram = torch.cuda.mem_get_info()[0] / 2**30
gate = {"avail_ram_gib": round(ms.ap / 2**30, 2), "commit_headroom_gib": round(ms.af / 2**30, 2), "free_vram_gib": round(free_vram, 2)}
log("generation_gate", ok=(gate["avail_ram_gib"] >= 3 and gate["commit_headroom_gib"] >= 8 and gate["free_vram_gib"] >= 2), **gate,
    rule="空きRAM>=3GiB かつ コミット残量>=8GiB かつ 空きVRAM>=2GiB（判定のみ。生成はしない）")
log("done_load_only", parts=PARTS)   # 推論はしない
