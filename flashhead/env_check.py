# FlashHead 専用環境の基本動作確認（モデルの重みは使わない）
import importlib, json, subprocess, sys, shutil, warnings
warnings.filterwarnings("ignore")
res = {"python": sys.version.split()[0], "executable": sys.executable}
def imp(name, attr="__version__"):
    try:
        m = importlib.import_module(name); res[name] = getattr(m, attr, "ok")
    except Exception as e:
        res[name] = f"NG: {type(e).__name__}: {str(e)[:120]}"
for n in ["torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "numpy", "cv2", "mediapipe",
          "librosa", "soundfile", "pyloudnorm", "imageio", "imageio_ffmpeg", "skimage", "scipy", "einops", "easydict",
          "ftfy", "loguru", "yaml", "PIL", "tokenizers", "huggingface_hub"]:
    imp(n)
import torch, torch.nn.functional as F
res["torch_cuda_build"] = torch.version.cuda
res["cuda_available"] = torch.cuda.is_available()
res["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
res["capability"] = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
res["cudnn"] = torch.backends.cudnn.version()
import torchvision
try:
    x = torch.rand(1, 3, 64, 64, device="cuda"); b = torch.tensor([[0, 0., 0., 32., 32.]], device="cuda")
    res["torchvision_ops_cuda"] = list(torchvision.ops.roi_align(x, b, (8, 8)).shape)
except Exception as e:
    res["torchvision_ops_cuda"] = f"NG: {e}"
import mediapipe as mp
res["mediapipe_face_detection_solution"] = hasattr(mp.solutions, "face_detection")
try:
    import numpy as np
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        r = fd.process(np.zeros((256, 256, 3), np.uint8))
    res["mediapipe_face_detection_run"] = "ok (no face on blank image: %s)" % (r.detections is None)
except Exception as e:
    res["mediapipe_face_detection_run"] = f"NG: {type(e).__name__}: {str(e)[:150]}"
from torch.nn.attention import sdpa_kernel, SDPBackend
q = torch.randn(1, 12, 9216, 128, device="cuda", dtype=torch.bfloat16)   # Pro と同じ head 数・head_dim・33フレーム相当のトークン数
sd = {}
for b in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]:
    try:
        torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel([b]):
            o = F.scaled_dot_product_attention(q, q, q); torch.cuda.synchronize()
        sd[b.name] = "OK (peak %.2f GiB)" % (torch.cuda.max_memory_allocated() / 2**30)
    except Exception as e:
        sd[b.name] = "NG: " + str(e).splitlines()[0][:90]
    torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
o = F.scaled_dot_product_attention(q, q, q); torch.cuda.synchronize()
sd["default(auto)"] = "OK (peak %.2f GiB)" % (torch.cuda.max_memory_allocated() / 2**30)
res["sdpa_bf16_1x12x9216x128"] = sd
import imageio_ffmpeg
res["imageio_ffmpeg_exe"] = imageio_ffmpeg.get_ffmpeg_exe()
ff = shutil.which("ffmpeg"); res["ffmpeg_on_path"] = ff
if ff:
    out = subprocess.run([ff, "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    res["ffmpeg_version"] = out[:80]
    enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    res["ffmpeg_libx264_aac"] = ("libx264" in enc, " aac " in enc)
print(json.dumps(res, ensure_ascii=False, indent=1))
