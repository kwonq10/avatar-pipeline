# 音声の長さと「GPU で検証済みの長さか」の判定（GUI・job_manager・fh_worker の全部がこれを使う。標準ライブラリだけ）
# - 長さは、いつも「PCM の WAV のサンプル数 ÷ サンプルレート」で測る（ワーカーが実際に処理する WAV と同じ値）
#   元の MP3 などを ffprobe で測った値は、エンコーダの遅延・つめ物のぶん少しずれるので、判定には使わない
# - 判定はフレーム数（round(秒 × 25)）で行う。検証済みの範囲は fh_limits.json の gpu_verified_max_frames（例：T1c の 208 フレーム）
#   秒の小数の丸め（例：8.333 と 8.333061）で判定が変わらないようにするため。表示の丸めは判定に使わない
import wave

FPS = 25

def wav_seconds(path):
    """PCM の WAV の長さ（秒）。サンプル数 ÷ サンプルレート"""
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()

def target_frames(seconds):
    """映像のフレーム数（ワーカーと同じ式）"""
    return round(seconds * FPS)

def verified_frames(limits):
    """検証済みのフレーム数。gpu_verified_max_frames が無い古い設定では、秒から計算する"""
    if limits.get("gpu_verified_max_frames") is not None:
        return int(limits["gpu_verified_max_frames"])
    return target_frames(float(limits["gpu_verified_max_seconds"]))

def is_unverified(seconds, limits):
    """GPU 生成で未検証の長さか（目標のフレーム数が、検証済みのフレーム数を超えるか）"""
    return target_frames(seconds) > verified_frames(limits)

def describe(seconds, limits):
    """表示用の文（判定には使わない）"""
    return (f"{seconds:.3f}秒（{target_frames(seconds)} フレーム）・検証済みは {verified_frames(limits)} フレーム"
            f"（{float(limits['gpu_verified_max_seconds']):.3f}秒）まで")
