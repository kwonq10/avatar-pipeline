"""GUI-only messages and startup checks. No model imports or job control."""
import os
import socket
import subprocess


def gui_port(value=None):
    raw = os.environ.get("AVATAR_GUI_PORT", "7861") if value is None else value
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise ValueError("AVATAR_GUI_PORTには1〜65535の整数を指定してください。通常は7861です。") from None
    if not 1 <= port <= 65535:
        raise ValueError("AVATAR_GUI_PORTには1〜65535の整数を指定してください。通常は7861です。")
    return port


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def port_busy_message(port):
    return (f"ポート {port} は別のプログラムが使用しています。新しいGUIは起動しません。\n"
            f"既存GUIなら http://127.0.0.1:{port} を利用してください。\n"
            "別のGUIを試す場合はAVATAR_GUI_PORTに空いている別ポートを指定してください。\n"
            "実行中のジョブを確認せず、既存GUIやPythonを強制終了しないでください。")


def require_input(path, label, extensions=None):
    if not path:
        raise ValueError(f"{label}を選んでください（必須）。左側の入力欄から選択できます。")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label}が見つかりません。移動・削除されていないか確認し、選び直してください。")
    if extensions and os.path.splitext(path)[1].lower() not in extensions:
        raise ValueError("対応していない音声形式です。WAV・MP3・FLAC・M4A・AAC・OGG・Opusを選んでください。")


def user_error(exc, action="操作"):
    """Keep the backend's Japanese rejection and add an actionable GUI hint."""
    detail = str(exc)
    if isinstance(exc, PermissionError):
        hint = "ファイル・保存先のアクセス権と、他のアプリで使用中でないか確認してください。保存先は書き込み可能なフォルダを選んでください。"
    elif isinstance(exc, FileNotFoundError) and exc.filename and os.path.basename(str(exc.filename)).lower() in ("ffmpeg", "ffmpeg.exe", "ffprobe", "ffprobe.exe"):
        hint = "ffmpeg・ffprobeが見つかりません。PATHとdocs/SETUP.mdを確認してください。"
    elif isinstance(exc, FileNotFoundError):
        hint = "入力を選び直してください。実行環境が見つからない場合はdocs/SETUP.mdを確認してください。"
    elif isinstance(exc, MemoryError) or "コミット残量" in detail or "空きRAM" in detail:
        hint = "不要なアプリを閉じ、処理状況の空きRAM・コミット残量を確認してから再試行してください。安全基準は変更しないでください。"
    elif "温度" in detail:
        hint = "GPUが冷えるまで待ち、温度を取得できることを確認してください。安全基準を下げず、自動で再試行しないでください。"
    elif isinstance(exc, subprocess.TimeoutExpired):
        hint = "読み込みが時間内に終わりませんでした。ファイルを別のプレイヤーで開けるか確認してから、選び直してください。"
    elif "出力フォルダ" in detail or "保存先" in detail:
        hint = "保存先には書き込み可能なフォルダの絶対パスを指定してください。"
    elif "GPU 生成モード" in detail or "未承認" in detail:
        hint = "通常モードではGPU生成できません。GPUを使う場合だけstart_gui_gpu.cmdで起動し、毎回の安全確認を行ってください。"
    else:
        hint = "入力ファイルを別のアプリで開けるか、保存先と実行環境が利用できるか確認してください。"
    return f"{action}できませんでした。{hint}\n詳細：{type(exc).__name__}: {detail}"
