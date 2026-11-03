"""Standard-library launcher: diagnose before importing Gradio/job_manager."""
from importlib import metadata
from pathlib import Path
import runpy
import sys

from gui_support import gui_port, port_in_use, port_busy_message, user_error


def main():
    try:
        port = gui_port()
    except ValueError as exc:
        print(f"[起動エラー] {exc}", flush=True)
        return 2
    if port_in_use(port):
        print("[起動エラー] " + port_busy_message(port), flush=True)
        return 3
    try:
        version = metadata.version("gradio")
    except metadata.PackageNotFoundError:
        print("[起動エラー] このPythonにGradioがありません。GUI用の既存仮想環境を指定してください。docs/SETUP.mdを確認してください。", flush=True)
        return 2
    if version != "5.50.0":
        print(f"[起動エラー] 対応するGradioは5.50.0です（現在 {version}）。既存のGUI用環境を指定してください。自動インストールは行いません。", flush=True)
        return 2
    gui = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(gui))
    sys.argv = [str(gui / "app.py"), *sys.argv[1:]]
    print(f"GUIを起動します：http://127.0.0.1:{port}\n終了前に、実行中のジョブがないことを確認してください。ブラウザを閉じるだけでは終了しません。", flush=True)
    try:
        runpy.run_path(str(gui / "app.py"), run_name="__main__")
    except KeyboardInterrupt:
        print("GUIの終了操作を受け取りました。実行中だったジョブの終了を意味するものではありません。", flush=True)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:
        print("[起動エラー] " + user_error(exc, "GUIを起動"), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
