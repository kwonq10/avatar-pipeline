"""CPU-only startup/messages/lock tests. No materials, models, or job generation.

Run with the existing GUI Python: python -B tests_cpu/test_windows_usability.py
Temporary files/processes belong only to these tests. Production jobs are untouched.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "gui/tools"))
from gui_support import gui_port, port_in_use, port_busy_message, require_input, user_error
import launch_gui
import job_manager as jm


class StartupTests(unittest.TestCase):
    def test_port_validation(self):
        for value in ["abc", "0", "65536", ""]:
            with self.assertRaisesRegex(ValueError, "整数"):
                gui_port(value)
        self.assertEqual(gui_port("7863"), 7863)

    def test_busy_port_launcher_does_not_import_app(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); sock.listen()
            port = sock.getsockname()[1]
            self.assertTrue(port_in_use(port))
            env = dict(os.environ, AVATAR_GUI_PORT=str(port), PYTHONUTF8="1")
            r = subprocess.run([sys.executable, "-B", str(ROOT / "gui/tools/launch_gui.py")], env=env, capture_output=True, encoding="utf-8", timeout=10)
        self.assertEqual(r.returncode, 3)
        self.assertIn("強制終了しない", r.stdout)
        self.assertNotIn("Traceback", r.stderr)

    def test_invalid_port_launcher(self):
        env = dict(os.environ, AVATAR_GUI_PORT="not-a-port", PYTHONUTF8="1")
        r = subprocess.run([sys.executable, "-B", str(ROOT / "gui/tools/launch_gui.py")], env=env, capture_output=True, encoding="utf-8", timeout=10)
        self.assertEqual(r.returncode, 2)
        self.assertIn("AVATAR_GUI_PORT", r.stdout)

    def test_missing_gradio(self):
        with patch.object(launch_gui, "port_in_use", return_value=False), patch.object(launch_gui.metadata, "version", side_effect=launch_gui.metadata.PackageNotFoundError), contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(launch_gui.main(), 2)
        self.assertIn("既存仮想環境", text.getvalue())

    def test_other_gradio_version(self):
        with patch.object(launch_gui, "port_in_use", return_value=False), patch.object(launch_gui.metadata, "version", return_value="0.0"), contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(launch_gui.main(), 2)
        self.assertIn("自動インストールは行いません", text.getvalue())

    def test_normal_and_interrupt_exit(self):
        for error in [None, KeyboardInterrupt()]:
            with patch.object(launch_gui, "port_in_use", return_value=False), patch.object(launch_gui.metadata, "version", return_value="5.50.0"), patch.object(launch_gui.runpy, "run_path", side_effect=error), patch.object(sys, "argv", ["launcher"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch_gui.main(), 0)

    def test_startup_failure_message(self):
        with patch.object(launch_gui, "port_in_use", return_value=False), patch.object(launch_gui.metadata, "version", return_value="5.50.0"), patch.object(launch_gui.runpy, "run_path", side_effect=PermissionError("denied")), patch.object(sys, "argv", ["launcher"]), contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(launch_gui.main(), 1)
        self.assertIn("アクセス権", text.getvalue())


class MessageTests(unittest.TestCase):
    def test_missing_inputs(self):
        for label in ["キャラクター画像", "音声ファイル"]:
            with self.assertRaisesRegex(ValueError, label):
                require_input(None, label)

    def test_missing_and_unsupported_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "unsupported.txt"; p.write_text("test")
            with self.assertRaisesRegex(ValueError, "音声形式"):
                require_input(str(p), "音声", jm.AUDIO_EXT)
            with self.assertRaises(FileNotFoundError):
                require_input(str(p) + "missing", "画像")

    def test_recovery_hints(self):
        self.assertIn("アクセス権", user_error(PermissionError()))
        self.assertIn("不要なアプリ", user_error(MemoryError()))
        self.assertIn("冷えるまで", user_error(RuntimeError("GPU 温度 80℃ 未満")))
        self.assertIn("コミット残量", user_error(RuntimeError("コミット残量不足")))
        self.assertIn("start_gui_gpu.cmd", user_error(RuntimeError("この GUI は GPU 生成モードではありません")))
        self.assertIn("絶対パス", user_error(RuntimeError("出力フォルダが不正")))
        self.assertIn("PATH", user_error(FileNotFoundError(2, "missing", "ffprobe")))


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.old_env = os.environ.get("GRADIO_TEMP_DIR")
        os.environ["GRADIO_TEMP_DIR"] = cls.tmp.name
        import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        if cls.old_env is None: os.environ.pop("GRADIO_TEMP_DIR", None)
        else: os.environ["GRADIO_TEMP_DIR"] = cls.old_env
        cls.tmp.cleanup()

    def test_missing_file_inline_messages(self):
        a = self.app
        image = a.on_image(str(Path(self.tmp.name) / "missing.png"))
        audio = a.on_audio(str(Path(self.tmp.name) / "missing.wav"))
        self.assertIn("選び直", image[0]); self.assertEqual(image[1], [])
        self.assertIn("選び直", audio[0]); self.assertFalse(audio[1]["interactive"])

    def test_decoder_errors_are_shown(self):
        a = self.app
        p = Path(self.tmp.name) / "broken.wav"; p.write_bytes(b"not audio")
        with patch.object(jm, "audio_seconds", side_effect=RuntimeError("decode failed")):
            msg, update = a.on_audio(str(p))
        self.assertIn("音声を読み込みできません", msg); self.assertFalse(update["interactive"])
        with patch.object(jm, "check_image", side_effect=PermissionError("denied")):
            msg, gallery, _ = a.on_image(str(p))
        self.assertIn("アクセス権", msg); self.assertEqual(gallery, [])

    def test_unsupported_audio(self):
        p = Path(self.tmp.name) / "not-audio.txt"; p.write_text("test")
        msg, update = self.app.on_audio(str(p))
        self.assertIn("対応していない音声形式", msg); self.assertFalse(update["interactive"])

    def test_invalid_inputs_do_not_start_job(self):
        a = self.app
        with patch.object(jm, "start_job") as start, patch.object(a, "refresh", return_value="unchanged"), patch.object(a.gr, "Warning"):
            self.assertEqual(a.start_test(None, None, "", "", 42, False, {}), "unchanged")
            start.assert_not_called()

    def test_bad_output_directory_does_not_start_job(self):
        a = self.app
        p = Path(self.tmp.name) / "input.wav"; p.write_bytes(b"test")
        with patch.object(jm, "start_job") as start, patch.object(a, "refresh", return_value="unchanged"), patch.object(a.gr, "Warning") as warning:
            a.start_test(str(p), str(p), "", "relative/path", 42, False, {})
            start.assert_not_called()
            self.assertIn("絶対パス", warning.call_args.args[0])

    def test_not_gpu_mode_does_not_issue_permit(self):
        a = self.app
        with patch.object(a, "GPU_MODE", False), patch.object(jm, "issue_gpu_permit") as permit, patch.object(a.gr, "Warning"):
            a.prepare_gpu(None, None, "", "", 42, False, False, None)
            permit.assert_not_called()

    def test_download_and_mute_controls_preserved(self):
        a = self.app
        self.assertEqual(a.download_btn.label, "⬇ 完成動画をダウンロード")
        self.assertFalse(a.download_btn.interactive)
        self.assertIn("v.muted = true", a.HEAD)
        self.assertIn("@property --rb-angle", a.HEAD)


class JobLifecycleTests(unittest.TestCase):
    def test_live_lock_stop_request_and_stale_lock(self):
        with tempfile.TemporaryDirectory() as d:
            lock = str(Path(d) / ".gpu.lock"); job = Path(d) / "fixture"; job.mkdir()
            with patch.object(jm, "LOCK", lock), patch.object(jm, "JOBS", d):
                Path(lock).write_text(json.dumps({"job_id": "fixture", "pid": os.getpid()}))
                self.assertEqual(jm.current_job(), "fixture")
                self.assertEqual(jm.stop_job(), "fixture")
                self.assertTrue((job / "STOP").exists())
                Path(lock).write_text(json.dumps({"job_id": "fixture", "pid": 0}))
                self.assertIsNone(jm.current_job())
                self.assertFalse(Path(lock).exists())

    @unittest.skipUnless(os.name == "nt", "Windows process-group semantics")
    def test_independent_cpu_child_survives_parent_exit(self):
        # A harmless CPU process uses exactly the flags of the real monitor.
        # It exits itself after 1s. No production process is terminated.
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'],creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid,flush=True)"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 0)
        pid = int(r.stdout.strip()); self.assertTrue(jm._pid_alive(pid))
        deadline = time.monotonic() + 4
        while jm._pid_alive(pid) and time.monotonic() < deadline: time.sleep(.05)
        self.assertFalse(jm._pid_alive(pid))



@unittest.skipUnless(os.name == "nt", "Windows CMD and Windows PowerShell 5.1")
class CmdDiagnosticsTests(unittest.TestCase):
    def test_cmd_diagnostics_stdout_stderr(self):
        # Use real CMD parsing: a mocked exit code missed truncated UTF-8 echo lines.
        # The busy listener prevents app/Gradio/model imports, even for the GPU cmd.
        powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            ps = folder / "invoke.ps1"
            ps.write_text(
                '$ErrorActionPreference = "Stop"\n'
                'if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -ne 1) { exit 99 }\n'
                '[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)\n'
                '$OutputEncoding = [Console]::OutputEncoding\n'
                '& $env:ComSpec /d /c $args[0]\n'
                'exit $LASTEXITCODE\n', encoding="ascii")
            for script in ["start_gui.cmd", "start_gui_gpu.cmd"]:
                target = ROOT / "gui" / script
                for cp in [932, 65001]:
                    for host in ["cmd", "powershell51"]:
                        for case in ["missing", "busy", "missing-special-path"]:
                            with self.subTest(script=script, cp=cp, host=host, case=case):
                                env = dict(os.environ, PYTHONUTF8="1")
                                env.pop("AVATAR_GUI_PYTHON", None)
                                with socket.socket() as sock:
                                    sock.bind(("127.0.0.1", 0)); sock.listen()
                                    env["AVATAR_GUI_PORT"] = str(sock.getsockname()[1])
                                    if case == "busy":
                                        env["AVATAR_GUI_PYTHON"] = sys.executable
                                    elif case == "missing-special-path":
                                        env["AVATAR_GUI_PYTHON"] = str(folder / "missing (GPU) & test.exe")
                                    wrapper = folder / "invoke.cmd"
                                    wrapper.write_bytes((
                                        f'@echo off\r\nchcp {cp} >nul\r\n'
                                        f'call "{target}"\r\nexit /b %errorlevel%\r\n'
                                    ).encode("utf-8"))
                                    command = [os.environ.get("ComSpec", "cmd.exe"), "/d", "/c", str(wrapper)]
                                    if host == "powershell51":
                                        command = [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(ps), str(wrapper)]
                                    result = subprocess.run(command, env=env, input="\n", capture_output=True,
                                                            encoding="utf-8", errors="replace", timeout=20)
                                self.assertEqual(result.returncode, 3 if case == "busy" else 2)
                                self.assertEqual(result.stderr, "")
                                self.assertNotIn("\ufffd", result.stdout)
                                if case == "busy":
                                    self.assertIn("強制終了しない", result.stdout)
                                    self.assertIn("http://127.0.0.1:", result.stdout)
                                else:
                                    expected_path = env.get("AVATAR_GUI_PYTHON", str(ROOT / "gui/.venv/Scripts/python.exe"))
                                    for line in [
                                        "[起動エラー] GUI用のPython仮想環境が見つかりません。",
                                        f'確認した場所: "{expected_path}"',
                                        "既存環境を使う場合はAVATAR_GUI_PYTHONにGUI用python.exeの絶対パスを指定してください。",
                                        "自動で環境を作成・変更しません。docs\\SETUP.mdの「GUIの起動」を確認してください。",
                                    ]:
                                        self.assertIn(line, result.stdout)

if __name__ == "__main__":
    unittest.main(verbosity=2)
