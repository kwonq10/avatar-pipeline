@echo off
rem Avatar-Pipeline GUI in GPU generation mode (127.0.0.1:7861 only).
rem GPU permission is NOT stored anywhere: it exists only while this window runs,
rem and every job still needs the in-GUI confirmation (one-time permit) and the backend safety checks.
cd /d D:\Avatar-Pipeline\gui
set PYTHONUTF8=1
"D:\Avatar-Pipeline\gui\.venv\Scripts\python.exe" app.py --gpu-enabled

if errorlevel 1 pause
