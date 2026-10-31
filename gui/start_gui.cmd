@echo off
rem Avatar-Pipeline 動画制作 GUI を起動する（127.0.0.1:7861 のみ。モデルは読み込まない）
cd /d D:\Avatar-Pipeline\gui
set PYTHONUTF8=1
"D:\Avatar-Pipeline\gui\.venv\Scripts\python.exe" app.py

if errorlevel 1 pause
