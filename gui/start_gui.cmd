@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Avatar-Pipeline 動画制作 GUI を起動する（127.0.0.1:7861 のみ。モデルは読み込まない）
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] GUI Python environment was not found.
    echo Expected: "%CD%\.venv\Scripts\python.exe"
    echo See ..\docs\SETUP.md before starting the GUI.
    popd
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py

set "GUI_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%GUI_EXIT_CODE%"=="0" pause
exit /b %GUI_EXIT_CODE%
