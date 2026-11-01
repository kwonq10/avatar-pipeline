@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Avatar-Pipeline GUI in GPU generation mode (127.0.0.1:7861 only).
rem GPU permission is NOT stored anywhere: it exists only while this window runs,
rem and every job still needs the in-GUI confirmation (one-time permit) and the backend safety checks.
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
".venv\Scripts\python.exe" app.py --gpu-enabled

set "GUI_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%GUI_EXIT_CODE%"=="0" pause
exit /b %GUI_EXIT_CODE%
