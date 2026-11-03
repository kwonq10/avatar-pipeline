@echo off
setlocal EnableExtensions DisableDelayedExpansion
for /f "tokens=2 delims=:" %%A in ('chcp') do set "GUI_OLD_CP=%%A"
chcp 65001 >nul
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
if defined AVATAR_GUI_PYTHON (
    set "GUI_PYTHON=%AVATAR_GUI_PYTHON%"
) else (
    set "GUI_PYTHON=%CD%\.venv\Scripts\python.exe"
)
if not exist "%GUI_PYTHON%" goto missing_python
"%GUI_PYTHON%" -B tools\launch_gui.py
set "GUI_EXIT_CODE=%ERRORLEVEL%"
goto finish
:missing_python
echo [起動エラー] GUI用のPython仮想環境が見つかりません。

echo 確認した場所: "%GUI_PYTHON%"

echo 既存環境を使う場合はAVATAR_GUI_PYTHONにGUI用python.exeの絶対パスを指定してください。

echo 自動で環境を作成・変更しません。docs\SETUP.mdの「GUIの起動」を確認してください。

set "GUI_EXIT_CODE=2"
:finish
if not "%GUI_EXIT_CODE%"=="0" pause
popd
if defined GUI_OLD_CP chcp %GUI_OLD_CP% >nul
exit /b %GUI_EXIT_CODE%
