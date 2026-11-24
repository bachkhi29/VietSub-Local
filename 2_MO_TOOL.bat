@echo off
setlocal
cd /d "%~dp0"
title LocalBachSub
if not exist ".venv\Scripts\python.exe" (
    echo Hay chay 1_CAI_DAT.bat truoc.
    pause
    exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
where ollama >nul 2>&1
if not errorlevel 1 (
    tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
    if errorlevel 1 start "" /B ollama serve >nul 2>&1
)
echo Dang mo LocalBachSub. Giu cua so nay trong luc xu ly.
echo Dung tool: Ctrl+C; neu hoi Terminate batch job thi nhan Y.
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
