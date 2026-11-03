@echo off
setlocal
cd /d "%~dp0"
title VietSub Local - Cai dat
echo VietSub Local - Cai dat lan dau. Can Internet va vai GB dung luong.
echo Khong can quyen Administrator.
echo.
if exist ".venv\Scripts\python.exe" goto dependencies
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
    py -3.11 -m venv .venv
    goto checkvenv
)
py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
    py -3.12 -m venv .venv
    goto checkvenv
)
echo Chua thay Python 3.11/3.12 64-bit.
echo Cai Python 3.11 tu https://www.python.org/downloads/release/python-3119/
echo Chon Windows installer (64-bit), bat Python Launcher va Add Python to PATH.
echo Sau do chay lai file nay. Xem BAT_DAU.html de co huong dan.
pause
exit /b 1
:checkvenv
if not exist ".venv\Scripts\python.exe" goto failed
:dependencies
".venv\Scripts\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 ".venv\Scripts\python.exe" -m ensurepip --upgrade
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Dang cai thu vien NVIDIA cho RTX 3050...
".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
if errorlevel 1 (
    echo Chua cai duoc thu vien GPU. App van co the thu bang CPU + small.
    echo Giu lai anh loi de kiem tra, hoac chay lai file nay.
)
echo.
where ollama >nul 2>&1
if errorlevel 1 goto needollama
echo Dang tai mo hinh dich qwen3:4b. Lan dau khoang 2.5 GB.
ollama pull qwen3:4b
if errorlevel 1 (
    echo Chua tai duoc mo hinh. Mo ung dung Ollama roi chay: ollama pull qwen3:4b
    pause
    exit /b 1
)
echo.
echo Xong. Chay 2_MO_TOOL.bat de bat dau.
pause
exit /b 0
:needollama
echo Chua thay Ollama. Cai tu https://ollama.com/download/windows
echo Mo Ollama, dong cua so nay, roi chay lai 1_CAI_DAT.bat.
pause
exit /b 1
:failed
echo.
echo Cai dat chua thanh cong. Chup lai phan loi o tren de kiem tra.
pause
exit /b 1
