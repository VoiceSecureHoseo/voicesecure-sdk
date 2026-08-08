@echo off
setlocal
cd /d "%~dp0"

set CAMPPLUS_ONNX=%~dp0campplus.onnx

echo ===================================================
echo Voice Train - evaluate
echo ===================================================
echo.

set PY_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
if not exist "%PY_EXE%" set PY_EXE=python

"%PY_EXE%" evaluate.py --checkpoint checkpoints\best.pt --wav_dir kss_mini %*

pause
