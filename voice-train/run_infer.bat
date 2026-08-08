@echo off
setlocal
cd /d "%~dp0"

set CAMPPLUS_ONNX=%~dp0campplus.onnx

if "%~1"=="" (
    echo usage: run_infer.bat input.wav [output.wav]
    echo example: run_infer.bat kss_mini\1\1_0000.wav
    pause
    exit /b 1
)

set PY_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
if not exist "%PY_EXE%" set PY_EXE=python

"%PY_EXE%" infer.py --checkpoint checkpoints\best.pt --input %1 --output %2

pause
