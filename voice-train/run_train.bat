@echo off
setlocal
cd /d "%~dp0"

set KSS_DIR=%~dp0kss_mini
set CAMPPLUS_ONNX=%~dp0campplus.onnx
set KOREAN_SPEECH_DIR=
set SAMPLE_WAV=

echo ===================================================
echo Voice Train
echo KSS_DIR = %KSS_DIR%
echo CAMPPLUS_ONNX = %CAMPPLUS_ONNX%
echo ===================================================
echo.

set PY_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
if not exist "%PY_EXE%" set PY_EXE=python

"%PY_EXE%" train.py %*

pause
