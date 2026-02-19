@echo off
echo ===================================================
echo   Shield-Ryzen V2 — Real-Time Deepfake Detection
echo ===================================================
echo.
echo [1] Launch Detection System (NPU Mode)
echo [2] Validate System Integrity (Run Tests)
echo [3] Export Torch Model to ONNX
echo [4] Quantize ONNX Model (INT8)
echo [5] Exit
echo.
set /p choice="Select Action [1-5]: "

if "%choice%"=="1" goto run
if "%choice%"=="2" goto validate
if "%choice%"=="3" goto export
if "%choice%"=="4" goto quantize
if "%choice%"=="5" goto exit

:run
python shield.py --audit
pause
goto exit

:validate
python validate_system.py
pause
goto exit

:export
python export_onnx.py
pause
goto exit

:quantize
python quantize_ryzen.py
pause
goto exit

:exit
exit
