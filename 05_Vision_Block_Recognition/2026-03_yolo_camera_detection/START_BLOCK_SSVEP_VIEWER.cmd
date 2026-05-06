@echo off
setlocal

for %%I in ("%~dp0..\..") do set "CODE_ROOT=%%~fI"
for %%I in ("%~dp0..\..\..") do set "BRAIN_ROOT=%%~fI"
set "RESOLVER=%CODE_ROOT%\tools\resolve_brain_python.cmd"
set "SCRIPT=%~dp0block_center_ssvep_single.py"
set "WEIGHTS=%CODE_ROOT%\datasets\vision\models\best.pt"
set "SOURCE=http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"

if not exist "%RESOLVER%" (
  echo Interpreter resolver not found:
  echo %RESOLVER%
  set "EXIT_CODE=1"
  goto :fail
)

for /f "usebackq delims=" %%I in (`call "%RESOLVER%"`) do set "PYTHON=%%I"

if not exist "%PYTHON%" (
  echo Python interpreter not found:
  echo %PYTHON%
  set "EXIT_CODE=1"
  goto :fail
)

if not exist "%SCRIPT%" (
  echo Vision script not found:
  echo %SCRIPT%
  set "EXIT_CODE=1"
  goto :fail
)

if /I "%~1"=="--help" goto :show_help
if /I "%~1"=="-h" goto :show_help

if not exist "%WEIGHTS%" (
  echo Weight file not found:
  echo Tried:
  echo   %CODE_ROOT%\datasets\vision\models\best.pt
  echo Or set BRAIN_VISION_WEIGHTS to another local .pt file.
  set "EXIT_CODE=1"
  goto :fail
)

echo [05 realtime] Starting small block recognition
echo [05 realtime] Python: %PYTHON%
echo [05 realtime] Weights: %WEIGHTS%
echo [05 realtime] Source: "%SOURCE%"
echo.

pushd "%CODE_ROOT%"
"%PYTHON%" "%SCRIPT%" ^
  --weights "%WEIGHTS%" ^
  --source "%SOURCE%" ^
  --device auto ^
  --imgsz 512 ^
  --max-det 6 ^
  --warmup-runs 1 ^
  %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" goto :fail
exit /b %EXIT_CODE%

:show_help
pushd "%CODE_ROOT%"
"%PYTHON%" "%SCRIPT%" --help
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" goto :fail
exit /b %EXIT_CODE%

:fail
echo.
echo [05 realtime] Failed with exit code %EXIT_CODE%.
echo [05 realtime] Script: %SCRIPT%
echo [05 realtime] Python: %PYTHON%
echo [05 realtime] Weights: %WEIGHTS%
echo [05 realtime] Source: "%SOURCE%"
echo.
if /I not "%BRAIN_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
