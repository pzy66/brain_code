@echo off
setlocal

for %%I in ("%~dp0..\..") do set "CODE_ROOT=%%~fI"
for %%I in ("%~dp0..\..\..") do set "BRAIN_ROOT=%%~fI"
set "RESOLVER=%CODE_ROOT%\tools\resolve_brain_python.cmd"
set "SCRIPT=%~dp0block_center_ssvep_single.py"
set "WEIGHTS=%CODE_ROOT%\hybrid_controller\models\vision\best.pt"
if not exist "%WEIGHTS%" set "WEIGHTS=%BRAIN_ROOT%\dataset\camara\best.pt"
set "SOURCE=http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"

if not exist "%RESOLVER%" (
  echo Interpreter resolver not found:
  echo %RESOLVER%
  exit /b 1
)

for /f "usebackq delims=" %%I in (`call "%RESOLVER%"`) do set "PYTHON=%%I"

if not exist "%PYTHON%" (
  echo Python interpreter not found:
  echo %PYTHON%
  exit /b 1
)

if not exist "%SCRIPT%" (
  echo Vision script not found:
  echo %SCRIPT%
  exit /b 1
)

if /I "%~1"=="--help" goto :show_help
if /I "%~1"=="-h" goto :show_help

if not exist "%WEIGHTS%" (
  echo Weight file not found:
  echo Tried:
  echo   %CODE_ROOT%\hybrid_controller\models\vision\best.pt
  echo   %BRAIN_ROOT%\dataset\camara\best.pt
  exit /b 1
)

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

exit /b %EXIT_CODE%

:show_help
pushd "%CODE_ROOT%"
"%PYTHON%" "%SCRIPT%" --help
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
