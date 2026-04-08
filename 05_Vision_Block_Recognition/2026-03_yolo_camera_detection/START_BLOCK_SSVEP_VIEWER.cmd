@echo off
setlocal

for %%I in ("%~dp0..\..\..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0block_center_ssvep_single.py"
set "WEIGHTS=%PROJECT_ROOT%\dataset\camara\best.pt"
set "SOURCE=http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"

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

if not exist "%WEIGHTS%" (
  echo Weight file not found:
  echo %WEIGHTS%
  exit /b 1
)

pushd "%PROJECT_ROOT%"
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
