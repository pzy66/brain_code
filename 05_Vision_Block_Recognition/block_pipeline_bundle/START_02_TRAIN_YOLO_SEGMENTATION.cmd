@echo off
setlocal

for %%I in ("%~dp0..\..") do set "CODE_ROOT=%%~fI"
for %%I in ("%~dp0..\..\..") do set "BRAIN_ROOT=%%~fI"
set "RESOLVER=%CODE_ROOT%\tools\resolve_brain_python.cmd"
set "SCRIPT=%~dp002_train_yolo_segmentation.py"

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
  echo Training script not found:
  echo %SCRIPT%
  exit /b 1
)

pushd "%CODE_ROOT%"
"%PYTHON%" "%SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
