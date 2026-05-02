@echo off
setlocal

for %%I in ("%~dp0..\..") do set "CODE_ROOT=%%~fI"
set "RESOLVER=%CODE_ROOT%\tools\resolve_brain_python.cmd"
set "SCRIPT=%~dp003_collect_images.py"

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
  echo Collector script not found:
  echo %SCRIPT%
  exit /b 1
)

pushd "%CODE_ROOT%"
"%PYTHON%" "%SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
