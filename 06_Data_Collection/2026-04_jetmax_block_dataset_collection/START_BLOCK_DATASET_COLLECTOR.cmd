@echo off
setlocal

for %%I in ("%~dp0..\..\..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0block_dataset_collector.py"

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

pushd "%PROJECT_ROOT%"
"%PYTHON%" "%SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
