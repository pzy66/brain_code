@echo off
setlocal

for %%I in ("%~dp0..\..\..") do set "PROJECT_ROOT=%%~fI"
set "SCRIPT=%~dp0async_fbcca_idle_standalone.py"
set "PYTHON_EXE=%HYBRID_PYTHON_EXE%"
if not defined PYTHON_EXE set "PYTHON_EXE=C:\Users\P1233\miniconda3\envs\brain-vision\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Missing interpreter. Tried:
    echo   %HYBRID_PYTHON_EXE%
    echo   C:\Users\P1233\miniconda3\envs\brain-vision\python.exe
    echo   %PROJECT_ROOT%\.venv\Scripts\python.exe
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo Online decoder script not found:
    echo   %SCRIPT%
    exit /b 1
)

echo Using interpreter: %PYTHON_EXE%
pushd "%PROJECT_ROOT%"
"%PYTHON_EXE%" "%SCRIPT%" online %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
