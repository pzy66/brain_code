@echo off
setlocal

set "OVERRIDE=%BRAIN_PYTHON_EXE%"
if defined OVERRIDE (
  if exist "%OVERRIDE%" (
    echo %OVERRIDE%
    exit /b 0
  )
  >&2 echo [resolve-brain-python] BRAIN_PYTHON_EXE is set but missing: %OVERRIDE%
  exit /b 1
)

set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%\.venv\python.exe") do (
  if exist "%%~fI" (
    echo %%~fI
    exit /b 0
  )
)

set "CANDIDATE_1=%USERPROFILE%\miniconda3\envs\brain-vision\python.exe"
set "CANDIDATE_2=%USERPROFILE%\anaconda3\envs\brain-vision\python.exe"
set "CANDIDATE_3=%USERPROFILE%\mambaforge\envs\brain-vision\python.exe"

for %%I in ("%CANDIDATE_1%" "%CANDIDATE_2%" "%CANDIDATE_3%") do (
  if exist "%%~fI" (
    echo %%~fI
    exit /b 0
  )
)

>&2 echo [resolve-brain-python] project .venv and brain-vision interpreter not found.
>&2 echo [resolve-brain-python] Set BRAIN_PYTHON_EXE or create env:
>&2 echo [resolve-brain-python]   %%CD%%\.venv\python.exe
>&2 echo [resolve-brain-python]   %%USERPROFILE%%\miniconda3\envs\brain-vision\python.exe
exit /b 1
