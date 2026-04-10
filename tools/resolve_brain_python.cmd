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

set "CANDIDATE_1=%USERPROFILE%\miniconda3\envs\brain-vision\python.exe"
set "CANDIDATE_2=%USERPROFILE%\anaconda3\envs\brain-vision\python.exe"
set "CANDIDATE_3=%USERPROFILE%\mambaforge\envs\brain-vision\python.exe"

for %%I in ("%CANDIDATE_1%" "%CANDIDATE_2%" "%CANDIDATE_3%") do (
  if exist "%%~fI" (
    echo %%~fI
    exit /b 0
  )
)

>&2 echo [resolve-brain-python] brain-vision interpreter not found.
>&2 echo [resolve-brain-python] Set BRAIN_PYTHON_EXE or create env:
>&2 echo [resolve-brain-python]   C:\Users\P1233\miniconda3\envs\brain-vision\python.exe
exit /b 1
