@echo off
setlocal
cd /d "%~dp0"
for /f "usebackq delims=" %%P in (`tools\resolve_brain_python.cmd`) do set "BRAIN_PYTHON=%%P"
if not defined BRAIN_PYTHON (
    echo Failed to resolve Python runtime.
    pause
    exit /b 1
)
"%BRAIN_PYTHON%" run_pretrain_ui.py
if errorlevel 1 pause
