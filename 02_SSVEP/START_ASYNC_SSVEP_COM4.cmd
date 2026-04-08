@echo off
setlocal
cd /d "%~dp0"

set "LAUNCHER=%~dp02026-04_async_fbcca_idle_decoder\START_ASYNC_SSVEP_VALIDATION_UI.cmd"

if not exist "%LAUNCHER%" (
    echo Missing launcher:
    echo   %LAUNCHER%
    exit /b 1
)

echo Starting async SSVEP validation UI on COM4 ^(board_id=0^)
call "%LAUNCHER%" --serial-port COM4 --board-id 0 --windowed %*
exit /b %ERRORLEVEL%
