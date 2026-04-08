@echo off
setlocal
cd /d "%~dp0"

set "LAUNCHER=%~dp02026-04_async_fbcca_idle_decoder\START_ASYNC_SSVEP_ONLINE.cmd"
set "PROFILE=%~dp02026-04_async_fbcca_idle_decoder\profiles\default_profile.json"

if not exist "%LAUNCHER%" (
    echo Missing launcher:
    echo   %LAUNCHER%
    exit /b 1
)

echo Starting async SSVEP online decoder on COM4 ^(board_id=0^)
call "%LAUNCHER%" --serial-port COM4 --board-id 0 --profile "%PROFILE%" %*
exit /b %ERRORLEVEL%
