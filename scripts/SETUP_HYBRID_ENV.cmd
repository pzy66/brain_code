@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0setup_desktop_env.ps1" %*
