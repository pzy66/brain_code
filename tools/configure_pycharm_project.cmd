@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0configure_pycharm_project.ps1" %*
exit /b %errorlevel%
