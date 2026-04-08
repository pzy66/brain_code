@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0benchmark_jetmax_viewer.ps1" %*
