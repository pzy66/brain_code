@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0..\scripts\run_desktop_fixed_world_real.ps1" -RobotHost 192.168.149.1 %*
