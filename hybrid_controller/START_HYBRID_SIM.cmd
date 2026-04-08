@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0..\scripts\run_fake_robot_server_fixed_world.ps1" %*
