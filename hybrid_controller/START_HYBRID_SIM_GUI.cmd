@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0..\scripts\run_desktop_fixed_world_sim.ps1" %*
