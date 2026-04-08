$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. $PSScriptRoot\_resolve_hybrid_python.ps1

$PythonExe = Resolve-HybridRunPython -RepoRoot $RepoRoot
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements-hybrid-controller.txt
& $PythonExe -m pip install -r requirements-hybrid-runtime-optional.txt
& $PythonExe -m pip install requests paramiko

Write-Host "brain-vision environment is ready: $PythonExe"
