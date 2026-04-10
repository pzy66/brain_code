param(
    [switch]$SkipRemove
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$condaExe = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
$envFile = Join-Path $repoRoot "environment.brain-vision.yml"
$exportFile = Join-Path $repoRoot "environment.brain-vision.lock.yml"

if (-not (Test-Path -LiteralPath $condaExe)) {
    throw "Conda not found: $condaExe"
}
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Environment file not found: $envFile"
}

if (-not $SkipRemove) {
    & $condaExe env remove -n brain-vision -y
}

& $condaExe env create -f $envFile
& $condaExe env export -n brain-vision > $exportFile

Write-Host "Rebuilt brain-vision and exported lock file:"
Write-Host "  $exportFile"
