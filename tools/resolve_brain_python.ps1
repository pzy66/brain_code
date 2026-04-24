param()

$override = $env:BRAIN_PYTHON_EXE
if ($override) {
    if (Test-Path -LiteralPath $override) {
        Write-Output (Resolve-Path -LiteralPath $override).Path
        exit 0
    }
    Write-Error "[resolve-brain-python] BRAIN_PYTHON_EXE is set but missing: $override"
    exit 1
}

$projectPython = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\python.exe"
if (Test-Path -LiteralPath $projectPython) {
    Write-Output (Resolve-Path -LiteralPath $projectPython).Path
    exit 0
}

$candidates = @(
    (Join-Path $env:USERPROFILE "miniconda3\envs\brain-vision\python.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\envs\brain-vision\python.exe"),
    (Join-Path $env:USERPROFILE "mambaforge\envs\brain-vision\python.exe")
)

foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        Write-Output (Resolve-Path -LiteralPath $candidate).Path
        exit 0
    }
}

Write-Error "[resolve-brain-python] project .venv and brain-vision interpreter not found. Set BRAIN_PYTHON_EXE or create the local .venv / conda env."
exit 1
