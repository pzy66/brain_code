param(
    [switch]$NoInstall,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = & (Join-Path $PSScriptRoot "resolve_brain_python.cmd")
if (-not $python) {
    throw "Failed to resolve Python runtime."
}

Push-Location $repoRoot
try {
    & $python -c "import PyInstaller, roslibpy, paramiko" 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($NoInstall) {
            throw "PyInstaller, roslibpy, or paramiko is missing. Re-run without -NoInstall or install .[packaging,hybrid]."
        }
        & $python -m pip install "pyinstaller>=6,<7" "roslibpy>=1.6,<2" "paramiko>=3,<4"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install packaging dependencies."
        }
    }

    & $python -m PyInstaller --noconfirm --clean ".\packaging\BrainRobotWorkbench.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    $distDir = Join-Path $repoRoot "dist\BrainRobotWorkbench"
    $exePath = Join-Path $distDir "BrainRobotWorkbench.exe"
    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Build finished but exe was not found: $exePath"
    }
    $readmeSource = Join-Path $repoRoot "packaging\BrainRobotWorkbench_PORTABLE_README.txt"
    if (Test-Path -LiteralPath $readmeSource) {
        Copy-Item -LiteralPath $readmeSource -Destination (Join-Path $distDir "BrainRobotWorkbench_PORTABLE_README.txt") -Force
    }
    Write-Host "Built: $exePath"

    if (-not $NoZip) {
        $zipPath = Join-Path $repoRoot "dist\BrainRobotWorkbench.zip"
        if (Test-Path -LiteralPath $zipPath) {
            Remove-Item -LiteralPath $zipPath -Force
        }
        Compress-Archive -LiteralPath $distDir -DestinationPath $zipPath -CompressionLevel Optimal
        Write-Host "Zipped: $zipPath"
    }
}
finally {
    Pop-Location
}
