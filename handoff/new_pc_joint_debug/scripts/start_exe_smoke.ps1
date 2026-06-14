param(
    [string]$ExePath = ""
)

$ErrorActionPreference = "Stop"

if (-not $ExePath) {
    $candidate = Get-ChildItem -Path . -Recurse -Filter "BrainRobotWorkbench.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $candidate) {
        throw "BrainRobotWorkbench.exe not found under current directory. Pass -ExePath explicitly."
    }
    $ExePath = $candidate.FullName
}

$exe = Get-Item -LiteralPath $ExePath
$args = @(
    "--demo-connected",
    "--smoke-test-ms", "1000",
    "--no-camera-auto-start",
    "--no-vision-auto-start",
    "--no-eeg-signal-auto-start"
)

$process = Start-Process -FilePath $exe.FullName -ArgumentList $args -WorkingDirectory $exe.DirectoryName -Wait -PassThru
Write-Host "Exe: $($exe.FullName)"
Write-Host "ExitCode: $($process.ExitCode)"
exit $process.ExitCode
