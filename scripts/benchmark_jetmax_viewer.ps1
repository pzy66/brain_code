param(
    [int]$DurationSec = 15,
    [string]$Source = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80",
    [string]$Weights = "",
    [int]$Imgsz = 512,
    [int]$MaxDet = 6,
    [switch]$Fullscreen
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$viewer = Join-Path $projectRoot "05_Vision_Block_Recognition\2026-03_yolo_camera_detection\block_center_ssvep_single.py"
$logsDir = Join-Path $projectRoot "logs"

if (-not (Test-Path -LiteralPath $viewer)) {
    throw "Viewer script not found: $viewer"
}

. $PSScriptRoot\_resolve_hybrid_python.ps1
$python = Resolve-HybridRunPython -RepoRoot $projectRoot

if ([string]::IsNullOrWhiteSpace($Weights)) {
    $Weights = Join-Path (Split-Path -Parent $projectRoot) "dataset\camara\best.pt"
}
if (-not (Test-Path -LiteralPath $Weights)) {
    throw "Weights not found: $Weights"
}

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $logsDir "jetmax_viewer_benchmark_${stamp}.jsonl"
$stderrPath = Join-Path $logsDir "jetmax_viewer_benchmark_${stamp}.stderr.log"

$arguments = @(
    $viewer,
    "--weights", $Weights,
    "--source", $Source,
    "--device", "0",
    "--imgsz", $Imgsz.ToString(),
    "--max-det", $MaxDet.ToString(),
    "--warmup-runs", "1",
    "--exit-after-sec", $DurationSec.ToString()
)
if ($Fullscreen) {
    $arguments += "--fullscreen"
}

function Quote-CmdArg {
    param([string]$Value)
    return '"' + ($Value -replace '"', '""') + '"'
}

Write-Host "Running viewer benchmark..."
Write-Host "source=$Source"
Write-Host "weights=$Weights"
Write-Host "stdout=$stdoutPath"
Write-Host "stderr=$stderrPath"

$oldLocation = Get-Location
Push-Location $projectRoot
try {
    $cmdArgs = @($viewer) + $arguments[1..($arguments.Count - 1)]
    $cmdLine = "{0} {1} 1>{2} 2>{3}" -f (
        Quote-CmdArg $python
    ), (
        ($cmdArgs | ForEach-Object { Quote-CmdArg ([string]$_) }) -join " "
    ), (
        Quote-CmdArg $stdoutPath
    ), (
        Quote-CmdArg $stderrPath
    )
    cmd.exe /d /c $cmdLine
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

$rows = @()
if (Test-Path -LiteralPath $stdoutPath) {
    $rows = Get-Content -LiteralPath $stdoutPath |
        Where-Object { $_.Trim() -ne "" } |
        ForEach-Object { $_ | ConvertFrom-Json }
}

$summary = [ordered]@{
    exit_code = $exitCode
    duration_sec = $DurationSec
    samples = $rows.Count
    source = $Source
    stdout = $stdoutPath
    stderr = $stderrPath
}

if ($rows.Count -gt 0) {
    $summary.capture_fps = [Math]::Round((($rows | Measure-Object -Property capture_fps -Average).Average), 3)
    $summary.app_packet_fps = [Math]::Round((($rows | Measure-Object -Property packet_fps -Average).Average), 3)
    $summary.queue_age_ms = [Math]::Round((($rows | Measure-Object -Property queue_age_ms -Average).Average), 3)
    $summary.infer_ms = [Math]::Round((($rows | Measure-Object -Property infer_ms -Average).Average), 3)
    $summary.post_ms = [Math]::Round((($rows | Measure-Object -Property post_ms -Average).Average), 3)
    $summary.detected_count = [Math]::Round((($rows | Measure-Object -Property detected_count -Average).Average), 3)
    if ($rows.Count -gt 1) {
        $firstTs = [datetime]$rows[0].timestamp
        $lastTs = [datetime]$rows[-1].timestamp
        $spanSec = ($lastTs - $firstTs).TotalSeconds
        if ($spanSec -gt 0) {
            $summary.measured_packet_fps = [Math]::Round((($rows.Count - 1) / $spanSec), 3)
        }
    }
}

$summary | ConvertTo-Json -Compress
