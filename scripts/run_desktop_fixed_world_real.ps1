param(
    [string]$RobotHost = "192.168.149.1",
    [int]$RobotPort = 8888,
    [string]$TimingProfile = "formal",
    [string]$ScenarioName = "basic",
    [string]$VisionMode = "fixed_cyl_slots"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. $PSScriptRoot\_resolve_hybrid_python.ps1
$PythonExe = Resolve-HybridRunPython -RepoRoot $RepoRoot

& $PythonExe -m hybrid_controller.app `
    --robot-mode real `
    --robot-host $RobotHost `
    --robot-port $RobotPort `
    --vision-mode $VisionMode `
    --move-source sim `
    --decision-source sim `
    --timing-profile $TimingProfile `
    --scenario-name $ScenarioName `
    --stage-motion-sec 300 `
    --continue-motion-sec 300
