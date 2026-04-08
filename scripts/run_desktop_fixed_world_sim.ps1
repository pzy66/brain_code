param(
    [string]$RobotHost = "127.0.0.1",
    [int]$RobotPort = 8899,
    [string]$TimingProfile = "fast",
    [string]$ScenarioName = "basic",
    [string]$VisionMode = "fixed_cyl_slots"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. $PSScriptRoot\_resolve_hybrid_python.ps1
$PythonExe = Resolve-HybridRunPython -RepoRoot $RepoRoot

& $PythonExe -m hybrid_controller.app `
    --robot-mode fake-remote `
    --robot-host $RobotHost `
    --robot-port $RobotPort `
    --vision-mode $VisionMode `
    --move-source sim `
    --decision-source sim `
    --timing-profile $TimingProfile `
    --scenario-name $ScenarioName
