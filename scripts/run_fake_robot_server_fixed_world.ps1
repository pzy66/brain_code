param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8899,
    [string]$TimingProfile = "fast",
    [string]$ScenarioName = "basic",
    [string]$VisionMode = "fixed_cyl_slots"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. $PSScriptRoot\_resolve_hybrid_python.ps1
$PythonExe = Resolve-HybridRunPython -RepoRoot $RepoRoot

& $PythonExe -m hybrid_controller.debug.fake_robot_server `
    --host $HostName `
    --port $Port `
    --timing-profile $TimingProfile `
    --scenario-name $ScenarioName `
    --vision-mode $VisionMode
