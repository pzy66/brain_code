param(
    [string]$WorkspaceRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$brainCodeRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
if (-not $WorkspaceRoot) {
    $WorkspaceRoot = (Split-Path -Parent $brainCodeRoot)
}
$workspace = (Resolve-Path -LiteralPath $WorkspaceRoot).Path

function Test-UnderRoot {
    param(
        [string]$Path,
        [string]$Root
    )
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd("\")
    return $fullPath.Equals($fullRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $fullPath.StartsWith($fullRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)
}

function Add-CleanTarget {
    param(
        [System.Collections.Generic.HashSet[string]]$Set,
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not (Test-UnderRoot -Path $resolved -Root $workspace)) {
        throw "refusing path outside workspace: $resolved"
    }
    if ($resolved -ieq $workspace -or $resolved -ieq $brainCodeRoot) {
        throw "refusing to clean workspace root: $resolved"
    }
    [void]$Set.Add($resolved)
}

Write-Output "[clean] workspace=$workspace"
Write-Output "[clean] brain_code=$brainCodeRoot"
Write-Output "[clean] dry_run=$([bool]$DryRun)"

$targets = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

$excludedRoots = @(
    (Join-Path $brainCodeRoot ".git"),
    (Join-Path $brainCodeRoot "01_MI\mi_classifier_latest\datasets"),
    (Join-Path $brainCodeRoot "01_MI\mi_classifier_latest\runtime"),
    (Join-Path $brainCodeRoot "02_SSVEP\artifacts\datasets"),
    (Join-Path $brainCodeRoot "02_SSVEP\artifacts\deployed_profiles"),
    (Join-Path $brainCodeRoot "02_SSVEP\artifacts\runs"),
    (Join-Path $brainCodeRoot "hybrid_controller\models")
) | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { (Resolve-Path -LiteralPath $_).Path }

function Test-ExcludedRoot {
    param([string]$Path)
    foreach ($excluded in $excludedRoots) {
        if (Test-UnderRoot -Path $Path -Root $excluded) {
            return $true
        }
    }
    return $false
}

function Get-CandidateDirectories {
    param([string]$Root)
    $stack = [System.Collections.Generic.Stack[string]]::new()
    $stack.Push((Resolve-Path -LiteralPath $Root).Path)
    while ($stack.Count -gt 0) {
        $current = $stack.Pop()
        Get-ChildItem -LiteralPath $current -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $path = $_.FullName
            if (-not (Test-ExcludedRoot -Path $path)) {
                Write-Output $path
                $stack.Push($path)
            }
        }
    }
}

$directPatterns = @(
    "__pycache__",
    ".pytest_cache",
    ".pytest_tmp*",
    ".tmp*",
    ".ultralytics",
    "pytest-cache-files-*",
    "pytest_tmp*",
    "pytest_temp*",
    "tmp_pytest*",
    "tmp_ssvep_wheels",
    "ssvep_gpu_runtime"
)

# The workspace root is local material. Only top-level cache/temp names are
# considered there; recursive cleaning is limited to the formal Git repository.
foreach ($root in @($workspace, $brainCodeRoot)) {
    foreach ($pattern in $directPatterns) {
        Get-ChildItem -LiteralPath $root -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like $pattern } |
            ForEach-Object { Add-CleanTarget -Set $targets -Path $_.FullName }
    }
}

$nestedNames = @(
    "__pycache__",
    ".pytest_cache",
    ".ultralytics"
)

foreach ($name in $nestedNames) {
    Get-CandidateDirectories -Root $brainCodeRoot |
        Where-Object { (Split-Path -Leaf $_) -eq $name } |
        ForEach-Object { Add-CleanTarget -Set $targets -Path $_ }
}

$nestedNamePatterns = @(
    ".pytest_tmp*",
    ".tmp*",
    "pytest-cache-files-*",
    "pytest_tmp*",
    "pytest_temp*",
    "tmp_pytest*"
)

foreach ($pattern in $nestedNamePatterns) {
    Get-CandidateDirectories -Root $brainCodeRoot |
        Where-Object { (Split-Path -Leaf $_) -like $pattern } |
        ForEach-Object { Add-CleanTarget -Set $targets -Path $_ }
}

$explicitRelativeTargets = @(
    "artifacts\pytest_tmp",
    "artifacts\pytest_tmp_hc_comm",
    "brain_code\02_SSVEP\artifacts\gpu_runtime\cupy_cache",
    "brain_code\02_SSVEP\artifacts\gpu_runtime\tmp",
    "brain_code\02_SSVEP\artifacts\stimulus_current_pc_smoke.png",
    "brain_code\02_SSVEP\artifacts\stimulus_current_pc_smoke_report.json",
    "brain_code\02_SSVEP\artifacts\ui_smoke_collection.png",
    "brain_code\02_SSVEP\artifacts\ui_smoke_realtime.png"
)

foreach ($relative in $explicitRelativeTargets) {
    Add-CleanTarget -Set $targets -Path (Join-Path $workspace $relative)
}

$orderedTargets = $targets | Sort-Object { $_.Length } -Descending
foreach ($target in $orderedTargets) {
    if ($DryRun) {
        Write-Output "[clean] would remove: $target"
        continue
    }
    try {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
        Write-Output "[clean] removed: $target"
    }
    catch {
        Write-Output "[clean] skip: $target ($($_.Exception.Message))"
    }
}
