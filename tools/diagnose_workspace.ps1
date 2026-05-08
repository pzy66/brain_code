param(
    [int]$LargestCount = 20,
    [switch]$IncludeIgnored
)

$ErrorActionPreference = "Stop"

$brainCodeRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$workspace = (Split-Path -Parent $brainCodeRoot)

Write-Output "[diagnose] workspace=$workspace"
Write-Output "[diagnose] brain_code=$brainCodeRoot"

$trackedRaw = git -C $brainCodeRoot ls-files -z
$tracked = @($trackedRaw -join "" -split "`0" | Where-Object { $_ })
Write-Output "[diagnose] tracked_files=$($tracked.Count)"

$trackedFileRows = @(
    $tracked | ForEach-Object {
        $relative = $_
        $local = $relative -replace "/", [System.IO.Path]::DirectorySeparatorChar
        $path = Join-Path $brainCodeRoot $local
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $item = Get-Item -LiteralPath $path
            [PSCustomObject]@{
                Bytes = $item.Length
                MiB = [Math]::Round($item.Length / 1MB, 2)
                Path = $relative
            }
        }
    }
)
$totalBytes = ($trackedFileRows | Measure-Object -Property Bytes -Sum).Sum
Write-Output "[diagnose] tracked_size_mib=$([Math]::Round($totalBytes / 1MB, 2))"

Write-Output ""
Write-Output "[diagnose] top-level tracked counts"
$tracked |
    ForEach-Object { ($_ -split "/")[0] } |
    Group-Object |
    Sort-Object Count -Descending |
    Select-Object -First 20 Count, Name |
    Format-Table -AutoSize |
    Out-String |
    Write-Output

Write-Output "[diagnose] extension counts"
$tracked |
    ForEach-Object {
        $ext = [System.IO.Path]::GetExtension($_).ToLowerInvariant()
        if ($ext) { $ext } else { "<none>" }
    } |
    Group-Object |
    Sort-Object Count -Descending |
    Select-Object -First 20 Count, Name |
    Format-Table -AutoSize |
    Out-String |
    Write-Output

Write-Output "[diagnose] largest tracked files"
$trackedFileRows |
    Sort-Object MiB -Descending |
    Select-Object -First $LargestCount MiB, Path |
    Format-Table -AutoSize |
    Out-String |
    Write-Output

$artifactPatterns = [ordered]@{
    "root_datasets" = "^datasets/"
    "ssvep_artifacts" = "^02_SSVEP/artifacts/"
    "ssvep_runs" = "^02_SSVEP/artifacts/runs/"
    "ssvep_datasets" = "^02_SSVEP/artifacts/datasets/"
    "ssvep_deployed_profiles" = "^02_SSVEP/artifacts/deployed_profiles/"
    "root_artifacts" = "^artifacts/"
    "hybrid_models" = "^hybrid_controller/models/"
    "logs" = "^logs/"
}

Write-Output "[diagnose] tracked artifact categories"
foreach ($entry in $artifactPatterns.GetEnumerator()) {
    $matches = @($trackedFileRows | Where-Object { $_.Path -match $entry.Value })
    $bytes = ($matches | Measure-Object -Property Bytes -Sum).Sum
    Write-Output "[diagnose] $($entry.Key)_files=$($matches.Count) size_mib=$([Math]::Round($bytes / 1MB, 2))"
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$statusOutput = git -C $brainCodeRoot status --short --ignored 2>&1
$ErrorActionPreference = $previousErrorActionPreference
$permissionWarnings = @($statusOutput | Where-Object { "$_" -like "*Permission denied*" })
$ignoredEntries = @($statusOutput | Where-Object { "$_" -like "!! *" })

Write-Output "[diagnose] ignored_entries=$($ignoredEntries.Count)"
Write-Output "[diagnose] permission_warnings=$($permissionWarnings.Count)"
foreach ($warning in $permissionWarnings) {
    Write-Output "[diagnose] $warning"
}

if ($IncludeIgnored) {
    Write-Output ""
    Write-Output "[diagnose] ignored entries"
    $ignoredEntries | Write-Output
}
