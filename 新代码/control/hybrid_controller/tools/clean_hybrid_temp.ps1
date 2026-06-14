param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$targets = @(
    ".tmp_pytest",
    ".tmp_pytest*",
    "pytest-cache-files-*",
    "pytest_tmp_py",
    "tmp_test_dir",
    "tmp_pytest*",
    ".pytest_tmp",
    ".pytest_tmp_run",
    ".pytest_cache",
    "__pycache__"
)

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
Write-Output "[clean] root=$resolvedRoot"

$rootEntries = Get-ChildItem -LiteralPath $resolvedRoot -Force -ErrorAction SilentlyContinue
$pathsToClean = New-Object System.Collections.Generic.HashSet[string]
foreach ($pattern in $targets) {
    foreach ($entry in $rootEntries) {
        if ($entry.Name -like $pattern) {
            [void]$pathsToClean.Add($entry.FullName)
        }
    }
}

foreach ($path in $pathsToClean) {
    try {
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
        Write-Output "[clean] removed: $path"
    }
    catch {
        Write-Output "[clean] skip: $path ($($_.Exception.Message))"
    }
}

# Clean nested __pycache__ folders recursively as well.
$nestedCaches = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -Directory -Filter "__pycache__" -Force -ErrorAction SilentlyContinue
foreach ($entry in $nestedCaches) {
    $path = $entry.FullName
    try {
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
        Write-Output "[clean] removed: $path"
    }
    catch {
        Write-Output "[clean] skip: $path ($($_.Exception.Message))"
    }
}
