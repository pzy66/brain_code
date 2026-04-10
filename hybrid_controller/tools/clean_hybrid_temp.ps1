param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$targets = @(
    ".tmp_pytest",
    "pytest-cache-files-*",
    "pytest_tmp_py",
    "tmp_test_dir",
    ".pytest_tmp",
    ".pytest_cache",
    "__pycache__"
)

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
Write-Output "[clean] root=$resolvedRoot"

foreach ($pattern in $targets) {
    $matches = Get-ChildItem -LiteralPath $resolvedRoot -Force -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -like $pattern
    }
    foreach ($entry in $matches) {
        $path = $entry.FullName
        try {
            Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
            Write-Output "[clean] removed: $path"
        }
        catch {
            Write-Output "[clean] skip: $path ($($_.Exception.Message))"
        }
    }
}
