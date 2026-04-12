param(
    [string]$PythonExe = "C:\Users\P1233\miniconda3\envs\brain-vision\python.exe",
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
Write-Output "[check] root=$resolvedRoot"
Write-Output "[check] python=$PythonExe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python not found: $PythonExe"
}

$skipDirs = @(
    ".tmp_pytest",
    "pytest-cache-files-",
    "pytest_tmp_py",
    "tmp_test_dir",
    "tmp_pytest",
    ".pytest_tmp",
    ".pytest_tmp_run",
    ".pytest_cache",
    "__pycache__"
)

$pyFiles = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Filter *.py -ErrorAction SilentlyContinue | Where-Object {
    $full = $_.FullName
    foreach ($token in $skipDirs) {
        if ($full -like "*\\$token*" -or $full -like "*\\$token\\*") {
            return $false
        }
    }
    return $true
}

Write-Output "[check] py_compile count=$($pyFiles.Count)"
foreach ($file in $pyFiles) {
    & $PythonExe -m py_compile $file.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "py_compile failed: $($file.FullName)"
    }
}

& $PythonExe -m pytest "$resolvedRoot\\tests" -q -o addopts= --basetemp "$resolvedRoot\\tmp_pytest_checks"
if ($LASTEXITCODE -ne 0) {
    throw "pytest failed"
}

Write-Output "[check] done"
