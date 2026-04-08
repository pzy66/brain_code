$ErrorActionPreference = "Stop"

function Resolve-HybridRunPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $candidates = @()

    if (-not [string]::IsNullOrWhiteSpace($env:HYBRID_PYTHON_EXE)) {
        $candidates += $env:HYBRID_PYTHON_EXE
    }

    $candidates += @(
        "C:\Users\P1233\miniconda3\envs\brain-vision\python.exe"
    )

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if ($candidate -like "*.exe" -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw "brain-vision interpreter not found. Expected: C:\Users\P1233\miniconda3\envs\brain-vision\python.exe"
}

function Resolve-HybridBootstrapPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $candidates = @()

    if (-not [string]::IsNullOrWhiteSpace($env:HYBRID_BASE_PYTHON)) {
        $candidates += $env:HYBRID_BASE_PYTHON
    }

    $candidates += @(
        "C:\Users\P1233\miniconda3\envs\brain-vision\python.exe"
    )

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "brain-vision interpreter not found. Expected: C:\Users\P1233\miniconda3\envs\brain-vision\python.exe"
}
