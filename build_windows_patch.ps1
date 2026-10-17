param(
    [Parameter(Mandatory = $true)]
    [string]$Baseline,
    [string]$Target = "dist\E7_BP_Helper",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing virtual environment at .venv. Create it and install requirements-recommender.txt first."
}

$baselinePath = $Baseline
if (-not [System.IO.Path]::IsPathRooted($baselinePath)) {
    $baselinePath = Join-Path $PSScriptRoot $baselinePath
}
$targetPath = $Target
if (-not [System.IO.Path]::IsPathRooted($targetPath)) {
    $targetPath = Join-Path $PSScriptRoot $targetPath
}

$arguments = @(
    (Join-Path $PSScriptRoot "workflow_scripts\build_windows_patch.py"),
    "--baseline", $baselinePath,
    "--target", $targetPath
)
if ($Output) {
    $outputPath = $Output
    if (-not [System.IO.Path]::IsPathRooted($outputPath)) {
        $outputPath = Join-Path $PSScriptRoot $outputPath
    }
    $arguments += @("--output", $outputPath)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Windows patch build failed with exit code $LASTEXITCODE"
}
