param(
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing virtual environment at .venv. Create it and install requirements-recommender.txt first."
}

$versionDir = Join-Path $PSScriptRoot "build"
New-Item -ItemType Directory -Force -Path $versionDir | Out-Null
$versionPath = Join-Path $versionDir "windows_version.json"
$versionJson = "{`n  `"version`": `"$Version`",`n  `"patch_protocol`": 2`n}`n"
$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($versionPath, $versionJson, $utf8)

Write-Host "Building frontend..."
Push-Location (Join-Path $PSScriptRoot "frontend")
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& npm.cmd run build
$frontendExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorAction
Pop-Location
if ($frontendExitCode -ne 0) {
    throw "Frontend build failed with exit code $frontendExitCode"
}

Write-Host "Building runtime artifacts..."
& $python (Join-Path $PSScriptRoot "workflow_scripts\build_runtime_artifacts.py")
if ($LASTEXITCODE -ne 0) {
    throw "Runtime artifact build failed with exit code $LASTEXITCODE"
}

Write-Host "Ensuring PyInstaller is installed..."
& $python -m pip install --disable-pip-version-check pyinstaller

Write-Host "Building onedir executable..."
& $python -m PyInstaller (Join-Path $PSScriptRoot "E7_BP_Helper.spec") --noconfirm
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$packagedVersion = Join-Path $PSScriptRoot "dist\E7_BP_Helper\windows_version.json"
Copy-Item -LiteralPath $versionPath -Destination $packagedVersion -Force
& $python (Join-Path $PSScriptRoot "workflow_scripts\build_windows_patch.py") --write-app-manifest (Join-Path $PSScriptRoot "dist\E7_BP_Helper")
if ($LASTEXITCODE -ne 0) {
    throw "Could not write the Windows file manifest"
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Version: $Version"
Write-Host "Output folder: dist\E7_BP_Helper\"
Write-Host "Run: dist\E7_BP_Helper\E7_BP_Helper.exe"
