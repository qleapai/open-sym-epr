<#
  build_standalone.ps1  -  Assemble a fully self-contained Windows package for
  Open-Sym-EPR that embeds its own Python runtime, so it installs and runs on any
  Windows desktop with NO system Python and NO internet.

  Output:  dist\OpenSymEPR\                (the standalone folder)
           dist\OpenSymEPR_Windows_x64.zip (distributable archive)

  Run once on a build machine (needs internet to fetch the embeddable runtime):
      powershell -ExecutionPolicy Bypass -File build_standalone.ps1
#>
param(
  [string]$Root   = (Split-Path -Parent $MyInvocation.MyCommand.Path),
  [string]$PyVer  = "3.11.9",
  [string]$Cache  = "e:\pip_cache"
)
$ErrorActionPreference = "Stop"
$dist  = Join-Path $Root "dist"
$stage = Join-Path $dist "OpenSymEPR"
$pydir = Join-Path $stage "python"
$app   = Join-Path $stage "app"

Write-Host "== Clean staging ==" -ForegroundColor Cyan
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $pydir, $app | Out-Null

Write-Host "== 1/6  Fetch embeddable Python $PyVer ==" -ForegroundColor Cyan
$zip = Join-Path $dist "python-embed.zip"
if (-not (Test-Path $zip)) {
  Invoke-WebRequest "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip" -OutFile $zip
}
Expand-Archive $zip -DestinationPath $pydir -Force

Write-Host "== 2/6  Enable site-packages in the embeddable runtime ==" -ForegroundColor Cyan
$pth = Get-ChildItem (Join-Path $pydir "python*._pth") | Select-Object -First 1
$lines = Get-Content $pth.FullName | ForEach-Object { $_ -replace '^\s*#\s*import site', 'import site' }
if ($lines -notcontains 'Lib\site-packages') { $lines += 'Lib\site-packages' }
Set-Content $pth.FullName $lines -Encoding ascii

Write-Host "== 3/6  Bootstrap pip ==" -ForegroundColor Cyan
$getpip = Join-Path $pydir "get-pip.py"
if (-not (Test-Path $getpip)) { Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile $getpip }
& (Join-Path $pydir "python.exe") $getpip --no-warn-script-location

Write-Host "== 4/6  Install Open-Sym-EPR + all dependencies into the runtime ==" -ForegroundColor Cyan
$cacheArg = @(); if (Test-Path $Cache) { $cacheArg = @("--cache-dir", $Cache) }
& (Join-Path $pydir "python.exe") -m pip install --no-warn-script-location @cacheArg "$Root"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "== 5/6  Copy application files ==" -ForegroundColor Cyan
foreach ($f in "streamlit_app.py","app_simepr.py","app_openspin.py","README.md","LICENSE","CITATION.cff") {
  if (Test-Path (Join-Path $Root $f)) { Copy-Item (Join-Path $Root $f) $app }
}
if (Test-Path (Join-Path $Root ".streamlit")) { Copy-Item (Join-Path $Root ".streamlit") $app -Recurse }
if (Test-Path (Join-Path $Root "examples"))   { Copy-Item (Join-Path $Root "examples")   $app -Recurse }

Write-Host "== 6/6  Write launcher + zip ==" -ForegroundColor Cyan
$launcher = @'
@echo off
REM Open-Sym-EPR - standalone launcher (bundled Python, no install needed)
cd /d "%~dp0"
echo Starting Open-Sym-EPR at http://localhost:8501  (close this window to stop)
start "" http://localhost:8501
"%~dp0python\python.exe" -m streamlit run "%~dp0app\streamlit_app.py" --server.headless true --browser.gatherUsageStats false
pause
'@
Set-Content (Join-Path $stage "Open-Sym-EPR.bat") $launcher -Encoding ascii

$readme = @'
Open-Sym-EPR - Standalone Windows package
=========================================
This folder contains everything needed to run Open-Sym-EPR, including its own
Python runtime. No installation and no internet connection are required.

TO RUN:  double-click  Open-Sym-EPR.bat
         Your browser opens at http://localhost:8501
         Close the black terminal window to stop the app.

You may copy this whole folder to any Windows 10/11 (64-bit) computer and run it
from anywhere (Desktop, USB drive, network share).
'@
Set-Content (Join-Path $stage "READ ME FIRST.txt") $readme -Encoding ascii

$archive = Join-Path $dist "OpenSymEPR_Windows_x64.zip"
if (Test-Path $archive) { Remove-Item $archive -Force }
Compress-Archive -Path $stage -DestinationPath $archive
$sizeMB = [math]::Round((Get-Item $archive).Length / 1MB, 1)
Write-Host "DONE. Package: $stage" -ForegroundColor Green
Write-Host "DONE. Archive: $archive ($sizeMB MB)" -ForegroundColor Green
