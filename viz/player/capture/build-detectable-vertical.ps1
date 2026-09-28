# Encodes the silent 9:16 "Detectable All Along" cut into assets/film.
# Requires: viz/ served at 127.0.0.1:8767 (python -m http.server 8767
# from viz/), Playwright Chromium, ffmpeg on PATH, viz/data/scene.json
# from collect.py (capture.js refuses the placeholder).
#
# Outputs:
#   detectable-all-along-vertical.mp4   1080x1920 30fps 90s
#   detectable-all-along-poster-vertical.jpg  still from the creature beat

param()

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$lockPath = Join-Path $env:TEMP 'itasorl-detectable-vertical-build.lock'
try {
  $script:BuildLock = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'Write', 'None')
} catch {
  throw "another build-detectable-vertical run is active; wait for it or delete $lockPath"
}

function Assert-Decodes([string]$path) {
  $errs = & ffmpeg -nostdin -v error -i $path -f null - 2>&1
  if ($LASTEXITCODE -ne 0 -or $errs) {
    throw "corrupt output: $path`n$(($errs | Select-Object -First 4) -join "`n")"
  }
}

$env:CHROME_EXE = "$env:LOCALAPPDATA\ms-playwright\chromium-1234\chrome-win64\chrome.exe"
if (-not (Test-Path $env:CHROME_EXE)) {
  $env:CHROME_EXE = "$env:LOCALAPPDATA\ms-playwright\chromium-1223\chrome-win64\chrome.exe"
}
$film = (Resolve-Path "$PSScriptRoot\..\..\..\assets\film").Path
$out = Join-Path $film 'detectable-all-along-vertical.mp4'
$poster = Join-Path $film 'detectable-all-along-poster-vertical.jpg'

$env:CAP_MODE = 'full'
$env:CAP_FPS = '30'
$env:CAP_URL = 'http://127.0.0.1:8767/player/index.html?layout=vertical&play=0'
$env:CAP_OUT = $out

if (Test-Path $out) { Remove-Item $out -Force }
Write-Host "=== capture vertical 1080x1920"
node capture.js
if ($LASTEXITCODE -ne 0) {
  if (Test-Path $out) { Remove-Item $out -Force }
  throw "capture failed"
}
Assert-Decodes $out

$probe = (& ffprobe -v error -select_streams v:0 `
  -show_entries stream=width,height -show_entries format=duration `
  -of default=noprint_wrappers=1 $out) -join "`n"
Write-Host $probe
if ($probe -notmatch 'width=1080' -or $probe -notmatch 'height=1920') {
  throw "wrong frame size:`n$probe"
}
$durLine = ($probe -split "`n" | Where-Object { $_ -like 'duration=*' } | Select-Object -First 1)
$dur = [double](($durLine -split '=', 2)[1])
if ([math]::Abs($dur - 90) -gt 0.15) {
  throw "wrong duration:`n$probe"
}

$env:SHOT_URL = $env:CAP_URL
$env:SHOT_T = '6000'
$env:SHOT_OUT = $poster
Write-Host "=== poster"
node shot.js
if ($LASTEXITCODE -ne 0) { throw "poster failed" }

Write-Host "BUILD_DONE -> $out"
