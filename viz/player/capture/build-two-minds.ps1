# Rebuilds the full "Two Minds" clip set into assets/film/clips.
# Requires: the brain renderer served at 127.0.0.1:8766 (python -m http.server 8766
# from viz/player/brain), Playwright Chromium, ffmpeg on PATH.
#
# Outputs (18 files):
#   two-minds-<sec>.mp4            1920x1080 30fps 12s seamless loop
#   two-minds-<sec>-vertical.mp4   1080x1920 30fps 12s seamless loop
#   two-minds-<sec>-loop.gif       560w 10fps
#   two-minds-tour[-vertical].mp4  57.6s fadeblack stitch of the 5 sections
#   two-minds-tour-loop.gif        420w 7fps

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$env:CHROME_EXE = "$env:LOCALAPPDATA\ms-playwright\chromium-1234\chrome-win64\chrome.exe"
if (-not (Test-Path $env:CHROME_EXE)) {
  $env:CHROME_EXE = "$env:LOCALAPPDATA\ms-playwright\chromium-1223\chrome-win64\chrome.exe"
}
$clips = (Resolve-Path "$PSScriptRoot\..\..\..\assets\film\clips").Path
$secs = 'overview', 'senses', 'process', 'memory', 'actions'

$env:CAP_MODE = 'full'
$env:CAP_FPS = '30'
foreach ($sec in $secs) {
  foreach ($lay in 'wide', 'vertical') {
    $suffix = if ($lay -eq 'vertical') { '-vertical' } else { '' }
    $env:CAP_URL = "http://127.0.0.1:8766/index.html?sec=$sec&layout=$lay"
    $env:CAP_OUT = "$clips\two-minds-$sec$suffix.mp4"
    Write-Host "=== capture $sec $lay"
    node capture-brain.js
    if ($LASTEXITCODE -ne 0) { throw "capture failed: $sec $lay" }
  }
}

# Tour = fadeblack stitch; offset_n = running length - 0.6s per join -> 57.6s total.
$fc = "[0:v][1:v]xfade=transition=fadeblack:duration=0.6:offset=11.4[v1];" +
      "[v1][2:v]xfade=transition=fadeblack:duration=0.6:offset=22.8[v2];" +
      "[v2][3:v]xfade=transition=fadeblack:duration=0.6:offset=34.2[v3];" +
      "[v3][4:v]xfade=transition=fadeblack:duration=0.6:offset=45.6[v4]"
foreach ($suffix in @('', '-vertical')) {
  $in = @()
  foreach ($s in $secs) { $in += @('-i', "$clips\two-minds-$s$suffix.mp4") }
  Write-Host "=== tour$suffix"
  ffmpeg -y @in -filter_complex $fc -map '[v4]' -c:v libx264 -preset medium -crf 17 `
    -pix_fmt yuv420p -movflags +faststart "$clips\two-minds-tour$suffix.mp4"
  if ($LASTEXITCODE -ne 0) { throw "tour failed$suffix" }
}

foreach ($s in $secs) {
  Write-Host "=== gif $s"
  ffmpeg -y -i "$clips\two-minds-$s.mp4" `
    -vf "fps=10,scale=560:-2:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" `
    "$clips\two-minds-$s-loop.gif"
  if ($LASTEXITCODE -ne 0) { throw "gif failed $s" }
}
Write-Host "=== gif tour"
ffmpeg -y -i "$clips\two-minds-tour.mp4" `
  -vf "fps=7,scale=420:-2:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" `
  "$clips\two-minds-tour-loop.gif"
if ($LASTEXITCODE -ne 0) { throw "gif failed tour" }

Write-Host "BUILD_DONE"
