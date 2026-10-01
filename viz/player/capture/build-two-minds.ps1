# Rebuilds the full "Two Minds" clip set into assets/film/clips.
# Requires: the brain renderer served at 127.0.0.1:8877 (python -m http.server 8877
# --bind 127.0.0.1 from viz/player/brain), Playwright Chromium, ffmpeg on PATH.
# Port 8877 rather than 8766: a parallel session's node server has bound 8766
# alongside the python one before and hijacked every capture request.
#
# Outputs (22 files):
#   two-minds-<sec>.mp4            1920x1080 30fps 12s seamless loop
#   two-minds-<sec>-vertical.mp4   1080x1920 30fps 12s seamless loop
#   two-minds-<sec>-loop.gif       560w 10fps (title and end get no gif)
#   two-minds-tour[-vertical].mp4  68.1s fadeblack stitch: title card (trimmed
#                                  to 7.2s) + the 5 sections + end card
#                                  (trimmed to 4.5s)
#   two-minds-tour-loop.gif        420w 7fps

param([switch]$Textless, [switch]$Resume)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Two runs writing the same clip names interleave inside the same files and
# corrupt every output (valid moov, shredded H.264). Hold an exclusive lock for
# the life of the process; Windows releases the handle when powershell exits.
$lockPath = Join-Path $env:TEMP 'itasorl-two-minds-build.lock'
try {
  $script:BuildLock = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'Write', 'None')
} catch {
  throw "another build-two-minds run is active; wait for it or delete $lockPath"
}

# A clip that decodes with errors poisons the tour stitch silently, so every
# output is decode-checked before anything downstream consumes it.
#
# The stderr redirect runs in cmd, not PowerShell. In 5.1, `2>&1` on a native
# exe wraps every stderr line in an ErrorRecord, so under
# $ErrorActionPreference='Stop' a corrupt file (a killed capture leaves a
# moov-less mp4) blew up inside the check itself with a NativeCommandError,
# instead of reaching the throw below that actually names the bad file.
function Assert-Decodes([string]$path) {
  $log = [System.IO.Path]::GetTempFileName()
  try {
    $null = cmd /c "ffmpeg -nostdin -v error -i `"$path`" -f null - 2>`"$log`""
    $code = $LASTEXITCODE
    $errs = @(Get-Content $log -ErrorAction SilentlyContinue)
  } finally {
    Remove-Item $log -Force -ErrorAction SilentlyContinue
  }
  if ($code -ne 0 -or $errs.Count -gt 0) {
    throw "corrupt output: $path`n$(($errs | Select-Object -First 4) -join "`n")"
  }
}

$env:CHROME_EXE = "$env:LOCALAPPDATA\ms-playwright\chromium-1234\chrome-win64\chrome.exe"
if (-not (Test-Path $env:CHROME_EXE)) {
  $env:CHROME_EXE = "$env:LOCALAPPDATA\ms-playwright\chromium-1223\chrome-win64\chrome.exe"
}
$clips = (Resolve-Path "$PSScriptRoot\..\..\..\assets\film\clips").Path
$secs = 'overview', 'senses', 'process', 'memory', 'actions'
# Title and end chapters are captured like sections but only feed the tour.
$capSecs = @('title') + $secs + @('end')
$renderer = "$PSScriptRoot\..\brain\brain.js"
# Textless (VO) clips get their own names so they never clobber the labeled set.
$mode = if ($Textless) { '-textless' } else { '' }

$env:CAP_MODE = 'full'
$env:CAP_FPS = '30'
foreach ($sec in $capSecs) {
  foreach ($lay in 'wide', 'vertical') {
    $suffix = if ($lay -eq 'vertical') { '-vertical' } else { '' }
    $q = if ($Textless) { '&text=0' } else { '' }
    $env:CAP_URL = "http://127.0.0.1:8877/index.html?sec=$sec&layout=$lay$q"
    $env:CAP_OUT = "$clips\two-minds-$sec$mode$suffix.mp4"
    # -Resume: skip clips already captured from the current renderer.
    if ($Resume -and (Test-Path $env:CAP_OUT) -and
        (Get-Item $env:CAP_OUT).LastWriteTime -gt (Get-Item $renderer).LastWriteTime -and
        (Get-Item $env:CAP_OUT).Length -gt 100KB) {
      Write-Host "=== skip $sec $lay (exists)"
      continue
    }
    Write-Host "=== capture $sec $lay"
    # Headless Chromium dies mid-run now and then; one clean retry absorbs it.
    # A failed attempt leaves a partial mp4 that must not survive (the -Resume
    # freshness check would skip it), so delete before each attempt and on the
    # way out of a failure.
    $ok = $false
    foreach ($attempt in 1, 2) {
      if (Test-Path $env:CAP_OUT) { Remove-Item $env:CAP_OUT -Force }
      node capture-brain.js
      if ($LASTEXITCODE -eq 0) { $ok = $true; break }
      Write-Host "=== capture $sec $lay attempt $attempt failed, retrying"
    }
    if (-not $ok) {
      if (Test-Path $env:CAP_OUT) { Remove-Item $env:CAP_OUT -Force }
      throw "capture failed twice: $sec $lay"
    }
    Assert-Decodes $env:CAP_OUT
  }
}

# Tour = fadeblack stitch. Chapter lengths are NOT uniform: six identical 12s
# chapters beat like a metronome, and senses and process are the two thinnest
# (no readout, static content), so they run 9s while the chapters carrying the
# event and the payoff keep the full loop. Their fourth caption starts at
# u 0.57 and holds past the cut, so nothing is orphaned mid-sentence.
# Trims are also set by reading time: the title card holds 9.5s because two
# beats at ~4.3s each is what they need to be read, and the end card holds 6s
# so the repo link is still there after the eye has finished the number.
#   title 9.5 | overview 12 | senses 9 | process 9 | memory 12 | actions 12 | end 6
# offset_n = (running length) - n * 0.6  ->  65.9s total (was 68.1).
$tourPlan = @(
  @{ sec = 'title';    trim = '9.5' },
  @{ sec = 'overview'; trim = $null },
  @{ sec = 'senses';   trim = '9'   },
  @{ sec = 'process';  trim = '9'   },
  @{ sec = 'memory';   trim = $null },
  @{ sec = 'actions';  trim = $null },
  @{ sec = 'end';      trim = '6'   }
)
$offsets = '8.9', '20.3', '28.7', '37.1', '48.5', '59.9'
$fc = ($offsets | ForEach-Object -Begin { $i = 0 } -Process {
  $i++
  $src = if ($i -eq 1) { '[0:v][1:v]' } else { "[v$($i-1)][${i}:v]" }
  "$src" + "xfade=transition=fadeblack:duration=0.6:offset=$_" + "[v$i];"
}) -join ''
$fc = $fc.TrimEnd(';')
$last = "[v$($offsets.Count)]"

foreach ($suffix in @('', '-vertical')) {
  $in = @()
  foreach ($c in $tourPlan) {
    if ($c.trim) { $in += @('-t', $c.trim) }
    $in += @('-i', "$clips\two-minds-$($c.sec)$mode$suffix.mp4")
  }
  Write-Host "=== tour$suffix"
  $tour = "$clips\two-minds-tour$mode$suffix.mp4"
  ffmpeg -nostdin -y -hide_banner -loglevel error @in -filter_complex $fc -map $last `
    -c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p -movflags +faststart $tour
  if ($LASTEXITCODE -ne 0) { throw "tour failed$suffix" }
  Assert-Decodes $tour
}

# The two gif steps pass -hide_banner -loglevel error, matching the tour stitch
# above. ffmpeg writes its version banner to stderr, and under a caller that
# redirects (2>&1) PowerShell 5.1 wraps every stderr line in an ErrorRecord, so
# $ErrorActionPreference='Stop' killed the build here on a run whose captures and
# tour stitch had all succeeded. Silencing the banner makes the step robust
# however it is invoked; $LASTEXITCODE still gates it and real errors still print.
foreach ($s in $secs) {
  Write-Host "=== gif $s"
  ffmpeg -nostdin -y -hide_banner -loglevel error -i "$clips\two-minds-$s$mode.mp4" `
    -vf "fps=10,scale=560:-2:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" `
    "$clips\two-minds-$s$mode-loop.gif"
  if ($LASTEXITCODE -ne 0) { throw "gif failed $s" }
}
Write-Host "=== gif tour"
ffmpeg -nostdin -y -hide_banner -loglevel error -i "$clips\two-minds-tour$mode.mp4" `
  -vf "fps=7,scale=420:-2:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" `
  "$clips\two-minds-tour$mode-loop.gif"
if ($LASTEXITCODE -ne 0) { throw "gif failed tour" }

Write-Host "BUILD_DONE"
