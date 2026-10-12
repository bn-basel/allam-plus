# Pause this project's GPU work. If a training run is going, it saves a checkpoint at the end of its current step and
# stops; the block runner then starts no further step (notes\STOP). If no training is going (a test is running), this
# stops the running step at once, like stop_gpu.ps1. Either way, starting the same block command again continues.
# Usage:  .\scripts\pause_training.ps1             (the run that is training now)
#         .\scripts\pause_training.ps1 tools_v2
param([string]$Run = "")
$root = Join-Path $PSScriptRoot ".."
New-Item -ItemType File -Path (Join-Path $root "notes\STOP") -Force | Out-Null
$training = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'scripts/train_long\.py --run (\S+)' })
if ($training.Count -eq 0) {
    Write-Host "No training is running; stopping the current step instead."
    & (Join-Path $PSScriptRoot "stop_gpu.ps1")
    exit 0
}
if (-not $Run) { $null = $training[0].CommandLine -match 'scripts/train_long\.py --run (\S+)'; $Run = $Matches[1] }
$dir = Join-Path $root "runs\$Run"
if (-not (Test-Path $dir)) { Write-Host "No run folder runs\$Run"; exit 1 }
New-Item -ItemType File -Path (Join-Path $dir "PAUSE") -Force | Out-Null
Write-Host "Pause requested for $Run. It saves a checkpoint after the current step (up to a few minutes) and stops;"
Write-Host "no further step of the block starts. To continue, start the same block command again."
