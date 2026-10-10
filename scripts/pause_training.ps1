# Pause a training run cleanly: it saves a checkpoint at the end of the current step and stops.
# Usage:  .\scripts\pause_training.ps1            (pauses long_v1)
#         .\scripts\pause_training.ps1 long_v2
param([string]$Run = "long_v1")
$dir = Join-Path $PSScriptRoot "..\runs\$Run"
if (-not (Test-Path $dir)) { Write-Host "No run folder runs\$Run"; exit 1 }
New-Item -ItemType File -Path (Join-Path $dir "PAUSE") -Force | Out-Null
Write-Host "Pause requested for $Run. Training saves a checkpoint after the current step (up to a few minutes: a step is 8 long examples) and stops."
Write-Host "Continue later with:  .\scripts\resume_training.ps1 $Run"
