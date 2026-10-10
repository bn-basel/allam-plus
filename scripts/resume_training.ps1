# Continue a paused or stopped Phase 2 run from its latest checkpoint, with the settings it was started with
# (read from runs/<run>/data.json).
# Usage:  .\scripts\resume_training.ps1 long_v1
# Close games and other GPU-heavy apps first.
param([string]$Run = "long_v1")
$root = Join-Path $PSScriptRoot ".."
$data = Join-Path $root "runs\$Run\data.json"
if (-not (Test-Path $data)) { Write-Host "No runs\$Run\data.json: was this run ever started?"; exit 1 }
$a = (Get-Content $data -Raw | ConvertFrom-Json).args
Remove-Item (Join-Path $root "runs\$Run\PAUSE") -ErrorAction SilentlyContinue
Write-Host "Resuming $Run. Progress is in runs\$Run\train.log. To pause again: .\scripts\pause_training.ps1 $Run"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $root
uv run python scripts/train_long.py --run $Run --data $a.data --epochs $a.epochs --lr $a.lr --rank $a.rank `
    --accum $a.accum --warmup $a.warmup --save-every $a.save_every --seed $a.seed --resume
