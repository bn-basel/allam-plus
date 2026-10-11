# Stop this project's GPU work now: no new step starts, and the running step is ended (every step resumes later).
# Usage (PowerShell, in C:\WIP\projects\allam-plus):  .\scripts\stop_gpu.ps1
$root = Join-Path $PSScriptRoot ".."
New-Item -ItemType File -Path (Join-Path $root "notes\STOP") -Force | Out-Null
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'allam-plus|gpu_block|phase3_|scripts/(tools_bench|tool_dev|train_long|receptionist_eval|needle|lceval|arabicmmlu|ifeval|export_gguf)\.py' -and $_.Name -match 'python|uv|sh|bash|llama-server' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host "Stopped. Training (if it was running) continues from its last checkpoint next time."
