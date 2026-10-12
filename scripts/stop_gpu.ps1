# Stop this project's GPU work now: no new step starts, and the running step is ended (every step resumes later).
# Usage (PowerShell, in C:\WIP\projects\allam-plus):  .\scripts\stop_gpu.ps1
$root = Join-Path $PSScriptRoot ".."
New-Item -ItemType File -Path (Join-Path $root "notes\STOP") -Force | Out-Null
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'allam-plus|gpu_block|phase[34]_|pick_and_test|agent_test_served|demo/agent\.py|scripts/(agent_bench|tools_bench|tool_dev|train_long|receptionist_eval|needle|lceval|arabicmmlu|ifeval|export_gguf|finalize_gguf|speed_memory)\.py|ollama' -and $_.Name -match 'python|uv|sh|bash|llama-server|llama-bench|ollama' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host "Stopped. Training (if it was running) continues from its last checkpoint next time."
