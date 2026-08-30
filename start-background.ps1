# 后台启动本地视觉模型服务（无窗口，日志写入 logs\）
$ErrorActionPreference = "Stop"
$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = Join-Path $base "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$py = Join-Path $base ".venv\Scripts\python.exe"
$server = Join-Path $base "server.py"
$p = Start-Process -FilePath $py -ArgumentList @($server, "--warmup") `
    -WorkingDirectory $base -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs "server.out.log") `
    -RedirectStandardError (Join-Path $logs "server.err.log") -PassThru
Set-Content -Path (Join-Path $logs "server.pid") -Value $p.Id
Write-Host "本地视觉模型服务已后台启动 (PID $($p.Id))，端口 8321"
