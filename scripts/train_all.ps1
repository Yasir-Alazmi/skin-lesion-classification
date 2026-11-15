<#
Detached training launcher: EfficientNet-B3 then ViT-B/16 on the lesion-grouped split,
each best-by-validation checkpoint evaluated once on the test set.

Starts ONE hidden python process (no PowerShell pipe, so nothing can block on output),
redirects stdout / stderr to log files, and returns immediately.  Progress is written
by python to <DataRoot>\logs\status.json after every epoch; use scripts\train_status.ps1.
An interrupted run continues with:  scripts\train_all.ps1 -Resume

Usage (repo root):  powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1 [-Epochs 40] [-Patience 7] [-Resume]
Everything large goes to $DataRoot (outside git): cache, checkpoints, logs, runs.
#>
param(
    [string]$DataRoot = "D:\claude project\1-core\skin-lesion-data",
    [int]$Epochs = 40,
    [int]$Patience = 7,
    [int]$Workers = 6,
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
$logs = Join-Path $DataRoot "logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$pidFile = Join-Path $logs "train.pid"
$status = Join-Path $logs "status.json"

if (Test-Path $pidFile) {
    $old = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue
    if ($old) { throw "training already running (PID $($old.Id)); see scripts\train_status.ps1" }
}

$env:SKIN_DATA_DIR = $DataRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out = Join-Path $logs "train_$stamp.out.log"
$err = Join-Path $logs "train_$stamp.err.log"

$argList = "-m src.train --model both --epochs $Epochs --patience $Patience --workers $Workers " +
    "--status-file `"$status`""
if ($Resume) { $argList += " --resume" }

$p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $repo `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
$p.Id | Set-Content $pidFile
Set-Content -Path (Join-Path $logs "latest.txt") -Value "$out`n$err"
Write-Host "Started training, PID $($p.Id)"
Write-Host "  stdout: $out"
Write-Host "  stderr: $err"
Write-Host "  status: $status   (powershell -File scripts\train_status.ps1)"
