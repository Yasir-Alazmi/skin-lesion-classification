<#
Print the training status (status.json), whether the process is alive, and the last
log lines.  Usage: powershell -ExecutionPolicy Bypass -File scripts\train_status.ps1 [-Lines 15]
#>
param(
    [string]$DataRoot = "D:\claude project\1-core\skin-lesion-data",
    [int]$Lines = 15
)
$logs = Join-Path $DataRoot "logs"
$status = Join-Path $logs "status.json"
$pidFile = Join-Path $logs "train.pid"

if (Test-Path $pidFile) {
    $procId = [int](Get-Content $pidFile)
    $alive = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($alive) { Write-Host "process: RUNNING (PID $procId)" } else { Write-Host "process: not running (PID $procId)" }
} else {
    Write-Host "process: no train.pid (never launched)"
}
if (Test-Path $status) {
    Write-Host "--- status.json"
    Get-Content $status
} else {
    Write-Host "no status.json yet"
}
$latest = Join-Path $logs "latest.txt"
if (Test-Path $latest) {
    foreach ($f in (Get-Content $latest)) {
        Write-Host "--- tail $f"
        if (Test-Path $f) { Get-Content $f -Tail $Lines }
    }
}
