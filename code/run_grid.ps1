# Self-restarting launcher for the FogField v2 grid (scheduled task `fogfield_v2`, every 10 min,
# MultipleInstances = IgnoreNew, action `conhost.exe --headless powershell.exe ... -File run_grid.ps1`).
#
# Phase 1: four workers share main, milp, equilibrium and lll_schedule (disjoint slices of the
#          remaining ledger units, see run_study in fogfield_src_fullrun.py).
# Phase 2: two workers share scalability and viewbound, whose wall-clock and per-decision
#          timings are reported in the paper, so they run with little CPU contention.
# Each worker has its own log. After phase 2, grid_status.py writes logs_v2\GRID_DONE once every
# study is complete, and later triggers exit at once.
$ErrorActionPreference = "Continue"
$here = "E:\network research\paper FogField\build"
Set-Location $here
if (Test-Path "$here\logs_v2\GRID_DONE") { exit 0 }
New-Item -ItemType Directory -Force "$here\logs_v2" | Out-Null

$py = "C:\Users\amr\AppData\Local\Programs\Python\Python311\python.exe"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONWARNINGS = "ignore"
$env:FOGFIELD_BUDGET_MIN = "720"
$env:FOGFIELD_STOP_AFTER_EXPERIMENTS = "1"

function Log($msg) {
    Add-Content "$here\logs_v2\launcher.log" ((Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $msg)
}

function Run-Phase($name, $studies, $nw) {
    Log "phase $name start: $nw worker(s), studies $studies"
    $env:FOGFIELD_STUDIES = $studies
    $env:FOGFIELD_NWORKERS = "$nw"
    $procs = @()
    for ($k = 0; $k -lt $nw; $k++) {
        $env:FOGFIELD_WORKER = "$k"
        $log = "$here\logs_v2\${name}_worker_$k.log"
        $p = Start-Process -FilePath $py -ArgumentList @("-u", "driver.py", "--mode", "grind") `
            -WorkingDirectory $here -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
            -NoNewWindow -PassThru
        $procs += $p
        Start-Sleep -Seconds 2
    }
    $procs | Wait-Process
    Log "phase $name end"
}

Log "launcher start (pid $PID)"
Run-Phase "p1" "main,milp,equilibrium,lll_schedule" 4
Run-Phase "p2" "scalability,viewbound" 2
& $py grid_status.py *>> "$here\logs_v2\launcher.log"
Log "launcher end"
