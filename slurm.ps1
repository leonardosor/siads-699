# SLURM Helper Script for Windows PowerShell
# Provides easy commands to interact with local SLURM cluster

param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet('start', 'stop', 'status', 'submit', 'queue', 'logs', 'cancel', 'help')]
    [string]$Command,
    
    [Parameter(Position=1)]
    [string]$JobId,
    
    [int]$Epochs = 250,
    [int]$Batch = 4,
    [switch]$Optuna,
    [int]$Trials = 20
)

$ConfigDir = "d:\docs\MADS\699\config"
$WorkspaceDir = "d:\docs\MADS\699"

function Show-Help {
    Write-Host @"
SLURM Helper Script for Local Development

Usage: .\slurm.ps1 <command> [options]

Commands:
    start           Start SLURM containers
    stop            Stop SLURM containers
    status          Show cluster status
    submit          Submit a training job
    queue           Show job queue
    logs [job_id]   View job logs
    cancel [job_id] Cancel a job
    help            Show this help message

Submit Options:
    -Epochs <n>     Number of epochs (default: 250)
    -Batch <n>      Batch size (default: 4)
    -Optuna         Enable Optuna optimization
    -Trials <n>     Optuna trials (default: 20)

Examples:
    .\slurm.ps1 start
    .\slurm.ps1 submit -Epochs 100 -Batch 8
    .\slurm.ps1 submit -Optuna -Trials 20 -Epochs 50
    .\slurm.ps1 queue
    .\slurm.ps1 logs 5
    .\slurm.ps1 cancel 5
    .\slurm.ps1 stop
"@
}

function Start-Slurm {
    Write-Host "Starting SLURM containers..." -ForegroundColor Cyan
    Push-Location $ConfigDir
    docker-compose up -d slurm-controller slurm-node1
    Pop-Location
    Write-Host "SLURM cluster started!" -ForegroundColor Green
}

function Stop-Slurm {
    Write-Host "Stopping SLURM containers..." -ForegroundColor Cyan
    Push-Location $ConfigDir
    docker-compose stop slurm-controller slurm-node1
    Pop-Location
    Write-Host "SLURM cluster stopped!" -ForegroundColor Green
}

function Get-SlurmStatus {
    Write-Host "SLURM Cluster Status:" -ForegroundColor Cyan
    docker exec slurm-controller sinfo
    Write-Host "`nContainer Status:" -ForegroundColor Cyan
    docker ps --filter "name=slurm" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

function Submit-Job {
    Write-Host "Submitting training job..." -ForegroundColor Cyan
    
    $env_vars = "EPOCHS=$Epochs,BATCH=$Batch"
    
    if ($Optuna) {
        $env_vars += ",USE_OPTUNA=1,N_TRIALS=$Trials"
        Write-Host "  Mode: Optuna Optimization" -ForegroundColor Yellow
        Write-Host "  Trials: $Trials" -ForegroundColor Yellow
    } else {
        $env_vars += ",USE_OPTUNA=0"
        Write-Host "  Mode: Standard Training" -ForegroundColor Yellow
    }
    
    Write-Host "  Epochs: $Epochs" -ForegroundColor Yellow
    Write-Host "  Batch: $Batch" -ForegroundColor Yellow
    
    $result = docker exec slurm-controller bash -c "cd /workspace/src/training && sbatch --export=ALL,$env_vars /workspace/src/training/batch_job_local.sh"
    
    Write-Host "`n$result" -ForegroundColor Green
    
    if ($result -match "Submitted batch job (\d+)") {
        $jobId = $Matches[1]
        Write-Host "`nJob submitted successfully! Job ID: $jobId" -ForegroundColor Green
        Write-Host "`nMonitor with:" -ForegroundColor Cyan
        Write-Host "  .\slurm.ps1 queue" -ForegroundColor White
        Write-Host "  .\slurm.ps1 logs $jobId" -ForegroundColor White
    }
}

function Get-JobQueue {
    Write-Host "Job Queue:" -ForegroundColor Cyan
    docker exec slurm-controller squeue
}

function Get-JobLogs {
    if (-not $JobId) {
        Write-Host "Error: Job ID required" -ForegroundColor Red
        Write-Host "Usage: .\slurm.ps1 logs <job_id>" -ForegroundColor Yellow
        return
    }
    
    $logFile = "/workspace/logs/capstone_local-$JobId.log"
    Write-Host "Viewing logs for Job $JobId..." -ForegroundColor Cyan
    Write-Host "Log file: $logFile" -ForegroundColor Gray
    Write-Host "Press Ctrl+C to exit`n" -ForegroundColor Gray
    
    docker exec slurm-controller tail -f $logFile
}

function Stop-Job {
    if (-not $JobId) {
        Write-Host "Error: Job ID required" -ForegroundColor Red
        Write-Host "Usage: .\slurm.ps1 cancel <job_id>" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Cancelling job $JobId..." -ForegroundColor Cyan
    docker exec slurm-controller scancel $JobId
    Write-Host "Job $JobId cancelled!" -ForegroundColor Green
}

# Main command dispatcher
switch ($Command) {
    'start'  { Start-Slurm }
    'stop'   { Stop-Slurm }
    'status' { Get-SlurmStatus }
    'submit' { Submit-Job }
    'queue'  { Get-JobQueue }
    'logs'   { Get-JobLogs }
    'cancel' { Stop-Job }
    'help'   { Show-Help }
}
