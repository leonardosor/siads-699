# Quick Start: Local SLURM Training

## Start SLURM Cluster

```powershell
cd d:\docs\MADS\699\config
docker-compose up -d slurm-controller slurm-node1
```

## Submit Training Job

### Option 1: Interactive (Inside SLURM Container)
```powershell
# Enter the SLURM controller
docker exec -it slurm-controller bash

# Use the helper script
cd /workspace/src/training
./submit_local_job.sh --epochs 100 --batch 4

# Or submit directly
sbatch /workspace/src/training/batch_job_local.sh
```

### Option 2: One-Line from Windows
```powershell
docker exec slurm-controller bash -c "cd /workspace/src/training && ./submit_local_job.sh --epochs 100 --batch 4"
```

### Option 3: With Custom Parameters
```powershell
docker exec slurm-controller sbatch --export=ALL,EPOCHS=100,BATCH=8,USE_OPTUNA=1,N_TRIALS=10 /workspace/src/training/batch_job_local.sh
```

## Monitor Jobs

```powershell
# Check job queue
docker exec slurm-controller squeue

# View job details
docker exec slurm-controller scontrol show job <JOB_ID>

# Watch log file (replace <JOB_ID> with actual job ID)
docker exec slurm-controller tail -f /workspace/logs/capstone_local-<JOB_ID>.log
```

## Cancel Job

```powershell
docker exec slurm-controller scancel <JOB_ID>
```

## Check Results

Training outputs will be in:
- Run directory: `d:\docs\MADS\699\models\experiments\active\`
- Artifacts: `d:\docs\MADS\699\models\artifacts\`
- Logs: `d:\docs\MADS\699\logs\`

## Stop SLURM Cluster

```powershell
cd d:\docs\MADS\699\config
docker-compose stop slurm-controller slurm-node1
```

For detailed documentation, see `src/training/SLURM_LOCAL_README.md`
