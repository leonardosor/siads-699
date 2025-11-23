# Local SLURM Training Setup

This directory contains scripts for running YOLOv8 training jobs on the local SLURM cluster.

## Files

- `batch_job.sh` - Original Great Lakes HPC batch script
- `batch_job_local.sh` - Local SLURM batch script (adapted for local development)
- `submit_local_job.sh` - Helper script to submit jobs to local SLURM
- `test_slurm_job.sh` - Simple test job to verify SLURM is working

## Prerequisites

1. SLURM containers must be running:
   ```powershell
   cd config
   docker-compose up -d slurm-controller slurm-node1
   ```

2. Verify SLURM is working:
   ```powershell
   docker exec slurm-controller sinfo
   ```

## Usage

### Method 1: Using the Helper Script (Recommended)

```bash
# Access the SLURM controller
docker exec -it slurm-controller bash

# Navigate to training directory
cd /workspace/src/training

# Submit a standard training job
./submit_local_job.sh --epochs 100 --batch 4

# Submit with Optuna optimization
./submit_local_job.sh --optuna --trials 20 --epochs 50

# Get help
./submit_local_job.sh --help
```

### Method 2: Direct sbatch Submission

```bash
# Access the SLURM controller
docker exec -it slurm-controller bash

# Fix line endings and submit
sed -i 's/\r$//' /workspace/src/training/batch_job_local.sh
sbatch /workspace/src/training/batch_job_local.sh

# Or with custom parameters
sbatch --export=ALL,EPOCHS=100,BATCH=8 /workspace/src/training/batch_job_local.sh
```

### Method 3: From Windows PowerShell

```powershell
# Submit job from Windows
docker exec slurm-controller bash -c "sed -i 's/\r$//' /workspace/src/training/batch_job_local.sh && sbatch /workspace/src/training/batch_job_local.sh"

# Check job status
docker exec slurm-controller squeue

# View logs
docker exec slurm-controller tail -f /workspace/logs/capstone_local-<JOB_ID>.log
```

## Monitoring Jobs

```bash
# View job queue
squeue

# View detailed job info
scontrol show job <JOB_ID>

# View job output (while running or after completion)
tail -f /workspace/logs/capstone_local-<JOB_ID>.log

# Cancel a job
scancel <JOB_ID>
```

## Environment Variables

You can customize training behavior with environment variables:

- `EPOCHS` - Number of training epochs (default: 250)
- `BATCH` - Batch size (default: 4)
- `IMGSZ` - Image size (default: 1024)
- `PATIENCE` - Early stopping patience (default: 60)
- `USE_OPTUNA` - Enable Optuna optimization: 0 or 1 (default: 0)
- `N_TRIALS` - Number of Optuna trials (default: 20)
- `RUN_NAME` - Custom run name (default: auto-generated timestamp)
- `HYPERPARAMS` - Additional hyperparameters (default: "--cache")

Example:
```bash
sbatch --export=ALL,EPOCHS=100,BATCH=8,USE_OPTUNA=1,N_TRIALS=10 /workspace/src/training/batch_job_local.sh
```

## Output Locations

- **Training runs**: `models/experiments/active/`
- **Artifacts (tar.gz)**: `models/artifacts/`
- **Job logs**: `logs/capstone_local-<JOB_ID>.log`

## Differences from Great Lakes Script

The local script differs from `batch_job.sh` in the following ways:

1. **Partition**: Uses `mypartition` instead of `spgpu`
2. **Resources**: No GPU/memory/CPU requirements (SLURM manages allocation)
3. **Modules**: No module loading (uses container environment directly)
4. **Account**: No account specification needed
5. **Email**: No email notifications
6. **Environment**: Uses the devcontainer Python environment instead of conda

## Troubleshooting

### Line Ending Issues
If you get "DOS line breaks" error:
```bash
docker exec slurm-controller sed -i 's/\r$//' /workspace/src/training/batch_job_local.sh
```

### Job Not Starting
Check node status:
```bash
docker exec slurm-controller sinfo
```

All nodes should show as "idle". If not, restart SLURM containers:
```powershell
cd config
docker-compose restart slurm-controller slurm-node1
```

### Python/PyTorch Not Available
The SLURM nodes run independently from your devcontainer. To use Python/PyTorch:
- The xenonmiddleware/slurm image includes basic Python
- For full environment, you may need to install packages in the SLURM containers or mount conda environments

### Job Output Not Found
Ensure the logs directory exists:
```bash
docker exec slurm-controller mkdir -p /workspace/logs
```

## Testing

Run a simple test job to verify everything works:
```bash
docker exec slurm-controller bash -c "sed -i 's/\r$//' /workspace/src/training/test_slurm_job.sh && sbatch /workspace/src/training/test_slurm_job.sh"
```

Check the output:
```bash
docker exec slurm-controller ls -la /workspace/slurm_test_*.log
docker exec slurm-controller cat /workspace/slurm_test_*.log
```
