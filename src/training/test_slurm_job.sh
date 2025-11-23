#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=/workspace/slurm_test_%j.log

echo "Hello from SLURM!"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
date
python3 --version 2>&1 || echo "Python not available"
