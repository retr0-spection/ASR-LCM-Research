#!/bin/bash
#SBATCH --job-name=Latent-Training-v2
#SBATCH --output=/home-mscluster/onailana/ASR-LCM-Research/scripts/test.log
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2


source ~/.bashrc
echo "Virt env active"
conda activate env


echo "===== SLURM ENVIRONMENT VARIABLES ====="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"

# Python test script
python - << EOF
import os
import torch

local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"Local rank: {local_rank}")
print(f"Number of visible GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Optional: test setting the device
if local_rank < torch.cuda.device_count():
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} using GPU {torch.cuda.current_device()}")
else:
    print(f"Process {local_rank} has no GPU available")
EOF

