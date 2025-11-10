#!/bin/bash
#SBATCH --job-name=Latent-Training-Final
#SBATCH --output=/home-mscluster/onailana/ASR-LCM-Research/training_denoising.log
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=48:00:00              # just in case; prevents job kill
#SBATCH --mem=0                      # use all available memory

# ------------------------------
# Environment setup
# ------------------------------
source ~/.bashrc
echo "Activating virtual environment..."
conda activate env

echo "Starting Latent Consistency Model training (DDP mode)"

# ------------------------------
# NCCL + Torch Distributed config
# ------------------------------
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ------------------------------
# Sanity echo
# ------------------------------
echo "Job Node: $(hostname)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ------------------------------
# Run distributed training
# ------------------------------
torchrun --standalone --nproc_per_node=1 ~/ASR-LCM-Research/training6.py \
    --csv_path /datasets/onailana/codes/code_data.csv \
    --val_csv_path /datasets/onailana/test_codes/test_code_data.csv \
    --d_model 256 \
    --n_heads 2 \
    --num_layers 8 \
    --batch_size 2 \
    --val_batch_size 2 \
    --epochs 1500 \
    --lr 1e-4 \
    --checkpoint_dir /datasets/onailana/checkpoints6 \
    --lambda_waveform 0.5 \
    --lambda_stft 0.5 \
    --wandb_project "lcm-distill" \
    --resume 1 \
    --log_grads

# ------------------------------
# Post-job cleanup
# ------------------------------
echo "Training complete at $(date)"

