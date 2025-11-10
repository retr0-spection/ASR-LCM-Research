#!/bin/bash
#SBATCH --job-name=Latent-Training-v2
#SBATCH --output=/home-mscluster/onailana/ASR-LCM-Research/training2.log
#SBATCH --partition=bigbatch
#SBATCH --nodes=1


source ~/.bashrc
echo "Virt env active"
conda activate env
echo "Training LCM Model"


torchrun  ~/ASR-LCM-Research/training3.py --csv_path /datasets/onailana/codes/code_data.csv \
  --val_csv_path  /datasets/onailana/test_codes/test_code_data.csv \
  --val_batch_size 2 \
  --d_model 256 \
  --n_heads 8 \
  --num_layers 8 \
  --batch_size 2 \
  --effective_batch_size 2 \
  --max_batch_size 2 \
  --epochs 1500 \
  --lr 1e-4 \
  --checkpoint_dir /datasets/onailana/checkpoints2 \
  --save_every 1 \
  --log_level INFO \
  --warmup_steps 500 \
  --resume 1
