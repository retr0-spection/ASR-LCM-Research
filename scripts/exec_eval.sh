#!/bin/bash
#SBATCH --job-name=LCM-short
#SBATCH --output=/home-mscluster/onailana/eval_result_short.txt
#SBATCH --partition=bigbatch

source ~/.bashrc
echo "Virt env active"
conda activate env
echo "Running Model Evaluation"
python ~/ASR-LCM-Research/evaluate.py --csv_path /datasets/onailana/test_codes/test_code_data.csv --checkpoint_path /datasets/onailana/checkpoints3/lcm_epoch19.pt --batch_size 2

