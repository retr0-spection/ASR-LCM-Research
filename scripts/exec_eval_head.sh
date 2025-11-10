#!/bin/bash
#SBATCH --job-name=Atten_Vis
#SBATCH --output=/home-mscluster/onailana/eval_head_res.txt
#SBATCH --partition=bigbatch

source ~/.bashrc
echo "Virt env active"
conda activate env
echo "Running Model Evaluation"
python ~/ASR-LCM-Research/eval_heads.py --csv_path /datasets/onailana/test_codes/test_code_data.csv --checkpoint_path /datasets/onailana/checkpoints1/lcm_epoch20.pt --batch_size 2

