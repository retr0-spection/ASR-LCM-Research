#!/bin/bash
#SBATCH --job-name=VAL-Prep
#SBATCH --output=/home-mscluster/onailana/procval_result.txt
#SBATCH --partition=bigbatch

source ~/.bashrc
echo "Virt env active"
conda activate env
echo "Running Val Data preprocessing"
python ~/ASR-LCM-Research/process_val.py
