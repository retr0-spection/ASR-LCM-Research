#!/bin/bash
#SBATCH --job-name=Hotfix
#SBATCH --output=/home-mscluster/onailana/result.txt
#SBATCH --partition=bigbatch

source ~/.bashrc
echo "Virt env active"
conda activate env
echo "Running hotfix script"
python ~/ASR-LCM-Research/scripts/hotfix.py 
