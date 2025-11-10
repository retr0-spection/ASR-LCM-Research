#!/bin/bash
#SBATCH --job-name=Latent-Code-Gen
#SBATCH --output=/home-mscluster/onailana/result.txt
#SBATCH --partition=biggpu

source ~/.bashrc
echo "Virt env active"
conda activate env
echo "Running code generation script"
python ~/ASR-LCM-Research/codes_generation.py 
