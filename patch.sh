#!/bin/bash
#SBATCH --job-name=Patch-Script
#SBATCH --output=/home-mscluster/onailana/ASR-LCM-Research/patch.log
#SBATCH --partition=biggpu

source ~/.bashrc
echo "Virt env active"
conda activate env
echo "Running patch script"


python patch_script.py
