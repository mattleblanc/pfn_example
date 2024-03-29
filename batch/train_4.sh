#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o logs/slurm-%j.out
#SBATCH -J PFN-qg-l4

source tensorflow.venv/bin/activate

python pfn_train.py  --doEarlyStopping --latentSize=4 --makeROCs --label="l4" 
