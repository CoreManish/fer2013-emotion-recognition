#!/bin/bash
#SBATCH --account=agni
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=results/train_%x_%j.log

cd $HOME/fer-project
source venv/bin/activate
cd src
python -u train.py --model $1 --root ../data
