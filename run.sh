#!/bin/bash
#SBATCH --job-name=gfn-tm
#SBATCH --partition=default
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=<stephen.lu@mail.mcgill.ca
#SBATCH --mail-type=ALL

cd $project/gfn-tm
module purge
module load gcc python/3.10 scipy-stack/2023b
source ~/pyenv/py310/bin/activate

export PYTHONUNBUFFERED=TRUE

python run.py --dataset 20ng --model ETM --inference_method variational --K 50 --n_iter 1000
