#!/bin/bash
#SBATCH --job-name=pytorch_test
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=1 --gpu_cmode=shared
#SBATCH --time=4:00:00
#SBATCH --account=PAS2152

cd $SLURM_SUBMIT_DIR

module load miniconda3/23.3.1-py310  cuda/12.3.0

source pytorch

python train_dynamics2.py