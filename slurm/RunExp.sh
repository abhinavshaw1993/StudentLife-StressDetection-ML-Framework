#!/bin/bash
#
#SBATCH --mem=15000
#SBATCH --job-name=1-node-GridSearchMentalHealth
#SBATCH --partition=longq
#SBATCH --output=ClassificationExp-%A.out
#SBATCH --error=ClassificationExp-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=12
#SBATCH --mail-user=abhinavshaw@umass.edu

# Thread Limiting
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/Projects/StudentLife-StressDetection-ML-Framework/src/main
PYTHONPATH=../ python run_experiments.py