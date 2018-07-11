#!/bin/bash
#
#SBATCH --mem=30000
#SBATCH --job-name=Grid-Regress
#SBATCH --partition=longq
#SBATCH --output=RegressionExp-%A.out
#SBATCH --error=RegressionExp-%A.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=12
#SBATCH --time=15-00:00
#SBATCH --mail-user=abhinavshaw@umass.edu

# Thread Limiting
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/Projects/StudentLife-StressDetection-ML-Framework/src/main
PYTHONPATH=../ python run_regressor.py