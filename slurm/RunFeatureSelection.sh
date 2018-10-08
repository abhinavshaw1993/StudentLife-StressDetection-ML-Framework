#!/bin/bash
#
#SBATCH --mem=30000
#SBATCH --job-name=Grid-Classif
#SBATCH --partition=longq
#SBATCH --output=feature_selection-%A.out
#SBATCH --error=feature_selection-%A.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=30
#SBATCH --time=10-00:00
#SBATCH --mail-user=abhinavshaw@umass.edu

# Thread Limiting
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export OMP_NUM_THREADS=30

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/Projects/StudentLife-StressDetection-ML-Framework/src/main
PYTHONPATH=../ python run_feature_selection.py