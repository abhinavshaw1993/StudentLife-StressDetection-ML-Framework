#!/bin/bash
#
#SBATCH --mem=3000
#SBATCH --job-name=Grid-Classif
#SBATCH --partition=defq
#SBATCH --output=confidence_interval-%A.out
#SBATCH --error=confidence_interval-%A.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=abhinavshaw@umass.edu

# Thread Limiting
export MKL_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export OMP_NUM_THREADS=5

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/Projects/StudentLife-StressDetection-ML-Framework/src/main
PYTHONPATH=../ python run_confidence_intervals.py