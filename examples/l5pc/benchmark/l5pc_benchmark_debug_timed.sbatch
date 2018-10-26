#!/bin/bash

#SBATCH --job-name=l5pc_benchmark
#SBATCH --error=logs/l5pc_benchmark.stdout
#SBATCH --output=logs/l5pc_benchmark.stdout
#SBATCH -t 00:05:00
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -L SCRATCH
#SBATCH --mail-user=tom.chartrand@alleninstitute.org
#SBATCH --mail-type=ALL

set -e
set -x

./start_timed.sh 2 1
