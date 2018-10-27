#!/bin/bash

#SBATCH --job-name=l5pc_benchmark_timed
#SBATCH --error=logs/l5pc_benchmark_timed.stdout
#SBATCH --output=logs/l5pc_benchmark_timed.stdout
#SBATCH -t 5:00:00
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -L SCRATCH
#SBATCH --mail-user=tom.chartrand@alleninstitute.org
#SBATCH --mail-type=ALL

set -e
set -x

./start_timed.sh 50 2
