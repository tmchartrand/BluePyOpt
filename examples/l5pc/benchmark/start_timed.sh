#!/bin/bash

set -e
set -x

PWD=$(pwd)
LOGS=$PWD/logs
mkdir -p $LOGS

cd ..

OFFSPRING_SIZE=128
MAX_NGEN=$1

export IPYTHONDIR=${PWD}/.ipython
export IPYTHON_PROFILE=bm_timed.${SLURM_JOBID}
ipcontroller --init --ip='*' --sqlitedb --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 10
srun -n 64 --output="${LOGS}/engine_%j_%2t.out" ipengine --timeout=300 --profile=${IPYTHON_PROFILE} &
sleep 10

CHECKPOINTS_DIR="checkpoints/bm_timed.${SLURM_JOBID}"
mkdir -p ${CHECKPOINTS_DIR}

pids=""
for seed in $(seq $2); do
    python opt_l5pc.py                     \
        -vv                                \
        --timed                            \
        --offspring_size=${OFFSPRING_SIZE} \
        --max_ngen=${MAX_NGEN}             \
        --seed=${seed}                     \
        --ipyparallel                      \
        --start                            \
        --checkpoint "${CHECKPOINTS_DIR}/bm_timed_seed${seed}.pkl" &
    pids+="$! "
done

wait $pids
