#!/usr/bin/env bash

set -x

PARTITION=$1
shift

srun -p ${PARTITION} \
    --job-name=RE \
    --gres=gpu:1 \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    python $@ &
