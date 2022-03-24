#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

for h in `seq 1 5`;
do
    for l in 1 2 5 10;
    do
        args="--aug --cure-l $l --cure-h $h"
        python -u train_pytorch.py $args
        for id in `seq 1 4`; do
            python -u test_pytorch.py $args --id $id
        done
    done
done