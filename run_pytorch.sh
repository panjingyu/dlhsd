#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

export args="--aug --cure-l 1 --cure-h 5e-1"

python -u train_pytorch.py $args
for id in `seq 1 4`;
do
    python -u test_pytorch.py $args --id $id
done
