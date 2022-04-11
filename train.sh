#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src/

# args="--aug --cure-l 1 --cure-h 1"
args="--aug"

# python -u train_pytorch.py $args
# for id in `seq 1 4`;
# do
#     python -u test_pytorch.py $args --id $id
# done
for id in `seq 1 4`;
do
    python -u attack_pytorch.py $args --id $id
done
