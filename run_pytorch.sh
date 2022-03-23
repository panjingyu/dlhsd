#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python -u train_pytorch.py --aug
for id in `seq 1 4`; do
    python -u test_pytorch.py --aug --id $id
done
