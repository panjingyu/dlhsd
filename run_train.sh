#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python -u train.py --aug
for id in `seq 1 4`;
do
    python -u test.py --aug --id $id
done
# python -u attack.py --aug --cure-l 5e-6 --cure-h 5e-4 --id 1
# python -u attack.py --aug --cure-l 5e-6 --cure-h 5e-4 --id 2
# python -u attack.py --aug --cure-l 5e-6 --cure-h 5e-4 --id 3
# python -u attack.py --aug --cure-l 5e-6 --cure-h 5e-4 --id 4
