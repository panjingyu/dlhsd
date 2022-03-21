#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# python -u train.py | tee train.aug-nocure.log
# python -u test.py --aug --id 4
# python -u train.py --aug --cure-l 1e-6 --cure-h 5e-4
# python -u test.py --aug --cure-l 1e-6 --cure-h 5e-4 --id 1
# python -u test.py --aug --cure-l 1e-6 --cure-h 5e-4 --id 2
# python -u test.py --aug --cure-l 1e-6 --cure-h 1e-4 --id 2
# python -u test.py --aug --cure-l 1e-6 --cure-h 5e-4 --id 3
# python -u test.py --aug --cure-l 1e-6 --cure-h 5e-4 --id 4
# python -u attack.py --aug --id 1
# python -u attack.py --aug --cure-l 1e-6 --cure-h 5e-4 --id 1 --log tmp
python -u train.py --aug --cure-l 5e-6 --cure-h 5e-4 --log tmp
