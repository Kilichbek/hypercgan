#!/bin/bash
# FID (30k generated images vs. all real test images) and CLIP R-precision of a trained model.
# Usage: bash scripts/evaluate.sh path/to/network-snapshot.pkl datasets/cub_test.zip [num_gpus]
set -e

NETWORK=$1
TEST_DATA=$2
GPUS=${3:-1}

python calc_metrics.py --network=$NETWORK --data=$TEST_DATA --metrics=fid50k_full --gpus=$GPUS
python clip_rprecision.py --network=$NETWORK --data=$TEST_DATA
