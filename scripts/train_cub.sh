#!/bin/bash
# HyperCGAN on CUB-200-2011 (256x256), settings used in the paper: 8 GPUs, batch 128, ADA (bg).
# Usage: bash scripts/train_cub.sh [extra train.py options, e.g. --resume=path/to/network-snapshot.pkl]
set -e

DATA_DIR=${DATA_DIR:-datasets}
OUTDIR=${OUTDIR:-training-runs/cub}
GPUS=${GPUS:-8}

python train.py \
    --outdir=$OUTDIR \
    --data=$DATA_DIR/cub_train.zip \
    --gpus=$GPUS \
    --batch=128 \
    --mirror=1 \
    --aug=ada \
    --augpipe=bg \
    --temp=0.5 \
    --itd=5 \
    --itc=10 \
    --gamma=10 \
    "$@"
