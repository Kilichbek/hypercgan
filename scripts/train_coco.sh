#!/bin/bash
# HyperCGAN on MS-COCO 2014 (256x256), settings used in the paper: 8 GPUs, batch 128, no augmentation.
# Usage: bash scripts/train_coco.sh [extra train.py options, e.g. --resume=path/to/network-snapshot.pkl]
set -e

DATA_DIR=${DATA_DIR:-datasets}
OUTDIR=${OUTDIR:-training-runs/coco}
GPUS=${GPUS:-8}

python train.py \
    --outdir=$OUTDIR \
    --data=$DATA_DIR/coco_train.zip \
    --gpus=$GPUS \
    --batch=128 \
    --mirror=1 \
    --aug=noaug \
    --temp=0.5 \
    --itd=5 \
    --itc=10 \
    --gamma=10 \
    "$@"
