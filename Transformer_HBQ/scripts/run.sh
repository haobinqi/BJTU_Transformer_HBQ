#!/usr/bin/env bash
set -e

# 修改下面路径为你的 IWSLT 数据目录（包含 train.en, train.de 等）
IWSLT_DIR="/dataset/iwslt2017"
OUTDIR="./results"
SEED=42
GPU=0

export CUDA_VISIBLE_DEVICES=${GPU}

python3 src/train.py \
  --data_dir ${IWSLT_DIR} \
  --outdir ${OUTDIR} \
  --epochs 20 \
  --batch_size 4096 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 1024 \
  --num_layers 4 \
  --lr 2e-4 \
  --dropout 0.1 \
  --seed ${SEED} \
  --save_every 1000 \
  --device cuda
