#!/bin/bash
set -e

python train.py \
  --data_dir /path/to/ACDC/training \
  --output_dir ./runs/acdc_ssl \
  --epochs 100 \
  --batch_size 8 \
  --num_clusters 4 \
  --amp
