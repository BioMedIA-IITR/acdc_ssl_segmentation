#!/bin/bash
set -e

python inference.py \
  --data_dir /path/to/ACDC/testing \
  --checkpoint ./runs/acdc_ssl/checkpoints/best.pth \
  --output_dir ./runs/acdc_ssl/test_segmentations \
  --num_clusters 4
