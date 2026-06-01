#!/usr/bin/env bash
set -euo pipefail

CSV=${1:-data/casia-b_pose_coco.csv}
CKPT=${2:-ckpts/ssl_best.pt}

# SSL pretrain (if not already done)
python -m src.train.train_ssl --csv "$CSV" --save "$CKPT"

# R1
python -m src.train.train_id --csv "$CSV" --ckpt "$CKPT" --region R1
# R2
python -m src.train.train_id --csv "$CSV" --ckpt "$CKPT" --region R2
# ALL avg
python -m src.train.train_id --csv "$CSV" --ckpt "$CKPT" --region ALL --fusion avg
# ALL concat
python -m src.train.train_id --csv "$CSV" --ckpt "$CKPT" --region ALL --fusion concat
