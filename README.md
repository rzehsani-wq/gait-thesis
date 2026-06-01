# Self-Supervised Transformer-Based Gait Recognition Under Clothing Variation

Master's thesis project, Bishop's University, 2026.

This codebase implements:
- A pose-based factorized spatial-temporal transformer for gait recognition
- Self-supervised pretraining via masked-joint reconstruction
- Supervised fine-tuning with ArcFace metric learning
- Evaluation on CASIA-B via centroid-based kNN retrieval

## Structure

src/
├── datasets/       # CASIA-B and DenseGait dataset loaders
├── models/         # Transformer backbone, ID head, ArcFace head
├── train/          # SSL pretraining and supervised fine-tuning scripts
├── utils/          # Pose normalization, region partitioning, parsing, logging
└── data/           # Pose augmentation transforms

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Usage

### Self-supervised pretraining

```bash
# On CASIA-B:
python -m src.train.train_ssl --dataset casia \
    --csv data/casia-b_pose_coco.csv \
    --seq_len 56 --stride 8 --epochs 50 --batch_size 64 \
    --save ckpts/ssl_casia_v2.pt

# On DenseGait:
python -m src.train.train_ssl --dataset densegait \
    --csv data/densegait_pose_coco_filtered.csv \
    --poses_dir data/poses \
    --seq_len 56 --stride 4 --epochs 50 --batch_size 64 \
    --save ckpts/ssl_densegait_v2.pt
```

### Supervised fine-tuning + kNN evaluation

```bash
python -m src.train.train_id \
    --csv data/casia-b_pose_coco.csv \
    --ckpt ckpts/ssl_casia_v2.pt \
    --region R1 --seq_len 56 --stride 2 \
    --gallery_cond nm --probe_cond cl \
    --gallery_per_subject 4 --knn_rule sim --knn_k 1 \
    --knn_view_match 1 --use_arcface 1 \
    --arc_s 16 --arc_m 0.20 --warmup_steps 300 \
    --steps 1500 --batch_size 64 \
    --lr_head 2e-3 --lr_backbone 3e-6 \
    --exp_name baseline_nm2cl
```

## Notes

Data files (CASIA-B CSV, DenseGait pose arrays) are not committed.
See `preprocessing_validation.md` for details on the preprocessing pipeline.