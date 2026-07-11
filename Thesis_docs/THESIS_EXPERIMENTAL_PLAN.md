# Experimental Plan and Conventions

## Status

- Week 1 (bug fixes, architecture upgrade): complete.
- Week 2 (SSL pretraining): complete.
  - `ssl_casia.pt`: 50 epochs, CASIA-B
  - `ssl_densegait.pt`: 20 epochs, DenseGait, final reconstruction loss ~0.003
- Week 3 (Fine-tuning experiments): complete.
- Week 4 (Ablations): complete.
- Week 5+ (Writing): Chapter 3 done. Chapters 1, 2, 4, 5 remain.

## Fine-tuning Recipe (verified, do not modify without justification)

```bash
!cd /content/gait-thesis && PYTHONPATH=. python src/train/train_id.py \
    --ckpt /content/drive/MyDrive/thesis/checkpoints/ssl/ssl_<source>.pt \
    --csv data/casia-b_pose_coco.csv \
    --region R1 \
    --seq_len 56 --stride 8 \
    --steps 6000 \
    --batch_size 64 \
    --use_arcface 1 --arc_s 16 --arc_m 0.20 \
    --warmup_steps 500 \
    --arc_s_min 4.0 --arc_s_warm 1000 \
    --freeze_backbone 0 --lr_backbone 1e-6 --lr_head 1e-3 \
    --pose_aug 0 --balance_cond 0 \
    --eval_protocol knn --gallery_cond nm --probe_cond <cond> \
    --gallery_per_subject 4 \
    --knn_rule sim --knn_k 1 --knn_view_match 1 \
    --limit_subjects 0 \
    --seed <0|1|2> \
    --log_dir /content/drive/MyDrive/thesis/runs/ \
    --exp_name <descriptive>
```

## What NOT to Do

- Use `ssl_best.pt` or any pre-Week-2 checkpoint (architecture mismatch, will fail strict load).
- Use `--limit_subjects 60` for reportable runs (non-standard 12-subject eval).
- Use `--gallery_cond` including the probe condition (contamination).
- Use `--seq_len 30 --stride 30` (outside SSL pretraining context window).
- Use `--use_arcface 0` for headline numbers (verified too weak).
- Run more experiments now. Focus on writing.

## Complete Experiment Matrix (all done)

**Phase 1: Main protocol (18 runs)**
- 3 protocols × 2 pretraining sources × 3 seeds = 18 runs
- Results in headline table below

**Phase 2: SSL contribution baselines (3-5 runs)**
- 3 protocols × random init × 1-3 seeds

**Phase 3: Ablations (13+ runs)**
- Gallery size (3 seeds each on nm→cl): g ∈ {1, 2, 4, 8}
- Region: R1, R2, ALL (3 seeds each on nm→cl)
- View matching: 0 vs 1 (3 seeds each on nm→cl)
- Backbone freeze: 0 vs 1 (3 seeds each on nm→cl)
- ArcFace: 0 vs 1 (3 seeds each on nm→cl)

## Naming Convention Used

Pattern: `<pretraining>_<region>_<protocol>_<seed>` with suffixes for ablations.
- `densegait_R1_nm2cl_s0`, `casia_R1_nm2cl_s0`
- `scratch_R1_nm2cl_s0`
- `densegait_R1_nm2cl_g8_s0` (gallery ablation, g=8)

## Known Issues, Now Fixed

- ArcFace scale ramp bug (line 952 of `train_id.py`) — fixed. Current runs use the corrected schedule.

## Known Confounds Disclosed in Thesis

- DenseGait SSL was 20 epochs vs CASIA's 50 (compute-constrained). Disclosed in Section 3.3.3.
- CASIA-SSL saw all 124 subjects during pretraining (no labels); DenseGait-SSL had full subject separation. Disclosed in Sections 3.3.4 and 3.7.1.
- nm→nm evaluation has gallery-probe overlap at the window level. Disclosed in Section 3.7.2.