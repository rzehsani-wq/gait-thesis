# Thesis Context

**Title (proposed):** Self-Supervised Transformer-Based Gait Recognition Under Clothing Variation  
**Student:** Reza Ehsani, MSc Computer Science, Bishop's University  
**Supervisors:** Dr. Madjid Allili (main), Dr. Ayoub (secondary)  
**Status:** Chapter 3 complete and reviewed. Chapters 1, 2, 4, 5 remaining. Supervisor has approved revision plan; no further alignment meeting needed.  
**Submission target:** ~5 weeks from now.

## One-paragraph summary

This thesis investigates whether a factorized spatial-temporal transformer architecture, pretrained via self-supervised masked-joint reconstruction, can perform competitive pose-based gait recognition under clothing variation. The method is evaluated on CASIA-B under the standard subject-disjoint protocol across three conditions (matched walking nm→nm, carrying bg, clothing change cl), with SSL pretraining conducted on two sources: in-domain (CASIA-B) and large-scale in-the-wild (DenseGait, 217k tracklets).

## Contributions (three, not four — audit contribution was dropped)

1. **A factorized spatial-temporal transformer architecture** for pose-based gait recognition under clothing variation, designed with explicit inductive biases for small-dataset regimes. Distinguished from existing methods by:
   - Anatomy-agnostic factorized attention (vs. GaitPT's hierarchical body-part groupings; vs. GaitGraph's skeletal graph convolutions)
   - Pure masked-joint reconstruction SSL without attribute supervision (vs. GaitFormer's multi-task SSL)
   - Single-scale temporal modeling with relative positional bias (vs. GaitPTv2's multi-scale)
   - Learned features from raw coordinates (vs. PoseGait's hand-crafted geometric features)

2. **Empirical evidence that in-domain SSL pretraining (CASIA) outperforms large-scale in-the-wild SSL pretraining (DenseGait)** on the controlled benchmark by a consistent 1.8-2.3 points across all three evaluation protocols, despite ~10× more pretraining data in the in-the-wild source. Suggests domain alignment dominates scale at this scale.

3. **Quantification of SSL pretraining benefit as a function of evaluation condition difficulty**: +5.6 points on matched walking, +11.8 points on carrying, +13.6 points on clothing variation. SSL representations are most valuable when surface appearance cues become unreliable.

Note: A protocol-audit contribution was considered and dropped on honesty grounds — the identified issues were personal implementation issues, not documented field-wide patterns.

## Model

- **Type:** Factorized spatial-temporal transformer, ~813k parameters (846k with supervised head)
- **Tokens:** T×J per sample (frame-joint pairs); for T=56, J=17: 952 tokens
- **Layers:** Two pairs of (Spatial, Temporal) attention blocks, spatial-first (S-T-S-T). d_model=128, nhead=4, depth_pairs=2.
- **Input:** Per-joint coordinate embedding (2 → d_model) + learned joint-ID embedding + learned time positional embedding
- **RPE:** Relative temporal positional bias on temporal blocks only (log-bucketed)
- **Pooling for identification:** Mean over all T×J tokens
- **Variable joint count:** Via `joint_indices` parameter (same architecture serves R1, R2, ALL)

## SSL Pretraining

- **Objective:** Masked-joint reconstruction, 35% mask ratio, MSE loss on masked positions only
- **Sequence:** seq_len=56 frames, stride 8 (CASIA) or 4 (DenseGait)
- **Optimizer:** AdamW, lr=3e-4, weight_decay=1e-4, gradient clipping=1.0, AMP
- **Duration:** 50 epochs for CASIA, 20 epochs for DenseGait (a known asymmetry disclosed in the thesis)
- **Checkpoints:** `ssl_casia.pt` and `ssl_densegait.pt` (in `MyDrive/thesis/checkpoints/ssl/`)

## Fine-tuning Recipe (verified working configuration)

```bash
--ckpt /content/drive/MyDrive/thesis/checkpoints/ssl/ssl_<source>.pt
--csv data/casia-b_pose_coco.csv
--region R1
--seq_len 56 --stride 8
--steps 6000
--batch_size 64
--use_arcface 1 --arc_s 16 --arc_m 0.20
--warmup_steps 500
--arc_s_min 4.0 --arc_s_warm 1000
--freeze_backbone 0 --lr_backbone 1e-6 --lr_head 1e-3
--pose_aug 0 --balance_cond 0
--gallery_cond nm --probe_cond <cond>
--gallery_per_subject 4
--knn_rule sim --knn_k 1 --knn_view_match 1
--limit_subjects 0
```

Two-phase loss: CE on auxiliary linear head for steps 0-499, ArcFace on MetricHead for steps 500-5999. Scale $s$ ramps linearly from 4 to 16 over 1000 steps starting at step 500. Loss shows a discontinuity at the transition because Phase 2 starts with an essentially untrained metric head; validation accuracy continues climbing through the transition.

## Evaluation Protocol

- **Split:** Subject-disjoint 80/20 (~99 train, ~25 val subjects). SSL pretraining sees all 124 subjects for the CASIA-SSL configuration; DenseGait-SSL has full separation.
- **Gallery:** Random subsample of val-subject nm windows, stratified by view when `view_match=1`, using fixed random seed at eval time.
- **Probe:** All val-subject windows of the target condition. For nm→nm, gallery windows are not excluded from the probe set — accuracy reflects mixture of near-duplicate and cross-clip retrieval, transparently noted.
- **Retrieval:** Centroid k-NN with k=1, cosine similarity, L2-normalized embeddings.

## Data Preprocessing

- Mid-hip centering, median shoulder-width scaling, per-sequence std normalization to ~1, clipping to ±8
- Applied uniformly to both CASIA-B and DenseGait
- Confidence threshold (0.5 for CASIA, 0.3 for DenseGait)
- DenseGait joint order recovered empirically (documented in Appendix A of the thesis)

## Region Definitions (COCO-17)

- R1 (upper body): head + arms = 11 joints, indices 0-10
- R2 (lower body): legs = 6 joints, indices 11-16
- ALL: full 17 joints

## Datasets

- **CASIA-B:** 124 subjects, walking under nm (6 sequences), bg (2), cl (2), across 11 view angles (0° to 180°). Pose in COCO-17 format via HRNet (from GaitGraph release).
- **DenseGait (Cosma & Radoi 2022):** 217k tracklets, 410 hours, from public surveillance streams. Poses via AlphaPose, custom 18-joint ordering (recovered empirically for COCO-17 alignment).

## What NOT to include in remaining chapters

- Do not reference the original thesis draft or its low numbers.
- Do not frame architectural choices as "fixing bugs."
- Do not claim novelty around evaluation protocol pitfalls (dropped as contribution).
- Do not overclaim vs. SOTA — position honestly as competitive with older pose-only methods.

## Honest SOTA Positioning

Competitive with PoseGait (~36% on nm→cl) and older pose-only methods. Below recent state-of-the-art (GaitPT, GaitFormer, GaitPTv2 report higher nm→cl). Position as "competitive with established pose-only baselines" rather than "state-of-the-art."

Verify these SOTA numbers against original papers before publishing:

| Method | Modality | nm→nm | nm→bg | nm→cl |
|---|---|---|---|---|
| GaitSet (2019) | Silhouette | ~96 | ~87 | ~70 |
| GaitPart (2020) | Silhouette | ~96 | ~92 | ~79 |
| PoseGait (2020) | Pose | ~69 | ~45 | ~36 |
| GaitGraph (2021) | Pose | ~88 | ~75 | ~66 |
| **This work (CASIA-SSL)** | Pose | **96.0** | **66.1** | **38.9** |
| **This work (DenseGait-SSL)** | Pose | 93.7 | 63.9 | 37.1 |

## Chapter 3 Status

Complete. Structure and content reviewed. Two open items:
- Verify CASIA reconstruction loss value on page 12 (load ssl_casia.pt metadata)
- Optional: add one-sentence note that pose_aug=0 in Section 3.6.3