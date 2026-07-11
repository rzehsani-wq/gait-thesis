# Thesis Context (for writing chat)

## Basics

**Title (proposed):** Self-Supervised Transformer-Based Gait Recognition Under Clothing Variation  
**Student:** Reza Ehsani, MSc Computer Science, Bishop's University  
**Supervisors:** Dr. Madjid Allili (main), Dr. Ayoub (secondary)  
**Status:** Final revision phase. All experiments complete. Beginning rewrite of revised draft.  
**Submission target:** ~5 weeks from now.

## One-paragraph summary

The thesis investigates whether a factorized spatial-temporal transformer architecture, pretrained via self-supervised masked-joint reconstruction, can perform competitive pose-based gait recognition under clothing variation. The method is evaluated on CASIA-B under the standard subject-disjoint protocol across three conditions (matched walking nm→nm, carrying bg, clothing change cl), and self-supervised pretraining is conducted on two sources: in-domain (CASIA-B itself) and large-scale in-the-wild (DenseGait, 217k tracklets).

## Contributions

1. **A factorized spatial-temporal transformer architecture** for pose-based gait recognition under clothing variation, designed with explicit inductive biases for small-dataset regimes. Distinguished from existing pose-based gait methods by:
   - Anatomy-agnostic factorized attention (vs. GaitPT's hierarchical body-part groupings; vs. GaitGraph's skeletal graph convolutions)
   - Pure masked-joint reconstruction SSL without attribute supervision (vs. GaitFormer's multi-task SSL)
   - Single-scale temporal modeling with relative positional bias (vs. GaitPTv2's multi-scale)
   - Learned features from raw coordinates (vs. PoseGait's hand-crafted geometric features)

2. **Empirical evidence that in-domain SSL pretraining (CASIA) outperforms large-scale in-the-wild SSL pretraining (DenseGait)** on the controlled benchmark by a consistent 1.8–2.3 points across all three evaluation protocols, despite ~10× more pretraining data in the in-the-wild source. This suggests that domain alignment dominates dataset scale at the scale evaluated.

3. **Quantification of SSL pretraining benefit as a function of evaluation condition difficulty**: +5.6 points on nm→nm (matched walking), +11.8 points on nm→bg (carrying), and +13.6 points on nm→cl (clothing variation). SSL representations are most valuable precisely when surface appearance cues become unreliable.

## Headline results

Standard CASIA-B protocol, all 124 subjects, subject-disjoint 80/20 split, 3 seeds (mean ± std):

| Protocol | DenseGait-SSL | CASIA-SSL | Random init |
|---|---|---|---|
| nm→nm | 93.7 ± 2.3 | **96.0 ± 1.5** | 88.1 |
| nm→bg | 63.9 ± 5.6 | **66.1 ± 5.5** | 52.1 |
| nm→cl | 37.1 ± 5.5 | **38.9 ± 6.7** | 23.5 |

Random init is single seed for nm→bg and nm→nm; three seeds for nm→cl (23.5% is the mean: 27.8 / 24.7 / 17.9).

## Ablation results (all on nm→cl, CASIA-SSL, 3 seeds unless noted)

**Gallery size** (mean ± std):
| g | Mean | Std |
|---|---|---|
| 1 | 33.5 | 4.5 |
| 2 | 36.8 | 5.5 |
| **4** (default) | **39.1** | 6.6 |
| 8 | 39.2 | 6.1 |
Monotonic growth then saturation at g=4. Choose g=4 as default for all main runs.

**Region:**
| Region | Joints | Mean | Std |
|---|---|---|---|
| R1 (head+arms) | 11 | 39.1 | 6.6 |
| R2 (legs) | 6 | 32.0 | 3.2 |
| ALL (full) | 17 | **44.0** | 6.4 |
ALL outperforms R1 by ~5 points. Upper-body region (R1) clearly beats lower-body region (R2) in isolation, but combining them is best. Headline numbers use R1 because that's where 3-seed runs were done for the main protocols; ALL is shown only in the ablation.

**View matching:**
| view_match | Mean | Std |
|---|---|---|
| 0 | 30.4 | 10.9 |
| 1 (default) | 39.1 | 6.6 |
View matching helps by +8.7 points and reduces seed variance by 39%.

**Backbone freeze:**
| freeze | Mean | Std |
|---|---|---|
| 0 (unfrozen, default) | 39.1 | 6.6 |
| 1 (frozen) | 34.4 | 7.2 |
Unfreezing helps by +4.7 points. Frozen-backbone baseline of 34.4% is still meaningful — SSL features alone provide most of the benefit; backbone refinement adds ~5 points.

**Metric loss:**
| Loss | Mean | Std |
|---|---|---|
| CE only | 19.3 | 0.4 |
| ArcFace (default) | 39.1 | 6.6 |
ArcFace is the single largest contributor: +19.8 points over CE. Note the dramatic std difference — CE-only converges to a similar mediocre solution every seed (0.4 std), while ArcFace produces real seed variance because it actually fits.

## Model architecture

- **Type:** Factorized spatial-temporal transformer
- **Token granularity:** T×J tokens (one per frame-joint pair). For T=56 frames and J=17 joints, 952 tokens per sample.
- **Input embedding:** Per-joint coordinate embedding (2 → d_model) + learned joint-identity embedding (17 entries × d_model) + learned absolute time positional embedding
- **Encoder structure:** Two pairs of (Spatial, Temporal) attention blocks, spatial-first ordering (S-T-S-T)
  - Spatial blocks: self-attention across J joints within each frame, no positional bias
  - Temporal blocks: self-attention across T frames within each joint, with relative temporal positional bias (log-bucketed)
- **Hyperparameters:** d_model=128, nhead=4, depth_pairs=2, dropout=0.1, mlp_ratio=4
- **Total parameters:** ~813,000 in the SSL reconstruction model; ~846,000 with the supervised classification/metric head
- **Pooling for identification:** Mean over all T×J tokens after encoding
- **Variable joint count:** Supported via joint_indices parameter; allows region-based input without separate models

## SSL pretraining details

- **Objective:** Masked-joint reconstruction (35% of (frame, joint) positions randomly masked, MSE loss on masked positions only)
- **Window size:** seq_len = 56 frames
- **Stride:** 8 for CASIA, 4 for DenseGait
- **Optimizer:** AdamW, lr = 3e-4, weight decay = 1e-4, gradient clipping = 1.0, AMP
- **Batch size:** 64
- **Epochs:** 50 for CASIA (`ssl_casia.pt`), 20 for DenseGait (`ssl_densegait.pt`) — discrepancy noted as a confound
- **Final reconstruction loss:** ~0.0027 for DenseGait at epoch 20

## Supervised fine-tuning recipe (the verified working configuration)

```bash
--ckpt ckpts/ssl_<source>.pt
--csv data/casia-b_pose_coco.csv
--region R1 (or ALL for ablation)
--seq_len 56 --stride 8
--steps 6000
--batch_size 64
--use_arcface 1 --arc_s 16 --arc_m 0.20
--warmup_steps 500
--arc_s_min 4.0 --arc_s_warm 1000
--freeze_backbone 0 --lr_backbone 1e-6 --lr_head 1e-3
--pose_aug 0 --balance_cond 0
--gallery_cond nm --probe_cond <nm|bg|cl>
--gallery_per_subject 4
--knn_rule sim --knn_k 1 --knn_view_match 1
--limit_subjects 0
```

Standard CASIA-B protocol; subject-disjoint 80/20 split (~99 train subjects, ~25 val subjects).

## Datasets

### CASIA-B
- 124 subjects, each walking under nm (6 sequences), bg (2), cl (2)
- 11 view angles per condition per subject (0°, 18°, 36°, ..., 180°)
- Pose extracted via standard methods (coordinates per joint per frame)
- 17 COCO joints used for analysis

### DenseGait (Cosma & Radoi 2022)
- 217k tracklets, 410 hours total
- Collected from public surveillance streams, extracted with AlphaPose
- 18 joints per frame in custom ordering (mapped to COCO-17 via empirically-recovered remap)

## Data preprocessing

- Mid-hip centering, median shoulder-width scaling, per-sequence std normalization to ≈1, clipping to ±8
- Applied uniformly to both CASIA-B and DenseGait (despite DenseGait being already pre-normalized by the authors; we apply our normalization on top to align distributions)
- Confidence threshold applied to filter low-confidence joints from normalization estimates

## Region definitions (COCO-17)

- **R1 (upper body):** head (nose, eyes, ears) + arms (shoulders, elbows, wrists). 11 joints. Indices [0-10].
- **R2 (lower body):** hips, knees, ankles. 6 joints. Indices [11-16].
- **ALL:** Full 17-joint skeleton.

## What NOT to include in the rewrite

- **Do not mention the original draft.** The thesis is forward-facing.
- **Do not reference the previous low results.** The numbers above are the numbers.
- **Do not frame anything as "we fixed bugs."** Everything is presented as the protocol used.
- **Do not invent contributions beyond the three listed above.** Stay honest.
- **Do not claim novelty around evaluation protocol pitfalls.** That was strategic over-framing and has been dropped.

## Honest positioning relative to SOTA

The thesis is positioned within the competitive band of older pose-only methods (PoseGait, GaitGraph). It does NOT beat recent state-of-the-art pose methods (GaitPT, GaitFormer, GaitPTv2 report higher nm→cl numbers). The honest framing is:

> "Our results are competitive with established pose-only baselines such as PoseGait. While the absolute numbers fall below the most recent state-of-the-art methods that benefit from larger models, more extensive pretraining, or more complex architectures, the proposed approach demonstrates the viability of small-dataset, parameter-efficient factorized transformers for this task."

Do not overclaim. Do not say "we close the gap with silhouette models." Do not say "we outperform method X" unless we actually do.

## SOTA comparison table (verify before publishing)

These approximate numbers need verification against original papers:

| Method | Modality | nm→nm | nm→bg | nm→cl |
|---|---|---|---|---|
| GaitSet (Chao et al. 2019) | Silhouette | ~96 | ~87 | ~70 |
| GaitPart (Fan et al. 2020) | Silhouette | ~96 | ~92 | ~79 |
| PoseGait (Liao et al. 2020) | Pose | ~69 | ~45 | ~36 |
| GaitGraph (Teepe et al. 2021) | Pose | ~88 | ~75 | ~66 |
| **This work (CASIA-SSL)** | **Pose** | **96.0** | **66.1** | **38.9** |
| **This work (DenseGait-SSL)** | **Pose** | **93.7** | **63.9** | **37.1** |

Verify each number against the original paper before publishing. Be explicit in the table caption that different methods may use slightly different protocols.

## Comparison framing for the architecture chapter

The architectural contribution should be framed against five reference points:

- **vs. GaitPT (Cosma 2022):** anatomy-agnostic factorized attention vs. hierarchical body-part groupings
- **vs. GaitFormer (Cosma & Radoi 2022):** pure masked-joint reconstruction SSL vs. multi-task SSL with attributes
- **vs. GaitPTv2:** single-scale temporal vs. multi-scale temporal
- **vs. GaitGraph (Teepe 2021):** self-attention vs. graph convolution
- **vs. PoseGait (Liao 2020):** learned features from raw coordinates vs. hand-crafted geometric features

Unifying design philosophy: "This work investigates whether a transformer architecture with minimal anatomical priors can match the performance of methods that rely on explicit structural priors (anatomical hierarchies, skeletal graphs, hand-crafted features, attribute supervision)."