# Experimental Plan and Conventions

This document describes the specific architectural and experimental choices for 
this MSc thesis on pose-based SSL transformer gait recognition. Hyperparameters 
were chosen to match the SSL pretraining setup, NOT taken from generic 
transformer recipes. Departures from these defaults require an explicit 
justification.

## What we are actually evaluating

The thesis is a controlled empirical study of pose-based self-supervised 
transformer models for gait recognition under clothing variation, on CASIA-B.

Specifically, we want to answer:
1. Does the architecture work end-to-end (sanity question)?
2. What is the achievable Rank-1 on CASIA-B nm→cl using a pose-only 
   SSL-pretrained transformer with the standard protocol?
3. Does SSL pretraining on in-the-wild data (DenseGait) transfer better, worse, 
   or equally to in-domain pretraining (CASIA-B)?
4. How sensitive are results to evaluation choices (gallery size, view 
   matching, region selection, ArcFace vs CE)?

We are NOT trying to beat state of the art. We want defensible, reproducible 
numbers under correct protocols.

## Architecture (fixed, do not change)

- Factorized spatial-temporal transformer (Option B from architecture choice).
- Two pairs of (Spatial, Temporal) attention blocks. Spatial-first ordering.
- Spatial blocks attend across joints within a frame. Temporal blocks attend 
  across frames within a joint, with relative temporal positional bias.
- d_model = 128, nhead = 4, depth_pairs = 2.
- Per-joint coordinate embedding + learned joint-ID embedding + learned time 
  positional embedding.
- ~813k parameters total.

## Pretraining setup (already complete)

Both SSL pretrainings used masked-joint reconstruction (mask 35% of 
(frame, joint) positions; MSE loss on masked positions only) with:

- `seq_len = 56`
- `stride = 8` for CASIA, `stride = 4` for DenseGait
- batch_size = 64

The model's learned positional embeddings and temporal RPE buckets are 
calibrated for T=56 contexts. Fine-tuning with substantially different 
seq_len means the model operates outside its trained range. KEEP seq_len = 56 
during fine-tuning.

Current checkpoints:
- `ssl_casia.pt`: 50 epochs (TO BE TRAINED if not yet done)
- `ssl_densegait.pt`: 20 epochs (final reconstruction loss 0.0027)

## Fine-tuning conventions

### Sequence configuration (do not deviate without reason)

- `seq_len = 56` — matches SSL pretraining.
- `stride = 8` for training data, can be larger (e.g., 30) for eval if 
  desired for speed.

### Recommended starting recipe

For initial baseline runs, use:

```bash
--ckpt ckpts/ssl_<source>_v2.pt
--region R1                          # historically stronger than ALL or R2
--seq_len 56 --stride 8
--use_arcface 1 --arc_s 16 --arc_m 0.20 --warmup_steps 300
--freeze_backbone 1                  # safer starting point
--lr_head 1e-3
--steps 2000-4000                    # not 8000 — fewer steps with frozen backbone
--batch_size 64
--eval_protocol knn
--knn_rule sim --knn_k 1
--gallery_per_subject 4
--knn_view_match 1
--limit_subjects 0                   # FULL 124 subjects, NOT --limit_subjects 60 
--seed 0
```

### Why these specific choices

- **R1 (head + arms)**: Empirically dominant in our preliminary results, and 
  consistent with the thesis hypothesis about upper-body cues. ALL with 17 
  joints introduces more variance without proportional gain.
- **ArcFace s=16, m=0.20, warmup=300**: Earlier ArcFace runs with s=12, 
  m=0.28 and abrupt switching caused training instability (loss spikes 
  from 3.9 to 7.7 at the CE→ArcFace transition). Gentler hyperparameters 
  and longer warmup fix this.
- **Frozen backbone for first runs**: Tests whether SSL features alone (with 
  a learned metric head) are sufficient. If frozen+ArcFace works (which I 
  expect), we then test unfrozen as an additional condition.
- **Steps 2000-4000 with frozen backbone**: With backbone frozen, the head 
  converges quickly. 8000 steps is wasteful and may even degrade if the 
  learning rate schedule isn't tuned.
- **`limit_subjects = 0`**: Use all 124 CASIA-B subjects. NEVER use 
  `limit_subjects = 60` for reportable runs. That gives 12 val subjects 
  which inflates accuracy and is non-standard.
- **`gallery_cond=nm, probe_cond=cl`**: Cross-condition only. NEVER use 
  `gallery_cond=nm,bg,cl` — that contaminates the gallery with the probe 
  condition and is not the standard protocol.
- **`knn_view_match=1`** with the fixed `parse_view_from_folder` (returns 
  parts[3] = view angle, not parts[2] = sequence number).

### What to avoid

Configurations that have been verified problematic:

- `seq_len=30, stride=30`: Too short, non-overlapping, outside SSL pretraining 
  range. Recent baseline run with this config gave 28% / 22% / 17% across 
  three seeds — high variance, train_acc stuck at 28%, model not fitting.
- Unfrozen backbone with `lr_backbone >= 1e-5` for many steps without 
  ArcFace: backbone drifts away from useful SSL features.
- `region=ALL` as the first comparison: too many joints, weaker signal than R1.
- `--gallery_cond` that includes the probe condition.
- `--use_arcface 0` for the headline results: CE alone produces unstable 
  embeddings under subject-disjoint evaluation. We learned this from the old 
  thesis runs (~10-15% range with CE).
- Old checkpoints `ssl_best.pt`, `ssl_casia.pt`, `ssl_densegait.pt`: these 
  are from the old architecture and will fail to load strictly (verified). 
  Only use `ssl_casia_v2.pt` and `ssl_densegait_v2.pt`.

## What "good" looks like at each stage

- **Sanity check (limit_subjects=20, R1, seq_len=56, ArcFace)**: Rank-1 in 
  the 50-70% range. Random chance on 4 val subjects = 25%.
- **Baseline full run (R1, frozen, ArcFace, seq_len=56)**: Rank-1 on nm→cl in 
  the 30-45% range. Pose-based methods (PoseGait, GaitGraph) report 22-40% 
  on this protocol.
- **Across-seed variance**: Standard deviation < 3 percentage points. If 
  variance is >5pp, something is unstable.
- **Train accuracy**: Should rise above 50% by step 2000 if learning is 
  working. If train_acc stays under 30% after 4000+ steps, the model is 
  not fitting and the recipe needs to change.

## Experiment matrix (planned)

Three protocols × three SSL initializations × three seeds = 27 runs.

Protocols:
1. nm→nm (matched walking condition)
2. nm→bg (carrying objects)
3. nm→cl (clothing change)

SSL initializations:
- `ssl_casia_v2.pt` (in-domain pretraining)
- `ssl_densegait_v2.pt` (in-the-wild pretraining)
- Random init (no pretraining; baseline for SSL contribution)

For each combination, run three seeds (0, 1, 2) and report mean ± std.

Then ablations (single seed each, on nm→cl only):
- Gallery size: {1, 2, 4, 8} sequences per subject
- Region: {R1, R2, ALL}
- View matching: {0, 1}
- ArcFace: {on, off}
- Backbone: {frozen, unfrozen}

## Important: each run must produce

- `config.json` — full hyperparameter dump
- `metrics.csv` — per-step training metrics (loss, train_acc, val_top1, 
  val_top5 at each eval)
- `best.json` — best val Rank-1 and the step at which it occurred
- `confmat_step*.csv` — confusion matrix at each eval

These are what `analysis/collect_runs.py` aggregates. Without them, the 
results can't be summarized for the thesis.

## On the silent strict=False bug

`load_ssl_backbone` now refuses partial loads by default. If you see 
`STRICT LOAD FAILED` in any run's stdout, STOP — that run is loading a 
random-init backbone and its results are not interpretable. This was the 
single biggest bug class in the original thesis draft.