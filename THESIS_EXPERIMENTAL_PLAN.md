# Experimental Plan and Conventions

This document describes the specific architectural and experimental choices 
for this MSc thesis on pose-based SSL transformer gait recognition. 
Hyperparameters were chosen to match the SSL pretraining setup, NOT taken 
from generic transformer recipes. Departures from these defaults require an 
explicit justification.

## Status

- Week 1 (bug fixes, architecture upgrade): complete.
- Week 2 (SSL pretraining): complete.
  - `ssl_casia.pt`: 50 epochs on CASIA-B, masked-joint reconstruction.
  - `ssl_densegait.pt`: 20 epochs on DenseGait, final reconstruction loss 0.0027.
  - Both use the new factorized spatial-temporal architecture (813,066 params).
- Week 3 (CASIA-B fine-tuning experiments): in progress.
  - First proper run (seed 0, ssl_densegait, R1, ArcFace, unfrozen backbone): 
    **best Rank-1 = 36.67% on nm→cl, all 124 subjects, standard protocol**.
  - This is 3.7× higher than the original thesis draft's 9.86% on the same 
    nominal protocol — confirming that the low original numbers were 
    artifacts of bugs and not fundamental limits of the approach.

## What we are evaluating

The thesis is a controlled empirical study of pose-based self-supervised 
transformer models for gait recognition under clothing variation on CASIA-B.

Specifically:
1. What is the achievable Rank-1 on CASIA-B nm→cl / nm→bg / nm→nm using a 
   pose-only SSL-pretrained transformer with the standard protocol?
2. Does SSL pretraining on in-the-wild data (DenseGait) transfer better, 
   worse, or equally to in-domain pretraining (CASIA-B)?
3. Does SSL pretraining help at all vs. random init?
4. How sensitive are results to evaluation choices (gallery size, view 
   matching, region, ArcFace vs CE)?

NOT trying to beat state of the art. Goal: defensible reproducible numbers 
under correct protocols.

## Architecture (fixed, do not change)

- Factorized spatial-temporal transformer (Option B).
- Two pairs of (Spatial, Temporal) attention blocks, spatial-first.
- Spatial blocks: attention across joints within a frame.
- Temporal blocks: attention across frames within a joint, with relative 
  temporal positional bias.
- d_model = 128, nhead = 4, depth_pairs = 2.
- Per-joint coordinate embedding + learned joint-ID embedding + learned 
  time positional embedding.
- ~813k parameters (846k with the supervised classification/metric head).

## Pretraining (already complete; do not redo unless explicitly agreed)

Both SSL pretrainings used masked-joint reconstruction (mask 35% of 
(frame, joint) positions; MSE loss on masked positions only) with:

- `seq_len = 56`
- `stride = 8` for CASIA, `stride = 4` for DenseGait
- batch_size = 64

The model's learned positional embeddings and temporal RPE buckets are 
calibrated for T=56 contexts. Fine-tuning with substantially different 
seq_len means operating outside the trained range. **KEEP seq_len = 56 
during fine-tuning.**

Checkpoints (in `/content/drive/MyDrive/thesis/checkpoints/ssl/`):
- `ssl_casia.pt`: 50 epochs on CASIA-B
- `ssl_densegait.pt`: 20 epochs on DenseGait (loss 0.0027)

Both load cleanly into the new architecture (`58/58 keys strict OK`).

## VERIFIED WORKING fine-tuning recipe

This is the recipe that produced 36.67% on nm→cl with seed 0. Use it as the 
baseline for all main experiments. Vary one parameter at a time for ablations.

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

Where `<source>` is `casia` or `densegait`, `<cond>` is `nm` or `bg` or `cl`, 
and `<seed>` is 0, 1, or 2.

**Steps: 6000 (revised from earlier 4000).** At 4000 steps val_top1 was still 
rising, with no clear plateau. 6000 gives enough headroom while staying 
clear of overfitting on the ~57k training windows / 99 train subjects.

### Why these choices

- **R1 (head + arms, 11 joints)**: Strongest region in preliminary results. 
  Also matches the thesis hypothesis that upper-body cues survive clothing 
  variation better than lower-body. ALL with 17 joints has more parameters 
  competing in input space and weaker signal.
- **`freeze_backbone=0` with `lr_backbone=1e-6`**: Frozen backbone with 
  ArcFace alone wasn't enough — val plateaued around 23%. Unfrozen at very 
  low LR lets the backbone refine SSL features for the identification task 
  while not destroying them. This combination reached 36.67%.
- **ArcFace s=16, m=0.20**: Earlier ArcFace runs with s=12, m=0.28 caused 
  severe training instability. These gentler hyperparameters work.
- **`warmup_steps=500, arc_s_warm=1000, arc_s_min=4`**: Smoother transition 
  from CE to ArcFace. Note: a ramp bug in train_id.py (line 952) currently 
  causes ArcFace scale to jump instead of ramp at warmup completion. The 
  spike is recoverable but suboptimal. Fix this when convenient (one-line 
  change), then re-run for cleaner numbers.
- **`gallery_per_subject=4`**: Standard practice. Matches the gallery size 
  used in nm-01..04 of the CASIA-B protocol.
- **`limit_subjects=0`**: ALL 124 subjects. Never use limit_subjects=60 for 
  reportable runs — that's a non-standard 12-subject evaluation.
- **`gallery_cond=nm, probe_cond=cl`**: Cross-condition only. NEVER 
  `gallery_cond=nm,bg,cl` — contaminates gallery with probe condition.
- **`pose_aug=0, balance_cond=0`**: Off for headline runs. Old thesis §5.1 
  noted these didn't reproducibly help; we confirm by disabling for clean 
  comparison.

## Experiment matrix (planned and in progress)

### Phase 1: Main protocol numbers (Table 4.1 of thesis)

3 protocols × 2 pretraining sources × 3 seeds = **18 runs**.

| protocol | pretraining | seeds | exp_name pattern |
|---|---|---|---|
| nm→nm | ssl_densegait.pt | 0, 1, 2 | `densegait_R1_nm2nm_s{seed}` |
| nm→bg | ssl_densegait.pt | 0, 1, 2 | `densegait_R1_nm2bg_s{seed}` |
| nm→cl | ssl_densegait.pt | 0, 1, 2 | `densegait_R1_nm2cl_s{seed}` |
| nm→nm | ssl_casia.pt | 0, 1, 2 | `casia_R1_nm2nm_s{seed}` |
| nm→bg | ssl_casia.pt | 0, 1, 2 | `casia_R1_nm2bg_s{seed}` |
| nm→cl | ssl_casia.pt | 0, 1, 2 | `casia_R1_nm2cl_s{seed}` |

Report mean ± std across seeds in Table 4.1.

### Phase 2: SSL contribution baseline (one row of Table 4.6)

3 protocols × random init × 1 seed = **3 runs**.

| protocol | pretraining | seed | exp_name |
|---|---|---|---|
| nm→nm | (none, `--ckpt ""`) | 0 | `scratch_R1_nm2nm_s0` |
| nm→bg | (none) | 0 | `scratch_R1_nm2bg_s0` |
| nm→cl | (none) | 0 | `scratch_R1_nm2cl_s0` |

If random-init beats SSL-pretrained on any protocol, that's an interesting 
negative finding and worth additional seeds to confirm.

### Phase 3: Ablations (single seed each, on nm→cl only)

Each ablation varies ONE parameter from the baseline recipe.

| ablation | parameter | values | runs |
|---|---|---|---|
| Gallery size | `--gallery_per_subject` | 1, 2, 4, 8 | 4 |
| Region | `--region` | R1, R2, ALL | 3 (R1 already done) |
| View matching | `--knn_view_match` | 0, 1 | 2 (1 already done) |
| ArcFace vs CE | `--use_arcface` | 0, 1 | 2 (1 already done) |
| Backbone | `--freeze_backbone` | 0, 1 | 2 (0 already done) |

**Total ablation runs: ~10 new + already done.**

### Phase 4 (optional): Confirmation reruns after fixing the ramp bug

After fixing the ArcFace ramp (line 952 of train_id.py), re-run seed 0 of 
the main protocols to see if cleaner loss curves yield 1-3 additional points. 
Worth doing only if there's time after Phases 1-3.

## Conventions for run logging

Every run must produce in its `runs/<exp_name>/` directory:
- `config.json` — full hyperparameter dump
- `metrics.csv` — per-step training metrics (loss, train_acc, val_top1, 
  val_top5 at each eval)
- `best.json` — best val Rank-1 and the step at which it occurred
- `confmat_step*.csv` — confusion matrix at final eval

These feed into `analysis/collect_runs.py` for aggregation into a master 
results table.

## Naming convention

Pattern: `<pretraining>_<region>_<protocol>_<seed>` with optional suffixes 
for ablations.

Examples:
- `densegait_R1_nm2cl_s0` — main run
- `casia_R1_nm2nm_s2` — main run
- `densegait_R1_nm2cl_g8_s0` — gallery size ablation, g=8
- `densegait_R1_nm2cl_R2_s0` — region ablation, R2
- `scratch_R1_nm2cl_s0` — random-init baseline

## Things to NEVER do

- Use `ckpts/ssl_best.pt` or any pre-Week-2 checkpoint (wrong architecture, 
  will fail strict load).
- Use `--limit_subjects 60` for reportable runs (non-standard 12-subject eval).
- Use `--gallery_cond` that includes the probe condition (e.g., 
  `nm,bg,cl` while probing `cl`).
- Use `--seq_len 30 --stride 30` (outside SSL pretraining context window).
- Use `--use_arcface 0` for headline numbers (CE-only is too weak under 
  subject-disjoint eval; verified by old thesis runs landing at 10-15%).
- Use `--pose_aug 1` or `--balance_cond 1` for headline runs (only for 
  ablations testing whether they help, with that hypothesis stated).

## On the strict-load check

`load_ssl_backbone` now refuses partial loads. Every run should print:
   `[ckpt] loaded <path>: 58/58 keys (strict OK)`
If you see anything else (especially "STRICT LOAD FAILED" or "WARNING: 
proceeding with partial load"), STOP — that run is loading a random-init 
backbone for the unmatched keys and its results are not interpretable.

## Known issues

1. **ArcFace ramp bug** (train_id.py line 952): `steps_after_warm` is 
   computed as `args.steps - warmup_left` instead of `step - args.warmup_steps`. 
   Result: ArcFace scale jumps to full value immediately after warmup ends 
   instead of ramping linearly. Causes a loss spike from ~4.6 to ~7.8. 
   Models recover but lose some efficiency. One-line fix; apply when 
   convenient. ---> It's already applied (replaced with `steps_after_warm = step - int(args.warmup_steps) if use_warmup else step`), so the code is updated but still it spikes!

2. **`acc=0.000` during ArcFace phase**: Normal. The angular margin makes 
   correct classification harder by design. Watch `val_top1` instead.