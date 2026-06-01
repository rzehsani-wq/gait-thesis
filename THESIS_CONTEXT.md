# Thesis Context

**Title:** Self-Supervised Transformer-Based Gait Recognition Under Clothing 
Variation: A Systematic Evaluation  
**Student:** Reza Ehsani, MSc Computer Science, Bishop's University  
**Status:** Student  
**Target submission:** Before mid-July

## One-line summary

Pose-based factorized spatial-temporal transformer for gait recognition, with 
self-supervised pretraining via masked-joint reconstruction. Evaluated on 
CASIA-B under clothing-variation protocols, with DenseGait used for 
in-the-wild SSL pretraining.

## Where I am right now

[update this section as you progress; example below]

Week 1 of 7-8 week revision plan. Bug-fix and architecture-upgrade phase 
complete. Data layer verified working. Ready to start SSL pretraining on 
Google Colab Pro.

Week 2: Need to set up the GitHub and Colab to start pretraining. Then pretrain the model on CASIA-B and DenseGait.

## The plan

- Week 1: Bug fixes and architecture upgrade. (Done)
- Week 2: SSL pretraining on CASIA-B and DenseGait (Colab Pro).
- Week 3: Sanity check on small subjects, then full 124-subject runs.
- Week 4: Ablations (gallery size, region, view-matching, ArcFace vs CE).
- Week 5: Buffer / additional experiments.
- Week 6: Rewrite Chapter 4 with real numbers; revise Chapters 1, 3, 5.
- Week 7: Final polish, abstract, bibliography.

## Key technical decisions

- **Regions:** R1 = head + arms (11 joints); R2 = legs (6 joints); ALL = 17 
  joints. Matches thesis §3.3.4.
- **Architecture:** Factorized spatial-temporal transformer. 2 pairs of 
  (Spatial, Temporal) attention blocks. Spatial = attention across joints 
  within frame. Temporal = attention across frames within joint, with 
  relative temporal positional bias. d_model=128, nhead=4. ~813k params.
- **Tokenization:** T*J tokens per sample (one per frame-joint pair). Each 
  token = per-joint coord embedding + learned joint-ID embed + learned 
  time-position embed.
- **SSL objective:** Masked-joint reconstruction. 35% mask ratio. MSE loss 
  on masked positions only.
- **DenseGait joint remap:** Recovered empirically from 10k+ frames; saved 
  in `densegait_dataset.py` as `DENSEGAIT_TO_COCO17`. Original .npy files 
  use a custom ordering, not canonical OpenPose COCO-18.
- **Normalization:** Mid-hip centered, median shoulder-width scaled, then 
  per-sequence std-rescaled to ≈1, then clipped to ±8. Applied uniformly to 
  both CASIA-B and DenseGait.
- **Fine-tuning:** ArcFace with s=16, m=0.20, warmup 300 steps. Centroid kNN 
  for retrieval. L2-normalized embeddings.
- **Evaluation protocol:** Subject-disjoint split, 80/20 train/val. 
  Standard CASIA-B condition mapping (gallery from nm-01..04, probe from 
  nm-05/06 / bg-01/02 / cl-01/02). View-matched, with view = parts[3] of 
  folder name (the angle, not the sequence number).

## Bugs that were found and fixed

1. `parse_view_from_folder` returned sequence number (parts[2]) instead of 
   view angle (parts[3]). All view-matched ablations in the original draft 
   were meaningless.
2. `load_ssl_backbone` used `strict=False` silently. Original `ssl_best.pt` 
   was from an older architecture; loading it into the current model 
   silently dropped 22 of 28 keys, leaving the transformer body 
   random-initialized. ALL old supervised runs used a random-init backbone 
   without realizing it.
3. SSL objective was cross-region regression (predict R1 from R2, or vice 
   versa) with zero-padded targets, not masked autoencoding.
4. R1/R2 region definitions in thesis text did not match code: code had 
   R1 = arms+legs, R2 = face+hip-center; thesis claimed upper/lower body.
5. DenseGait poses were not normalized (CASIA-B was). Caused apparent 
   "negative transfer" that wasn't real.
6. DenseGait joint order in the .npy files does not match canonical 
   OpenPose COCO-18. The COCO18_TO_COCO17 map in the original code 
   scrambled joints almost completely.
7. `conf_thr` parameter was parsed but never applied. Low-confidence joint 
   detections corrupted normalization estimates.
8. Thesis claimed "all 124 subjects" but original runs used 
   `--limit_subjects 60`, evaluating on only 12 val subjects.
9. Original "good" runs used `gallery_cond=nm,bg,cl` (gallery contained 
   same condition as probe). The 70-72% numbers were within-condition 
   retrieval, not cross-condition.

## Code state

- **Repo:** [github URL once set up]
- **Branch:** main
- **Tag for Week 1 complete state:** `week1-done`
- **Data location:** Google Drive at `MyDrive/thesis/data/` (CASIA-B CSV + 
  DenseGait poses dir). NOT in git.
- **Cloud GPU:** Google Colab Pro.

## Open questions / decisions not yet made

- Whether to extend architecture ablation in week 7 if time allows.
- How to handle DenseGait pretraining if it takes >12 hours on Colab.
- Whether to include a separate "limitations" subsection in Chapter 3 or 
  fold the protocol-audit story into Chapter 5.

## Supervisor

Madjid and Ayoub, reading current thesis draft. Feedback expected by June 7. 
Approach: don't disclose bug fixes proactively; wait for feedback, then 
fold corrections into the response naturally.