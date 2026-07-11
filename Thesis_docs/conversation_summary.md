# Strategic Chat Summary

This chat handled the strategic and diagnostic phase of the MSc thesis 
revision (May-July 2026). Key outcomes below.

## Diagnostic phase (~2 weeks)

- Identified and fixed 6 pipeline bugs in the original codebase:
  1. Silent `strict=False` checkpoint loading (22/28 keys dropped)
  2. View parsing returning sequence number instead of angle
  3. DenseGait joint ordering scrambled (custom order, not canonical OpenPose)
  4. DenseGait poses unnormalized (CASIA-B was)
  5. R1/R2 region definitions mislabeled vs. thesis description
  6. SSL objective implemented as cross-region regression instead of masked autoencoding
- Also fixed: ArcFace scale ramp bug in train_id.py line 952

## Architecture redesign

- Redesigned encoder as factorized spatial-temporal transformer (Option B):
  two pairs of (Spatial, Temporal) attention blocks, spatial-first ordering
- Per-joint coordinate embedding + joint-ID embedding + temporal positional embedding
- Variable joint-count handling via joint_indices parameter
- ~813k parameters

## SSL re-pretraining

- ssl_casia.pt: 50 epochs, CASIA-B
- ssl_densegait.pt: 20 epochs, DenseGait (Colab Pro+ compute limit)

## Fine-tuning recipe

Established verified working recipe: R1 region, seq_len=56, ArcFace with 
s=16 m=0.20, warmup 500 steps, scale ramp 1000 steps, gentle backbone LR 
1e-6, gallery_per_subject=4, view_match=1.

## Results (mean across 3 seeds)

| Protocol | DenseGait-SSL | CASIA-SSL | Random init |
|---|---|---|---|
| nm→nm | 93.7 ± 2.3 | **96.0 ± 1.5** | 88.1 |
| nm→bg | 63.9 ± 5.6 | **66.1 ± 5.5** | 52.1 |
| nm→cl | 37.1 ± 5.5 | **38.9 ± 6.7** | 23.5 |

Old thesis draft reported 9.86% on nm→cl. New number is 38.9%, competitive 
with PoseGait (~36%).

## Ablations completed

- Gallery size (g = 1, 2, 4, 8): monotonic to g=4, plateau at g=8
- Region (R1, R2, ALL): ALL best (44.0%), R1 (39.1%), R2 (32.0%)
- View matching: +8.7 points with view matching on
- Backbone freeze: +4.7 points from unfreezing
- ArcFace: +19.8 points over CE-only

## Supervisor communication

- Revision plan (3 pages) sent to Madjid: approved without meeting needed
- Ayoub also aligned (via Madjid's collective response)
- No further supervisor alignment required before writing

## Contribution framing decided

Three contributions (four originally; protocol-audit contribution was 
dropped on honesty grounds):

1. Factorized spatial-temporal transformer architecture
2. In-domain vs. in-the-wild SSL transfer finding
3. Condition-dependent SSL benefit quantification

## Chapter 3 status

Complete. Structure covers architecture (3.2), SSL (3.3), region 
partitioning (3.4), preprocessing (3.5), fine-tuning (3.6), evaluation (3.7).
All technical subtleties correctly incorporated (nm→nm overlap, view 
matching double effect, CE→ArcFace transition dynamics, DenseGait joint 
remap procedure).

## Remaining work

Chapters 1, 2, 4, 5. Draft in the writing chat with the documents from 
this handoff.