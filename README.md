# Self-Supervised Transformer-Based Gait Recognition Under Clothing Variation

This codebase implements:

- A pose-based factorized spatial-temporal transformer (~813K parameters) for
  gait recognition
- Self-supervised pretraining via masked-joint reconstruction
- Supervised fine-tuning with ArcFace metric learning
- Evaluation on CASIA-B via cross-view centroid retrieval

## Repository layout

    src/
      datasets/     CASIA-B and DenseGait dataset loaders
      models/       Transformer backbone, ID head, ArcFace head
      train/        SSL pretraining and supervised fine-tuning scripts
      utils/        Pose normalization, region partitioning, parsing, logging
      data/         Pose augmentation transforms
      configs/      Example YAML configuration
      smoke_test.py Quick architecture sanity check
    analysis/
      collect_runs.py           Walks runs/ to build a master CSV
      notebooks/                Example analysis notebook
    scripts/
      run_matrix.sh             Runs the main ablation matrix
      bench.py                  Reproduces Table 13 efficiency numbers
    logs/                       Per-seed/per-config result CSVs that back
                                Tables 2, 3, 5, 6, 8, 9, 10, 11, 12 of the paper

## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # POSIX (macOS / Linux)
# .venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

Tested with Python 3.10–3.12 and PyTorch 2.x on NVIDIA GPUs (A100 for the
reported numbers; the code runs on any CUDA GPU with ≥4 GB of memory).

## Reproducing the paper tables

The per-seed, per-configuration result logs that back the paper's numerical
tables are shipped in `logs/`. Each CSV corresponds to one paper table; see
`logs/README.md` for the schema. From a fresh run, the logs are regenerated
from `runs/` with `analysis/collect_runs.py`.

### Table 2 (main results, R1 region, standard protocol)

```bash
# SSL pretrain on CASIA-B training subjects (74 subjects)
python -m src.train.train_ssl --dataset casia \
    --csv data/casia-b_pose_coco.csv \
    --seq_len 56 --stride 8 --epochs 50 --batch_size 64 \
    --lr 3e-4 --save ckpts/ssl_casia.pt

# Fine-tune under the three probe conditions; repeat for seeds 0..6
for seed in 0 1 2 3 4 5 6; do
  for probe in nm bg cl; do
    python -m src.train.train_id \
      --csv data/casia-b_pose_coco.csv \
      --ckpt ckpts/ssl_casia.pt \
      --region R1 --seq_len 56 \
      --gallery_cond nm --probe_cond "$probe" \
      --gallery_per_subject 4 --knn_k 1 --knn_view_match 0 \
      --use_arcface 1 --arc_s 16 --arc_m 0.20 \
      --steps 6000 --batch_size 64 \
      --lr_head 1e-3 --lr_backbone 1e-4 \
      --seed "$seed" \
      --exp_name "table2_casia_r1_${probe}_seed${seed}"
  done
done
```

Repeat with `--ckpt ckpts/ssl_densegait.pt` and `--ckpt ""` (no SSL) for the
DenseGait-SSL and Random-init columns. The main-table hyperparameters are
documented in Table 1 of the paper.

### Table 10 (2×2 region-by-initialization, standard protocol)

Same as Table 2 but add `--region R2` and vary `--ckpt` across
{CASIA-SSL, Random}.

### Table 13 (computational efficiency)

```bash
python scripts/bench.py --batch_size 64 --seq_len 56 --joints 17 --precision fp16
```

Prints parameters, FLOPs per sample, peak GPU memory, latency, and throughput.
Absolute numbers match the paper within rounding on an NVIDIA A100 at FP16.

## Data

The CASIA-B dataset is distributed by the Institute of Automation, Chinese
Academy of Sciences, upon request. The HRNet pose extractions used in this
work are available in the [GaitGraph repository](https://github.com/tteepe/GaitGraph).
The DenseGait dataset is publicly available as described in
[Cosma and Rădoi (2022)](https://doi.org/10.3390/s22186803). Preprocessed
CSVs that index these datasets for this codebase are not committed; see
`preprocessing_validation.md` for details on the preprocessing pipeline.