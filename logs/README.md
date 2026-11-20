# Result logs

This directory contains the per-seed, per-configuration results that back the
numerical tables in the paper. One CSV per table. Each row is a single
fine-tuning or SSL-pretraining run; columns include the configuration knobs
plus the final Rank-1 (and Rank-5 where applicable) accuracy.

## Files

| File                          | Backs paper table | What it contains |
|-------------------------------|-------------------|-------------------|
| `table02_main_results.csv`    | Table 2           | 7 seeds × 3 inits × 3 probes (R1 region) = 63 rows |
| `table03_ssl_contribution.csv`| Table 3           | Paired (SSL − Random) per-seed differences used in the paired t-tests |
| `table05_domain_comparison.csv`| Table 5          | Paired (CASIA-SSL − DenseGait-SSL) per-seed differences |
| `table06_lr_sweep.csv`        | Table 6           | 3 LRs × 3 probes, CASIA-SSL, seed 0 |
| `table08_graph_prior.csv`     | Table 8           | 7 seeds × 3 probes, graph-prior variant and main |
| `table09_spatial_only.csv`    | Table 9           | 7 seeds × 3 probes, spatial-only variant and main |
| `table10_region_standard.csv` | Table 10          | 7 seeds × 3 probes × {R1, R2} × {CASIA-SSL, Random} |
| `table11_data_scale.csv`      | Table 11          | 7 seeds × 3 probes × {CASIA-SSL, DenseGait-full, DenseGait-subsampled-8140} |
| `table12_data_scale_tests.csv`| Table 12          | Paired-t inputs for the scale-vs-domain tests |

## Column conventions

Each run row carries, at minimum:

- `run_id`             — unique string identifier
- `seed`               — random seed (fine-tuning initialization)
- `init`               — one of {CASIA-SSL, DenseGait-SSL, DenseGait-SSL-sub8140, Random, Graph-prior, Spatial-only}
- `region`             — one of {R1, R2, ALL}
- `probe_cond`         — one of {nm, bg, cl}
- `backbone_lr`        — backbone learning rate
- `rank1`              — final Rank-1 accuracy (%)
- `rank5`              — final Rank-5 accuracy (%) where available

Paired-test CSVs (03, 05, 12) additionally include the paired-difference,
t-statistic, df, and p-value per row.

## Regenerating

These CSVs are produced from the per-run JSON summaries under `runs/` (not
tracked in the repository — large) by:

    python -m analysis.collect_runs --runs_dir runs --out analysis/runs_master.csv
    python scripts/export_table_logs.py --master analysis/runs_master.csv --out logs/

The export script is a thin selector that filters `runs_master.csv` into the
per-table CSVs listed above.
