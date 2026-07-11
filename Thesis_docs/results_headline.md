# Headline Results

## Main Protocol Table (Table 4.1)

Rank-1 accuracy on CASIA-B, all 124 subjects, subject-disjoint 80/20 split, 
mean ± std across 3 seeds. R1 region, ArcFace, gallery=4, view-matched.

| Protocol | DenseGait-SSL | CASIA-SSL | Random init |
|---|---|---|---|
| nm→nm | 93.7 ± 2.3 | **96.0 ± 1.5** | 88.1 (seed 0) |
| nm→bg | 63.9 ± 5.6 | **66.1 ± 5.5** | 52.1 (seed 0) |
| nm→cl | 37.1 ± 5.5 | **38.9 ± 6.7** | 23.5 ± 5.0 |

Random init nm→cl: 27.8 / 24.7 / 17.9 across 3 seeds.

## SSL Contribution (Table 4.6)

| Protocol | Random init | Best SSL | Δ (CASIA-SSL) |
|---|---|---|---|
| nm→nm | 88.1 | 96.0 | +7.9 |
| nm→bg | 52.1 | 66.1 | +14.0 |
| nm→cl | 23.5 | 38.9 | +15.4 |

Note: SSL benefit grows with condition difficulty.

## Gallery Size Ablation (nm→cl, CASIA-SSL, 3 seeds)

| g | Mean | Std |
|---|---|---|
| 1 | 33.54 | 4.45 |
| 2 | 36.83 | 5.46 |
| 4 (default) | 39.12 | 6.63 |
| 8 | 39.15 | 6.09 |

## Region Ablation (nm→cl, CASIA-SSL, 3 seeds)

| Region | Mean | Std |
|---|---|---|
| R1 (11 joints) | 39.12 | 6.63 |
| R2 (6 joints) | 31.97 | 3.19 |
| ALL (17 joints) | 43.95 | 6.40 |

## View Matching Ablation (nm→cl, CASIA-SSL, 3 seeds)

| view_match | Mean | Std |
|---|---|---|
| 0 (off) | 30.43 | 10.94 |
| 1 (on, default) | 39.12 | 6.63 |

## Backbone Freeze Ablation (nm→cl, CASIA-SSL, 3 seeds)

| freeze | Mean | Std |
|---|---|---|
| 0 (unfrozen, default) | 39.12 | 6.63 |
| 1 (frozen) | 34.44 | 7.21 |

## ArcFace Ablation (nm→cl, CASIA-SSL, 3 seeds)

| use_arcface | Mean | Std |
|---|---|---|
| 0 (CE only) | 19.25 | 0.36 |
| 1 (ArcFace, default) | 39.12 | 6.63 |

Note: CE-only converges to same mediocre solution every seed (very low std). 
ArcFace introduces real seed variance because it actually fits identity 
discrimination.