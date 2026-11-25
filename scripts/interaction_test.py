"""Region x initialization interaction test for Table 10.

For each probe condition and each seed s, the interaction contrast is

    I_s = (R2_s - R1_s) under CASIA-SSL  -  (R2_s - R1_s) under random init

A one-sample t-test on the seven I_s values (df = 6) tests whether the
R2-R1 difference depends on initialization. Seeds are paired across the
two initializations by seed index only, so a Welch two-sample t-test on
the two sets of within-initialization differences is also reported; it
does not depend on that pairing.

Usage:
    python scripts/interaction_test.py logs/table10_region_standard.csv
"""
import sys

import pandas as pd
from scipy import stats


def main(path):
    df = pd.read_csv(path)
    df = df[df["protocol"] == "standard"]
    table = df.pivot_table(index=["probe_cond", "seed"],
                           columns=["init", "region"], values="top1")

    print(f"{'cond':<5}{'SSL R2-R1':>11}{'Rand R2-R1':>12}{'I mean':>9}"
          f"{'I sd':>7}{'t(6)':>8}{'p':>10}{'Welch p':>10}  I>0")
    for cond in ["nm", "bg", "cl"]:
        x = table.loc[cond]
        d_ssl = x[("casia", "R2")] - x[("casia", "R1")]
        d_rand = x[("random", "R2")] - x[("random", "R1")]
        contrast = d_ssl - d_rand
        one_sample = stats.ttest_1samp(contrast, 0.0)
        welch = stats.ttest_ind(d_ssl, d_rand, equal_var=False)
        print(f"nm->{cond:<2}{d_ssl.mean():>+10.2f}{d_rand.mean():>+12.2f}"
              f"{contrast.mean():>+9.2f}{contrast.std(ddof=1):>7.2f}"
              f"{one_sample.statistic:>8.2f}{one_sample.pvalue:>10.2g}"
              f"{welch.pvalue:>10.2g}  {(contrast > 0).sum()}/{len(contrast)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/table10_region_standard.csv")
