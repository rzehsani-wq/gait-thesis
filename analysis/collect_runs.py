"""
analysis/collect_runs.py

Walk the runs/ directory and produce a master CSV summarizing every run.
One row per run. Columns are a flattened mix of config + final metrics.

Usage:
    python -m analysis.collect_runs --runs_dir runs --out analysis/runs_master.csv
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] could not parse {path}: {e}")
        return None


def _read_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] could not parse {path}: {e}")
        return None


def _extract_final_metrics(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Pull the most-useful summary numbers out of a per-step metrics CSV."""
    out: Dict[str, Any] = {}
    if metrics_df is None or metrics_df.empty:
        return out

    out["n_steps_logged"] = len(metrics_df)

    # If there's a val_top1 column, get the best and last values
    for col in ("val_top1", "val_top5", "val_loss"):
        if col in metrics_df.columns:
            out[f"{col}_best"] = float(metrics_df[col].max() if "top" in col 
                                       else metrics_df[col].min())
            out[f"{col}_last"] = float(metrics_df[col].iloc[-1])

    # Training trajectory: final train loss and accuracy
    for col in ("train_loss", "train_acc", "loss", "acc"):
        if col in metrics_df.columns:
            out[f"{col}_last"] = float(metrics_df[col].iloc[-1])
    return out


def _flatten_config(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dicts: {a: {b: 1}} -> {a.b: 1}"""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_config(v, prefix=key))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def collect_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Collect one run's results into a single flat dict."""
    cfg = _read_json(run_dir / "config.json") or {}
    best = _read_json(run_dir / "best.json") or {}
    metrics = _read_metrics_csv(run_dir / "metrics.csv")

    row: Dict[str, Any] = {"run_name": run_dir.name}
    row.update(_flatten_config(cfg))
    row.update({f"best.{k}": v for k, v in best.items()})
    row.update(_extract_final_metrics(metrics))

    # Best confusion matrix file (if any) — just record the filename
    confmats = sorted(run_dir.glob("confmat_step*.csv"))
    if confmats:
        row["last_confmat_file"] = str(confmats[-1])

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs", help="Directory containing run subfolders")
    ap.add_argument("--out", default="analysis/runs_master.csv", help="Output CSV path")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"runs directory not found: {runs_dir}")

    rows = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        row = collect_run(sub)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[warn] no runs found")
        return

    df = pd.DataFrame(rows)
    # Sort columns: run_name first, then config, then results
    cols = ["run_name"]
    config_cols = sorted([c for c in df.columns 
                          if c not in cols and not c.startswith("best.") 
                          and not c.endswith(("_best", "_last"))
                          and c != "n_steps_logged" 
                          and c != "last_confmat_file"])
    result_cols = sorted([c for c in df.columns if c not in cols + config_cols])
    df = df[cols + config_cols + result_cols]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[ok] wrote {len(df)} rows to {args.out}")
    print(f"     {len(config_cols)} config columns, {len(result_cols)} result columns")


if __name__ == "__main__":
    main()