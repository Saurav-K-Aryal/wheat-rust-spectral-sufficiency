"""The Health versus Rust decision, from the saved out-of-fold probabilities (no model is refitted).

Three-class macro-F1 mixes the disease decision between Health and Rust with the easier split between vegetation and
the Other class. For every method and outer fold this script computes
  hr_auc              ROC AUC of the Rust score p_Rust / (p_Health + p_Rust) on the true Health and Rust patches
  hr_auc_veg          the same on vegetated patches only
  f1_Health, f1_Rust  F1 of the three-class argmax, as in the fold metrics
Health patches include empty mosaic patches and sparse canopy that no Rust patch shows, so part of the all-patch AUC
reflects canopy cover. The vegetated subset keeps the patches that are not empty and have a patch-mean NDVI of at
least 0.5 (results/audit/composition_ndvi.csv), which removes most of that effect.

Writes to results/analysis/
  hr_folds.csv        the per-fold values above
  hr_metrics.csv      per method, the all-patch AUC and the F1 values, mean and standard deviation over folds
  hr_veg_metrics.csv  per method, the vegetated AUC, with the numbers of vegetated Health and Rust patches
  health_errors.csv   per method, the shares of Health patches predicted Health, Rust, and Other, pooled over folds
usage: python scripts/analysis/03_health_rust.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.analysis._shared import RESULTS, analysis_file, read_experiment

# every experiment whose methods appear in the Health versus Rust tables or contrasts
OOF_EXPERIMENTS = ["E1", "E2", "E4", "E9", "E9t", "E4t", "E3t"]
PROBA = ["p_Health", "p_Rust", "p_Other"]


def vegetated():
    comp = pd.read_csv(os.path.join(RESULTS, "audit", "composition_ndvi.csv"))
    return comp[(~comp["empty"].astype(bool)) & (comp.ndvi_mean >= 0.5)]


def per_fold(O):
    rows = []
    for (key, r, k), g in O.groupby(["key", "repeat", "fold"]):
        P = g[PROBA].values
        y = g.y.values
        m = (y == 0) | (y == 1)
        s = P[m, 1] / (P[m, 0] + P[m, 1] + 1e-12)
        auc = roc_auc_score((y[m] == 1).astype(int), s) if len(np.unique(y[m])) == 2 else np.nan
        pred = P.argmax(1)
        f1 = {}
        for c, name in [(0, "Health"), (1, "Rust")]:
            tp = ((pred == c) & (y == c)).sum()
            fp = ((pred == c) & (y != c)).sum()
            fn = ((pred != c) & (y == c)).sum()
            f1[name] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        rows.append(dict(key=key, repeat=r, fold=k, hr_auc=auc, f1_Health=f1["Health"], f1_Rust=f1["Rust"]))
    return pd.DataFrame(rows)


def per_fold_subset(O, keep_ids):
    """The Health versus Rust AUC on a subset of patches; folds left with one class are skipped."""
    rows = []
    for (key, r, k), g in O.groupby(["key", "repeat", "fold"]):
        g = g[g.sample_id.isin(keep_ids)]
        m = g.y.isin([0, 1]).values
        y = g.y.values[m]
        if len(np.unique(y)) < 2:
            continue
        P = g[["p_Health", "p_Rust"]].values[m]
        rows.append(dict(key=key, repeat=r, fold=k, hr_auc=roc_auc_score((y == 1).astype(int),
                                                                          P[:, 1] / (P.sum(1) + 1e-12))))
    return pd.DataFrame(rows)


def main():
    O = pd.concat([read_experiment(e, "oof") for e in OOF_EXPERIMENTS], ignore_index=True)
    veg = vegetated()
    counts = veg.label.value_counts()

    F = per_fold(O)
    Fveg = per_fold_subset(O, set(veg.sample_id))
    F.merge(Fveg.rename(columns={"hr_auc": "hr_auc_veg"}), on=["key", "repeat", "fold"], how="left").to_csv(
        analysis_file("hr_folds.csv"), index=False)

    F.groupby("key").agg(n_units=("hr_auc", "size"), hr_auc_mean=("hr_auc", "mean"), hr_auc_sd=("hr_auc", "std"),
                         f1_Health_mean=("f1_Health", "mean"), f1_Rust_mean=("f1_Rust", "mean")).reset_index().to_csv(
        analysis_file("hr_metrics.csv"), index=False)

    s = Fveg.groupby("key").hr_auc.agg(["mean", "std", "size"]).rename(
        columns={"mean": "hr_auc_mean", "std": "hr_auc_sd", "size": "n_units"}).reset_index()
    s["n_veg_health"] = int(counts.get("Health", 0))
    s["n_veg_rust"] = int(counts.get("Rust", 0))
    s.to_csv(analysis_file("hr_veg_metrics.csv"), index=False)

    rows = []
    for key, g in O.groupby("key"):
        h = g[g.y == 0]
        pred = h[PROBA].values.argmax(1)
        rows.append(dict(key=key, n_health=len(h), recall_health=float((pred == 0).mean()),
                         health_as_rust=float((pred == 1).mean()), health_as_other=float((pred == 2).mean())))
    pd.DataFrame(rows).to_csv(analysis_file("health_errors.csv"), index=False)
    print(f"Health versus Rust metrics for {F.key.nunique()} methods; vegetated patches "
          + ", ".join(f"{c} {int(n)}" for c, n in counts.items()))


if __name__ == "__main__":
    main()
