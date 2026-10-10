"""Whether the vegetated-patch results depend on the NDVI threshold that defines 'vegetated'.

The paper scores the Health versus Rust decision on vegetated patches as well as on all of them, because no Rust patch
has a patch-mean NDVI below 0.5 while about a fifth of the Health patches do, so part of the all-patch AUC measures
canopy cover rather than disease (scripts/analysis/03_health_rust.py). The threshold of 0.5 was fixed from the audit,
not from results, but it is one choice among several. This script repeats the vegetated AUC contrasts of the paper
(the hr_vegetated family of scripts/analysis/04_contrasts.py) with thresholds from 0.3 to 0.7, using the same saved
out-of-fold probabilities, the same paired test, and Holm's adjustment within each threshold. A higher threshold
keeps fewer and denser Health patches, so the comparison also shows how much of each advantage survives as the
canopy-cover contrast between the classes shrinks. At 0.5 the script reproduces the paper's values and stops if they
differ.

A threshold removes only Health patches, because no Rust patch falls below 0.5, so the Health patches that remain are
greener on average than the Rust patches, and greenness alone still separates the two classes a little. The script
therefore records that imbalance (the AUC of patch-mean NDVI itself, lower NDVI counting as Rust) at every threshold,
and at the paper's threshold it repeats the contrasts with an NDVI-balanced AUC. Patches are grouped into NDVI
quintiles of the vegetated Health and Rust patches, and each patch is weighted by the share of its quintile among
all patches divided by its share among the patches of its class, so that both classes have the same NDVI
distribution at the level of quintiles. The weights come from the audit's NDVI, never from a model.

Writes to results/analysis/
  ndvi_threshold.csv      one row per threshold and contrast, with the numbers of vegetated Health and Rust patches,
                          the 95 percent credible interval, and the verbal label of the posteriors
  ndvi_threshold_auc.csv  the vegetated AUC of every method at every threshold, mean and standard deviation over the
                          outer folds that hold both classes
  ndvi_imbalance.csv      per threshold, the median patch-mean NDVI of the vegetated Health and Rust patches and the AUC
                          of NDVI alone, without and with the balancing weights
  ndvi_balance.csv        the vegetated contrasts at the paper's threshold with the NDVI-balanced AUC
usage: python scripts/analysis/10_ndvi_threshold.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.analysis._shared import RESULTS, analysis_file, load_script, read_experiment
from wheatrust.stats import correlated_posterior, verdict

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
PAPER_THRESHOLD = 0.5
N_BINS = 5  # NDVI quintiles for the balancing weights

hr_mod = load_script("03_health_rust")
con = load_script("04_contrasts")


def describe(name):
    """The description 04_contrasts.py gives the same contrast."""
    return con.HR_DESC[name.removeprefix("t_")] + (", TabPFN" if name.startswith("t_") else ", FLAML")


def vegetated(comp, threshold):
    return comp[(~comp["empty"].astype(bool)) & (comp.ndvi_mean >= threshold)]


def balance_weights(veg):
    """Weight per vegetated Health or Rust patch that gives both classes the pooled NDVI distribution over quintiles."""
    hr = veg[veg.label.isin(["Health", "Rust"])].copy()
    edges = np.quantile(hr.ndvi_mean, np.linspace(0, 1, N_BINS + 1))
    hr["bin"] = np.clip(np.searchsorted(edges, hr.ndvi_mean, side="right") - 1, 0, N_BINS - 1)
    pooled = hr.bin.value_counts(normalize=True)
    within = hr.groupby("label").bin.value_counts(normalize=True)
    hr["weight"] = [pooled[b] / within[(c, b)] for c, b in zip(hr.label, hr.bin)]
    return hr.set_index("sample_id").weight


def ndvi_auc(veg, weights=None):
    """AUC of patch-mean NDVI for Rust against Health, lower NDVI counting as Rust."""
    hr = veg[veg.label.isin(["Health", "Rust"])]
    w = None if weights is None else weights.loc[hr.sample_id].to_numpy()
    return roc_auc_score((hr.label == "Rust").astype(int), -hr.ndvi_mean, sample_weight=w)


def per_fold_weighted(O, weights):
    """The Health versus Rust AUC of every method and outer fold on the weighted vegetated patches."""
    rows = []
    for (key, r, k), g in O.groupby(["key", "repeat", "fold"]):
        g = g[g.sample_id.isin(weights.index) & g.y.isin([0, 1])]
        y = g.y.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        P = g[["p_Health", "p_Rust"]].to_numpy()
        rows.append(dict(key=key, repeat=r, fold=k,
                         hr_auc=roc_auc_score((y == 1).astype(int), P[:, 1] / (P.sum(1) + 1e-12),
                                              sample_weight=weights.loc[g.sample_id].to_numpy())))
    return pd.DataFrame(rows)


def main():
    keys = sorted({k for _, a, b in con.HR for k in (a, b)})
    O = pd.concat([read_experiment(e, "oof") for e in hr_mod.OOF_EXPERIMENTS], ignore_index=True)
    O = O[O.key.isin(keys)]
    comp = pd.read_csv(os.path.join(RESULTS, "audit", "composition_ndvi.csv"))
    rows, aucs = [], []
    for t in THRESHOLDS:
        veg = vegetated(comp, t)
        counts = veg.label.value_counts()
        F = hr_mod.per_fold_subset(O, set(veg.sample_id))
        specs = [(name, a, b, describe(name),
                  *con.aligned([con.fold_series(F, a, "hr_auc"), con.fold_series(F, b, "hr_auc")]))
                 for name, a, b in con.HR]
        for r in con.family_rows("hr_vegetated", specs, "hr_auc", "vegetated"):
            post = correlated_posterior(r["diff"], r["diff_sd"], r["n_paired"])
            r.update(threshold=t, n_health=int(counts.get("Health", 0)), n_rust=int(counts.get("Rust", 0)),
                     ci95_lo=post.ppf(0.025), ci95_hi=post.ppf(0.975),
                     verdict=verdict(r["p_better"], r["p_equiv"], r["p_worse"]))
            rows.append(r)
        s = F.groupby("key").hr_auc.agg(["mean", "std", "size"]).reset_index()
        s.columns = ["key", "hr_auc_mean", "hr_auc_sd", "n_units"]
        s.insert(0, "threshold", t)
        s["n_health"], s["n_rust"] = int(counts.get("Health", 0)), int(counts.get("Rust", 0))
        aucs.append(s)
    out = pd.DataFrame(rows)
    out = out[["threshold", "n_health", "n_rust"] + [c for c in con.COLS if c in out]
              + ["ci95_lo", "ci95_hi", "verdict"]]

    # the paper's threshold must give the paper's numbers
    paper = pd.read_csv(analysis_file("contrasts.csv"), float_precision="round_trip")
    paper = paper[paper.family == "hr_vegetated"].set_index("name")
    here = out[out.threshold == PAPER_THRESHOLD].set_index("name")
    cols = ["n_paired", "diff", "diff_sd", "p_better", "p_equiv", "p_worse", "p_corrected", "p_holm"]
    worst = max(np.abs(here.loc[paper.index, c].astype(float) - paper[c].astype(float)).max() for c in cols)
    if worst > 1e-9:
        sys.exit(f"at NDVI {PAPER_THRESHOLD} the contrasts differ from contrasts.csv by {worst:.2e}; stopping")

    out.to_csv(analysis_file("ndvi_threshold.csv"), index=False)
    pd.concat(aucs, ignore_index=True).to_csv(analysis_file("ndvi_threshold_auc.csv"), index=False)

    # how unequal the canopy of the two classes stays after each threshold, and the balanced contrasts at 0.5
    imb = []
    for t in THRESHOLDS:
        veg = vegetated(comp, t)
        w = balance_weights(veg)
        med = veg.groupby("label").ndvi_mean.median()
        imb.append(dict(threshold=t, ndvi_median_health=med["Health"], ndvi_median_rust=med["Rust"],
                        ndvi_auc=ndvi_auc(veg), ndvi_auc_balanced=ndvi_auc(veg, w),
                        weight_min=w.min(), weight_max=w.max()))
    pd.DataFrame(imb).to_csv(analysis_file("ndvi_imbalance.csv"), index=False)
    veg = vegetated(comp, PAPER_THRESHOLD)
    Fw = per_fold_weighted(O, balance_weights(veg))
    specs = [(name, a, b, describe(name),
              *con.aligned([con.fold_series(Fw, a, "hr_auc"), con.fold_series(Fw, b, "hr_auc")]))
             for name, a, b in con.HR]
    bal = []
    for r in con.family_rows("hr_vegetated_balanced", specs, "hr_auc", "vegetated, NDVI-balanced"):
        post = correlated_posterior(r["diff"], r["diff_sd"], r["n_paired"])
        r.update(threshold=PAPER_THRESHOLD, ci95_lo=post.ppf(0.025), ci95_hi=post.ppf(0.975),
                 verdict=verdict(r["p_better"], r["p_equiv"], r["p_worse"]))
        bal.append(r)
    bal = pd.DataFrame(bal)
    means = Fw.groupby("key").hr_auc.mean()
    bal["auc_a"], bal["auc_b"] = bal.a.map(means), bal.b.map(means)
    bal.to_csv(analysis_file("ndvi_balance.csv"), index=False)
    show = out[out.name.isin(["hs_vs_ms", "t_hs_vs_ms", "ms5_vs_rgb3", "t_ms5_vs_rgb3"])]
    print(f"vegetated contrasts at NDVI thresholds {THRESHOLDS}; the {PAPER_THRESHOLD} values match the paper")
    print(show[["threshold", "n_health", "n_rust", "name", "n_paired", "diff", "ci95_lo", "ci95_hi", "p_better",
                "verdict"]].round(3).to_string(index=False))
    print(pd.DataFrame(imb).round(3).to_string(index=False))
    print(bal[bal.name.isin(["hs_vs_ms", "t_hs_vs_ms", "ms5_vs_rgb3", "t_ms5_vs_rgb3", "stretch"])][
        ["name", "n_paired", "auc_a", "auc_b", "diff", "ci95_lo", "ci95_hi", "p_better", "verdict"]].round(3)
        .to_string(index=False))


if __name__ == "__main__":
    main()
