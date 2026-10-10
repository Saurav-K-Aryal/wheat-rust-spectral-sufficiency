"""Tidy summaries of the shipped per-fold results, and every method compared with the baseline.

Reads results/<experiment>/fold_metrics.csv for the experiments listed in _shared.EXPERIMENTS, results/E1/oof.csv.gz,
and results/E10/crossbatch.csv. Writes to results/analysis/
  all_fold_metrics.csv        every outer fold of every method, keyed '<experiment>:<variant>'
  e1_summary.csv              the baseline's metrics, mean and standard deviation over outer folds
  e1_selected_estimators.csv  how often FLAML's search chose each estimator for the baseline
  e1_confusion_pooled.csv     the baseline's confusion matrix, pooled over every outer fold
  e3t_views.csv, e4_views.csv, e5_ablation.csv, e9_sensors.csv, e9t_sensors.csv
                              one row per method with its summary and its paired comparison with the baseline on the
                              same folds (_shared.paired_test, Holm within each file)
  refit_gain.csv              what the last quarter of an outer training fold adds (see refit_gain below)
  crossbatch_summary.csv      cross-batch transfer, mean macro-F1 per learner, input, and split
  crossbatch_cost.csv         what training on one HS file format and testing on the other costs, relative to a
                              random split of the same sizes and class counts
usage: python scripts/analysis/01_summaries.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from scripts.analysis._shared import (ANALYSIS, BASE, EXPERIMENTS, FLAML_VIEW, N_TEST, N_TRAIN, RESULTS,
                                      analysis_file, read_experiment)
from wheatrust.io_data import CLASSES
from wheatrust.stats import compare_to, summarize

MET = ["macro_f1", "accuracy", "balanced_accuracy", "brier", "log_loss"] + [
    f"{m}_{c}" for m in ("precision", "recall", "f1") for c in CLASSES]


def pooled_confusion(oo, key):
    o = oo[oo.key == key]
    pred = o[["p_Health", "p_Rust", "p_Other"]].values.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(o.y.values, pred):
        cm[t, p] += 1
    return pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])


def compare_with_baseline(FM, keys, name):
    """Summary of each key and of the baseline, merged with the paired comparison of each key with the baseline."""
    summ = summarize(FM[FM.key.isin(keys + [BASE])].assign(variant=lambda d: d.key), MET)
    c = compare_to(FM.assign(variant=FM.key), BASE, keys)
    summ.merge(c, on="variant", how="left").to_csv(analysis_file(f"{name}.csv"), index=False)


def refit_gain(FM):
    """FLAML on the full outer training fold (480 patches) against FLAML on three stratified draws of three quarters
    of it (360 patches, E3f), on the same validation patches. A submission refits on all 600 training patches, 120
    more than an outer fold, so this paired gain shows how much a refit alone can raise a score. The three draws are
    averaged within each fold, and the interval for the macro-F1 gain uses the corrected resampled t distribution."""
    rows = []
    for view, full_key in FLAML_VIEW.items():
        full = FM[FM.key == full_key]
        part = FM[(FM.exp == "E3f") & FM.variant.str.fullmatch(rf"flaml_{view}_f3_4_d\d+")]
        row = dict(view=view)
        for metric in ("macro_f1", "accuracy"):
            a = full.groupby(["repeat", "fold"])[metric].mean()
            b = part.groupby(["repeat", "fold"])[metric].mean()
            idx = a.index.intersection(b.index)
            d = (a.loc[idx] - b.loc[idx]).values
            row[f"{metric}_gain"] = d.mean()
            if metric == "macro_f1":
                J = len(d)
                se_c = d.std(ddof=1) * np.sqrt(1.0 / J + N_TEST / N_TRAIN)
                q = tdist.ppf(0.975, J - 1)
                row.update(n_folds=J, macro_f1_gain_sd=d.std(ddof=1), macro_f1_gain_lo=d.mean() - q * se_c,
                           macro_f1_gain_hi=d.mean() + q * se_c)
        rows.append(row)
    cols = ["view", "n_folds", "macro_f1_gain", "macro_f1_gain_sd", "macro_f1_gain_lo", "macro_f1_gain_hi",
            "accuracy_gain"]
    pd.DataFrame(rows)[cols].to_csv(analysis_file("refit_gain.csv"), index=False)


def crossbatch():
    """The HS files come in two formats (125 and 126 bands, the 126th a constant fill). Each model was trained on
    the training files of one format and tested on those of the other and, as the reference, on stratified random
    splits with the same sizes and class counts. The cost of a direction is the random split minus the transfer."""
    df = pd.read_csv(os.path.join(RESULTS, "E10", "crossbatch.csv"))
    s = df.groupby(["learner", "view", "name"]).agg(m=("macro_f1", "mean"), sd=("macro_f1", "std"),
                                                    n=("macro_f1", "size"))
    s.reset_index().to_csv(analysis_file("crossbatch_summary.csv"), index=False)
    rows = []
    for (lrn, vw), g in s.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        for direction, rnd in [("125_to_126", "random_489_to_111"), ("126_to_125", "random_111_to_489")]:
            rows.append(dict(learner=lrn, view=vw, direction=direction, transfer=g.loc[direction].m,
                             random=g.loc[rnd].m, cost=g.loc[rnd].m - g.loc[direction].m))
    pd.DataFrame(rows).to_csv(analysis_file("crossbatch_cost.csv"), index=False)


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    FM = pd.concat([read_experiment(e) for e in EXPERIMENTS], ignore_index=True)
    FM.to_csv(analysis_file("all_fold_metrics.csv"), index=False)

    e1 = FM[FM.key == BASE]
    summarize(e1.assign(variant=e1.key), MET).to_csv(analysis_file("e1_summary.csv"), index=False)
    sel = e1.selected_estimator.value_counts().rename_axis("estimator").reset_index(name="n_folds")
    sel["share"] = sel.n_folds / sel.n_folds.sum()
    sel.to_csv(analysis_file("e1_selected_estimators.csv"), index=False)
    pooled_confusion(read_experiment("E1", "oof"), BASE).to_csv(analysis_file("e1_confusion_pooled.csv"))

    keys = sorted(set(FM.key))
    for prefix, name in [("E3t:", "e3t_views"), ("E4:", "e4_views"), ("E5:", "e5_ablation"), ("E9:", "e9_sensors"),
                         ("E9t:", "e9t_sensors")]:
        compare_with_baseline(FM, [k for k in keys if k.startswith(prefix)], name)
    refit_gain(FM)
    crossbatch()
    print(f"{len(FM)} outer folds of {FM.key.nunique()} methods summarised in {os.path.relpath(ANALYSIS)}")


if __name__ == "__main__":
    main()
