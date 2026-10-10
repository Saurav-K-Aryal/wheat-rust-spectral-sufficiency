"""How consistently the single-view permutation ranks the spectral windows.

The permutation runs (scripts/training) apply each fold's model to its validation patches with one 40 nm window of
the cleaned HS cube, or one MS band, shuffled across those patches (features recomputed, three permutations per
window), and record how much the Health versus Rust AUC and macro-F1 drop (results/E8/hr_window_folds.csv, and
hr_window_folds_veg.csv for the FLAML models scored on vegetated patches). Per model this script ranks the windows
within every fold by their AUC drop and reports
  Kendall's W        agreement of the per-fold rankings, from 0 (none) to 1 (every fold ranks alike)
  rank               each window's place when ordered by its mean drop over folds
  share_folds_top3   the share of folds in which the window was among the three largest drops
and compares the top window with every other window on the same folds (_shared.paired_test, ROPE 0.02 AUC), which
shows which windows cannot be told apart from the top one.

Writes to results/analysis/ hr_window_stats.csv, hr_window_pairs.csv, and the same for vegetated patches,
hr_window_stats_veg.csv and hr_window_pairs_veg.csv.
usage: python scripts/analysis/05_window_stats.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import pandas as pd

from scripts.analysis._shared import RESULTS, analysis_file, paired_test


def kendall_w(ranks):
    """Kendall's coefficient of concordance for m folds (rows) ranking n windows (columns), without tie correction."""
    m, n = ranks.shape
    Rj = ranks.sum(axis=0)
    return float(12 * ((Rj - Rj.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n)))


def window_stats(W, column):
    stats_rows, pair_rows = [], []
    for model, g in W.groupby("model"):
        piv = g.groupby(["window", "repeat", "fold"])[column].mean().reset_index().pivot_table(
            index=["repeat", "fold"], columns="window", values=column)
        ranks = piv.rank(axis=1, ascending=False)
        kw = kendall_w(ranks)
        means = piv.mean(axis=0).sort_values(ascending=False)
        top = means.index[0]
        for i, (w, v) in enumerate(means.items()):
            stats_rows.append({"model": model, "window": w, f"mean_{column}": float(v), "rank": i + 1, "kendall_w": kw,
                               "share_folds_top3": float((ranks[w] <= 3).mean()), "n_folds": len(piv)})
            if w == top:
                continue
            pb, pe, pw, p = paired_test(piv[top].values, piv[w].values)
            pair_rows.append(dict(model=model, top=top, window=w, diff=float(piv[top].mean() - piv[w].mean()),
                                  p_top_better=pb, p_equiv=pe, p_top_worse=pw, p_corrected=p))
    return pd.DataFrame(stats_rows), pd.DataFrame(pair_rows)


def main():
    W = pd.read_csv(os.path.join(RESULTS, "E8", "hr_window_folds.csv"))
    s, p = window_stats(W, "drop_hr_auc")
    s.to_csv(analysis_file("hr_window_stats.csv"), index=False)
    p.to_csv(analysis_file("hr_window_pairs.csv"), index=False)
    V = pd.read_csv(os.path.join(RESULTS, "E8", "hr_window_folds_veg.csv"))
    V["window"] = V.window.astype(str)
    s, p = window_stats(V, "drop_hr_auc_veg")
    s.drop(columns="share_folds_top3").to_csv(analysis_file("hr_window_stats_veg.csv"), index=False)
    p.to_csv(analysis_file("hr_window_pairs_veg.csv"), index=False)
    print(f"window statistics for {W.model.nunique()} models, and {V.model.nunique()} on vegetated patches")


if __name__ == "__main__":
    main()
