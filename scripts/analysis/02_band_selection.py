"""Which HS bands the in-fold band selection picked.

Band selection ran inside every outer training fold (scripts/training), greedily adding the cleaned HS band that most
improved the inner three-fold macro-F1 of shrinkage LDA on patch-mean reflectance. Each fold's ranked list is stored
as results/E9/selections/sel_seed<s>.json with s = 1000 * repeat + fold, so a budget of k bands is the first k of it.

Writes to results/analysis/
  e9_selection_frequency.csv  per budget k, how often each wavelength was among the first k (share of folds)
  e9_selections_by_fold.csv   the first eight selected wavelengths and the inner LDA scores of every fold
usage: python scripts/analysis/02_band_selection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import glob
import json
import os

import pandas as pd

from scripts.analysis._shared import RESULTS, analysis_file

BUDGETS = [1, 2, 3, 5, 8, 16, 32]


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "E9", "selections", "*.json")))
    rows, first = [], []
    for f in files:
        s = json.load(open(f))
        r, k = divmod(int(os.path.basename(f).split("seed")[1].split(".")[0]), 1000)
        for kk in BUDGETS:
            for w in s["wavelengths"][:kk]:
                rows.append(dict(k=kk, repeat=r, fold=k, wavelength=w))
        first.append(dict(seed=os.path.basename(f), **{f"band{i + 1}": w for i, w in enumerate(s["wavelengths"][:8])},
                          **{f"inner_k{kk}": s["inner_lda_macro_f1"][kk - 1] for kk in BUDGETS}))
    d = pd.DataFrame(rows)
    n_units = d.groupby("k")[["repeat", "fold"]].apply(lambda x: len(x.drop_duplicates())).rename("n_units")
    fr = d.groupby(["k", "wavelength"]).size().rename("count").reset_index().merge(n_units.reset_index(), on="k")
    fr["frequency"] = fr["count"] / fr.n_units
    fr.to_csv(analysis_file("e9_selection_frequency.csv"), index=False)
    pd.DataFrame(first).to_csv(analysis_file("e9_selections_by_fold.csv"), index=False)
    print(f"band selections of {len(files)} outer folds summarised")


if __name__ == "__main__":
    main()
