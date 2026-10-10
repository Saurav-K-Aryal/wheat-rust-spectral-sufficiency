"""How stable the in-fold band selection is across the 50 outer folds.

Each outer training fold selected its own ranked list of cleaned HS bands (results/E9/selections), and a budget of k
bands takes the first k. For every pair of folds this script compares the two k-band sets in two ways.

  Kuncheva's consistency index (Kuncheva 2007), (r n - k^2) / (k (n - k)) for r shared bands out of n = 101
  candidates. It is 1 for identical sets and 0 for the overlap expected from random sets of the same size.
  The share of bands with a match within 8 nm (two band steps) in the other set, averaged over both directions.
  Neighbouring bands are nearly collinear, so a selector can swap one for its neighbour without changing what it
  measures, which the exact index counts as a disagreement. The chance level of this share comes from 2,000 pairs
  of random k-band sets.

Writes results/analysis/selection_stability.csv, one row per budget.
usage: python scripts/analysis/13_selection_stability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

from scripts.analysis._shared import RESULTS, analysis_file
from wheatrust.io_data import CLEAN_IDX, HS_WL

BUDGETS = [1, 2, 3, 5, 8, 16, 32]
TOLERANCE_NM = 8.0
N_RANDOM = 2000
SEED = 20261009


def kuncheva(a, b, n):
    k = len(a)
    if k == n:
        return float("nan")
    r = len(set(a) & set(b))
    return (r * n - k * k) / (k * (n - k))


def near_share(a, b, tol=TOLERANCE_NM):
    """Share of the wavelengths of a and of b that have a partner within tol nm in the other set, averaged."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = (np.abs(a[:, None] - b[None, :]) <= tol + 1e-9).any(1).mean()
    ba = (np.abs(b[:, None] - a[None, :]) <= tol + 1e-9).any(1).mean()
    return (ab + ba) / 2


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "E9", "selections", "*.json")))
    lists = [json.load(open(f))["wavelengths"] for f in files]
    candidates = HS_WL[CLEAN_IDX]
    n = len(candidates)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in BUDGETS:
        sets = [lst[:k] for lst in lists]
        pairs = list(itertools.combinations(range(len(sets)), 2))
        ki = [kuncheva(sets[i], sets[j], n) for i, j in pairs]
        jac = [len(set(sets[i]) & set(sets[j])) / len(set(sets[i]) | set(sets[j])) for i, j in pairs]
        near = [near_share(sets[i], sets[j]) for i, j in pairs]
        chance = [near_share(rng.choice(candidates, k, replace=False), rng.choice(candidates, k, replace=False))
                  for _ in range(N_RANDOM)]
        rows.append(dict(k=k, n_folds=len(sets), n_pairs=len(pairs), n_candidates=n, kuncheva_mean=np.mean(ki),
                         jaccard_mean=np.mean(jac), near_share_mean=np.mean(near), near_share_chance=np.mean(chance),
                         tolerance_nm=TOLERANCE_NM))
    out = pd.DataFrame(rows)
    out.to_csv(analysis_file("selection_stability.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
