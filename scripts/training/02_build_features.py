"""Feature cache with every engineered feature table and the patch mean spectra, computed once for all 900 patches.

The features are fixed per-patch transforms without fitted state (wheatrust.features), so computing them for all
patches at once leaks nothing between folds; PCA and every other fitted step run later inside the outer folds.

Reads   the array cache (built from DATA on first use)
Writes  cache/features/<table>.pkl          rgb, ms, hsc, hsf, cross, rgbhs, mshs (indexed by sample_id)
        cache/features/spectra_<view>.pkl   valid-pixel mean spectra of the cleaned (hsc) and full (hsf) HS views
        results/features/feature_dictionary.csv   table, view, family, and name of every feature
usage: python scripts/training/02_build_features.py      (a few minutes)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

from wheatrust.common import FEAT_DIR  # noqa: E402
from wheatrust.features import all_feature_tables  # noqa: E402
from wheatrust.io_data import RESULTS, Arrays  # noqa: E402


def main():
    t0 = time.time()
    A = Arrays()
    T, spectra = all_feature_tables(A)
    os.makedirs(FEAT_DIR, exist_ok=True)
    rows = []
    for k, df in T.items():
        df.to_pickle(os.path.join(FEAT_DIR, f"{k}.pkl"))
        for c in df.columns:
            v, fam, name = c.split("__", 2)
            rows.append(dict(table=k, feature=c, view=v, family=fam, name=name))
    for k, df in spectra.items():
        df.to_pickle(os.path.join(FEAT_DIR, f"spectra_{k}.pkl"))
    os.makedirs(os.path.join(RESULTS, "features"), exist_ok=True)
    fd = pd.DataFrame(rows)
    fd.to_csv(os.path.join(RESULTS, "features", "feature_dictionary.csv"), index=False)
    print(fd.groupby(["table", "family"]).size().to_string())
    print({k: v.shape for k, v in T.items()})
    print("non-finite values:", {k: int((~v.astype(float).notna()).sum().sum()) for k, v in T.items()})
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
