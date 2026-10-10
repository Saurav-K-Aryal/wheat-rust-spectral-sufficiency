"""How many of the engineered MS features each MS band enters.

Window permutation scores a band by the drop in performance when that band is permuted and every feature is
recomputed from the permuted image (scripts/training/06_window_permutation.py). A band that feeds many features can
therefore rank high partly because it feeds many features. This script counts, for each of the five MS bands, the
features of the MS view that change when only that band changes. It runs the feature code itself
(wheatrust.features.ms_features) on a small synthetic image and on copies in which one band receives independent
pixel noise, so the counts follow the code rather than a hand-made list. Noise rather than a common scaling is used
because texture features quantise each patch between its own percentiles and would not notice a scaling. No data are
read.

Writes results/analysis/ms_feature_use.csv, one row per band.
usage: python scripts/analysis/14_ms_feature_use.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import numpy as np
import pandas as pd

from scripts.analysis._shared import analysis_file
from wheatrust.features import ms_features

MS_WL = [482, 550, 650, 738, 834]


def main():
    rng = np.random.default_rng(20261010)
    # a few plausible vegetation-like patches (reflectance rising towards the NIR), with no empty pixels
    base = np.array([0.04, 0.08, 0.05, 0.25, 0.40])
    img = np.clip(base + rng.normal(0, 0.02, size=(4, 16, 16, 5)), 0.01, None)
    ref = ms_features(img)
    rows = []
    for i, wl in enumerate(MS_WL):
        alt = img.copy()
        alt[..., i] = np.clip(alt[..., i] + rng.normal(0, 0.01, size=alt.shape[:3]), 0.01, None)
        changed = ~np.isclose(ms_features(alt).to_numpy(float), ref.to_numpy(float), rtol=1e-9, atol=1e-12).all(0)
        rows.append(dict(band_nm=wl, n_features=int(changed.sum()), n_total=ref.shape[1]))
    out = pd.DataFrame(rows)
    out.to_csv(analysis_file("ms_feature_use.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
