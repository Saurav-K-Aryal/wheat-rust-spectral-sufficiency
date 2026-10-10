"""How well each HS band separates Rust from Health patches on its own, without any model.

The paper's wavelength importance permutes windows upstream of the engineered features, so it partly reflects how
many features use a window. This model-free check complements it. For every HS band, and for the ten 40 nm windows
of the window permutation (scripts/training/06_window_permutation.py), the patch-mean reflectance of each training
patch is compared between the true Rust and Health patches with the ROC AUC, P(Rust value > Health value) with ties
counted half. An AUC of 0.5 means no separation, values above 0.5 mean Rust patches reflect more, and values below
0.5 mean they reflect less. Two subsets are scored, every non-empty patch and the vegetated patches the paper uses
(not empty, patch-mean NDVI of at least 0.5, results/audit/composition_ndvi.csv).

Intervals are 95 percent percentile intervals from 2,000 bootstrap draws of patches. Patches whose tiles overlap
form one group (results/folds/folds.csv) and are drawn together, like they share a fold in cross-validation.

Reads   the training HS cubes through the array cache (built from DATA on first use), results/audit, results/folds
Writes  results/audit/band_separability.csv   one row per band or window and subset
usage: python scripts/eda/04_band_separability.py      (about a minute, needs the competition data)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from wheatrust.io_data import CLEAN_IDX, HS_WL, RESULTS, Arrays  # noqa: E402

N_BOOT = 2000
SEED = 20261009
VEG_NDVI = 0.5


def windows():
    """The ten 40 nm windows over the cleaned range, as in scripts/training/06_window_permutation.py."""
    wl_clean = HS_WL[CLEAN_IDX]
    out = []
    for i in range(10):
        lo, hi = 490 + 40 * i, 530 + 40 * i
        sel = (wl_clean >= lo) & ((wl_clean < hi) if i < 9 else (wl_clean <= 890))
        out.append((f"{lo}-{hi if i < 9 else 890}", np.asarray(CLEAN_IDX)[sel]))
    return out


def mean_spectra(A):
    """Valid-pixel mean reflectance of every training patch (rows in the order of A.tr); NaN for empty patches."""
    out = np.full((len(A.tr), len(HS_WL)), np.nan)
    for row, i in enumerate(A.tr):  # one patch at a time keeps the memory small
        cube = np.asarray(A.hs[i], dtype=np.float64) / 10000.0  # digital numbers to reflectance
        valid = ~(cube == 0).all(-1)  # empty mosaic pixels are zero in every band
        if valid.any():
            out[row] = cube[valid].mean(0)
    return out


def auc_columns(X, rust):
    """ROC AUC of every column of X for Rust (rust True) against Health, from average ranks."""
    R = rankdata(X, axis=0)
    n1 = int(rust.sum())
    n0 = len(rust) - n1
    return (R[rust].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    A = Arrays()
    ids = A.sample_id[A.tr]
    labels = A.label[A.tr]
    spec = mean_spectra(A)
    wins = windows()
    X = np.column_stack([spec] + [spec[:, idx].mean(1) for _, idx in wins])
    names = [f"{w:.0f}" for w in HS_WL] + [w for w, _ in wins]
    kinds = ["band"] * len(HS_WL) + ["window"] * len(wins)
    centre = list(HS_WL) + [float(np.mean(HS_WL[idx])) for _, idx in wins]
    clean = [bool(i in set(CLEAN_IDX)) for i in range(len(HS_WL))] + [True] * len(wins)

    comp = pd.read_csv(os.path.join(RESULTS, "audit", "composition_ndvi.csv")).set_index("sample_id").loc[ids]
    folds = pd.read_csv(os.path.join(RESULTS, "folds", "folds.csv"))
    group = folds[folds.repeat == 0].set_index("sample_id").group.loc[ids].to_numpy()
    nonempty = ~comp["empty"].astype(bool).to_numpy() & np.isfinite(spec).all(1)
    subsets = {"all": nonempty, "vegetated": nonempty & (comp.ndvi_mean.to_numpy() >= VEG_NDVI)}

    rng = np.random.default_rng(SEED)
    rows = []
    for subset, keep in subsets.items():
        m = keep & np.isin(labels, ["Health", "Rust"])
        Xs, rust, g = X[m], labels[m] == "Rust", group[m]
        point = auc_columns(Xs, rust)
        members = {u: np.where(g == u)[0] for u in np.unique(g)}
        units = np.array(list(members))
        boot = np.empty((N_BOOT, X.shape[1]))
        for b in range(N_BOOT):
            idx = np.concatenate([members[u] for u in rng.choice(units, len(units), replace=True)])
            boot[b] = auc_columns(Xs[idx], rust[idx])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for j in range(X.shape[1]):
            rows.append(dict(kind=kinds[j], name=names[j], wavelength_nm=centre[j], in_clean_range=clean[j],
                             subset=subset, n_health=int((~rust).sum()), n_rust=int(rust.sum()), auc=point[j],
                             auc_lo95=lo[j], auc_hi95=hi[j], separation=abs(point[j] - 0.5)))
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "audit", "band_separability.csv")
    out.to_csv(path, index=False)
    for subset in subsets:
        w = out[(out.kind == "window") & (out.subset == subset)].sort_values("separation", ascending=False)
        print(f"{subset}: windows by separation " + ", ".join(f"{r.name} ({r.auc:.2f})" for r in w.itertuples()))
    print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
