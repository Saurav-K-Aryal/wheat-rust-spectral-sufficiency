"""Band budgets and placements (E9 and E9t).

A band set is either chosen inside each outer training fold by greedy forward selection over the cleaned HS bands,
or fixed at the HS bands behind the RGB view (482, 550, and 650 nm) or the MS view (482, 550, 650, 738, and 834 nm).
Every band set, selected or fixed, is turned into the same generic features (generic_band_features), so budgets
and placements are compared like for like rather than through view-specific indices.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from wheatrust.features import masked_stats, sdiv
from wheatrust.io_data import CLEAN_IDX, HS_WL, RESULTS


def band_cube(ids, spec):
    """(n, 32, 32, k) reflectance cube of the listed samples for a list of k HS band indices."""
    from wheatrust.runner import arrays

    A = arrays()
    pos = pd.Series(np.arange(len(A.sample_id)), index=A.sample_id).loc[ids].values
    return np.asarray(A.hs[pos][..., list(spec)], dtype=np.float32) / 10000.0


def generic_band_features(cube, valid=None, nd=True):
    """Generic features of any band set, as (matrix, column names).

    The features are the valid-pixel fraction and five valid-pixel statistics per band. For sets of up to eight
    bands (and nd True) they add the mean, standard deviation, and 10th and 90th percentiles of every pairwise
    normalised difference (R_i - R_j) / (R_i + R_j). Their number grows quadratically with the number of bands, so
    larger sets go without them. The eight-band control (nd False) tests whether that switch explains how the scores
    behave above eight bands.
    """
    n, h, w, k = cube.shape
    if valid is None:
        valid = ~(cube == 0).all(-1)
    cols = {"meta_valid_frac": valid.mean((1, 2))}
    for j in range(k):
        st = masked_stats(cube[..., j], valid)
        for si, s in enumerate(["mean", "std", "p10", "p50", "p90"]):
            cols[f"b{j}_{s}"] = st[:, si]
    if k <= 8 and nd:
        for i in range(k):
            for j in range(i + 1, k):
                nd = sdiv(cube[..., i] - cube[..., j], cube[..., i] + cube[..., j])
                st = masked_stats(nd, valid)
                for si, s in [(0, "mean"), (1, "std"), (2, "p10"), (4, "p90")]:
                    cols[f"nd{i}_{j}_{s}"] = st[:, si]
    return pd.DataFrame(cols).values.astype(np.float32), list(cols)


def forward_select(Xs, y, kmax, seed):
    """Greedy forward selection of up to kmax columns of Xs (patch-mean reflectance per band).

    Each step adds the band that most improves the macro-F1 of shrinkage LDA (Ledoit-Wolf shrinkage) under
    stratified 3-fold cross-validation inside the data given. Shrinkage keeps the covariance estimate stable when
    neighbouring bands are almost collinear. Returns the chosen column indices in order and the score after each step.
    """
    skf = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xs, y))
    chosen, scores = [], []
    for _ in range(kmax):
        best, best_j = -1, None
        for j in range(Xs.shape[1]):
            if j in chosen:
                continue
            cols = chosen + [j]
            pred = np.zeros(len(y), int)
            for tr, va in skf:
                m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xs[tr][:, cols], y[tr])
                pred[va] = m.predict(Xs[va][:, cols])
            s = f1_score(y, pred, average="macro")
            if s > best + 1e-12:
                best, best_j = s, j
        chosen.append(best_j)
        scores.append(best)
    return chosen, scores


def selection(ids_tr, y_tr, seed, kmax=32, cache_dir=None):
    """Forward selection of up to kmax cleaned HS bands on the given training samples, cached as JSON.

    Budgets of 1, 2, 3, 5, 8, 16, and 32 bands are prefixes of one selection order, so a fold needs only one search.
    The cache file is named after the seed (1000 * repeat + fold for an outer fold), so E9 and E9t read the same
    selection in every fold.
    """
    path = os.path.join(cache_dir, f"sel_seed{seed}.json") if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from wheatrust.common import spectra

    S = spectra("hsc").loc[ids_tr].values
    ch, sc = forward_select(S, y_tr, kmax, seed)
    out = {"bands_hs_index": [int(CLEAN_IDX[j]) for j in ch], "wavelengths": [float(HS_WL[CLEAN_IDX[j]]) for j in ch],
           "inner_lda_macro_f1": [float(s) for s in sc]}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
    return out


SELECTION_DIR = os.path.join(RESULTS, "E9", "selections")


def band_input(inp, ids_tr, ids_va, y_tr, seed):
    """(X_tr, X_va, columns) for a band-set input, either selected bands {'kind': 'bandsel', 'k': k} or fixed
    bands {'kind': 'bands', 'wavelengths': [...]}, each fixed wavelength taken from the nearest HS band."""
    kind = inp["kind"]
    if kind == "bandsel":
        sel = selection(ids_tr, y_tr, seed, cache_dir=SELECTION_DIR)
        spec = sel["bands_hs_index"][: inp["k"]]
    elif kind == "bands":
        spec = [int(np.argmin(np.abs(HS_WL - w))) for w in inp["wavelengths"]]
    else:
        raise ValueError(kind)
    nd = inp.get("nd", True)
    Ftr, cols = generic_band_features(band_cube(ids_tr, spec), nd=nd)
    Fva, _ = generic_band_features(band_cube(ids_va, spec), nd=nd)
    return Ftr, Fva, cols
