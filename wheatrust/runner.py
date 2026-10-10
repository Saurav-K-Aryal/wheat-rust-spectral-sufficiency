"""Experiment units, each one learner fitted on one outer training fold and scored on its validation fold.

A unit is (experiment, variant, repeat, fold). run_unit builds the input from the cached feature tables, fits the
learner on the outer training part of the saved fold, predicts the untouched validation part, and writes one pickle
to results/<exp>/units/<variant>/r<repeat>_f<fold>.pkl. Units already on disk are skipped and every file is written
atomically, so a driver can be stopped and restarted at any time. collect() merges the pickles of an experiment into
the tidy tables results/<exp>/fold_metrics.csv and results/<exp>/oof.csv.gz.

Seeds depend only on the repeat and the fold (1000 * repeat + fold), so a unit gives the same result whenever and in
whatever order it runs on one platform. Across platforms, FLAML can settle on a different one of several nearly
equivalent configurations, so every unit records the platform it ran on.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from wheatrust.common import decide, feature_matrix, fold_indices, load_folds, metrics, spectra
from wheatrust.io_data import RESULTS, Arrays

_A = None


def arrays():
    """The cached arrays, loaded once per process."""
    global _A
    if _A is None:
        _A = Arrays()
    return _A


def platform_label():
    """Operating system and processor of this machine, such as linux-x86_64 or macos-arm64."""
    system = {"Darwin": "macos"}.get(platform.system(), platform.system().lower())
    return f"{system}-{platform.machine().lower()}"


def single_thread():
    """Limit this process to one CPU thread (BLAS, OpenMP, and torch), as in the runs of the study.

    Thread counts can change the order of floating-point sums, so they are fixed to make units repeatable. Parallel
    runs use several worker processes instead.
    """
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def build_input(inp, ids_tr, ids_va, y_tr, seed):
    """(X_tr, X_va, columns) for an input spec.

    Everything fitted here (PCA, band selection, recursive feature elimination) sees only the training rows.
    """
    kind = inp["kind"]
    if kind == "features":
        tabs = inp.get("tables", [])
        Xtr_df = feature_matrix(tabs, ids_tr, inp.get("families"), inp.get("exclude"), _flt(inp)) if tabs else None
        Xva_df = feature_matrix(tabs, ids_va, inp.get("families"), inp.get("exclude"), _flt(inp)) if tabs else None
        parts_tr, parts_va, cols = [], [], []
        if Xtr_df is not None and Xtr_df.shape[1]:
            parts_tr.append(Xtr_df.values)
            parts_va.append(Xva_df.values)
            cols += list(Xtr_df.columns)
        if inp.get("pca"):
            src, n = inp["pca"]["source"], inp["pca"]["n"]
            S = spectra(src)
            str_, sva = S.loc[ids_tr].values, S.loc[ids_va].values
            p = PCA(n_components=n, random_state=seed).fit(str_)
            parts_tr.append(p.transform(str_))
            parts_va.append(p.transform(sva))
            cols += [f"{src}__pca__pc{i + 1}" for i in range(p.n_components_)]
        return np.hstack(parts_tr), np.hstack(parts_va), cols
    if kind in ("bandsel", "bands"):
        from wheatrust.bands import band_input

        return band_input(inp, ids_tr, ids_va, y_tr, seed)
    if kind == "rfecv":
        # recursive feature elimination (10 percent of the features per step) scored by inner 3-fold macro-F1
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import StratifiedKFold
        import lightgbm

        Xtr, Xva, cols = build_input(inp["base"], ids_tr, ids_va, y_tr, seed)
        sel = RFECV(lightgbm.LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1), step=0.1,
                    cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="f1_macro",
                    min_features_to_select=5, n_jobs=1).fit(Xtr, y_tr)
        keep = np.where(sel.support_)[0]
        return Xtr[:, keep], Xva[:, keep], [cols[i] for i in keep]
    raise ValueError(kind)


def _flt(inp):
    f = inp.get("filter")
    if f is None:
        return None
    if f == "band_only":
        # raw band statistics, keeping the valid-pixel fraction
        return lambda c: c.split("__")[1] in ("band", "meta")
    raise ValueError(f)


def assemble(inp, T_tr, T_va, seed):
    """Rebuild a 'features' input from given tables instead of the cache, as (X_tr, X_va).

    T_tr and T_va map table names (and spectra_<view> for the PCA input) to frames already restricted to the
    training and validation rows. The window permutation uses it to score a saved fold model on features recomputed
    from permuted cubes; the PCA is refitted on the training tables with the unit's seed, which reproduces the
    projection the model was trained with.
    """

    def mat(T):
        tabs = inp.get("tables", [])
        if not tabs:
            return None
        df = pd.concat([T[t] for t in tabs], axis=1)
        fam, exc, flt = inp.get("families"), inp.get("exclude"), _flt(inp)
        if fam is not None:
            df = df[[c for c in df.columns if c.split("__")[1] in fam]]
        if exc is not None:
            df = df[[c for c in df.columns if f"{c.split('__')[0]}:{c.split('__')[1]}" not in exc
                     and c.split("__")[1] not in exc]]
        if flt is not None:
            df = df[[c for c in df.columns if flt(c)]]
        return df.values

    Xtr, Xva = mat(T_tr), mat(T_va)
    parts_tr = [Xtr] if Xtr is not None else []
    parts_va = [Xva] if Xva is not None else []
    if inp.get("pca"):
        src, n = inp["pca"]["source"], inp["pca"]["n"]
        p = PCA(n_components=n, random_state=seed).fit(T_tr[f"spectra_{src}"].values)
        parts_tr.append(p.transform(T_tr[f"spectra_{src}"].values))
        parts_va.append(p.transform(T_va[f"spectra_{src}"].values))
    return np.hstack(parts_tr), np.hstack(parts_va)


def make_learner(spec, seed, columns=None):
    """A learner from a learner spec (see wheatrust.experiments)."""
    from wheatrust.learners import FlamlLearner, TabPFNLearner

    k = spec["kind"]
    if k == "flaml":
        return FlamlLearner(estimator_list=spec.get("estimator_list", "auto"), budget=spec["budget"], seed=seed,
                            tune_thr=spec.get("tune_thr", True))
    if k == "tabpfn":
        return TabPFNLearner(seed=seed, n_estimators=spec.get("n_estimators", "auto"))
    raise ValueError(k)


def subset(sub, tr, y, r, k):
    """Training positions of a unit, optionally reduced to a class-stratified random fraction (at least 5 per class).

    The generator is seeded from the repeat, fold, and draw, so every learner and view sees the same subset.
    """
    if sub is None:
        return tr
    rng = np.random.default_rng(10_000 * r + 100 * k + sub.get("draw", 0) + 7)
    if sub["kind"] == "fraction":
        keep = []
        for c in range(3):
            idx = tr[y[tr] == c]
            n = max(5, int(round(len(idx) * sub["frac"])))
            keep.append(rng.choice(idx, n, replace=False))
        return np.sort(np.concatenate(keep))
    raise ValueError(sub["kind"])


# ---------------------------------------------------------------------------
# one unit
# ---------------------------------------------------------------------------
def unit_path(exp, variant, r, k):
    return os.path.join(RESULTS, exp, "units", variant, f"r{r}_f{k}.pkl")


def model_path(exp, variant, r, k):
    return os.path.join(RESULTS, exp, "models", variant, f"r{r}_f{k}.pkl")


def run_unit(exp, variant, spec, r, k, folds=None):
    """Fit and score one unit; returns 'skip' if its file already exists and 'done' otherwise."""
    out = unit_path(exp, variant, r, k)
    if os.path.exists(out):
        return "skip"
    A = arrays()
    ids = A.sample_id[A.tr]
    y = A.y_train()
    folds = load_folds() if folds is None else folds
    tr, va = fold_indices(folds, ids, r, k)
    tr = subset(spec.get("subset"), tr, y, r, k)
    seed = 1000 * r + k
    t0 = time.time()
    Xtr, Xva, cols = build_input(spec["input"], ids[tr], ids[va], y[tr], seed)
    L = make_learner(spec["learner"], seed, cols).fit(Xtr, y[tr])
    t_fit = time.time() - t0
    t1 = time.time()
    P = np.asarray(L.predict_proba(Xva), dtype=float)
    t_pred = (time.time() - t1) / max(len(va), 1)
    res = dict(exp=exp, variant=variant, repeat=r, fold=k, platform=platform_label(), n_train=len(tr), n_val=len(va),
               n_features=Xtr.shape[1], fit_time_s=t_fit, pred_time_per_sample_s=t_pred, val_ids=ids[va],
               y_val=y[va], proba=P, describe=L.describe())
    res["metrics"] = metrics(y[va], P)
    thr = getattr(L, "thresholds", None)
    if thr is not None:
        res["thresholds"] = np.asarray(thr)
        res["metrics_thr"] = metrics(y[va], P, decide(P, "threshold", thresholds=thr))
    if spec.get("save_model"):
        mp = model_path(exp, variant, r, k)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        with open(mp + ".tmp", "wb") as f:
            pickle.dump(dict(learner=L, columns=cols, spec=spec), f)
        os.replace(mp + ".tmp", mp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(out + ".tmp", out)
    return "done"


def parse_repeats(text):
    """Outer repeats (or folds) from a command-line value such as '0-9' or '0,1,2'."""
    lo_hi = text.split("-")
    if len(lo_hi) == 2:
        return list(range(int(lo_hi[0]), int(lo_hi[1]) + 1))
    return [int(x) for x in text.split(",")]


def pending(exp, variants, repeats, folds=range(5)):
    """(variant, repeat, fold) of every unit without a file. A spec's own 'repeats' overrides the repeats given."""
    todo = []
    for v, spec in variants.items():
        for r in spec.get("repeats", repeats):
            for k in spec.get("folds", folds):
                if not os.path.exists(unit_path(exp, v, r, k)):
                    todo.append((v, r, k))
    return todo


# ---------------------------------------------------------------------------
# tidy tables
# ---------------------------------------------------------------------------
# The platform reruns are only compared on scores and predictions, so their table keeps these columns.
RERUN_COLUMNS = ["variant", "repeat", "fold", "n_train", "n_features", "platform"]


def collect(exp):
    """Merge the unit pickles of an experiment into results/<exp>/fold_metrics.csv and oof.csv.gz.

    fold_metrics.csv has one row per unit with the learner's description (selected estimator, configuration,
    thresholds), the argmax metrics, and the thresholded metrics (thr_ columns). oof.csv.gz has the out-of-fold
    class probabilities of every validation sample, rounded to five decimals. Returns both frames, or (None, None)
    when the experiment has no units.
    """
    from wheatrust.experiments import PLATFORM_RERUNS

    base = os.path.join(RESULTS, exp, "units")
    rows, oof = [], []
    if not os.path.isdir(base):
        return None, None
    for v in sorted(os.listdir(base)):
        for f in sorted(os.listdir(os.path.join(base, v))):
            if not f.endswith(".pkl"):
                continue
            with open(os.path.join(base, v, f), "rb") as fh:
                u = pickle.load(fh)
            row = dict(exp=exp, variant=v, repeat=u["repeat"], fold=u["fold"], n_train=u["n_train"],
                       n_features=u["n_features"], fit_time_s=u["fit_time_s"],
                       pred_time_per_sample_s=u["pred_time_per_sample_s"], platform=u["platform"])
            # the learner's own fit time replaces the unit's, which also includes building the input
            for kk, vv in u["describe"].items():
                if isinstance(vv, (list, np.ndarray)):
                    vv = json.dumps([round(float(x), 5) for x in np.ravel(vv)])
                row[kk] = vv
            if exp in PLATFORM_RERUNS:
                row = {c: row[c] for c in RERUN_COLUMNS}
            for kk, vv in u["metrics"].items():
                row[kk] = vv
            if "metrics_thr" in u and exp not in PLATFORM_RERUNS:
                for kk, vv in u["metrics_thr"].items():
                    row[f"thr_{kk}"] = vv
            rows.append(row)
            P = u["proba"]
            oof.append(pd.DataFrame(dict(variant=v, repeat=u["repeat"], fold=u["fold"], sample_id=u["val_ids"],
                                         y=u["y_val"], p_Health=P[:, 0], p_Rust=P[:, 1], p_Other=P[:, 2])))
    fm = pd.DataFrame(rows)
    oo = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()
    fm.to_csv(os.path.join(RESULTS, exp, "fold_metrics.csv"), index=False)
    oo.to_csv(os.path.join(RESULTS, exp, "oof.csv.gz"), index=False, float_format="%.5f")
    return fm, oo
