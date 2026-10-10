"""Outer folds, cached feature tables, and the metrics every experiment reports."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from wheatrust.io_data import CACHE, CLASSES, RESULTS

FOLD_FILE = os.path.join(RESULTS, "folds", "folds.csv")
FEAT_DIR = os.path.join(CACHE, "features")
N_REPEATS = 10
N_FOLDS = 5


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
def overlap_groups(train_ids):
    """Group training patches that share image content.

    The audit finds a few pairs of patches whose tiles overlap, shifted by half a tile, recognisable by identical
    4 x 4 pixel blocks across all 125 bands (kind shared_4x4_blocks in results/audit/duplicates.csv). If the two
    patches of a pair fell on opposite sides of an outer split, a model would be validated on pixels it had trained
    on. Pairs are therefore merged into connected groups (union-find), and every group stays inside one fold.
    """
    dup = pd.read_csv(os.path.join(RESULTS, "audit", "duplicates.csv"))
    dup = dup[(dup.kind == "shared_4x4_blocks") & (dup.a_split == "train") & (dup.b_split == "train")]
    parent = {s: s for s in train_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(dup.a, dup.b):
        parent[find(a)] = find(b)
    roots = {s: find(s) for s in train_ids}
    codes = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return np.array([codes[roots[s]] for s in train_ids])


def make_folds(train_ids, y, n_repeats=N_REPEATS, n_folds=N_FOLDS, seed=20260507):
    """Stratified grouped k-fold assignments for every repeat, as a long table (sample_id, repeat, fold, group)."""
    groups = overlap_groups(train_ids)
    rows = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for k, (_, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            for i in va:
                rows.append(dict(sample_id=train_ids[i], repeat=r, fold=k))
    df = pd.DataFrame(rows)
    df["group"] = df.sample_id.map(dict(zip(train_ids, groups)))
    return df


def load_folds():
    return pd.read_csv(FOLD_FILE)


def fold_indices(folds, train_ids, repeat, fold):
    """Positions (into train_ids) of the training and validation parts of one outer fold."""
    f = folds[folds.repeat == repeat].set_index("sample_id").loc[train_ids, "fold"].values
    return np.where(f != fold)[0], np.where(f == fold)[0]


# ---------------------------------------------------------------------------
# feature tables (written by scripts/training/02_build_features.py and 03_stretch_control.py)
# ---------------------------------------------------------------------------
_CACHE = {}


def table(name):
    """A cached feature table indexed by sample_id, loaded once per process."""
    if name not in _CACHE:
        path = os.path.join(FEAT_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing; run scripts/training/02_build_features.py "
                                    "and scripts/training/03_stretch_control.py first")
        _CACHE[name] = pd.read_pickle(path)
    return _CACHE[name]


def spectra(name):
    """Valid-pixel mean spectrum of every patch, for the cleaned ('hsc') or full ('hsf') HS view."""
    return table(f"spectra_{name}")


def feature_matrix(tables, ids, families=None, exclude_families=None, columns_filter=None):
    """Concatenate feature tables for the given samples and optionally keep or drop feature families.

    Columns are named <view>__<family>__<name>. exclude_families accepts a bare family ('glcm') or a view and family
    ('hsc:deriv'), which is how the leave-one-family-out ablation removes one family from one view.
    """
    df = pd.concat([table(t).loc[ids] for t in tables], axis=1)
    if families is not None:
        df = df[[c for c in df.columns if c.split("__")[1] in families]]
    if exclude_families is not None:
        df = df[[c for c in df.columns if f"{c.split('__')[0]}:{c.split('__')[1]}" not in exclude_families
                 and c.split("__")[1] not in exclude_families]]
    if columns_filter is not None:
        df = df[[c for c in df.columns if columns_filter(c)]]
    return df


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y, proba, pred=None):
    """Scores of one set of predictions; the decision is the argmax of the probabilities unless pred is given.

    Probabilities are clipped at 1e-7 and renormalised so that the log loss stays finite for overconfident models.
    """
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1)
    proba = proba / proba.sum(1, keepdims=True)
    if pred is None:
        pred = proba.argmax(1)
    out = dict(
        macro_f1=f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        brier=float(np.mean(np.sum((proba - np.eye(3)[y]) ** 2, 1))),
        log_loss=log_loss(y, proba, labels=[0, 1, 2]),
    )
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"] = p[i]
        out[f"recall_{c}"] = r[i]
        out[f"f1_{c}"] = f[i]
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    out["confusion"] = json.dumps(cm.tolist())
    return out


def decide(proba, rule="argmax", thresholds=None):
    """Class decisions from probabilities, by plain argmax or by argmax of p / t with per-class thresholds t."""
    proba = np.asarray(proba, dtype=float)
    if rule == "argmax":
        return proba.argmax(1)
    if rule == "threshold":
        return (proba / np.asarray(thresholds)[None, :]).argmax(1)
    raise ValueError(rule)


def tune_thresholds(y, proba, grid=np.linspace(0.2, 2.0, 37)):
    """Per-class multiplicative thresholds that maximise macro-F1, found by coordinate ascent.

    The decision is argmax(p / t), with t normalised so that t[0] = 1. The thresholds are recorded with every FLAML
    unit (the thr_ columns of the fold tables), but reported scores and submissions use the plain argmax.
    """
    t = np.ones(3)
    best = f1_score(y, (proba / t).argmax(1), average="macro")
    for _ in range(3):
        for c in (1, 2):
            for g in grid:
                tt = t.copy()
                tt[c] = g
                s = f1_score(y, (proba / tt).argmax(1), average="macro")
                if s > best + 1e-12:
                    best, t = s, tt
    return t, best
