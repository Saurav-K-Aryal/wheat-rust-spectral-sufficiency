"""Adversarial validation and the HS file-format check.

Adversarial validation asks how well a classifier can tell training patches from holdout patches using only their
inputs. A cross-validated ROC AUC near 0.5 means the holdout looks like the training data; the gain of a LightGBM
model trained on that task names the features that differ most. No label is used.

The HS files come in two formats, 125 bands and 126 bands with a constant fill band. The same kind of classifier,
trained to tell the two formats apart within the training set from the engineered features alone, measures how far
the formats behave like separate processing batches (the cross-batch transfer of
scripts/training/07_crossbatch.py measures what that costs a model).

Reads   the feature cache (run scripts/training/02_build_features.py first) and the array cache
Writes  results/E12/adversarial.csv         train versus holdout AUC per view, five seeds of 5-fold CV
        results/E12/adversarial_top.csv     the 15 features with the largest gain, all views
        results/E12/batch_adversarial.csv   125- versus 126-band files within the training set
usage: python scripts/eda/03_adversarial_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wheatrust.common import feature_matrix  # noqa: E402
from wheatrust.io_data import RESULTS  # noqa: E402
from wheatrust.runner import arrays  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(RESULTS, "E12")
VIEWS = {"rgb": ["rgb"], "ms": ["ms"], "hsc": ["hsc"], "all": ["rgb", "ms", "hsc", "cross"]}


def cv_auc(X, target, seed):
    """Out-of-fold ROC AUC of a default LightGBM model under stratified 5-fold CV."""
    import lightgbm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    clf = lightgbm.LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1)
    p = cross_val_predict(clf, X, target, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return roc_auc_score(target, p)


def main():
    import lightgbm

    os.makedirs(OUT, exist_ok=True)
    A = arrays()
    ids = A.sample_id
    is_hold = (A.split == "val").astype(int)
    rows, top = [], None
    for v, tabs in VIEWS.items():
        X = feature_matrix(tabs, ids)
        aucs = [cv_auc(X.values, is_hold, seed) for seed in range(5)]
        rows.append(dict(view=v, auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs, ddof=1)), n_train=600,
                         n_holdout=300, n_features=X.shape[1]))
        if v == "all":
            clf = lightgbm.LGBMClassifier(n_jobs=1, random_state=0, verbose=-1, importance_type="gain")
            clf.fit(X.values, is_hold)
            top = (pd.DataFrame(dict(feature=X.columns, gain=clf.feature_importances_))
                   .sort_values("gain", ascending=False).head(15))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "adversarial.csv"), index=False)
    top.to_csv(os.path.join(OUT, "adversarial_top.csv"), index=False)

    tr = A.tr
    b = (A.hs_nbands_raw[tr] == 126).astype(int)
    X = feature_matrix(VIEWS["all"], ids[tr]).values
    aucs = [cv_auc(X, b, seed) for seed in range(5)]
    pd.DataFrame([dict(task="125_vs_126_band_files_train", auc_mean=float(np.mean(aucs)),
                       auc_sd=float(np.std(aucs, ddof=1)), n_126=int(b.sum()), n_125=int(len(b) - b.sum()))]
                 ).to_csv(os.path.join(OUT, "batch_adversarial.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    print(f"125 versus 126 band files: AUC {np.mean(aucs):.3f}")


if __name__ == "__main__":
    main()
