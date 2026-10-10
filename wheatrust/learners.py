"""The two learners of the study, FLAML and TabPFN.

Both expose fit(X, y), predict_proba(X), and describe(). Everything a learner fits (standardisation,
hyperparameters, thresholds) is learned from the data passed to fit, which is always an outer training fold or, for
the submissions, the full training set.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from wheatrust.common import tune_thresholds
from wheatrust.io_data import MODELS

warnings.filterwarnings("ignore")
FLAML_DEFAULT = "auto"


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _proba_from(est, X):
    """Class probabilities from a fitted estimator, or a softmax of its decision function if it has none."""
    if hasattr(est, "predict_proba"):
        try:
            p = est.predict_proba(X)
            if p is not None:
                return np.asarray(p)
        except Exception:
            pass
    d = est.decision_function(X)
    return _softmax(np.asarray(d))


# ---------------------------------------------------------------------------
# FLAML
# ---------------------------------------------------------------------------
class FlamlLearner:
    """Nested FLAML search on standardised inputs.

    The search maximises macro-F1 under stratified inner cross-validation with a fixed number of iterations
    (budget) and no time limit, so that it does not depend on machine speed, and refits the selected configuration
    on all the data given. With tune_thr, the selected configuration is then fitted on a stratified split of the
    same data into inner_splits folds, and per-class thresholds are tuned on the resulting out-of-fold
    probabilities. The thresholds are recorded, but every reported score uses the plain argmax.
    """

    def __init__(self, estimator_list=FLAML_DEFAULT, budget=100, seed=0, tune_thr=True, inner_splits=3):
        self.estimator_list = estimator_list
        self.budget = budget
        self.seed = seed
        self.tune_thr = tune_thr
        self.inner_splits = inner_splits

    def fit(self, X, y):
        from flaml import AutoML

        t0 = time.time()
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.automl = AutoML()
        self.automl.fit(
            Xs, y, task="classification", metric="macro_f1", estimator_list=self.estimator_list,
            max_iter=self.budget, time_budget=-1, eval_method="cv", n_splits=self.inner_splits,
            split_type="stratified", seed=self.seed, n_jobs=1, verbose=0, retrain_full=True,
            sample=False, auto_augment=False, log_file_name="", model_history=False,
            keep_search_state=False, early_stop=False,
        )
        self.fit_time = time.time() - t0
        self.inner_best = 1.0 - float(self.automl.best_loss)
        self.thresholds = np.ones(3)
        self.inner_oof = None
        if self.tune_thr:
            self.inner_oof = self._inner_oof(Xs, y)
            self.thresholds, self.inner_thr_score = tune_thresholds(y, self.inner_oof)
        return self

    def _config(self):
        cfg = dict(self.automl.best_config)
        return {k: v for k, v in cfg.items() if not k.startswith("FLAML_") and k != "learner"}

    def _inner_oof(self, Xs, y):
        cls = self.automl._state.learner_classes[self.automl.best_estimator]
        oof = np.zeros((len(y), 3))
        skf = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(Xs, y):
            est = cls(task="classification", n_jobs=1, **self._config())
            est.fit(Xs[tr], y[tr])
            oof[va] = _proba_from(est, Xs[va])
        return oof

    def predict_proba(self, X):
        p = self.automl.predict_proba(self.scaler.transform(X))
        if p is None:
            p = _proba_from(self.automl.model.estimator, self.scaler.transform(X))
        return np.asarray(p)

    def describe(self):
        return dict(
            selected_estimator=self.automl.best_estimator,
            best_config=str(self._config()),
            inner_best_macro_f1=self.inner_best,
            thresholds=list(map(float, self.thresholds)),
            fit_time_s=self.fit_time,
        )


# ---------------------------------------------------------------------------
# TabPFN v2
# ---------------------------------------------------------------------------
# the default TabPFN v2 classifier checkpoint (Hugging Face repository Prior-Labs/TabPFN-v2-clf)
TABPFN_CHECKPOINT = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"


def tabpfn_weights():
    """Path of the TabPFN v2 checkpoint under MODELS (searched recursively), or None if it is missing."""
    if not os.path.isdir(MODELS):
        return None
    for root, _, files in sorted(os.walk(MODELS)):
        if TABPFN_CHECKPOINT in files:
            return os.path.join(root, TABPFN_CHECKPOINT)
    return None


class TabPFNLearner:
    """TabPFN v2 from the local checkpoint, on the CPU, without tuning.

    The pretraining-limit check is disabled because the all-views input has more features than TabPFN was designed
    for. n_estimators is the ensemble size, either a number or 'auto' for the library default.
    """

    def __init__(self, seed=0, n_estimators="auto"):
        self.seed = seed
        self.n_estimators = n_estimators

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        t0 = time.time()
        w = tabpfn_weights()
        if w is None:
            raise RuntimeError(f"{TABPFN_CHECKPOINT} not found under {MODELS} (see WHEATRUST_MODELS)")
        self.model = TabPFNClassifier(model_path=w, device="cpu", random_state=self.seed,
                                      n_estimators=self.n_estimators, ignore_pretraining_limits=True)
        self.model.fit(X, y)
        self.fit_time = time.time() - t0
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def describe(self):
        return dict(selected_estimator="tabpfn_v2", best_config=f"n_estimators={self.n_estimators}",
                    fit_time_s=self.fit_time)
