"""Keys, the paper's submissions, loaders, and the paired test, shared by scripts/analysis, scripts/tables, and
scripts/plots.

A key names one method as '<experiment>:<variant>', for example E4:ms_eng (FLAML on the engineered features of the
MS view) or E3t:tabpfn8_ms_f1_d0 (TabPFN with eight ensemble members on the same features, which is the full-size
point of its learning curve). Each experiment folder under results/ holds fold_metrics.csv, one row per outer fold,
and oof.csv.gz, the class probabilities that each outer fold's model gave its validation patches.
"""
import os

import pandas as pd

from wheatrust.io_data import RESULTS
from wheatrust.stats import corrected_ttest, correlated_bayes

ANALYSIS = os.path.join(RESULTS, "analysis")
LEADERBOARD = os.path.join(RESULTS, "leaderboard")

# The experiments whose per-fold results are shipped, each over 10 repeats of 5-fold cross-validation
EXPERIMENTS = {
    "E1": "FLAML baseline on all views",
    "E2": "TabPFN on all views with its automatic ensemble size, the procedure of submission S05",
    "E3f": "FLAML on three stratified draws of three quarters of each outer training fold, four views",
    "E3t": "TabPFN with eight ensemble members on the four views",
    "E4": "FLAML on single views and representations, with the single-source and stretch controls",
    "E4t": "TabPFN on the single-source and stretch controls",
    "E5": "FLAML feature-family and dimensionality ablations",
    "E9": "FLAML on band budgets selected in fold and on fixed band placements",
    "E9t": "TabPFN on the same band sets",
}

# Outer training and validation sizes (600 training patches, five folds). The corrected resampled t-test needs them.
N_TRAIN, N_TEST = 480, 120

BASE = "E1:flaml_all"  # FLAML on all views, the reference of every comparison table
FLAML_VIEW = {"rgb": "E4:rgb_eng", "ms": "E4:ms_eng", "hsc": "E4:hsc_eng", "all": BASE}
TABPFN_VIEW = {v: f"E3t:tabpfn8_{v}_f1_d0" for v in ("rgb", "ms", "hsc", "all")}

# The seven holdout submissions the paper reports, out of the ten the study made, each with the cross-validated
# procedure it refits on all training patches. S05 refits E2:tabpfn_all, whose automatic ensemble size resolves to
# the same eight members as the TabPFN runs of the paper.
SUBMISSIONS = {"S01": "E1:flaml_all", "S02": "E4:rgb_eng", "S03": "E4:ms_eng", "S04": "E4:hsc_eng",
               "S05": "E2:tabpfn_all", "S07": "E9:bandsel_k5", "S09": "E4:rgbhs_eng"}
SUBMISSIONS_FLAML = [s for s in SUBMISSIONS if s != "S05"]

NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
           10: "ten"}


def analysis_file(name):
    return os.path.join(ANALYSIS, name)


def load_script(name):
    """Import a numbered analysis script, for example load_script("04_contrasts"), so that a later analysis reuses
    its definitions (contrast lists, per-fold helpers) instead of copying them. Its main() does not run."""
    import importlib

    return importlib.import_module(f"scripts.analysis.{name}")


def read_experiment(exp, what="fold_metrics"):
    """fold_metrics.csv or oof.csv.gz of one experiment, with the key in the first column."""
    name = "fold_metrics.csv" if what == "fold_metrics" else "oof.csv.gz"
    df = pd.read_csv(os.path.join(RESULTS, exp, name), low_memory=False)
    df.insert(0, "key", exp + ":" + df.variant)
    return df


def all_fold_metrics(columns=None):
    """Every fold of every method (written by 01_summaries.py)."""
    return pd.read_csv(analysis_file("all_fold_metrics.csv"), usecols=columns, low_memory=False)


def paired_test(x, y):
    """Compare two methods on the outer folds they share (x and y aligned fold by fold).

    Outer folds of repeated cross-validation share most of their training data, so their scores are correlated and
    a plain paired t-test would be overconfident. The Bayesian correlated t-test (Corani and Benavoli) widens the
    posterior of the mean difference with the correlation rho = 1/5, the share of the data held out per fold, and
    returns the probabilities that x is better by more than the region of practical equivalence (ROPE, plus or
    minus 0.02), equivalent within it, or worse. The corrected resampled t-test of Nadeau and Bengio applies the
    same correction, 1/J + n_test/n_train, to give a two-sided p-value for Holm's adjustment.
    """
    pb, pe, pw = correlated_bayes(x, y)
    _, p = corrected_ttest(x, y, N_TRAIN, N_TEST)
    return pb, pe, pw, p
