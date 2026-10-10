"""Write outputs/numbers.tex, every number the paper quotes, as LaTeX macros.

Two kinds of macros are written.
* Named macros for facts of the data, the design, and single results, for example \\NTrain.
* Keyed results, read in the text with \\res{key}{field}, for example \\res{E4:ms_eng}{macro_f1_mean}. A key that
  is missing prints a bold ??key/field?? marker, so nothing disappears silently. A key is either a method key or a
  prefix naming the analysis, for example pw: and spw: for paired contrasts, hr: and hrveg: for the Health versus
  Rust AUC, hrwin: for the window permutation, selfreq: and selbin: for band selection, and kaggle: for the
  leaderboard.
Every value is read from files under results/ (results/analysis/ is written by scripts/analysis), and the package
versions come from requirements.txt. The file also defines the switch \\ifkagglescores, true once leaderboard scores
are recorded, which the manuscript uses to choose between its two versions of the holdout passages.
usage: python scripts/tables/01_numbers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import importlib.metadata
import json
import math
import os

import numpy as np
import pandas as pd

from scripts.analysis._shared import (ANALYSIS, BASE, LEADERBOARD, N_TEST, N_TRAIN, NUMWORD, SUBMISSIONS,
                                      SUBMISSIONS_FLAML, all_fold_metrics, analysis_file)
from scripts.tables._latex import f2, f3, pval, prob, prob_fine, signed, verdict
from wheatrust.experiments import BUDGET, PCA_HSC
from wheatrust.io_data import CLEAN_IDX, HS_WL, OUTPUTS, RESULTS, ROOT
from wheatrust.stats import ROPE, holm

L = ["% Written by scripts/tables/01_numbers.py from results/. Rebuild instead of editing.",
     "\\makeatletter",
     "\\providecommand{\\res}[2]{\\ifcsname res@#1@#2\\endcsname\\csname res@#1@#2\\endcsname"
     "\\else\\textbf{??\\detokenize{#1/#2}??}\\fi}",
     "\\makeatother"]
NAMED = {}
NA = "n/a"

WINDOW_NAMES = {"490-530": "blue-green", "530-570": "green", "570-610": "yellow", "610-650": "orange",
                "650-690": "red", "690-730": "lower red-edge", "730-770": "upper red-edge", "770-810": "NIR-shoulder",
                "810-850": "NIR", "850-890": "far NIR", "482": "blue", "550": "green", "650": "red",
                "738": "red-edge", "834": "NIR"}
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
           9: "ninth", 10: "tenth"}
# the platform labels of the fold metrics, as the paper describes them
PLATFORM_NAMES = {"linux-aarch64": "an arm64 Linux sandbox", "macos-arm64": "an Apple-silicon laptop under macOS",
                  "linux-x86_64": "an x86-64 cloud workspace"}
PREFIX = {"hr_all": "shr", "hr_vegetated": "shrv"}  # key prefix of a contrast family, spw: for the macro-F1 ones
ESTIMATOR_NAMES = {"lrl1": "L1-regularised logistic regression", "lgbm": "LightGBM", "xgboost": "XGBoost",
                   "xgb_limitdepth": "depth-limited XGBoost", "rf": "random forest", "extra_tree": "extra trees",
                   "sgd": "SGD"}


def name(n, v):
    assert n.isalpha(), n
    NAMED[n] = v
    L.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def res(key, field, v):
    L.append(f"\\expandafter\\def\\csname res@{key}@{field}\\endcsname{{{v}}}")


def fmt_field(col, v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return NA
    if col.startswith("n_") or col in ("n", "n_units", "n_paired"):
        return f"{int(round(v))}"
    if col.startswith("p_") and col not in ("p_corrected", "p_holm"):
        return prob(v, NA)
    if col in ("p_corrected", "p_holm"):
        return pval(v, NA)
    if col == "diff":
        return signed(v, missing=NA)
    return f3(v, NA)


def table_rows(path, keycol="variant"):
    """Every column of a summary table as a keyed macro of its row."""
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        for c in df.columns:
            if c == keycol:
                continue
            v = r[c]
            if isinstance(v, (np.floating, float, np.integer, int)):
                v = float(v)
            res(r[keycol], c, fmt_field(c, v))
            if c in ("macro_f1_mean", "accuracy_mean", "diff") and isinstance(v, float) and math.isfinite(v):
                res(r[keycol], c + "_pct", f"{100 * v:.1f}")
            if c == "diff" and isinstance(v, float) and math.isfinite(v):
                res(r[keycol], "absdiff", f3(abs(v), NA))


def wname(w):
    """A window such as '490-530' as the text writes it, '490 to 530'."""
    return str(w).replace("-", " to ")


def wlist(items):
    items = [wname(w) for w in items]
    return items[0] if len(items) == 1 else (" and ".join(items) if len(items) == 2
                                             else ", ".join(items[:-1]) + ", and " + items[-1])


def audit():
    S = json.load(open(os.path.join(RESULTS, "audit", "audit_summary.json")))
    name("NTrain", S["n_train"])
    name("NHoldout", S["n_holdout"])
    name("NPerClass", S["class_counts"]["Health"])
    name("NEmptyTrain", sum(S["empty_train_by_class"].values()))
    name("NEmptyHealth", S["empty_train_by_class"].get("Health", 0))
    name("NEmptyOther", S["empty_train_by_class"].get("Other", 0))
    name("NPartialTrain", sum(S["partial_train_by_class"].values()))
    name("NEmptyHoldout", S["empty_holdout"])
    name("NPartialHoldout", S["partial_holdout"])
    name("NHSOneTwentySixTrain", S["hs_bands_raw_counts"]["train_126"])
    name("NHSOneTwentySixPerClass", S["hs126_by_class"]["Health"])
    name("NHSOneTwentySixHoldout", S["hs_bands_raw_counts"]["val_126"])
    d = S["duplicates"]["shared_4x4_blocks"]
    name("NOverlapPairs", d["train_train"] + d["train_holdout"] + d["holdout_holdout"])
    name("NOverlapTrainTrain", d["train_train"])
    name("NOverlapTrainHoldout", d["train_holdout"])
    name("NOverlapHoldoutHoldout", d["holdout_holdout"])
    # Agreement of the RGB and MS views with their best-matching HS bands. A minimum is floored and a maximum is
    # ceiled, so that the bound the text states holds.
    prov = pd.DataFrame(S["provenance"])
    ms, rgb = prov[prov.view == "MS"], prov[prov.view == "RGB"]
    name("MSRsqPooledMin", f"{np.floor(ms.pooled_r2.min() * 1000) / 1000:.3f}")
    name("MSSlopeMin", f"{ms.pooled_slope.min():.2f}")
    name("MSSlopeMax", f"{ms.pooled_slope.max():.2f}")
    name("RGBRsqPerImageMin", f"{np.floor(rgb.median_per_image_r2.min() * 1000) / 1000:.3f}")
    name("RGBRsqPooledMax", f"{np.ceil(rgb.pooled_r2.max() * 100) / 100:.2f}")
    name("RGBClipLowPct", f"{100 * S['rgb_frac_at_0_median']:.1f}")
    name("RGBClipHighPct", f"{100 * S['rgb_frac_at_255_median']:.1f}")
    name("SNRClean", f"{S['snr_median_clean']:.0f}")
    name("SNRNoisyLow", f"{S['snr_median_noisy_low']:.0f}")
    name("SNRBlue", f"{S['snr_at_482']:.0f}")
    name("SNRNoisyHigh", f"{S['snr_median_noisy_high']:.0f}")
    nd = S["ndvi_by_class"]
    name("NDVIHealthMedian", f"{nd['Health']['ndvi_median_of_means']:.2f}")
    name("NDVIRustMedian", f"{nd['Rust']['ndvi_median_of_means']:.2f}")
    name("NDVIOtherMedian", f"{nd['Other']['ndvi_median_of_means']:.2f}")
    name("OtherLowNDVIPct", f"{100 * nd['Other']['frac_patches_mean_ndvi_lt_0p5']:.0f}")
    name("HealthLowNDVIPct", f"{100 * nd['Health']['frac_patches_mean_ndvi_lt_0p5']:.0f}")
    # how well patch-level ExG and VARI separate the classes (Kruskal-Wallis epsilon squared), over four choices of
    # the blue band, from the noisy 482 nm band to cleaner ones
    bl = pd.read_csv(os.path.join(RESULTS, "audit", "blue_index_noise.csv"))
    for idx, tag in [("ExG", "Exg"), ("VARI", "Vari")]:
        b = bl[bl["index"] == idx]
        name(f"Blue{tag}EpsLo", f"{b.epsilon_squared.min():.3f}")
        name(f"Blue{tag}EpsHi", f"{b.epsilon_squared.max():.3f}")
    off = pd.read_csv(os.path.join(RESULTS, "audit", "overlap_offsets.csv"))
    name("OverlapOffsetPx", int(off.offset_px.mode().iloc[0]))
    both = off[(off.a_split == "train") & (off.b_split == "train")]
    n_mixed = int((both.a.str.split("_").str[0] != both.b.str.split("_").str[0]).sum())  # ids start with the label
    name("NOverlapMixedLabelWord", NUMWORD.get(n_mixed, str(n_mixed)) if n_mixed else "none")
    return S


def design():
    base = all_fold_metrics(["key", "repeat", "fold"])
    base = base[base.key == BASE]
    n_repeats, n_folds = base.repeat.nunique(), base.fold.nunique()
    name("NRepeats", n_repeats)
    name("NFolds", n_folds)
    name("NOuterFolds", n_repeats * n_folds)
    name("NInnerFolds", 3)  # FLAML's inner stratified cross-validation (wheatrust.learners.FlamlLearner)
    name("FLAMLBudget", BUDGET)
    name("ROPE", f"{ROPE:.2f}")
    # the PCA scores of the cleaned (and full) HS spectrum are fitted inside each fold, so the feature dictionary
    # does not list them
    n_pca = PCA_HSC["n"]
    cnt = pd.read_csv(os.path.join(RESULTS, "features", "feature_dictionary.csv")).groupby("table").size()
    name("NFeatRGB", int(cnt["rgb"]))
    name("NFeatMS", int(cnt["ms"]))
    name("NFeatHSC", int(cnt["hsc"]) + n_pca)
    name("NFeatHSF", int(cnt["hsf"]) + n_pca)
    name("NFeatCross", int(cnt["cross"]))
    name("NFeatAll", int(cnt["rgb"] + cnt["ms"] + cnt["hsc"] + cnt["cross"]) + n_pca)
    name("NBandsClean", len(CLEAN_IDX))
    name("NBandsFull", len(HS_WL))
    name("NTrainFold", N_TRAIN)
    name("NValFold", N_TEST)


def versions():
    """Pinned versions from requirements.txt; a package it does not pin falls back to the installed version."""
    req = {}
    p = os.path.join(ROOT, "requirements.txt")
    if os.path.exists(p):
        req = {l.split("==")[0].strip().lower(): l.split("==")[1].strip() for l in open(p) if "==" in l}
    for pkg, n in [("flaml", "VerFLAML"), ("scikit-learn", "VerSklearn"), ("lightgbm", "VerLightGBM"),
                   ("xgboost", "VerXGBoost"), ("torch", "VerTorch"), ("tabpfn", "VerTabPFN")]:
        v = req.get(pkg)
        if v is None:
            try:
                v = importlib.metadata.version(pkg)
                print(f"{pkg} is not pinned in requirements.txt; using the installed version {v}")
            except importlib.metadata.PackageNotFoundError:
                v = NA
        name(n, v)


def summaries():
    """Method summaries, feature counts, the baseline's model choice and errors, cross-batch costs, and the
    adversarial validation."""
    for f in ["e1_summary", "e4_views", "e5_ablation", "e9_sensors", "e9t_sensors", "e3t_views"]:
        table_rows(analysis_file(f + ".csv"))
    fm = all_fold_metrics(["key", "n_features"])
    for k, g in fm.groupby("key"):
        if g.n_features.notna().any():
            res(k, "n_features", f"{g.n_features.mean():.0f}")
            res(k, "n_features_min", f"{g.n_features.min():.0f}")
            res(k, "n_features_max", f"{g.n_features.max():.0f}")
    # engineered features minus raw band statistics, per view, on the means
    ev = pd.read_csv(analysis_file("e4_views.csv")).set_index("variant").macro_f1_mean
    pairs = [("E4:rgb_eng", "E4:rgb_raw"), ("E4:ms_eng", "E4:ms_raw"), ("E4:hsc_eng", "E4:hsc_raw"),
             ("E4:hsf_eng", "E4:hsf_raw"), (BASE, "E4:all_raw")]
    d = [ev[a] - ev[b] for a, b in pairs]
    name("EngRawDiffMin", f3(min(d), NA))
    name("EngRawDiffMax", f3(max(d), NA))
    sel = pd.read_csv(analysis_file("e1_selected_estimators.csv")).sort_values(["n_folds", "estimator"],
                                                                                ascending=[False, True])
    first = sel.estimator.iloc[0]
    res("selrank", "first", first.replace("_", "\\_"))
    res("selrank", "first_name", ESTIMATOR_NAMES.get(first, first.replace("_", "\\_")))
    res("selrank", "first_n", int(sel.n_folds.iloc[0]))
    res("selrank", "nest", len(sel))
    res("selrank", "nestword", NUMWORD.get(len(sel), str(len(sel))))
    # share of the baseline's errors that confuse Health and Rust, pooled over outer folds
    cm = pd.read_csv(analysis_file("e1_confusion_pooled.csv"), index_col=0)
    err_total = int(cm.values.sum() - np.trace(cm.values))
    hr_err = int(cm.loc["true_Health", "pred_Rust"] + cm.loc["true_Rust", "pred_Health"])
    name("ErrHRSharePct", f"{100 * hr_err / err_total:.0f}")
    xb = pd.read_csv(analysis_file("crossbatch_cost.csv"))  # over learners, inputs, and directions
    name("XBCostMin", f3(xb.cost.min(), NA))
    name("XBCostMax", f3(xb.cost.max(), NA))
    for f, keyc in [("adversarial.csv", "view"), ("batch_adversarial.csv", "task")]:
        for _, r in pd.read_csv(os.path.join(RESULTS, "E12", f)).iterrows():
            res(f"adv:{r[keyc]}", "auc_mean", f2(r.auc_mean, NA))
            res(f"adv:{r[keyc]}", "auc_sd", f2(r.auc_sd, NA))


def health_rust():
    """The Health versus Rust AUC on all and on vegetated patches, and the errors on Health patches."""
    for _, r in pd.read_csv(analysis_file("hr_metrics.csv")).iterrows():
        res(f"hr:{r.key}", "auc", f3(r.hr_auc_mean, NA))
        res(f"hr:{r.key}", "auc_sd", f3(r.hr_auc_sd, NA))
        res(f"hr:{r.key}", "f1_Health", f3(r.f1_Health_mean, NA))
        res(f"hr:{r.key}", "f1_Rust", f3(r.f1_Rust_mean, NA))
    df = pd.read_csv(analysis_file("hr_veg_metrics.csv"))
    for _, r in df.iterrows():
        res(f"hrveg:{r.key}", "auc", f3(r.hr_auc_mean, NA))
        res(f"hrveg:{r.key}", "auc_sd", f3(r.hr_auc_sd, NA))
    name("NVegHealth", int(df.n_veg_health.iloc[0]))
    name("NVegRust", int(df.n_veg_rust.iloc[0]))
    for _, r in pd.read_csv(analysis_file("health_errors.csv")).iterrows():
        res(f"err:{r.key}", "h_as_r_pct", f"{100 * r.health_as_rust:.0f}")
        res(f"err:{r.key}", "h_as_o_pct", f"{100 * r.health_as_other:.0f}")
        res(f"err:{r.key}", "recall_health", f3(r.recall_health, NA))


def contrasts():
    """Contrasts are cited under several prefixes. spw: (macro-F1), shr: (Health versus Rust AUC on all patches),
    and shrv: (on vegetated patches) carry the Holm adjustment within the paper's families. pw: gives a macro-F1
    contrast's difference and posteriors, and hrpw:<name>:hr_auc the same for an AUC contrast on all patches."""
    df = pd.read_csv(analysis_file("contrasts.csv"))
    for _, r in df.iterrows():
        if r.family in ("planned", "tabpfn", "learner", "ablation"):
            k = f"pw:{r['name']}"
            res(k, "diff", signed(r["diff"], missing=NA))
            res(k, "absdiff", f3(abs(r["diff"]), NA))
            res(k, "diff_sd", f3(r["diff_sd"], NA))
            res(k, "mean_a", f3(r["mean_a"], NA))
            res(k, "mean_b", f3(r["mean_b"], NA))
            for c in ["p_better", "p_equiv", "p_worse"]:
                res(k, c, prob(r[c], NA))
            res(k, "p_holm", pval(r["p_holm"], NA))
            res(k, "n", int(r["n_paired"]))
        if r.family == "hr_all":
            k = f"hrpw:{r['name']}:hr_auc"
            res(k, "diff", signed(r["diff"], missing=NA))
            res(k, "absdiff", f3(abs(r["diff"]), NA))
            res(k, "mean_a", f3(r["mean_a"], NA))
            res(k, "mean_b", f3(r["mean_b"], NA))
            for c in ["p_better", "p_equiv", "p_worse"]:
                res(k, c, prob(r[c], NA))
            res(k, "p_holm", pval(r["p_holm"], NA))
        if r.family == "ablation":
            continue
        k = f"{PREFIX.get(r.family, 'spw')}:{r['name']}"
        res(k, "diff", signed(r["diff"], missing=NA))
        res(k, "absdiff", f3(abs(r["diff"]), NA))
        res(k, "mean_a", f3(r.mean_a, NA))
        res(k, "mean_b", f3(r.mean_b, NA))
        for c in ["p_better", "p_equiv", "p_worse"]:
            res(k, c, prob(r[c], NA))
            res(k, c.replace("p_", "p3_"), prob_fine(r[c], NA))
        res(k, "p_corrected", pval(r.p_corrected, NA))
        res(k, "p_holm", pval(r.p_holm, NA))
        res(k, "verdict", verdict(r.p_better, r.p_equiv, r.p_worse))
        res(k, "n", int(r.n_paired))
    # TabPFN minus FLAML on the four views, same inputs and folds
    lv = df[(df.family == "learner") & df["name"].isin([f"lrn_view_{v}" for v in ("rgb", "ms", "hsc", "all")])]
    name("LrnViewDiffMin", f3(lv["diff"].min(), NA))
    name("LrnViewDiffMax", f3(lv["diff"].max(), NA))
    ab = df[df.family == "ablation"]
    name("EFiveHolmMin", pval(ab.p_holm.min(), NA))
    k5 = int((ab.p_holm < 0.05).sum())
    name("EFiveHolmConfirmedWord", "none" if k5 == 0 else NUMWORD.get(k5, str(k5)))
    name("EFiveNContrasts", len(ab))
    # How often the corrected t-test confirms, after Holm, a difference the Bayesian test labels clear or probable,
    # over every contrast of the paper's Appendix B whose largest posterior is not equivalence
    sc = df[df.family != "ablation"]
    post = sc[["p_better", "p_equiv", "p_worse"]]
    top, direction = post.max(axis=1), post.idxmax(axis=1)
    labelled = sc[(direction != "p_equiv") & (top > 0.75)]
    for tag, g in [("Clear", labelled[top[labelled.index] > 0.95]), ("Probable", labelled[top[labelled.index] <= 0.95])]:
        k = int((g.p_holm < 0.05).sum())
        name(f"NHolm{tag}", len(g))
        name(f"NHolm{tag}Confirmed", k)
        name(f"NHolm{tag}ConfirmedWord", "none" if k == 0 else str(k))


def windows():
    """The window permutation (results/E8) and the statistics of its rankings (scripts/analysis/05_window_stats.py).
    Windows are named in the text by the spectral region they cover (WINDOW_NAMES)."""
    df = pd.read_csv(os.path.join(RESULTS, "E8", "hr_window_summary.csv"))
    for _, r in df.iterrows():
        res(f"hrwin:{r.model}:{r.window}", "auc_drop", f3(r.drop_hr_auc, NA))
        res(f"hrwin:{r.model}:{r.window}", "f1_drop", f3(r.drop_macro_f1, NA))
    for mdl, g in df.groupby("model"):
        for i, (_, r) in enumerate(g.sort_values("drop_hr_auc", ascending=False).head(3).iterrows()):
            res(f"hrwintop:{mdl}", f"w{i + 1}", wname(r.window))
            res(f"hrwintop:{mdl}", f"v{i + 1}", f3(r.drop_hr_auc, NA))
            res(f"hrwintop:{mdl}", f"name{i + 1}", WINDOW_NAMES.get(str(r.window), str(r.window)))
        for i, (_, r) in enumerate(g.sort_values("drop_hr_auc", ascending=True).head(2).iterrows()):
            res(f"hrwinlow:{mdl}", f"w{i + 1}", wname(r.window))
            res(f"hrwinlow:{mdl}", f"v{i + 1}", f3(r.drop_hr_auc, NA))
        t = g.sort_values("drop_macro_f1", ascending=False).iloc[0]
        res(f"hrwintopf1:{mdl}", "w1", wname(t.window))
        res(f"hrwintopf1:{mdl}", "v1", f3(t.drop_macro_f1, NA))
        res(f"hrwintopf1:{mdl}", "name1", WINDOW_NAMES.get(str(t.window), str(t.window)))

    ws = pd.read_csv(analysis_file("hr_window_stats.csv"))
    for mdl, g in ws.groupby("model"):
        res(f"hrwinW:{mdl}", "W", f2(g.kendall_w.iloc[0], NA))
        res(f"hrwinW:{mdl}", "n", int(g.n_folds.iloc[0]))
        for _, r in g.iterrows():
            k = f"hrwinrank:{mdl}:{r.window}"
            res(k, "rank", int(r["rank"]))
            res(k, "rankword", ORDINAL.get(int(r["rank"]), str(int(r["rank"]))))
            res(k, "top3_pct", f"{100 * r.share_folds_top3:.0f}")
    # windows whose drop could not be told apart from the top window's (top window not probably larger)
    wp = pd.read_csv(analysis_file("hr_window_pairs.csv"))
    for mdl, g in wp.groupby("model"):
        rk = ws[ws.model == mdl].set_index("window")["rank"]
        tied = g[g.p_top_better <= 0.75].copy()
        tied["rank"] = tied.window.map(lambda w: rk.get(w, np.nan))
        tied = tied.sort_values("rank")
        beyond = tied[tied["rank"] > 3]
        res(f"hrwintie:{mdl}", "beyondtop3", wlist(beyond.window) if len(beyond) else "none")
        if len(beyond):
            dv = ws[ws.model == mdl].set_index("window").mean_drop_hr_auc
            vals = [f3(dv[w], NA) for w in beyond.window]
            res(f"hrwintie:{mdl}", "beyondtop3_vals",
                vals[0] if len(vals) == 1 else (" and ".join(vals) if len(vals) == 2
                                                else ", ".join(vals[:-1]) + ", and " + vals[-1]))
            res(f"hrwintie:{mdl}", "pmax_beyond", prob(beyond.p_top_better.max(), NA))
        # the probability that the top window's drop exceeds the second window's by more than the ROPE
        second = rk[rk == 2].index
        if len(second):
            q = g[g.window.astype(str) == str(second[0])]
            if len(q):
                res(f"hrwintie:{mdl}", "p2", prob(q.p_top_better.iloc[0], NA))

    # the FLAML models scored on vegetated patches
    df = pd.read_csv(os.path.join(RESULTS, "E8", "hr_window_summary_veg.csv"))
    df["window"] = df.window.astype(str)
    for mdl, g in df.groupby("model"):
        for _, r in g.iterrows():
            res(f"hrwinveg:{mdl}:{r.window}", "auc_drop", f3(r.drop_hr_auc_veg, NA))
        for i, (_, r) in enumerate(g.sort_values("drop_hr_auc_veg", ascending=False).head(3).iterrows()):
            res(f"hrwinvegtop:{mdl}", f"w{i + 1}", wname(r.window))
            res(f"hrwinvegtop:{mdl}", f"v{i + 1}", f3(r.drop_hr_auc_veg, NA))
            res(f"hrwinvegtop:{mdl}", f"name{i + 1}", WINDOW_NAMES.get(str(r.window), str(r.window)))
    sv = pd.read_csv(analysis_file("hr_window_stats_veg.csv"))
    sv["window"] = sv.window.astype(str)
    pv = pd.read_csv(analysis_file("hr_window_pairs_veg.csv"))
    pv["window"] = pv.window.astype(str)
    for mdl, g in sv.groupby("model"):
        res(f"hrwinWveg:{mdl}", "W", f2(g.kendall_w.iloc[0], NA))
        n_tied = int(((pv.model == mdl) & (pv.p_top_better <= 0.75)).sum())
        res(f"hrwintieveg:{mdl}", "nothers", n_tied)
        res(f"hrwintieveg:{mdl}", "nothersword", NUMWORD.get(n_tied, str(n_tied)) if n_tied else "none of the")


def band_selection():
    """Which wavelengths the in-fold band selection picked, singly and in 20 nm bins.

    Many wavelengths share a selection frequency, so every ranking breaks ties by wavelength (shorter first) with a
    stable sort. An unstable sort would order tied bands differently on different processors."""
    df = pd.read_csv(analysis_file("e9_selection_frequency.csv"))
    for k in [1, 2, 3, 5, 8]:
        ranked = df[df.k == k].sort_values(["frequency", "wavelength"], ascending=[False, True], kind="stable")
        top = ranked.head(6)
        res(f"selfreq:{k}", "top", ", ".join(f"{w:.0f}" for w in top.wavelength))
        for i, (_, r) in enumerate(top.iterrows()):
            res(f"selfreq:{k}", f"w{i + 1}", f"{r.wavelength:.0f}")
            res(f"selfreq:{k}", f"f{i + 1}", f"{100 * r.frequency:.0f}")
        # the wavelengths that share the highest and the second-highest frequency, so that a tie is named in full
        for g, (freq, grp) in enumerate(list(ranked.groupby("frequency", sort=False))[:2]):
            res(f"selfreq:{k}", f"g{g + 1}", wlist([f"{w:.0f}" for w in sorted(grp.wavelength)]))
            res(f"selfreq:{k}", f"g{g + 1}f", f"{100 * freq:.0f}")
            res(f"selfreq:{k}", f"g{g + 1}n", len(grp))
    # selected bands per fold in 20 nm bins (Figure 4b), ranked, for the larger budgets
    edges = np.arange(490, 891, 20)
    for k in [5, 8]:
        f = df[df.k.astype(float) == k]
        idx = np.clip(np.digitize(f.wavelength.to_numpy(float), edges) - 1, 0, len(edges) - 2)
        per_bin = pd.Series(np.bincount(idx, weights=f.frequency.to_numpy(float), minlength=len(edges) - 1),
                            index=[f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)])
        top = per_bin.sort_values(ascending=False, kind="stable").head(4)  # ties keep the shorter bin first
        for i, (b, v) in enumerate(top.items()):
            res(f"selbin:{k}", f"b{i + 1}", wname(b))
            res(f"selbin:{k}", f"v{i + 1}", f"{v:.2f}")
        res(f"selbin:{k}", "list4", wlist(list(top.index)))
        # the most frequent bin within the blue-green edge and within the red edge, and its rank among all bins
        order = list(per_bin.sort_values(ascending=False, kind="stable").index)
        for reg, lo, hi in [("bluegreen", 490, 530), ("rededge", 690, 770)]:
            sub = per_bin[[b for b in per_bin.index if lo <= int(b.split("-")[0]) < hi]]
            b = sub.idxmax()
            res(f"selregion:{k}:{reg}", "bin", wname(b))
            res(f"selregion:{k}:{reg}", "per_fold", f"{sub.max():.2f}")
            res(f"selregion:{k}:{reg}", "rank", order.index(b) + 1)
    # the 20 nm bin that held the first selected band most often
    b1 = pd.read_csv(analysis_file("e9_selections_by_fold.csv")).band1.to_numpy(float)
    cnt = np.bincount(np.clip(np.digitize(b1, edges) - 1, 0, len(edges) - 2), minlength=len(edges) - 1)
    j = int(cnt.argmax())
    res("sel1bin", "bin", f"{edges[j]} to {edges[j + 1]}")
    res("sel1bin", "pct", f"{100 * cnt[j] / len(b1):.0f}")
    res("sel1bin", "n", len(b1))


def platform_desc(s):
    s = s.groupby(s.map(lambda p: PLATFORM_NAMES.get(p, str(p)))).size().sort_values(ascending=False, kind="stable")
    if len(s) == 1:
        return s.index[0]
    parts = [f"{n} ({v} units)" for n, v in s.items()]
    return " and ".join(parts) if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1]


def platforms():
    """Where the units of the paper ran, as one sentence. The post hoc controls are the stretch control
    (rgbhsst) and eight bands without pairwise differences (_nond)."""
    fp = all_fold_metrics(["key", "platform"])
    ctrl = fp.key.str.contains("rgbhsst|_nond")
    isf = fp.key.str.match(r"^(E1:flaml_all|E4:|E5:|E9:)")
    # E2:tabpfn_all is the TabPFN procedure that submission S05 refits
    ist = fp.key.str.match(r"^(E2:tabpfn_all|E3t:tabpfn8_(rgb|ms|hsc|all)_f1_d0|E4t:|E9t:)")
    parts = [f"The FLAML units of this paper ran on {platform_desc(fp[isf & ~ctrl].platform)}",
             f"the TabPFN units on {platform_desc(fp[ist & ~ctrl].platform)}",
             f"the post hoc controls on {platform_desc(fp[(isf | ist) & ctrl].platform)}"]
    name("PlatformSentence", ", ".join(parts[:-1]) + ", and " + parts[-1] + ".")
    rp = json.load(open(analysis_file("platform_check.json")))
    for lrn, tag in [("flaml", "Flaml"), ("tabpfn", "Tabpfn")]:
        name(f"Repro{tag}MeanDiff", f3(rp[lrn]["abs_mean_diff"], NA))
        name(f"Repro{tag}MaxFoldDiff", f3(rp[lrn]["max_fold_abs_diff"], NA))
        name(f"Repro{tag}DiffFolds", int(rp[lrn]["n_folds_different_predictions"]))
        name(f"Repro{tag}NFolds", int(rp[lrn]["n_folds"]))
    rx = json.load(open(analysis_file("platform_controls.json")))
    name("ReproXMaxMeanDiff", f3(max(v["abs_mean_diff"] for v in rx["pairs"].values()), NA))
    name("ReproXNUnits", sum(v["n_folds"] for v in rx["pairs"].values()))
    for n, v in rx["controls"].items():
        k = f"xctl:{n}"
        res(k, "diff", signed(v["diff"], missing=NA))
        res(k, "absdiff", f3(abs(v["diff"]), NA))
        for c in ["p_better", "p_equiv", "p_worse"]:
            res(k, c, prob(v[c], NA))
        res(k, "verdict", verdict(v["p_better"], v["p_equiv"], v["p_worse"]))


def leaderboard(S):
    """The holdout submissions and their leaderboard scores (scripts/analysis/08_leaderboard.py)."""
    meta = json.load(open(os.path.join(LEADERBOARD, "meta.json")))
    kv = pd.read_csv(analysis_file("leaderboard_vs_cv.csv"))
    for sid in SUBMISSIONS:
        row = kv[kv.sid == sid].iloc[0]
        for c in ["public", "private"]:
            if pd.notna(row[c]):
                res(f"kaggle:{sid}", c, f"{float(row[c]):.3f}")
        res(f"kaggle:{sid}", "lo95", f"{row.lo95:.3f}")
        res(f"kaggle:{sid}", "hi95", f"{row.hi95:.3f}")
        if pd.notna(row.private):
            res(f"kaggle:{sid}", "diff", signed(row.private_minus_cv, missing=NA))
    has_scores = bool(kv["private"].notna().any())
    L.append("\\newif\\ifkagglescores")
    L.append("\\kagglescorestrue" if has_scores else "\\kagglescoresfalse")
    name("HoldoutHeading", "Holdout Scores Confirm the Deficit of the RGB Product" if has_scores
         else "Holdout Submissions")
    name("KaggleMetric", str(kv.metric.iloc[0]).replace("_", " "))
    name("KagglePrivateN", int(kv.n_private.iloc[0]))
    name("KagglePublicN", meta["n_public"])
    name("NRefitExtra", S["n_train"] - N_TRAIN)  # how many more training patches a refit sees than an outer fold
    name("NSubmissionsWord", NUMWORD[meta["n_submissions"]])
    name("NSubmissionsAWord", NUMWORD[len(SUBMISSIONS)])
    name("NSubmissionsOther", NUMWORD[meta["n_submissions"] - len(SUBMISSIONS)])
    if not has_scores:
        return
    # rank agreement of CV accuracy and private score over the six FLAML submissions
    tau = pd.read_csv(analysis_file("leaderboard_tau.csv"))
    t = tau[(tau.score == "private") & (tau.set == "paper_flaml")].iloc[0]
    name("KendallTauAF", f"{t.tau:.2f}")
    name("KendallPAF", pval(t.p, NA))
    ka = kv[kv.sid.isin(list(SUBMISSIONS))].dropna(subset=["private"])
    n_in = int(ka.private_in_range.astype(bool).sum())
    name("KaggleInRangeWordA", f"All {NUMWORD[len(ka)]}" if n_in == len(ka)
         else f"{NUMWORD.get(n_in, str(n_in)).capitalize()} of the {NUMWORD[len(ka)]}")
    kf = kv[kv.sid.isin(SUBMISSIONS_FLAML)].dropna(subset=["private"])
    name("KaggleDiffMinAF", signed(kf.private_minus_cv.min(), missing=NA))
    name("KaggleDiffMaxAF", signed(kf.private_minus_cv.max(), missing=NA))
    # RGB features from the HS bands (S09) against the RGB PNG view (S02), on the private split and in CV accuracy
    kvi = kv.set_index("sid")
    cvm = pd.read_csv(analysis_file("leaderboard_cv_map.csv")).set_index("sid")
    name("KaggleRGBGap", f3(kvi.loc["S09", "private"] - kvi.loc["S02", "private"], NA))
    name("KaggleRGBGapCV", f3(cvm.loc["S09", "cv_acc"] - cvm.loc["S02", "cv_acc"], NA))
    oth = ka[ka.sid != "S02"]  # the paper's six submissions other than the RGB PNG view
    name("KaggleOtherMin", f3(oth.private.min(), NA))
    name("KaggleOtherMax", f3(oth.private.max(), NA))
    # S02 against each other submission over all holdout patches, with the sign-test bounds adjusted by Holm
    kp = pd.read_csv(analysis_file("leaderboard_pairs.csv"))
    rg = kp[((kp.a == "S02") & kp.b.isin(list(SUBMISSIONS))) | ((kp.b == "S02") & kp.a.isin(list(SUBMISSIONS)))]
    deficit = np.where(rg.a == "S02", -rg.d_all, rg.d_all)  # correct patches of the other minus those of S02
    adj = holm(rg.p_bound_all.values)
    name("KaggleRGBMinMore", int(deficit.min()))
    name("KaggleRGBMaxMore", int(deficit.max()))
    name("KaggleRGBNPairs", NUMWORD[len(rg)])
    name("KaggleRGBSignP", "$p<0.001$" if adj.max() < 0.001 else f"$p\\leq{adj.max():.3f}$")
    # the largest accuracy gain from 360 to 480 training patches, over the four views (scripts/analysis/02)
    g = pd.read_csv(analysis_file("refit_gain.csv"))
    name("SlopeFlamlAccMax", signed(g.accuracy_gain.max(), missing=NA))


def main():
    S = audit()
    design()
    versions()
    summaries()
    health_rust()
    contrasts()
    windows()
    band_selection()
    platforms()
    leaderboard(S)
    os.makedirs(OUTPUTS, exist_ok=True)
    out = os.path.join(OUTPUTS, "numbers.tex")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    n_keyed = sum(1 for line in L if "csname res@" in line)
    print(f"wrote {os.path.relpath(out)} with {len(NAMED)} named macros and {n_keyed} keyed values "
          f"from {os.path.relpath(ANALYSIS)}")


if __name__ == "__main__":
    main()
