"""Check the verbal statements of the paper against the numbers they summarise.

The Results call a difference clear when its largest posterior probability exceeds 0.95 and probable when it exceeds
0.75; below that it is inconclusive, or leaning when the stated direction still has the largest posterior. Each
entry of CLAIMS names a contrast of results/analysis/contrasts.csv, the direction the text states, and the strength
the text uses. The script recomputes the strength from the posteriors and prints MISMATCH for every claim whose
wording no longer fits. It also checks the Holm statements, statements about the two learners and about
wavelengths, and the statements of the holdout paragraph, each as a condition on the files under results/.

The manuscript is not part of this repository. Given its folder with --manuscript (main.tex and sections/*.tex), the
script also reports claims whose keys the text no longer cites (NOT FOUND), posteriors printed with two decimals
next to a label threshold (NEAR THRESHOLD), and holdout macros missing from the text.
Run it after scripts/tables/01_numbers.py, whose output it reads. The exit status is 1 when a check fails and
--strict is given.
usage: python scripts/tables/03_check_statements.py [--manuscript DIR] [--strict]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from scripts.analysis._shared import LEADERBOARD, SUBMISSIONS, SUBMISSIONS_FLAML, all_fold_metrics, analysis_file
from wheatrust.io_data import RESULTS
from wheatrust.stats import ROPE, holm, verdict

# (family, name, direction, strength, where). direction is better, equivalent, or worse. strength is clear or
# probable (the stated direction is the largest posterior and exceeds 0.95 or 0.75), leaning (the stated direction is
# the largest posterior, at most 0.75), weak (no posterior above 0.75, written as inconclusive), notclear (the text
# only says the evidence is short of clear), or notbetter (A is not shown better).
CLAIMS = [
    ("planned", "stretch", "better", "clear", "Results Q1, Discussion, abstract"),
    ("planned", "resample", "equivalent", "leaning", "Results Q1 (leaning towards equivalence)"),
    ("planned", "ms_vs_rgbhs", "better", "leaning", "Results Q1, Discussion (leaning better)"),
    ("planned", "hs_vs_ms", "better", "weak", "Results Q1, Discussion (posterior split)"),
    ("planned", "clean", "equivalent", "leaning", "Results Q1 (leaning equivalent)"),
    ("planned", "eng_vs_raw", "better", "clear", "Results Q1"),
    ("planned", "eng_vs_pca", "better", "probable", "Results Q1"),
    ("planned", "ms5_vs_rgb3", "better", "probable", "Results Q2, Discussion"),
    ("planned", "k3_vs_rgb3", "equivalent", "leaning", "Results Q2 (leaning equivalent)"),
    ("planned", "k5_vs_ms5", "worse", "leaning", "Results Q2 (leaned worse)"),
    ("tabpfn", "t_stretch", "better", "clear", "Results Q1, Discussion"),
    ("tabpfn", "t_resample", "equivalent", "clear", "Results Q1 (clearly equivalent)"),
    ("tabpfn", "t_ms_vs_rgbhs", "better", "clear", "Results Q1, Discussion"),
    ("tabpfn", "t_ms5_vs_rgb3", "better", "clear", "Results Q2, Discussion"),
    ("tabpfn", "t_k3_vs_rgb3", "equivalent", "leaning", "Results Q2 (leaned equivalent)"),
    ("tabpfn", "t_k5_vs_ms5", "worse", "probable", "Results Q2"),
    ("tabpfn", "t_k32_vs_ms5", "equivalent", "leaning", "Results Q2 (leaning equivalent)"),
    ("hr_all", "hs_vs_ms", "better", "probable", "Results Q1"),
    ("hr_all", "t_hs_vs_ms", "better", "probable", "Results Q1, Discussion"),
    ("hr_all", "all_vs_hsc", "better", "notbetter", "Results Q1 (did not help)"),
    ("hr_vegetated", "hs_vs_ms", "better", "probable", "Results Q1, Discussion, abstract"),
    ("hr_vegetated", "t_hs_vs_ms", "better", "probable", "Results Q1, Discussion"),
    ("learner", "lrn_view_rgb", "better", "leaning", "Results Q1, Discussion (leaning for RGB)"),
    ("learner", "lrn_view_ms", "better", "leaning", "Results Q1, Discussion (leaning for MS)"),
    ("learner", "lrn_view_hsc", "better", "probable", "Results Q1"),
    ("learner", "lrn_view_all", "better", "probable", "Results Q1"),
    ("learner", "lrn_bandsel_k32", "better", "clear", "Results Q2"),
    ("interaction", "cost_minus_rededge_flaml", "better", "probable", "Discussion, abstract (probably more than)"),
    ("interaction", "cost_minus_rededge_tabpfn", "equivalent", "weak",
     "Discussion (could not resolve), Conclusions (of unresolved size)"),
    ("interaction", "rededge_gain_tabpfn_minus_flaml", "better", "probable", "Discussion, conclusions"),
]
# claims about the post hoc controls
CONTROL_CLAIMS = [
    ("controls", "st_vs_hs", "worse", "clear", "Results Q1, Discussion (stretch reproduces the loss)"),
    ("controls", "png_vs_st", "better", "weak", "Results Q1 (inconclusive remainder)"),
    ("controls", "t_st_vs_hs", "worse", "clear", "Results Q1 (TabPFN, same picture)"),
    ("controls", "t_png_vs_st", "better", "weak", "Results Q1 (TabPFN, same picture)"),
    ("controls", "k8nond_vs_k8", "worse", "probable", "Results Q2"),
    ("controls", "t_k8nond_vs_k8", "worse", "leaning", "Results Q2"),
    ("controls", "hs_vs_ms_raw", "equivalent", "leaning", "Discussion (did not outperform, leaning equivalent)"),
    ("hr_all", "hs_vs_ms_raw", "better", "leaning", "Discussion (leaning better)"),
    ("hr_all", "ms5_vs_rgb3", "better", "clear", "Results Q2 (gap on the disease decision)"),
    ("hr_all", "t_ms5_vs_rgb3", "better", "clear", "Results Q2"),
]
# Holm-adjusted claims of the Results opening paragraph, as (prefix, name, confirmed after Holm)
HOLM_CLAIMS = [("spw", "stretch", True), ("spw", "t_stretch", True), ("spw", "t_ms_vs_rgbhs", True),
               ("spw", "eng_vs_raw", True), ("shr", "ms5_vs_rgb3", False), ("shr", "t_ms5_vs_rgb3", True),
               ("shrv", "hs_vs_ms", False), ("shrv", "t_hs_vs_ms", False), ("spw", "ms5_vs_rgb3", False),
               ("spw", "t_ms5_vs_rgb3", True), ("spw", "cost_minus_rededge_flaml", False)]
# A FLAML contrast and its TabPFN counterpart. Wherever either is probable or clear, the other must point the same way.
LEARNER_PAIRS = [("planned", "stretch", "tabpfn", "t_stretch"), ("planned", "resample", "tabpfn", "t_resample"),
                 ("planned", "ms_vs_rgbhs", "tabpfn", "t_ms_vs_rgbhs"), ("planned", "hs_vs_ms", "tabpfn", "t_hs_vs_ms"),
                 ("planned", "ms5_vs_rgb3", "tabpfn", "t_ms5_vs_rgb3"), ("planned", "k3_vs_rgb3", "tabpfn", "t_k3_vs_rgb3"),
                 ("planned", "k5_vs_ms5", "tabpfn", "t_k5_vs_ms5"), ("controls", "st_vs_hs", "controls", "t_st_vs_hs"),
                 ("controls", "png_vs_st", "controls", "t_png_vs_st"),
                 ("controls", "k8nond_vs_k8", "controls", "t_k8nond_vs_k8")]
for _sub in ("hr_all", "hr_vegetated"):
    LEARNER_PAIRS += [(_sub, n, _sub, "t_" + n) for n in ("hs_vs_ms", "ms_vs_rgbhs", "stretch", "ms5_vs_rgb3", "k5_vs_ms5")]
PREFIX = {"hr_all": "shr", "hr_vegetated": "shrv"}


def strength(p):
    return "clear" if p > 0.95 else "probable" if p > 0.75 else "weak"


def means():
    fm = all_fold_metrics(["key", "macro_f1", "accuracy"])
    return fm.groupby("key")[["macro_f1", "accuracy"]].mean()


def learner_claims(df):
    """'Both learners ranked the views and band placements in the same order' and related statements."""
    out = []
    m = means().macro_f1
    flaml = ["E4:rgb_eng", "E4:rgbhs_eng", "E4:ms_eng", "E4:hsc_eng", "E1:flaml_all"]
    tab = ["E3t:tabpfn8_rgb_f1_d0", "E4t:tabpfn_rgbhs_eng", "E3t:tabpfn8_ms_f1_d0", "E3t:tabpfn8_hsc_f1_d0",
           "E3t:tabpfn8_all_f1_d0"]
    same = list(pd.Series([m[k] for k in flaml]).rank()) == list(pd.Series([m[k] for k in tab]).rank())
    same = same and ((m["E9:fixed_ms5"] > m["E9:fixed_rgb3"]) == (m["E9t:tabpfn_fixed_ms5"] > m["E9t:tabpfn_fixed_rgb3"]))
    out.append(("both learners rank the views and the band placements in the same order", same))
    eng_raw = [("E4:rgb_eng", "E4:rgb_raw"), ("E4:ms_eng", "E4:ms_raw"), ("E4:hsc_eng", "E4:hsc_raw"),
               ("E4:hsf_eng", "E4:hsf_raw"), ("E1:flaml_all", "E4:all_raw")]
    out.append(("engineered above raw band statistics in every view", all(m[a] > m[b] for a, b in eng_raw)))
    out.append(("engineered above PCA scores alone for HS",
                m["E4:hsc_eng"] > m["E4:hsc_pca"] and m["E4:hsf_eng"] > m["E4:hsf_pca"]))
    out.append(("only FLAML drops by more than the ROPE beyond eight bands",
                m["E9:bandsel_k8"] - m["E9:bandsel_k16"] > ROPE and
                max(m["E9t:tabpfn_bandsel_k8"] - m["E9t:tabpfn_bandsel_k16"],
                    m["E9t:tabpfn_bandsel_k8"] - m["E9t:tabpfn_bandsel_k32"]) < ROPE))
    out.append(("stretched HS bands at least as costly as the PNG processing, both learners",
                m["E4:rgbhsst_eng"] <= m["E4:rgb_eng"] + 0.005 and
                m["E4t:tabpfn_rgbhsst_eng"] <= m["E3t:tabpfn8_rgb_f1_d0"] + 0.005))
    views4 = [(f, t) for f, t in zip(flaml, tab) if "rgbhs" not in f]  # RGB PNG, MS, cleaned HS, all views
    out.append(("TabPFN above FLAML on each of the four engineered views", all(m[t] > m[f] for f, t in views4)))
    out.append(("eight bands without differences at the level of 16 and 32 bands (within the ROPE)",
                abs(m["E9:bandsel_k8_nond"] - m["E9:bandsel_k16"]) < ROPE
                and abs(m["E9:bandsel_k8_nond"] - m["E9:bandsel_k32"]) < ROPE))
    for fa, na, fb, nb in LEARNER_PAIRS:
        if (fa, na) not in df.index or (fb, nb) not in df.index:
            continue
        a, b = df.loc[(fa, na)], df.loc[(fb, nb)]
        if max(a.p_better, a.p_worse) > 0.75 or max(b.p_better, b.p_worse) > 0.75:
            out.append((f"{fa}/{na} and {fb}/{nb} point the same way", (a["diff"] > 0) == (b["diff"] > 0)))

    # On one platform the controls keep their labels, except the TabPFN eight-band control, whose change the
    # Software and Hardware section states.
    for n, v in json.load(open(analysis_file("platform_controls.json")))["controls"].items():
        if ("controls", n) not in df.index:
            continue
        q = df.loc[("controls", n)]
        same = verdict(q.p_better, q.p_equiv, q.p_worse) == verdict(v["p_better"], v["p_equiv"], v["p_worse"])
        if n == "t_k8nond_vs_k8":
            out.append((f"control {n} changes its label on one platform, as the text states", not same))
        else:
            out.append((f"control {n} keeps its label on one platform", same))
    return out


def wavelength_claims():
    """Statements about wavelengths in the Results and Discussion, checked against the window statistics."""
    out = []
    ws = pd.read_csv(analysis_file("hr_window_stats.csv"))
    ws["window"] = ws.window.astype(str)
    rk = {(r.model, r.window): int(r["rank"]) for _, r in ws.iterrows()}
    top = {m: g.sort_values("rank").window.iloc[0] for m, g in ws.groupby("model")}
    second = {m: g.sort_values("rank").window.iloc[1] for m, g in ws.groupby("model")}
    out.append(("NIR band mattered most with both MS learners", top["E4:ms_eng"] == "834" == top["tabpfn8:ms"]))
    out.append(("second MS band differs between learners (red vs green)", second["E4:ms_eng"] != second["tabpfn8:ms"]))
    out.append(("738 nm ranked below the NIR band with both learners",
                rk[("E4:ms_eng", "738")] > 1 and rk[("tabpfn8:ms", "738")] > 1))
    out.append(("red-edge and blue-green windows in the FLAML and TabPFN HS top three",
                {"490-530"} <= {w for (m, w), r in rk.items() if m == "E4:hsc_eng" and r <= 3}
                and any(rk[("E4:hsc_eng", w)] <= 3 for w in ("690-730", "730-770"))
                and rk[("tabpfn8:hsc", "490-530")] <= 3
                and any(rk[("tabpfn8:hsc", w)] <= 3 for w in ("690-730", "730-770"))))
    out.append(("green window outside the FLAML HS top three but inside TabPFN's",
                rk[("E4:hsc_eng", "530-570")] > 3 and rk[("tabpfn8:hsc", "530-570")] <= 3))
    s = pd.read_csv(os.path.join(RESULTS, "E8", "hr_window_summary.csv"))
    f1top = s[s.model == "E4:hsc_eng"].sort_values("drop_macro_f1", ascending=False).window.iloc[0]
    out.append(("green window led three-class macro-F1 for the FLAML HS model", str(f1top) == "530-570"))
    sv = pd.read_csv(analysis_file("hr_window_stats_veg.csv"))
    sv["window"] = sv.window.astype(str)
    rkv = {(r.model, r.window): int(r["rank"]) for _, r in sv.iterrows()}
    out.append(("vegetated: NIR band first in the FLAML MS model", rkv[("E4:ms_eng", "834")] == 1))
    allm = ws[ws.model == "E4:hsc_eng"].set_index("window").mean_drop_hr_auc
    vegm = sv[sv.model == "E4:hsc_eng"].set_index("window").mean_drop_hr_auc_veg
    shrink = (allm - vegm).sort_values(ascending=False)
    out.append(("vegetated: drops of 650-690 and 690-730 shrank most", set(shrink.index[:2]) == {"650-690", "690-730"}))
    out.append(("vegetated: windows ranked less consistently (lower Kendall W than on all patches)",
                sv[sv.model == "E4:hsc_eng"].kendall_w.iloc[0] < ws[ws.model == "E4:hsc_eng"].kendall_w.iloc[0]))
    out.append(("vegetated: lower red-edge window 690-730 lost rank",
                rkv[("E4:hsc_eng", "690-730")] > rk[("E4:hsc_eng", "690-730")]))
    pr = pd.read_csv(analysis_file("hr_window_pairs.csv"))
    pr["window"] = pr.window.astype(str)
    sec = ws[(ws.model == "E4:ms_eng") & (ws["rank"] == 2)].window.iloc[0]
    out.append(("FLAML MS: NIR could not be told apart from the second band (the red band)",
                sec == "650" and float(pr[(pr.model == "E4:ms_eng") & (pr.window == sec)].p_top_better.iloc[0]) <= 0.75))
    w = ws.groupby("model").kendall_w.first()
    out.append(("Kendall W moderate for FLAML HS (0.3 to 0.7)", 0.3 <= w["E4:hsc_eng"] <= 0.7))
    fr = pd.read_csv(analysis_file("e9_selection_frequency.csv"))
    f8 = fr[fr.k.astype(float) == 8]
    edges = np.arange(490, 891, 20)
    idx = np.clip(np.digitize(f8.wavelength.to_numpy(float), edges) - 1, 0, len(edges) - 2)
    pb = pd.Series(np.bincount(idx, weights=f8.frequency.to_numpy(float), minlength=len(edges) - 1),
                   index=edges[:-1]).sort_values(ascending=False, kind="stable")
    top4 = set(pb.index[:4])
    out.append(("band selection favoured a red-edge and a blue-green bin (both in the top four bins)",
                any(690 <= b < 770 for b in top4) and any(490 <= b < 530 for b in top4)))
    return out


def robustness_claims():
    """Statements about the robustness analyses (scripts/analysis/09 to 13 and scripts/eda/04)."""
    out = []
    rb = pd.read_csv(analysis_file("contrast_robustness.csv")).set_index(["family", "name"])

    def same_label(fam, name, label):
        r = rb.loc[(fam, name)]
        return r.verdict_r01 == r.verdict_r02 == r.verdict_r03 == label

    out.append(("the costs of the RGB processing and of the stretch stay clear at ROPE 0.01, 0.02, and 0.03",
                same_label("planned", "stretch", "better, clear") and same_label("tabpfn", "t_stretch", "better, clear")
                and same_label("controls", "st_vs_hs", "worse, clear")
                and same_label("controls", "t_st_vs_hs", "worse, clear")))
    out.append(("the vegetated HS advantage with FLAML stays probable at all three ROPEs",
                same_label("hr_vegetated", "hs_vs_ms", "better, probable")))
    out.append(("the credible intervals of cleaned HS minus MS in macro-F1 include zero with both learners",
                rb.loc[("planned", "hs_vs_ms"), "ci95_lo"] < 0 < rb.loc[("planned", "hs_vs_ms"), "ci95_hi"]
                and rb.loc[("tabpfn", "t_hs_vs_ms"), "ci95_lo"] < 0 < rb.loc[("tabpfn", "t_hs_vs_ms"), "ci95_hi"]))
    out.append(("the vegetated HS advantage has a credible interval reaching below zero with FLAML, above zero with "
                "TabPFN", rb.loc[("hr_vegetated", "hs_vs_ms"), "ci95_lo"] < 0 < rb.loc[("hr_vegetated", "t_hs_vs_ms"),
                                                                                      "ci95_lo"]))
    out.append(("the RGB processing cost probably at least matched the red-edge and NIR gain with both learners",
                all(1 - rb.loc[("interaction", n), "p_worse_r02"] > 0.75
                    for n in ("cost_minus_rededge_flaml", "cost_minus_rededge_tabpfn"))))
    nd = pd.read_csv(analysis_file("ndvi_threshold.csv"))
    hs = nd[nd.name.isin(["hs_vs_ms", "t_hs_vs_ms"])]
    out.append(("the vegetated HS advantage is at least probable at every NDVI threshold, both learners",
                bool(((hs.p_better > 0.75) & (hs.p_better >= hs.p_equiv)).all())))
    ls = pd.read_csv(analysis_file("label_shift.csv"))
    j = ls[ls.sid == "joint"].iloc[0]
    out.append(("every joint holdout prior has an interval that contains one third",
                all(j[f"prior_{c}_lo95"] <= 1 / 3 <= j[f"prior_{c}_hi95"] for c in ("Health", "Rust", "Other"))))
    out.append(("under the estimated priors every submission would score below its CV accuracy",
                bool((ls[ls.sid != "joint"].gain_from_priors < 0).all())))
    man = pd.read_csv(os.path.join(LEADERBOARD, "submission_manifest.csv"))
    out.append(("every submission of the paper predicted more Rust than Health patches",
                bool((man.n_Rust > man.n_Health).all())))
    tau = pd.read_csv(analysis_file("holdout_tau.csv")).set_index(["set", "score"])
    out.append(("without S02 the rank agreement of the FLAML submissions is not significant at 0.05",
                float(tau.loc[("paper_flaml_without_S02", "private"), "p"]) > 0.05))
    st = pd.read_csv(analysis_file("selection_stability.csv")).set_index("k")
    out.append(("band selection is unstable (Kuncheva index below 0.2 for one to eight bands)",
                bool((st.loc[[1, 2, 3, 5, 8], "kuncheva_mean"] < 0.2).all())))
    out.append(("five-band sets share bands within 8 nm more often than random sets",
                st.loc[5, "near_share_mean"] > st.loc[5, "near_share_chance"]))
    fs = json.load(open(analysis_file("file_structure.json")))
    out.append(("the HS file format is the same in all three classes at every file number",
                fs["numbers_same_format_all_classes"] == fs["numbers_per_class"]))
    out.append(("no overlapping pair of the same class has adjacent file numbers", not fs["overlap_any_adjacent"]))
    path = os.path.join(RESULTS, "audit", "band_separability.csv")
    if os.path.exists(path):
        sep = pd.read_csv(path)
        veg = sep[(sep.subset == "vegetated") & sep.in_clean_range.astype(bool)]
        bands, wins = veg[veg.kind == "band"], veg[veg.kind == "window"].set_index("name")
        hv = pd.read_csv(analysis_file("hr_veg_metrics.csv")).set_index("key").hr_auc_mean
        out.append(("no single band separates vegetated Rust from Health well (AUC within 0.1 of 0.5)",
                    bool((bands.separation < 0.1).all())))
        out.append(("the cleaned-HS models separate vegetated patches far better than any single band",
                    min(hv["E4:hsc_eng"], hv["E3t:tabpfn8_hsc_f1_d0"]) > bands.auc.max() + 0.1))
        out.append(("on vegetated patches Rust reflects more from 570 to 690 nm and less from 770 to 890 nm",
                    bool((wins.loc[["570-610", "610-650", "650-690"], "auc"] > 0.5).all()
                         and (wins.loc[["770-810", "810-850", "850-890"], "auc"] < 0.5).all())))
        allw = sep[(sep.subset == "all") & (sep.kind == "window")].sort_values("separation", ascending=False)
        out.append(("on all patches the three most separating windows are darker in Rust patches",
                    bool((allw.head(3).auc < 0.5).all())))
        hw = pd.read_csv(analysis_file("hr_window_stats.csv"))
        top2 = hw[hw.model == "E4:hsc_eng"].sort_values("rank").window.astype(str).head(2)
        out.append(("on all patches the two windows that matter most to the FLAML HS model are among the three that "
                    "separate the classes most without a model", set(top2) <= set(allw.head(3).name.astype(str))))
        nir = allw.set_index("name").loc[["770-810", "810-850", "850-890"], "auc"]
        out.append(("on all patches the NIR windows from 770 to 890 nm do not separate the classes (AUC within 0.02 "
                    "of 0.5)", bool(((nir - 0.5).abs() < 0.02).all())))
    ws = pd.read_csv(analysis_file("hr_window_stats_veg.csv"))
    pv = pd.read_csv(analysis_file("hr_window_pairs_veg.csv"))
    hs_veg = ws[ws.model == "E4:hsc_eng"]
    out.append(("within vegetated canopy no window of the cube stood out (W below 0.3, at least four windows tied "
                "with the top one)", float(hs_veg.kendall_w.iloc[0]) < 0.3
                and int(((pv.model == "E4:hsc_eng") & (pv.p_top_better <= 0.75)).sum()) >= 4))
    imb = pd.read_csv(analysis_file("ndvi_imbalance.csv")).set_index("threshold")
    at = imb.loc[min(imb.index, key=lambda t: abs(t - 0.5))]
    out.append(("vegetated Health patches stay greener than Rust patches, so NDVI alone separates them above chance",
                at.ndvi_median_health > at.ndvi_median_rust and at.ndvi_auc > 0.55))
    out.append(("the NDVI imbalance grows with the threshold, and the balancing weights remove it (AUC within 0.02 "
                "of 0.5)", imb.ndvi_auc.iloc[-1] > imb.ndvi_auc.iloc[0] and abs(at.ndvi_auc_balanced - 0.5) < 0.02))
    bal = pd.read_csv(analysis_file("ndvi_balance.csv")).set_index("name")
    out.append(("with NDVI-balanced weights the vegetated HS advantage stays at least probable with both learners",
                all(bal.loc[n, "p_better"] > 0.75 and bal.loc[n, "p_better"] >= bal.loc[n, "p_equiv"]
                    for n in ("hs_vs_ms", "t_hs_vs_ms"))))
    hrv = pd.read_csv(analysis_file("contrasts.csv")).set_index(["family", "name"])
    out.append(("with NDVI-balanced weights the vegetated HS advantage is at least as large as without, both learners",
                all(bal.loc[n, "diff"] >= hrv.loc[("hr_vegetated", n), "diff"] for n in ("hs_vs_ms", "t_hs_vs_ms"))))
    use = pd.read_csv(analysis_file("ms_feature_use.csv")).set_index("band_nm").n_features
    allw = pd.read_csv(analysis_file("hr_window_stats.csv"))
    ms_rank = allw[allw.model == "E4:ms_eng"].sort_values("rank").window.astype(int).tolist()
    out.append(("the FLAML MS permutation ranking follows the number of features each band enters",
                ms_rank == use.sort_values(ascending=False).index.tolist()))
    fr = pd.read_csv(analysis_file("e9_selection_frequency.csv"))
    f5 = fr[fr.k == 5].frequency
    top, second = sorted(f5.unique(), reverse=True)[:2]
    out.append(("with five bands one wavelength leads alone and several share the second frequency",
                int((f5 == top).sum()) == 1 and int((f5 == second).sum()) > 1))
    return out


def holdout_claims(text):
    """Statements of the holdout paragraph, the limitations, the introduction, and the conclusions. Empty until
    leaderboard scores exist; every entry is (statement, holds)."""
    kv = pd.read_csv(analysis_file("leaderboard_vs_cv.csv")).set_index("sid")
    if kv.private.isna().all():
        return []
    meta = json.load(open(os.path.join(LEADERBOARD, "meta.json")))
    sc = pd.read_csv(os.path.join(LEADERBOARD, "scores.csv"))
    man = pd.read_csv(os.path.join(LEADERBOARD, "submission_manifest.csv"))
    out = [("the manifest, the scores, and the leaderboard analysis hold the same seven submissions",
            list(man.sid) == list(sc.sid) == list(kv.index) == list(SUBMISSIONS))]
    whole = all(abs(round(x * meta["n_public"]) / meta["n_public"] - x) <= 1.0001e-5 for x in sc.public_score) and \
        all(abs(round(x * meta["n_private"]) / meta["n_private"] - x) <= 1.0001e-5 for x in sc.private_score)
    out.append(("every returned score is a whole number of correct patches out of n_public and n_private, which "
                "sum to the holdout size, so the metric is accuracy",
                whole and meta["n_public"] + meta["n_private"] == meta["n_holdout"] and meta["metric"] == "accuracy"))
    f = kv.loc[SUBMISSIONS_FLAML]
    disc = [(a, b) for i, a in enumerate(f.index) for b in f.index[i + 1:]
            if (f.cv[a] - f.cv[b]) * (f.private[a] - f.private[b]) < 0]
    out.append(("the six FLAML configurations kept their CV order except that MS and cleaned HS traded places",
                sorted(map(sorted, disc)) == [["S03", "S04"]]))
    a7 = kv.loc[list(SUBMISSIONS)]
    others = a7.drop("S02").private
    out.append(("the PNG-based RGB view trailed far behind on the private split (lowest by at least 0.05)",
                a7.private.idxmin() == "S02" and others.min() - a7.private["S02"] >= 0.05))
    pr = pd.read_csv(analysis_file("leaderboard_pairs.csv"))
    rg = pr[((pr.a == "S02") & pr.b.isin(list(SUBMISSIONS))) | ((pr.b == "S02") & pr.a.isin(list(SUBMISSIONS)))]
    deficit = np.where(rg.a == "S02", -rg.d_all, rg.d_all)
    adj = holm(rg.p_bound_all.values)
    out.append(("over all holdout patches the RGB view classified fewer patches correctly than each other submission "
                "of the paper, beyond chance after Holm (upper bound of the sign test)",
                len(rg) == len(SUBMISSIONS) - 1 and (deficit > 0).all() and (adj < 0.05).all()))
    r9 = rg[(rg.a == "S09") | (rg.b == "S09")].iloc[0]
    s09, s02 = (r9.correct_all_a, r9.correct_all_b) if r9.a == "S09" else (r9.correct_all_b, r9.correct_all_a)
    out.append(("the same three bands taken from the cube classified many more holdout patches correctly than the "
                "PNGs (at least 20 of 300)", abs(int(r9.d_all)) >= 20 and int(s09) > int(s02)))
    rest = [s for s in SUBMISSIONS if s != "S02"]
    rr = pr[pr.a.isin(rest) & pr.b.isin(rest)]
    out.append(("the holdout cannot resolve the smaller differences (every other pair has a sign-test bound above "
                "0.05)", bool((rr.p_bound_all > 0.05).all()) and len(rr) == len(rest) * (len(rest) - 1) // 2))
    g = means()
    e2, e3 = g.loc["E2:tabpfn_all"], g.loc["E3t:tabpfn8_all_f1_d0"]
    out.append(("S05 (E2, automatic ensemble size) matches the paper's eight-member TabPFN on all views in CV "
                "(within 0.002)", bool((abs(e2 - e3) < 0.002).all())))
    cvm = pd.read_csv(analysis_file("leaderboard_cv_map.csv")).set_index("sid").loc[list(SUBMISSIONS)]
    cvm.loc["S05", ["cv_mean", "cv_acc"]] = [e3.macro_f1, e3.accuracy]
    out.append(("TabPFN on all views was the most accurate of the seven in CV (accuracy and macro-F1)",
                cvm.cv_acc.idxmax() == "S05" and cvm.cv_mean.idxmax() == "S05"))
    out.append(("TabPFN scored below the FLAML baseline on the private split but within its range",
                kv.private["S05"] < kv.private["S01"] and bool(kv.private_in_range["S05"])))
    out.append(("every FLAML score of the paper exceeded its CV accuracy", bool((f.private_minus_cv > 0).all())))
    gains = pd.read_csv(analysis_file("refit_gain.csv")).accuracy_gain
    out.append(("the last 120 training patches raised FLAML accuracy on average by less than the smallest private "
                "shift, so training-set size probably explains only part of it",
                len(gains) == 4 and gains.max() < f.private_minus_cv.min()))
    if text is not None:
        out.append(("the holdout heading states that the scores confirm the RGB deficit", "\\HoldoutHeading" in text))
        for mac in ["KaggleInRangeWordA", "KendallTauAF", "KaggleRGBSignP", "SlopeFlamlAccMax", "KaggleDiffMinAF"]:
            out.append((f"the holdout paragraph cites \\{mac}", f"\\{mac}" in text))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manuscript", help="folder with main.tex and sections/*.tex, for the checks on the text")
    ap.add_argument("--strict", action="store_true", help="exit with status 1 when a check fails")
    args = ap.parse_args()
    text = None
    if args.manuscript:
        files = sorted(glob.glob(os.path.join(args.manuscript, "sections", "*.tex"))) + \
            [os.path.join(args.manuscript, "main.tex")]
        text = "\n".join(open(f).read() for f in files)

    df = pd.read_csv(analysis_file("contrasts.csv")).set_index(["family", "name"])
    col = {"better": "p_better", "equivalent": "p_equiv", "worse": "p_worse"}
    bad, missing = [], []
    for fam, name, direction, want, where in CLAIMS + CONTROL_CLAIMS:
        if (fam, name) not in df.index:
            missing.append(f"{fam}/{name} not in contrasts.csv ({where})")
            continue
        r = df.loc[(fam, name)]
        ps = {d: float(r[c]) for d, c in col.items()}
        top = max(ps, key=ps.get)
        if want == "notbetter":
            ok = ps["better"] <= 0.5
            got = f"P(better) {ps['better']:.3f}"
        elif want == "notclear":
            ok = ps[direction] <= 0.95
            got = f"P({direction}) {ps[direction]:.3f}"
        elif want in ("weak", "leaning"):
            # inconclusive means no posterior above 0.75, and leaning also needs the stated direction to be largest
            ok = max(ps.values()) <= 0.75 and (want == "weak" or top == direction)
            got = f"{top} {ps[top]:.3f}"
        else:
            ok = top == direction and strength(ps[direction]) == want
            got = f"{top} {ps[top]:.3f} ({strength(ps[top])})"
        if not ok:
            bad.append(f"{fam}/{name}: text says {direction} {want}, posteriors give {got} [{where}]")
        if text is not None:
            key = f"{PREFIX.get(fam, 'spw')}:{name}"
            alt = {"planned": f"pw:{name}", "tabpfn": f"pw:{name}", "learner": f"pw:{name}",
                   "hr_all": f"hrpw:{name}:hr_auc"}.get(fam)
            summarised = name.startswith("lrn_view_")  # reported through \LrnViewDiffMin and \LrnViewDiffMax
            if key not in text and (alt is None or alt not in text) and not summarised:
                missing.append(f"{key} not cited in the text ({where})")
    print(f"{len(CLAIMS) + len(CONTROL_CLAIMS)} labelled claims checked, {len(bad)} mismatches"
          + (f", {len(missing)} not found" if text is not None or missing else ""))
    for b in bad:
        print("MISMATCH", b)
    for m in missing:
        print("NOT FOUND", m)
    if text is not None:
        # a posterior printed next to a label threshold should use the three-decimal p3_ field
        for (fam, name), r in df.iterrows():
            for c in ("p_better", "p_equiv", "p_worse"):
                x = float(r[c])
                if any(abs(x - t) < 0.01 for t in (0.75, 0.95)) and re.search(
                        re.escape(f"{PREFIX.get(fam, 'spw')}:{name}}}{{{c}}}"), text):
                    print(f"NEAR THRESHOLD {fam}/{name} {c} = {x:.4f} printed with two decimals; consider p3_ field")
    pref = {"spw": None, "shr": "hr_all", "shrv": "hr_vegetated"}
    for pre, name, confirmed in HOLM_CLAIMS:
        rows = df.xs(name, level="name")
        fam = pref[pre]
        r = rows.loc[fam] if fam else rows[~rows.index.str.startswith("hr")].iloc[0]
        if (float(r.p_holm) < 0.05) != confirmed:
            bad.append(f"Holm {pre}:{name} adjusted p {float(r.p_holm):.3f}, text says "
                       f"{'confirmed' if confirmed else 'not confirmed'} [Results opening paragraph]")
            print("MISMATCH", bad[-1])
    ablation = df.xs("ablation", level="family")
    if (ablation.p_holm < 0.05).any():
        print("NOTE some ablation contrasts are confirmed after Holm; the generated word in Appendix A reflects it")
    statements = wavelength_claims() + learner_claims(df) + holdout_claims(text) + robustness_claims()
    for claim, ok in statements:
        if not ok:
            bad.append(f"claim fails: {claim}")
            print("MISMATCH", bad[-1])
    print(f"{len(HOLM_CLAIMS) + len(statements)} Holm, wavelength, learner, holdout, and robustness statements "
          "checked; "
          + ("ALL CHECKS PASSED" if not bad else f"ATTENTION {len(bad)} FAILURE(S), see MISMATCH lines"))
    if bad and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
