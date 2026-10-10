"""Write outputs/robustness/, the numbers and tables of the robustness analyses.

  robustness.tex        LaTeX macros, named (for example \\RobNClearBH) and keyed (for example
                        \\res{rob:planned:hs_vs_ms}{lo}), read in the text after outputs/numbers.tex
  contrasts_f1_ci.tex, contrasts_hr_ci.tex, contrasts_learner_ci.tex
                        Tables A2 to A4 with the 95 percent credible interval in place of the unadjusted p-value and a
                        mark on every verdict that changes with a ROPE of 0.01 or 0.03. They keep the labels of the
                        tables they replace, so either version can be input.
  headline.tex          the contrasts behind the paper's main statements with credible intervals, the verdicts at
                        three ROPEs, and the global Benjamini-Hochberg adjustment, confirmatory contrasts first
  ndvi_threshold.tex    cleaned HS minus MS on vegetated patches for NDVI thresholds from 0.3 to 0.7
  holdout_all.tex       all ten holdout submissions of the study
  label_shift.tex       holdout class priors estimated from the submitted predictions, and the accuracy they imply
  crossbatch.tex        transfer between the two HS file formats per view and learner

Values come from results/analysis/ (scripts/analysis/01, 08, and 09 to 13) and, when it exists,
results/audit/band_separability.csv (scripts/eda/04_band_separability.py, which needs the images). Captions use
macros of outputs/numbers.tex, so the tables compile with it.
usage: python scripts/tables/04_robustness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import importlib
import json
import os

import pandas as pd

from scripts.analysis._shared import NUMWORD, all_fold_metrics, analysis_file
from scripts.tables._latex import f2, f3, prob, prob_fine, pval, signed, write_table
from wheatrust.io_data import CLASSES, OUTPUTS, RESULTS

OUT = os.path.join(OUTPUTS, "robustness")
SOURCE = "scripts/tables/04_robustness.py"
NA = "n/a"
L = ["% Written by scripts/tables/04_robustness.py from results/. Rebuild instead of editing.",
     "\\makeatletter",
     "\\providecommand{\\res}[2]{\\ifcsname res@#1@#2\\endcsname\\csname res@#1@#2\\endcsname"
     "\\else\\textbf{??\\detokenize{#1/#2}??}\\fi}",
     "\\makeatother"]
NAMED = {}
tables02 = importlib.import_module("scripts.tables.02_tables")  # display labels of Table 5 and its CV override

DAGGER = "$^{\\dagger}$"
THRESHOLDS_TEXT = ("Verdicts take the largest of the three posteriors and call it clear above 0.95, probable above "
                   "0.75, and leaning above 0.5, and inconclusive otherwise.")
SECT = "\\textsuperscript{\\S}"
FAMILY_TITLES = {"planned": "Planned contrasts, FLAML",
                 "tabpfn": f"TabPFN versions of the planned contrasts{DAGGER}",
                 "controls": f"Controls added after the first round of results{DAGGER}",
                 "interaction": f"Differences between two contrasts on the same folds{DAGGER}",
                 "hr_all": f"Health versus Rust AUC, all Health and Rust patches{DAGGER}",
                 "hr_vegetated": f"Health versus Rust AUC, vegetated patches only{DAGGER}",
                 "learner": f"TabPFN minus FLAML on the same inputs{DAGGER}"}
# the contrasts behind the main statements of the paper, in the order of the Results
HEADLINE = [("planned", "stretch"), ("planned", "resample"), ("planned", "ms_vs_rgbhs"), ("planned", "hs_vs_ms"),
            ("planned", "ms5_vs_rgb3"), ("planned", "k3_vs_rgb3"), ("planned", "k5_vs_ms5"),
            ("tabpfn", "t_stretch"), ("tabpfn", "t_ms_vs_rgbhs"), ("tabpfn", "t_hs_vs_ms"),
            ("tabpfn", "t_ms5_vs_rgb3"), ("tabpfn", "t_k5_vs_ms5"),
            ("controls", "st_vs_hs"), ("controls", "t_st_vs_hs"), ("controls", "hs_vs_ms_raw"),
            ("hr_vegetated", "hs_vs_ms"), ("hr_vegetated", "t_hs_vs_ms"),
            ("interaction", "cost_minus_rededge_flaml"), ("interaction", "cost_minus_rededge_tabpfn"),
            ("learner", "lrn_view_all")]
OTHER_LABELS = {"S06": "HybridSN-style network, cleaned HS cube", "S08": "FLAML, simulated five-band camera",
                "S10": "Shrinkage LDA, all views"}
VIEW_NAMES = {"all": "All views", "ms": "MS", "hsc": "Cleaned HS"}
DIRECTION_NAMES = {"125_to_126": "125 to 126 bands", "126_to_125": "126 to 125 bands"}


def name(n, v):
    assert n.isalpha() and n not in NAMED, n
    NAMED[n] = v
    L.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def res(key, field, v):
    L.append(f"\\expandafter\\def\\csname res@{key}@{field}\\endcsname{{{v}}}")


def word(n):
    n = int(n)
    return "none" if n == 0 else NUMWORD.get(n, str(n))


def wlist(items):
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else (" and ".join(items) if len(items) == 2
                                             else ", ".join(items[:-1]) + ", and " + items[-1])


def short(v):
    """A verdict in a few characters, for narrow table columns."""
    if v == "inconclusive":
        return "inconcl."
    abbr = {"better": "better", "worse": "worse", "equivalent": "equiv."}
    if v.startswith("leaning "):
        return "lean. " + abbr[v.split(" ", 1)[1]]
    direction, strength = v.split(", ")
    return {"clear": "clear", "probable": "prob."}[strength] + " " + abbr[direction]


def prob_t(x):
    """A posterior with two decimals, or three when two would round onto a label threshold (0.5, 0.75, 0.95)."""
    return prob_fine(x) if any(abs(x - t) < 0.005 for t in (0.5, 0.75, 0.95)) else prob(x)


def crint(lo, hi, d=3):
    return f"{signed(lo, d, NA)}, {signed(hi, d, NA)}"


def write(fname, caption, colspec, header, rows, label, note=None, size="\\footnotesize", placement="H"):
    write_table(os.path.join(OUT, fname + ".tex"), caption, colspec, header, rows, label, note=note, size=size,
                source=SOURCE, placement=placement)


# ---------------------------------------------------------------------------------------------------------------
# contrasts, with credible intervals, ROPE sensitivity, status, and multiplicity (scripts/analysis/09)
# ---------------------------------------------------------------------------------------------------------------
def contrast_macros(rob):
    for _, r in rob.iterrows():
        k = f"rob:{r.family}:{r['name']}"
        res(k, "diff", signed(r["diff"], missing=NA))
        res(k, "lo", signed(r.ci95_lo, missing=NA))
        res(k, "hi", signed(r.ci95_hi, missing=NA))
        res(k, "ppos", prob(r.p_positive, NA))
        for t in ("r01", "r02", "r03"):
            res(k, f"v{t[1:]}", r[f"verdict_{t}"])
            for c in ("better", "equiv", "worse"):
                res(k, f"p{c}{t[1:]}", prob(r[f"p_{c}_{t}"], NA))
            res(k, f"pnotworse{t[1:]}", prob(1.0 - r[f"p_worse_{t}"], NA))  # better or equivalent
        res(k, "qbh", pval(r.q_bh_global, NA))
        res(k, "qby", pval(r.q_by_global, NA))
        res(k, "pholmg", pval(r.p_holm_global, NA))
        res(k, "pholm", pval(r.p_holm, NA))
        res(k, "status", r.status)
    name("RobROPELow", "0.01")
    name("RobROPEHigh", "0.03")
    name("RobNContrasts", len(rob))
    name("RobNConfirmatory", int((rob.status == "confirmatory").sum()))
    name("RobNExploratory", int((rob.status == "exploratory").sum()))
    name("RobNAppendixContrasts", int((rob.family != "ablation").sum()))

    ms = pd.read_csv(analysis_file("multiplicity_summary.csv"))
    for status, stag in [("any", ""), ("confirmatory", "Conf"), ("exploratory", "Expl")]:
        for label, ltag in [("clear difference", "Clear"), ("probable difference", "Probable")]:
            r = ms[(ms.scope == "appendix") & (ms.status == status) & (ms.label == label)].iloc[0]
            name(f"RobN{stag}{ltag}", int(r.n))
            for col, atag in [("n_holm_family", "HolmFamily"), ("n_holm_global", "HolmGlobal"), ("n_bh_global", "BH"),
                              ("n_by_global", "BY"), ("n_bh_status", "BHStatus")]:
                name(f"RobN{stag}{ltag}{atag}", int(r[col]))
                name(f"RobN{stag}{ltag}{atag}Word", word(r[col]))
    rs = pd.read_csv(analysis_file("rope_summary.csv"))
    r = rs[(rs.scope == "appendix") & (rs.status == "any")].iloc[0]
    name("RobNLabelChange", int(r.n_changes_any))
    name("RobNLabelChangeLow", int(r.n_changes_r01))
    name("RobNLabelChangeHigh", int(r.n_changes_r03))
    head = rob.set_index(["family", "name"]).loc[HEADLINE]
    name("RobHeadlineN", len(head))
    name("RobHeadlineChanged", int(head.label_changes.sum()))
    name("RobHeadlineChangedWord", word(head.label_changes.sum()))


def contrast_tables_ci(rob):
    """Tables A2 to A4 as in scripts/tables/02_tables.py, with the credible interval in place of the unadjusted
    p-value."""
    header = ("\\textbf{Contrast (A minus B)} & \\textbf{Diff.} & \\textbf{95\\% CrI} & \\textbf{P(A$>$B)} & "
              "\\textbf{P(equiv.)} & \\textbf{P(A$<$B)} & \\textbf{$p$, Holm} & \\textbf{Verdict}")
    spec = ">{\\raggedright\\arraybackslash}X cccccc>{\\raggedright\\arraybackslash}p{2.75cm}"
    note = ("Diff.\\ is the mean paired difference over outer folds and 95\\% CrI its central 95 percent credible "
            "interval. The three probabilities are the posteriors of the Bayesian correlated $t$-test with a ROPE of "
            "plus or minus \\ROPE{}, and $p$, Holm is the two-sided corrected resampled $t$-test adjusted by Holm "
            "within each family (block). " + THRESHOLDS_TEXT + f" {SECT} marks a verdict that changes when the ROPE is "
            f"\\RobROPELow{{}} or \\RobROPEHigh{{}}; Table~\\ref{{tab:headline}} gives the labels at all three ROPEs "
            f"for the contrasts behind the main statements, and the code repository for every contrast. "
            f"{DAGGER}Added after the first round of results.")
    groups = [("contrasts_f1_ci", ["planned", "tabpfn", "controls", "interaction"],
               "Macro-F1 contrasts cited in the paper, by family.", "tab:contrasts-f1"),
              ("contrasts_hr_ci", ["hr_all", "hr_vegetated"],
               "Health versus Rust AUC contrasts cited in the paper, by family (ROPE of plus or minus \\ROPE{} AUC).",
               "tab:contrasts-hr"),
              ("contrasts_learner_ci", ["learner"], "Macro-F1 of TabPFN minus FLAML on identical inputs and folds.",
               "tab:contrasts-learner")]
    for fname, fams, cap, label in groups:
        rows = []
        for fam in fams:
            g = rob[rob.family == fam]
            if rows:
                rows.append("\\midrule")
            rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{FAMILY_TITLES[fam]}}}}}")
            for _, r in g.iterrows():
                desc = str(r.description)
                if fam in ("planned", "tabpfn"):
                    desc = desc.replace(", FLAML", "").replace(", TabPFN", "")
                desc = desc.replace("TabPFN minus FLAML, ", "")
                mark = SECT if r.label_changes else ""
                rows.append(f"{desc} & {signed(r['diff'])} & {crint(r.ci95_lo, r.ci95_hi)} & {prob_t(r.p_better)} & "
                            f"{prob_t(r.p_equiv)} & {prob_t(r.p_worse)} & {pval(r.p_holm)} & {r.verdict_r02}{mark}")
        write(fname, cap, spec, header, rows, label, note=note)


def headline_table(rob):
    idx = rob.set_index(["family", "name"])
    rows = []
    for status, title in [("confirmatory", "Confirmatory, planned before any result existed"),
                          ("exploratory", "Exploratory, added after the first round of results")]:
        if rows:
            rows.append("\\midrule")
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}}")
        for fam, nm in HEADLINE:
            r = idx.loc[(fam, nm)]
            if r.status != status:
                continue
            desc = str(r.description)
            desc = desc[0].upper() + desc[1:]
            if fam.startswith("hr_"):
                desc += " (AUC, " + ("vegetated patches" if fam == "hr_vegetated" else "all patches") + ")"
            rows.append(f"{desc} & {signed(r['diff'])} ({crint(r.ci95_lo, r.ci95_hi)}) & {short(r.verdict_r01)} & "
                        f"{short(r.verdict_r02)} & {short(r.verdict_r03)} & {pval(r.q_bh_global)}")
    write("headline", "Robustness of the contrasts behind the main statements. Diff.\\ is the mean paired "
          "difference over outer folds (macro-F1 unless marked AUC) with its central 95 percent credible interval. "
          "The three verdict columns label the posteriors of the Bayesian correlated $t$-test with a ROPE of plus "
          "or minus \\RobROPELow{}, \\ROPE{} (the paper's), and \\RobROPEHigh{}. " + THRESHOLDS_TEXT + " prob.\\ "
          "is probable, lean.\\ leaning, inconcl.\\ inconclusive, and equiv.\\ equivalent. $q$ is the "
          "Benjamini-Hochberg adjustment of the corrected resampled $t$-test over all \\RobNContrasts{} contrasts of "
          "the paper, and a difference counts as confirmed when $q$ is below 0.05 before rounding. Confirmatory "
          "contrasts were specified in the study plan before any result existed; the others were added after the "
          "first round of results.",
          ">{\\raggedright\\arraybackslash}X ccccc",
          "\\textbf{Contrast (A minus B)} & \\textbf{Diff. (95\\% CrI)} & \\textbf{ROPE \\RobROPELow{}} & "
          "\\textbf{ROPE \\ROPE{}} & \\textbf{ROPE \\RobROPEHigh{}} & \\textbf{$q$, BH}", rows, "tab:headline",
          placement="tbp")  # the paper cites this table in the main text, where a float avoids a half-empty page


# ---------------------------------------------------------------------------------------------------------------
# NDVI threshold of the vegetated subset (scripts/analysis/10)
# ---------------------------------------------------------------------------------------------------------------
def ndvi():
    nd = pd.read_csv(analysis_file("ndvi_threshold.csv"))
    ths = sorted(nd.threshold.unique())
    for _, r in nd.iterrows():
        k = f"robndvi:{r.threshold:.1f}:{r['name']}"
        res(k, "diff", signed(r["diff"], missing=NA))
        res(k, "lo", signed(r.ci95_lo, missing=NA))
        res(k, "hi", signed(r.ci95_hi, missing=NA))
        for c in ("better", "equiv", "worse"):
            res(k, f"p{c}", prob(r[f"p_{c}"], NA))
        res(k, "pholm", pval(r.p_holm, NA))
        res(k, "verdict", r.verdict)
        res(k, "n", int(r.n_paired))
    for t, g in nd.groupby("threshold"):
        res(f"robndvi:{t:.1f}", "nhealth", int(g.n_health.iloc[0]))
        res(f"robndvi:{t:.1f}", "nrust", int(g.n_rust.iloc[0]))
    name("RobNDVIList", wlist([f"{t:.1f}" for t in ths]))
    name("RobNDVIMin", f"{min(ths):.1f}")
    name("RobNDVIMax", f"{max(ths):.1f}")
    # the HS advantage over the thresholds the review asked about, around the paper's 0.5
    near = nd[nd.threshold.between(0.35, 0.65)]
    for nm, tag in [("hs_vs_ms", "Flaml"), ("t_hs_vs_ms", "Tabpfn")]:
        g = near[near.name == nm]
        name(f"RobVegHS{tag}DiffMin", signed(g["diff"].min(), missing=NA))
        name(f"RobVegHS{tag}DiffMax", signed(g["diff"].max(), missing=NA))
        name(f"RobVegHS{tag}PMin", prob(g.p_better.min(), NA))
        name(f"RobVegHS{tag}PMax", prob(g.p_better.max(), NA))
        name(f"RobVegHS{tag}Verdicts", wlist(sorted(set(g.verdict))))
    imb = pd.read_csv(analysis_file("ndvi_imbalance.csv")).set_index("threshold")
    paper = imb.loc[min(imb.index, key=lambda t: abs(t - 0.5))]
    name("RobVegNDVIMedHealth", f2(paper.ndvi_median_health, NA))
    name("RobVegNDVIMedRust", f2(paper.ndvi_median_rust, NA))
    name("RobVegNDVIAUC", f2(paper.ndvi_auc, NA))
    name("RobVegNDVIAUCBal", f2(paper.ndvi_auc_balanced, NA))
    name("RobVegNDVIAUCMin", f2(imb.ndvi_auc.min(), NA))
    name("RobVegNDVIAUCMax", f2(imb.ndvi_auc.max(), NA))
    bal = pd.read_csv(analysis_file("ndvi_balance.csv")).set_index("name")
    for nm, r in bal.iterrows():
        k = f"robbal:{nm}"
        res(k, "diff", signed(r["diff"], missing=NA))
        res(k, "lo", signed(r.ci95_lo, missing=NA))
        res(k, "hi", signed(r.ci95_hi, missing=NA))
        res(k, "pbetter", prob(r.p_better, NA))
        res(k, "verdict", r.verdict)
        res(k, "auca", f3(r.auc_a, NA))
        res(k, "aucb", f3(r.auc_b, NA))
    rows = []
    for t in ths:
        a = nd[(nd.threshold == t) & (nd.name == "hs_vs_ms")].iloc[0]
        b = nd[(nd.threshold == t) & (nd.name == "t_hs_vs_ms")].iloc[0]
        lab = f"{t:.1f}" + (" (paper)" if abs(t - 0.5) < 1e-9 else "")
        rows.append(f"{lab} & {int(a.n_health)}/{int(a.n_rust)} & {f2(imb.loc[t, 'ndvi_auc'])} & "
                    f"{signed(a['diff'])} ({crint(a.ci95_lo, a.ci95_hi)}) & {short(a.verdict)} & "
                    f"{signed(b['diff'])} ({crint(b.ci95_lo, b.ci95_hi)}) & {short(b.verdict)}")
    a, b = bal.loc["hs_vs_ms"], bal.loc["t_hs_vs_ms"]
    rows.append("\\midrule")
    at = nd[(nd.threshold - 0.5).abs() < 1e-9].iloc[0]
    rows.append(f"0.5, balanced & {int(at.n_health)}/{int(at.n_rust)} & {f2(paper.ndvi_auc_balanced)} & "
                f"{signed(a['diff'])} ({crint(a.ci95_lo, a.ci95_hi)}) & {short(a.verdict)} & "
                f"{signed(b['diff'])} ({crint(b.ci95_lo, b.ci95_hi)}) & {short(b.verdict)}")
    write("ndvi_threshold", "Cleaned HS minus MS in the Health versus Rust AUC on vegetated patches, for several "
          "NDVI thresholds of the vegetated subset (not empty and patch-mean NDVI at or above the threshold). "
          "Patches gives the vegetated Health and Rust patches, and NDVI AUC the separation of the two classes by "
          "patch-mean NDVI alone, lower NDVI counting as Rust, which shows that the Health patches left by a "
          "threshold stay greener than the Rust patches. The last row repeats the paper's threshold with weights "
          "that give both classes the same distribution over NDVI quintiles. Diff.\\ is the mean paired difference "
          "over outer folds with its central 95 percent credible interval, and the verdict labels the posteriors "
          "with a ROPE of plus or minus \\ROPE{} AUC. " + THRESHOLDS_TEXT + " prob.\\ is probable and lean.\\ "
          "leaning. The paper's threshold of 0.5 was fixed from the audit before the analysis.",
          "lcccccc",
          "& & & \\multicolumn{2}{c}{\\textbf{FLAML}} & \\multicolumn{2}{c}{\\textbf{TabPFN}} \\\\ "
          "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7} \\textbf{NDVI} & \\textbf{Patches} & \\textbf{NDVI AUC} & "
          "\\textbf{Diff. (95\\% CrI)} & \\textbf{Verdict} & \\textbf{Diff. (95\\% CrI)} & \\textbf{Verdict}",
          rows, "tab:ndvi-threshold")


# ---------------------------------------------------------------------------------------------------------------
# holdout, with all ten submissions, label shift, and rank agreement (scripts/analysis/11)
# ---------------------------------------------------------------------------------------------------------------
def holdout():
    ten = pd.read_csv(analysis_file("holdout_all.csv"))
    fm = all_fold_metrics(["key", "accuracy"])
    cv_acc = {sid: fm[fm.key == k].accuracy.mean() for sid, k in tables02.CV_KEY.items()}
    for _, r in ten.iterrows():
        k = f"robsub:{r.sid}"
        for c in ["public", "private", "cv_accuracy", "range_lo95", "range_hi95"]:
            res(k, c.replace("_", ""), f3(r[c], NA))
        res(k, "inrange", "inside" if r.private_in_range else ("above" if r.private > r.range_hi95 else "below"))
    n_in = int(ten.private_in_range.sum())
    name("RobNInRangeAll", f"All {NUMWORD[len(ten)]}" if n_in == len(ten)
         else f"{NUMWORD.get(n_in, str(n_in)).capitalize()} of the {NUMWORD[len(ten)]}")
    oth = ten[~ten.in_paper]
    name("RobOtherPrivateList", wlist([f3(v) for v in oth.private]))
    name("RobOtherSidList", wlist(list(oth.sid)))
    rows = []
    for _, r in ten.iterrows():
        label = tables02.LABELS.get(r.sid, OTHER_LABELS.get(r.sid, r.sid))
        cv = cv_acc.get(r.sid, r.cv_accuracy)
        rows.append(f"{r.sid} & {label} & {'yes' if r.in_paper else 'no'} & {f3(cv)} & "
                    f"{r.range_lo95:.3f} to {r.range_hi95:.3f} & {int(r.n_Health)}/{int(r.n_Rust)}/{int(r.n_Other)} & "
                    f"{r.public:.3f} & {r.private:.3f}")
    write("holdout_all", "All \\NSubmissionsWord{} holdout submissions of the study. The seven marked yes are those "
          "of Table~\\ref{tab:holdout}; the other three use a learner or an input outside the scope of this paper, "
          "and their code and per-fold results are not part of the repository. CV acc.\\ is the outer-fold mean of "
          "the same procedure, and the range is the central 95 percent of CV-based accuracy for a test set of "
          "\\KagglePrivateN{} samples, the size of the private split. Predicted counts are Health/Rust/Other among "
          "the \\NHoldout{} holdout samples. Public and Private give the leaderboard accuracy on \\KagglePublicN{} and "
          "\\KagglePrivateN{} holdout patches, recorded after all submissions were final.",
          "lXcccccc", "\\textbf{ID} & \\textbf{Model} & \\textbf{In paper} & \\textbf{CV acc.} & "
          "\\textbf{95\\% range} & \\textbf{Predicted} & \\textbf{Public} & \\textbf{Private}", rows,
          "tab:holdout-all")

    ls = pd.read_csv(analysis_file("label_shift.csv"))
    for _, r in ls.iterrows():
        k = f"robshift:{r.sid}"
        for c in CLASSES:
            res(k, f"prior{c}", f2(r[f"prior_{c}"], NA))
            res(k, f"prior{c}lo", f2(r[f"prior_{c}_lo95"], NA))
            res(k, f"prior{c}hi", f2(r[f"prior_{c}_hi95"], NA))
        if r.sid != "joint":
            res(k, "accexp", f3(r.accuracy_expected, NA))
            res(k, "accexplo", f3(r.accuracy_expected_lo95, NA))
            res(k, "accexphi", f3(r.accuracy_expected_hi95, NA))
            res(k, "cvacc", f3(r.cv_accuracy_pooled, NA))
            res(k, "gainprivate", signed(r.gain_private, missing=NA))
            res(k, "gainpriors", signed(r.gain_from_priors, missing=NA))
            res(k, "gainsize", signed(r.gain_from_training_size, missing=NA))
    j = ls[ls.sid == "joint"].iloc[0]
    for c in CLASSES:
        name(f"RobShiftJoint{c}", f2(j[f"prior_{c}"], NA))
        name(f"RobShiftJoint{c}Lo", f2(j[f"prior_{c}_lo95"], NA))
        name(f"RobShiftJoint{c}Hi", f2(j[f"prior_{c}_hi95"], NA))
    outside = [c for c in CLASSES if not (j[f"prior_{c}_lo95"] <= 1 / 3 <= j[f"prior_{c}_hi95"])]
    name("RobShiftClassesExcludingThird", wlist(outside) if outside else "none")
    sub = ls[ls.sid != "joint"]
    name("RobShiftGainMin", signed(sub.gain_from_priors.min(), missing=NA))
    name("RobShiftGainMax", signed(sub.gain_from_priors.max(), missing=NA))
    rows = []
    for _, r in ls.iterrows():
        pri = " & ".join(f"{f2(r[f'prior_{c}'])} ({f2(r[f'prior_{c}_lo95'])} to {f2(r[f'prior_{c}_hi95'])})"
                         for c in CLASSES)
        if r.sid == "joint":
            rows.append("\\midrule")
            rows.append(f"All seven, jointly & & {pri} & & & ")
        else:
            cv = cv_acc.get(r.sid, ten.set_index("sid").loc[r.sid, "cv_accuracy"])
            rows.append(f"{r.sid} & {int(r.n_pred_Health)}/{int(r.n_pred_Rust)}/{int(r.n_pred_Other)} & {pri} & "
                        f"{f3(cv)} & {f3(r.accuracy_expected)} & {f3(r.private)}")
    write("label_shift", "Holdout class priors estimated from the submitted predictions. For each submission the "
          "priors $q$ solve $M q = s$ with $q \\geq 0$ and $\\sum q = 1$, where $s$ holds the shares of the "
          "predicted classes among the \\NHoldout{} holdout patches and $M_{ij}$ the cross-validated probability of "
          "predicting class $i$ for a patch of class $j$; the last row fits all seven submissions at once. "
          "Intervals are central 95 percent bootstrap intervals over training and holdout patches. Training has "
          "\\NPerClass{} patches per class, priors of one third each. CV acc.\\ is the outer-fold mean accuracy, as in "
          "Table~\\ref{tab:holdout}, and Expected acc.\\ the accuracy under the joint priors, the sum over classes of "
          "prior times cross-validated recall. The cross-validated confusion matrix stands in for that of each "
          "refitted model. No holdout label is used.",
          "lccccccc",
          "\\textbf{ID} & \\textbf{Predicted} & \\textbf{Health prior} & \\textbf{Rust prior} & "
          "\\textbf{Other prior} & \\textbf{CV acc.} & \\textbf{Expected acc.} & \\textbf{Private}", rows,
          "tab:label-shift")

    tau = pd.read_csv(analysis_file("holdout_tau.csv"))
    pt = tau[tau.score == "private"].set_index("set")
    for s, tag in [("all", "All"), ("all_without_S02", "AllNoRGB"), ("paper", "Paper"),
                   ("paper_without_S02", "PaperNoRGB"), ("paper_flaml", "Flaml"),
                   ("paper_flaml_without_S02", "FlamlNoRGB")]:
        name(f"RobTau{tag}", f2(pt.loc[s, "tau"], NA))
        name(f"RobTau{tag}P", pval(pt.loc[s, "p"], NA))
        name(f"RobTau{tag}N", int(pt.loc[s, "n"]))


# ---------------------------------------------------------------------------------------------------------------
# file structure, band selection stability, cross-format transfer, band separability
# ---------------------------------------------------------------------------------------------------------------
def structure():
    fs = json.load(open(analysis_file("file_structure.json")))
    name("RobFormatNumbersAgree", fs["numbers_same_format_all_classes"])
    name("RobFormatNumbersTotal", fs["numbers_per_class"])
    name("RobFormatCommon", fs["n126_common_to_all_classes"])
    name("RobFormatLongestRun", fs["longest_run_126"])
    name("RobOverlapSameLabel", fs["n_overlap_same_label"])
    name("RobOverlapSameLabelWord", word(fs["n_overlap_same_label"]))
    name("RobOverlapTrainPairsWord", word(fs["n_overlap_train_pairs"]))
    name("RobOverlapGaps", wlist(fs["overlap_number_gaps_same_label"]))
    name("RobOverlapGapMin", fs["overlap_min_number_gap_same_label"])

    use = pd.read_csv(analysis_file("ms_feature_use.csv")).set_index("band_nm").n_features
    for wl, tag in [(482, "Blue"), (550, "Green"), (650, "Red"), (738, "RE"), (834, "NIR")]:
        name(f"RobMSFeat{tag}", int(use[wl]))
    name("RobMSFeatTotal", int(pd.read_csv(analysis_file("ms_feature_use.csv")).n_total.iloc[0]))

    st = pd.read_csv(analysis_file("selection_stability.csv"))
    for _, r in st.iterrows():
        k = f"robsel:{int(r.k)}"
        res(k, "ki", f2(r.kuncheva_mean, NA))
        res(k, "jaccard", f2(r.jaccard_mean, NA))
        res(k, "near", f2(r.near_share_mean, NA))
        res(k, "chance", f2(r.near_share_chance, NA))
    small = st[st.k <= 8]
    name("RobSelKIMin", f2(small.kuncheva_mean.min(), NA))
    name("RobSelKIMax", f2(small.kuncheva_mean.max(), NA))
    name("RobSelTolNm", f"{st.tolerance_nm.iloc[0]:.0f}")

    xb = pd.read_csv(analysis_file("crossbatch_cost.csv"))
    xs = pd.read_csv(analysis_file("crossbatch_summary.csv")).set_index(["learner", "view", "name"])
    rows = []
    for lrn in ["flaml", "tabpfn"]:
        if rows:
            rows.append("\\midrule")
        for view in ["all", "ms", "hsc"]:
            for direction, rnd in [("125_to_126", "random_489_to_111"), ("126_to_125", "random_111_to_489")]:
                c = xb[(xb.learner == lrn) & (xb.view == view) & (xb.direction == direction)].iloc[0]
                t, rr = xs.loc[(lrn, view, direction)], xs.loc[(lrn, view, rnd)]
                k = f"robxb:{lrn}:{view}:{direction}"
                res(k, "transfer", f3(c.transfer, NA))
                res(k, "random", f3(c.random, NA))
                res(k, "cost", f3(c.cost, NA))
                rows.append(f"{'FLAML' if lrn == 'flaml' else 'TabPFN'} & {VIEW_NAMES[view]} & "
                            f"{DIRECTION_NAMES[direction]} & {c.transfer:.3f} $\\pm$ {t.sd:.3f} & "
                            f"{c.random:.3f} $\\pm$ {rr.sd:.3f} & {f3(c.cost)}")
    n = int(xs.n.iloc[0])
    write("crossbatch", "Transfer between the two HS file formats. Each model was trained on the training files of "
          "one format and tested on those of the other (\\NHSOneTwentySixTrain{} of the \\NTrain{} training files "
          "carry 126 bands, \\NHSOneTwentySixPerClass{} per class), and, as the reference, on stratified random "
          "splits with the same sizes and class counts. Values are macro-F1, mean $\\pm$ standard deviation over "
          f"{NUMWORD.get(n, n)} seeds. Cost is the random split minus the transfer.", "lllccc",
          "\\textbf{Learner} & \\textbf{Input} & \\textbf{Train to test} & \\textbf{Transfer} & "
          "\\textbf{Random split} & \\textbf{Cost}", rows, "tab:crossbatch", size="\\small")

    path = os.path.join(RESULTS, "audit", "band_separability.csv")
    if not os.path.exists(path):
        print(f"{os.path.relpath(path)} is missing (scripts/eda/04_band_separability.py needs the images); "
              "its macros are skipped")
        return
    sep = pd.read_csv(path)
    for subset, tag in [("vegetated", "Veg"), ("all", "All")]:
        b = sep[(sep.kind == "band") & (sep.subset == subset) & sep.in_clean_range.astype(bool)]
        top = b.sort_values(["separation", "wavelength_nm"], ascending=[False, True]).iloc[0]
        name(f"RobSep{tag}MaxDev", f2(top.separation, NA))
        name(f"RobSep{tag}MaxBand", f"{top.wavelength_nm:.0f}")
        name(f"RobSep{tag}MaxAUC", f2(top.auc, NA))
        name(f"RobSep{tag}MaxLo", f2(top.auc_lo95, NA))
        name(f"RobSep{tag}MaxHi", f2(top.auc_hi95, NA))
        w = sep[(sep.kind == "window") & (sep.subset == subset)]
        for _, r in w.iterrows():
            k = f"robsep:{subset}:{r['name']}"
            res(k, "auc", f2(r.auc, NA))
            res(k, "lo", f2(r.auc_lo95, NA))
            res(k, "hi", f2(r.auc_hi95, NA))
        order = (w.sort_values(["separation", "wavelength_nm"], ascending=[False, True])["name"]
                 .str.replace("-", " to ").tolist())
        name(f"RobSep{tag}TopWindows", wlist(order[:3]))


def main():
    os.makedirs(OUT, exist_ok=True)
    rob = pd.read_csv(analysis_file("contrast_robustness.csv"))
    contrast_macros(rob)
    contrast_tables_ci(rob)
    headline_table(rob)
    ndvi()
    holdout()
    structure()
    with open(os.path.join(OUT, "robustness.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    n_keyed = sum(1 for line in L if "csname res@" in line)
    print(f"wrote {os.path.relpath(OUT)}: robustness.tex with {len(NAMED)} named macros and {n_keyed} keyed values, "
          f"and {len([f for f in os.listdir(OUT) if f != 'robustness.tex'])} tables")


if __name__ == "__main__":
    main()
