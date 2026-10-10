"""Write the nine tables of the paper to outputs/tables/*.tex.

  features.tex   engineered feature families per view (Table 1)
  views.tex      spectral configuration and representation with FLAML (Table 2)
  hr.tex         the views on the Health versus Rust decision with both learners (Table 3)
  sensors.tex    band budgets and fixed placements with both learners (Table 4)
  holdout.tex    the seven holdout submissions of the paper with their leaderboard scores (Table 5)
  ablation.tex   feature-family and dimensionality ablations (Table A1)
  contrasts_f1.tex, contrasts_hr.tex, contrasts_learner.tex
                 every contrast the paper cites, with the three posteriors, the corrected t-test, and Holm within
                 the paper's families (Tables A2 to A4)
Captions refer to macros of outputs/numbers.tex (for example \\NOuterFolds), so the tables compile with it. Every
value comes from results/ and results/analysis/.
usage: python scripts/tables/02_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import json
import os

import pandas as pd

from scripts.analysis._shared import (FLAML_VIEW, LEADERBOARD, SUBMISSIONS, TABPFN_VIEW, all_fold_metrics,
                                      analysis_file)
from scripts.tables._latex import f3, pm, prob, pval, signed, verdict, write_table
from wheatrust.experiments import PCA_HSC
from wheatrust.io_data import OUTPUTS, RESULTS
from wheatrust.stats import correlated_bayes

T = os.path.join(OUTPUTS, "tables")
SOURCE = "scripts/tables/02_tables.py"
TV = TABPFN_VIEW


def write(name, caption, colspec, header, rows, label=None, note=None, size="\\small"):
    write_table(os.path.join(T, name + ".tex"), caption, colspec, header, rows, label or f"tab:{name}", note=note,
                size=size, source=SOURCE)


# what each feature family contains, as Table 1 prints it
FEATURE_TEXT = {
    ("rgb", "band"): "Per-channel statistics", ("rgb", "chroma"): "Chromatic coordinates r, g, b",
    ("rgb", "index"): "ExG, ExR, ExGR, VARI, GLI, NGRDI, CIVE, MGRVI, RGBVI, TGI",
    ("rgb", "hsv"): "Hue, saturation, value statistics and circular hue mean", ("rgb", "glcm"): "GLCM texture of luminance",
    ("ms", "band"): "Per-band statistics (482, 550, 650, 738, 834 nm)",
    ("ms", "index"): "NDVI, GNDVI, BNDVI, red-edge NDVI, NDREr, NDWI, OSAVI, SAVI, EVI, MCARI, CIgreen, CIrededge, MSR, SR, R/G",
    ("ms", "glcm"): "GLCM texture of NIR and NDVI",
    ("hsc", "band"): "Per-band mean and standard deviation (490 to 890 nm)",
    ("hsc", "index"): "NDVI, OSAVI, PSSRa, RENDVI, PRI, ARI, ARI2, CRI1, PSRI, MCARI, TCARI, TCARI/OSAVI, MTCI, CIgreen, CIrededge, GM1, GM2, VOG1, linear REP",
    ("hsc", "deriv"): "First-derivative edges, positions and sums; pixel-level REP and red-edge slope",
    ("hsc", "shape"): "Green peak, red well, NIR plateau, continuum-removed red well, entropy, band ratios",
    ("hsc", "pca"): "PCA scores of the patch mean spectrum (fitted in fold)",
    ("cross", "vegcolor"): "RGB colour of MS-vegetated and soil pixels", ("cross", "agree"): "Pixel agreement of ExG and NDVI",
    ("cross", "mask"): "Vegetation fraction from MS NDVI",
}


def features():
    """Table 1, the engineered feature families per view, counted from the feature dictionary."""
    fd = pd.read_csv(os.path.join(RESULTS, "features", "feature_dictionary.csv"))
    vname = {"rgb": "RGB", "ms": "MS", "hsc": "HS (cleaned)", "cross": "Cross-modal"}
    rows = []
    for t in ["rgb", "ms", "hsc", "cross"]:
        g = fd[fd.table == t].groupby("family").size()
        # the PCA scores are fitted inside each fold, so the feature dictionary does not list them
        fams = [f for f in ["band", "chroma", "index", "hsv", "glcm", "deriv", "shape", "pca", "vegcolor", "agree",
                            "mask"] if f in g.index or (t == "hsc" and f == "pca")]
        for i, fam in enumerate(fams):
            n = PCA_HSC["n"] if (t == "hsc" and fam == "pca") else int(g[fam])
            rows.append(f"{vname[t] if i == 0 else ''} & {fam} & {n} & {FEATURE_TEXT.get((t, fam), '')}")
        if "meta" in g.index:
            rows.append(f" & meta & {int(g['meta'])} & Valid-pixel fraction")
        rows.append("\\midrule")
    write("features", "Engineered feature families per view. Statistics over valid pixels are the mean, standard "
          "deviation, and 10th, 50th, and 90th percentiles; the all-views set concatenates the four blocks. Index "
          "abbreviations are expanded in the Abbreviations section.",
          "llrX", "\\textbf{View} & \\textbf{Family} & \\textbf{Count} & \\textbf{Content}", rows[:-1])


def views():
    """Table 2, every view and representation with FLAML against the all-views baseline."""
    e4 = pd.read_csv(analysis_file("e4_views.csv"))
    # the row names are those of Table 3, Figure 3, and the contrast tables
    vk = [("RGB PNG", ["E4:rgb_raw", "E4:rgb_eng"]), ("RGB features from HS bands", ["E4:rgbhs_eng"]),
          ("RGB features from stretched HS bands$^{\\dagger}$", ["E4:rgbhsst_eng"]),
          ("MS file", ["E4:ms_raw", "E4:ms_eng"]), ("MS features from HS bands", ["E4:mshs_eng"]),
          ("Cleaned HS (101 bands)", ["E4:hsc_raw", "E4:hsc_eng", "E4:hsc_pca"]),
          ("Full HS (125 bands)", ["E4:hsf_raw", "E4:hsf_eng", "E4:hsf_pca"]),
          ("All views", ["E4:all_raw", "E1:flaml_all"])]

    def rep(k):
        return "raw band statistics" if k.endswith("raw") else "PCA (10 scores)" if k.endswith("pca") else "engineered"

    rows = []
    for view, keys in vk:
        for i, k in enumerate(keys):
            r = e4[e4.variant == k].iloc[0]
            if k == "E1:flaml_all":
                rows.append(f"{view if i == 0 else ''} & engineered (baseline) & {pm(r.macro_f1_mean, r.macro_f1_sd)} "
                            f"& {f3(r.accuracy_mean)} & reference & & & ")
            else:
                rows.append(f"{view if i == 0 else ''} & {rep(k)} & {pm(r.macro_f1_mean, r.macro_f1_sd)} & "
                            f"{f3(r.accuracy_mean)} & {signed(r['diff'])} & {prob(r.p_better)} & {prob(r.p_equiv)} & "
                            f"{prob(r.p_worse)}")
    write("views", "Spectral configuration and representation with the FLAML baseline learner. The single-source rows "
          "compute every RGB or MS feature directly from the matching HS bands at native resolution, without the "
          "provider's resampling, stretch, or 8-bit quantisation. $^{\\dagger}$The stretch control, added after the "
          "first round of results, applies the per-image 2 to 98 percent stretch and 8-bit quantisation to those HS "
          "bands. Acc.\\ is accuracy. $\\Delta$ is the mean paired difference in macro-F1 from the all-views engineered "
          "baseline on the same folds, and the probabilities are the posteriors of the Bayesian correlated $t$-test "
          "that the row is better than the baseline by more than \\ROPE{}, within plus or minus \\ROPE{} of it, or "
          "worse by more than \\ROPE{}. Values are means $\\pm$ standard deviations over the \\NOuterFolds{} outer "
          "folds.", ">{\\raggedright\\arraybackslash}p{3.6cm}lcccccc",
          "\\textbf{View} & \\textbf{Representation} & \\textbf{Macro-F1} & \\textbf{Acc.} & \\textbf{$\\Delta$} & "
          "\\textbf{P(better)} & \\textbf{P(equiv.)} & \\textbf{P(worse)}", rows)


def health_rust(mf):
    """Table 3, the views on the Health versus Rust decision with both learners."""
    hr = pd.read_csv(analysis_file("hr_metrics.csv")).set_index("key")
    hv = pd.read_csv(analysis_file("hr_veg_metrics.csv")).set_index("key")
    he = pd.read_csv(analysis_file("health_errors.csv")).set_index("key")

    def auc(tab, k):
        return f3(tab.loc[k, "hr_auc_mean"]) if k in tab.index else "--"

    def har(k):
        return f"{100 * he.loc[k, 'health_as_rust']:.0f}" if k in he.index else "--"

    def mf1(k):
        return f3(mf.loc[k, "mean"]) if k in mf.index else "--"

    items = [("RGB PNG", "E4:rgb_eng", TV["rgb"]),
             ("RGB features from HS bands", "E4:rgbhs_eng", "E4t:tabpfn_rgbhs_eng"),
             ("RGB features from stretched HS bands$^{\\dagger}$", "E4:rgbhsst_eng", "E4t:tabpfn_rgbhsst_eng"),
             ("MS file", "E4:ms_eng", TV["ms"]),
             ("MS features from HS bands", "E4:mshs_eng", "E4t:tabpfn_mshs_eng"),
             ("Cleaned HS", "E4:hsc_eng", TV["hsc"]),
             ("All views", "E1:flaml_all", TV["all"])]
    rows = [f"{lab} & {auc(hr, a)} & {auc(hr, b)} & {auc(hv, a)} & {auc(hv, b)} & {har(a)} & {har(b)} & "
            f"{mf1(a)} & {mf1(b)}" for lab, a, b in items]
    keys = [k for _, a, b in items for k in (a, b)]
    sd_all = hr.loc[[k for k in keys if k in hr.index], "hr_auc_sd"]
    sd_veg = hv.loc[[k for k in keys if k in hv.index], "hr_auc_sd"]
    nveg_h, nveg_r = int(hv.n_veg_health.iloc[0]), int(hv.n_veg_rust.iloc[0])
    header = ("& \\multicolumn{2}{c}{\\textbf{AUC, all patches}} & \\multicolumn{2}{c}{\\textbf{AUC, vegetated}} & "
              "\\multicolumn{2}{c}{\\textbf{Healthy as Rust (\\%)}} & \\multicolumn{2}{c}{\\textbf{Macro-F1}} \\\\ "
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9} "
              "\\textbf{View} & FLAML & TabPFN & FLAML & TabPFN & FLAML & TabPFN & FLAML & TabPFN")
    write("hr", "Views on the disease-relevant decision, with both learners and engineered features. The Health versus "
          "Rust ROC AUC uses the Rust score $p_{\\mathrm{R}}/(p_{\\mathrm{H}}+p_{\\mathrm{R}})$ on the true Health and "
          "Rust patches of each outer fold, either all of them or only the vegetated ones (not empty and patch mean "
          f"NDVI of at least 0.5, {nveg_h} Health and {nveg_r} Rust patches). Healthy as Rust is the share of Health "
          "patches that the three-class argmax labelled Rust, pooled over outer folds. AUC and macro-F1 values are "
          "means over \\NOuterFolds{} outer folds, and the standard deviation of the AUC over folds ranged from "
          f"{sd_all.min():.3f} to {sd_all.max():.3f} on all patches and from {sd_veg.min():.3f} to {sd_veg.max():.3f} "
          "on vegetated patches. $^{\\dagger}$The stretch control applies the per-image stretch and 8-bit quantisation "
          "of the RGB product to the HS bands. Everything except FLAML macro-F1 was added after the first round of "
          "results.", ">{\\raggedright\\arraybackslash}p{4.6cm}*{8}{>{\\centering\\arraybackslash}X}", header, rows)


def sensors(mf):
    """Table 4, band budgets selected in fold and the fixed placements, with both learners."""
    hr = pd.read_csv(analysis_file("hr_metrics.csv")).set_index("key")

    def mf1(k):
        return pm(mf.loc[k, "mean"], mf.loc[k, "std"]) if k in mf.index else "--"

    def auc(k):
        return pm(hr.loc[k, "hr_auc_mean"], hr.loc[k, "hr_auc_sd"]) if k in hr.index else "--"

    rows = []
    for k in [1, 2, 3, 5, 8, 16, 32]:
        a, b = f"E9:bandsel_k{k}", f"E9t:tabpfn_bandsel_k{k}"
        rows.append(f"Selected in fold, {k} band{'s' if k > 1 else ''} & {k} & {mf1(a)} & {mf1(b)} & {auc(a)} & {auc(b)}")
    rows.append("\\midrule")
    for lab, v, nb in [("Fixed RGB placement (482, 550, 650 nm)", "fixed_rgb3", 3),
                       ("Fixed MS placement (482, 550, 650, 738, 834 nm)", "fixed_ms5", 5)]:
        a, b = f"E9:{v}", f"E9t:tabpfn_{v}"
        rows.append(f"{lab} & {nb} & {mf1(a)} & {mf1(b)} & {auc(a)} & {auc(b)}")
    write("sensors", "Band budgets and fixed placements with both learners. Every band set uses the same generic band "
          "features (per-band statistics and, up to 8 bands, pairwise normalised differences), so values are "
          "comparable within the table but sit below the engineered views. Selection runs inside each outer training "
          "fold and is shared by both learners. AUC is the Health versus Rust ROC AUC. Mean $\\pm$ standard deviation "
          "over \\NOuterFolds{} outer folds. The TabPFN columns and the AUC were added after the first round of "
          "results.", ">{\\raggedright\\arraybackslash}p{3.6cm}cXXXX",
          "\\textbf{Band set} & \\textbf{Bands} & \\textbf{Macro-F1, FLAML} & \\textbf{Macro-F1, TabPFN} & "
          "\\textbf{AUC, FLAML} & \\textbf{AUC, TabPFN}", rows)


# Display labels; the Kaggle messages were fixed when the files were submitted
LABELS = {"S01": "FLAML, all views (baseline)", "S02": "FLAML, RGB view", "S03": "FLAML, MS view",
          "S04": "FLAML, cleaned HS view", "S05": "TabPFN, all views",
          "S07": "FLAML, five bands selected on all training samples",
          "S09": "FLAML, RGB features from the HS bands"}
# S05 refits E2:tabpfn_all, whose automatic ensemble size resolves to eight members (tabpfn 9.1.0), as in every
# other TabPFN run of the paper. Its CV columns therefore quote the eight-member run on all views that the text
# reports (E3t:tabpfn8_all_f1_d0, within 0.002 of E2:tabpfn_all, which scripts/tables/03_check_statements.py
# checks), while its 95 percent range still comes from the out-of-fold predictions of E2:tabpfn_all.
CV_KEY = {"S05": TV["all"]}


def holdout(fm):
    """Table 5, the seven submissions of the paper with their CV estimates, ranges, and leaderboard scores."""
    man = pd.read_csv(os.path.join(LEADERBOARD, "submission_manifest.csv")).set_index("sid")
    meta = json.load(open(os.path.join(LEADERBOARD, "meta.json")))
    kv = pd.read_csv(analysis_file("leaderboard_vs_cv.csv")).set_index("sid")
    cv = pd.read_csv(analysis_file("leaderboard_cv_map.csv")).set_index("sid")
    fa = fm[fm.key.isin(list(CV_KEY.values()))].groupby("key")[["macro_f1", "accuracy"]].mean()
    for sid, k in CV_KEY.items():
        cv.loc[sid, "cv_mean"], cv.loc[sid, "cv_acc"] = fa.loc[k, "macro_f1"], fa.loc[k, "accuracy"]
    scored = kv["private"].notna().any()
    rows = []
    for sid in SUBMISSIONS:
        r, k, c = man.loc[sid], kv.loc[sid], cv.loc[sid]
        row = (f"{sid} & {LABELS[sid]} & {f3(c.cv_mean)} & {f3(c.cv_acc)} & {k.lo95:.3f} to {k.hi95:.3f} & "
               f"{int(r.n_Health)}/{int(r.n_Rust)}/{int(r.n_Other)}")
        rows.append(row + (f" & {k.public:.3f} & {k.private:.3f}" if scored else ""))
    met = str(kv.metric.iloc[0]).replace("_", " ")
    npriv = int(kv.n_private.iloc[0])
    sizes = f" on {meta['n_public']} and {npriv} holdout patches"
    write("holdout", "Holdout submissions of this paper, \\NSubmissionsAWord{} of the \\NSubmissionsWord{} the study "
          "made. CV columns are outer-fold means of the same procedure. The "
          f"range is the central 95 percent of CV-based {met} for a test set of {npriv} samples"
          + (", the size of the private split," if scored else "") + " from resampled "
          "out-of-fold predictions, which reflects the sampling of a test set of that size only, not the variability "
          "of refitting or the \\NOverlapTrainHoldout{} holdout patches that overlap training patches. Predicted counts "
          "are Health/Rust/Other among the \\NHoldout{} holdout samples."
          + (f" Public and Private give the leaderboard {met}{sizes}, recorded after all "
             "submissions were final; they played no part in choosing any model or configuration." if scored else ""),
          "lXcccccc" if scored else "lXcccc",
          "\\textbf{ID} & \\textbf{Model} & \\textbf{CV macro-F1} & \\textbf{CV acc.} & \\textbf{95\\% range} & "
          "\\textbf{Predicted}" + (" & \\textbf{Public} & \\textbf{Private}" if scored else ""), rows)


def ablation(fm):
    """Table A1, each ablation against the full feature set of its view on paired outer folds."""
    def scores(k):
        return fm[fm.key == k].set_index(["repeat", "fold"]).macro_f1

    def drow(label, a, b):
        A, B = scores(a), scores(b)
        i = A.index.intersection(B.index).sort_values()
        x, y = A.loc[i].values, B.loc[i].values
        pb, pe, pw = correlated_bayes(x, y)
        return f"{label} & {pm(x.mean(), x.std(ddof=1))} & {signed((x - y).mean())} & {prob(pb)} & {prob(pe)} & {prob(pw)}"

    ab = [("RGB view", FLAML_VIEW["rgb"], [("without band statistics", "E5:rgb_minus_band"),
                                           ("without chromatic coordinates", "E5:rgb_minus_chroma"),
                                           ("without colour indices", "E5:rgb_minus_index"),
                                           ("without HSV", "E5:rgb_minus_hsv"), ("without GLCM", "E5:rgb_minus_glcm")]),
          ("MS view", FLAML_VIEW["ms"], [("without band statistics", "E5:ms_minus_band"),
                                         ("without vegetation indices", "E5:ms_minus_index"),
                                         ("without GLCM", "E5:ms_minus_glcm")]),
          ("HS view (cleaned)", FLAML_VIEW["hsc"], [("without band statistics", "E5:hsc_minus_band"),
                                                   ("without indices", "E5:hsc_minus_index"),
                                                   ("without derivative features", "E5:hsc_minus_deriv"),
                                                   ("without shape features", "E5:hsc_minus_shape"),
                                                   ("without PCA scores", "E5:hsc_minus_pca"),
                                                   ("PCA only, 3 scores", "E5:hsc_pca3"),
                                                   ("PCA only, 5 scores", "E5:hsc_pca5"),
                                                   ("PCA only, 10 scores", "E4:hsc_pca"),
                                                   ("PCA only, 20 scores", "E5:hsc_pca20"),
                                                   ("PCA only, 95 percent variance", "E5:hsc_pca95")]),
          ("All views", FLAML_VIEW["all"], [("without RGB", "E5:all_minus_rgb"), ("without MS", "E5:all_minus_ms"),
                                            ("without HS", "E5:all_minus_hsc"),
                                            ("without cross-modal", "E5:all_minus_cross"),
                                            ("RFECV selection in the inner loop", "E5:all_rfecv")])]
    rows = []
    for view, ref, items in ab:
        r = scores(ref)
        rows.append(f"\\textit{{{view}}}, full set & {pm(r.mean(), r.std(ddof=1))} & reference & & & ")
        rows += [drow(lab, k, ref) for lab, k in items]
        rows.append("\\midrule")
    write("ablation", "Feature-family and dimensionality ablations with the FLAML baseline learner. $\\Delta$ is the "
          "mean paired difference in macro-F1 from the full feature set of the same view on the same outer folds, and "
          "the probabilities are the posteriors of the Bayesian correlated $t$-test that the variant is better by more "
          "than \\ROPE{}, within plus or minus \\ROPE{}, or worse by more than \\ROPE{}. Macro-F1 values are means "
          "$\\pm$ standard deviations over the \\NOuterFolds{} outer folds.", "Xccccc", "\\textbf{Variant} & \\textbf{Macro-F1} & \\textbf{$\\Delta$} & \\textbf{P(better)} & "
          "\\textbf{P(equiv.)} & \\textbf{P(worse)}", rows[:-1])


def contrast_tables():
    """Tables A2 to A4, every contrast the paper cites (results/analysis/contrasts.csv), block by family."""
    df = pd.read_csv(analysis_file("contrasts.csv"))
    titles = {"planned": "Planned contrasts, FLAML",
              "tabpfn": "TabPFN versions of the planned contrasts$^{\\dagger}$",
              "controls": "Controls added after the first round of results$^{\\dagger}$",
              "interaction": "Differences between two contrasts on the same folds$^{\\dagger}$",
              "hr_all": "Health versus Rust AUC, all Health and Rust patches$^{\\dagger}$",
              "hr_vegetated": "Health versus Rust AUC, vegetated patches only$^{\\dagger}$",
              "learner": "TabPFN minus FLAML on the same inputs$^{\\dagger}$"}
    header = ("\\textbf{Contrast (A minus B)} & \\textbf{Diff.} & \\textbf{P(A$>$B)} & \\textbf{P(equiv.)} & "
              "\\textbf{P(A$<$B)} & \\textbf{$p$} & \\textbf{$p$, Holm} & \\textbf{Verdict}")
    spec = ">{\\raggedright\\arraybackslash}X cccccc>{\\raggedright\\arraybackslash}p{2.75cm}"
    note = ("Diff.\\ is the mean paired difference over outer folds. The three probabilities are the posteriors of the "
            "Bayesian correlated $t$-test with a ROPE of plus or minus \\ROPE{}, $p$ is the two-sided corrected "
            "resampled $t$-test, and Holm adjusts $p$ within each family (block). The verdict applies the thresholds "
            "of Section~\\ref{sec:results} to the largest posterior. $^{\\dagger}$Added after the first round of "
            "results.")
    groups = [("contrasts_f1", ["planned", "tabpfn", "controls", "interaction"],
               "Macro-F1 contrasts cited in the paper, by family.", "tab:contrasts-f1"),
              ("contrasts_hr", ["hr_all", "hr_vegetated"],
               "Health versus Rust AUC contrasts cited in the paper, by family (ROPE of plus or minus \\ROPE{} AUC).",
               "tab:contrasts-hr"),
              ("contrasts_learner", ["learner"], "Macro-F1 of TabPFN minus FLAML on identical inputs and folds.",
               "tab:contrasts-learner")]
    for fname, fams, cap, label in groups:
        rows = []
        for fam in fams:
            g = df[df.family == fam]
            if rows:
                rows.append("\\midrule")
            rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{titles[fam]}}}}}")
            for _, r in g.iterrows():
                # the block title already names the learner
                desc = str(r.description)
                if fam in ("planned", "tabpfn"):
                    desc = desc.replace(", FLAML", "").replace(", TabPFN", "")
                desc = desc.replace("TabPFN minus FLAML, ", "")
                rows.append(f"{desc} & {signed(r['diff'])} & {prob(r.p_better)} & {prob(r.p_equiv)} & "
                            f"{prob(r.p_worse)} & {pval(r.p_corrected)} & {pval(r.p_holm)} & "
                            f"{verdict(r.p_better, r.p_equiv, r.p_worse)}")
        write(fname, cap, spec, header, rows, label, note=note, size="\\footnotesize")


def main():
    fm = all_fold_metrics(["key", "repeat", "fold", "macro_f1", "accuracy"])
    mf = fm.groupby("key").macro_f1.agg(["mean", "std", "size"])
    features()
    views()
    health_rust(mf)
    sensors(mf)
    holdout(fm)
    ablation(fm)
    contrast_tables()
    print(f"{len(os.listdir(T))} tables written to {os.path.relpath(T)}")


if __name__ == "__main__":
    main()
