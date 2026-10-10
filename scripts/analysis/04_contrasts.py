"""Every paired contrast the paper cites, grouped into the families within which Holm's adjustment runs.

A contrast 'a minus b' compares two methods on the outer folds they share (_shared.paired_test). Holm's step-down
adjustment of the corrected t-test p-values controls the family-wise error within each family, and the families
follow the structure of the paper:
  planned       the planned view and band-placement contrasts with FLAML
  tabpfn        the same contrasts with TabPFN as the learner
  controls      the stretch control, eight bands without pairwise differences, and raw band statistics
  interaction   differences between two contrasts on the same folds, for example the cost of the RGB product's
                processing minus the gain from the red-edge and NIR bands
  learner       TabPFN minus FLAML on identical inputs and folds
  hr_all        Health versus Rust AUC contrasts on every Health and Rust patch
  hr_vegetated  the same on vegetated patches only
  ablation      the feature-family and dimensionality ablations of Appendix A, each against its own full view
Every family except planned and ablation was added after the first round of results, and the appendix tables mark
them as such. Macro-F1 contrasts read results/analysis/all_fold_metrics.csv, and the AUC contrasts read
results/analysis/hr_folds.csv (scripts/analysis/03_health_rust.py). The ROPE is 0.02 in either unit.

Writes results/analysis/contrasts.csv, one row per contrast.
usage: python scripts/analysis/04_contrasts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import pandas as pd

from scripts.analysis._shared import TABPFN_VIEW as TV
from scripts.analysis._shared import all_fold_metrics, analysis_file, paired_test
from wheatrust.stats import holm

MACRO = {
    "planned": [
        ("stretch", "E4:rgbhs_eng", "E4:rgb_eng", "RGB features from HS bands minus RGB PNG, FLAML"),
        ("resample", "E4:mshs_eng", "E4:ms_eng", "MS features from HS bands minus MS file, FLAML"),
        ("ms_vs_rgbhs", "E4:ms_eng", "E4:rgbhs_eng", "MS minus RGB features from HS bands, FLAML"),
        ("hs_vs_ms", "E4:hsc_eng", "E4:ms_eng", "cleaned HS minus MS, FLAML"),
        ("clean", "E4:hsc_eng", "E4:hsf_eng", "cleaned HS minus full HS, FLAML"),
        ("eng_vs_raw", "E4:hsc_eng", "E4:hsc_raw", "cleaned HS engineered minus raw band statistics, FLAML"),
        ("eng_vs_pca", "E4:hsc_eng", "E4:hsc_pca", "cleaned HS engineered minus 10 PCA scores, FLAML"),
        ("ms5_vs_rgb3", "E9:fixed_ms5", "E9:fixed_rgb3", "fixed MS minus fixed RGB placement, FLAML"),
        ("k3_vs_rgb3", "E9:bandsel_k3", "E9:fixed_rgb3", "three selected bands minus fixed RGB placement, FLAML"),
        ("k5_vs_ms5", "E9:bandsel_k5", "E9:fixed_ms5", "five selected bands minus fixed MS placement, FLAML"),
    ],
    "tabpfn": [
        ("t_stretch", "E4t:tabpfn_rgbhs_eng", TV["rgb"], "RGB features from HS bands minus RGB PNG, TabPFN"),
        ("t_resample", "E4t:tabpfn_mshs_eng", TV["ms"], "MS features from HS bands minus MS file, TabPFN"),
        ("t_ms_vs_rgbhs", TV["ms"], "E4t:tabpfn_rgbhs_eng", "MS minus RGB features from HS bands, TabPFN"),
        ("t_hs_vs_ms", TV["hsc"], TV["ms"], "cleaned HS minus MS, TabPFN"),
        ("t_ms5_vs_rgb3", "E9t:tabpfn_fixed_ms5", "E9t:tabpfn_fixed_rgb3", "fixed MS minus fixed RGB placement, TabPFN"),
        ("t_k3_vs_rgb3", "E9t:tabpfn_bandsel_k3", "E9t:tabpfn_fixed_rgb3",
         "three selected bands minus fixed RGB placement, TabPFN"),
        ("t_k5_vs_ms5", "E9t:tabpfn_bandsel_k5", "E9t:tabpfn_fixed_ms5",
         "five selected bands minus fixed MS placement, TabPFN"),
        ("t_k32_vs_ms5", "E9t:tabpfn_bandsel_k32", "E9t:tabpfn_fixed_ms5",
         "32 selected bands minus fixed MS placement, TabPFN"),
    ],
    "controls": [
        ("st_vs_hs", "E4:rgbhsst_eng", "E4:rgbhs_eng",
         "stretched and quantised HS bands minus HS bands, RGB features, FLAML"),
        ("png_vs_st", "E4:rgb_eng", "E4:rgbhsst_eng", "RGB PNG minus stretched and quantised HS bands, FLAML"),
        ("t_st_vs_hs", "E4t:tabpfn_rgbhsst_eng", "E4t:tabpfn_rgbhs_eng",
         "stretched and quantised HS bands minus HS bands, RGB features, TabPFN"),
        ("t_png_vs_st", TV["rgb"], "E4t:tabpfn_rgbhsst_eng", "RGB PNG minus stretched and quantised HS bands, TabPFN"),
        ("k8nond_vs_k8", "E9:bandsel_k8_nond", "E9:bandsel_k8",
         "eight selected bands without minus with pairwise normalised differences, FLAML"),
        ("t_k8nond_vs_k8", "E9t:tabpfn_bandsel_k8_nond", "E9t:tabpfn_bandsel_k8",
         "eight selected bands without minus with pairwise normalised differences, TabPFN"),
        ("hs_vs_ms_raw", "E4:hsc_raw", "E4:ms_raw", "cleaned HS minus MS, raw band statistics, FLAML"),
    ],
    "learner": [(f"lrn_view_{v}", TV[v], {"rgb": "E4:rgb_eng", "ms": "E4:ms_eng", "hsc": "E4:hsc_eng",
                                          "all": "E1:flaml_all"}[v],
                 "TabPFN minus FLAML, " + {"rgb": "RGB PNG", "ms": "MS view", "hsc": "cleaned HS view",
                                           "all": "all views"}[v])
                for v in ("rgb", "ms", "hsc", "all")]
               + [("lrn_rgbhs", "E4t:tabpfn_rgbhs_eng", "E4:rgbhs_eng", "TabPFN minus FLAML, RGB features from HS bands"),
                  ("lrn_mshs", "E4t:tabpfn_mshs_eng", "E4:mshs_eng", "TabPFN minus FLAML, MS features from HS bands"),
                  ("lrn_fixed_rgb3", "E9t:tabpfn_fixed_rgb3", "E9:fixed_rgb3", "TabPFN minus FLAML, fixed RGB placement"),
                  ("lrn_fixed_ms5", "E9t:tabpfn_fixed_ms5", "E9:fixed_ms5", "TabPFN minus FLAML, fixed MS placement")]
               + [(f"lrn_bandsel_k{k}", f"E9t:tabpfn_bandsel_k{k}", f"E9:bandsel_k{k}",
                   f"TabPFN minus FLAML, {k} selected band{'s' if k > 1 else ''}") for k in (1, 2, 3, 5, 8, 16, 32)],
}

# The ablations are compared with the model of their own view, never with each other
ABLATION = [(f"e5_{v}_minus_{f}", f"E5:{v}_minus_{f}", ref, f"{v} view without {f} features minus {v} view")
            for v, ref, fams in [("rgb", "E4:rgb_eng", ["band", "chroma", "glcm", "hsv", "index"]),
                                 ("ms", "E4:ms_eng", ["band", "glcm", "index"]),
                                 ("hsc", "E4:hsc_eng", ["band", "deriv", "index", "pca", "shape"])]
            for f in fams]
ABLATION += [(f"e5_all_minus_{v}", f"E5:all_minus_{v}", "E1:flaml_all", f"all views without {v} minus all views")
             for v in ["rgb", "ms", "hsc", "cross"]]
ABLATION += [("e5_all_rfecv", "E5:all_rfecv", "E1:flaml_all",
              "RFECV inside the outer training fold minus all views")]
ABLATION += [(f"e5_{p}_vs_pca10", f"E5:hsc_{p}", "E4:hsc_pca", f"cleaned HS {p} minus 10 PCA scores")
             for p in ["pca3", "pca5", "pca20", "pca95"]]

HR = [("hs_vs_ms", "E4:hsc_eng", "E4:ms_eng"), ("ms_vs_rgbhs", "E4:ms_eng", "E4:rgbhs_eng"),
      ("stretch", "E4:rgbhs_eng", "E4:rgb_eng"), ("all_vs_hsc", "E1:flaml_all", "E4:hsc_eng"),
      ("ms5_vs_rgb3", "E9:fixed_ms5", "E9:fixed_rgb3"), ("k5_vs_ms5", "E9:bandsel_k5", "E9:fixed_ms5"),
      ("t_hs_vs_ms", TV["hsc"], TV["ms"]), ("t_ms_vs_rgbhs", TV["ms"], "E4t:tabpfn_rgbhs_eng"),
      ("t_stretch", "E4t:tabpfn_rgbhs_eng", TV["rgb"]), ("t_ms5_vs_rgb3", "E9t:tabpfn_fixed_ms5", "E9t:tabpfn_fixed_rgb3"),
      ("t_k5_vs_ms5", "E9t:tabpfn_bandsel_k5", "E9t:tabpfn_fixed_ms5"),
      ("hs_vs_ms_raw", "E4:hsc_raw", "E4:ms_raw")]
HR_DESC = {"hs_vs_ms": "cleaned HS minus MS", "ms_vs_rgbhs": "MS minus RGB features from HS bands",
           "stretch": "RGB features from HS bands minus RGB PNG", "all_vs_hsc": "all views minus cleaned HS",
           "ms5_vs_rgb3": "fixed MS minus fixed RGB placement", "k5_vs_ms5": "five selected bands minus fixed MS placement",
           "hs_vs_ms_raw": "cleaned HS minus MS, raw band statistics"}

# (name, (a1, b1, a2, b2), description): the contrast a1 minus b1 minus the contrast a2 minus b2, fold by fold
INTERACTIONS = [
    ("cost_minus_rededge_flaml", ("E4:rgbhs_eng", "E4:rgb_eng", "E4:ms_eng", "E4:rgbhs_eng"),
     "RGB processing cost minus red-edge and NIR gain, FLAML"),
    ("cost_minus_rededge_tabpfn", ("E4t:tabpfn_rgbhs_eng", TV["rgb"], TV["ms"], "E4t:tabpfn_rgbhs_eng"),
     "RGB processing cost minus red-edge and NIR gain, TabPFN"),
    ("rededge_gain_tabpfn_minus_flaml", (TV["ms"], "E4t:tabpfn_rgbhs_eng", "E4:ms_eng", "E4:rgbhs_eng"),
     "red-edge and NIR gain with TabPFN minus with FLAML"),
]

COLS = ["family", "name", "a", "b", "description", "metric", "subset", "n_paired", "mean_a", "mean_b", "diff",
        "diff_sd", "p_better", "p_equiv", "p_worse", "p_corrected", "p_holm"]


def fold_series(F, key, metric):
    g = F[F.key == key].dropna(subset=[metric])
    return g.groupby(["repeat", "fold"])[metric].mean()


def aligned(series):
    """The given per-fold series on the folds they all share, in (repeat, fold) order."""
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    return [s.loc[idx].values for s in series]


def family_rows(family, specs, metric, subset):
    """Rows of one family from specs (name, a, b, description, x, y), with Holm's adjustment within the family."""
    rows = []
    for name, a, b, desc, x, y in specs:
        if len(x) < 5:
            print(f"skipped {family}/{name} ({len(x)} paired folds)")
            continue
        pb, pe, pw, p = paired_test(x, y)
        rows.append(dict(family=family, name=name, a=a, b=b, description=desc, metric=metric, subset=subset,
                         n_paired=len(x), mean_a=x.mean(), mean_b=y.mean(), diff=(x - y).mean(),
                         diff_sd=(x - y).std(ddof=1), p_better=pb, p_equiv=pe, p_worse=pw, p_corrected=p))
    for r, h in zip(rows, holm([r["p_corrected"] for r in rows])):
        r["p_holm"] = h
    return rows


def main():
    fm = all_fold_metrics(["key", "repeat", "fold", "macro_f1"])
    # read the AUCs back exactly; the default CSV parser can change the last bit of a float
    hr = pd.read_csv(analysis_file("hr_folds.csv"), float_precision="round_trip")
    out = []
    for family, items in list(MACRO.items()) + [("ablation", ABLATION)]:
        specs = [(name, a, b, desc, *aligned([fold_series(fm, a, "macro_f1"), fold_series(fm, b, "macro_f1")]))
                 for name, a, b, desc in items]
        out += family_rows(family, specs, "macro_f1", "all")
    for subset, column in [("all", "hr_auc"), ("vegetated", "hr_auc_veg")]:
        specs = [(name, a, b, HR_DESC[name.removeprefix("t_")] + (", TabPFN" if name.startswith("t_") else ", FLAML"),
                  *aligned([fold_series(hr, a, column), fold_series(hr, b, column)])) for name, a, b in HR]
        out += family_rows(f"hr_{subset}", specs, "hr_auc", subset)
    specs = []
    for name, keys, desc in INTERACTIONS:
        s = aligned([fold_series(fm, k, "macro_f1") for k in keys])
        specs.append((name, f"{keys[0]} minus {keys[1]}", f"{keys[2]} minus {keys[3]}", desc, s[0] - s[1], s[2] - s[3]))
    out += family_rows("interaction", specs, "macro_f1", "all")
    pd.DataFrame(out, columns=COLS).to_csv(analysis_file("contrasts.csv"), index=False)
    print(f"{len(out)} contrasts in {len(set(r['family'] for r in out))} families written")


if __name__ == "__main__":
    main()
