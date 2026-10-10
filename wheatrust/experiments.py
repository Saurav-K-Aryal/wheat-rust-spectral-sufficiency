"""Experiment registry with every configuration the paper reports, as plain dict specs.

An experiment maps variant names to specs. A spec names an input (which features or bands), a learner, and
optionally a training subset, the repeats to run, and whether to keep the fitted fold models. Every variant runs on
the same saved outer folds (10 repeats of stratified grouped 5-fold cross-validation), so any two variants can be
compared fold by fold.

  E1      the FLAML baseline on all views
  E2      TabPFN on all views, ensemble size left to the library ('auto')
  E3f     FLAML on class-stratified random three-quarter subsets of every outer training fold, three draws,
          for the four views
  E3t     TabPFN with eight members on the four views
  E4      views and representations with FLAML (engineered features, raw band statistics, PCA scores, the full HS
          view, the single-source control, and the stretch control)
  E4t     TabPFN on the single-source and stretch controls
  E5      feature-family and dimensionality ablations with FLAML
  E9      band budgets from forward selection and the fixed RGB and MS placements, with FLAML
  E9t     the band sets of E9 with TabPFN
  REPRO   platform rerun of the E1 and E2 units of the first repeat
  REPROX  platform rerun of the units that the stretch and eight-band controls are compared with
The platform reruns are ordinary units of the experiments above, written under their own name so that they sit
beside the originals instead of replacing them.

The cross-batch transfer (results/E10) uses the inputs and splits defined at the end of this module.
"""
from __future__ import annotations

from wheatrust.common import N_REPEATS

REPEATS = list(range(N_REPEATS))
BUDGET = 50  # FLAML iterations per search, where the pilot search had levelled off
# FLAML 2.7.0's default classification portfolio without CatBoost, whose trials took far longer than the others
BASE_EST = ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth", "sgd", "lrl1"]
ALL_TABLES = ["rgb", "ms", "hsc", "cross"]
# 10 PCA scores of the patch mean spectrum, fitted inside each outer training fold
PCA_HSC = {"source": "hsc", "n": 10}
PCA_HSF = {"source": "hsf", "n": 10}


def flaml(budget=BUDGET, est=None, **kw):
    d = {"kind": "flaml", "estimator_list": est or BASE_EST, "budget": budget}
    d.update(kw)
    return d


def tabpfn(n_estimators=8):
    return {"kind": "tabpfn", "n_estimators": n_estimators}


def feats(tables, pca=None, **kw):
    """A feature input. Extra keys are exclude (families to drop) and filter ('band_only' keeps band statistics)."""
    d = {"kind": "features", "tables": tables}
    if pca:
        d["pca"] = pca
    d.update(kw)
    return d


BASELINE_INPUT = feats(ALL_TABLES, PCA_HSC)
VIEW_INPUTS = {
    "rgb": feats(["rgb"]),
    "ms": feats(["ms"]),
    "hsc": feats(["hsc"], PCA_HSC),
    "all": BASELINE_INPUT,
}


def E1():
    return {"flaml_all": {"input": BASELINE_INPUT, "learner": flaml()}}


def E2():
    return {"tabpfn_all": {"input": BASELINE_INPUT, "learner": {"kind": "tabpfn"}}}


LC_FRACS = {"f3_4": 3 / 4}
LC_DRAWS = 3


def E3_flaml():
    v = {}
    for view, inp in VIEW_INPUTS.items():
        for fn, fr in LC_FRACS.items():
            for d in range(LC_DRAWS):
                v[f"flaml_{view}_{fn}_d{d}"] = {"input": inp, "learner": flaml(),
                                                "subset": {"kind": "fraction", "frac": fr, "draw": d}}
    return v


def E3_tabpfn8():
    return {f"tabpfn8_{view}_f1_d0": {"input": inp, "learner": tabpfn(8)} for view, inp in VIEW_INPUTS.items()}


def E4():
    v = {}
    v["rgb_raw"] = {"input": feats(["rgb"], filter="band_only"), "learner": flaml()}
    v["rgb_eng"] = {"input": feats(["rgb"]), "learner": flaml()}
    v["ms_raw"] = {"input": feats(["ms"], filter="band_only"), "learner": flaml()}
    # the fold models of the two single-view models are kept for the window permutation
    v["ms_eng"] = {"input": feats(["ms"]), "learner": flaml(), "save_model": True}
    v["hsc_raw"] = {"input": feats(["hsc"], filter="band_only"), "learner": flaml()}
    v["hsc_eng"] = {"input": feats(["hsc"], PCA_HSC), "learner": flaml(), "save_model": True}
    v["hsc_pca"] = {"input": feats([], PCA_HSC), "learner": flaml()}
    v["hsf_raw"] = {"input": feats(["hsf"], filter="band_only"), "learner": flaml()}
    v["hsf_eng"] = {"input": feats(["hsf"], PCA_HSF), "learner": flaml()}
    v["hsf_pca"] = {"input": feats([], PCA_HSF), "learner": flaml()}
    # single-source control, the RGB and MS feature sets computed from the HS bands at the same positions
    v["rgbhs_eng"] = {"input": feats(["rgbhs"]), "learner": flaml()}
    v["mshs_eng"] = {"input": feats(["mshs"]), "learner": flaml()}
    # stretch control (post hoc), RGB features from the HS bands after the provider's stretch and 8-bit quantisation
    v["rgbhsst_eng"] = {"input": feats(["rgbhsst"]), "learner": flaml()}
    v["all_raw"] = {"input": feats(["rgb", "ms", "hsc"], filter="band_only"), "learner": flaml()}
    return v


def E4_tabpfn():
    e4 = E4()
    return {f"tabpfn_{v}": {"input": e4[v]["input"], "learner": tabpfn(8)}
            for v in ["rgbhs_eng", "mshs_eng", "rgbhsst_eng"]}


def E5():
    v = {}
    fam = {"rgb": ["band", "chroma", "index", "hsv", "glcm"], "ms": ["band", "index", "glcm"],
           "hsc": ["band", "index", "deriv", "shape"]}
    for view, fl in fam.items():
        for f in fl:
            inp = feats([view], PCA_HSC if view == "hsc" else None, exclude=[f"{view}:{f}"])
            v[f"{view}_minus_{f}"] = {"input": inp, "learner": flaml()}
    v["hsc_minus_pca"] = {"input": feats(["hsc"]), "learner": flaml()}
    v["all_minus_cross"] = {"input": feats(["rgb", "ms", "hsc"], PCA_HSC), "learner": flaml()}
    v["all_minus_rgb"] = {"input": feats(["ms", "hsc"], PCA_HSC), "learner": flaml()}
    v["all_minus_ms"] = {"input": feats(["rgb", "hsc"], PCA_HSC), "learner": flaml()}
    v["all_minus_hsc"] = {"input": feats(["rgb", "ms", "cross"]), "learner": flaml()}
    for n in [3, 5, 20]:
        v[f"hsc_pca{n}"] = {"input": feats([], {"source": "hsc", "n": n}), "learner": flaml()}
    # as many components as explain 95 percent of the variance in the training fold
    v["hsc_pca95"] = {"input": feats([], {"source": "hsc", "n": 0.95}), "learner": flaml()}
    # recursive feature elimination with LightGBM inside each outer training fold
    v["all_rfecv"] = {"input": {"kind": "rfecv", "base": BASELINE_INPUT}, "learner": flaml()}
    return v


E9_KS = [1, 2, 3, 5, 8, 16, 32]


def E9():
    v = {}
    for k in E9_KS:
        v[f"bandsel_k{k}"] = {"input": {"kind": "bandsel", "k": k}, "learner": flaml()}
    v["fixed_rgb3"] = {"input": {"kind": "bands", "wavelengths": [482, 550, 650]}, "learner": flaml()}
    v["fixed_ms5"] = {"input": {"kind": "bands", "wavelengths": [482, 550, 650, 738, 834]}, "learner": flaml()}
    # eight-band control (post hoc), the eight selected bands without the pairwise normalised differences
    v["bandsel_k8_nond"] = {"input": {"kind": "bandsel", "k": 8, "nd": False}, "learner": flaml()}
    return v


def E9_tabpfn():
    return {f"tabpfn_{v}": {"input": spec["input"], "learner": tabpfn(8)} for v, spec in E9().items()}


def REPRO():
    return {"E1_flaml_all": {**E1()["flaml_all"], "repeats": [0]},
            "E2_tabpfn_all": {**E2()["tabpfn_all"], "repeats": [0]}}


def REPROX():
    return {"E4_rgbhs_eng": E4()["rgbhs_eng"], "E9_bandsel_k8": E9()["bandsel_k8"],
            "E4t_tabpfn_rgbhs_eng": E4_tabpfn()["tabpfn_rgbhs_eng"],
            "E9t_tabpfn_bandsel_k8": E9_tabpfn()["tabpfn_bandsel_k8"]}


REGISTRY = {"E1": E1, "E2": E2, "E3f": E3_flaml, "E3t": E3_tabpfn8, "E4": E4, "E4t": E4_tabpfn, "E5": E5,
            "E9": E9, "E9t": E9_tabpfn, "REPRO": REPRO, "REPROX": REPROX}
PLATFORM_RERUNS = ("REPRO", "REPROX")


def get(exp):
    return REGISTRY[exp]()


# Cross-batch transfer (results/E10). The HS files come in two formats (125 or 126 bands) that behave like
# processing batches, so models are trained on one batch and tested on the other, against random splits of the same
# sizes and class counts. Five seeds per split, for FLAML and for TabPFN with eight members.
CROSSBATCH_INPUTS = {"all": BASELINE_INPUT, "ms": VIEW_INPUTS["ms"], "hsc": VIEW_INPUTS["hsc"]}
CROSSBATCH_SPLITS = ["125_to_126", "126_to_125", "random_489_to_111", "random_111_to_489"]
CROSSBATCH_SEEDS = list(range(5))
CROSSBATCH_LEARNERS = ["flaml", "tabpfn"]
