"""Does a result depend on the machine it ran on?

The units of this study ran on three platforms (the platform column of every fold_metrics.csv). Two sets of
reruns, with unchanged code, seeds, and folds, measure how much that matters.
  results/REPRO   the repeat-0 units of the FLAML baseline (E1:flaml_all) and of TabPFN on all views (E2:tabpfn_all),
                  rerun on a second platform
  results/REPROX  the units that the post hoc controls are compared with (RGB features from the HS bands and eight
                  selected bands, with both learners), rerun on the platform where the controls ran, so that each
                  control contrast can also be computed with both of its sides from one platform
Floating-point results differ between processors and numerical libraries, so a rerun can end FLAML's search at
another configuration and change some predictions, and TabPFN's probabilities shift slightly. For each pair this
script reports the change of the mean macro-F1, the largest change in one fold, and the number of folds whose
predictions differ (argmax of the stored probabilities, which every oof.csv.gz rounds to five decimals).

Writes to results/analysis/
  platform_check.csv      per learner and fold of the REPRO pairs
  platform_check.json     their summary per learner
  platform_controls.json  the REPROX pairs, and the four control contrasts with both sides from one platform
usage: python scripts/analysis/06_platform_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from scripts.analysis._shared import RESULTS, analysis_file, paired_test

# each pair as (rerun folder, rerun variant, original folder, original variant)
PAIRS = {"flaml": ("REPRO", "E1_flaml_all", "E1", "flaml_all"),
         "tabpfn": ("REPRO", "E2_tabpfn_all", "E2", "tabpfn_all")}
PAIRS_X = {"flaml_rgbhs_eng": ("REPROX", "E4_rgbhs_eng", "E4", "rgbhs_eng"),
           "flaml_bandsel_k8": ("REPROX", "E9_bandsel_k8", "E9", "bandsel_k8"),
           "tabpfn_rgbhs_eng": ("REPROX", "E4t_tabpfn_rgbhs_eng", "E4t", "tabpfn_rgbhs_eng"),
           "tabpfn_bandsel_k8": ("REPROX", "E9t_tabpfn_bandsel_k8", "E9t", "tabpfn_bandsel_k8")}
# each control contrast as (control, the rerun of its comparison unit on the control's platform)
CTRL_X = {"st_vs_hs": (("E4", "rgbhsst_eng"), ("REPROX", "E4_rgbhs_eng")),
          "t_st_vs_hs": (("E4t", "tabpfn_rgbhsst_eng"), ("REPROX", "E4t_tabpfn_rgbhs_eng")),
          "k8nond_vs_k8": (("E9", "bandsel_k8_nond"), ("REPROX", "E9_bandsel_k8")),
          "t_k8nond_vs_k8": (("E9t", "tabpfn_bandsel_k8_nond"), ("REPROX", "E9t_tabpfn_bandsel_k8"))}


@lru_cache(maxsize=None)
def tables(folder):
    return (pd.read_csv(os.path.join(RESULTS, folder, "fold_metrics.csv"), low_memory=False),
            pd.read_csv(os.path.join(RESULTS, folder, "oof.csv.gz")))


def units(folder, variant):
    """Per (repeat, fold) of one variant, its macro-F1, argmax predictions, validation sample ids, and platform."""
    fm, oo = tables(folder)
    oo = oo[oo.variant == variant]
    pred = {k: (g.sample_id.values, g[["p_Health", "p_Rust", "p_Other"]].values.argmax(1))
            for k, g in oo.groupby(["repeat", "fold"])}
    out = {}
    for _, r in fm[fm.variant == variant].iterrows():
        k = (int(r["repeat"]), int(r.fold))
        out[k] = (float(r.macro_f1), pred[k][1], pred[k][0], str(r.platform))
    return out


def matched(a, b):
    keys = sorted(set(a) & set(b))
    for k in keys:
        assert list(a[k][2]) == list(b[k][2]), f"validation patches differ in fold {k}"
    return keys


def main():
    rows, summary = [], {}
    for lrn, (nf, nv, of, ov) in PAIRS.items():
        a, b = units(nf, nv), units(of, ov)
        for k in matched(a, b):
            rows.append(dict(learner=lrn, repeat=k[0], fold=k[1], platform_new=a[k][3], platform_old=b[k][3],
                             macro_f1_new=a[k][0], macro_f1_old=b[k][0], n_pred_diff=int((a[k][1] != b[k][1]).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(analysis_file("platform_check.csv"), index=False)
    for lrn, g in df.groupby("learner"):
        d = g.macro_f1_new - g.macro_f1_old
        summary[lrn] = dict(n_folds=int(len(g)), mean_new=float(g.macro_f1_new.mean()),
                            mean_old=float(g.macro_f1_old.mean()),
                            abs_mean_diff=float(abs(g.macro_f1_new.mean() - g.macro_f1_old.mean())),
                            max_fold_abs_diff=float(d.abs().max()),
                            n_folds_different_predictions=int((g.n_pred_diff > 0).sum()),
                            platforms_new=sorted(g.platform_new.unique()), platforms_old=sorted(g.platform_old.unique()))
    with open(analysis_file("platform_check.json"), "w") as f:
        json.dump(summary, f, indent=1)

    controls = {"pairs": {}, "controls": {}}
    for name, (nf, nv, of, ov) in PAIRS_X.items():
        a, b = units(nf, nv), units(of, ov)
        keys = matched(a, b)
        x, y = np.array([a[k][0] for k in keys]), np.array([b[k][0] for k in keys])
        controls["pairs"][name] = dict(n_folds=len(keys), mean_new=float(x.mean()), mean_old=float(y.mean()),
                                       abs_mean_diff=float(abs(x.mean() - y.mean())),
                                       max_fold_abs_diff=float(np.abs(x - y).max()),
                                       n_folds_different_predictions=int(sum((a[k][1] != b[k][1]).any() for k in keys)),
                                       platform_new=sorted({a[k][3] for k in keys}),
                                       platform_old=sorted({b[k][3] for k in keys}))
    for name, ((ca, va), (cb, vb)) in CTRL_X.items():
        a, b = units(ca, va), units(cb, vb)
        keys = matched(a, b)
        x, y = np.array([a[k][0] for k in keys]), np.array([b[k][0] for k in keys])
        pb, pe, pw, p = paired_test(x, y)
        controls["controls"][name] = dict(n_folds=len(keys), mean_a=float(x.mean()), mean_b=float(y.mean()),
                                          diff=float((x - y).mean()), p_better=pb, p_equiv=pe, p_worse=pw,
                                          p_corrected=p,
                                          platforms=sorted({a[k][3] for k in keys} | {b[k][3] for k in keys}))
    with open(analysis_file("platform_controls.json"), "w") as f:
        json.dump(controls, f, indent=1)
    print(df.round(4).to_string(index=False))
    for lrn, s in summary.items():
        print(f"{lrn}: mean macro-F1 changed by {s['abs_mean_diff']:.4f}, predictions differ in "
              f"{s['n_folds_different_predictions']} of {s['n_folds']} folds")


if __name__ == "__main__":
    main()
