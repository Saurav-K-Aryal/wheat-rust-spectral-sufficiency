"""How much the paper's contrasts depend on the ROPE and on multiplicity, and which of them were planned.

Reads results/analysis/contrasts.csv (scripts/analysis/04_contrasts.py) and writes to results/analysis/

  contrast_robustness.csv   one row per contrast with
      ci95_lo, ci95_hi            central 95 percent credible interval of the mean paired difference
      p_positive                  posterior probability that the difference is above zero
      p_better_rNN, p_equiv_rNN, p_worse_rNN, verdict_rNN
                                  the three posteriors and their verbal label for a ROPE of 0.0NN (01, 02, 03)
      label_changes               whether the label at 0.01 or at 0.03 differs from the label at 0.02
      status                      confirmatory or exploratory (see STATUS below)
      p_holm_global               Holm over all contrasts
      q_bh_global, q_by_global    Benjamini-Hochberg and Benjamini-Yekutieli adjusted p-values over all contrasts
      p_holm_status, q_bh_status  Holm and Benjamini-Hochberg within the confirmatory or the exploratory contrasts
  multiplicity_summary.csv  per scope, status, and label, how many contrasts there are and how many of the labelled
                            differences each adjustment confirms at 0.05
  rope_summary.csv          per scope and status, how many labels change when the ROPE is 0.01 or 0.03

The posterior of the Bayesian correlated t-test is a Student t that depends on the paired fold differences only
through their number, mean, and standard deviation, which contrasts.csv stores, so no fold has to be read again
(wheatrust.stats.correlated_posterior). At the paper's ROPE of 0.02 the script reproduces the stored posteriors and
stops if they differ. The same ROPE values serve macro-F1 and the Health versus Rust AUC.

The p-values are those of the corrected resampled t-test. It shares the variance correction of the Bayesian test, so
the adjusted p-values are a multiplicity check on the same evidence rather than an independent confirmation.
Benjamini-Hochberg assumes independent or positively dependent tests; Benjamini-Yekutieli does not, and is given
beside it because contrasts that share methods and folds are correlated.

Scopes. 'appendix' holds the contrasts of the appendix tables of paired contrasts (every family except the
ablations), the set over which the paper counts clear and probable differences; 'ablation' holds the feature-family
and dimensionality ablations; 'all' holds both. The global adjustments always run over all contrasts.
usage: python scripts/analysis/09_contrast_robustness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import numpy as np
import pandas as pd

from scripts.analysis._shared import analysis_file
from wheatrust.stats import ROPE, benjamini_hochberg, correlated_posterior, holm, verdict

ROPES = (0.01, 0.02, 0.03)
ALPHA = 0.05

# Confirmatory contrasts were specified in the study plan before any result existed. They are the view,
# representation, single-source, and band-placement contrasts with FLAML (family 'planned') and the feature-family
# and dimensionality ablations ('ablation'). Every other family was added after the first round of results and is
# exploratory, namely the TabPFN versions of the planned contrasts, the stretch and eight-band controls, the
# differences between contrasts, TabPFN against FLAML on the same inputs, and the Health versus Rust AUC on all and
# on vegetated patches. The appendix tables mark the same families with a dagger.
STATUS = {"planned": "confirmatory", "ablation": "confirmatory"}

ADJUSTMENTS = {"holm_family": "p_holm", "holm_global": "p_holm_global", "bh_global": "q_bh_global",
               "by_global": "q_by_global", "holm_status": "p_holm_status", "bh_status": "q_bh_status"}
LABELS = ["clear difference", "probable difference", "leaning difference", "equivalence", "inconclusive"]


def tag(r):
    return f"r{round(100 * r):02d}"


def label_class(v):
    """'better, clear' becomes 'clear difference', 'leaning equivalent' becomes 'equivalence', and so on."""
    if v == "inconclusive":
        return v
    if v.startswith("leaning "):
        direction, strength = v.split(" ", 1)[1], "leaning"
    else:
        direction, strength = v.split(", ")
    return "equivalence" if direction == "equivalent" else f"{strength} difference"


def posteriors(df):
    """Credible intervals and the posteriors at every ROPE, with the check against the stored posteriors."""
    out = []
    for _, r in df.iterrows():
        post = correlated_posterior(r["diff"], r["diff_sd"], int(r["n_paired"]))
        row = dict(ci95_lo=post.ppf(0.025), ci95_hi=post.ppf(0.975), p_positive=1.0 - post.cdf(0.0))
        for rope in ROPES:
            # the same expressions as wheatrust.stats.correlated_bayes
            pw = post.cdf(-rope)
            pb = 1.0 - post.cdf(rope)
            pe = 1.0 - pb - pw
            t = tag(rope)
            row.update({f"p_better_{t}": pb, f"p_equiv_{t}": pe, f"p_worse_{t}": pw,
                        f"verdict_{t}": verdict(pb, pe, pw)})
        out.append(row)
    out = pd.DataFrame(out, index=df.index)
    t = tag(ROPE)
    worst = max(np.abs(out[f"{c}_{t}"] - df[c]).max() for c in ["p_better", "p_equiv", "p_worse"])
    if worst > 1e-9:
        sys.exit(f"the rebuilt posteriors differ from contrasts.csv by {worst:.2e} at ROPE {ROPE}; stopping")
    return out


def multiplicity(df):
    p = df.p_corrected.values
    df["p_holm_global"] = holm(p)
    df["q_bh_global"] = benjamini_hochberg(p)
    df["q_by_global"] = benjamini_hochberg(p, arbitrary_dependence=True)
    df["p_holm_status"] = np.nan
    df["q_bh_status"] = np.nan
    for _, g in df.groupby("status"):
        df.loc[g.index, "p_holm_status"] = holm(g.p_corrected.values)
        df.loc[g.index, "q_bh_status"] = benjamini_hochberg(g.p_corrected.values)
    return df


def scopes(df):
    return {"appendix": df[df.family != "ablation"], "ablation": df[df.family == "ablation"], "all": df}


def summaries(df):
    t = tag(ROPE)
    df["label"] = df[f"verdict_{t}"].map(label_class)
    mult, rope = [], []
    for scope, d in scopes(df).items():
        for status in ["confirmatory", "exploratory", "any"]:
            g = d if status == "any" else d[d.status == status]
            for lab in LABELS:
                h = g[g.label == lab]
                row = dict(scope=scope, status=status, label=lab, n=len(h))
                for name, col in ADJUSTMENTS.items():
                    # a p-value tests for a difference, so confirmation is counted only for difference labels
                    row[f"n_{name}"] = int((h[col] < ALPHA).sum()) if lab.endswith("difference") else np.nan
                mult.append(row)
            rope.append(dict(scope=scope, status=status, n=len(g),
                             n_changes_r01=int((g[f"verdict_{tag(0.01)}"] != g[f"verdict_{t}"]).sum()),
                             n_changes_r03=int((g[f"verdict_{tag(0.03)}"] != g[f"verdict_{t}"]).sum()),
                             n_changes_any=int(g.label_changes.sum())))
    return pd.DataFrame(mult), pd.DataFrame(rope)


def main():
    # read the floats back exactly; the default parser can change the last bit
    df = pd.read_csv(analysis_file("contrasts.csv"), float_precision="round_trip")
    df = pd.concat([df, posteriors(df)], axis=1)
    t = tag(ROPE)
    df["label_changes"] = (df[f"verdict_{tag(0.01)}"] != df[f"verdict_{t}"]) | \
                          (df[f"verdict_{tag(0.03)}"] != df[f"verdict_{t}"])
    df["status"] = df.family.map(STATUS).fillna("exploratory")
    df = multiplicity(df)
    mult, rope = summaries(df)
    df.drop(columns="label").to_csv(analysis_file("contrast_robustness.csv"), index=False)
    mult.to_csv(analysis_file("multiplicity_summary.csv"), index=False)
    rope.to_csv(analysis_file("rope_summary.csv"), index=False)

    print(f"{len(df)} contrasts, {int((df.status == 'confirmatory').sum())} confirmatory; posteriors at ROPE "
          f"{ROPE} reproduced")
    m = mult[(mult.scope == "appendix") & (mult.status == "any") & mult.label.str.endswith("difference")]
    print(m[["label", "n"] + [f"n_{k}" for k in ADJUSTMENTS]].to_string(index=False))
    r = rope[rope.status == "any"]
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
