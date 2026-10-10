"""Compare the leaderboard scores of the paper's seven submissions with cross-validation.

Reads results/leaderboard/ (submission_manifest.csv, scores.csv, meta.json), results/analysis/all_fold_metrics.csv,
the out-of-fold predictions of E1, E2, E4, and E9, and the submission files in submissions/. Writes to
results/analysis/
  leaderboard_cv_map.csv  per submission, the cross-validated procedure it refits, with its CV macro-F1 and accuracy
  leaderboard_vs_cv.csv   per submission, the CV estimate on the leaderboard metric, the public and private scores, the
                          central 95 percent range of that metric for a test set the size of the private split, and
                          whether the private score falls in it
  leaderboard_tau.csv     Kendall's tau between CV means and scores, over the seven submissions and over the six
                          FLAML ones
  leaderboard_pairs.csv   per pair of submissions, their agreement on the holdout and in CV, the differences in correct
                          patches, and an upper bound on the p-value of the paired sign test
The range resamples the pooled out-of-fold predictions of one repeat at a time at the private size
(wheatrust.stats.holdout_interval), so it reflects the sampling of a test set of that size only. The scores were
recorded after every submission was final and played no part in any choice of model or configuration.
usage: python scripts/analysis/08_leaderboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau

from scripts.analysis._shared import (LEADERBOARD, SUBMISSIONS, SUBMISSIONS_FLAML, all_fold_metrics, analysis_file,
                                      read_experiment)
from wheatrust.io_data import CLASSES, ROOT
from wheatrust.stats import holdout_interval

SETS = {"paper": list(SUBMISSIONS), "paper_flaml": SUBMISSIONS_FLAML}
CLASS_NAMES = np.array(CLASSES)


def norm(x):
    """Kaggle shows scores as fractions and the organisers quote percentages; store fractions."""
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x > 1.0, x / 100.0, x)


def sign_bound(d, n_max):
    """Upper bound on the two-sided exact sign-test p-value for two submissions whose numbers of correct patches
    differ by d, when their predictions differ on n_max patches.

    The sign test would use the discordant patches, those that exactly one of the two classifies correctly, but the
    leaderboard reveals only how many patches each gets right. With three classes a patch on which the predictions
    differ can also be wrong for both, so the number of discordant patches can be any n with d <= n <= n_max and
    n = d (mod 2). Each such n gives the p-value of (n + d)/2 successes in n fair trials, and the largest of them
    bounds the true p-value from above."""
    d = abs(int(d))
    if d == 0:
        return 1.0
    ps = [binomtest((n + d) // 2, n, 0.5).pvalue for n in range(d, int(n_max) + 1, 2)]
    return max(ps) if ps else float("nan")


def cv_map(fm):
    rows = []
    for sid, key in SUBMISSIONS.items():
        g = fm[fm.key == key]
        rows.append(dict(sid=sid, key=key, cv_mean=g.macro_f1.mean(), cv_sd=g.macro_f1.std(ddof=1),
                         cv_acc=g.accuracy.mean(), n_units=len(g)))
    out = pd.DataFrame(rows)
    out.to_csv(analysis_file("leaderboard_cv_map.csv"), index=False)
    return out


def pairs(man, sc, meta, cvm, O):
    """Paired comparisons of submissions from the numbers of correct patches the scores imply."""
    if sc[["public_score", "private_score"]].isna().any().any():
        return pd.DataFrame()
    pred = {r.sid: pd.read_csv(os.path.join(ROOT, "submissions", r.file)).set_index("Id").Category
            for _, r in man.iterrows()}
    s = sc.set_index("sid")
    cor_pub = (s.public_score * meta["n_public"]).round().astype(int)
    cor_pri = (s.private_score * meta["n_private"]).round().astype(int)
    oof = {}
    for sid, k in cvm.set_index("sid").key.items():
        o = O[O.variant == k]
        oof[sid] = pd.Series(CLASS_NAMES[o[["p_Health", "p_Rust", "p_Other"]].values.argmax(1)],
                             index=pd.MultiIndex.from_arrays([o.repeat.values, o.sample_id.values]))
    rows = []
    for a, b in itertools.combinations(list(man.sid), 2):
        pa, pb = pred[a], pred[b].reindex(pred[a].index)
        n_dis = int((pa != pb).sum())
        j = pd.concat([oof[a].rename("a"), oof[b].rename("b")], axis=1, join="inner")
        cv_agree = j.groupby(level=0).apply(lambda g: float((g.a == g.b).mean())).mean() if len(j) else np.nan
        d_all = int(cor_pub[a] + cor_pri[a] - cor_pub[b] - cor_pri[b])
        d_pri = int(cor_pri[a] - cor_pri[b])
        rows.append(dict(a=a, b=b, n_holdout=len(pa), n_disagree=n_dis, holdout_agree=1 - n_dis / len(pa),
                         cv_agree=cv_agree, correct_all_a=int(cor_pub[a] + cor_pri[a]),
                         correct_all_b=int(cor_pub[b] + cor_pri[b]), d_all=d_all, d_private=d_pri,
                         p_bound_all=sign_bound(d_all, n_dis), p_bound_private=sign_bound(d_pri, n_dis)))
    out = pd.DataFrame(rows)
    out.to_csv(analysis_file("leaderboard_pairs.csv"), index=False)
    return out


def main():
    man = pd.read_csv(os.path.join(LEADERBOARD, "submission_manifest.csv"))
    sc = pd.read_csv(os.path.join(LEADERBOARD, "scores.csv"))
    for c in ["public_score", "private_score"]:
        sc[c] = norm(sc[c])
    meta = json.load(open(os.path.join(LEADERBOARD, "meta.json")))
    metric = meta.get("metric", "accuracy")
    n_priv = int(round(meta["n_holdout"] * meta["private_fractions"][0]))
    fm = all_fold_metrics(["key", "repeat", "fold", "macro_f1", "accuracy"])
    cvm = cv_map(fm)
    # out-of-fold predictions of the procedures the submissions refit
    exps = sorted({k.split(":")[0] for k in SUBMISSIONS.values()})
    O = pd.concat([read_experiment(e, "oof") for e in exps], ignore_index=True)
    O["variant"] = O.key  # holdout_interval selects rows by variant
    rows = []
    for _, r in cvm.iterrows():
        f = fm[fm.key == r.key]
        cv = f[metric].mean()
        h = holdout_interval(O, r.key, sizes=(n_priv,), n_boot=4000, metric=metric).iloc[0]
        s = sc[sc.sid == r.sid]
        pub = float(s.public_score.iloc[0]) if len(s) else np.nan
        pri = float(s.private_score.iloc[0]) if len(s) else np.nan
        rows.append(dict(sid=r.sid, key=r.key, metric=metric, cv=cv, cv_sd=f[metric].std(), n_private=n_priv,
                         lo95=h.lo95, hi95=h.hi95, public=pub, private=pri, private_minus_cv=pri - cv,
                         private_in_range=bool(h.lo95 <= pri <= h.hi95) if np.isfinite(pri) else np.nan))
    kv = pd.DataFrame(rows)
    kv.to_csv(analysis_file("leaderboard_vs_cv.csv"), index=False)
    taus = []
    for set_name, members in SETS.items():
        g0 = kv[kv.sid.isin(members)]
        for col in ["private", "public"]:
            g = g0.dropna(subset=[col])
            if len(g) >= 3:
                t, p = kendalltau(g.cv, g[col])
                taus.append(dict(set=set_name, score=col, metric=metric, n=len(g), tau=t, p=p))
    tau = pd.DataFrame(taus, columns=["set", "score", "metric", "n", "tau", "p"])
    tau.to_csv(analysis_file("leaderboard_tau.csv"), index=False)
    pr = pairs(man, sc, meta, cvm, O)
    print(kv.round(3).to_string(index=False))
    print(tau.round(3).to_string(index=False))
    print(f"{len(pr)} pairs of submissions compared")


if __name__ == "__main__":
    main()
