"""Three checks of the holdout results, covering all ten submissions, the class balance of the holdout, and rank
agreement without the RGB submission.

1. All ten submissions. The paper reports the seven submissions whose learner and input it describes. The study made
   three more (a HybridSN-style network on the HS cube, FLAML on a simulated five-band camera, and shrinkage LDA on
   all views), whose code and per-fold results belong to work outside this paper and are not part of this
   repository. Their scores, predicted class counts, and CV accuracy with its range at the private size are recorded
   in results/leaderboard/other_submissions.csv, so that every submission can be reported.

2. Class priors of the holdout. Every FLAML submission scored above its CV accuracy, and most predicted many more Rust
   than Health patches. One explanation is label shift, a holdout whose class balance differs from the 200/200/200 of
   training, which adversarial validation on the inputs cannot detect. Black-box shift estimation (Lipton, Wang, and
   Smola 2018) estimates the holdout priors q from a classifier's predictions alone. If the class-conditional
   distributions are the same in training and holdout, the expected share of predictions per class is M q, where
   M[i, j] = P(predicted i | true j) comes from the out-of-fold predictions of the procedure the submission refits.
   Solving M q = (predicted shares) for q >= 0 with sum 1 gives one estimate per submission, and stacking the seven
   systems gives a joint estimate. Under q, a submission would be expected to reach sum_j q_j recall_j, which
   separates the part of its holdout gain that the class balance explains from the rest (a larger training set and
   chance). Intervals come from 2,000 bootstrap draws that resample training patches (in the out-of-fold
   predictions of one repeat at a time) and holdout patches, both jointly across submissions. No holdout label is
   used or guessed; the check reads only the submitted predictions, as the leaderboard did.

3. Rank agreement. Kendall's tau between CV accuracy and the leaderboard scores, over the paper's submissions, its
   FLAML ones, and all ten, each with and without the RGB submission S02, whose low scores could drive the rank
   agreement alone. The values for the paper's sets reproduce results/analysis/leaderboard_tau.csv.

Reads results/leaderboard/, results/analysis/leaderboard_vs_cv.csv and refit_gain.csv, the out-of-fold predictions of
the submitted procedures, and submissions/. Writes to results/analysis/
  holdout_all.csv    the ten submissions with predicted counts, CV accuracy, range, and scores
  label_shift.csv    per submission and jointly, the estimated holdout priors with 95 percent intervals, the CV recall
                     per class, and the accuracy expected under the estimated priors
  holdout_tau.csv    Kendall's tau per set of submissions and score
usage: python scripts/analysis/11_holdout_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import kendalltau

from scripts.analysis._shared import (LEADERBOARD, SUBMISSIONS, SUBMISSIONS_FLAML, analysis_file, read_experiment)
from wheatrust.io_data import CLASSES, ROOT

N_BOOT = 2000
SEED = 20261009
# the views of the FLAML submissions whose gain from more training data was measured (scripts/analysis/01)
REFIT_VIEW = {"S01": "all", "S02": "rgb", "S03": "ms", "S04": "hsc"}


def all_ten():
    man = pd.read_csv(os.path.join(LEADERBOARD, "submission_manifest.csv")).set_index("sid")
    kv = pd.read_csv(analysis_file("leaderboard_vs_cv.csv")).set_index("sid")
    rows = []
    for sid in SUBMISSIONS:
        m, k = man.loc[sid], kv.loc[sid]
        rows.append(dict(sid=sid, in_paper=True, key=k.key, message=m.message, n_Health=int(m.n_Health),
                         n_Rust=int(m.n_Rust), n_Other=int(m.n_Other), cv_accuracy=k.cv, range_lo95=k.lo95,
                         range_hi95=k.hi95, public=k.public, private=k.private))
    oth = pd.read_csv(os.path.join(LEADERBOARD, "other_submissions.csv"))
    for _, o in oth.iterrows():
        rows.append(dict(sid=o.sid, in_paper=False, key="", message=o.message, n_Health=int(o.n_Health),
                         n_Rust=int(o.n_Rust), n_Other=int(o.n_Other), cv_accuracy=o.cv_accuracy,
                         range_lo95=o.range_lo95, range_hi95=o.range_hi95, public=o.public_score,
                         private=o.private_score))
    df = pd.DataFrame(rows).sort_values("sid").reset_index(drop=True)
    df["private_minus_cv"] = df.private - df.cv_accuracy
    df["private_in_range"] = (df.private >= df.range_lo95) & (df.private <= df.range_hi95)
    return df


def confusion_given_true(y, pred, k=3):
    """M[i, j] = P(predicted i | true j); every column sums to one."""
    M = np.zeros((k, k))
    for j in range(k):
        m = y == j
        M[:, j] = np.bincount(pred[m], minlength=k) / m.sum()
    return M


def priors(Ms, shares):
    """The class priors q >= 0 with sum 1 that minimise sum_s ||M_s q - shares_s||^2. A heavily weighted extra row
    makes the sum one; the small remainder is normalised away."""
    A, b = np.vstack(Ms), np.concatenate(shares)
    w = 1e3 * len(Ms)
    q, _ = nnls(np.vstack([A, w * np.ones(A.shape[1])]), np.append(b, w))
    return q / q.sum()


def label_shift(ten):
    meta = json.load(open(os.path.join(LEADERBOARD, "meta.json")))
    man = pd.read_csv(os.path.join(LEADERBOARD, "submission_manifest.csv")).set_index("sid")
    refit = pd.read_csv(analysis_file("refit_gain.csv")).set_index("view").accuracy_gain
    exps = sorted({k.split(":")[0] for k in SUBMISSIONS.values()})
    O = pd.concat([read_experiment(e, "oof") for e in exps], ignore_index=True)
    sids = list(SUBMISSIONS)
    # out-of-fold labels and argmax predictions per repeat, rows sorted by sample so that a patch index means the
    # same patch for every submission
    Y, P = {}, {}
    for sid in sids:
        o = O[O.key == SUBMISSIONS[sid]].sort_values(["repeat", "sample_id"])
        reps = sorted(o.repeat.unique())
        Y[sid] = np.stack([o[o.repeat == r].y.to_numpy() for r in reps])
        P[sid] = np.stack([o[o.repeat == r][["p_Health", "p_Rust", "p_Other"]].to_numpy().argmax(1) for r in reps])
    ref = Y[sids[0]]
    assert all(np.array_equal(Y[s], ref) for s in sids), "out-of-fold predictions of different patches"
    n_rep, n_tr = ref.shape
    # holdout predictions, rows sorted by Id
    H = {}
    for sid in sids:
        s = pd.read_csv(os.path.join(ROOT, "submissions", man.loc[sid, "file"])).sort_values("Id")
        H[sid] = s.Category.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    n_ho = len(H[sids[0]])

    def estimate(rep_rows, ho_rows, rep=None):
        """Confusion matrices, predicted shares, and priors for one set of training and holdout patches."""
        Ms, mus = {}, {}
        for sid in sids:
            if rep is None:  # point estimate, averaged over every repeat
                Ms[sid] = np.mean([confusion_given_true(Y[sid][r], P[sid][r]) for r in range(n_rep)], axis=0)
            else:
                Ms[sid] = confusion_given_true(Y[sid][rep, rep_rows], P[sid][rep, rep_rows])
            mus[sid] = np.bincount(H[sid][ho_rows], minlength=3) / len(ho_rows)
        single = {sid: priors([Ms[sid]], [mus[sid]]) for sid in sids}
        joint = priors([Ms[s] for s in sids], [mus[s] for s in sids])
        return Ms, single, joint

    all_tr, all_ho = np.arange(n_tr), np.arange(n_ho)
    Ms, single, joint = estimate(all_tr, all_ho)
    rng = np.random.default_rng(SEED)
    boot_single = {sid: [] for sid in sids}
    boot_joint, boot_expected = [], {sid: [] for sid in sids}
    for b in range(N_BOOT):
        tr = rng.integers(0, n_tr, n_tr)
        ho = rng.integers(0, n_ho, n_ho)
        bM, bs, bj = estimate(tr, ho, rep=b % n_rep)
        for sid in sids:
            boot_single[sid].append(bs[sid])
            boot_expected[sid].append(float(bj @ np.diag(bM[sid])))
        boot_joint.append(bj)

    def interval(samples):
        a = np.asarray(samples)
        return np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)

    tenk = ten.set_index("sid")
    rows = []
    for sid in sids + ["joint"]:
        q = joint if sid == "joint" else single[sid]
        lo, hi = interval(boot_joint if sid == "joint" else boot_single[sid])
        row = dict(sid=sid, key=SUBMISSIONS.get(sid, "all seven"))
        for j, c in enumerate(CLASSES):
            row.update({f"prior_{c}": q[j], f"prior_{c}_lo95": lo[j], f"prior_{c}_hi95": hi[j]})
        if sid != "joint":
            rec = np.diag(Ms[sid])
            acc = float(np.mean([(P[sid][r] == Y[sid][r]).mean() for r in range(n_rep)]))
            exp_acc = float(joint @ rec)
            elo, ehi = interval(boot_expected[sid])
            pri = float(tenk.loc[sid, "private"])
            row.update({f"recall_{c}": rec[j] for j, c in enumerate(CLASSES)})
            row.update(n_pred_Health=int(man.loc[sid, "n_Health"]), n_pred_Rust=int(man.loc[sid, "n_Rust"]),
                       n_pred_Other=int(man.loc[sid, "n_Other"]), cv_accuracy_pooled=acc,
                       accuracy_expected=exp_acc, accuracy_expected_lo95=float(elo), accuracy_expected_hi95=float(ehi),
                       private=pri, gain_private=pri - acc, gain_from_priors=exp_acc - acc,
                       gain_from_training_size=refit.get(REFIT_VIEW.get(sid), np.nan))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(2, "n_holdout", n_ho)
    out.insert(3, "n_private", int(meta["n_private"]))
    return out


def taus(ten):
    sets = {"paper": list(SUBMISSIONS), "paper_flaml": SUBMISSIONS_FLAML, "all": list(ten.sid)}
    rows = []
    for name, members in sets.items():
        for drop in (False, True):
            ids = [s for s in members if not (drop and s == "S02")]
            g = ten[ten.sid.isin(ids)]
            for col in ["private", "public"]:
                t, p = kendalltau(g.cv_accuracy, g[col])
                rows.append(dict(set=name + ("_without_S02" if drop else ""), score=col, n=len(g), tau=t, p=p))
    out = pd.DataFrame(rows)
    old = pd.read_csv(analysis_file("leaderboard_tau.csv"))
    for _, r in old.iterrows():
        new = out[(out.set == r.set) & (out.score == r.score)].iloc[0]
        if abs(new.tau - r.tau) > 1e-12:
            sys.exit(f"Kendall's tau for {r.set}/{r.score} differs from leaderboard_tau.csv; stopping")
    return out


def main():
    ten = all_ten()
    ten.to_csv(analysis_file("holdout_all.csv"), index=False)
    ls = label_shift(ten)
    ls.to_csv(analysis_file("label_shift.csv"), index=False)
    tau = taus(ten)
    tau.to_csv(analysis_file("holdout_tau.csv"), index=False)
    print(ten[["sid", "in_paper", "cv_accuracy", "range_lo95", "range_hi95", "private", "private_in_range"]]
          .round(3).to_string(index=False))
    show = ["sid"] + [f"prior_{c}{s}" for c in CLASSES for s in ("", "_lo95", "_hi95")]
    print(ls[show].round(3).to_string(index=False))
    print(ls[["sid", "cv_accuracy_pooled", "accuracy_expected", "private", "gain_private", "gain_from_priors",
              "gain_from_training_size"]].dropna(subset=["private"]).round(3).to_string(index=False))
    print(tau[tau.score == "private"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
