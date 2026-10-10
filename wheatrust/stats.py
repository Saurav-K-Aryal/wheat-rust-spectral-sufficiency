"""Paired comparisons on repeated k-fold cross-validation, and holdout uncertainty.

Scores from repeated k-fold cross-validation are not independent, because the training sets of different folds
overlap. Both tests below therefore inflate the variance of the mean difference by a correlation rho between folds,
set to the test fraction 1/k (Nadeau and Bengio 2003), which is 0.2 for the 5-fold design used throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

ROPE = 0.02  # region of practical equivalence, in macro-F1


def paired(fm, a, b, metric="macro_f1"):
    """Per-fold scores of variants a and b on the (repeat, fold) units they share, aligned and sorted."""
    A = fm[fm.variant == a].set_index(["repeat", "fold"])[metric]
    B = fm[fm.variant == b].set_index(["repeat", "fold"])[metric]
    idx = A.index.intersection(B.index).sort_values()
    return A.loc[idx].values, B.loc[idx].values, idx


def correlated_bayes(x, y, n_folds=5, rope=ROPE):
    """Bayesian correlated t-test (Corani and Benavoli 2015; Benavoli et al. 2017) on paired CV scores.

    The posterior of the mean difference x - y is a Student t with J - 1 degrees of freedom, location mean(d), and
    scale sqrt((1/J + rho/(1 - rho)) var(d)), where J is the number of paired folds and rho = 1 / n_folds.
    Returns P(x better by more than rope), P(|difference| <= rope), and P(x worse by more than rope). It equals
    baycomp.CorrelatedTTest when J is a multiple of n_folds.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    J = len(d)
    if J < 2 or np.allclose(d.var(ddof=1), 0):
        m = d.mean() if J else 0.0
        return float(m > rope), float(abs(m) <= rope), float(m < -rope)
    rho = 1.0 / n_folds
    scale = np.sqrt((1.0 / J + rho / (1.0 - rho)) * d.var(ddof=1))
    post = st.t(df=J - 1, loc=d.mean(), scale=scale)
    p_worse = post.cdf(-rope)
    p_better = 1.0 - post.cdf(rope)
    return float(p_better), float(1.0 - p_better - p_worse), float(p_worse)


def correlated_posterior(mean_diff, sd_diff, n_paired, n_folds=5):
    """The posterior of correlated_bayes as a frozen scipy Student t, rebuilt from the number, mean, and standard
    deviation of the paired fold differences, which is all the posterior depends on. Its ppf gives credible
    intervals, and its cdf the probabilities for any ROPE."""
    rho = 1.0 / n_folds
    scale = np.sqrt((1.0 / n_paired + rho / (1.0 - rho)) * sd_diff ** 2)
    return st.t(df=n_paired - 1, loc=mean_diff, scale=scale)


def verdict(p_better, p_equiv, p_worse):
    """Verbal label of the three posteriors. The largest one is clear above 0.95, probable above 0.75, and leaning
    above 0.5, and otherwise the comparison is inconclusive. These are the thresholds stated in the Results."""
    lab, v = max([("better", p_better), ("equivalent", p_equiv), ("worse", p_worse)], key=lambda t: t[1])
    if v > 0.95:
        return f"{lab}, clear"
    if v > 0.75:
        return f"{lab}, probable"
    return f"leaning {lab}" if v > 0.5 else "inconclusive"


def corrected_ttest(x, y, n_train, n_test):
    """Corrected resampled t-test (Nadeau and Bengio 2003) for repeated k-fold CV, as (t, two-sided p).

    The variance of the mean difference is (1/J + n_test/n_train) var(d) instead of var(d) / J.
    """
    d = np.asarray(x) - np.asarray(y)
    J = len(d)
    if np.allclose(d.std(ddof=1), 0):
        return 0.0, 1.0
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * st.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


def holm(pvals):
    """Holm step-down adjusted p values (Holm 1979), for the contrasts of one family."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals, arbitrary_dependence=False):
    """Adjusted p-values (q-values) of the Benjamini-Hochberg step-up procedure, which controls the false discovery
    rate when the tests are independent or positively dependent (Benjamini and Hochberg 1995). With
    arbitrary_dependence the p-values are further multiplied by 1 + 1/2 + ... + 1/m, the Benjamini-Yekutieli
    procedure, which holds under any dependence (Benjamini and Yekutieli 2001)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    if arbitrary_dependence:
        scaled = scaled * np.sum(1.0 / np.arange(1, m + 1))
    q = np.minimum.accumulate(scaled[::-1])[::-1]  # step-up, so each q is the smallest bound from its rank onwards
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def compare_to(fm, baseline, variants, metric="macro_f1", n_train=480, n_test=120, rope=ROPE):
    """Every variant against one baseline on shared folds, with both tests and Holm over the variants compared."""
    rows = []
    for v in variants:
        if v == baseline:
            continue
        x, y, idx = paired(fm, v, baseline, metric)
        if len(x) < 5:
            continue
        pb, pe, pw = correlated_bayes(x, y, rope=rope)
        t, p = corrected_ttest(x, y, n_train, n_test)
        rows.append(dict(variant=v, baseline=baseline, n_paired=len(x), mean=x.mean(), sd=x.std(ddof=1),
                         baseline_mean=y.mean(), diff=(x - y).mean(), diff_sd=(x - y).std(ddof=1),
                         p_better=pb, p_equiv=pe, p_worse=pw, t_corrected=t, p_corrected=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df.p_corrected.values)
    return df


def summarize(fm, metrics=("macro_f1", "accuracy", "balanced_accuracy", "brier", "log_loss")):
    """Number of units, mean, and standard deviation of each metric per variant."""
    g = fm.groupby("variant")
    out = g.size().rename("n_units").to_frame()
    for m in metrics:
        if m in fm:
            out[f"{m}_mean"] = g[m].mean()
            out[f"{m}_sd"] = g[m].std(ddof=1)
    return out.reset_index()


def holdout_interval(oof, variant, sizes=(300, 150), n_boot=5000, seed=0, metric="macro_f1"):
    """Plausible score range on a test set of a given size, by resampling pooled out-of-fold predictions.

    Each bootstrap draw takes n samples with replacement from the pooled out-of-fold predictions of one repeat
    (every training sample once), cycling through the repeats, so the spread reflects test-set sampling at that
    size. Returns the mean and the 80 and 95 percent percentile intervals per size.
    """
    from sklearn.metrics import f1_score, accuracy_score

    rng = np.random.default_rng(seed)
    o = oof[oof.variant == variant]
    res = []
    for n in sizes:
        vals = []
        reps = sorted(o.repeat.unique())
        for b in range(n_boot):
            r = reps[b % len(reps)]
            d = o[o.repeat == r]
            i = rng.integers(0, len(d), n)
            yy = d.y.values[i]
            pp = d[["p_Health", "p_Rust", "p_Other"]].values[i].argmax(1)
            vals.append(f1_score(yy, pp, average="macro") if metric == "macro_f1" else accuracy_score(yy, pp))
        v = np.array(vals)
        res.append(dict(variant=variant, n=n, metric=metric, mean=v.mean(), lo95=np.percentile(v, 2.5),
                        hi95=np.percentile(v, 97.5), lo80=np.percentile(v, 10), hi80=np.percentile(v, 90)))
    return pd.DataFrame(res)
