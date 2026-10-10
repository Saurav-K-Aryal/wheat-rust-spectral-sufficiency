"""Window permutation of the single-view models, scored on the Health versus Rust decision.

Grouped permutation importance of wavelength windows for the cleaned-HS model (E4 hsc_eng) and the MS model
(E4 ms_eng). In each outer validation fold, the bands of one window are permuted across the validation patches with
one permutation shared by all bands of the window, so whole 32 x 32 sub-cubes of those bands are swapped between
patches. The features are recomputed from the permuted cubes and the fold model is scored again, three times per
window and fold. Each row records the drop in three-class macro-F1 and in the Health versus Rust AUC of the Rust
score p_Rust / (p_Health + p_Rust) on the true Health and Rust patches.

Windows are read from single-view models because permuting a window of the all-views model would also permute the
RGB, MS, and cross-modal inputs derived from its bands. For the cleaned HS cube, ten 40 nm windows partition the
cleaned range exactly (490 to 530 nm up to 850 to 890 nm, the last one including 890 nm); for MS, each band is
permuted on its own.

  flaml    the saved E4 fold models (scripts/training/05_run_units.py --exp E4). Every permutation is also scored on
           vegetated patches only, those that are not empty and have a patch mean NDVI of at least 0.5
           (results/audit/composition_ndvi.csv), because part of the Health versus Rust separation can come from
           canopy cover alone.
  tabpfn8  TabPFN with eight members fitted in every fold on the same inputs, with the same permutations.

Reads   the array and feature caches, results/folds/folds.csv, results/E4/models, and results/audit
Writes  results/E8/hr_units/<learner>/r<repeat>_f<fold>.pkl and results/E8/hr_units_veg/flaml/r<repeat>_f<fold>.pkl
        results/E8/hr_window_folds.csv and hr_window_summary.csv (both learners, all patches)
        results/E8/hr_window_folds_veg.csv and hr_window_summary_veg.csv (FLAML, vegetated patches)
usage: python scripts/training/06_window_permutation.py [--learners flaml tabpfn8] [--repeats 0-9] [--folds 0-4]
                                                       [--workers N]
       python scripts/training/06_window_permutation.py --collect
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse  # noqa: E402
import glob  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wheatrust.common import N_FOLDS, fold_indices, load_folds, metrics, table  # noqa: E402
from wheatrust.experiments import REPEATS, VIEW_INPUTS  # noqa: E402
from wheatrust.features import hs_features, ms_features  # noqa: E402
from wheatrust.io_data import CLEAN_IDX, HS_WL, RESULTS  # noqa: E402
from wheatrust.runner import arrays, assemble, model_path, parse_repeats, single_thread  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(RESULTS, "E8")
N_PERM = 3
MS_WL = [482, 550, 650, 738, 834]
LEARNERS = ["flaml", "tabpfn8"]


def hs_windows():
    """(name, HS band indices) of the ten 40 nm windows over the cleaned range."""
    wl_clean = HS_WL[CLEAN_IDX]
    out = []
    for i in range(10):
        lo, hi = 490 + 40 * i, 530 + 40 * i
        sel = (wl_clean >= lo) & ((wl_clean < hi) if i < 9 else (wl_clean <= 890))
        out.append((f"{lo}-{hi if i < 9 else 890}", np.asarray(CLEAN_IDX)[sel]))
    return out


def rust_auc(y, P, keep=None):
    """Health versus Rust ROC AUC of p_Rust / (p_Health + p_Rust) on the true Health and Rust patches in keep."""
    from sklearn.metrics import roc_auc_score

    m = (y == 0) | (y == 1)
    if keep is not None:
        m &= keep
    if len(np.unique(y[m])) < 2:
        return np.nan
    s = P[m, 1] / (P[m, 0] + P[m, 1] + 1e-12)
    return roc_auc_score((y[m] == 1).astype(int), s)


def vegetated_ids():
    comp = pd.read_csv(os.path.join(RESULTS, "audit", "composition_ndvi.csv"))
    return set(comp[(~comp["empty"].astype(bool)) & (comp.ndvi_mean >= 0.5)].sample_id)


def fold_data(r, k):
    """Training and validation positions of an outer fold, with the validation cubes as reflectance."""
    A = arrays()
    ids = A.sample_id[A.tr]
    y_all = A.y_train()
    tr, va = fold_indices(load_folds(), ids, r, k)
    pos = {s: i for i, s in enumerate(A.sample_id)}
    pv = [pos[s] for s in ids[va]]
    hs = np.asarray(A.hs[pv], dtype=np.float32) / 10000.0
    ms = np.asarray(A.ms[pv], dtype=np.float32) / 10000.0
    return dict(ids=ids, y_all=y_all, tr=tr, va=va, y=y_all[va], seed=1000 * r + k, hs=hs, ms=ms,
                mask_hs=~(hs == 0).all(-1), mask_ms=~(ms == 0).all(-1))


def hsc_tables(d, cube):
    """Cleaned-HS feature table and mean spectra of the validation patches, recomputed from a (permuted) cube."""
    ids_va = d["ids"][d["va"]]
    T, spec = hs_features(cube[..., CLEAN_IDX], HS_WL[CLEAN_IDX], d["mask_hs"], "hsc")
    T.index = ids_va
    return {"hsc": T.astype(np.float32),
            "spectra_hsc": pd.DataFrame(spec, index=ids_va, columns=[f"{int(w)}" for w in HS_WL[CLEAN_IDX]])}


def ms_tables(d, cube):
    T = ms_features(cube, d["mask_ms"], "ms")
    T.index = d["ids"][d["va"]]
    return {"ms": T.astype(np.float32)}


def unit_flaml(r, k, veg_ids):
    """Permutations of one fold for the saved FLAML models, as (rows on all patches, rows with vegetated AUC)."""
    d = fold_data(r, k)
    ids, tr, va, y, seed, hs, ms = d["ids"], d["tr"], d["va"], d["y"], d["seed"], d["hs"], d["ms"]
    veg = np.array([s in veg_ids for s in ids[va]])
    rows, rows_veg = [], []
    rng = np.random.default_rng(seed + 77)

    def add(model, window, n_bands, rep, base, new):
        (b_f1, b_auc, b_veg), (f1, auc, aveg) = base, new
        rows.append(dict(model=model, window=window, n_bands=n_bands, repeat=r, fold=k, perm=rep, base_macro_f1=b_f1,
                         base_hr_auc=b_auc, drop_macro_f1=b_f1 - f1, drop_hr_auc=b_auc - auc))
        rows_veg.append(dict(model=model, window=window, repeat=r, fold=k, perm=rep, base_hr_auc=b_auc,
                             base_hr_auc_veg=b_veg, drop_macro_f1=b_f1 - f1, drop_hr_auc=b_auc - auc,
                             drop_hr_auc_veg=b_veg - aveg))

    def scorer(model, T_tr, tables_of):
        def score(cube):
            _, Xva = assemble(model["spec"]["input"], T_tr, tables_of(d, cube), seed)
            P = np.asarray(model["learner"].predict_proba(Xva))
            return metrics(y, P)["macro_f1"], rust_auc(y, P), rust_auc(y, P, veg)
        return score

    with open(model_path("E4", "hsc_eng", r, k), "rb") as f:
        M = pickle.load(f)
    score = scorer(M, {"hsc": table("hsc").loc[ids[tr]], "spectra_hsc": table("spectra_hsc").loc[ids[tr]]},
                   hsc_tables)
    base = score(hs)
    for window, bands in hs_windows():
        for rep in range(N_PERM):
            perm = rng.permutation(len(va))
            h2 = hs.copy()
            h2[..., bands] = hs[perm][..., bands]
            add("E4:hsc_eng", window, int(len(bands)), rep, base, score(h2))

    with open(model_path("E4", "ms_eng", r, k), "rb") as f:
        M2 = pickle.load(f)
    score = scorer(M2, {"ms": table("ms").loc[ids[tr]]}, ms_tables)
    base = score(ms)
    for j, wl in enumerate(MS_WL):
        for rep in range(N_PERM):
            perm = rng.permutation(len(va))
            m2 = ms.copy()
            m2[..., j] = ms[perm][..., j]
            add("E4:ms_eng", f"{wl}", 1, rep, base, score(m2))
    return rows, rows_veg


def unit_tabpfn(r, k):
    """Permutations of one fold for TabPFN with eight members, fitted on the outer training part of the fold.

    The permutations are drawn in the same order as for FLAML. The permuted copies of the validation fold are
    predicted eight at a time (960 rows per call, which bounds memory); TabPFN scores every test row against the
    training rows only, so stacking the copies leaves each prediction unchanged up to floating-point rounding.
    """
    from wheatrust.learners import TabPFNLearner

    d = fold_data(r, k)
    ids, y_all, tr, va, y, seed, hs, ms = d["ids"], d["y_all"], d["tr"], d["va"], d["y"], d["seed"], d["hs"], d["ms"]
    rng = np.random.default_rng(seed + 77)
    perms_hs = [(w, b, rep, rng.permutation(len(va))) for w, b in hs_windows() for rep in range(N_PERM)]
    perms_ms = [(j, rep, rng.permutation(len(va))) for j in range(5) for rep in range(N_PERM)]

    def run(inp, T_tr, tables_of, cubes):
        mats = [assemble(inp, T_tr, tables_of(d, cube), seed)[1] for cube in cubes]
        Xtr, _ = assemble(inp, T_tr, T_tr, seed)
        L = TabPFNLearner(seed=seed, n_estimators=8).fit(Xtr, y_all[tr])
        P = np.vstack([np.asarray(L.predict_proba(np.vstack(mats[c:c + 8]))) for c in range(0, len(mats), 8)])
        return [P[i * len(va):(i + 1) * len(va)] for i in range(len(mats))]

    rows = []
    cubes = [hs]
    for _, bands, _, perm in perms_hs:
        h2 = hs.copy()
        h2[..., bands] = hs[perm][..., bands]
        cubes.append(h2)
    T_tr = {"hsc": table("hsc").loc[ids[tr]], "spectra_hsc": table("spectra_hsc").loc[ids[tr]]}
    Ps = run(VIEW_INPUTS["hsc"], T_tr, hsc_tables, cubes)
    b_f1, b_auc = metrics(y, Ps[0])["macro_f1"], rust_auc(y, Ps[0])
    for (w, bands, rep, _), P in zip(perms_hs, Ps[1:]):
        rows.append(dict(model="tabpfn8:hsc", window=w, n_bands=len(bands), repeat=r, fold=k, perm=rep,
                         base_macro_f1=b_f1, base_hr_auc=b_auc, drop_macro_f1=b_f1 - metrics(y, P)["macro_f1"],
                         drop_hr_auc=b_auc - rust_auc(y, P)))
    cubes = [ms]
    for j, _, perm in perms_ms:
        m2 = ms.copy()
        m2[..., j] = ms[perm][..., j]
        cubes.append(m2)
    Ps = run(VIEW_INPUTS["ms"], {"ms": table("ms").loc[ids[tr]]}, ms_tables, cubes)
    b_f1, b_auc = metrics(y, Ps[0])["macro_f1"], rust_auc(y, Ps[0])
    for (j, rep, _), P in zip(perms_ms, Ps[1:]):
        rows.append(dict(model="tabpfn8:ms", window=f"{MS_WL[j]}", n_bands=1, repeat=r, fold=k, perm=rep,
                         base_macro_f1=b_f1, base_hr_auc=b_auc, drop_macro_f1=b_f1 - metrics(y, P)["macro_f1"],
                         drop_hr_auc=b_auc - rust_auc(y, P)))
    return rows


def unit_file(learner, r, k, sub="hr_units"):
    return os.path.join(OUT, sub, learner, f"r{r}_f{k}.pkl")


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(rows, f)
    os.replace(path + ".tmp", path)


def done(learner, r, k):
    if learner == "flaml":
        return os.path.exists(unit_file("flaml", r, k)) and os.path.exists(unit_file("flaml", r, k, "hr_units_veg"))
    return os.path.exists(unit_file(learner, r, k))


def run_task(task):
    learner, r, k = task
    t0 = time.time()
    try:
        if done(learner, r, k):
            return task, "skip", 0.0
        if learner == "flaml":
            if not all(os.path.exists(model_path("E4", v, r, k)) for v in ("hsc_eng", "ms_eng")):
                return task, "missing E4 fold model (run 05_run_units.py --exp E4)", 0.0
            rows, rows_veg = unit_flaml(r, k, vegetated_ids())
            write(unit_file("flaml", r, k), rows)
            write(unit_file("flaml", r, k, "hr_units_veg"), rows_veg)
        else:
            write(unit_file(learner, r, k), unit_tabpfn(r, k))
        return task, "done", time.time() - t0
    except Exception:
        return task, "error\n" + traceback.format_exc(), time.time() - t0


def summarise(df):
    """Mean drop per window over folds, after averaging the three permutations within each fold."""
    g = df.groupby(["model", "window", "repeat", "fold"])[["drop_macro_f1", "drop_hr_auc"]].mean().reset_index()
    s = g.groupby(["model", "window"]).agg(drop_macro_f1=("drop_macro_f1", "mean"), drop_hr_auc=("drop_hr_auc", "mean"),
                                           drop_hr_auc_sd=("drop_hr_auc", "std"), n=("drop_hr_auc", "size"))
    s.reset_index().to_csv(os.path.join(OUT, "hr_window_summary.csv"), index=False)
    return s


def collect():
    parts = [pd.DataFrame(pickle.load(open(f, "rb")))
             for f in sorted(glob.glob(os.path.join(OUT, "hr_units", "*", "r*_f*.pkl")))]
    if not parts:
        print("no permutation units")
        return
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["model", "repeat", "fold", "window", "perm"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "hr_window_folds.csv"), index=False)
    print(summarise(df).round(4).to_string())

    fs = sorted(glob.glob(os.path.join(OUT, "hr_units_veg", "flaml", "r*_f*.pkl")))
    if not fs:
        return
    dv = pd.concat([pd.DataFrame(pickle.load(open(f, "rb"))) for f in fs], ignore_index=True)
    dv = dv.sort_values(["model", "repeat", "fold", "window", "perm"]).reset_index(drop=True)
    dv.to_csv(os.path.join(OUT, "hr_window_folds_veg.csv"), index=False)
    # consistency check, the all-patch drops beside the vegetated ones must equal those of hr_window_folds.csv
    ref = pd.read_csv(os.path.join(OUT, "hr_window_folds.csv"))
    ref = ref[ref.model.isin(["E4:hsc_eng", "E4:ms_eng"])]
    ref["window"] = ref.window.astype(str)
    m = dv.merge(ref[["model", "window", "repeat", "fold", "perm", "drop_hr_auc"]],
                 on=["model", "window", "repeat", "fold", "perm"], suffixes=("", "_ref"))
    dev = float(np.abs(m.drop_hr_auc - m.drop_hr_auc_ref).max()) if len(m) else np.nan
    print(f"{len(dv)} vegetated rows, {len(m)} matched, largest deviation of the all-patch drops {dev:.2e}")
    g = dv.groupby(["model", "window", "repeat", "fold"])[["drop_hr_auc", "drop_hr_auc_veg"]].mean().reset_index()
    s = g.groupby(["model", "window"]).agg(drop_hr_auc=("drop_hr_auc", "mean"),
                                           drop_hr_auc_veg=("drop_hr_auc_veg", "mean"),
                                           drop_hr_auc_veg_sd=("drop_hr_auc_veg", "std"),
                                           n=("drop_hr_auc_veg", "size")).reset_index()
    s["check_max_abs_dev"] = dev
    s.to_csv(os.path.join(OUT, "hr_window_summary_veg.csv"), index=False)
    print(s.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--learners", nargs="+", default=LEARNERS, choices=LEARNERS)
    ap.add_argument("--repeats", default=None, help="outer repeats, such as 0-9 or 0,1 (default all ten)")
    ap.add_argument("--folds", default=None, help="outer folds, such as 0-4 or 0 (default all five)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes, one CPU thread each")
    ap.add_argument("--collect", action="store_true", help="only rewrite the tables from the units present")
    a = ap.parse_args()
    if not a.collect:
        reps = parse_repeats(a.repeats) if a.repeats else REPEATS
        folds = parse_repeats(a.folds) if a.folds else range(N_FOLDS)
        todo = [(lrn, r, k) for lrn in a.learners for r in reps for k in folds if not done(lrn, r, k)]
        print(f"pending units: {len(todo)}", flush=True)
        single_thread()
        bad = []
        pool = mp.Pool(a.workers, initializer=single_thread) if a.workers > 1 else None
        for task, st, dt in (pool.imap_unordered(run_task, todo) if pool else map(run_task, todo)):
            print(f"{task[0]} r{task[1]} f{task[2]} {st.splitlines()[0]} {dt:.0f}s", flush=True)
            if st not in ("done", "skip"):
                bad.append((task, st))
        if pool:
            pool.close()
            pool.join()
        for task, st in bad:
            print(task, st, file=sys.stderr)
        if bad:
            sys.exit(1)
    missing = [(lrn, r, k) for lrn in LEARNERS for r in REPEATS for k in range(N_FOLDS) if not done(lrn, r, k)]
    if missing and not a.collect:
        print(f"{len(missing)} units missing, tables not rewritten (use --collect to force)")
        return
    collect()


if __name__ == "__main__":
    main()
