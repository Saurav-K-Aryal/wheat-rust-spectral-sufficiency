"""Cross-batch transfer between the two HS file formats.

489 training files have 125 HS bands and 111 have a 126th fill band, and the two formats behave like separate
processing batches (scripts/eda/03_adversarial_validation.py). Here a model is trained on one batch and tested on
the other, and the same is done on random splits with the same sizes and class counts as the reference, so the
difference measures what the batch shift costs. Inputs are all views, MS, and cleaned HS; learners are the FLAML
baseline search (without threshold tuning) and TabPFN with eight members; five seeds per split
(wheatrust.experiments, CROSSBATCH_*).

Reads   the array and feature caches, and for TabPFN the checkpoint under MODELS
Writes  results/E10/crossbatch/<unit>.pkl (one per learner, view, split, and seed; not part of the repository)
        results/E10/crossbatch.csv
usage: python scripts/training/07_crossbatch.py [--views all ms hsc] [--learners flaml tabpfn] [--workers N]
       python scripts/training/07_crossbatch.py --collect
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wheatrust.common import metrics  # noqa: E402
from wheatrust.experiments import (BASE_EST, BUDGET, CROSSBATCH_INPUTS, CROSSBATCH_LEARNERS,  # noqa: E402
                                   CROSSBATCH_SEEDS, CROSSBATCH_SPLITS)
from wheatrust.io_data import RESULTS  # noqa: E402
from wheatrust.runner import arrays, build_input, single_thread  # noqa: E402

UNITS = os.path.join(RESULTS, "E10", "crossbatch")
COLUMNS = ["name", "seed", "n_train", "n_test", "macro_f1", "accuracy", "recall_Rust", "selected", "view", "learner"]


def unit_path(view, name, s, learner="flaml"):
    """File of one unit. FLAML units of the all-views input carry no prefix, which fixes the row order of the table."""
    stem = f"{name}_s{s}" if view == "all" else f"{view}__{name}_s{s}"
    if learner != "flaml":
        stem = f"{learner}__{view}__{name}_s{s}"
    return os.path.join(UNITS, stem + ".pkl")


def run_unit(view, name, s, learner="flaml"):
    """Train on one file batch (or one side of a random split) and test on the rest."""
    from wheatrust.learners import FlamlLearner, TabPFNLearner

    out = unit_path(view, name, s, learner)
    if os.path.exists(out):
        return "skip"
    A = arrays()
    ids = A.sample_id[A.tr]
    y = A.y_train()
    b = A.hs_nbands_raw[A.tr]
    rng = np.random.default_rng(500 + s)
    if name == "125_to_126":
        tr, te = np.where(b == 125)[0], np.where(b == 126)[0]
    elif name == "126_to_125":
        tr, te = np.where(b == 126)[0], np.where(b == 125)[0]
    else:
        # random split with the class counts of the batch of the same size
        n_tr = 489 if name == "random_489_to_111" else 111
        ref_tr = np.where(b == (125 if n_tr == 489 else 126))[0]
        cnt = np.bincount(y[ref_tr], minlength=3)
        tr = np.concatenate([rng.choice(np.where(y == c)[0], cnt[c], replace=False) for c in range(3)])
        te = np.setdiff1d(np.arange(len(y)), tr)
    seed = 7000 + s
    Xtr, Xte, cols = build_input(CROSSBATCH_INPUTS[view], ids[tr], ids[te], y[tr], seed)
    if learner == "tabpfn":
        L = TabPFNLearner(seed=seed, n_estimators=8).fit(Xtr, y[tr])
    else:
        L = FlamlLearner(estimator_list=BASE_EST, budget=BUDGET, seed=seed, tune_thr=False).fit(Xtr, y[tr])
    P = L.predict_proba(Xte)
    m = metrics(y[te], P)
    res = dict(view=view, name=name, seed=s, learner=learner, n_train=len(tr), n_test=len(te),
               macro_f1=m["macro_f1"], accuracy=m["accuracy"], recall_Rust=m["recall_Rust"],
               selected=L.describe()["selected_estimator"])
    os.makedirs(UNITS, exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(out + ".tmp", out)
    return "done"


def run_task(task):
    t0 = time.time()
    try:
        return task, run_unit(*task), time.time() - t0
    except Exception:
        return task, "error\n" + traceback.format_exc(), time.time() - t0


def all_units(views=tuple(CROSSBATCH_INPUTS), learners=CROSSBATCH_LEARNERS):
    return [(v, n, s, lrn) for lrn in learners for v in views for s in CROSSBATCH_SEEDS for n in CROSSBATCH_SPLITS]


def collect():
    fs = sorted(glob.glob(os.path.join(UNITS, "*.pkl")))
    if not fs:
        print("no cross-batch units")
        return
    d = pd.DataFrame([pickle.load(open(f, "rb")) for f in fs])[COLUMNS]
    d.to_csv(os.path.join(RESULTS, "E10", "crossbatch.csv"), index=False)
    print(d.groupby(["learner", "view", "name"]).macro_f1.mean().round(3).unstack().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--views", nargs="+", default=list(CROSSBATCH_INPUTS), choices=list(CROSSBATCH_INPUTS))
    ap.add_argument("--learners", nargs="+", default=CROSSBATCH_LEARNERS, choices=CROSSBATCH_LEARNERS)
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes, one CPU thread each")
    ap.add_argument("--collect", action="store_true", help="only rewrite the table from the units present")
    a = ap.parse_args()
    if not a.collect:
        todo = [t for t in all_units(a.views, a.learners) if not os.path.exists(unit_path(*t))]
        print(f"pending units: {len(todo)}", flush=True)
        single_thread()
        bad = []
        pool = mp.Pool(a.workers, initializer=single_thread) if a.workers > 1 else None
        for task, st, dt in (pool.imap_unordered(run_task, todo) if pool else map(run_task, todo)):
            print(f"{task[3]} {task[0]} {task[1]} s{task[2]} {st.splitlines()[0]} {dt:.0f}s", flush=True)
            if st not in ("done", "skip"):
                bad.append((task, st))
        if pool:
            pool.close()
            pool.join()
        for task, st in bad:
            print(task, st, file=sys.stderr)
        if bad:
            sys.exit(1)
    missing = [t for t in all_units() if not os.path.exists(unit_path(*t))]
    if missing and not a.collect:
        print(f"{len(missing)} units missing, table not rewritten (use --collect to force)")
        return
    collect()


if __name__ == "__main__":
    main()
