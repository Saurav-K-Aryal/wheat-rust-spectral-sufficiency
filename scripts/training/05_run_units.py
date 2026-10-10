"""Experiment units with FLAML and TabPFN, and their tidy result tables.

Runs every pending unit (experiment, variant, repeat, fold) of the named experiments of wheatrust.experiments with a
pool of worker processes, one CPU thread each. Units already on disk are skipped and each unit is written
atomically, so the script can be stopped and restarted at any time. When an experiment has all its units, its
tables are rewritten from the units; --collect rewrites them from whatever units exist.

The platform reruns are experiments of their own (REPRO and REPROX) that reuse the specs of E1, E2, E4, E4t, E9,
and E9t, so running them on another machine writes the same units to their own results folders.

Reads   the feature cache, results/folds/folds.csv, results/E9/selections (band budgets), and for TabPFN the
        checkpoint under MODELS (see wheatrust.learners)
Writes  results/<exp>/units/<variant>/r<repeat>_f<fold>.pkl   one pickle per unit (not part of the repository)
        results/<exp>/models/<variant>/r<repeat>_f<fold>.pkl  fitted fold models of E4 ms_eng and hsc_eng
        results/<exp>/fold_metrics.csv and oof.csv.gz        the tidy tables
usage: python scripts/training/05_run_units.py --exp E1 E4 [--only REGEX] [--skip REGEX] [--repeats 0-9]
                                               [--folds 0-4] [--workers N]
       python scripts/training/05_run_units.py --exp REPRO REPROX --workers 2      (platform reruns)
       python scripts/training/05_run_units.py --exp E4 --collect                  (tables only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

from wheatrust import experiments, runner  # noqa: E402
from wheatrust.common import N_FOLDS, load_folds  # noqa: E402

_STATE = {}


def init_worker():
    runner.single_thread()
    _STATE["folds"] = load_folds()


def run_one(task):
    """Run one unit and report (task, status, seconds, error text)."""
    exp, v, r, k = task
    if "folds" not in _STATE:
        init_worker()
    if exp not in _STATE:
        _STATE[exp] = experiments.get(exp)
    t0 = time.time()
    try:
        st = runner.run_unit(exp, v, _STATE[exp][v], r, k, _STATE["folds"])
        return task, st, time.time() - t0, None
    except Exception:
        return task, "error", time.time() - t0, traceback.format_exc()


def select(exp, only, skip):
    V = experiments.get(exp)
    if only:
        V = {k: v for k, v in V.items() if re.search(only, k)}
    if skip:
        V = {k: v for k, v in V.items() if not re.search(skip, k)}
    return V


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", nargs="+", required=True, choices=sorted(experiments.REGISTRY))
    ap.add_argument("--only", default=None, help="regular expression on variant names to run")
    ap.add_argument("--skip", default=None, help="regular expression on variant names to leave out")
    ap.add_argument("--repeats", default=None, help="outer repeats, such as 0-9 or 0,1 (default all ten)")
    ap.add_argument("--folds", default=None, help="outer folds, such as 0-4 or 0 (default all five)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes, one CPU thread each")
    ap.add_argument("--collect", action="store_true", help="only rewrite the tables from the units present")
    a = ap.parse_args()

    if not a.collect:
        reps = runner.parse_repeats(a.repeats) if a.repeats else experiments.REPEATS
        folds = runner.parse_repeats(a.folds) if a.folds else range(N_FOLDS)
        todo = [(exp, v, r, k) for exp in a.exp
                for v, r, k in runner.pending(exp, select(exp, a.only, a.skip), reps, folds)]
        todo.sort(key=lambda t: (t[2], t[3], t[0], t[1]))
        print(f"pending units: {len(todo)}", flush=True)
        failed = []
        if a.workers > 1:
            with mp.Pool(a.workers, initializer=init_worker) as pool:
                results = pool.imap_unordered(run_one, todo)
                for task, st, dt, err in results:
                    print(f"{task[0]} {task[1]} r{task[2]} f{task[3]} {st} {dt:.1f}s", flush=True)
                    if err:
                        failed.append(task)
                        print(err, file=sys.stderr, flush=True)
        else:
            for t in todo:
                task, st, dt, err = run_one(t)
                print(f"{task[0]} {task[1]} r{task[2]} f{task[3]} {st} {dt:.1f}s", flush=True)
                if err:
                    failed.append(task)
                    print(err, file=sys.stderr, flush=True)
        if failed:
            print(f"{len(failed)} units failed; rerun to retry them", flush=True)
            sys.exit(1)

    for exp in a.exp:
        missing = runner.pending(exp, experiments.get(exp), experiments.REPEATS)
        if missing and not a.collect:
            print(f"{exp}: {len(missing)} units missing, tables not rewritten (use --collect to force)")
            continue
        fm, _ = runner.collect(exp)
        if fm is None:
            print(f"{exp}: no units")
        else:
            print(f"{exp}: {len(fm)} units collected, mean macro-F1 by variant")
            print(fm.groupby("variant").macro_f1.mean().round(3).to_string())


if __name__ == "__main__":
    main()
