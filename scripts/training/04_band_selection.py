"""Band selection for the band budgets, forward selection of up to 32 cleaned HS bands in every outer training fold.

Inside each outer training fold, greedy forward selection over the patch-mean reflectance of the 101 cleaned bands
adds the band that most improves the inner 3-fold macro-F1 of shrinkage LDA (wheatrust.bands.forward_select). The
budgets of E9 and E9t are prefixes of the selection order, so one search per fold serves every budget and both
learners. The E9 and E9t units compute a missing selection themselves; running this step first avoids parallel
workers repeating the same search.

Reads   the feature cache (patch mean spectra) and results/folds/folds.csv
Writes  results/E9/selections/sel_seed<1000 * repeat + fold>.json (existing files are kept)
usage: python scripts/training/04_band_selection.py [--repeats 0-9] [--folds 0-4] [--workers N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

from wheatrust.bands import SELECTION_DIR, selection  # noqa: E402
from wheatrust.common import N_FOLDS, fold_indices, load_folds  # noqa: E402
from wheatrust.experiments import REPEATS  # noqa: E402
from wheatrust.runner import arrays, parse_repeats, single_thread  # noqa: E402


def select_fold(rk):
    r, k = rk
    t0 = time.time()
    A = arrays()
    ids = A.sample_id[A.tr]
    y = A.y_train()
    tr, _ = fold_indices(load_folds(), ids, r, k)
    sel = selection(ids[tr], y[tr], 1000 * r + k, cache_dir=SELECTION_DIR)
    return r, k, sel["wavelengths"][:5], time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", default=None, help="outer repeats, such as 0-9 or 0,1 (default all ten)")
    ap.add_argument("--folds", default=None, help="outer folds, such as 0-4 or 0 (default all five)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes, one CPU thread each")
    a = ap.parse_args()
    reps = parse_repeats(a.repeats) if a.repeats else REPEATS
    folds = parse_repeats(a.folds) if a.folds else range(N_FOLDS)
    todo = [(r, k) for r in reps for k in folds
            if not os.path.exists(os.path.join(SELECTION_DIR, f"sel_seed{1000 * r + k}.json"))]
    print(f"selections to compute: {len(todo)}", flush=True)
    single_thread()
    if a.workers > 1:
        with mp.Pool(a.workers, initializer=single_thread) as pool:
            for r, k, wl, dt in pool.imap_unordered(select_fold, todo):
                print(f"repeat {r} fold {k}: first five bands {wl} ({dt:.0f}s)", flush=True)
    else:
        for rk in todo:
            r, k, wl, dt = select_fold(rk)
            print(f"repeat {r} fold {k}: first five bands {wl} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
