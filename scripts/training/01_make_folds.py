"""Outer folds, 10 repeats of stratified grouped 5-fold cross-validation over the 600 training patches.

The folds are stratified by class and grouped so that overlapping patches always fall in the same fold (see
wheatrust.common.overlap_groups). They are saved once and shared by every experiment, which makes every comparison
between configurations paired.

Reads   the array cache and results/audit/duplicates.csv (scripts/eda/01_audit.py)
Writes  results/folds/folds.csv (sample_id, repeat, fold, group); an existing file is kept unless --force is given
usage: python scripts/training/01_make_folds.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse  # noqa: E402
import os  # noqa: E402

from wheatrust.common import FOLD_FILE, make_folds  # noqa: E402
from wheatrust.io_data import Arrays  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="overwrite an existing folds.csv")
    a = ap.parse_args()
    if os.path.exists(FOLD_FILE) and not a.force:
        print("folds already exist, not overwriting:", FOLD_FILE)
        return
    A = Arrays()
    ids = A.sample_id[A.tr]
    y = A.y_train()
    df = make_folds(ids, y)
    os.makedirs(os.path.dirname(FOLD_FILE), exist_ok=True)
    df.to_csv(FOLD_FILE, index=False)
    df["y"] = df.sample_id.map(dict(zip(ids, y)))
    print("validation samples per class and fold (0 Health, 1 Rust, 2 Other)")
    print(df.groupby(["repeat", "fold"]).y.value_counts().unstack().describe().round(1).to_string())
    print("groups with more than one patch:", int((df[df.repeat == 0].group.value_counts() > 1).sum()))
    print("wrote", FOLD_FILE)


if __name__ == "__main__":
    main()
