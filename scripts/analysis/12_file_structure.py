"""How the competition files are organised, for the question whether acquisition batches or plots could stand behind
the class signal. Only the audit tables are read; no image is opened.

1. The two HS file formats. Some HS files carry a 126th band of constant fill, and the two formats behave like
   separate processing batches (adversarial validation separates them, results/E12/batch_adversarial.csv). Training
   files are named <class>_hyper_<n> with n from 1 to 200 in every class. The script records which numbers carry
   the 126th band in each class, whether the three classes agree number by number, and how the numbers cluster.
   A format that followed the class, or concentrated in one class, would confound the format batch with the label.
2. The overlapping tiles. For each pair of training patches whose tiles overlap by half a tile
   (results/audit/overlap_offsets.csv), the difference of their file numbers. If the numbering followed the order in
   which tiles were cut from the mosaics, overlapping tiles would have nearby numbers, and blocks of numbers could
   stand in for spatial blocks in a grouped cross-validation.

Writes results/analysis/file_structure.json and results/analysis/overlap_pairs.csv.
usage: python scripts/analysis/12_file_structure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import json
import os

import pandas as pd

from scripts.analysis._shared import RESULTS, analysis_file
from wheatrust.io_data import CLASSES


def number(sample_id):
    return int(sample_id.rsplit("_", 1)[1])


def longest_run(values):
    """Length of the longest run of consecutive integers in a set."""
    v = sorted(values)
    best = run = 1 if v else 0
    for a, b in zip(v, v[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


def formats():
    inv = pd.read_csv(os.path.join(RESULTS, "audit", "file_inventory.csv"))
    tr = inv[inv.split == "train"].copy()
    tr["number"] = tr.sample_id.map(number)
    wide = tr.pivot(index="number", columns="label", values="hs_bands_raw")[CLASSES]
    agree = wide.nunique(axis=1) == 1
    sets = {c: set(wide.index[wide[c] == 126]) for c in CLASSES}
    common = set.intersection(*sets.values())
    numbers = sorted(common)
    ho = inv[inv.split == "val"]
    return dict(
        numbers_per_class=int(len(wide)),
        numbers_same_format_all_classes=int(agree.sum()),
        n126_per_class={c: len(s) for c, s in sets.items()},
        n126_common_to_all_classes=len(common),
        n126_holdout=int((ho.hs_bands_raw == 126).sum()),
        n_holdout=int(len(ho)),
        numbers_126=numbers,
        longest_run_126=longest_run(numbers),
        numbers_126_below_101=int(sum(n <= 100 for n in numbers)),
    )


def overlaps():
    off = pd.read_csv(os.path.join(RESULTS, "audit", "overlap_offsets.csv"))
    tt = off[(off.a_split == "train") & (off.b_split == "train")].copy()
    tt["label_a"] = tt.a.str.split("_").str[0]
    tt["label_b"] = tt.b.str.split("_").str[0]
    tt["number_a"] = tt.a.map(number)
    tt["number_b"] = tt.b.map(number)
    tt["number_gap"] = (tt.number_a - tt.number_b).abs()
    tt["same_label"] = tt.label_a == tt.label_b
    return tt[["a", "b", "label_a", "label_b", "same_label", "number_a", "number_b", "number_gap", "dy", "dx"]]


def main():
    s = formats()
    ov = overlaps()
    same = ov[ov.same_label]
    s.update(n_overlap_train_pairs=int(len(ov)), n_overlap_same_label=int(len(same)),
             overlap_number_gaps_same_label=[int(g) for g in sorted(same.number_gap)],
             overlap_min_number_gap_same_label=int(same.number_gap.min()) if len(same) else None,
             overlap_any_adjacent=bool((same.number_gap <= 1).any()))
    ov.to_csv(analysis_file("overlap_pairs.csv"), index=False)
    with open(analysis_file("file_structure.json"), "w") as f:
        json.dump(s, f, indent=2)
    print(f"HS format identical across the three classes at {s['numbers_same_format_all_classes']} of "
          f"{s['numbers_per_class']} file numbers; {s['n126_common_to_all_classes']} numbers carry 126 bands in every "
          f"class (longest run of consecutive numbers {s['longest_run_126']})")
    print(f"{s['n_overlap_same_label']} of {s['n_overlap_train_pairs']} overlapping training pairs share a label; "
          f"their file numbers differ by {s['overlap_number_gaps_same_label']}")


if __name__ == "__main__":
    main()
