"""Holdout refits and the submission files of the paper.

Each submission refits its procedure on all 600 training patches, with the same input and learner as in
cross-validation, and predicts the 300 holdout patches with the argmax rule. Holdout inputs are used for nothing
but these predictions.

  S01  FLAML baseline, all views          S05  TabPFN, all views
  S02  FLAML, RGB view                    S07  FLAML, five bands selected on the full training set
  S03  FLAML, MS view                     S09  FLAML, RGB features computed from the HS bands
  S04  FLAML, cleaned HS view

Every file is validated against the competition format (header Id,Category, one row per holdout sample, Ids equal
to the holdout HS file names, categories Health, Rust, or Other).

Reads   the array and feature caches, and for S05 the TabPFN checkpoint under MODELS
Writes  results/holdout/<sid>.pkl                     holdout probabilities (not part of the repository)
        results/holdout/selections/sel_seed20260507.json   the five bands of S07
        submissions/<sid>_<name>.csv
        results/leaderboard/submission_manifest.csv   once all seven submissions exist
usage: python scripts/training/08_submissions.py [S01 S03 ...]     (default all seven)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wheatrust.experiments import BASE_EST, BASELINE_INPUT, BUDGET, PCA_HSC, feats  # noqa: E402
from wheatrust.io_data import CLASSES, DATA, RESULTS, ROOT  # noqa: E402
from wheatrust.runner import arrays, build_input, make_learner, platform_label, single_thread  # noqa: E402

SEED = 20260507
SUBS = {
    "S01": ("flaml", "S01_flaml_all.csv", BASELINE_INPUT, "S01 FLAML baseline all views"),
    "S02": ("flaml", "S02_flaml_rgb.csv", feats(["rgb"]), "S02 FLAML baseline RGB view"),
    "S03": ("flaml", "S03_flaml_ms.csv", feats(["ms"]), "S03 FLAML baseline MS view"),
    "S04": ("flaml", "S04_flaml_hs_clean.csv", feats(["hsc"], PCA_HSC), "S04 FLAML baseline cleaned HS view"),
    "S05": ("tabpfn", "S05_tabpfn_all.csv", BASELINE_INPUT, "S05 TabPFN all views"),
    "S07": ("flaml", "S07_flaml_bandsel5.csv", {"kind": "bandsel", "k": 5}, "S07 FLAML optimally placed five bands"),
    "S09": ("flaml", "S09_flaml_rgb_from_hs.csv", feats(["rgbhs"]),
            "S09 FLAML single-source RGB features from HS bands"),
}
SUB_DIR = os.path.join(ROOT, "submissions")
HO_DIR = os.path.join(RESULTS, "holdout")
MANIFEST = os.path.join(RESULTS, "leaderboard", "submission_manifest.csv")


def refit(sid):
    """Refit one submission on all training patches and store its holdout probabilities."""
    kind, fname, inp, msg = SUBS[sid]
    out = os.path.join(HO_DIR, f"{sid}.pkl")
    if os.path.exists(out):
        return "skip"
    A = arrays()
    ids_tr, ids_va = A.sample_id[A.tr], A.sample_id[A.va]
    y = A.y_train()
    t0 = time.time()
    if inp.get("kind") == "bandsel":
        # bands selected once on the full training set, with the selection cached beside the holdout results
        from wheatrust.bands import band_cube, generic_band_features, selection

        sel = selection(ids_tr, y, SEED, cache_dir=os.path.join(HO_DIR, "selections"))
        spec = sel["bands_hs_index"][:5]
        Xtr, cols = generic_band_features(band_cube(ids_tr, spec))
        Xva, _ = generic_band_features(band_cube(ids_va, spec))
        extra = {"selected_wavelengths": sel["wavelengths"][:5]}
    else:
        Xtr, Xva, cols = build_input(inp, ids_tr, ids_va, y, SEED)
        extra = {}
    spec = {"kind": "flaml", "estimator_list": BASE_EST, "budget": BUDGET} if kind == "flaml" else {"kind": "tabpfn"}
    L = make_learner(spec, SEED, cols).fit(Xtr, y)
    P = np.asarray(L.predict_proba(Xva))
    res = dict(sid=sid, message=msg, file=fname, holdout_ids=ids_va, proba=P, describe=L.describe(),
               n_features=len(cols), fit_s=time.time() - t0, platform=platform_label(), **extra)
    os.makedirs(HO_DIR, exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(out + ".tmp", out)
    return "done"


def write_submission(ids_val, proba, fname):
    os.makedirs(SUB_DIR, exist_ok=True)
    pred = np.asarray(proba).argmax(1)
    df = pd.DataFrame({"Id": [f"{s}.tif" for s in ids_val], "Category": [CLASSES[i] for i in pred]})
    df.to_csv(os.path.join(SUB_DIR, fname), index=False)
    return df


def validate(path, holdout_ids):
    """Problems of a submission file against the competition format (an empty list if there are none)."""
    df = pd.read_csv(path)
    errs = []
    if list(df.columns) != ["Id", "Category"]:
        errs.append(f"header {list(df.columns)}")
    if len(df) != len(holdout_ids):
        errs.append(f"{len(df)} rows, expected {len(holdout_ids)}")
    if df.Id.duplicated().any():
        errs.append("duplicate Ids")
    exp_ids = {f"{s}.tif" for s in holdout_ids}
    if set(df.Id) != exp_ids:
        errs.append(f"Id mismatch: {len(set(df.Id) - exp_ids)} unexpected, {len(exp_ids - set(df.Id))} missing")
    bad = set(df.Category) - set(CLASSES)
    if bad:
        errs.append(f"invalid categories {bad}")
    files = set(os.listdir(os.path.join(DATA, "val", "HS")))
    if not set(df.Id) <= files:
        errs.append("Ids not matching holdout HS filenames")
    return errs


def materialise():
    """Write and validate the submission file of every refit present, and the manifest once all exist."""
    A = arrays()
    ids_va = A.sample_id[A.va]
    rows = []
    for sid in SUBS:
        p = os.path.join(HO_DIR, f"{sid}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            r = pickle.load(f)
        assert list(r["holdout_ids"]) == list(ids_va)
        write_submission(r["holdout_ids"], r["proba"], r["file"])
        errs = validate(os.path.join(SUB_DIR, r["file"]), ids_va)
        pred = np.asarray(r["proba"]).argmax(1)
        rows.append(dict(sid=r["sid"], file=r["file"], message=r["message"], valid=not errs, errors="; ".join(errs),
                         n_Health=int((pred == 0).sum()), n_Rust=int((pred == 1).sum()), n_Other=int((pred == 2).sum()),
                         selected=r["describe"].get("selected_estimator", "")))
    df = pd.DataFrame(rows)
    print(df.to_string())
    if len(df) == len(SUBS):
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        df.to_csv(MANIFEST, index=False)
        print("wrote", MANIFEST)
    else:
        print(f"{len(SUBS) - len(df)} submissions missing, manifest not rewritten")
    if len(df) and not df.valid.all():
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sids", nargs="*", default=list(SUBS), help="submissions to refit (default all seven)")
    a = ap.parse_args()
    unknown = set(a.sids) - set(SUBS)
    if unknown:
        ap.error(f"unknown submissions {sorted(unknown)}; choose from {list(SUBS)}")
    single_thread()
    for s in a.sids:
        print(s, refit(s), flush=True)
    materialise()


if __name__ == "__main__":
    main()
