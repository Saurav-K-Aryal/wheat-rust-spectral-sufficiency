"""Record the leaderboard scores of the paper's submissions and identify the leaderboard split from them.

The scores come from the Kaggle listing of submissions,
    kaggle competitions submissions -c beyond-visible-spectrum-ai-for-agriculture-2026 -v
which covers every submission on an account. Only the entries whose message matches one in
results/leaderboard/submission_manifest.csv are kept, so no other submission enters the repository.

Kaggle shows each score truncated to five decimals. If every public score is a whole number of correct patches out of
the same n_public, every private score likewise out of n_private, and n_public + n_private equals the number of
holdout patches, then the leaderboard metric is accuracy and the split sizes are identified. The script tries every
pair of sizes that sums to the holdout size and requires exactly one pair to fit every score; macro-F1 would almost
never produce such values.

Reads   a listing, by default results/leaderboard/listing.csv (the paper's entries of the study's listing)
Writes  results/leaderboard/listing.csv          the matched rows, when a different listing is given
        results/leaderboard/scores.csv           public and private score per submission
        results/analysis/leaderboard_counts.csv  the implied numbers of correct patches per split
        results/leaderboard/meta.json            metric and split sizes, only when the file does not exist yet; an
                                                 existing file is checked against the scores and left unchanged
usage: python scripts/analysis/07_leaderboard_import.py [PATH_TO_LISTING]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import json
import os
from io import StringIO

import numpy as np
import pandas as pd

from scripts.analysis._shared import ANALYSIS, LEADERBOARD, RESULTS, analysis_file

TOL = 1.0001e-5  # one unit in the fifth decimal, the resolution of the listing


def read_listing(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if "fileName" in l and "description" in l)  # skip CLI notices
    k = pd.read_csv(StringIO("\n".join(lines[start:])), dtype={"ref": str})
    k.columns = [c.strip() for c in k.columns]
    k["description"] = k["description"].astype(str).str.strip()
    return k


def fits(scores, n):
    """True when every score equals k/n for an integer k, up to the listing's resolution."""
    s = np.asarray(scores, dtype=float)
    return bool(np.all(np.abs(np.round(s * n) / n - s) <= TOL))


def main(argv):
    shipped = os.path.join(LEADERBOARD, "listing.csv")
    src = argv[0] if argv else shipped
    man = pd.read_csv(os.path.join(LEADERBOARD, "submission_manifest.csv"))
    n_holdout = int(json.load(open(os.path.join(RESULTS, "audit", "audit_summary.json")))["n_holdout"])
    k = read_listing(src)
    keep, n_entries = [], 0
    for _, m in man.iterrows():
        hit = k[k.description == m.message]
        n_entries += len(hit)
        if len(hit) == 0:
            print(f"{m.sid}: no entry with the message {m.message!r}")
            continue
        if len(hit) > 1:
            print(f"warning: {len(hit)} entries share the message of {m.sid}; keeping the earliest")
            hit = hit.sort_values("date").head(1)
        keep.append(hit)
    ours = pd.concat(keep).sort_values("date") if keep else k.iloc[:0]
    print(f"kept {len(ours)} of {len(k)} listed submissions; {len(k) - len(ours)} other entries ignored")
    if os.path.abspath(src) != os.path.abspath(shipped):
        ours.to_csv(shipped, index=False)
    bad = ours[~ours.status.astype(str).str.contains("COMPLETE")]
    if len(bad):
        print("warning: not complete:", ", ".join(bad.description))
    ours = ours[ours.status.astype(str).str.contains("COMPLETE")].copy()

    by_message = ours.set_index("description")
    pd.DataFrame([dict(sid=m.sid, file=m.file, message=m.message,
                       public_score=by_message.publicScore.get(m.message, np.nan),
                       private_score=by_message.privateScore.get(m.message, np.nan)) for _, m in man.iterrows()]
                 ).to_csv(os.path.join(LEADERBOARD, "scores.csv"), index=False)

    pub = pd.to_numeric(ours.publicScore, errors="coerce").dropna().values
    pri = pd.to_numeric(ours.privateScore, errors="coerce").dropna().values
    pairs = [(n, n_holdout - n) for n in range(1, n_holdout) if fits(pub, n) and fits(pri, n_holdout - n)]
    print(f"split sizes (public, private) that fit every score and sum to {n_holdout}: {pairs}")
    sid_of = dict(zip(man.message, man.sid))
    rows = []
    for _, r in ours.iterrows():
        row = dict(sid=sid_of[r.description], message=r.description, date=r.date,
                   public_score=float(r.publicScore), private_score=float(r.privateScore))
        if len(pairs) == 1:
            n_pub, n_pri = pairs[0]
            row.update(n_public=n_pub, public_correct=int(round(r.publicScore * n_pub)),
                       n_private=n_pri, private_correct=int(round(r.privateScore * n_pri)))
        rows.append(row)
    os.makedirs(ANALYSIS, exist_ok=True)
    pd.DataFrame(rows).sort_values("sid").to_csv(analysis_file("leaderboard_counts.csv"), index=False)
    if len(pairs) != 1:
        print("the split is not identified by the scores; meta.json is not written")
        return
    n_pub, n_pri = pairs[0]
    p = os.path.join(LEADERBOARD, "meta.json")
    if os.path.exists(p):
        old = json.load(open(p))
        agrees = (old.get("metric") == "accuracy" and old.get("n_public") == n_pub and old.get("n_private") == n_pri
                  and old.get("private_fractions") == [n_pri / n_holdout])
        print(f"{os.path.relpath(p)} {'agrees with' if agrees else 'DIFFERS from'} the scores "
              f"(accuracy on {n_pub} public and {n_pri} private patches); left unchanged")
        return
    meta = dict(metric="accuracy", private_fractions=[n_pri / n_holdout], n_public=n_pub, n_private=n_pri,
                n_holdout=n_holdout, n_submissions=len(ours), n_study_entries=n_entries,
                basis=(f"Inferred from the returned scores: every public score is a whole number of correct patches "
                       f"out of {n_pub} and every private score out of {n_pri}, and {n_pub} + {n_pri} = {n_holdout} "
                       "holdout samples, the only pair of sizes that fits all scores. This confirms accuracy as the "
                       "leaderboard metric."))
    json.dump(meta, open(p, "w"), indent=2)
    print(f"wrote {os.path.relpath(p)}: accuracy on {n_pub} public and {n_pri} private patches")


if __name__ == "__main__":
    main(sys.argv[1:])
