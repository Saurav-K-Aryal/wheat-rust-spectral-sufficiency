"""Pixel offset between the overlapping patch pairs found by the audit.

scripts/eda/01_audit.py lists the pairs of HS patches that share exact 4 x 4 pixel blocks
(results/audit/duplicates.csv) but not how far one tile is shifted against the other. This script recovers the
shift from the positions of the shared blocks in both patches; the most frequent displacement is the offset.

Reads   the array cache and results/audit/duplicates.csv
Writes  results/audit/overlap_offsets.csv (one row per pair, with the modal displacement and its support)
usage: python scripts/eda/02_overlap_offsets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import hashlib  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wheatrust.io_data import RESULTS  # noqa: E402
from wheatrust.runner import arrays  # noqa: E402


def blocks(a):
    """Positions of every 4 x 4 block without empty pixels, keyed by the hash of its contents."""
    out = {}
    for y in range(a.shape[0] - 3):
        for x in range(a.shape[1] - 3):
            w = a[y:y + 4, x:x + 4]
            if (w == 0).all(-1).any():
                continue
            out.setdefault(hashlib.md5(np.ascontiguousarray(w).tobytes()).digest(), []).append((y, x))
    return out


def main():
    A = arrays()
    pos = {s: i for i, s in enumerate(A.sample_id)}
    d = pd.read_csv(os.path.join(RESULTS, "audit", "duplicates.csv"))
    d = d[d.kind == "shared_4x4_blocks"]
    rows = []
    for _, r in d.iterrows():
        a, b = np.asarray(A.hs[pos[r.a]]), np.asarray(A.hs[pos[r.b]])
        ba, bb = blocks(a), blocks(b)
        disp = Counter()
        for h, pa in ba.items():
            for (y1, x1) in pa:
                for (y2, x2) in bb.get(h, []):
                    disp[(y1 - y2, x1 - x2)] += 1
        (dy, dx), n = disp.most_common(1)[0]
        rows.append(dict(a=r.a, b=r.b, a_split=r.a_split, b_split=r.b_split, dy=dy, dx=dx,
                         offset_px=max(abs(dy), abs(dx)), diagonal=bool(dy and dx), n_blocks_at_offset=n,
                         n_shared_blocks=int(r.n_shared_blocks), tile_px=a.shape[0]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "audit", "overlap_offsets.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
