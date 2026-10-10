"""Figure 2, patch-mean NDVI by class.

Every non-empty training patch as a jittered point over a box plot of its class (box from the first to the third
quartile, whiskers to 1.5 times the interquartile range). Health patches include sparse canopy that no Rust patch
shows, which is why the paper also scores the disease decision on vegetated patches (patch-mean NDVI of at least
0.5).

Reads results/audit/composition_ndvi.csv; writes outputs/figures/fig_ndvi.pdf.
usage: python scripts/plots/02_fig_ndvi.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots._style import CLASS_STYLE, CLASSES, STD_W, finalize, load_csv, run, warn
from wheatrust.io_data import RESULTS


def fig_ndvi(name, source):
    tag = "fig_ndvi"
    comp = load_csv(tag, os.path.join(RESULTS, "audit", "composition_ndvi.csv"), ["label", "empty", "ndvi_mean"])
    if comp is None:
        return False
    empty = comp["empty"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    comp = comp[~empty & comp["ndvi_mean"].notna()]
    classes = [c for c in CLASSES if (comp["label"] == c).any()]
    for c in CLASSES:
        if c not in classes:
            warn(tag, f"no non-empty {c} patches")
    if not classes:
        return False
    fig, ax = plt.subplots(figsize=(STD_W, 2.3), layout="constrained")
    data = [comp.loc[comp["label"] == c, "ndvi_mean"].to_numpy(float) for c in classes]
    rng = np.random.default_rng(20260507)  # fixed seed so the jitter is reproducible
    for i, (c, v) in enumerate(zip(classes, data)):
        ax.scatter(i + rng.uniform(-0.17, 0.17, v.size), v, s=7, color=CLASS_STYLE[c]["color"], alpha=0.55,
                   lw=0, marker=CLASS_STYLE[c]["marker"], zorder=2)
    ax.boxplot(data, positions=range(len(classes)), widths=0.5, whis=1.5, showfliers=False,
               patch_artist=True, zorder=3,
               boxprops=dict(facecolor="none", edgecolor="black", lw=0.9),
               medianprops=dict(color="black", lw=1.4), whiskerprops=dict(color="black", lw=0.8),
               capprops=dict(color="black", lw=0.8))
    ax.set_xticks(range(len(classes)), [f"{c}\nn = {len(v)}" for c, v in zip(classes, data)])
    ax.set_xlim(-0.6, len(classes) - 0.4)
    ax.set_ylabel("Patch mean NDVI (unitless)")
    ax.xaxis.grid(False)
    ax.tick_params(axis="x", length=0)
    return finalize(fig, name, source)


if __name__ == "__main__":
    sys.exit(run(fig_ndvi, "fig_ndvi.pdf", "scripts/plots/02_fig_ndvi.py"))
