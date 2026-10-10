"""Figure 3, macro-F1 of every view and representation with FLAML.

One row per view, with raw band statistics, engineered features, and PCA scores side by side, each with its 95
percent interval over outer folds; the dashed line marks the all-views engineered baseline. The single-source rows
(RGB and MS from HS bands) compute the same features directly from the matching HS bands, and the stretch control
applies the per-image 2 to 98 percent stretch and 8-bit quantisation of the RGB product to those bands.

Reads results/analysis/e4_views.csv (scripts/analysis/01_summaries.py); writes outputs/figures/fig_views.pdf.
usage: python scripts/plots/03_fig_views.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.analysis._shared import analysis_file
from scripts.plots._style import OI, STD_W, ci_halfwidth, finalize, load_csv, points, run, warn

VIEW_ROWS = [
    ("rgb", "RGB PNG"),
    ("rgbhs", "RGB features from HS bands"),
    ("rgbhsst", "RGB features from stretched HS bands"),
    ("ms", "MS file"),
    ("mshs", "MS features from HS bands"),
    ("hsc", "Cleaned HS (101 bands)"),
    ("hsf", "Full HS (125 bands)"),
    ("all", "All views"),
]
REP_STYLE = {  # representation -> (label, colour, marker, vertical dodge)
    "raw": ("Raw band statistics", OI["orange"], "s", 0.22),
    "eng": ("Engineered features", OI["blue"], "o", 0.0),
    "pca": ("PCA scores", OI["green"], "^", -0.22),
}


def fig_views(name, source):
    tag = "fig_views"
    df = load_csv(tag, analysis_file("e4_views.csv"), ["variant", "macro_f1_mean", "macro_f1_sd", "n_units"])
    if df is None:
        return False
    recs = []
    for _, r in df.iterrows():
        v = str(r["variant"])
        if v == "E1:flaml_all":
            view, rep = "all", "eng"
        else:
            m = re.fullmatch(r"E4:([a-z]+)_(raw|eng|pca)", v)
            if not m or m.group(1) not in dict(VIEW_ROWS):
                warn(tag, f"unrecognised variant {v} not plotted")
                continue
            view, rep = m.group(1), m.group(2)
        mean = float(r["macro_f1_mean"])
        if np.isfinite(mean):
            recs.append(dict(view=view, rep=rep, mean=mean, hw=ci_halfwidth(r["macro_f1_sd"], r["n_units"])))
    if not recs:
        return False
    recs = pd.DataFrame(recs)
    rows = [(k, lab) for k, lab in VIEW_ROWS if (recs["view"] == k).any()]
    for k, lab in VIEW_ROWS:
        if (k, lab) not in rows:
            warn(tag, f"no variants for view row {lab}")
    fig, ax = plt.subplots(figsize=(STD_W, 0.36 * len(rows) + 0.75), layout="constrained")
    ypos = {k: len(rows) - 1 - i for i, (k, _) in enumerate(rows)}
    handles = []
    for rep, (lab, col, mk, dy) in REP_STYLE.items():
        sub = recs[recs["rep"] == rep]
        if sub.empty:
            continue
        points(ax, sub["mean"], [ypos[v] + dy for v in sub["view"]], sub["hw"], horizontal=True, color=col,
               marker=mk, ms=4.2)
        handles.append(Line2D([], [], ls="none", marker=mk, color=col, ms=4.2, label=lab))
    base = recs[recs["view"].eq("all") & recs["rep"].eq("eng")]
    if len(base):
        ax.axvline(base["mean"].iloc[0], color=OI["black"], ls="--", lw=0.8, zorder=1)
        handles.append(Line2D([], [], color=OI["black"], ls="--", lw=0.8, label="All-views engineered baseline"))
    else:
        warn(tag, "E1:flaml_all missing; no baseline reference line")
    ax.set_yticks([ypos[k] for k, _ in rows], [lab for _, lab in rows])
    for k in ypos.values():  # faint separators between view rows
        ax.axhline(k - 0.5, color="#EEEEEE", lw=0.5, zorder=0)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.yaxis.grid(False)
    ax.tick_params(axis="y", length=0)
    lo = np.nanmin(recs["mean"] - recs["hw"].fillna(0))
    hi = np.nanmax(recs["mean"] + recs["hw"].fillna(0))
    ax.set_xlim(lo - 0.015, hi + 0.015)
    ax.set_xlabel("Macro-F1 (mean and 95% interval over outer folds)")
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    return finalize(fig, name, source)


if __name__ == "__main__":
    sys.exit(run(fig_views, "fig_views.pdf", "scripts/plots/03_fig_views.py"))
