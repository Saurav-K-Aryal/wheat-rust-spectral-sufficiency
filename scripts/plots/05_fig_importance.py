"""Figure 5, which spectral windows the decisions rely on.

Drop of the Health versus Rust AUC (top row) and of three-class macro-F1 (bottom row) when one 40 nm window of the
cleaned HS cube (left) or one band of the MS view (right) is permuted across the validation patches of an outer fold.
The AUC row shows FLAML and TabPFN on all Health and Rust patches and FLAML on vegetated patches only; the macro-F1 row
shows FLAML. Points are means over outer folds (each fold averages its permutations) with 95 percent intervals, and
series are dodged sideways so that their bars stay readable.

Reads results/E8/hr_window_folds.csv and hr_window_folds_veg.csv (the permutation runs of scripts/training); writes
outputs/figures/fig_importance.pdf.
usage: python scripts/plots/05_fig_importance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from scripts.plots._style import FULL_W, OI, finalize, letter, mean_ci, points, run, warn, window_centre
from wheatrust.io_data import RESULTS

# (file, HS model, MS model, column, label, colour, marker, line style, dodge in nm), per row of panels
AUC_SERIES = [("hr_window_folds.csv", "E4:hsc_eng", "E4:ms_eng", "drop_hr_auc", "FLAML, all patches", OI["vermillion"],
               "o", "-", -5.0),
              ("hr_window_folds_veg.csv", "E4:hsc_eng", "E4:ms_eng", "drop_hr_auc_veg", "FLAML, vegetated patches",
               OI["pink"], "^", (0, (1, 1.5)), 0.0),
              ("hr_window_folds.csv", "tabpfn8:hsc", "tabpfn8:ms", "drop_hr_auc", "TabPFN, all patches", OI["orange"],
               "s", (0, (4, 2)), 5.0)]
F1_SERIES = [("hr_window_folds.csv", "E4:hsc_eng", "E4:ms_eng", "drop_macro_f1", "FLAML, all patches", OI["blue"], "D",
              ":", 0.0)]


def draw(ax, data, series, which, handles):
    xfun = window_centre if which == 0 else (lambda w: float(w))
    for fname, hs_model, ms_model, col_, lab, col, mk, ls, dx in series:
        model = hs_model if which == 0 else ms_model
        d = data[fname]
        d = d[d.model == model].copy()
        if d.empty:
            warn("fig_importance", f"no permutation rows for {model} in {fname}")
            continue
        d["x"] = d.window.map(xfun)
        pf = d.groupby(["x", "repeat", "fold"], as_index=False)[col_].mean()
        sm = pd.DataFrame([(x,) + mean_ci(g[col_]) for x, g in pf.groupby("x")],
                          columns=["x", "mean", "hw", "n"]).sort_values("x")
        xx = sm.x.to_numpy(float) + (dx if which == 0 else dx * 2.5)
        ax.plot(xx, sm["mean"], color=col, ls=ls, lw=1.0, zorder=3)
        points(ax, xx, sm["mean"], sm["hw"], color=col, marker=mk, ms=3.4, zorder=4)
        if handles is not None and which == 0:
            handles.append(Line2D([], [], color=col, ls=ls, lw=1.0, marker=mk, ms=3.4, label=lab))
    ax.axhline(0, color="#777777", lw=0.6, zorder=2)


def fig_importance(name, source):
    data = {f: pd.read_csv(os.path.join(RESULTS, "E8", f)) for f in ("hr_window_folds.csv", "hr_window_folds_veg.csv")}
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.6), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.45, 0.75]))
    h_auc, h_f1 = [], []
    for row, series, handles, ylabel in [(0, AUC_SERIES, h_auc, "Drop in Health versus Rust AUC"),
                                         (1, F1_SERIES, h_f1, "Drop in three-class macro-F1")]:
        axh, axm = axes[row]
        letter(axh, 2 * row)
        letter(axm, 2 * row + 1)
        draw(axh, data, series, 0, handles)
        draw(axm, data, series, 1, None)
        axh.set_ylabel(ylabel)
        lo, hi = min(axh.get_ylim()[0], axm.get_ylim()[0]), max(axh.get_ylim()[1], axm.get_ylim()[1])
        axh.set_ylim(lo, hi)
        axm.set_ylim(lo, hi)
        cs = list(range(510, 871, 40))
        axh.set_xticks(cs, [str(c) for c in cs], rotation=45)
        axm.set_xticks([482, 550, 650, 738, 834], ["482", "550", "650", "738", "834"], rotation=45)
        axh.set_xlabel("40 nm window centre, cleaned HS models (nm)")
        axm.set_xlabel("Band, MS models (nm)")
        if row == 0:  # the macro-F1 row holds one series, which its axis label and the caption name
            axm.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92, edgecolor="none",
                       facecolor="white", handlelength=2.6)
    return finalize(fig, name, source)


if __name__ == "__main__":
    sys.exit(run(fig_importance, "fig_importance.pdf", "scripts/plots/05_fig_importance.py"))
