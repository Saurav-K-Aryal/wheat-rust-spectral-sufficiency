"""Figure 4, band budgets and the bands that in-fold selection picked.

(a) Macro-F1 of 1 to 32 bands selected inside each outer training fold, with FLAML (filled markers) and TabPFN (open
markers), beside the fixed RGB (482, 550, 650 nm) and MS (482, 550, 650, 738, 834 nm) placements at 3 and 5 bands;
every band set uses the same generic band features. Above eight bands those features no longer include the pairwise
normalised differences, so the curves break there. Markers at the same budget are dodged slightly by learner. The
dotted line is the all-views FLAML baseline, and the bars are 95 percent intervals over outer folds. (b) How many of the selected bands per fold fell into each 20 nm bin, for budgets of
five and eight bands, with dotted lines at the HS bands that match the MS channels.

Reads results/analysis/all_fold_metrics.csv and e9_selection_frequency.csv, and results/audit/provenance.csv;
writes outputs/figures/fig_sensor.pdf.
usage: python scripts/plots/04_fig_sensor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.analysis._shared import BASE, all_fold_metrics, analysis_file
from scripts.plots._style import (FULL_W, GUIDE, OI, finalize, letter, mean_ci, ms_centres, ms_guides, points, run,
                                  set_log_ticks, warn)

KS = [1, 2, 3, 5, 8, 16, 32]


def fig_sensor(name, source):
    fm = all_fold_metrics(["key", "repeat", "fold", "macro_f1"])
    freq = pd.read_csv(analysis_file("e9_selection_frequency.csv"))
    centres = ms_centres("fig_sensor")

    def stat(k):
        v = fm.loc[fm.key == k, "macro_f1"].to_numpy(float)
        return mean_ci(v) if len(v) else (np.nan, np.nan, 0)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL_W, 2.55), layout="constrained",
                                   gridspec_kw=dict(width_ratios=[1.2, 1.0]))
    letter(axa, 0)
    handles, ys = [], []
    for lrn, pre, col, mk, filled in [("FLAML", "E9:", OI["blue"], "o", True),
                                      ("TabPFN", "E9t:tabpfn_", OI["orange"], "s", False)]:
        pts = [(k,) + stat(f"{pre}bandsel_k{k}") for k in KS]
        pts = [p for p in pts if p[3] > 0]
        if not pts:
            warn("fig_sensor", f"no band-budget results for {lrn}")
            continue
        k_, m_, h_, _ = map(np.array, zip(*pts))
        dx = 2 ** (-0.05 if lrn == "FLAML" else 0.05)  # small dodge on the log axis so the learners do not overlap
        # the generic features drop the pairwise differences above eight bands, so the line is not drawn across
        for seg in (k_ <= 8, k_ > 8):
            axa.plot(k_[seg] * dx, m_[seg], color=col, lw=1.1, zorder=3)
        points(axa, k_ * dx, m_, h_, color=col, marker=mk, filled=filled, ms=4.0, zorder=4)
        handles.append(Line2D([], [], color=col, marker=mk, mfc=col if filled else "white", mec=col, ms=4.0, lw=1.1,
                              label=f"Selected bands, {lrn}"))
        ys += list(m_ - np.nan_to_num(h_)) + list(m_ + np.nan_to_num(h_))
        for v, nb, fcol, fmk, flab in [("fixed_rgb3", 3, OI["vermillion"], "^", "Fixed RGB placement"),
                                       ("fixed_ms5", 5, OI["green"], "D", "Fixed MS placement")]:
            m, h, n = stat(f"{pre}{v}")
            if n == 0:
                continue
            # a wider dodge than the selected bands, so the two kinds of marker at 3 and 5 bands stay apart
            points(axa, nb * 2 ** (-0.16 if lrn == "FLAML" else 0.16), m, h, color=fcol, marker=fmk, filled=filled,
                   ms=4.6, zorder=5)
            ys += [m - np.nan_to_num(h), m + np.nan_to_num(h)]
            handles.append(Line2D([], [], ls="none", color=fcol, marker=fmk, mfc=fcol if filled else "white", mec=fcol,
                                  ms=4.6, label=f"{flab}, {lrn}"))
    axa.axvline(np.sqrt(8 * 16), color=GUIDE, ls=(0, (1, 2)), lw=0.7, zorder=1)
    # the legend fills column by column, so pairing the learners puts FLAML in the first row and TabPFN in the second
    order = ["Selected bands", "Fixed RGB placement", "Fixed MS placement"]
    handles = [h for o in order for lrn in ("FLAML", "TabPFN") for h in handles if h.get_label() == f"{o}, {lrn}"]
    base = stat(BASE)
    if base[2]:
        axa.axhline(base[0], color=OI["black"], ls=":", lw=0.9, zorder=2)
        handles.append(Line2D([], [], color=OI["black"], ls=":", lw=0.9, label="All-views baseline (FLAML)"))
        ys.append(base[0])
    axa.set_xscale("log", base=2)
    set_log_ticks(axa.xaxis, KS, lambda v, _: f"{v:.0f}")
    axa.set_xlim(1 / 2 ** 0.45, 32 * 2 ** 0.45)
    if ys:
        axa.set_ylim(min(ys) - 0.02, max(ys) + 0.02)
    axa.set_xlabel("Number of spectral bands")
    axa.set_ylabel("Macro-F1 (outer folds)")

    letter(axb, 1)
    edges = np.arange(490, 891, 20)
    bh, ymax = [], 0.0
    for k, col, off in [(5, OI["blue"], -0.25), (8, OI["sky"], 0.25)]:
        f = freq[freq["k"].astype(float) == k]
        if f.empty:
            continue
        # frequency is the share of folds that selected a wavelength, so the bin sums are bands per fold
        idx = np.clip(np.digitize(f.wavelength.to_numpy(float), edges) - 1, 0, len(edges) - 2)
        per_bin = np.bincount(idx, weights=f.frequency.to_numpy(float), minlength=len(edges) - 1)
        axb.bar(edges[:-1] + 10 + off * 20 * 0.8, per_bin, width=20 * 0.4, color=col, edgecolor="white", lw=0.4,
                zorder=3)
        bh.append(Patch(facecolor=col, edgecolor="white", label=f"{k} bands"))
        ymax = max(ymax, float(per_bin.max()))
    ms_guides(axb, centres)
    bh.append(Line2D([], [], color=GUIDE, ls=":", lw=0.8, label="MS band centres"))
    axb.set_xlim(470, 910)
    axb.set_ylim(0, ymax * 1.3 + 0.05)
    axb.set_xlabel("Wavelength bin (nm)")
    axb.set_ylabel("Selected bands per fold")
    axb.legend(handles=bh, loc="upper right", handlelength=1.4, frameon=True, framealpha=0.92, edgecolor="none",
               facecolor="white")
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    return finalize(fig, name, source)


if __name__ == "__main__":
    sys.exit(run(fig_sensor, "fig_sensor.pdf", "scripts/plots/04_fig_sensor.py"))
