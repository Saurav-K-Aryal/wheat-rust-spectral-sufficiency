"""Supplementary figure, how well each HS band separates Rust from Health patches on its own.

The ROC AUC of the patch-mean reflectance of every HS band for Rust against Health patches, on vegetated patches
(solid) and on every non-empty patch (dashed), with 95 percent bootstrap intervals as bands. Values above 0.5 mean
that Rust patches reflect more at that wavelength. The cleaned range is shaded and the five MS band centres are
dotted.

Reads results/audit/band_separability.csv (scripts/eda/04_band_separability.py, which needs the images) and
results/audit/band_snr.csv; writes outputs/figures/fig_band_separability.pdf.
usage: python scripts/plots/06_fig_band_separability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import matplotlib.pyplot as plt

from scripts.plots._style import (GUIDE, OI, STD_W, clean_range, finalize, load_csv, ms_centres, ms_guides, run,
                                  shade_clean)
from wheatrust.io_data import RESULTS

STYLE = {"vegetated": dict(color=OI["vermillion"], ls="-", label="Vegetated patches"),
         "all": dict(color=OI["blue"], ls="--", label="All non-empty patches")}


def fig_band_separability(name, source):
    tag = "fig_band_separability"
    sep = load_csv(tag, os.path.join(RESULTS, "audit", "band_separability.csv"),
                   ["kind", "subset", "wavelength_nm", "auc", "auc_lo95", "auc_hi95"])
    if sep is None:
        return False
    snr = load_csv(tag, os.path.join(RESULTS, "audit", "band_snr.csv"), ["wavelength_nm"])
    fig, ax = plt.subplots(figsize=(STD_W, 2.4), layout="constrained")
    shade_clean(ax, clean_range(tag, snr) if snr is not None else None, label="Cleaned range")
    ms_guides(ax, ms_centres(tag), label="MS band centres")
    ax.axhline(0.5, color=GUIDE, lw=0.8, zorder=1)
    for subset, st in STYLE.items():
        d = sep[(sep.kind == "band") & (sep.subset == subset)].sort_values("wavelength_nm")
        if d.empty:
            continue
        n_h, n_r = int(d.n_health.iloc[0]), int(d.n_rust.iloc[0])
        ax.fill_between(d.wavelength_nm, d.auc_lo95, d.auc_hi95, color=st["color"], alpha=0.15, lw=0, zorder=2)
        ax.plot(d.wavelength_nm, d.auc, color=st["color"], ls=st["ls"], lw=1.2, zorder=3,
                label=f"{st['label']} ({n_h} Health, {n_r} Rust)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("AUC, Rust against Health")
    ax.set_xlim(450, 946)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)  # above the panel, clear of the curves
    return finalize(fig, name, source)


if __name__ == "__main__":
    sys.exit(run(fig_band_separability, "fig_band_separability.pdf", "scripts/plots/06_fig_band_separability.py"))
