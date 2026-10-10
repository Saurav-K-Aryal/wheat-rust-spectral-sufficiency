"""Figure 1, the spectra of the three classes and the noise of the HS bands.

(a) Median and interquartile range, over the fully valid training patches of each class, of the patch-mean
reflectance spectrum. (b) The spectral-residual signal-to-noise ratio of each band on a log axis. The deviation of a
pixel from the mean of its two neighbouring bands estimates the noise, and the ratio is the median over patches of
mean reflectance over that noise. The shaded span is the cleaned range kept for the cleaned HS view, and dotted lines
mark the HS bands that match the five MS channels.

Reads results/audit/band_snr.csv and results/audit/provenance.csv (written by the data audit, scripts/eda), so the
figure needs no image data. Writes outputs/figures/fig_data.pdf.
usage: python scripts/plots/01_fig_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from scripts.plots._style import (CLASS_STYLE, CLASSES, FULL_W, OI, blank, clean_range, finalize, letter, load_csv,
                                  ms_centres, ms_guides, run, set_log_ticks, shade_clean, warn)
from wheatrust.io_data import RESULTS

DN_SCALE = 10000.0  # reflectance = digital number / 10000


def fig_data(name, source):
    tag = "fig_data"
    snr = load_csv(tag, os.path.join(RESULTS, "audit", "band_snr.csv"), ["wavelength_nm"])
    if snr is None:
        return False
    snr = snr.sort_values("wavelength_nm")
    wl = snr["wavelength_nm"].to_numpy(float)
    rng = clean_range(tag, snr)
    centres = ms_centres(tag)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.45), layout="constrained")
    drawn = 0

    # (a) class median spectra with interquartile bands
    letter(ax1, 0)
    n_cls = 0
    for c in CLASSES:
        cols = [f"{c}_median", f"{c}_q25", f"{c}_q75"]
        if not all(k in snr for k in cols):
            warn(tag, f"band_snr.csv lacks {c} spectrum columns; class skipped")
            continue
        st = CLASS_STYLE[c]
        med, q25, q75 = (snr[k].to_numpy(float) / DN_SCALE for k in cols)
        ax1.fill_between(wl, q25, q75, color=st["color"], alpha=0.15, lw=0, zorder=2)
        ax1.plot(wl, med, color=st["color"], ls=st["ls"], lw=1.3, label=f"{c} median", zorder=3)
        n_cls += 1
    if n_cls:
        shade_clean(ax1, rng)
        ms_guides(ax1, centres)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Reflectance (DN/10000)")
        ax1.set_xlim(wl.min() - 4, wl.max() + 4)
        ax1.set_ylim(bottom=0)
        h, lab = ax1.get_legend_handles_labels()
        h.append(Patch(facecolor="#808080", alpha=0.35, lw=0))
        lab.append("Interquartile range")
        ax1.legend(h, lab, loc="upper left", handlelength=2.2, frameon=True, framealpha=0.92, edgecolor="none",
                   facecolor="white")
        drawn += 1
    else:
        blank(ax1)

    # (b) spectral-residual SNR on a log axis
    letter(ax2, 1)
    if "snr_spectral_residual" in snr:
        s = snr["snr_spectral_residual"].to_numpy(float)
        ok = np.isfinite(s) & (s > 0)
        shade_clean(ax2, rng, label=f"Cleaned range ({rng[0]:g} to {rng[1]:g} nm)" if rng else None)
        ms_guides(ax2, centres, label="MS band centres")
        ax2.plot(wl[ok], s[ok], color=OI["black"], lw=1.0, marker="o", ms=1.8, zorder=3)
        ax2.set_yscale("log")
        lo, hi = s[ok].min(), s[ok].max()
        ax2.set_ylim(lo / 1.6, hi * 1.8)
        cand = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
        set_log_ticks(ax2.yaxis, [t for t in cand if lo / 1.6 <= t <= hi * 1.8])
        ax2.set_xlim(wl.min() - 4, wl.max() + 4)
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Spectral-residual SNR (ratio)")
        ax2.legend(loc="upper left", handlelength=2.2, frameon=True, framealpha=0.92, edgecolor="none",
                   facecolor="white")
        drawn += 1
    else:
        warn(tag, "band_snr.csv lacks snr_spectral_residual; panel (b) skipped")
        blank(ax2)
    if not drawn:
        plt.close(fig)
        return False
    return finalize(fig, name, source)


if __name__ == "__main__":
    sys.exit(run(fig_data, "fig_data.pdf", "scripts/plots/01_fig_data.py"))
