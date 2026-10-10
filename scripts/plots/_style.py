"""Shared style and helpers for the figures of the paper.

Design. Okabe-Ito colours, always paired with a marker, line style, or hatch so that no meaning relies on colour
alone; 8 pt sans-serif text (7 pt ticks and legends); TrueType fonts embedded (pdf.fonttype 42); no titles; light
solid grids; widths of 5.3 in (standard) and 6.7 in (full); bold panel letters at the top left of each panel.

Unless a figure says otherwise, error bars are 95 percent intervals of the mean over outer folds from the corrected
resampled t distribution of Nadeau and Bengio, t(0.975, n - 1) * sd * sqrt(1/n + 120/480). Outer folds of repeated
cross-validation share training data, so the naive standard error would be too small.

A missing file, column, or variant prints a WARNING line and the affected series or panel is skipped.
"""
from __future__ import annotations

import math
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from scripts.analysis._shared import N_TEST, N_TRAIN
from wheatrust.io_data import OUTPUTS, RESULTS

matplotlib.use("Agg")  # write files only, no display

FIGURES = os.path.join(OUTPUTS, "figures")

# Okabe-Ito colour-blind-safe palette
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
}
SHADE = "#000000"  # cleaned-range shading, drawn at low alpha
GUIDE = "#555555"  # dotted MS band-centre guides and reference lines
STD_W, FULL_W = 5.3, 6.7  # inches (13.5 cm and 17 cm)
CLASSES = ["Health", "Rust", "Other"]
CLASS_STYLE = {
    "Health": dict(color=OI["green"], ls="-", marker="o"),
    "Rust": dict(color=OI["vermillion"], ls="--", marker="s"),
    "Other": dict(color=OI["blue"], ls="-.", marker="^"),
}
# The HS bands that match the MS channels best, used only if results/audit/provenance.csv is unavailable
MS_CENTRES_FALLBACK = [482.0, 550.0, 650.0, 738.0, 834.0]

WARNINGS: list[str] = []


def warn(tag: str, msg: str) -> None:
    text = f"[{tag}] {msg}"
    WARNINGS.append(text)
    print("WARNING " + text, flush=True)


def setup_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E2E2E2",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.labelpad": 3.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        "errorbar.capsize": 1.5,
        "hatch.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "mathtext.default": "regular",
        "axes.unicode_minus": True,
    })


def load_csv(tag: str, path: str, required=(), **kw) -> pd.DataFrame | None:
    """Read a CSV, or warn and return None if it is missing, incomplete, or empty."""
    rel = os.path.relpath(path)
    if not os.path.exists(path):
        warn(tag, f"missing file {rel}")
        return None
    df = pd.read_csv(path, **kw)
    missing = [c for c in required if c not in df.columns]
    if missing:
        warn(tag, f"{rel} lacks columns {missing}")
        return None
    if df.empty:
        warn(tag, f"{rel} has no rows")
        return None
    return df


def ci_halfwidth(sd, n) -> float:
    """Half-width of a 95 percent interval for a mean over n outer folds, from the corrected resampled variance
    t(0.975, n - 1) * sd * sqrt(1/n + n_test/n_train). NaN when undefined (n < 2 or sd missing)."""
    from scipy.stats import t as tdist

    try:
        sd, n = float(sd), float(n)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(sd) and np.isfinite(n)) or n < 2:
        return float("nan")
    return float(tdist.ppf(0.975, n - 1)) * sd * math.sqrt(1.0 / n + N_TEST / N_TRAIN)


def mean_ci(values) -> tuple[float, float, int]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    sd = v.std(ddof=1) if v.size > 1 else float("nan")
    return float(v.mean()), ci_halfwidth(sd, v.size), int(v.size)


def points(ax, x, y, err, *, horizontal=False, color="k", marker="o", filled=True, ms=4.0,
           label=None, zorder=3, elw=0.8, alpha=1.0):
    """Markers with 95 percent interval bars (bars only where the half-width is finite)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    err = np.atleast_1d(np.asarray(err, dtype=float)) * np.ones_like(x)
    ok = np.isfinite(err) & np.isfinite(x) & np.isfinite(y)
    if ok.any():
        kw = dict(fmt="none", ecolor=color, elinewidth=elw, capthick=elw, zorder=zorder - 0.1, alpha=alpha)
        if horizontal:
            ax.errorbar(x[ok], y[ok], xerr=err[ok], **kw)
        else:
            ax.errorbar(x[ok], y[ok], yerr=err[ok], **kw)
    return ax.plot(x, y, ls="none", marker=marker, ms=ms, mfc=color if filled else "white", mec=color,
                   mew=0.9, color=color, label=label, zorder=zorder, alpha=alpha)[0]


def set_log_ticks(axis, ticks, fmt=None) -> None:
    ticks = sorted(set(float(t) for t in ticks))
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(fmt or (lambda v, _: f"{v:g}")))


def letter(ax, i: int):
    """Bold panel letter as a left title; finalize() aligns it with the panel's left edge."""
    t = ax.set_title(f"({'abcdefghijklmnop'[i]})", loc="left", fontsize=8, fontweight="bold", pad=4)
    ax.panel_letter = t
    return t


def blank(ax) -> None:
    ax.set_axis_off()
    if hasattr(ax, "panel_letter"):
        ax.panel_letter.set_text("")


def finalize(fig, name: str, source: str) -> bool:
    """Freeze the constrained layout, align panel letters, and save the PDF to outputs/figures."""
    fig.canvas.draw()
    fig.set_layout_engine("none")  # keep the computed positions fixed from here on
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        t = getattr(ax, "panel_letter", None)
        if t is None or not t.get_text() or not ax.axison:
            continue
        # left edge of the panel including tick labels and axis label, excluding legends
        tight = ax.get_tightbbox(renderer, bbox_extra_artists=[])
        win = ax.get_window_extent(renderer)
        if tight is not None and win.width > 0:
            t.set_x((tight.x0 - win.x0) / win.width)
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    # no dates in the metadata, so that a rebuild writes the same file
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None, "ModDate": None, "Creator": source})
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}", flush=True)
    return True


def clean_range(tag: str, snr: pd.DataFrame):
    if "in_clean_range" in snr:
        flag = snr["in_clean_range"].astype(str).str.lower().isin(["true", "1", "1.0"])
        wl = snr.loc[flag, "wavelength_nm"]
        if len(wl):
            return float(wl.min()), float(wl.max())
    warn(tag, "in_clean_range column unavailable; cleaned-range shading skipped")
    return None


def ms_centres(tag: str) -> list[float]:
    """Wavelengths of the HS bands that match the five MS channels best (results/audit/provenance.csv)."""
    prov = load_csv(tag, os.path.join(RESULTS, "audit", "provenance.csv"), ["view", "best_hs_wavelength_nm"])
    if prov is not None:
        c = prov.loc[prov["view"] == "MS", "best_hs_wavelength_nm"].dropna().astype(float)
        if len(c):
            return sorted(c.tolist())
    warn(tag, "MS band centres taken from the design values")
    return list(MS_CENTRES_FALLBACK)


def shade_clean(ax, rng, label=None) -> None:
    if rng is not None:
        ax.axvspan(rng[0], rng[1], color=SHADE, alpha=0.06, lw=0, zorder=0, label=label)


def ms_guides(ax, centres, label=None) -> None:
    for i, c in enumerate(centres):
        ax.axvline(c, color=GUIDE, ls=":", lw=0.8, zorder=1, label=label if i == 0 else None)


def window_centre(w: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", str(w))
    return (float(m.group(1)) + float(m.group(2))) / 2 if m else float("nan")


def run(fig_fn, name: str, source: str) -> int:
    """Draw one figure with the shared style; exit status 1 when it could not be drawn."""
    setup_style()
    ok = fig_fn(name, source)
    if not ok:
        print(f"skipped {name} (no drawable data)")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s), listed above with WARNING")
    return 0 if ok else 1
