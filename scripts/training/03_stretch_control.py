"""Stretch control, the RGB features of the three HS bands behind the RGB view after the provider's processing.

The provided RGB PNGs are the HS bands at 650, 550, and 482 nm after a per-image, per-channel linear contrast
stretch with 8-bit quantisation, resampled to a 64 x 64 grid. This table applies the same 2 to 98 percent stretch
and quantisation to those HS bands on their own 32 x 32 grid with the HS valid-pixel mask. Compared with the RGB
features of the unstretched HS bands (rgbhs) it isolates the stretch and quantisation; compared with the PNG
features (rgb) it isolates the remaining differences, namely the finer grid, the PNG valid-pixel mask, and texture
computed on that grid.

Reads   the array cache
Writes  cache/features/rgbhsst.pkl
usage: python scripts/training/03_stretch_control.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import os  # noqa: E402

import numpy as np  # noqa: E402

from wheatrust.common import FEAT_DIR  # noqa: E402
from wheatrust.features import HS_WL, band, rgb_features, stretch_8bit  # noqa: E402
from wheatrust.io_data import Arrays  # noqa: E402


def main():
    A = Arrays()
    hs = np.asarray(A.hs, dtype=np.float32) / 10000.0
    mask = ~(hs == 0).all(-1)
    b = [band(w, HS_WL) for w in (650, 550, 482)]
    rgb_st = stretch_8bit(hs[..., b], mask)
    T = rgb_features(rgb_st, mask, "rgbhsst")
    T.index = A.sample_id
    T = T.astype(np.float32)
    os.makedirs(FEAT_DIR, exist_ok=True)
    out = os.path.join(FEAT_DIR, "rgbhsst.pkl")
    T.to_pickle(out)
    clip0 = float(((rgb_st == 0) & mask[..., None]).mean() / mask.mean())
    clip1 = float(((rgb_st == 1) & mask[..., None]).mean() / mask.mean())
    print(f"wrote {out} {T.shape}; share of valid pixel values at 0 and 1: {clip0:.4f}, {clip1:.4f}")


if __name__ == "__main__":
    main()
