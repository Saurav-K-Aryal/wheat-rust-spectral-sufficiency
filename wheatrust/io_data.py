"""Paths, data access, and the conventions of the three views.

Every location the code reads or writes is defined here.
  ROOT     the repository root
  DATA     the competition data, never written (environment variable WHEATRUST_DATA, default ROOT/data)
  CACHE    arrays and feature tables computed once from DATA (ROOT/cache)
  RESULTS  experiment results (ROOT/results)
  OUTPUTS  numbers, tables, and figures of the paper (ROOT/outputs)
  MODELS   the TabPFN checkpoint (environment variable WHEATRUST_MODELS, default ROOT/models)

The competition ships every patch as an RGB PNG, a five-band MS GeoTIFF, and an HS GeoTIFF. All three are views of
one hyperspectral acquisition with a Cubert S185 snapshot camera. The audit (scripts/eda/01_audit.py) finds each MS
band to be a single HS band resampled onto a grid twice as fine, and each RGB channel to be a single HS band after a
per-image contrast stretch, so the views differ in processing and band placement, not in sensor.

Conventions
* HS cubes are (N, 32, 32, 125) uint16. Some files carry a 126th band of constant fill, which is dropped. The raw
  band count is kept (hs_nbands_raw) because the two file formats behave like separate processing batches.
* The files carry no wavelengths. The camera records 125 bands between 450 and 950 nm, and HS band i (counted from
  zero) is taken to be centred at 450 + 4 i nm, the 4 nm step reported for this camera. Under this assignment the
  HS bands that the audit matches to the MS channels lie within 2 nm of the nominal MS centres given with the data.
* MS cubes are (N, 64, 64, 5) uint16 and RGB images (N, 64, 64, 3) uint8.
* MS and HS digital numbers divided by 10 000 give reflectance.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.path.expanduser(os.environ.get("WHEATRUST_DATA") or os.path.join(ROOT, "data")))
CACHE = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")
OUTPUTS = os.path.join(ROOT, "outputs")
MODELS = os.path.abspath(os.path.expanduser(os.environ.get("WHEATRUST_MODELS") or os.path.join(ROOT, "models")))

CLASSES = ["Health", "Rust", "Other"]
HS_WL = 450.0 + 4.0 * np.arange(125)
# MS band centres as given in the dataset description; the audit places them at 482, 550, 650, 738, and 834 nm
MS_WL_NOMINAL = np.array([480.0, 550.0, 650.0, 740.0, 833.0])
# The dataset notes flag the first 10 and the last 14 HS bands as possibly noisy. The cleaned view keeps the 101
# bands in between (0-based indices 10 to 110, 490 to 890 nm).
CLEAN_IDX = np.arange(10, 111)


def sample_table() -> pd.DataFrame:
    """One row per sample with its RGB, MS, and HS files.

    Files are recorded as data/<split>/<view>/<name>, the layout of the default data folder, whatever WHEATRUST_DATA
    points to, so the table and every result copied from it stay the same on any machine. data_file() resolves them.
    """
    rows = []
    for split in ["train", "val"]:
        ids = set()
        files = {}
        for mod, ext in [("RGB", ".png"), ("MS", ".tif"), ("HS", ".tif")]:
            for p in glob.glob(os.path.join(DATA, split, mod, "*" + ext)):
                name = os.path.basename(p)
                sid = name[: -len(ext)]
                ids.add(sid)
                files[(sid, mod)] = f"data/{split}/{mod}/{name}"
        for sid in sorted(ids):
            # training files are named <class>_hyper_<n>; holdout names carry no label
            label = sid.split("_")[0] if split == "train" else None
            rows.append(
                dict(
                    sample_id=sid,
                    split=split,
                    label=label,
                    rgb_file=files.get((sid, "RGB")),
                    ms_file=files.get((sid, "MS")),
                    hs_file=files.get((sid, "HS")),
                )
            )
    df = pd.DataFrame(rows)
    df["missing_modality"] = df[["rgb_file", "ms_file", "hs_file"]].isna().any(axis=1)
    return df


def data_file(rel: str) -> str:
    """Absolute path of a file recorded in the sample table as data/<split>/<view>/<name>."""
    return os.path.join(DATA, *rel.replace("\\", "/").split("/")[1:])


def build_cache(force: bool = False) -> str:
    """Read every file once and store the three views as uint16 and uint8 arrays in CACHE."""
    marker = os.path.join(CACHE, "done.txt")
    if os.path.exists(marker) and not force:
        return CACHE
    os.makedirs(CACHE, exist_ok=True)
    df = sample_table()
    if not len(df):
        raise FileNotFoundError(f"no competition data under {DATA} (set WHEATRUST_DATA or see data/README.md)")
    hs, ms, rgb, nb = [], [], [], []
    for r in df.itertuples():
        a = tifffile.imread(data_file(r.hs_file))
        nb.append(a.shape[-1])
        hs.append(a[..., :125].astype(np.uint16))
        ms.append(tifffile.imread(data_file(r.ms_file)).astype(np.uint16))
        rgb.append(np.asarray(Image.open(data_file(r.rgb_file)).convert("RGB")).astype(np.uint8))
    np.save(os.path.join(CACHE, "hs.npy"), np.stack(hs))
    np.save(os.path.join(CACHE, "ms.npy"), np.stack(ms))
    np.save(os.path.join(CACHE, "rgb.npy"), np.stack(rgb))
    np.save(os.path.join(CACHE, "hs_nbands_raw.npy"), np.array(nb, dtype=np.int16))
    df.to_csv(os.path.join(CACHE, "sample_table.csv"), index=False)
    with open(marker, "w") as f:
        f.write("ok\n")
    return CACHE


class Arrays:
    """The cached arrays (memory-mapped) with the sample table, split into training and holdout positions."""

    def __init__(self):
        build_cache()
        df = pd.read_csv(os.path.join(CACHE, "sample_table.csv"))
        self.table = df
        self.sample_id = np.array(df.sample_id.tolist(), dtype=str)
        self.split = np.array(df.split.tolist(), dtype=str)
        self.label = np.array(df.label.fillna("").tolist(), dtype=str)
        self.hs = np.load(os.path.join(CACHE, "hs.npy"), mmap_mode="r")
        self.ms = np.load(os.path.join(CACHE, "ms.npy"), mmap_mode="r")
        self.rgb = np.load(os.path.join(CACHE, "rgb.npy"), mmap_mode="r")
        self.hs_nbands_raw = np.load(os.path.join(CACHE, "hs_nbands_raw.npy"))
        self.tr = np.where(self.split == "train")[0]
        self.va = np.where(self.split == "val")[0]

    def y_train(self):
        """Training labels as class indices (0 Health, 1 Rust, 2 Other), in the order of self.tr."""
        return np.array([CLASSES.index(c) for c in self.label[self.tr]])
