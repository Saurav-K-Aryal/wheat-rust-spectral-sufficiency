"""Data audit of files, view provenance, band noise, duplicates and overlaps, and canopy cover.

Reads the competition data (DATA) through the array cache, which it builds on first use. Holdout inputs enter only
the file inventory and the duplicate and overlap checks.

Writes to results/audit/
  file_inventory.csv      shape, type, tags, and valid-pixel fractions of every file
  sample_table.csv        the RGB, MS, and HS file of every sample
  geotiff_metadata.csv    coordinate reference system, transform, and tags of every MS and HS file
  provenance.csv          the HS band that best explains each MS and RGB channel
  band_snr.csv            spectral-residual SNR per HS band and class median spectra
  blue_index_noise.csv    ExG and VARI computed with different blue bands
  duplicates.csv          exact duplicates, overlapping patches (shared 4 x 4 blocks), and near duplicates
  nearest_neighbour.csv   nearest patch of every fully valid patch under rotations and flips
  composition_ndvi.csv    patch NDVI statistics of the training patches
  audit_summary.json      every number above that the paper quotes

usage: python scripts/eda/01_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402

from wheatrust.io_data import CLASSES, CLEAN_IDX, DATA, HS_WL, RESULTS, Arrays, data_file, sample_table  # noqa: E402

OUT = os.path.join(RESULTS, "audit")


def r2(x, y):
    """Squared Pearson correlation of two vectors."""
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) ** 2 / ((x @ x) * (y @ y) + 1e-12))


def inventory(A, tab, hs, ms, rgb, S):
    """File formats, valid-pixel fractions, and the constant fill band of the 126-band HS files."""
    tr = A.split == "train"
    va = A.split == "val"
    inv = []
    for r in tab.itertuples():
        with tifffile.TiffFile(data_file(r.hs_file)) as tf:
            p = tf.pages[0]
            hs_shape, hs_dtype, hs_geo = p.shape, str(p.dtype), tf.is_geotiff
            hs_tags = sorted(t.name for t in p.tags.values())
        with tifffile.TiffFile(data_file(r.ms_file)) as tf:
            p = tf.pages[0]
            ms_shape, ms_dtype, ms_geo = p.shape, str(p.dtype), tf.is_geotiff
        im = Image.open(data_file(r.rgb_file))
        inv.append(
            dict(
                sample_id=r.sample_id,
                split=r.split,
                label=r.label,
                hs_shape="x".join(map(str, hs_shape)),
                hs_dtype=hs_dtype,
                hs_bands_raw=hs_shape[-1],
                ms_shape="x".join(map(str, ms_shape)),
                ms_dtype=ms_dtype,
                rgb_size=f"{im.size[0]}x{im.size[1]}",
                rgb_mode=im.mode,
                hs_geotiff=hs_geo,
                ms_geotiff=ms_geo,
                hs_tags=";".join(hs_tags),
            )
        )
    inv = pd.DataFrame(inv)
    inv["hs_valid_frac"] = (~(hs == 0).all(-1)).mean((1, 2))
    inv["ms_valid_frac"] = (~(ms == 0).all(-1)).mean((1, 2))
    inv["rgb_valid_frac"] = (~(rgb == 0).all(-1)).mean((1, 2))
    inv["empty"] = inv.hs_valid_frac == 0
    inv["partially_empty"] = (inv.hs_valid_frac > 0) & (inv.hs_valid_frac < 0.99)
    fill = []
    for r in tab.itertuples():
        a = tifffile.imread(data_file(r.hs_file))
        if a.shape[-1] == 126:
            u = np.unique(a[..., 125])
            fill.append(";".join(map(str, u[:4])))
        else:
            fill.append("")
    inv["hs_band126_values"] = fill
    inv.to_csv(os.path.join(OUT, "file_inventory.csv"), index=False)
    tab.to_csv(os.path.join(OUT, "sample_table.csv"), index=False)

    cc = inv[inv.split == "train"].groupby("label").size().reindex(CLASSES)
    S["n_train"] = int(tr.sum())
    S["n_holdout"] = int(va.sum())
    S["class_counts"] = {k: int(v) for k, v in cc.items()}
    S["missing_modality"] = int(tab.missing_modality.sum())
    # holdout Ids in a submission are HS file names, so the three files of a holdout sample must share one stem
    S["holdout_ids_share_stem"] = bool(
        all(
            os.path.basename(r.ms_file) == os.path.basename(r.hs_file) == r.sample_id + ".tif"
            and os.path.basename(r.rgb_file) == r.sample_id + ".png"
            for r in tab[tab.split == "val"].itertuples()
        )
    )
    S["sample_submission_present"] = bool(any(f.lower().startswith("sample") for f in os.listdir(DATA)))
    S["hs_bands_raw_counts"] = {
        f"{s}_{b}": int(v) for (s, b), v in inv.groupby(["split", "hs_bands_raw"]).size().items()
    }
    S["hs126_by_class"] = {
        k: int(v) for k, v in inv[(inv.split == "train") & (inv.hs_bands_raw == 126)].groupby("label").size().items()
    }
    S["empty_train_by_class"] = {
        k: int(v) for k, v in inv[(inv.split == "train") & inv["empty"]].groupby("label").size().items()
    }
    S["partial_train_by_class"] = {
        k: int(v) for k, v in inv[(inv.split == "train") & inv.partially_empty].groupby("label").size().items()
    }
    S["empty_holdout"] = int(inv[(inv.split == "val")]["empty"].sum())
    S["partial_holdout"] = int(inv[(inv.split == "val")].partially_empty.sum())
    S["shapes"] = {
        "hs": sorted(inv.hs_shape.unique().tolist()),
        "ms": sorted(inv.ms_shape.unique().tolist()),
        "rgb": sorted(inv.rgb_size.unique().tolist()),
        "rgb_mode": sorted(inv.rgb_mode.unique().tolist()),
        "dtypes": sorted(set(inv.hs_dtype) | set(inv.ms_dtype)),
    }
    return inv


def georeferencing(tab, inv, S):
    """Whether the GeoTIFFs carry a coordinate system or a transform (they do not)."""
    geo = []
    try:
        import rasterio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for r in tab.itertuples():
                for mod in ["hs_file", "ms_file"]:
                    with rasterio.open(data_file(getattr(r, mod))) as ds:
                        geo.append(
                            dict(
                                sample_id=r.sample_id,
                                file=getattr(r, mod),
                                crs=str(ds.crs),
                                transform_is_identity=ds.transform.is_identity,
                                tags=json.dumps(ds.tags()),
                            )
                        )
    except Exception as e:
        geo.append(dict(sample_id="", file="", crs=f"rasterio failed: {e}", transform_is_identity=None, tags=""))
    geo = pd.DataFrame(geo)
    geo.to_csv(os.path.join(OUT, "geotiff_metadata.csv"), index=False)
    S["n_files_with_crs"] = int((geo.crs.notna() & (geo.crs != "None")).sum())
    S["n_files_non_identity_transform"] = int((geo.transform_is_identity == False).sum())  # noqa: E712
    S["any_geotiff_tags"] = bool(inv.hs_geotiff.any() or inv.ms_geotiff.any())
    S["tiff_tag_set"] = sorted(set(";".join(inv.hs_tags).split(";")))


def provenance(A, hs, ms, rgb, full, S):
    """Which HS band each MS and RGB channel comes from, and the per-image stretch of the RGB PNGs.

    Each MS and RGB channel is block-averaged 2 x 2 onto the 32 x 32 HS grid and correlated with every HS band,
    per image (the per-image R^2, insensitive to any per-image gain and offset) and pooled over all fully valid
    training patches in the original units (the pooled R^2 and slope, which an image-wise stretch destroys).
    """
    N = len(hs)
    tr = A.split == "train"
    ms_d = ms.reshape(N, 32, 2, 32, 2, 5).mean((2, 4))
    rgb_d = rgb.reshape(N, 32, 2, 32, 2, 3).mean((2, 4))
    idx = np.where(full & tr)[0]
    rows = []
    for view, M, nb, nominal in [
        ("MS", ms_d, 5, [480, 550, 650, 740, 833]),
        ("RGB", rgb_d, 3, ["red", "green", "blue"]),
    ]:
        for b in range(nb):
            per_img = np.zeros((len(idx), 125))
            for j, i in enumerate(idx):
                y = M[i][..., b].ravel()
                H = hs[i].reshape(-1, 125)
                yc = y - y.mean()
                Hc = H - H.mean(0)
                per_img[j] = (Hc.T @ yc) ** 2 / ((Hc**2).sum(0) * (yc @ yc) + 1e-12)
            med = np.median(per_img, 0)
            k = int(med.argmax())
            yp = M[idx][..., b].ravel()
            xp = hs[idx][..., k].ravel()
            pooled = r2(xp, yp)
            slope, intercept = np.polyfit(xp, yp, 1)
            rows.append(
                dict(
                    view=view,
                    channel=b,
                    nominal=nominal[b],
                    best_hs_band_1based=k + 1,
                    best_hs_wavelength_nm=HS_WL[k],
                    median_per_image_r2=float(med[k]),
                    p05_per_image_r2=float(np.percentile(per_img[:, k], 5)),
                    second_best_band_r2=float(np.sort(med)[-2]),
                    pooled_r2=pooled,
                    pooled_slope=float(slope),
                    pooled_intercept=float(intercept),
                    n_images=len(idx),
                )
            )
    prov = pd.DataFrame(rows)
    prov.to_csv(os.path.join(OUT, "provenance.csv"), index=False)
    S["provenance"] = prov.to_dict(orient="records")

    # Signs of a per-image stretch. Every PNG spans 0 to 255 and a fixed share of pixels sits at each end, while
    # the HS reflectance level varies between patches.
    ne = np.where(full)[0]
    r_flat = rgb[ne].reshape(len(ne), -1, 3)
    S["rgb_frac_at_0_median"] = float(np.median((r_flat == 0).mean(1)))
    S["rgb_frac_at_255_median"] = float(np.median((r_flat == 255).mean(1)))
    S["rgb_min_is_0_frac_images"] = float(((r_flat.min(1) == 0).all(1)).mean())
    S["rgb_max_is_255_frac_images"] = float(((r_flat.max(1) == 255).all(1)).mean())
    S["hs650_patch_mean_cv"] = float(hs[ne][..., 49].mean((1, 2)).std() / hs[ne][..., 49].mean())
    # A 2 to 98 percent linear stretch of HS bands, upsampled to the PNG grid, against the PNG channels of the first
    # 300 fully valid patches. The bands are 0-based 49, 24, and 7 (646, 546, and 478 nm), each one band below
    # the RGB bands that the provenance regression identifies.
    stretch_err = []
    for i in ne[:300]:
        for c, b in enumerate([49, 24, 7]):
            x = np.repeat(np.repeat(hs[i][..., b], 2, 0), 2, 1)
            lo, hi = np.percentile(x, [2, 98])
            pred = np.clip((x - lo) / (hi - lo + 1e-9) * 255, 0, 255)
            stretch_err.append(np.corrcoef(pred.ravel(), rgb[i][..., c].ravel())[0, 1])
    S["rgb_stretch_reconstruction_corr_median"] = float(np.median(stretch_err))
    S["ms_blockmean_r2_min"] = float(prov[prov.view == "MS"].median_per_image_r2.min())


def band_noise(A, hs, full, S):
    """Spectral-residual SNR per band and class median spectra (fully valid training patches).

    The noise of band b is estimated from its deviation from the mean of its two neighbours, R_b - (R_b-1 + R_b+1)/2,
    whose standard deviation is sqrt(1.5) times the white-noise level when the spectrum is locally linear. The SNR
    is the patch mean over that level, with the median taken over patches.
    """
    tr = A.split == "train"
    fidx = np.where(full & tr)[0]
    X = hs[fidx].reshape(len(fidx), -1, 125)
    resid = X[:, :, 1:-1] - 0.5 * (X[:, :, :-2] + X[:, :, 2:])
    noise = resid.std(1) / np.sqrt(1.5)
    sig = X[:, :, 1:-1].mean(1)
    snr = np.median(sig / (noise + 1e-9), 0)
    band_rows = pd.DataFrame(
        dict(
            band_1based=np.arange(2, 125),
            wavelength_nm=HS_WL[1:-1],
            snr_spectral_residual=snr,
            in_clean_range=[(b - 1) in set(CLEAN_IDX) for b in range(2, 125)],
        )
    )
    for c in CLASSES:
        m = fidx[A.label[fidx] == c]
        spec = hs[m].reshape(len(m), -1, 125).mean(1)
        for q, nm in [(50, "median"), (25, "q25"), (75, "q75")]:
            band_rows[f"{c}_{nm}"] = np.percentile(spec, q, axis=0)[1:-1]
    band_rows.to_csv(os.path.join(OUT, "band_snr.csv"), index=False)
    # snr starts at 1-based band 2, so snr[b - 2] belongs to 1-based band b
    S["snr_median_clean"] = float(np.median(snr[[(b - 1) in set(CLEAN_IDX) for b in range(2, 125)]]))
    S["snr_median_noisy_low"] = float(np.median(snr[: 10 - 1]))
    S["snr_median_noisy_high"] = float(np.median(snr[111 - 1:]))
    S["snr_at_482"] = float(snr[8 - 1])
    S["snr_at_490"] = float(snr[10 - 1])


def blue_index_noise(A, hs, full):
    """Whether blue-dependent RGB indices are limited by the noise of the 482 nm band.

    ExG and VARI are computed from HS bands with four choices of blue (482, 490, or 498 nm, or the mean of 478 to
    486 nm) and with red and green at 0-based bands 49 and 24 (646 and 546 nm). For each, the within-patch noise
    (a robust standard deviation of horizontal pixel differences) is set against the spread of patch medians, and
    the class separation of the patch median is measured by the Kruskal-Wallis epsilon squared and by the
    cross-validated macro-F1 of a logistic regression on that single value.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = A.split == "train"
    labs = np.array([CLASSES.index(c) if c else -1 for c in A.label])
    fi = np.where(full & tr)[0]
    R = hs[fi][..., 49]
    G = hs[fi][..., 24]
    blues = {"482nm_b9": hs[fi][..., 8], "490nm_b11": hs[fi][..., 10], "498nm_b13": hs[fi][..., 12],
             "mean478_486": hs[fi][..., 7:10].mean(-1)}
    idx_rows = []
    for bname, B in blues.items():
        tot = R + G + B + 1e-6
        exg = 2 * G / tot - R / tot - B / tot
        vari = (G - R) / np.where(np.abs(G + R - B) < 1e-3, 1e-3, G + R - B)
        for iname, I in [("ExG", exg), ("VARI", vari)]:
            pm = np.median(I.reshape(len(fi), -1), 1)
            d = np.diff(I, axis=2).reshape(len(fi), -1)
            hf = np.median(1.4826 * np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1) / np.sqrt(2))
            between = float(np.std(pm))
            kw = stats.kruskal(*[pm[labs[fi] == k] for k in range(3)])
            eps2 = (kw.statistic - 3 + 1) / (len(pm) - 3)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            pred = cross_val_predict(clf, pm.reshape(-1, 1), labs[fi],
                                     cv=StratifiedKFold(5, shuffle=True, random_state=0))
            idx_rows.append(
                dict(
                    index=iname,
                    blue=bname,
                    within_patch_noise=float(hf),
                    between_patch_sd=between,
                    noise_to_between_ratio=float(hf / between),
                    kruskal_H=float(kw.statistic),
                    epsilon_squared=float(eps2),
                    single_index_cv_macro_f1=float(f1_score(labs[fi], pred, average="macro")),
                    n=len(fi),
                )
            )
    pd.DataFrame(idx_rows).to_csv(os.path.join(OUT, "blue_index_noise.csv"), index=False)


def dihedral(x):
    """The eight rotations and flips of a stack of (N, H, W, ...) patches."""
    out = []
    for k in range(4):
        r_ = np.rot90(x, k, axes=(1, 2))
        out.append(r_)
        out.append(r_[:, :, ::-1])
    return out


def duplicates(A, hs, valid_hs, S):
    """Exact duplicates, overlapping patches, near duplicates, and adjacency of patch edges, over all 900 patches."""
    N = len(hs)
    dup_rows = []
    hashes = {}
    for i in range(N):
        if not valid_hs[i].any():
            continue
        for mod, arr in [("HS", A.hs), ("MS", A.ms), ("RGB", A.rgb)]:
            h = hashlib.md5(np.ascontiguousarray(arr[i]).tobytes()).hexdigest()
            hashes.setdefault((mod, h), []).append(i)
    for (mod, h), lst in hashes.items():
        if len(lst) > 1:
            for a in lst:
                for b in lst:
                    if a < b:
                        dup_rows.append(dict(kind="exact", modality=mod, a=A.sample_id[a], b=A.sample_id[b],
                                             a_split=A.split[a], b_split=A.split[b]))
    # Overlap means two different patches containing the same 4 x 4 pixel block (all 125 bands, no empty pixel) at any
    # position. Identical blocks of 16 pixels x 125 bands do not arise by chance, so a shared block means the two
    # tiles were cut from overlapping parts of the same image.
    blk = {}
    nonempty = np.where(valid_hs.mean((1, 2)) > 0)[0]
    for i in nonempty:
        a = A.hs[i]
        for y in range(0, 29):
            for x in range(0, 29):
                w = a[y: y + 4, x: x + 4]
                if (w == 0).all(-1).any():
                    continue
                blk.setdefault(hashlib.md5(np.ascontiguousarray(w).tobytes()).digest(), set()).add(i)
    pairs = {}
    for s_ in blk.values():
        if len(s_) > 1:
            s_ = sorted(s_)
            for a in s_:
                for b in s_:
                    if a < b:
                        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    for (a, b), n_ in pairs.items():
        dup_rows.append(dict(kind="shared_4x4_blocks", modality="HS", a=A.sample_id[a], b=A.sample_id[b],
                             a_split=A.split[a], b_split=A.split[b], n_shared_blocks=n_))
    # Near duplicates by cosine similarity of a coarse descriptor (8 x 8 spatial blocks of the cleaned bands), taking
    # the best match over the eight rotations and flips of the second patch.
    coarse = hs[:, :, :, CLEAN_IDX].reshape(N, 8, 4, 8, 4, len(CLEAN_IDX)).mean((2, 4))
    ne_idx = np.where(valid_hs.mean((1, 2)) > 0.99)[0]
    base = coarse[ne_idx].reshape(len(ne_idx), -1)
    base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)
    best = np.full((len(ne_idx), len(ne_idx)), -1.0, dtype=np.float32)
    for t in dihedral(coarse[ne_idx]):
        v = t.reshape(len(ne_idx), -1)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        best = np.maximum(best, base @ v.T)
    np.fill_diagonal(best, -1)
    nn_sim = best.max(1)
    pd.DataFrame(dict(sample_id=A.sample_id[ne_idx], split=A.split[ne_idx], nn_cosine=nn_sim,
                      nn_id=A.sample_id[ne_idx[best.argmax(1)]], nn_split=A.split[ne_idx[best.argmax(1)]])).to_csv(
        os.path.join(OUT, "nearest_neighbour.csv"), index=False)
    thr = 0.9995
    iu = np.argwhere(np.triu(best > thr, 1))
    for a, b in iu:
        dup_rows.append(dict(kind=f"near_cosine>{thr}", modality="HS", a=A.sample_id[ne_idx[a]],
                             b=A.sample_id[ne_idx[b]], a_split=A.split[ne_idx[a]], b_split=A.split[ne_idx[b]],
                             cosine=float(best[a, b])))
    dup = pd.DataFrame(dup_rows)
    dup.to_csv(os.path.join(OUT, "duplicates.csv"), index=False)
    if len(dup):
        S["duplicates"] = {
            k: {
                "train_train": int(((g.a_split == "train") & (g.b_split == "train")).sum()),
                "train_holdout": int((g.a_split != g.b_split).sum()),
                "holdout_holdout": int(((g.a_split == "val") & (g.b_split == "val")).sum()),
            }
            for k, g in dup.groupby("kind")
        }
    else:
        S["duplicates"] = {}
    S["nn_cosine_quantiles"] = {str(q): float(np.percentile(nn_sim, q)) for q in [1, 50, 99]}

    # Mosaic adjacency. If patches were cut edge to edge from one image, the right edge of one patch would continue
    # into the left edge (or the bottom into the top) of another as smoothly as neighbouring pixel columns inside a
    # patch. Every edge is compared with every other patch's opposite edge on 11 normalised bands.
    E = hs[ne_idx][..., CLEAN_IDX[::10]]
    E = E / (E.mean((1, 2, 3), keepdims=True) + 1e-9)
    inner = np.median(np.abs(E[:, :, 1:] - E[:, :, :-1]).mean((1, 3)))
    best_edge = []
    for name, ea, eb in [("right-left", E[:, :, -1], E[:, :, 0]), ("bottom-top", E[:, -1], E[:, 0])]:
        a = ea.reshape(len(ne_idx), -1)
        b = eb.reshape(len(ne_idx), -1)
        d = np.empty((len(a), len(b)), dtype=np.float32)
        for s0 in range(0, len(a), 64):
            d[s0: s0 + 64] = np.abs(a[s0: s0 + 64, None, :] - b[None, :, :]).mean(-1)
        np.fill_diagonal(d, np.inf)
        best_edge.append(d.min(1))
    best_edge = np.minimum(*best_edge)
    S["edge_inner_median_step"] = float(inner)
    S["edge_best_match_median"] = float(np.median(best_edge))
    S["edge_pairs_below_inner_step"] = int((best_edge < inner).sum())


def composition(A, hs, valid_hs, S):
    """Canopy cover of the training patches from pixel NDVI.

    NDVI uses 0-based HS bands 49 and 95 (646 and 830 nm). Patches with a mean NDVI of at least 0.5 that are not
    empty count as vegetated in the Health versus Rust analysis.
    """
    tr = A.split == "train"
    red, nir = hs[..., 49], hs[..., 95]
    ndvi = (nir - red) / (nir + red + 1e-6)
    comp = []
    for i in np.where(tr)[0]:
        v = valid_hs[i]
        if v.sum() == 0:
            comp.append(dict(sample_id=A.sample_id[i], label=A.label[i], empty=True))
            continue
        x = ndvi[i][v]
        comp.append(
            dict(
                sample_id=A.sample_id[i],
                label=A.label[i],
                empty=False,
                valid_frac=float(v.mean()),
                ndvi_mean=float(x.mean()),
                ndvi_median=float(np.median(x)),
                ndvi_p10=float(np.percentile(x, 10)),
                frac_ndvi_lt_0p3=float((x < 0.3).mean()),
                frac_ndvi_lt_0p5=float((x < 0.5).mean()),
            )
        )
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(OUT, "composition_ndvi.csv"), index=False)
    g = comp[~comp["empty"].astype(bool)].groupby("label")
    S["ndvi_by_class"] = {
        c: {
            "n_nonempty": int(len(g.get_group(c))),
            "ndvi_median_of_means": float(g.get_group(c).ndvi_mean.median()),
            "ndvi_q25": float(g.get_group(c).ndvi_mean.quantile(0.25)),
            "ndvi_q75": float(g.get_group(c).ndvi_mean.quantile(0.75)),
            "frac_patches_mean_ndvi_lt_0p5": float((g.get_group(c).ndvi_mean < 0.5).mean()),
            "median_frac_pixels_ndvi_lt_0p3": float(g.get_group(c).frac_ndvi_lt_0p3.median()),
        }
        for c in CLASSES
    }
    kw = stats.kruskal(*[g.get_group(c).ndvi_mean.values for c in CLASSES])
    S["ndvi_kruskal_H"] = float(kw.statistic)
    S["ndvi_kruskal_p"] = float(kw.pvalue)


def main():
    os.makedirs(OUT, exist_ok=True)
    A = Arrays()
    hs = np.asarray(A.hs, dtype=np.float32)
    ms = np.asarray(A.ms, dtype=np.float32)
    rgb = np.asarray(A.rgb, dtype=np.float32)
    valid_hs = ~(hs == 0).all(-1)
    full = valid_hs.mean((1, 2)) > 0.999  # fully valid patches
    S = {}
    tab = sample_table()
    inv = inventory(A, tab, hs, ms, rgb, S)
    georeferencing(tab, inv, S)
    provenance(A, hs, ms, rgb, full, S)
    band_noise(A, hs, full, S)
    blue_index_noise(A, hs, full)
    duplicates(A, hs, valid_hs, S)
    composition(A, hs, valid_hs, S)
    with open(os.path.join(OUT, "audit_summary.json"), "w") as f:
        json.dump(S, f, indent=2, default=float)
    prov = pd.DataFrame(S["provenance"])
    print(prov[["view", "nominal", "best_hs_wavelength_nm", "median_per_image_r2", "pooled_r2"]].round(3).to_string())
    print({k: S[k] for k in ["n_train", "n_holdout", "class_counts", "duplicates"]})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
