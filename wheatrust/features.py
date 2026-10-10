"""Engineered features for the RGB, MS, and HS views.

Every function here is a fixed per-sample transform with no fitted state, so the features of all 900 patches are
computed once (scripts/training/02_build_features.py) without any leakage between folds. Everything that is fitted,
such as standardisation, PCA, and band selection, is fitted inside outer training folds by wheatrust.runner.

Conventions
* Statistics use valid pixels only. MS and HS pixels are invalid when they are zero in every band; RGB has its own
  rule (rgb_valid_mask). A completely empty patch receives zeros and valid_frac = 0.
* MS and HS digital numbers are divided by 10 000 to give reflectance, and RGB values are scaled to [0, 1]. The RGB
  channels were contrast stretched per image by the data provider, so RGB indices describe colour within a patch,
  not absolute reflectance.
* Feature names are <view>__<family>__<name>. The families drive the leave-one-family-out ablation (E5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage.feature import graycomatrix, graycoprops

from wheatrust.io_data import CLEAN_IDX, HS_WL

EPS = 1e-6
STAT_NAMES = ["mean", "std", "p10", "p50", "p90"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sdiv(a, b):
    """Division that replaces denominators within 1e-6 of zero by +-1e-6, keeping their sign."""
    b = np.where(np.abs(b) < EPS, np.where(b < 0, -EPS, EPS), b)
    return a / b


def masked_stats(img, mask):
    """Mean, standard deviation, and 10th, 50th, and 90th percentiles over the valid pixels of each patch.

    img and mask are (N, H, W); the result is (N, 5). Values are clipped to +-1e4 first so that ratio indices with
    near-zero denominators cannot dominate the moments.
    """
    x = np.where(mask, img, np.nan).reshape(len(img), -1).astype(np.float64)
    x = np.clip(x, -1e4, 1e4)
    out = np.zeros((len(img), 5))
    has = mask.reshape(len(img), -1).any(1)
    if has.any():
        xs = x[has]
        out[has, 0] = np.nanmean(xs, 1)
        out[has, 1] = np.nanstd(xs, 1)
        out[has, 2:] = np.nanpercentile(xs, [10, 50, 90], axis=1).T
    return out


def add_stats(cols, prefix, img, mask):
    st = masked_stats(img, mask)
    for j, s in enumerate(STAT_NAMES):
        cols[f"{prefix}_{s}"] = st[:, j]


def glcm_features(gray, mask, levels=32, distances=(1, 2), prefix=""):
    """GLCM texture (Haralick et al. 1973) averaged over four directions, at distances of one and two pixels.

    gray is (N, H, W) and is quantised to 32 levels per patch between the 1st and 99th percentiles of its valid
    pixels. Invalid pixels get an extra level, and the rows and columns of that level are dropped from the
    co-occurrence matrix, so pairs involving invalid pixels never count. Patches with fewer than 16 valid pixels
    keep zeros.
    """
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    out = {f"{prefix}{p}_d{d}": np.zeros(len(gray)) for p in props for d in distances}
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for i in range(len(gray)):
        m = mask[i]
        if m.sum() < 16:
            continue
        g = gray[i]
        lo, hi = np.percentile(g[m], [1, 99])
        q = np.clip(((g - lo) / (hi - lo + EPS) * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)
        q[~m] = levels
        P = graycomatrix(q, distances=list(distances), angles=angles, levels=levels + 1, symmetric=True, normed=False)
        P = P[:levels, :levels].astype(np.float64)
        s = P.sum(axis=(0, 1), keepdims=True)
        P = P / np.where(s == 0, 1, s)
        for p in props:
            v = graycoprops(P, p)  # (n_dist, n_angle)
            for di, d in enumerate(distances):
                out[f"{prefix}{p}_d{d}"][i] = np.nan_to_num(v[di].mean())
    return out


# ---------------------------------------------------------------------------
# RGB view
# ---------------------------------------------------------------------------
def rgb_valid_mask(rgb, max_clip_frac=0.05):
    """Valid-pixel mask of the RGB view.

    The provider's per-image stretch sends about 2 percent of the pixels of every channel to 0, so scattered black
    pixels are real data. Black pixels count as no-data only when they cover more than max_clip_frac of the patch,
    which happens where a patch reaches the edge of the mosaic.
    """
    black = (rgb == 0).all(-1)
    frac = black.mean((1, 2))
    return np.where((frac > max_clip_frac)[:, None, None], ~black, True)


def rgb_features(rgb, mask=None, view="rgb"):
    """Channel statistics, chromatic coordinates, colour indices, HSV, and GLCM texture of rgb (N, H, W, 3), with
    values in [0, 1]."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if mask is None:
        mask = ~(rgb == 0).all(-1)
    cols = {}
    cols[f"{view}__meta__valid_frac"] = mask.mean((1, 2))
    for c, x in zip("RGB", [R, G, B]):
        add_stats(cols, f"{view}__band__{c}", x, mask)
    tot = R + G + B
    r, g, b = sdiv(R, tot), sdiv(G, tot), sdiv(B, tot)
    for c, x in zip("rgb", [r, g, b]):
        add_stats(cols, f"{view}__chroma__{c}", x, mask)
    exg = 2 * g - r - b  # Woebbecke et al. 1995
    exr = 1.4 * r - g  # Meyer et al. 1998
    idx = {
        "ExG": exg,
        "ExR": exr,
        "ExGR": exg - exr,  # Meyer and Neto 2008
        "VARI": np.clip(sdiv(G - R, G + R - B), -5, 5),  # Gitelson et al. 2002
        "GLI": sdiv(2 * G - R - B, 2 * G + R + B),  # Louhaichi et al. 2001
        "NGRDI": sdiv(G - R, G + R),  # Tucker 1979
        "CIVE": 0.441 * r - 0.811 * g + 0.385 * b + 18.78745,  # Kataoka et al. 2003
        "MGRVI": sdiv(G**2 - R**2, G**2 + R**2),  # Bendig et al. 2015
        "RGBVI": sdiv(G**2 - B * R, G**2 + B * R),  # Bendig et al. 2015
        "TGI": -0.5 * (190 * (R - G) - 120 * (R - B)),  # Hunt et al. 2013, broadband form
    }
    for k, x in idx.items():
        add_stats(cols, f"{view}__index__{k}", x, mask)
    # HSV, with the hue also summarised as a circular mean because it wraps around at red
    mx = np.max(rgb, -1)
    mn = np.min(rgb, -1)
    delta = mx - mn
    V = mx
    S = sdiv(delta, mx) * (mx > 0)
    H = np.zeros_like(mx)
    dz = delta > EPS
    rc = sdiv(mx - R, delta)
    gc = sdiv(mx - G, delta)
    bc = sdiv(mx - B, delta)
    H = np.where(R == mx, bc - gc, np.where(G == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    H = np.where(dz, (H / 6.0) % 1.0, 0.0)
    for k, x in [("H", H), ("S", S), ("V", V)]:
        add_stats(cols, f"{view}__hsv__{k}", x, mask)
    ang = 2 * np.pi * H
    cs = masked_stats(np.cos(ang), mask)[:, 0]
    sn = masked_stats(np.sin(ang), mask)[:, 0]
    cols[f"{view}__hsv__H_circmean_cos"] = cs
    cols[f"{view}__hsv__H_circmean_sin"] = sn
    cols[f"{view}__hsv__H_resultant"] = np.sqrt(cs**2 + sn**2)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    for k, v in glcm_features(gray, mask, prefix=f"{view}__glcm__").items():
        cols[k] = v
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# MS view
# ---------------------------------------------------------------------------
def ms_features(ms, mask=None, view="ms"):
    """Band statistics, vegetation indices, and GLCM texture of ms (N, H, W, 5), reflectance at 482, 550, 650, 738,
    and 834 nm. Indices defined for other wavelengths use the nearest MS band, as noted beside each one."""
    Bl, G, R, RE, NIR = [ms[..., i] for i in range(5)]
    if mask is None:
        mask = ~(ms == 0).all(-1)
    cols = {f"{view}__meta__valid_frac": mask.mean((1, 2))}
    for name, x in zip(["B482", "G550", "R650", "RE738", "NIR834"], [Bl, G, R, RE, NIR]):
        add_stats(cols, f"{view}__band__{name}", x, mask)
    idx = {
        "NDVI": sdiv(NIR - R, NIR + R),  # Rouse et al. 1974
        "GNDVI": sdiv(NIR - G, NIR + G),  # Gitelson et al. 1996
        "BNDVI": sdiv(NIR - Bl, NIR + Bl),
        "RENDVI": sdiv(NIR - RE, NIR + RE),  # red-edge NDVI, Gitelson and Merzlyak 1994 adapted to 738/834 nm
        "NDREr": sdiv(RE - R, RE + R),
        "NDWI": sdiv(G - NIR, G + NIR),  # McFeeters 1996
        "OSAVI": 1.16 * sdiv(NIR - R, NIR + R + 0.16),  # Rondeaux et al. 1996
        "SAVI": 1.5 * sdiv(NIR - R, NIR + R + 0.5),  # Huete 1988
        "EVI": np.clip(2.5 * sdiv(NIR - R, NIR + 6 * R - 7.5 * Bl + 1), -5, 5),  # Huete et al. 2002
        "MCARI": ((RE - R) - 0.2 * (RE - G)) * sdiv(RE, R),  # Daughtry et al. 2000, 738 for 700 and 650 for 670 nm
        "CIgreen": sdiv(NIR, G) - 1,  # Gitelson et al. 2003
        "CIrededge": sdiv(NIR, RE) - 1,  # Gitelson et al. 2003
        "MSR": sdiv(sdiv(NIR, R) - 1, np.sqrt(np.clip(sdiv(NIR, R) + 1, EPS, None))),  # Chen 1996
        "SR": sdiv(NIR, R),  # Jordan 1969
        "RG": sdiv(R, G),
    }
    for k, x in idx.items():
        add_stats(cols, f"{view}__index__{k}", np.clip(x, -50, 50), mask)
    for k, v in glcm_features(NIR, mask, prefix=f"{view}__glcm__NIR_").items():
        cols[k] = v
    for k, v in glcm_features(idx["NDVI"], mask, prefix=f"{view}__glcm__NDVI_").items():
        cols[k] = v
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# HS view
# ---------------------------------------------------------------------------
def band(wl, wavelengths):
    """Index of the band centre closest to wl among wavelengths."""
    return int(np.argmin(np.abs(wavelengths - wl)))


def hs_features(hs, wavelengths, mask=None, view="hs", band_stats=("mean", "std")):
    """Band statistics, narrow-band indices, spectral shape, and red-edge features of hs (N, H, W, B), reflectance
    with band centres wavelengths.

    Indices take the HS band nearest each published wavelength and are skipped when a wavelength lies more than 2 nm
    outside the cube. Returns the feature table and the (N, B) valid-pixel mean spectrum of every patch.
    """
    N = len(hs)
    if mask is None:
        mask = ~(hs == 0).all(-1)
    cols = {f"{view}__meta__valid_frac": mask.mean((1, 2))}
    has = mask.reshape(N, -1).any(1)
    m3 = mask[..., None]
    cnt = np.maximum(mask.sum((1, 2)), 1)[:, None]
    mean_spec = (hs * m3).sum((1, 2)) / cnt  # N, B
    if "mean" in band_stats:
        for j, w in enumerate(wavelengths):
            cols[f"{view}__band__mean_{int(w)}"] = mean_spec[:, j]
    if "std" in band_stats:
        var = ((hs - mean_spec[:, None, None, :]) ** 2 * m3).sum((1, 2)) / cnt
        sd = np.sqrt(var)
        for j, w in enumerate(wavelengths):
            cols[f"{view}__band__std_{int(w)}"] = sd[:, j]

    def Rw(wl):
        return hs[..., band(wl, wavelengths)]

    lo, hi = wavelengths.min(), wavelengths.max()

    def ok(*wls):
        return all(lo - 2 <= w <= hi + 2 for w in wls)

    # pigment and structural indices, computed per pixel from the nearest bands
    idx = {}
    if ok(800, 670):
        idx["NDVI"] = sdiv(Rw(800) - Rw(670), Rw(800) + Rw(670))  # Rouse et al. 1974
        idx["OSAVI"] = 1.16 * sdiv(Rw(800) - Rw(670), Rw(800) + Rw(670) + 0.16)  # Rondeaux et al. 1996
        idx["PSSRa"] = sdiv(Rw(800), Rw(680))  # Blackburn 1998
    if ok(750, 705):
        idx["RENDVI"] = sdiv(Rw(750) - Rw(705), Rw(750) + Rw(705))  # Gitelson and Merzlyak 1994
    if ok(531, 570):
        idx["PRI"] = sdiv(Rw(531) - Rw(570), Rw(531) + Rw(570))  # Penuelas, Filella and Gamon 1995 (531 and 570 nm)
    if ok(550, 700):
        idx["ARI"] = sdiv(1.0, Rw(550)) - sdiv(1.0, Rw(700))  # Gitelson et al. 2001
    if ok(550, 700, 800):
        idx["ARI2"] = Rw(800) * (sdiv(1.0, Rw(550)) - sdiv(1.0, Rw(700)))  # Gitelson et al. 2001
        idx["CIrededge"] = sdiv(Rw(800), Rw(720)) - 1  # Gitelson et al. 2003
        idx["CIgreen"] = sdiv(Rw(800), Rw(550)) - 1  # Gitelson et al. 2003
    if ok(510, 550):
        idx["CRI1"] = sdiv(1.0, Rw(510)) - sdiv(1.0, Rw(550))  # Gitelson et al. 2002
    if ok(500, 680, 750):
        idx["PSRI"] = sdiv(Rw(680) - Rw(500), Rw(750))  # Merzlyak et al. 1999
    if ok(700, 670, 550):
        mc = ((Rw(700) - Rw(670)) - 0.2 * (Rw(700) - Rw(550))) * sdiv(Rw(700), Rw(670))
        idx["MCARI"] = mc  # Daughtry et al. 2000
        tc = 3 * ((Rw(700) - Rw(670)) - 0.2 * (Rw(700) - Rw(550)) * sdiv(Rw(700), Rw(670)))
        idx["TCARI"] = tc  # Haboudane et al. 2002
        if "OSAVI" in idx:
            idx["TCARI_OSAVI"] = sdiv(tc, idx["OSAVI"])  # Haboudane et al. 2002
    if ok(754, 709, 681):
        idx["MTCI"] = sdiv(Rw(754) - Rw(709), Rw(709) - Rw(681))  # Dash and Curran 2004
    if ok(750, 550):
        idx["GM1"] = sdiv(Rw(750), Rw(550))  # Gitelson and Merzlyak 1997
    if ok(750, 700):
        idx["GM2"] = sdiv(Rw(750), Rw(700))  # Gitelson and Merzlyak 1997
    if ok(740, 720):
        idx["VOG1"] = sdiv(Rw(740), Rw(720))  # Vogelmann et al. 1993
    if ok(670, 700, 740, 780):
        rre = 0.5 * (Rw(670) + Rw(780))
        idx["REP_linear"] = 700 + 40 * sdiv(rre - Rw(700), Rw(740) - Rw(700))  # Guyot and Baret 1988
    for k, x in idx.items():
        add_stats(cols, f"{view}__index__{k}", np.clip(x, -1e3, 1e3), mask)

    # Shape of the patch mean spectrum after Savitzky-Golay smoothing (7-band window, second-order polynomial). The
    # features are the first-derivative extremes and their positions on the blue, yellow, and red edges, the green
    # peak, the red well, the NIR plateau, the continuum-removed red well, spectral entropy, and band ratios.
    sh = {}
    wl = wavelengths
    step = float(np.median(np.diff(wl)))
    win = 7 if len(wl) >= 7 else (len(wl) // 2) * 2 - 1
    sm = savgol_filter(mean_spec, win, 2, axis=1)
    d1 = savgol_filter(mean_spec, win, 2, deriv=1, delta=step, axis=1)

    def rng_(a, b):
        return np.where((wl >= a) & (wl <= b))[0]

    for nm, (a, b) in {"blue": (490, 530), "yellow": (550, 582), "red": (680, 760)}.items():
        r_ = rng_(a, b)
        if len(r_) >= 3:
            dd = d1[:, r_]
            if nm == "yellow":
                sh[f"D_{nm}_min"] = dd.min(1)
                sh[f"Pos_{nm}_min"] = wl[r_][dd.argmin(1)]
            else:
                sh[f"D_{nm}_max"] = dd.max(1)
                sh[f"Pos_{nm}_max"] = wl[r_][dd.argmax(1)]
            sh[f"SD_{nm}"] = dd.sum(1) * step
    if "D_red_max" in sh and "SD_blue" in sh:
        sh["SDr_SDb_ratio"] = sdiv(sh["SD_red"], sh["SD_blue"])
        sh["SDr_SDy_norm"] = sdiv(sh["SD_red"] - sh["SD_yellow"], sh["SD_red"] + sh["SD_yellow"])
    g = rng_(510, 560)
    if len(g):
        sh["green_peak_R"] = sm[:, g].max(1)
        sh["green_peak_pos"] = wl[g][sm[:, g].argmax(1)]
    rw = rng_(640, 690)
    if len(rw):
        sh["red_well_R"] = sm[:, rw].min(1)
        sh["red_well_pos"] = wl[rw][sm[:, rw].argmin(1)]
    nir = rng_(780, 890)
    if len(nir):
        sh["nir_plateau_mean"] = sm[:, nir].mean(1)
        sh["nir_plateau_slope"] = np.polyfit(wl[nir], sm[:, nir].T, 1)[0]
    # continuum removal of the red chlorophyll well, against a straight line from 550 to 760 nm
    c0, c1 = band(550, wl), band(760, wl)
    if wl[c0] >= lo and wl[c1] <= hi and c1 > c0 + 2:
        seg = sm[:, c0 : c1 + 1]
        line = seg[:, :1] + (seg[:, -1:] - seg[:, :1]) * (np.arange(seg.shape[1]) / (seg.shape[1] - 1))
        cr = sdiv(seg, line)
        sh["CR_depth_red"] = 1 - cr.min(1)
        sh["CR_area_red"] = (1 - cr).sum(1) * step
        sh["CR_pos_red"] = wl[c0 : c1 + 1][cr.argmin(1)]
    p = np.clip(sm, EPS, None)
    p = p / p.sum(1, keepdims=True)
    sh["spectral_entropy"] = -(p * np.log(p)).sum(1) / np.log(p.shape[1])
    for nm_, (a, b) in {"NIR_R": (800, 670), "R_G": (670, 550), "G_B": (550, 495), "RE_R": (720, 670),
                        "NIR_RE": (800, 720), "R750_R550": (750, 550), "R700_R670": (700, 670)}.items():
        if ok(a, b):
            sh[f"ratio_{nm_}"] = sdiv(sm[:, band(a, wl)], sm[:, band(b, wl)])
    sh["mean_reflectance"] = sm.mean(1)
    for k, v in sh.items():
        v = np.where(has, v, 0.0)
        fam = "deriv" if k.startswith(("D_", "Pos_", "SD", "red_edge")) else "shape"
        cols[f"{view}__{fam}__{k}"] = np.clip(np.nan_to_num(v), -1e4, 1e4)

    # red-edge position and slope per pixel, summarised over the patch, which describes how uneven the red edge is
    r_ = rng_(680, 760)
    if len(r_) >= 5:
        rep = np.zeros(hs.shape[:3], dtype=np.float32)
        slope = np.zeros(hs.shape[:3], dtype=np.float32)
        for s0 in range(0, N, 64):
            dpx = savgol_filter(hs[s0 : s0 + 64], win, 2, deriv=1, delta=step, axis=-1)[..., r_]
            rep[s0 : s0 + 64] = wl[r_][dpx.argmax(-1)]
            slope[s0 : s0 + 64] = dpx.max(-1)
        add_stats(cols, f"{view}__deriv__REP_maxderiv_px", rep, mask)
        add_stats(cols, f"{view}__deriv__RE_slope_px", slope, mask)
    for k in list(cols):
        cols[k] = np.where(has, cols[k], 0.0) if not k.endswith("valid_frac") else cols[k]
    return pd.DataFrame(cols), mean_spec


# ---------------------------------------------------------------------------
# cross-modal RGB x MS features
# ---------------------------------------------------------------------------
def cross_features(rgb, ms, mask, view="cross"):
    """Features that combine the co-registered RGB (stretched) and MS (reflectance) views, both (N, 64, 64, .).

    MS NDVI above 0.5 marks vegetated pixels. The features are the vegetated fraction, the RGB colour of vegetated
    and soil pixels, and how well RGB greenness (ExG) agrees with MS NDVI pixel by pixel.
    """
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    NIR, Rm = ms[..., 4], ms[..., 2]
    ndvi = sdiv(NIR - Rm, NIR + Rm)
    tot = R + G + B
    r, g, b = sdiv(R, tot), sdiv(G, tot), sdiv(B, tot)
    exg = 2 * g - r - b
    veg = mask & (ndvi > 0.5)
    soil = mask & (ndvi <= 0.5)
    cols = {f"{view}__mask__veg_frac": sdiv(veg.sum((1, 2)), np.maximum(mask.sum((1, 2)), 1))}
    N = len(rgb)
    corr = np.zeros(N)
    for i in range(N):
        m = mask[i]
        if m.sum() > 10:
            a, c = exg[i][m], ndvi[i][m]
            if a.std() > EPS and c.std() > EPS:
                corr[i] = np.corrcoef(a, c)[0, 1]
    cols[f"{view}__agree__corr_ExG_NDVI"] = corr
    for nm, x in [("r", r), ("g", g), ("b", b), ("ExG", exg), ("VARI", np.clip(sdiv(G - R, G + R - B), -5, 5))]:
        cols[f"{view}__vegcolor__{nm}_veg_mean"] = masked_stats(x, veg)[:, 0]
        cols[f"{view}__vegcolor__{nm}_veg_std"] = masked_stats(x, veg)[:, 1]
        cols[f"{view}__vegcolor__{nm}_soil_mean"] = masked_stats(x, soil)[:, 0]
    green_px = mask & (exg > 0.05)
    cols[f"{view}__agree__NDVI_on_RGBgreen_mean"] = masked_stats(ndvi, green_px)[:, 0]
    cols[f"{view}__agree__frac_green_and_veg"] = sdiv((green_px & veg).sum((1, 2)), np.maximum(mask.sum((1, 2)), 1))
    return pd.DataFrame(cols)


def stretch_8bit(img, mask, lo=2.0, hi=98.0):
    """Per-image, per-channel linear stretch between the lo and hi percentiles of the valid pixels, clipped to
    [0, 1] and quantised to 8 bits, the processing that the audit found in the provided RGB PNGs.

    img is (N, H, W, C) reflectance and mask (N, H, W). The stretch control applies it to the three HS bands behind
    the RGB view (scripts/training/03_stretch_control.py), which separates the stretch and quantisation from the
    other differences of the PNG.
    """
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        for c in range(img.shape[-1]):
            x = img[i, ..., c]
            a, b = np.percentile(x[m], [lo, hi])
            y = np.clip((x - a) / (b - a), 0.0, 1.0) if b > a else np.zeros_like(x)
            out[i, ..., c] = np.round(y * 255.0) / 255.0
        out[i][~m] = 0.0
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def all_feature_tables(A):
    """Every feature table (indexed by sample_id) and the patch mean spectra, from an Arrays-like object A.

    rgb, ms, hsc (cleaned HS), hsf (full HS), and cross are the views of the paper. rgbhs and mshs are the
    single-source control, the RGB and MS feature sets computed directly from the HS bands at the RGB and MS
    positions, which removes the stretch and the resampling while keeping the band placement.
    """
    hs = np.asarray(A.hs, dtype=np.float32) / 10000.0
    ms = np.asarray(A.ms, dtype=np.float32) / 10000.0
    rgb = np.asarray(A.rgb, dtype=np.float32) / 255.0
    ids = A.sample_id
    mask_hs = ~(hs == 0).all(-1)
    mask_ms = ~(ms == 0).all(-1)
    mask_rgb = rgb_valid_mask(rgb)
    T = {}
    T["rgb"] = rgb_features(rgb, mask_rgb, "rgb")
    T["ms"] = ms_features(ms, mask_ms, "ms")
    hs_clean = hs[..., CLEAN_IDX]
    T["hsc"], spec_c = hs_features(hs_clean, HS_WL[CLEAN_IDX], mask_hs, "hsc")
    T["hsf"], spec_f = hs_features(hs, HS_WL, mask_hs, "hsf")
    T["cross"] = cross_features(rgb, ms, mask_ms, "cross")
    b_rgb = [band(w, HS_WL) for w in (650, 550, 482)]
    b_ms = [band(w, HS_WL) for w in (482, 550, 650, 738, 834)]
    T["rgbhs"] = rgb_features(hs[..., b_rgb], mask_hs, "rgbhs")
    T["mshs"] = ms_features(hs[..., b_ms], mask_hs, "mshs")
    for k in T:
        T[k].index = ids
        T[k] = T[k].astype(np.float32)
    spectra = {
        "hsc": pd.DataFrame(spec_c, index=ids, columns=[f"{int(w)}" for w in HS_WL[CLEAN_IDX]]),
        "hsf": pd.DataFrame(spec_f, index=ids, columns=[f"{int(w)}" for w in HS_WL]),
    }
    return T, spectra
