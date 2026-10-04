"""Denoising, baselines, peak detection and bond assignment."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

from .model import UNet1D
from .preprocess import GRID

STEP = GRID[1] - GRID[0]
HERE = os.path.dirname(__file__)


# ---------------------------------------------------------------- denoisers
def load_model(path, in_ch):
    m = UNet1D(in_ch)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def unet(model, X):
    """X: (N, C, L) normalised. Averages over the sign / branch-order symmetries."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(dev)
    res = []
    for b in range(0, len(X), 256):
        x = torch.as_tensor(np.asarray(X[b:b + 256], np.float32), device=dev)
        outs = [model(x), -model(-x)]
        if x.shape[1] == 2:
            outs += [model(x.flip(1)).flip(1), -model(-x.flip(1)).flip(1)]
        res.append(torch.stack(outs).mean(0).cpu().numpy())
    return np.concatenate(res)


def savgol(X, window=11, order=3):
    return savgol_filter(X, window, order, axis=-1)


def gauss(X, sigma=2.5):
    return gaussian_filter1d(X, sigma, axis=-1)


def branch_average(X):
    """Classical two-branch estimate: smooth, align sign of -V branch to +V, average."""
    S = gauss(X)
    s = np.sign(np.sum(S[:, 0] * S[:, 1], axis=-1))[:, None]
    avg = (S[:, 0] + s * S[:, 1]) / 2
    return np.stack([avg, s * avg], axis=1)


# ---------------------------------------------------------------- peaks
def detect(y, prominence, min_sep_cm=40.0):
    """Signed peaks (maxima of y and of -y). Returns list of (wavenumber, height)."""
    out = []
    for sgn in (1, -1):
        idx, _ = find_peaks(sgn * y, prominence=prominence, distance=max(1, int(min_sep_cm / STEP)))
        out += [(GRID[i], y[i]) for i in idx]
    return sorted(out)


def match(a, b, tol=24.0):
    """Greedy one-to-one matching of peak positions; returns list of (i, j)."""
    pairs, used = [], set()
    for i, (wa, _) in enumerate(a):
        cands = [(abs(wa - wb), j) for j, (wb, _) in enumerate(b) if j not in used and abs(wa - wb) <= tol]
        if cands:
            _, j = min(cands)
            used.add(j)
            pairs.append((i, j))
    return pairs


def f1(pred, truth, tol=24.0):
    m = len(match(pred, truth, tol))
    p = m / len(pred) if pred else 1.0
    r = m / len(truth) if truth else 1.0
    return 2 * p * r / (p + r + 1e-12), p, r


def classify_peaks(Y, thr, single_thr, tentative_thr=None, tol=32.0):
    """Peaks in the two denoised branches (on the common midpoint axis).

    confirmed   -- found in both branches within `tol` cm-1 (prominence >= thr in each)
    single-half -- found in one branch only, with prominence >= single_thr
    tentative   -- found in one branch only, with tentative_thr <= prominence < single_thr
    Returns a list of dicts sorted by wavenumber.
    """
    a, b = detect(Y[0], thr), detect(Y[1], thr)
    pairs = match(a, b, tol)
    out, used_a, used_b = [], {i for i, _ in pairs}, {j for _, j in pairs}
    for i, j in pairs:
        (wa, ha), (wb, hb) = a[i], b[j]
        out.append(dict(wavenumber=(wa + wb) / 2, height_pos=ha, height_neg=hb,
                        strength=float(np.sqrt(abs(ha * hb))), confidence="confirmed",
                        symmetry="odd" if ha * hb < 0 else "even"))
    for peaks, used, br in ((a, used_a, "+V"), (b, used_b, "-V")):
        for k, (w, h) in enumerate(peaks):
            if k in used:
                continue
            y = Y[0] if br == "+V" else Y[1]
            if detect_one(y, w, single_thr):
                tier = f"single-half ({br})"
            elif tentative_thr is not None and detect_one(y, w, tentative_thr):
                tier = f"tentative ({br})"
            else:
                continue
            out.append(dict(wavenumber=w, height_pos=h if br == "+V" else np.nan,
                            height_neg=h if br == "-V" else np.nan, strength=float(abs(h)),
                            confidence=tier, symmetry=""))
    return sorted(out, key=lambda d: d["wavenumber"])


def detect_one(y, w, prominence):
    """True if a peak at wavenumber w survives the stricter prominence."""
    return any(abs(w2 - w) < 1e-6 for w2, _ in detect(y, prominence))


# ---------------------------------------------------------------- bonds
def bond_table():
    return pd.read_csv(os.path.join(HERE, "ftir_bands.csv"))


def candidates(wavenumber, uncertainty=0.0, table=None, tol=20.0):
    """All FTIR rows whose band (widened by tol + the position uncertainty) contains the peak."""
    t = bond_table() if table is None else table
    u = tol + (0.0 if not np.isfinite(uncertainty) else uncertainty)
    hit = t[(t.wn_low - u <= wavenumber) & (wavenumber <= t.wn_high + u)].copy()
    hit["dist"] = np.maximum(0, np.maximum(hit.wn_low - wavenumber, wavenumber - hit.wn_high))
    return hit


def assign(wavenumber, uncertainty=0.0, table=None, tol=20.0, top=3):
    """Top candidate bonds: bands containing the peak first, then nearest, narrowest first."""
    hit = candidates(wavenumber, uncertainty, table, tol)
    hit["width"] = hit.wn_high - hit.wn_low
    hit = hit.sort_values(["dist", "width"]).head(top)
    return "; ".join(f"{r.vibration} [{r.group}] {r.wn_low}-{r.wn_high}" for r in hit.itertuples())
