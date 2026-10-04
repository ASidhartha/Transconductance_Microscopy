"""Turn a raw bias sweep into two aligned, normalised branches on a uniform grid.

IETS features appear at |eV| = h*nu for both bias polarities, so the +V branch and
the mirrored -V branch are two measurements of the same vibrational spectrum.
We found peak *positions* reproduce across the two branches while their *signs* do
not consistently follow the ideal odd symmetry, so the branches are kept as two
separate channels rather than being antisymmetrised.
"""
from __future__ import annotations

import numpy as np

# 200..3648 cm-1 in 8 cm-1 steps -> 432 points (divisible by 16 for the U-Net).
# Below ~200 cm-1 is the zero-bias region; above ~3650 the sweep-edge artefact dominates.
GRID = 200.0 + 8.0 * np.arange(432)


def detrend(y: np.ndarray, x: np.ndarray = GRID, deg: int = 3) -> np.ndarray:
    """Remove a slowly varying background with a robust (iteratively reweighted) polynomial."""
    t = (x - x.mean()) / (x.max() - x.min())
    w = np.ones_like(y)
    for _ in range(5):
        c = np.polyfit(t, y, deg, w=w)
        r = y - np.polyval(c, t)
        s = 1.4826 * np.median(np.abs(r)) + 1e-30
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2.5 * s))  # Huber weights
    return y - np.polyval(c, t)


def robust_scale(y: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(y - np.median(y))) + 1e-30)


def branches(wavenumber: np.ndarray, d2: np.ndarray):
    """Return (pos, neg) branches resampled onto GRID and detrended, in original units."""
    w, d = np.asarray(wavenumber), np.asarray(d2)
    pos = np.interp(GRID, w[w > 0], d[w > 0])
    wn, dn = -w[w < 0][::-1], d[w < 0][::-1]
    neg = np.interp(GRID, wn, dn)
    return detrend(pos), detrend(neg)


def prepare(wavenumber, d2):
    """Two-channel normalised input (2, L) and the scale needed to restore units."""
    p, n = branches(wavenumber, d2)
    scale = robust_scale(np.concatenate([p, n]))
    return np.stack([p, n]) / scale, scale
