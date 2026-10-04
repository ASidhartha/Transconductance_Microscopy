"""Align the +V and mirrored -V branches of a sweep.

The two branches share features, but not at the same wavenumber: features in the
+V branch sit at T(w) = k*w + c for a feature at w in the -V branch, with k and c
varying from spectrum to spectrum (typically k ~ 0.9-1.1 in the corrected data). Neither polarity's axis
is known to be correct, so both branches are resampled onto the *midpoint* axis
u = (w + T(w)) / 2, and the half-difference |T(w) - w| / 2 is kept as the position
uncertainty caused by the polarity mismatch.

An alignment is only used when its correlation beats the chance level obtained by
running the same search on randomly paired (unrelated) branches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import GRID

K_RANGE = np.linspace(0.85, 1.15, 31)  # new data: fitted k is 0.87-1.10 (median ~1.0)
C_RANGE = np.arange(-400, 401, 8.0)
MIN_OVERLAP = 0.6  # fraction of the grid that must overlap after warping


@dataclass
class Alignment:
    k: float = 1.0
    c: float = 0.0
    r: float = 0.0          # signed correlation at the optimum
    r0: float = 0.0         # signed correlation with no warping
    accepted: bool = False

    def T(self, w):
        """-V wavenumber -> matching +V wavenumber."""
        return self.k * w + self.c

    def mid(self, w):
        return (w + self.T(w)) / 2

    def uncertainty(self, u):
        """Half-distance between the two branch positions of a feature at midpoint u."""
        w = (2 * u - self.c) / (1 + self.k)
        return np.abs(self.T(w) - w) / 2


def _search(p, m):
    """Best (k, c, r) such that m(w) ~ r * p(k w + c)."""
    best = (1.0, 0.0, 0.0)
    for k in K_RANGE:
        for c in C_RANGE:
            pw = np.interp(k * GRID + c, GRID, p, left=np.nan, right=np.nan)
            ok = ~np.isnan(pw)
            if ok.mean() < MIN_OVERLAP:
                continue
            r = np.corrcoef(pw[ok], m[ok])[0, 1]
            if abs(r) > abs(best[2]):
                best = (k, c, r)
    return best


def null_threshold(branches, n=150, q=95, seed=0):
    """q-th percentile of the best |r| found when aligning unrelated branches."""
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        i, j = rng.integers(0, len(branches), 2)
        if i != j:
            vals.append(abs(_search(branches[i][0], branches[j][1])[2]))
    return float(np.percentile(vals, q)), vals


def align(p, m, threshold):
    k, c, r = _search(p, m)
    r0 = float(np.corrcoef(p, m)[0, 1])
    ok = abs(r) > threshold
    return Alignment(k, c, r, r0, ok) if ok else Alignment(1.0, 0.0, r, r0, False)


def to_midpoint(p, m, a: Alignment):
    """Resample both branches onto GRID interpreted as the midpoint axis u.

    Returns (2, L) array; points outside a branch's measured range are set to 0 and a
    (2, L) validity mask is returned alongside.
    """
    w_neg = (2 * GRID - a.c) / (1 + a.k)          # -V coordinate of midpoint u
    w_pos = a.T(w_neg)                            # +V coordinate of midpoint u
    pos = np.interp(w_pos, GRID, p, left=np.nan, right=np.nan)
    neg = np.interp(w_neg, GRID, m, left=np.nan, right=np.nan)
    out = np.stack([pos, neg])
    mask = ~np.isnan(out)
    return np.nan_to_num(out), mask
