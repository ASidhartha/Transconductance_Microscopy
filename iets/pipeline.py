"""Shared preparation: load -> split branches -> align -> resample onto midpoint axis."""
from __future__ import annotations

import json
import os

import numpy as np

from .align import C_RANGE, K_RANGE, Alignment, align, null_threshold, to_midpoint
from .data import load_all
from .preprocess import GRID, branches, robust_scale

CACHE = "outputs/alignment.json"
NULL_Q = 95  # acceptance: beat the 95th percentile of randomly paired branches


def _accept(a: Alignment, thr: float) -> bool:
    at_edge = abs(a.c) >= C_RANGE.max() or a.k <= K_RANGE.min() + 1e-6 or a.k >= K_RANGE.max() - 1e-6
    return abs(a.r) > thr and not at_edge


def alignments(spectra, recompute=False):
    """Per-spectrum Alignment objects (cached in outputs/alignment.json)."""
    key = [f"{s.device}/{s.pair}" for s in spectra]
    if os.path.exists(CACHE) and not recompute:
        c = json.load(open(CACHE))
        if c.get("keys") == key and "threshold" in c:
            return [Alignment(**a) for a in c["align"]], c["threshold"]
    raw = [branches(s.wavenumber, s.d2) for s in spectra]
    norm = [np.stack(b) / robust_scale(np.concatenate(b)) for b in raw]
    thr, null = null_threshold(norm, n=200, q=NULL_Q)
    out = []
    for p, m in norm:
        a = align(p, m, 0.0)
        a.accepted = _accept(a, thr)
        if not a.accepted:
            a = Alignment(1.0, 0.0, a.r, a.r0, False)
        out.append(a)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(dict(keys=key, threshold=thr, null=null, align=[a.__dict__ for a in out]),
              open(CACHE, "w"), indent=1, default=float)
    return out, thr


def prepare_all(spectra=None):
    """Returns spectra, X (N,2,L) normalised on the midpoint axis, masks, scales, alignments."""
    spectra = spectra or load_all()
    A, _ = alignments(spectra)
    X, M, SC = [], [], []
    for s, a in zip(spectra, A):
        p, m = branches(s.wavenumber, s.d2)
        x, mask = to_midpoint(p, m, a)
        sc = robust_scale(x[mask])
        X.append(x / sc)
        M.append(mask)
        SC.append(sc)
    return spectra, np.stack(X).astype(np.float32), np.stack(M), np.array(SC), A


def residual_noise(X, A):
    """Noise estimates from aligned spectra: what one branch does not share with the other.

    For each accepted alignment, fit m = b * p by least squares and take (m - b p)/sqrt(1 + b^2)
    as a single-branch noise realisation (in the spectrum's normalised units)."""
    out = []
    for x, a in zip(X, A):
        if not a.accepted:
            continue
        p, m = x
        b = float(np.dot(p, m) / (np.dot(p, p) + 1e-12))
        out.append((m - b * p) / np.sqrt(1 + b * b))
    return np.stack(out)


__all__ = ["alignments", "prepare_all", "residual_noise", "GRID"]
