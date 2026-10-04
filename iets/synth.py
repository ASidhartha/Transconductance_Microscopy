"""Synthetic two-branch IETS spectra with realistic noise, for supervised training.

Clean signal: a sum of vibrational features at random positions (uniform over the
grid -- deliberately NOT drawn from the FTIR table, so the model cannot learn to
hallucinate peaks at "expected" bond positions).

  * Most features appear in both bias branches, with an amplitude ratio and a sign
    relation (odd, even or mixed) that vary, as observed in the data.
  * Some features appear in one branch only; the target keeps them, so strong
    features are not deleted merely because the other branch lacks them.
  * The branches are only approximately aligned: after the per-spectrum alignment
    step a residual shift/stretch remains, so the -V branch is warped slightly.
  * Amplitudes span 0.5-30x the noise level, matching the real data (largest
    features are typically 8x and up to ~30x the noise).

Noise: phase-randomised surrogates of the *residual* noise measured on aligned real
spectra (the part of one branch not shared with the other), which keeps the real
noise's correlation length without containing the real signal.
"""
from __future__ import annotations

import numpy as np

from .preprocess import GRID, detrend, robust_scale

L = GRID.size


class NoiseBank:
    def __init__(self, traces: np.ndarray):
        t = traces - traces.mean(axis=1, keepdims=True)
        t = t / t.std(axis=1, keepdims=True)
        self.amp = np.abs(np.fft.rfft(t, axis=1))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        idx = rng.integers(0, len(self.amp), n)
        phase = rng.uniform(0, 2 * np.pi, (n, self.amp.shape[1]))
        phase[:, 0] = 0
        x = np.fft.irfft(self.amp[idx] * np.exp(1j * phase), n=L, axis=1)
        return x / x.std(axis=1, keepdims=True)


def _lineshape(x, mu, sig, kind):
    z = (x - mu) / sig
    g = np.exp(-0.5 * z**2)
    if kind == 0:
        return g
    return -z * g / 0.6065  # derivative-of-Gaussian (dispersive), unit peak height


def make_batch(rng: np.random.Generator, bank: NoiseBank, n: int):
    """Return inputs X (n,2,L) and clean targets Y (n,2,L), both normalised like real data."""
    X = np.empty((n, 2, L), np.float32)
    Y = np.empty((n, 2, L), np.float32)
    noise = bank.sample(rng, 2 * n).reshape(n, 2, L)
    t = np.linspace(-1, 1, L)
    for i in range(n):
        clean = np.zeros((2, L))
        k = 0 if rng.random() < 0.10 else rng.integers(1, 9)
        global_sym = rng.choice([-1.0, 1.0])
        # residual misalignment of the -V branch after the alignment step
        stretch, shift = np.exp(rng.normal(0, 0.02)), rng.normal(0, 12.0)
        w_neg = GRID * stretch + shift
        for _ in range(k):
            mu = rng.uniform(GRID[0] + 30, GRID[-1] - 30)
            sig = rng.uniform(12, 70)
            kind = int(rng.random() < 0.3)
            a = rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(0.5), np.log(30.0)))
            where = rng.random()
            if where < 0.8 or where >= 0.9:   # +V branch (80% both, 10% +V only)
                clean[0] += a * _lineshape(GRID, mu, sig, kind)
            if where < 0.9:                   # -V branch (80% both, 10% -V only)
                rel = global_sym if rng.random() < 0.7 else rng.choice([-1.0, 1.0])
                ratio = np.exp(rng.normal(0, 0.4))
                clean[1] += rel * ratio * a * _lineshape(w_neg, mu + rng.normal(0, 4.0),
                                                         sig * np.exp(rng.normal(0, .1)), kind)
        lvl = np.exp(rng.normal(0, 0.3, 2))[:, None]
        base = np.stack([np.polyval(rng.normal(0, 1.5, 4), t) for _ in range(2)])
        x = clean + lvl * noise[i] + base
        x = np.stack([detrend(c) for c in x])
        y = np.stack([detrend(c) for c in clean])
        s = robust_scale(x)
        X[i], Y[i] = x / s, y / s
    return X, Y
