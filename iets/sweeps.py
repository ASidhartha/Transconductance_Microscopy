"""Repeated raw I-V sweeps (ALD MoS2 hand-off data): loading, artefact removal and
regularised second derivatives.

Unlike the rGO files, which hold one averaged sweep with exported derivatives, these files keep
every repeated sweep with its time stamps. That makes three things possible that the averaged
data do not allow:

* separating components that are locked to the bias (the same in every repeat) from components
  locked to time (different phase in every repeat: interference lines, drift);
* estimating the noise of the averaged sweep from the scatter of the repeats (bootstrap);
* choosing the amount of smoothing by cross-validation between independent halves of the
  repeats, which needs no noise model and no ground truth.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu

from .data import EV_TO_CM1

KT_300K = 0.025852  # eV
THERMAL_FWHM_V = 5.4 * KT_300K  # IETS line width at room temperature (Lambe & Jaklevic)


@dataclass
class RawSweep:
    kind: str            # "type1" (0 -> +0.5 V) or "type2" (-0.5 -> +0.5 V)
    location: str        # e.g. "pos1", "centre2"
    V: np.ndarray        # (n,) programmed bias, identical in every repeat
    I: np.ndarray        # (R, n) current of every repeat
    t: np.ndarray        # (R, n) seconds since the start of that repeat
    glitches: np.ndarray = field(default=None)  # (R, n) bool, filled by clean()
    lines: list = field(default_factory=list)   # per repeat: list of (f, A, B) removed by clean()

    @property
    def name(self):
        return f"{self.kind}/{self.location}"

    @property
    def R(self):
        return self.I.shape[0]

    @property
    def dv(self):
        return float(np.mean(np.diff(self.V)))

    @property
    def dt(self):
        return float(np.mean(np.diff(self.t, axis=1)))

    @property
    def rate(self):
        """Sweep rate in V/s."""
        return self.dv / self.dt


def load_raw(path: str) -> RawSweep:
    d = pd.read_csv(path)
    d.columns = ["rep", "var2", "point", "V", "I", "t"]
    reps = [g for _, g in d.groupby("rep")]
    V = reps[0].V.values.astype(float)
    I = np.array([g.I.values for g in reps], float)
    t = np.array([g.t.values - g.t.values[0] for g in reps], float)
    kind = "type1" if "type1" in path.lower() else "type2"
    loc = os.path.basename(path).replace("_mos2", "").replace("mos2_", "").replace(".csv", "")
    return RawSweep(kind, loc, V, I, t)


def load_mos2(root: str = "data/mos2") -> list[RawSweep]:
    out = [load_raw(f) for f in sorted(glob.glob(os.path.join(root, "*", "*.csv")))]
    return sorted(out, key=lambda s: (s.kind, s.location))


# ---------------------------------------------------------------- smoothing primitives

def _diff_matrix(n: int, order: int) -> sparse.csc_matrix:
    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D


class Whittaker:
    """Penalised least squares f = argmin |y - f|^2 + lam |D_q f|^2 (Whittaker-Eilers smoother).

    Equivalent to a Gaussian-process / Tikhonov estimate with a q-th order random-walk prior.
    The factorisation is cached, so repeated calls with the same (n, lam, q) are cheap."""

    def __init__(self, n: int, lam: float, order: int = 4):
        self.n, self.lam, self.order = n, lam, order
        D = _diff_matrix(n, order)
        self._lu = splu((sparse.eye(n, format="csc") + lam * (D.T @ D)).tocsc())

    def __call__(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)
        if y.ndim == 1:
            return self._lu.solve(y)
        return np.stack([self._lu.solve(r) for r in y])

    def kernel_fwhm(self) -> float:
        """FWHM (in samples) of the equivalent kernel of the second derivative at the centre."""
        e = np.zeros(self.n); e[self.n // 2] = 1.0
        h = second_diff(self(e))
        h = np.abs(h) / np.abs(h).max()
        # the central lobe of the d2 kernel is negative; its half-width is what limits resolution
        c = self.n // 2
        lo = c
        while lo > 0 and h[lo - 1] >= 0.5:
            lo -= 1
        hi = c
        while hi < self.n - 1 and h[hi + 1] >= 0.5:
            hi += 1
        return float(hi - lo + 1)


def second_diff(y: np.ndarray, dv: float = 1.0) -> np.ndarray:
    """Central second difference along the last axis (edges repeated)."""
    y = np.asarray(y, float)
    d = np.empty_like(y)
    d[..., 1:-1] = y[..., 2:] - 2 * y[..., 1:-1] + y[..., :-2]
    d[..., 0], d[..., -1] = d[..., 1], d[..., -2]
    return d / dv**2


def first_diff(y: np.ndarray, dv: float = 1.0) -> np.ndarray:
    return np.gradient(y, axis=-1) / dv


# ---------------------------------------------------------------- artefact removal

def find_glitches(I: np.ndarray, z_thr: float = 7.0, lam: float = 2.0) -> np.ndarray:
    """Isolated single-sample spikes in each repeat (instrument glitches).

    The repeat is high-passed with a light Whittaker smoother; samples whose residual exceeds
    z_thr robust standard deviations are flagged together with their direct neighbours."""
    W = Whittaker(I.shape[1], lam, order=2)
    mask = np.zeros(I.shape, bool)
    for k, y in enumerate(I):
        r = y - W(y)
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        bad = np.abs(r - np.median(r)) > z_thr * s
        mask[k] = bad | np.roll(bad, 1) | np.roll(bad, -1)
        mask[k, 0] = mask[k, -1] = False
    return mask


def _interp_masked(y, m):
    if not m.any():
        return y
    x = np.arange(len(y))
    y = y.copy()
    y[m] = np.interp(x[m], x[~m], y[~m])
    return y


def fit_lines(t: np.ndarray, r: np.ndarray, band=(0.1, 2.4), max_lines: int = 5,
              p_false: float = 1e-3, pad: int = 8):
    """Narrow interference lines in a high-passed time series r(t).

    A line is accepted when its periodogram ordinate exceeds the local mean level by the
    threshold at which the chance of a false line anywhere in the band is p_false (periodogram
    ordinates of noise are exponential, so P(I/<I> > x) = exp(-x) per independent frequency).
    Frequency, amplitude and phase are then refined by least squares and the line subtracted;
    the search is repeated for further lines. Returns [(f, A, B)] with model A cos + B sin."""
    from scipy.optimize import least_squares
    from scipy.signal import medfilt

    t = t - t[0]
    dt = np.mean(np.diff(t))
    scale = np.std(r) or 1.0
    y = r / scale
    n = len(y)
    lines = []
    n_indep = max(1, int((band[1] - band[0]) * n * dt))
    thr = np.log(n_indep / p_false)
    win = np.hanning(n)
    for _ in range(max_lines):
        F = np.fft.rfft((y - y.mean()) * win, n * pad)
        P = np.abs(F) ** 2
        f = np.fft.rfftfreq(n * pad, dt)
        bg = medfilt(P[::pad], 21) / np.log(2)  # local mean of the exponential distribution
        bg = np.interp(f, f[::pad], bg)
        sel = (f >= band[0]) & (f <= band[1])
        i = np.argmax(np.where(sel, P / np.maximum(bg, 1e-30), 0))
        if P[i] / bg[i] < thr:
            break
        f0 = f[i]

        def res(p):
            ph = 2 * np.pi * p[2] * t
            return y - (p[0] * np.cos(ph) + p[1] * np.sin(ph))

        c, s = np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t)
        A0, B0 = np.linalg.lstsq(np.c_[c, s], y, rcond=None)[0]
        p = least_squares(res, [A0, B0, f0], x_scale=[1, 1, 1 / (n * dt)]).x
        lines.append((p[2], p[0] * scale, p[1] * scale))
        y = res(p)
    return lines


def line_model(t: np.ndarray, lines) -> np.ndarray:
    t = t - t[0]
    out = np.zeros_like(t, float)
    for f, A, B in lines:
        out += A * np.cos(2 * np.pi * f * t) + B * np.sin(2 * np.pi * f * t)
    return out


def demodulate_line(t: np.ndarray, r: np.ndarray, f0: float, bw: float = 0.03) -> np.ndarray:
    """Interference line with slowly varying amplitude and phase (complex demodulation).

    r is multiplied by exp(-2 pi i f0 t), low-passed with a Gaussian of standard deviation
    1/(2 pi bw) seconds and remodulated. A line whose frequency wanders by less than ~bw
    (mains frequency drifts by a few 10 mHz over a sweep) is captured, while structure that
    is not concentrated near f0 is attenuated by the narrow pass band."""
    from scipy.ndimage import gaussian_filter1d
    dt = np.mean(np.diff(t))
    z = r * np.exp(-2j * np.pi * f0 * t)
    sig = 1.0 / (2 * np.pi * bw) / dt
    env = gaussian_filter1d(z.real, sig, mode="reflect") + 1j * gaussian_filter1d(z.imag, sig, mode="reflect")
    return 2 * np.real(env * np.exp(2j * np.pi * f0 * t))


HP_PERIOD_S = 12.0  # high-pass used before the line search: removes the I-V shape, keeps 0.1-2.5 Hz


def _lam_for_period(period_samples: float, order: int) -> float:
    """Whittaker lambda whose transfer function is 1/2 at the given period."""
    w = 2 * np.pi / period_samples
    return 1.0 / (2 * (1 - np.cos(w))) ** order


def clean(s: RawSweep, t_skip: float = 20.0, remove_lines: bool = True,
          remove_glitches: bool = True) -> np.ndarray:
    """Return the repeats with glitches interpolated and interference lines subtracted.

    Lines are searched for in each repeat separately, after the first t_skip seconds (the
    sweep-start transient) and after a high-pass that removes structure slower than
    HP_PERIOD_S. A line coherent over the whole sweep is subtracted over the whole sweep."""
    I = s.I.copy()
    s.glitches = find_glitches(I) if remove_glitches else np.zeros(I.shape, bool)
    for k in range(s.R):
        I[k] = _interp_masked(I[k], s.glitches[k])
    s.lines = [[] for _ in range(s.R)]
    if remove_lines:
        lam = _lam_for_period(HP_PERIOD_S / s.dt, 2)
        W = Whittaker(I.shape[1], lam, order=2)
        for k in range(s.R):
            keep = s.t[k] >= t_skip
            tk = s.t[k]
            r = I[k] - W(I[k])
            s.lines[k] = fit_lines(tk[keep], r[keep])
            for f, A, B in s.lines[k]:
                # the detection fixes f0; amplitude and phase are followed through the sweep
                line = demodulate_line(tk[keep], r[keep], f)
                I[k, keep] -= line
                r[keep] -= line
    return I
    return I


# ---------------------------------------------------------------- regularised derivative

def legacy_sg2(y: np.ndarray, dv: float, w1: int = 31, w2: int = 31, poly: int = 3) -> np.ndarray:
    """The rGO-paper practice: Savitzky-Golay smoothing, two gradients, Savitzky-Golay again."""
    from scipy.signal import savgol_filter
    a = savgol_filter(y, w1, poly)
    d2 = np.gradient(np.gradient(a)) / dv**2
    return savgol_filter(d2, w2, poly)


def sg_d2(y: np.ndarray, dv: float, w: int, poly: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter
    return savgol_filter(y, w, poly, deriv=2, delta=dv, axis=-1)


def cv_split_loss(I: np.ndarray, estimator, dv: float, window: np.ndarray, n_split: int = 60,
                  rng=None, I_target: np.ndarray | None = None, return_noise: bool = False):
    """Cross-validated error of a second-derivative estimator (noise2noise between repeats).

    The repeats are split at random into halves A and B; the estimator applied to mean(A) is
    compared with the unbiased but noisy central second difference of mean(B). Because the
    noise of A and B is independent, the expected loss is the estimator's mean-squared error
    plus the noise variance of the target, so its minimum picks the best estimator without any
    ground truth. The target is taken from I_target (default I); passing the uncleaned repeats
    makes any bias-locked signal removed by a cleaning step count as error. With return_noise,
    the target noise variance (half the variance of the difference of the two halves' second
    differences) is returned too, so that loss - noise estimates the mean-squared error."""
    rng = np.random.default_rng(0) if rng is None else rng
    I_target = I if I_target is None else I_target
    R = I.shape[0]
    h = R // 2
    out, noise = [], []
    for _ in range(n_split):
        p = rng.permutation(R)
        A, B = p[:h], p[h:2 * h]
        est = estimator(I[A].mean(0))
        tgt = second_diff(I_target[B].mean(0), dv)
        out.append(np.mean((est[window] - tgt[window]) ** 2))
        if return_noise:
            ta = second_diff(I_target[A].mean(0), dv)
            noise.append(0.5 * np.mean((ta[window] - tgt[window]) ** 2))
    if return_noise:
        return float(np.mean(out)), float(np.mean(noise))
    return float(np.mean(out))


LAMBDAS = np.logspace(2, 12, 41)


def choose_lambda(I: np.ndarray, dv: float, window: np.ndarray, order: int = 4,
                  lambdas=LAMBDAS, n_split: int = 60):
    """Lambda that minimises the split-half CV loss for R/2 repeats, rescaled to R repeats.

    Halving the number of repeats doubles the noise variance, and the optimal lambda of a
    penalised least-squares estimator scales with the noise variance, so lambda_R = lambda_{R/2}/2."""
    n = I.shape[1]
    losses = []
    for lam in lambdas:
        W = Whittaker(n, lam, order)
        losses.append(cv_split_loss(I, lambda y: second_diff(W(y), dv), dv, window, n_split))
    losses = np.array(losses)
    i = int(np.argmin(losses))
    # parabolic refinement in log(lambda)
    lam_half = float(lambdas[i])
    if 0 < i < len(lambdas) - 1:
        x = np.log(lambdas[i - 1:i + 2]); yv = losses[i - 1:i + 2]
        curv = yv[2] - 2 * yv[1] + yv[0]
        if curv > 0:
            lam_half = float(np.exp(x[1] - (x[1] - x[0]) * (yv[2] - yv[0]) / (2 * curv)))
    return lam_half / 2.0, lam_half, losses


@dataclass
class Spectrum2:
    """Second-derivative spectrum of one location with its repeat-bootstrap uncertainty."""
    name: str
    V: np.ndarray
    t: np.ndarray            # time since sweep start of each bias point (first repeat)
    I_mean: np.ndarray
    G: np.ndarray            # dI/dV of the regularised fit
    d2: np.ndarray           # regularised d2I/dV2
    d2_boot: np.ndarray      # (B, n) bootstrap replicates
    lam: float
    fwhm_v: float            # resolution: FWHM of the equivalent d2 kernel, volts
    valid: np.ndarray        # bool, outside the sweep-start transient and the sweep ends

    @property
    def sigma(self):
        return self.d2_boot.std(0, ddof=1)

    @property
    def wavenumber(self):
        return self.V * EV_TO_CM1


def regularised_spectrum(s: RawSweep, I: np.ndarray, lam: float, order: int = 4,
                         n_boot: int = 300, t_skip: float = 15.0, edge: int = 6,
                         rng=None) -> Spectrum2:
    rng = np.random.default_rng(1) if rng is None else rng
    n = I.shape[1]
    W = Whittaker(n, lam, order)
    m = I.mean(0)
    f = W(m)
    d2 = second_diff(f, s.dv)
    boots = np.empty((n_boot, n))
    for b in range(n_boot):
        idx = rng.integers(0, s.R, s.R)
        boots[b] = second_diff(W(I[idx].mean(0)), s.dv)
    valid = s.t[0] >= t_skip
    valid[:edge] = valid[-edge:] = False
    return Spectrum2(s.name, s.V, s.t[0], m, first_diff(f, s.dv), d2, boots, lam,
                     W.kernel_fwhm() * abs(s.dv), valid)


# ---------------------------------------------------------------- half-data Fourier correlation

FRC_HALF = 1.0 / 3.0  # FRC of the halves at which the Wiener gain of the full average is 1/2


def fourier_ring_correlation(I: np.ndarray, window: np.ndarray, dv: float, n_split: int = 60,
                             rng=None, min_width: int = 2):
    """Correlation between the Fourier coefficients of two independent halves of the repeats,
    in bins of bias frequency (cycles per volt).

    This is the one-dimensional analogue of the Fourier shell correlation used in cryo-electron
    microscopy to measure resolution from half-maps. Differentiation multiplies every Fourier
    coefficient by (2 pi i f)^2 in both halves, so the curve is the same for I, dI/dV and
    d2I/dV2: it tells at which bias scales the data carry reproducible information, whatever
    derivative estimator is used afterwards. Each half is linearly detrended and Tukey-tapered
    over the analysis window. Returns (centre frequencies, mean FRC, coefficients per bin)."""
    from scipy.signal.windows import tukey
    rng = np.random.default_rng(0) if rng is None else rng
    R = I.shape[0]
    h = R // 2
    idx = np.where(window)[0]
    n = len(idx)
    x = np.arange(n)
    taper = tukey(n, 0.2)
    df = 1.0 / (n * abs(dv))
    # bins of Fourier indices 1..n/2, at least min_width wide, widening by ~20% per bin
    ib, k0 = [], 1
    while k0 <= n // 2:
        ib.append(k0)
        k0 += max(min_width, int(round(0.2 * k0)))
    edges = np.r_[ib, n // 2 + 1] - 0.5
    k = np.arange(n // 2 + 1)
    masks = [(k >= lo) & (k < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    out = []
    for _ in range(n_split):
        p = rng.permutation(R)
        a, b = I[p[:h]].mean(0)[idx], I[p[h:2 * h]].mean(0)[idx]
        a = a - np.polyval(np.polyfit(x, a, 1), x)
        b = b - np.polyval(np.polyfit(x, b, 1), x)
        A, B = np.fft.rfft(a * taper), np.fft.rfft(b * taper)
        out.append([np.real(np.sum(A[m] * np.conj(B[m]))) /
                    np.sqrt(np.sum(np.abs(A[m]) ** 2) * np.sum(np.abs(B[m]) ** 2)) for m in masks])
    fc = np.sqrt(np.maximum(edges[:-1], 0.5) * edges[1:]) * df
    return fc, np.mean(out, 0), np.array([m.sum() for m in masks])


def frc_cutoff(fc: np.ndarray, c: np.ndarray, level: float = FRC_HALF, f_min: float = 0.0) -> float:
    """Lowest frequency above f_min at which the FRC falls below `level` (log-interpolated)."""
    for i in range(1, len(fc)):
        if fc[i] > f_min and c[i] < level <= c[i - 1]:
            w = (c[i - 1] - level) / (c[i - 1] - c[i])
            return float(np.exp(np.log(fc[i - 1]) + w * (np.log(fc[i]) - np.log(fc[i - 1]))))
    return float(fc[0]) if c[0] < level else float(fc[-1])


def lam_for_cutoff(f_cut: float, dv: float, order: int = 4) -> float:
    """Whittaker lambda whose gain is 1/2 at f_cut (cycles per volt)."""
    return _lam_for_period(1.0 / (f_cut * abs(dv)), order)
