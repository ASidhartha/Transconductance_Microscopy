"""Feature detection in repeated-sweep second-derivative spectra, with permutation statistics.

Every test uses the repeats themselves as the reference, so none needs a noise model:

* within a location, a sign-flip permutation test with a max-statistic over the bias axis
  (family-wise error control, as in neuroimaging; Nichols & Holmes 2002);
* between the two bias polarities and between locations, the chance of a coincident feature is
  measured by scanning each trace over all positions (equivalent to a circular-shift null).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import stats

from .data import EV_TO_CM1
from .sweeps import RawSweep, Whittaker, lam_for_cutoff, second_diff, THERMAL_FWHM_V

BASELINE_CUTOFF = 3.0  # cycles per volt: slower structure is treated as background


@dataclass
class Traces:
    """Per-repeat feature traces x_k(V) = d2I/dV2 - background, divided by the median dI/dV."""
    name: str
    kind: str
    V: np.ndarray
    t: np.ndarray
    x: np.ndarray            # (R, n)
    valid: np.ndarray        # (n,) bool
    fcut: float              # cycles per volt used for the smoothing
    fwhm_v: float            # resolution (equivalent-kernel FWHM), volts
    G0: float                # median dI/dV over the valid window (S)

    @property
    def R(self):
        return self.x.shape[0]

    def mean(self):
        return self.x.mean(0)

    def tstat(self, x=None):
        """Pseudo-t: mean over repeats divided by a standard error whose variance is smoothed
        along the bias axis (Gaussian, sigma = 2 resolution FWHM, within the valid window).

        With 5-20 repeats the per-point variance is too unstable for a plain t statistic;
        pooling it over neighbouring bias points (the smoothed-variance pseudo-t of permutation
        neuroimaging) raises the power while the sign-flip test, which recomputes the same
        statistic for every flip, stays exact."""
        from scipy.ndimage import gaussian_filter1d
        x = self.x if x is None else x
        var = x.var(0, ddof=1)
        dv = abs(float(np.mean(np.diff(self.V))))
        sig = 2 * self.fwhm_v / dv
        w = self.valid.astype(float)
        num = gaussian_filter1d(var * w, sig, mode="nearest")
        den = gaussian_filter1d(w, sig, mode="nearest")
        v = np.where(den > 1e-6, num / np.maximum(den, 1e-6), var)
        return x.mean(0) / (np.sqrt(v / x.shape[0]) + 1e-300)


def background_basis(V, t, y):
    """Basis of the smooth background in the window: a cubic in V plus two decaying
    exponentials in t, the time since the sweep start (the tail of the sweep-start transient).

    The two time constants are picked from a small grid by fitting the mean current y; the
    returned basis is then projected out of every repeat separately. Keeping the operation
    linear and identical for all repeats is what keeps the sign-flip test exact: fitting the
    mean and subtracting one curve from all repeats would leave a common, noise-driven term."""
    x = (V - V.mean()) / np.ptp(V)
    P = np.vander(x, 4)
    best, best_r = P, np.sum((y - P @ np.linalg.lstsq(P, y, rcond=None)[0]) ** 2)
    for t1 in (3.0, 10.0):
        for t2 in (30.0, 100.0, 300.0):
            B = np.c_[P, np.exp(-t / t1), np.exp(-t / t2)]
            r = np.sum((y - B @ np.linalg.lstsq(B, y, rcond=None)[0]) ** 2)
            if r < best_r:
                best, best_r = B, r
    return best


def project_out(B, Y):
    """Residual of each row of Y after least-squares projection onto the columns of B."""
    Q, _ = np.linalg.qr(B)
    return Y - (Y @ Q) @ Q.T


def apply_operator(s: RawSweep, Y: np.ndarray, I_ref: np.ndarray, fcut: float, valid: np.ndarray,
                   order: int = 4) -> np.ndarray:
    """The linear map from currents to feature traces used by make_traces, with the background
    basis and dI/dV taken from the measured repeats I_ref, applied to the rows of Y. Used to
    pass template features through exactly the processing that the data received."""
    idx = np.where(valid)[0]
    raw = I_ref[:, idx]
    G0 = float(np.median(np.gradient(raw.mean(0)) / s.dv))
    B = background_basis(s.V[idx], s.t[0][idx], raw.mean(0))
    seg = project_out(B, np.atleast_2d(Y)[:, idx])
    n = seg.shape[1]
    W = Whittaker(n, lam_for_cutoff(fcut, s.dv, order), order)
    Wb = Whittaker(n, lam_for_cutoff(BASELINE_CUTOFF, s.dv, order), order)
    out = np.zeros((seg.shape[0], len(s.V)))
    out[:, idx] = (second_diff(W(seg), s.dv) - second_diff(Wb(seg), s.dv)) / G0
    return out


def make_traces(s: RawSweep, I: np.ndarray, fcut: float, valid: np.ndarray, order: int = 4) -> Traces:
    """Smooth, differentiate and remove the background within the analysis window only.

    Working on the window alone keeps the steep sweep-start transient (excluded from the
    window) from leaking into the background estimate as ringing."""
    idx = np.where(valid)[0]
    raw = I[:, idx]
    G0 = float(np.median(np.gradient(raw.mean(0)) / s.dv))
    seg = project_out(background_basis(s.V[idx], s.t[0][idx], raw.mean(0)), raw)
    n = seg.shape[1]
    W = Whittaker(n, lam_for_cutoff(fcut, s.dv, order), order)
    Wb = Whittaker(n, lam_for_cutoff(BASELINE_CUTOFF, s.dv, order), order)
    d2 = second_diff(W(seg), s.dv)
    base = second_diff(Wb(seg), s.dv)
    x = np.zeros(I.shape)
    x[:, idx] = (d2 - base) / G0
    v = valid.copy()
    # the second difference is undefined at the two ends of the segment
    v[idx[:2]] = v[idx[-2:]] = False
    return Traces(s.name, s.kind, s.V, s.t[0], x, v, fcut, W.kernel_fwhm() * abs(s.dv), G0)


def signflip_maxt(tr: Traces, n_perm: int = 5000, rng=None):
    """Sign-flip permutation distribution of max |t| over the valid window.

    Under the null hypothesis that a location has no reproducible feature beyond the smooth
    background, each repeat's trace is symmetric about zero, so flipping the sign of whole
    repeats leaves the joint distribution unchanged. The max over the bias axis controls the
    family-wise error over all positions. All 2^R patterns are used when R <= 12."""
    rng = np.random.default_rng(2) if rng is None else rng
    R = tr.R
    x = tr.x
    if R <= 12:
        signs = np.array(list(product([1, -1], repeat=R)), float)
    else:
        signs = rng.choice([1.0, -1.0], size=(n_perm, R))
        signs[0] = 1.0
    out = np.empty(len(signs))
    for i, sg in enumerate(signs):
        out[i] = np.max(np.abs(tr.tstat(x * sg[:, None]))[tr.valid])
    return out


@dataclass
class Feature:
    location: str
    kind: str
    V: float
    sigma_V: float
    fwhm_V: float
    amplitude: float         # (d2I/dV2)/(dI/dV), 1/V
    t: float                 # time since sweep start, s
    tstat: float
    p_fwer: float            # sign-flip max-t p value within the location
    polarity: str = "n/a"
    p_polarity: float = np.nan
    n_locations: int = 0     # other locations with a same-sign feature at this bias
    n_tested: int = 0
    p_recur: float = np.nan
    q_recur: float = np.nan

    @property
    def wavenumber(self):
        return self.V * EV_TO_CM1


def _extrema(y, valid):
    i = np.arange(1, len(y) - 1)
    m = ((y[i] > y[i - 1]) & (y[i] >= y[i + 1])) | ((y[i] < y[i - 1]) & (y[i] <= y[i + 1]))
    i = i[m]
    return i[valid[i]]


def _fwhm(y, i, dv):
    h = abs(y[i]) / 2
    lo = i
    while lo > 0 and np.sign(y[lo - 1]) == np.sign(y[i]) and abs(y[lo - 1]) >= h:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and np.sign(y[hi + 1]) == np.sign(y[i]) and abs(y[hi + 1]) >= h:
        hi += 1
    return (hi - lo + 1) * abs(dv)


SMALL_R = 7  # below this, sign flipping has too few patterns; a parametric max-t is used


def parametric_maxt_p(tr: Traces, tabs: np.ndarray) -> np.ndarray:
    """Family-wise p value of |t| for few repeats: pointwise Student-t (R - 1 degrees of
    freedom), Sidak-corrected for the number of independent resolution elements in the window."""
    dv = abs(float(np.mean(np.diff(tr.V))))
    n_eff = max(1.0, tr.valid.sum() * dv / tr.fwhm_v)
    p = 2 * stats.t.sf(np.abs(tabs), tr.R - 1)
    return 1 - (1 - p) ** n_eff


def detect(tr: Traces, alpha: float = 0.05, n_boot: int = 300, rng=None, extra_null=None,
           thr_override: float | None = None):
    """Features whose |t| exceeds the family-wise (1 - alpha) threshold.

    The threshold is the sign-flip max-t quantile, or the parametric one for fewer than SMALL_R
    repeats; with extra_null (max |t| of synthetic feature-free data sets), the stricter of the
    two is used. Position uncertainty: the repeats are resampled with replacement and the extremum
    of the same sign nearest to the feature (within one resolution FWHM) is located again."""
    rng = np.random.default_rng(3) if rng is None else rng
    t = tr.tstat()
    m = tr.mean()
    dv = float(np.mean(np.diff(tr.V)))
    half = max(2, int(round(tr.fwhm_v / abs(dv))))
    if tr.R >= SMALL_R:
        null = signflip_maxt(tr)
        thr = float(np.quantile(null, 1 - alpha))
        pfun = lambda a: (1 + np.sum(null >= a)) / (1 + len(null))
    else:
        null = np.array([])
        grid = np.linspace(0, 200, 20001)
        pg = parametric_maxt_p(tr, grid)
        thr = float(grid[np.argmax(pg <= alpha)])
        pfun = lambda a: float(parametric_maxt_p(tr, np.array([a]))[0])
    if extra_null is not None and len(extra_null):
        # calibration on synthetic feature-free data built from this location's own repeats:
        # the stricter of the two thresholds and the larger of the two p values are used
        en = np.asarray(extra_null)
        thr = max(thr, float(np.quantile(en, 1 - alpha)))
        pf0 = pfun
        pfun = lambda a: max(pf0(a), (1 + np.sum(en >= a)) / (1 + len(en)))
    if thr_override is not None:  # list candidates below the family-wise threshold too
        thr = thr_override
    feats = []
    for i in _extrema(t, tr.valid):
        if abs(t[i]) < thr:
            continue
        pos = []
        for _ in range(n_boot):
            mb = tr.x[rng.integers(0, tr.R, tr.R)].mean(0)
            lo, hi = max(0, i - half), min(len(mb), i + half + 1)
            seg = np.sign(m[i]) * mb[lo:hi]
            pos.append(tr.V[lo + int(np.argmax(seg))])
        p = pfun(abs(t[i]))
        feats.append(Feature(tr.name, tr.kind, float(tr.V[i]), float(np.std(pos)),
                             _fwhm(m, i, dv), float(m[i]), float(tr.t[i]), float(t[i]), float(p)))
    return feats, thr, null


def _hits(tr: Traces, sign: float, tol: float, p_point: float = 0.05):
    """Boolean over the bias axis: a same-sign, pointwise-significant feature lies within +-tol."""
    tcrit = stats.t.ppf(1 - p_point / 2, tr.R - 1)
    t = tr.tstat()
    sig = tr.valid & (np.sign(t) == sign) & (np.abs(t) >= tcrit)
    dv = abs(float(np.mean(np.diff(tr.V))))
    k = max(1, int(round(tol / dv)))
    return np.convolve(sig.astype(float), np.ones(2 * k + 1), "same") > 0


def recurrence(features, traces: dict, tol_floor: float = 0.0):
    """For every feature, count other locations with a coincident same-sign feature.

    The chance of a coincidence at location j is the fraction of its valid bias positions at
    which a same-sign significant feature lies within the tolerance, which is what a random
    circular shift of trace j would give. The count of coincidences then follows a
    Poisson-binomial distribution, whose upper tail is the p value. Tolerance: the larger of
    the two resolutions, plus both position uncertainties."""
    for f in features:
        ps, hits = [], 0
        for name, tr in traces.items():
            if name == f.location:
                continue
            if not (tr.V[tr.valid].min() <= f.V <= tr.V[tr.valid].max()):
                continue
            tol = max(tol_floor, max(tr.fwhm_v, f.fwhm_V) / 2 + 2 * f.sigma_V)
            h = _hits(tr, np.sign(f.amplitude), tol)
            ps.append(h[tr.valid].mean())
            hits += bool(h[np.argmin(np.abs(tr.V - f.V))])
        f.n_locations, f.n_tested = hits, len(ps)
        f.p_recur = poisson_binomial_sf(hits, np.array(ps)) if ps else np.nan
    p = np.array([f.p_recur for f in features], float)
    q = bh(p)
    for f, qq in zip(features, q):
        f.q_recur = float(qq)
    return features


def polarity(features, traces: dict):
    """Type-2 sweeps cover both polarities. A feature at +V0 is looked for at -V0 in the same
    location, with the same sign (even) or the opposite sign (odd); chance level as above."""
    for f in features:
        tr = traces[f.location]
        if f.kind != "type2" or abs(f.V) < 0.03:
            continue
        if not (tr.V[tr.valid].min() <= -f.V <= tr.V[tr.valid].max()):
            f.polarity = "not testable"
            continue
        tol = tr.fwhm_v / 2 + 2 * f.sigma_V
        j = np.argmin(np.abs(tr.V + f.V))
        even = _hits(tr, np.sign(f.amplitude), tol)
        odd = _hits(tr, -np.sign(f.amplitude), tol)
        if even[j] or odd[j]:
            f.polarity = "even" if even[j] else "odd"
            ch = (even if even[j] else odd)[tr.valid].mean()
            f.p_polarity = float(min(1.0, 2 * ch))  # two possible symmetries
        else:
            f.polarity = "absent"
    return features


def poisson_binomial_sf(k: int, p: np.ndarray) -> float:
    """P(X >= k) for X a sum of independent Bernoulli(p_i)."""
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for pi in p:
        pmf[1:] = pmf[1:] * (1 - pi) + pmf[:-1] * pi
        pmf[0] *= (1 - pi)
    return float(pmf[k:].sum())


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (NaN kept)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def consensus(traces: dict, grid: np.ndarray, names=None, mirror: str | None = None):
    """Stouffer combination of the per-location t statistics on a common bias grid.

    mirror='odd' or 'even' adds each Type-2 location's negative half, mirrored onto positive
    bias (with the sign flipped for 'odd'), as a further replicate."""
    names = list(traces) if names is None else names
    Zs = []
    for nm in names:
        tr = traces[nm]
        z = stats.norm.ppf(stats.t.cdf(tr.tstat(), tr.R - 1))
        z = np.where(tr.valid, z, np.nan)
        Zs.append(np.interp(grid, tr.V, z, left=np.nan, right=np.nan))
        if mirror and tr.kind == "type2":
            zm = np.interp(-grid, tr.V, z, left=np.nan, right=np.nan)
            Zs.append(-zm if mirror == "odd" else zm)
    Zs = np.array(Zs)
    n = np.sum(np.isfinite(Zs), 0)
    Z = np.nansum(Zs, 0) / np.sqrt(np.maximum(n, 1))
    return np.where(n >= 2, Z, np.nan), Zs, n


def consensus_null(Zs: np.ndarray, grid: np.ndarray, n_perm: int = 2000, rng=None):
    """Max |Z| of the consensus when each trace is circularly shifted within its own valid range."""
    rng = np.random.default_rng(4) if rng is None else rng
    out = np.empty(n_perm)
    for b in range(n_perm):
        S = np.full_like(Zs, np.nan)
        for i, z in enumerate(Zs):
            ok = np.where(np.isfinite(z))[0]
            if len(ok) == 0:
                continue
            seg = z[ok]
            S[i, ok] = np.roll(seg, rng.integers(0, len(seg)))
        n = np.sum(np.isfinite(S), 0)
        Z = np.nansum(S, 0) / np.sqrt(np.maximum(n, 1))
        out[b] = np.nanmax(np.abs(np.where(n >= 2, Z, np.nan)))
    return out


def thermal_flag(fwhm_v: float) -> bool:
    """True if a feature is narrower than the room-temperature inelastic line width."""
    return fwhm_v < THERMAL_FWHM_V
