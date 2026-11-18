"""Denoising and feature statistics of the ALD MoS2 repeated sweeps (data/mos2).

Run:  .venv/bin/python mos2_analysis.py           (~10 min on 8 cores)
Writes outputs/mos2/: summary.json, noise.csv, peaks.csv, frc.csv, benchmark.csv and
spectra.npz (everything the figure script needs).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import welch, savgol_filter

from iets.data import EV_TO_CM1
from iets.features import (BASELINE_CUTOFF, background_basis, consensus, consensus_null, detect, make_traces, polarity,
                           project_out, recurrence, thermal_flag, Traces)
from iets.sweeps import (THERMAL_FWHM_V, Whittaker, _lam_for_period, clean, first_diff,
                         fourier_ring_correlation, frc_cutoff, lam_for_cutoff, legacy_sg2,
                         load_mos2, second_diff)

OUT = "outputs/mos2"
T_SKIP = 20.0     # s: sweep-start transient excluded from all spectral analysis
EDGE = 6          # samples dropped at the end of the sweep
GRID = np.arange(-0.5, 0.5 + 1e-9, 0.002)


def window(s):
    w = s.t[0] >= T_SKIP
    w[:EDGE] = False
    w[-EDGE:] = False
    return w


# ------------------------------------------------------------------ noise characterisation

def noise_summary(s, Ic):
    """Statistics of the repeat-to-repeat fluctuations n_k = I_k - mean_j I_j (no bias-locked part)."""
    w = window(s)
    res_raw = (s.I - s.I.mean(0))[:, w]
    res = (Ic - Ic.mean(0))[:, w]
    fs = 1 / s.dt
    f, P_raw = welch(res_raw, fs=fs, nperseg=min(128, w.sum()), axis=1)
    _, P = welch(res, fs=fs, nperseg=min(128, w.sum()), axis=1)
    P_raw, P = np.median(P_raw, 0), np.median(P, 0)
    lo = (f > 0.02) & (f < 0.2)
    gamma = -np.polyfit(np.log(f[lo]), np.log(P[lo]), 1)[0]
    white = np.median(P[f > 1.5])
    i_rms = np.sqrt(np.mean(Ic.mean(0)[w] ** 2))
    S01 = np.interp(0.1, f, P) / i_rms**2
    # autocorrelation length of the cleaned fluctuations, in samples
    r = res - res.mean(1, keepdims=True)
    ac = np.mean([np.correlate(x, x, "full")[len(x) - 1:] / np.dot(x, x) for x in r], 0)
    corr_len = int(np.argmax(ac < np.exp(-1)))
    lines = [l for rep in s.lines for l in rep]
    main = [l for rep in s.lines for l in rep[:1]]
    f_main = np.array([l[0] for l in main]) if main else np.array([np.nan])
    amp_main = np.array([np.hypot(l[1], l[2]) for l in main]) if main else np.array([np.nan])
    # conductance drift between repeats and zero-current offset of the up-sweep
    late = s.t[0] > 40
    G = np.array([np.polyfit(s.V[late], i[late], 1)[0] for i in Ic])
    drift = np.polyfit(np.arange(s.R) * (s.t[0, -1] + s.dt), G / G.mean(), 1)[0] * 3600
    V0 = np.nan
    if s.kind == "type2":
        m = Ic.mean(0)
        k = np.where(np.diff(np.sign(m)) > 0)[0]
        k = k[s.t[0][k] > T_SKIP]
        if len(k):
            V0 = float(np.interp(0, m[k[0]:k[0] + 2], s.V[k[0]:k[0] + 2]))
    return dict(location=s.name, kind=s.kind, repeats=s.R, points=len(s.V), step_mV=s.dv * 1e3,
                dwell_s=s.dt, rate_mV_per_s=s.rate * 1e3, G_nS=float(G.mean() * 1e9),
                G_drift_pct_per_h=float(drift * 100), V_zero_current_mV=V0 * 1e3 if V0 == V0 else np.nan,
                lsb_pA=_lsb(s.I) * 1e12,
                glitches=int(s.glitches.sum()), glitch_per_1000=float(1000 * s.glitches.mean()),
                lines_total=len(lines), repeats_with_line=int(sum(bool(l) for l in s.lines)),
                f_line_median_Hz=float(np.nanmedian(f_main)),
                f_line_iqr_Hz=float(np.subtract(*np.nanpercentile(f_main, [75, 25]))),
                mains_Hz_median=float(np.nanmedian(10 / s.dt + f_main)),
                line_amp_pA=float(np.nanmedian(amp_main) * 1e12),
                fluct_rms_pA=float(res.std() * 1e12), fluct_rms_raw_pA=float(res_raw.std() * 1e12),
                psd_gamma=float(gamma), psd_white_A2_per_Hz=float(white), SI_over_I2_at_0p1Hz=float(S01),
                corr_len_mV=corr_len * abs(s.dv) * 1e3), (f, P_raw, P)


def _lsb(I):
    for p in range(-9, -17, -1):
        q = I.ravel()[:2000] / 10.0**p
        if np.all(np.abs(q - np.round(q)) < 1e-6):
            return 10.0**p
    return np.nan


def transient_fit(s, Ic):
    """Mean current after the bias step that starts each sweep:
    I = g V + c + A1 exp(-t/tau1) + A2 exp(-t/tau2), fitted in pA and seconds.
    Amplitudes are returned as voltage equivalents A/g (mV), so that locations of different
    conductance can be compared."""
    m = Ic.mean(0) * 1e12
    t = s.t[0]
    V = s.V
    sel = t > 0.5

    def model(X, g, c, A1, tau1, A2, tau2):
        tt, vv = X
        return g * vv + c + A1 * np.exp(-tt / tau1) + A2 * np.exp(-tt / tau2)

    late = t > 60
    g0, c0 = np.polyfit(V[late], m[late], 1)
    a0 = m[sel][0] - (g0 * V[sel][0] + c0)
    best = None
    for t1, t2 in ((1.0, 15.0), (2.0, 30.0), (3.0, 60.0)):
        try:
            p, _ = curve_fit(model, (t[sel], V[sel]), m[sel], p0=[g0, c0, 0.7 * a0, t1, 0.3 * a0, t2],
                             bounds=([-np.inf, -np.inf, -np.inf, 0.2, -np.inf, 5.0],
                                     [np.inf, np.inf, np.inf, 5.0, np.inf, 1000.0]), maxfev=40000)
        except RuntimeError:
            continue
        r = np.sum((m[sel] - model((t[sel], V[sel]), *p)) ** 2)
        if best is None or r < best[1]:
            best = (p, r)
    p = best[0] if best else np.full(6, np.nan)
    g = p[0]
    resid = m[sel] - model((t[sel], V[sel]), *p)
    return dict(location=s.name, tau1_s=p[3], A1_mV=p[2] / g * 1e3, tau2_s=p[5], A2_mV=p[4] / g * 1e3,
                rms_resid_pA=float(np.std(resid)), step_at_start_V=0.5 if s.kind == "type1" else 1.0)


# ------------------------------------------------------------------ per-location analysis

def analyse_location(s, extra_null=None):
    Ic = clean(s)
    w = window(s)
    fc, c, nb = fourier_ring_correlation(Ic, w, s.dv)
    fc_raw, c_raw, _ = fourier_ring_correlation(s.I, w, s.dv)
    fcut = frc_cutoff(fc, c)
    tr = make_traces(s, Ic, fcut, w)
    feats, thr, null = detect(tr, extra_null=extra_null)
    # candidates: the three strongest extrema of the location, whatever their significance
    cands, _, _ = detect(tr, extra_null=extra_null, thr_override=0.0)
    cands = sorted(cands, key=lambda f: -abs(f.tstat))[:3]
    # legacy processing for comparison: averaged raw repeats, SG before and after differentiation
    legacy = legacy_sg2(s.I.mean(0), s.dv) / tr.G0
    raw_d2 = second_diff(s.I.mean(0), s.dv) / tr.G0
    nz, spec = noise_summary(s, Ic)
    nz["frc_cutoff_per_V"] = fcut
    nz["resolution_mV"] = 1e3 / fcut
    nz["kernel_fwhm_mV"] = tr.fwhm_v * 1e3
    nz["maxt_threshold"] = thr
    nz["max_abs_t"] = float(np.max(np.abs(tr.tstat())[tr.valid]))
    nz["V_at_max_abs_t"] = float(tr.V[np.argmax(np.abs(tr.tstat()) * tr.valid)])
    nz["n_features"] = len(feats)
    nz["trace_noise_per_V"] = float(np.median(tr.x.std(0, ddof=1)[w]) / np.sqrt(tr.R))
    # how much of the conventionally processed spectrum the removed artefacts account for
    lines_only = np.zeros_like(s.I)
    I_g = s.I.copy()
    for k in range(s.R):
        I_g[k] = np.where(s.glitches[k], np.interp(np.arange(len(s.V)), np.where(~s.glitches[k])[0],
                                                   s.I[k][~s.glitches[k]]), s.I[k])
    lines_only = I_g - Ic
    glitch_only = s.I - I_g
    leg_lines = legacy_sg2(lines_only.mean(0), s.dv) / tr.G0
    leg_glitch = legacy_sg2(glitch_only.mean(0), s.dv) / tr.G0
    lw = w.copy()
    nz["legacy_var_frac_lines"] = float(np.var(leg_lines[lw]) / np.var(legacy[lw]))
    nz["legacy_var_frac_glitches"] = float(np.var(leg_glitch[lw]) / np.var(legacy[lw]))
    nz["legacy_corr_lines"] = float(np.corrcoef(leg_lines[lw], legacy[lw])[0, 1])
    nz["legacy_noise_per_V"] = float(1.4826 * np.median(np.abs(np.diff(legacy[w])) / np.sqrt(2)))
    nz["raw_noise_per_V"] = float(1.4826 * np.median(np.abs(np.diff(raw_d2[w])) / np.sqrt(2)))
    tf = transient_fit(s, Ic)
    return dict(name=s.name, s=s, Ic=Ic, tr=tr, leg_lines=leg_lines, cands=cands, feats=feats, thr=thr, null=null,
                frc=(fc, c, nb, c_raw), legacy=legacy, raw_d2=raw_d2, noise=nz, psd=spec, transient=tf)


# ------------------------------------------------------------------ benchmark with injected features

def _inject(V, G0, rng, valid, K):
    """K second-derivative features (Gaussian, random sign) placed in the valid window, and the
    current they produce (double cumulative integral)."""
    lo, hi = V[valid].min(), V[valid].max()
    feats = []
    d2 = np.zeros_like(V)
    tries = 0
    while len(feats) < K and tries < 200:
        tries += 1
        fw = np.exp(rng.uniform(np.log(0.02), np.log(0.15)))
        v0 = rng.uniform(lo + fw, hi - fw)
        if any(abs(v0 - f[0]) < 1.5 * (fw + f[1]) for f in feats):
            continue
        a = np.exp(rng.uniform(np.log(1.0), np.log(40.0))) * rng.choice([-1, 1])
        feats.append((v0, fw, a))
        d2 += a * G0 * np.exp(-0.5 * ((V - v0) / (fw / 2.3548)) ** 2)
    dv = np.mean(np.diff(V))
    d1 = np.cumsum(d2) * dv
    I = np.cumsum(d1 - d1.mean()) * dv
    return feats, d2 / G0, I - I.mean()


N_NULL = 80      # feature-free synthetic data sets per location (calibration)
N_BENCH = 160    # synthetic data sets per location in total


def _bench_one(args):
    """One synthetic data set: this location's repeat-to-repeat fluctuations, each repeat's
    sign flipped at random (which keeps interference, drift, glitches and their time structure
    but removes anything common to the repeats), added to the feature-free background of the
    location and to K injected features (K = 0 for the first N_NULL seeds)."""
    loc, seed = args
    s = [x for x in load_mos2() if x.name == loc][0]
    rng = np.random.default_rng(seed)
    R = s.R
    w = window(s)
    m = s.I.mean(0)
    base = m.copy()
    base[w] = m[w] - project_out(background_basis(s.V[w], s.t[0][w], m[w]), m[w][None])[0]
    G0 = float(np.median(np.gradient(base)[w] / s.dv))
    K = 0 if seed % N_BENCH < N_NULL else int(rng.integers(1, 4))
    truth, d2_true, I_feat = _inject(s.V, G0, rng, w, K)
    eps = rng.choice([-1.0, 1.0], R)
    noise = (s.I - m) * np.sqrt(R / (R - 1)) * eps[:, None]
    s.I = base + I_feat + noise
    out = []
    Ic = clean(s)
    fc, c, _ = fourier_ring_correlation(Ic, w, s.dv, n_split=30)
    fcut = frc_cutoff(fc, c)
    tr = make_traces(s, Ic, fcut, w)
    feats, _, _ = detect(tr, n_boot=30)
    out.append(("pipeline", tr.tstat(), tr.mean(), tr.fwhm_v, tr.valid))
    tr2 = make_traces(s, s.I, fcut, w)
    y2 = tr2.mean()
    out.append(("no-cleaning", tr2.tstat(), y2, tr2.fwhm_v, tr2.valid))
    Wb = Whittaker(len(s.V), lam_for_cutoff(3.0, s.dv), 4)
    leg = (legacy_sg2(s.I.mean(0), s.dv) - second_diff(Wb(s.I.mean(0)), s.dv)) / G0
    out.append(("legacy SG", leg / _mad(leg[tr.valid]), leg, 31 * abs(s.dv), tr.valid))
    d2_feat = d2_true - second_diff(Wb(I_feat), s.dv) / G0
    return dict(loc=loc, seed=seed, truth=truth, d2_true=d2_feat, V=s.V, results=out,
                own=[(f.V, f.amplitude) for f in feats])


def _mad(y):
    return 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-300


def _peaks_above(score, valid, thr, V):
    i = np.arange(1, len(score) - 1)
    m = (((score[i] > score[i - 1]) & (score[i] >= score[i + 1])) |
         ((score[i] < score[i - 1]) & (score[i] <= score[i + 1])))
    i = i[m & valid[i]]
    return [(V[j], score[j]) for j in i if abs(score[j]) >= thr]


def null_maxima(runs):
    """Max |score| over the window in the feature-free synthetic data sets, per location and method."""
    out = {}
    for r in runs:
        if r["truth"]:
            continue
        for meth, score, _, _, valid in r["results"]:
            out.setdefault(r["loc"], {}).setdefault(meth, []).append(float(np.max(np.abs(score[valid]))))
    return out


def score_benchmark(runs, nulls, alpha=0.05):
    """Matched-false-positive comparison. Every method detects local extrema of its own score
    (t statistic for the pipeline, robust noise units for the others) above a threshold set,
    per location, to the (1 - alpha) quantile of the score maximum in the feature-free data sets.
    Scored on the data sets with injected features only."""
    rows = []
    for r in runs:
        if not r["truth"]:
            continue
        for meth, score, y, fw, valid in r["results"]:
            thr = float(np.quantile(nulls[r["loc"]][meth], 1 - alpha))
            det = _peaks_above(score, valid, thr, r["V"])
            rmse = float(np.sqrt(np.mean((y[valid] - r["d2_true"][valid]) ** 2)))
            matched = set()
            for v0, fwt, a in r["truth"]:
                tol = max(fwt / 2, fw / 2)
                hit = [k for k, (v, sc) in enumerate(det) if abs(v - v0) <= tol and np.sign(sc) == np.sign(a)]
                rows.append(dict(loc=r["loc"], seed=r["seed"], method=meth, kind="truth", V0=v0, fwhm=fwt,
                                 amp=a, detected=bool(hit),
                                 pos_err=min(abs(det[k][0] - v0) for k in hit) if hit else np.nan, rmse=rmse))
                matched.update(hit)
            # a detection is false only if it lies outside every injected feature and its
            # side lobes: the background high-pass (BASELINE_CUTOFF) gives each feature
            # opposite-sign satellites out to ~1/(2 f_b) from its centre
            lobe = 0.5 / BASELINE_CUTOFF
            far = [v for v, sc in det if all(abs(v - v0) > fwt + fw + lobe for v0, fwt, a in r["truth"])]
            rows.append(dict(loc=r["loc"], seed=r["seed"], method=meth, kind="spectrum", V0=np.nan, fwhm=np.nan,
                             amp=np.nan, detected=bool(far), pos_err=float(len(far)),
                             rmse=rmse, n_true=len(r["truth"])))
    return pd.DataFrame(rows)


def rgo_residual():
    """Fluctuations of the averaged rGO sweeps on the bias scales of the MoS2 interference
    (periods < 40 mV), for comparison with the pickup amplitude found on MoS2."""
    from iets.data import load_device
    r, imax = [], []
    for dev in ("pure", "impure"):
        for sp in load_device(dev):
            I = sp.current[40:-10]
            r.append(np.std(I - Whittaker(len(I), _lam_for_period(32, 2), 2)(I)))
            imax.append(np.max(np.abs(I)))
    return dict(median=float(np.median(r) * 1e12), p10=float(np.percentile(r, 10) * 1e12),
                Imax_median_uA=float(np.median(imax) * 1e6))


# ------------------------------------------------------------------ main

def main():
    os.makedirs(OUT, exist_ok=True)
    sweeps = load_mos2()
    jobs = [(s.name, 1000 * i + j) for i, s in enumerate(sweeps) for j in range(N_BENCH)]
    with ProcessPoolExecutor(8) as ex:
        runs = list(ex.map(_bench_one, jobs, chunksize=4))
    nulls = null_maxima(runs)
    own_fwer = {}
    for r in runs:
        if not r["truth"]:
            own_fwer.setdefault(r["loc"], []).append(bool(r["own"]))
    res = [analyse_location(s, extra_null=nulls[s.name]["pipeline"]) for s in sweeps]
    traces = {r["name"]: r["tr"] for r in res}

    feats = [f for r in res for f in r["feats"]]
    polarity(feats, traces)
    recurrence(feats, traces)
    cols = ["location", "kind", "V", "wavenumber_cm1", "sigma_V_mV", "sigma_cm1", "fwhm_mV", "fwhm_cm1",
            "narrower_than_thermal", "amplitude_per_V", "t_since_start_s", "tstat", "p_fwer", "polarity",
            "p_polarity", "n_other_locations", "n_tested", "p_recurrence", "q_recurrence"]
    peaks = pd.DataFrame([dict(location=f.location, kind=f.kind, V=f.V, wavenumber_cm1=f.wavenumber,
                               sigma_V_mV=f.sigma_V * 1e3, sigma_cm1=f.sigma_V * EV_TO_CM1,
                               fwhm_mV=f.fwhm_V * 1e3, fwhm_cm1=f.fwhm_V * EV_TO_CM1,
                               narrower_than_thermal=thermal_flag(f.fwhm_V),
                               amplitude_per_V=f.amplitude, t_since_start_s=f.t, tstat=f.tstat,
                               p_fwer=f.p_fwer, polarity=f.polarity, p_polarity=f.p_polarity,
                               n_other_locations=f.n_locations, n_tested=f.n_tested,
                               p_recurrence=f.p_recur, q_recurrence=f.q_recur) for f in feats], columns=cols)
    peaks.sort_values(["kind", "location", "V"]).to_csv(f"{OUT}/peaks.csv", index=False, float_format="%.4g")

    cands = [f for r in res for f in r["cands"]]
    polarity(cands, traces)
    recurrence(cands, traces)
    pd.DataFrame([dict(location=f.location, kind=f.kind, V=f.V, wavenumber_cm1=f.wavenumber,
                       sigma_V_mV=f.sigma_V * 1e3, fwhm_mV=f.fwhm_V * 1e3, amplitude_per_V=f.amplitude,
                       t_since_start_s=f.t, tstat=f.tstat, p_fwer=f.p_fwer, polarity=f.polarity,
                       p_polarity=f.p_polarity, n_other_locations=f.n_locations, n_tested=f.n_tested,
                       p_recurrence=f.p_recur, q_recurrence=f.q_recur) for f in cands]
                 ).to_csv(f"{OUT}/candidates.csv", index=False, float_format="%.4g")

    noise = pd.DataFrame([r["noise"] for r in res])
    noise.to_csv(f"{OUT}/noise.csv", index=False, float_format="%.4g")
    trans = pd.DataFrame([r["transient"] for r in res])
    trans.to_csv(f"{OUT}/transient.csv", index=False, float_format="%.4g")
    frc_rows = []
    for r in res:
        fc, c, nb, c_raw = r["frc"]
        for a, b, n, d in zip(fc, c, nb, c_raw):
            frc_rows.append(dict(location=r["name"], f_per_V=a, frc=b, frc_raw=d, n_coef=n))
    pd.DataFrame(frc_rows).to_csv(f"{OUT}/frc.csv", index=False, float_format="%.4g")

    # consensus across locations, positive bias (and mirrored negative halves)
    cons = {}
    for mode in (None, "odd", "even"):
        Z, Zs, n = consensus(traces, GRID, mirror=mode)
        null = consensus_null(Zs, GRID, n_perm=1000)
        thr = float(np.quantile(null, 0.95))
        sel = GRID > 0
        cons[str(mode)] = dict(Z=Z, n=n, thr=thr, max_absZ=float(np.nanmax(np.abs(Z[sel]))),
                               V_at_max=float(GRID[sel][np.nanargmax(np.abs(Z[sel]))]),
                               p_max=float((1 + np.sum(null >= np.nanmax(np.abs(Z[sel])))) / (1 + len(null))))

    # polarity correlation (Type 2): +V half against the mirrored -V half, as a function of displacement
    t2 = [k for k in traces if k.startswith("type2")]
    lags = np.arange(-0.2, 0.2001, 0.01)

    def on_grid(tr):
        y = np.where(tr.valid, tr.mean(), np.nan)
        return np.interp(GRID, tr.V, y, left=np.nan, right=np.nan)

    def pol_r(a, b):
        out = []
        for L in lags:
            bm = np.interp(-(GRID + L), GRID, b, left=np.nan, right=np.nan)
            m = np.isfinite(a) & np.isfinite(bm) & (GRID > 0.03)
            out.append(np.corrcoef(a[m], bm[m])[0, 1] if m.sum() > 30 else np.nan)
        return np.array(out)

    Y = {k: on_grid(traces[k]) for k in t2}
    pol_same = np.array([pol_r(Y[k], Y[k]) for k in t2])
    pol_cross = np.array([pol_r(Y[a], Y[b]) for a in t2 for b in t2 if a != b])

    bench = score_benchmark(runs, nulls)
    bench.to_csv(f"{OUT}/benchmark.csv", index=False, float_format="%.4g")

    def bsum(m):
        b = bench[bench.method == m]
        t = b[b.kind == "truth"]
        sp = b[b.kind == "spectrum"]
        return dict(sensitivity=float(t.detected.mean()),
                    sens_amp_ge5=float(t[np.abs(t.amp) >= 5].detected.mean()),
                    sens_amp_ge10=float(t[np.abs(t.amp) >= 10].detected.mean()),
                    sens_amp_ge10_fwhm_ge50=float(t[(np.abs(t.amp) >= 10) & (t.fwhm >= 0.05)].detected.mean()),
                    false_per_spectrum=float(sp.pos_err.mean()),
                    spectra_with_false=float(sp.detected.mean()),
                    pos_err_mV=float(np.nanmedian(t.pos_err) * 1e3),
                    rmse_median=float(sp.rmse.median()), n_truth=int(len(t)), n_spectra=int(len(sp)))

    summary = dict(
        thermal_fwhm_mV=THERMAL_FWHM_V * 1e3, thermal_fwhm_cm1=THERMAL_FWHM_V * EV_TO_CM1,
        locations=len(res), repeats_total=int(sum(r["s"].R for r in res)),
        n_features=len(feats),
        n_features_type2=int(sum(f.kind == "type2" for f in feats)),
        n_narrower_than_thermal=int(sum(thermal_flag(f.fwhm_V) for f in feats)),
        n_recurring_q05=int(np.sum(peaks.q_recurrence < 0.05)) if len(peaks) else 0,
        n_polarity_reproduced=int(np.sum(peaks.polarity.isin(["odd", "even"]))) if len(peaks) else 0,
        n_polarity_testable=int(np.sum(peaks.polarity.isin(["odd", "even", "absent"]))) if len(peaks) else 0,
        consensus={k: {kk: vv for kk, vv in v.items() if kk not in ("Z", "n")} for k, v in cons.items()},
        polarity_lag0_same=float(np.nanmean(pol_same[:, len(lags) // 2])),
        polarity_lag0_cross_sd=float(np.nanstd(pol_cross[:, len(lags) // 2])),
        polarity_same_mean_by_lag=dict(zip(np.round(lags * 1e3).astype(int).tolist(),
                                           np.round(np.nanmean(pol_same, 0), 3).tolist())),
        polarity_cross_q99=float(np.nanquantile(np.abs(np.nanmean(pol_cross, 0)), 0.99)),
        benchmark={m: bsum(m) for m in ("pipeline", "no-cleaning", "legacy SG")},
        signflip_fwer_on_null=float(np.mean([v for vv in own_fwer.values() for v in vv])),
        signflip_fwer_by_location={k: float(np.mean(v)) for k, v in own_fwer.items()},
        transient=dict(tau1_median_s=float(trans.tau1_s.median()), tau2_median_s=float(trans.tau2_s.median()),
                       tau1_range=[float(trans.tau1_s.min()), float(trans.tau1_s.max())],
                       tau2_range=[float(trans.tau2_s.min()), float(trans.tau2_s.max())],
                       resid_over_mean_noise_median=float(np.median(
                           trans.rms_resid_pA.values / (noise.fluct_rms_pA.values / np.sqrt(noise.repeats.values))))),
        noise_median={k: float(noise[k].median()) for k in
                      ("fluct_rms_pA", "fluct_rms_raw_pA", "psd_gamma", "corr_len_mV", "resolution_mV",
                       "kernel_fwhm_mV", "trace_noise_per_V", "legacy_noise_per_V", "raw_noise_per_V")},
        mains_Hz_range=[float(noise.mains_Hz_median.min()), float(noise.mains_Hz_median.max())],
    )
    summary["rgo_residual_rms_pA"] = rgo_residual()
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    np.savez_compressed(
        f"{OUT}/spectra.npz", grid=GRID, lags=lags, pol_same=pol_same, pol_cross=pol_cross, t2=np.array(t2),
        **{f"cons_{k}": v["Z"] for k, v in cons.items()}, **{f"consn_{k}": v["n"] for k, v in cons.items()},
        names=np.array([r["name"] for r in res]),
        **{f"V__{r['name']}": r["s"].V for r in res}, **{f"t__{r['name']}": r["s"].t[0] for r in res},
        **{f"x__{r['name']}": r["tr"].x for r in res}, **{f"valid__{r['name']}": r["tr"].valid for r in res},
        **{f"legacy__{r['name']}": r["legacy"] for r in res},
        **{f"leglines__{r['name']}": r["leg_lines"] for r in res}, **{f"rawd2__{r['name']}": r["raw_d2"] for r in res},
        **{f"Iraw__{r['name']}": r["s"].I for r in res}, **{f"Iclean__{r['name']}": r["Ic"] for r in res},
        **{f"glitch__{r['name']}": r["s"].glitches for r in res},
        **{f"psd__{r['name']}": np.vstack(r["psd"]) for r in res},
        **{f"null__{r['name']}": r["null"] for r in res},
    )
    print(json.dumps(summary, indent=1, default=float))
    print(peaks.to_string())


if __name__ == "__main__":
    main()
