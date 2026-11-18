"""Upper limits on a feature common to the whole MoS2 film (pooled over locations).

    .venv/bin/python mos2_pooled.py        (after mos2_analysis.py; ~1 min)

For a trial feature, a Gaussian in (d2I/dV2)/(dI/dV) of width w centred at V0, the
template current is passed through exactly the processing each location's data received
(apply_operator). Each repeat's amplitude is its least-squares projection onto the processed
template. Amplitudes are combined over the repeats of a location (mean, standard error) and over
locations by a random-effects model. The family-wise null is built by flipping the signs of
whole repeats independently in every location. Writes outputs/mos2/pooled.csv and
pooled_summary.json.
"""
import json

import numpy as np
import pandas as pd

from iets.features import apply_operator, make_traces
from iets.sweeps import clean, fourier_ring_correlation, frc_cutoff, load_mos2
from mos2_analysis import window

OUT = "outputs/mos2"
WIDTHS = (0.03, 0.06, 0.14)          # V; 0.14 V is the room-temperature inelastic line width
GRID = np.arange(-0.25, 0.4501, 0.005)  # V0 over the window covered by at least the Type-2 locations
N_PERM = 2000
# main region: at least 40 s into the Type-2 sweeps (the relaxation has decayed below the noise);
# start region: 20-40 s, where residual time-locked background cannot be excluded
REGIONS = {"main": (-0.10, 0.45), "start": (-0.25, -0.10)}


def templates(s, w):
    """Template currents for unit amplitude (1 V^-1 of dI/dV) at every V0 of GRID."""
    dv = s.dv
    G0 = 1.0  # amplitudes are in units of dI/dV, so the template is built per unit conductance
    T = []
    for v0 in GRID:
        d2 = np.exp(-0.5 * ((s.V - v0) / (w / 2.3548)) ** 2)
        d1 = np.cumsum(d2) * dv
        T.append(np.cumsum(d1 - d1.mean()) * dv)
    return np.array(T) * G0


def location_amplitudes(s, Ic, fcut, valid, w):
    tr = make_traces(s, Ic, fcut, valid)
    # the operator divides by the location's own dI/dV; multiply the template by it so that
    # amplitudes come out in units of dI/dV
    T = apply_operator(s, templates(s, w) * tr.G0, Ic, fcut, valid)
    x = tr.x[:, valid]
    Tv = T[:, valid]
    norm = np.sum(Tv ** 2, 1)
    ok = (GRID >= s.V[valid].min() + w) & (GRID <= s.V[valid].max() - w) & (norm > 0)
    A = np.full((tr.R, len(GRID)), np.nan)
    A[:, ok] = (x @ Tv[ok].T) / norm[ok]
    return A


def combine(A_list, signs=None):
    """Random-effects (DerSimonian-Laird) estimate and standard error at every V0."""
    m, se = [], []
    for k, A in enumerate(A_list):
        a = A if signs is None else A * signs[k][:, None]
        m.append(np.nanmean(a, 0))
        se.append(np.nanstd(a, 0, ddof=1) / np.sqrt(a.shape[0]))
    m, se = np.array(m), np.array(se)
    w = 1 / se ** 2
    ok = np.isfinite(m)
    w = np.where(ok, w, 0)
    mf = np.nansum(w * np.nan_to_num(m), 0) / np.sum(w, 0)
    Q = np.nansum(w * (np.nan_to_num(m) - mf) ** 2, 0)
    k = ok.sum(0)
    c = np.sum(w, 0) - np.sum(w ** 2, 0) / np.sum(w, 0)
    tau2 = np.maximum(0, (Q - (k - 1)) / np.maximum(c, 1e-300))
    ws = np.where(ok, 1 / (se ** 2 + tau2), 0)
    mu = np.nansum(ws * np.nan_to_num(m), 0) / np.sum(ws, 0)
    s_mu = 1 / np.sqrt(np.sum(ws, 0))
    return mu, s_mu, k


def main():
    rng = np.random.default_rng(5)
    sweeps = load_mos2()
    prep = []
    for s in sweeps:
        Ic = clean(s)
        valid = window(s)
        fc, c, _ = fourier_ring_correlation(Ic, valid, s.dv)
        prep.append((s, Ic, frc_cutoff(fc, c), valid))
    rows, summ = [], {}
    for w in WIDTHS:
        A_list = [location_amplitudes(s, Ic, fcut, valid, w) for s, Ic, fcut, valid in prep]
        mu, s_mu, k = combine(A_list)
        z = mu / s_mu
        ul = np.abs(mu) + 1.96 * s_mu
        for v, a, b, n_, u in zip(GRID, mu, s_mu, k, ul):
            rows.append(dict(width_mV=w * 1e3, V0=v, amplitude=a, se=b, n_locations=int(n_), upper95=u))
        out = {}
        for name, reg in REGIONS.items():
            sel = (GRID >= reg[0]) & (GRID <= reg[1]) & (k >= 3)
            null = []
            for _ in range(N_PERM):
                sg = [rng.choice([-1.0, 1.0], A.shape[0]) for A in A_list]
                m0, s0, _ = combine(A_list, sg)
                null.append(np.nanmax(np.abs(m0 / s0)[sel]))
            null = np.array(null)
            zmax = float(np.nanmax(np.abs(z[sel])))
            out[name] = dict(max_abs_z=zmax, V_at_max=float(GRID[sel][np.nanargmax(np.abs(z[sel]))]),
                             z_threshold_fwer05=float(np.quantile(null, 0.95)),
                             p_fwer=float((1 + np.sum(null >= zmax)) / (1 + N_PERM)),
                             upper95_median=float(np.nanmedian(ul[sel])),
                             upper95_p90=float(np.nanpercentile(ul[sel], 90)),
                             locations_median=float(np.median(k[sel])))
        summ[f"{w * 1e3:.0f}mV"] = out
    pd.DataFrame(rows).to_csv(f"{OUT}/pooled.csv", index=False, float_format="%.4g")
    json.dump(summ, open(f"{OUT}/pooled_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
