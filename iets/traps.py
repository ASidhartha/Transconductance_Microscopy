"""Charge-trap model of the repeated MoS2 sweeps.

The film is described as a conductance G in parallel with J trap branches. Each branch is a
capacitance C_j that charges through the film with time constant tau_j (an R-C branch), so
its current obeys

    tau_j di_j/dt + i_j = C_j dV/dt,

and the measured current is I(t) = G(t) V(t) + I_0 + sum_j i_j(t). G drifts linearly over the
session. The whole measurement is simulated sample by sample over every repeat in order,
including the bias jump between repeats and the time spent there. One parameter set
therefore has to explain the start-up relaxation, the zero-current offset of the up-sweep
and the differences between the first and later repeats together.

C_j is the trapped charge per volt of branch j. C_j / e is the number of trapped carriers
per volt, which is the quantity compared between locations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

E = 1.602176634e-19


@dataclass
class TrapFit:
    location: str
    J: int
    G0: float              # S, at the start of the session
    drift: float           # fractional change of G per hour
    I0: float              # A
    V_pre: float           # bias held before the first repeat (V)
    C: np.ndarray          # F, per branch
    tau: np.ndarray        # s, per branch
    rms: float             # A, residual
    n: int                 # samples used
    bic: float
    r2_relax: float        # fraction of the non-ohmic signal explained

    @property
    def carriers_per_volt(self):
        return self.C / E


def session(s):
    """Concatenate the repeats into one time series with absolute time.

    The repeats follow each other without a recorded gap. Each repeat is placed after the
    previous one with one dwell time in between, which is when the bias jumps back to the start."""
    t, V, I, rep = [], [], [], []
    t0 = 0.0
    for k in range(s.R):
        tk = s.t[k] + t0
        t.append(tk)
        V.append(s.V)
        I.append(s.I[k])
        rep.append(np.full(len(s.V), k))
        t0 = tk[-1] + s.dt
    return np.concatenate(t), np.concatenate(V), np.concatenate(I), np.concatenate(rep)


def branch_currents(t, V, C, tau, V_pre):
    """Summed trap-branch current at the sample times.

    Between samples the bias is taken to change linearly; a branch then evolves exactly as
    i <- i e^{-d/tau} + C (dV/d)(1 - e^{-d/tau}). The bias jump between repeats is treated
    as a step at the start of the interval (i += C dV / tau), followed by decay. Before the
    first sample the device rests at V_pre and the bias then steps to V[0]. The dwell d is
    constant to 0.1%, so its mean is used and the recursion becomes a first-order filter."""
    from scipy.signal import lfilter
    d = float(np.mean(np.diff(t)))
    dV = np.diff(V)
    jump = np.abs(dV) > 0.05
    total = np.zeros(len(t))
    for c, ta in zip(C, tau):
        e = np.exp(-d / ta)
        u = np.where(jump, c * dV / ta * e, c * dV / d * (1 - e))
        total += lfilter([1.0], [1.0, -e], np.r_[c * (V[0] - V_pre) / ta, u])
    return total


def fit(s, J=2, mask=None, tau_init=(1.5, 18.0, 150.0), use=None):
    """Least-squares fit of the trap model to all repeats of a location (glitches masked)."""
    t, V, I, rep = session(s)
    ok = np.ones_like(I, bool) if mask is None else ~mask.ravel()
    if use is not None:  # fit only these repeats (the simulation still runs over the whole session)
        ok = ok & np.isin(rep, use)
    scale = 1e-12

    def unpack(p):
        G0, drift, I0, V_pre = p[0] * 1e-9, p[1], p[2] * scale, p[3]
        C = np.exp(p[4:4 + J]) * 1e-12
        tau = np.exp(p[4 + J:4 + 2 * J])
        return G0, drift, I0, V_pre, C, tau

    def res(p):
        G0, drift, I0, V_pre, C, tau = unpack(p)
        model = G0 * (1 + drift * t / 3600) * V + I0 + branch_currents(t, V, C, tau, V_pre)
        return ((model - I) / scale)[ok]

    late = np.concatenate([s.t[k] > 60 for k in range(s.R)])
    g, c = np.polyfit(V[late], I[late], 1)
    best = None
    for c_init in (100.0, 1000.0, 10000.0):
        p0 = np.r_[g * 1e9, 0.0, c / scale, 0.0, np.log(np.full(J, c_init)), np.log(tau_init[:J])]
        lo = np.r_[-np.inf, -1, -np.inf, -0.5, np.full(J, np.log(1e-2)), np.log([0.2, 3, 30][:J])]
        hi = np.r_[np.inf, 1, np.inf, 0.5, np.full(J, np.log(1e7)), np.log([5, 100, 1000][:J])]
        try:
            r = least_squares(res, np.clip(p0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), x_scale="jac",
                              max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    G0, drift, I0, V_pre, C, tau = unpack(p)
    o = np.argsort(tau)
    C, tau = C[o], tau[o]
    rr = res(p) * scale
    n = int(ok.sum())
    k = 4 + 2 * J
    bic = n * np.log(np.mean(rr ** 2)) + k * np.log(n)
    nonohm = (I - (G0 * (1 + drift * t / 3600) * V + I0))[ok]
    r2 = 1 - np.sum(rr ** 2) / np.sum((nonohm - nonohm.mean()) ** 2)
    return TrapFit(s.name, J, G0, drift, I0, V_pre, C, tau, float(np.sqrt(np.mean(rr ** 2))), n,
                   float(bic), float(r2)), (t, V, I, rep, ok, G0 * (1 + drift * t / 3600) * V + I0
                                            + branch_currents(t, V, C, tau, V_pre))
