"""Do the mirrored negative-bias (-V) readings carry the same spectral information as +V?

Tests, on both the raw second derivative (DerivativeY2_2) and the normalised signal
alpha = (d2I/dV2)/(dI/dV) (the quantity in the 'Alpha' column, which is only filled for V > 0):

  1. Cross-correlation between the +V branch and the mirrored -V branch of the SAME electrode
     pair, as a function of a rigid wavenumber shift, averaged over all pairs of a device,
     compared with the same quantity for RANDOMLY paired (unrelated) spectra.
     Same-pair >> random-pair  ->  the -V branch contains the +V features (shared information).
     Peak away from zero shift  ->  the two branches disagree on feature positions, i.e. a
     +V-only analysis carries an undetectable systematic position error.
  2. Bootstrap confidence interval of the device-level shift.
  3. Symmetry: sign of the same-pair correlation at that shift (even vs odd in V).

    .venv/bin/python polarity_check.py      # -> outputs/polarity/
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iets.data import DEVICES, read_csv
from iets.preprocess import GRID, detrend

OUT = "outputs/polarity"
SHIFTS = np.arange(-600, 601, 8.0)
CM_PER_V = 8065.54
C_POS, C_NEG, C_NULL = "#2a6cb3", "#c8562a", "0.55"
DEV_COL = {"impure": C_NEG, "pure": C_POS}
rng = np.random.default_rng(0)


def load(device):
    out = {}
    for f in sorted(glob.glob(os.path.join(DEVICES[device], "*.csv"))):
        pair = os.path.basename(f).split("_alpha")[0]
        df = read_csv(f)
        w, d2, d1 = df.wavenumber.values, df.d2_ss.values, df.d1_s.values
        a = d2 / d1                                    # normalised IETS signal, both polarities
        res = {}
        for name, y in (("d2", d2), ("alpha", a)):
            pos = np.interp(GRID, w[w > 0], y[w > 0])
            neg = np.interp(GRID, -w[w < 0][::-1], y[w < 0][::-1])   # mirrored, sign NOT flipped
            res[name] = (detrend(pos), detrend(neg))
        out[pair] = res
    return out


WINDOWS = {"full": (200, 3648), "low": (200, 800), "high": (800, 3648)}


def xcorr(p, m, win=(200, 3648)):
    """r(c) = corr(p(w + c), m(w)) over -V positions w in `win`: positive c means +V features
    sit c cm-1 above the -V ones."""
    r = np.empty(len(SHIFTS))
    inwin = (GRID >= win[0]) & (GRID <= win[1])
    for k, c in enumerate(SHIFTS):
        pw = np.interp(GRID + c, GRID, p, left=np.nan, right=np.nan)
        ok = ~np.isnan(pw) & inwin
        r[k] = np.corrcoef(pw[ok], m[ok])[0, 1] if ok.sum() >= 40 else np.nan   # >= 320 cm-1 overlap
    return r


def analyse(B, win):
    same = np.array([xcorr(p, m, win) for p, m in B])
    ij = [(i, j) for i, j in rng.integers(0, len(B), (600, 2)) if i != j]
    null = np.array([xcorr(B[i][0], B[j][1], win) for i, j in ij])
    # null distribution of an N-spectrum average: random-pair curves resampled in groups of N
    grp = np.array([null[rng.integers(0, len(null), len(B))].mean(0) for _ in range(2000)])
    ms = same.mean(0)
    j, j0 = int(np.nanargmax(np.abs(ms))), int(np.argmin(np.abs(SHIFTS)))
    # look-elsewhere-corrected chance level: the largest |mean r| a random group reaches at ANY shift
    gmax = np.nanmax(np.abs(grp), 1)
    boot = [SHIFTS[np.nanargmax(np.abs(same[rng.integers(0, len(B), len(B))].mean(0)))] for _ in range(1000)]
    best_each = SHIFTS[np.nanargmax(np.abs(np.nan_to_num(same)), axis=1)]
    res = dict(n=len(B), window=list(win), best_shift_cm=float(SHIFTS[j]),
               best_shift_ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
               best_shift_mV=float(SHIFTS[j] / CM_PER_V * 1e3), plusV_only_bias_cm=float(SHIFTS[j] / 2),
               mean_r_at_shift=float(ms[j]), mean_r_at_zero=float(ms[j0]),
               chance_max_abs_r_99=float(np.percentile(gmax, 99)),
               p_value=float((np.sum(gmax >= abs(ms[j])) + 1) / (len(gmax) + 1)),
               z_vs_null=float((ms[j] - grp[:, j].mean()) / grp[:, j].std()),
               z_at_zero=float((ms[j0] - grp[:, j0].mean()) / grp[:, j0].std()),
               frac_even=float(np.mean(same[:, j] > 0)),
               per_spectrum_shift_median=float(np.median(best_each)),
               per_spectrum_shift_iqr=[float(x) for x in np.percentile(best_each, [25, 75])])
    lo, hi = np.nanpercentile(grp, [0.5, 99.5], axis=0)
    return res, (ms, lo, hi, best_each, same)


os.makedirs(OUT, exist_ok=True)
summary, curves = {}, {}
for dev in DEVICES:
    data = load(dev)
    summary[dev] = {}
    for q in ("d2", "alpha"):
        B = [data[p][q] for p in data]
        for wname, win in WINDOWS.items():
            res, cv = analyse(B, win)
            summary[dev][f"{q}_{wname}"] = res
            curves[(dev, q, wname)] = cv
            print(f"{dev:6s} {q:5s} {wname:4s} shift {res['best_shift_cm']:+5.0f} cm-1 "
                  f"(95% CI {res['best_shift_ci95'][0]:+.0f}..{res['best_shift_ci95'][1]:+.0f}) = "
                  f"{res['best_shift_mV']:5.1f} mV; mean r {res['mean_r_at_shift']:+.3f} "
                  f"(chance max |r| {res['chance_max_abs_r_99']:.3f}, p={res['p_value']:.4f}); "
                  f"r at 0 {res['mean_r_at_zero']:+.3f} (z={res['z_at_zero']:+.1f}); even {100 * res['frac_even']:.0f}%")
json.dump(summary, open(f"{OUT}/polarity_summary.json", "w"), indent=2)
np.savez_compressed(f"{OUT}/polarity_curves.npz", shifts=SHIFTS, **{
    f"{d}_{q}_{w}_{part}": v for (d, q, w), cv in curves.items()
    for part, v in zip(("mean", "lo", "hi", "best_each"), cv[:4])})

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "sans-serif"})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))

# (a) example: alpha on +V and mirrored -V, impure device
ex_dev, ex_q = "impure", "alpha"
ms, lo, hi, best_each, same = curves[(ex_dev, ex_q, "full")]
shift = summary[ex_dev][f"{ex_q}_full"]["best_shift_cm"]
k = int(np.argmax(same[:, int(np.argmin(np.abs(SHIFTS - shift)))]))
pair = list(load(ex_dev))[k]
p, m = load(ex_dev)[pair][ex_q]
sc = 1.4826 * np.median(np.abs(np.concatenate([p, m])))
ax = axes[0]
ax.plot(GRID, p / sc, color=C_POS, lw=0.9, label="+V")
ax.plot(GRID, m / sc, color=C_NEG, lw=0.9, label="−V (mirrored)")
ax.plot(GRID, np.interp(GRID - shift, GRID, m, left=np.nan, right=np.nan) / sc - 12, color=C_NEG, lw=0.9, ls="-")
ax.plot(GRID, p / sc - 12, color=C_POS, lw=0.9)
ax.text(GRID[-1], 6.5, "as measured", ha="right", fontsize=7)
ax.text(GRID[-1], -12 + 6.5, f"−V shifted by {shift:+.0f} cm$^{{-1}}$", ha="right", fontsize=7)
ax.set_xlabel("|eV| expressed as wavenumber (cm$^{-1}$)")
ax.set_ylabel("α (noise units)")
ax.set_yticks([])
ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
ax.set_title(f"a  {ex_dev} device, pair {pair}", loc="left", fontsize=8, fontweight="bold")

# (b) device-averaged cross-correlation vs shift
ax = axes[1]
for dev in DEVICES:
    ms, lo, hi, *_ = curves[(dev, "alpha", "full")]
    ax.fill_between(SHIFTS, lo, hi, color=DEV_COL[dev], alpha=0.12, lw=0)
    ax.plot(SHIFTS, ms, color=DEV_COL[dev], lw=1.6, label=f"{dev}")
    ax.plot(SHIFTS, curves[(dev, "alpha", "high")][0], color=DEV_COL[dev], lw=0.9, ls="--",
            label=f"{dev}, >800 cm$^{{-1}}$ only")
    s = summary[dev]["alpha_full"]
    ax.annotate(f"{s['best_shift_cm']:+.0f}", (s["best_shift_cm"], s["mean_r_at_shift"]), xytext=(14 if dev == "impure" else -14, 3),
                textcoords="offset points", ha="left" if dev == "impure" else "right", fontsize=7, color="0.15")
ax.axvline(0, color="0.3", lw=0.5, ls=":")
ax.axhline(0, color="0.3", lw=0.4)
ax.set_xlabel("shift of −V relative to +V (cm$^{-1}$)")
ax.set_ylabel("mean ±V correlation of α")
ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
ax.set_title("b  same pair vs chance (band)", loc="left", fontsize=8, fontweight="bold")

# (c) per-spectrum best shift
ax = axes[2]
bins = np.arange(-600, 601, 50)
for dev in DEVICES:
    ax.hist(curves[(dev, "alpha", "full")][3], bins, color=DEV_COL[dev], alpha=0.55, label=dev, edgecolor="white", lw=0.5)
ax.axvline(0, color="0.3", lw=0.5, ls=":")
ax.set_xlabel("best shift per spectrum (cm$^{-1}$)")
ax.set_ylabel("electrode pairs")
ax.legend(fontsize=7, frameon=False)
ax.set_title("c  per electrode pair", loc="left", fontsize=8, fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_polarity.{ext}")
print("wrote", OUT)
