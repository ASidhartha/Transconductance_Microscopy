"""Denoise every spectrum, pick peaks, map them to bonds, and draw bond heatmaps.

    .venv/bin/python apply.py

Outputs (in outputs/):
  denoised/<device>/<pair>.csv   raw and denoised branches on the common (midpoint) axis
  peaks.csv                      every peak with confidence tier, position uncertainty, candidate bonds
  spectra_<device>.pdf           raw vs denoised for every electrode pair
  heatmaps_<device>.pdf          spatial maps for the bands in iets/bands.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata

from iets.analysis import assign, classify_peaks, load_model, unet
from iets.bands import BANDS
from iets.data import duplicated_pairs
from iets.pipeline import prepare_all
from iets.preprocess import GRID

C_POS, C_NEG, C_POS_L, C_NEG_L = "#2a6cb3", "#c8562a", "#a3bfe0", "#eab49c"

tiers = json.load(open("outputs/metrics.json"))["synthetic"]["tiers"]
THR, SINGLE_THR, TENT_THR = tiers["threshold"], tiers["single_threshold"], tiers["tentative_threshold"]
model = load_model("models/unet_2branch.pt", 2)

S, X, MASK, SC, A = prepare_all()
D = unet(model, X) * MASK          # nothing is reported where a branch has no data
DUP = duplicated_pairs(S)

rows = []
for s, x, y, sc, a, mk in zip(S, X, D, SC, A, MASK):
    os.makedirs(f"outputs/denoised/{s.device}", exist_ok=True)
    w_neg = (2 * GRID - a.c) / (1 + a.k)
    pd.DataFrame(dict(wavenumber_mid=GRID, wavenumber_pos_axis=a.T(w_neg), wavenumber_neg_axis=w_neg,
                      raw_pos=np.where(mk[0], x[0] * sc, np.nan), raw_neg=np.where(mk[1], x[1] * sc, np.nan),
                      denoised_pos=np.where(mk[0], y[0] * sc, np.nan),
                      denoised_neg=np.where(mk[1], y[1] * sc, np.nan))
                 ).to_csv(f"outputs/denoised/{s.device}/{s.pair}.csv", index=False)
    mx, my = s.midpoint
    for p in classify_peaks(y, THR, SINGLE_THR, TENT_THR):
        u = p["wavenumber"]
        unc = float(a.uncertainty(u)) if a.accepted else np.nan
        wn = (2 * u - a.c) / (1 + a.k)
        rows.append(dict(device=s.device, pair=s.pair, x=mx, y=my, wavenumber=round(u, 1),
                         wavenumber_pos_axis=round(float(a.T(wn)), 1), wavenumber_neg_axis=round(float(wn), 1),
                         position_uncertainty=round(unc, 1) if a.accepted else np.nan,
                         aligned=a.accepted, confidence=p["confidence"], symmetry=p["symmetry"],
                         strength=p["strength"] * sc, snr=round(p["strength"], 2),
                         duplicated_across_devices=s.pair in DUP,
                         candidate_bonds=assign(u, 0.0 if not a.accepted else unc)))
peaks = pd.DataFrame(rows)
peaks.to_csv("outputs/peaks.csv", index=False)
print(f"{len(peaks)} peaks; alignment accepted for {sum(a.accepted for a in A)}/{len(A)} spectra")
print(peaks.groupby(["device", "confidence"]).size().unstack(fill_value=0))


# ------------------------------------------------------------ per-spectrum plots
def plot_pair(axl, axr, i):
    s, x, y, sc, a, mk = S[i], X[i], D[i], SC[i] * 1e9, A[i], MASK[i]
    for ax, Z, cols, lab in ((axl, x, (C_POS_L, C_NEG_L), "raw"), (axr, y, (C_POS, C_NEG), "denoised")):
        ax.plot(GRID, np.where(mk[0], Z[0] * sc, np.nan), color=cols[0], lw=1, label=f"{lab} +V")
        ax.plot(GRID, np.where(mk[1], Z[1] * sc, np.nan), color=cols[1], lw=1, label=f"{lab} −V (mirrored)")
        ax.axhline(0, color="k", lw=0.3)
    lim = np.nanmax(np.abs(x[mk])) * sc * 1.1
    for ax in (axl, axr):
        ax.set_ylim(-lim, lim)
    for p in peaks[(peaks.device == s.device) & (peaks.pair == s.pair)].itertuples():
        ls = {"c": "-", "s": "--", "t": ":"}[p.confidence[0]]
        for ax in (axl, axr):
            ax.axvline(p.wavenumber, color="k", lw=0.7, ls=ls, alpha=0.6 if ls == "-" else 0.45)
        axr.text(p.wavenumber, lim * 0.97, f" {p.wavenumber:.0f}", fontsize=6.5, va="top", rotation=90)
    note = (f"aligned: −V ≈ ±(+V)({a.k:.2f}·w{a.c:+.0f}), r={abs(a.r):.2f}" if a.accepted
            else "halves could not be aligned — shown as measured")
    dup = "  [nearly identical data in both device folders]" if s.pair in DUP else ""
    axl.set_ylabel(f"{s.pair}\nnA/V²", fontsize=8)
    axl.set_title(f"{s.device} {s.pair} — {note}{dup}", fontsize=7.5, loc="left")


for dev in ("impure", "pure"):
    idx = [i for i, s in enumerate(S) if s.device == dev]
    with PdfPages(f"outputs/spectra_{dev}.pdf") as pdf:
        for k in range(0, len(idx), 5):
            fig, axes = plt.subplots(5, 2, figsize=(13, 14), sharex=True)
            for row, i in zip(axes, idx[k:k + 5]):
                plot_pair(row[0], row[1], i)
            for row in axes[len(idx[k:k + 5]):]:
                row[0].axis("off")
                row[1].axis("off")
            axes[0, 0].legend(fontsize=6.5, loc="lower left")
            axes[0, 1].legend(fontsize=6.5, loc="lower left")
            for ax in axes[-1]:
                ax.set_xlabel("Wavenumber, common axis (cm$^{-1}$)")
            fig.suptitle(f"{dev} device — left: raw (before), right: denoised (after).  "
                         "Lines: solid = confirmed in both halves, dashed = strong single-half, dotted = tentative", fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

# ------------------------------------------------------------ bond heatmaps
xi, yi = np.meshgrid(np.linspace(1, 5, 120), np.linspace(1, 4, 100))
for dev in ("impure", "pure"):
    idx = [i for i, s in enumerate(S) if s.device == dev]
    mid = np.array([S[i].midpoint for i in idx])
    with PdfPages(f"outputs/heatmaps_{dev}.pdf") as pdf:
        for name, (lo, hi) in BANDS.items():
            m = (GRID >= lo) & (GRID <= hi)
            vals = np.array([np.max(np.abs(D[i][:, m]).mean(0)) * SC[i] for i in idx]) * 1e9
            sub = peaks[(peaks.device == dev) & peaks.wavenumber.between(lo, hi)]
            fig, ax = plt.subplots(figsize=(7, 6))
            zi = griddata(mid, vals, (xi, yi), method="cubic")
            cf = ax.contourf(xi, yi, np.clip(zi, 0, None), 20, cmap="magma_r")
            fig.colorbar(cf, label="denoised |d²I/dV²| in band, mean of ±V (nA/V²)")
            for i, (px, py) in zip(idx, mid):
                t = sub[sub.pair == S[i].pair].confidence
                mk = "o" if (t == "confirmed").any() else ("s" if t.str.startswith("single").any() else ".")
                ax.plot(px, py, mk, color="k" if mk != "." else "0.5", ms=7 if mk != "." else 3,
                        mfc="none" if mk == "s" else None)
            ax.set_title(f"{dev}: {name} ({lo}–{hi} cm$^{{-1}}$)\n● confirmed peak in band   □ single-half peak",
                         fontsize=10)
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            pdf.savefig(fig)
            plt.close(fig)
print("wrote outputs/")
