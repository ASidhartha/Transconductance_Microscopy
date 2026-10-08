"""Coupling coefficient alpha = (d2I/dV2) / (2 K0) before and after denoising (2 mm electrode pitch).

    .venv/bin/python alpha_analysis.py

Run after apply.py. Constants and geometry are in iets/alpha.py (taken from
matlab/alpha_calculation_updated.m). Outputs (in outputs/alpha/):
  <device>/<pair>.csv        alpha on the common (midpoint) axis: measured and denoised, both halves
  alpha_spectra_<device>.pdf original alpha (+V, as in the MATLAB script) vs denoised alpha, every pair
  peaks_alpha.csv            peaks.csv plus K0, pair length and alpha at each peak (measured and denoised)
  alpha_summary.json         per-device statistics quoted in the manuscript
  matlab_check.csv           ratio of the MATLAB Alpha column (Alpha_2mm/, if present) to the alpha here
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from iets.alpha import alpha, is_diagonal, k0, pair_length
from iets.analysis import load_model, unet
from iets.data import duplicated_pairs, read_csv
from iets.pipeline import prepare_all
from iets.preprocess import GRID, detrend

OUT = "outputs/alpha"
MATLAB = {"impure": "Alpha_2mm/Impure Alpha", "pure": "Alpha_2mm/Pure Alpha"}
C_POS, C_NEG, C_POS_L, C_NEG_L = "#2a6cb3", "#c8562a", "#a3bfe0", "#eab49c"

S, X, MASK, SC, A = prepare_all()
D = unet(load_model("models/unet_2branch.pt", 2), X) * MASK
DUP = duplicated_pairs(S)
peaks = pd.read_csv("outputs/peaks.csv")
idx_of = {(s.device, s.pair): i for i, s in enumerate(S)}


def own_axis(i):
    """Original alpha on the +V half's own axis (as the MATLAB script computes it), the denoised +V
    half mapped back onto that axis with the removed background restored, and the background."""
    s, a, sc, k = S[i], A[i], SC[i], 2 * k0(S[i].node_a, S[i].node_b)
    w = s.wavenumber
    keep = w >= 100
    orig_w, orig = w[keep], s.d2[keep] / k
    raw = np.interp(GRID, w[w > 0], s.d2[w > 0])
    bg = raw - detrend(raw)
    w_pos = a.T((2 * GRID - a.c) / (1 + a.k))          # +V axis position of each midpoint-grid point
    ok = MASK[i][0]
    den_w = w_pos[ok]
    den = (D[i][0][ok] * sc + np.interp(den_w, GRID, bg)) / k
    return orig_w, orig, den_w, den, GRID, bg / k


# ------------------------------------------------------------ per-pair alpha on the common axis
for s, x, y, sc, mk in zip(S, X, D, SC, MASK):
    os.makedirs(f"{OUT}/{s.device}", exist_ok=True)
    k = 2 * k0(s.node_a, s.node_b)
    pd.DataFrame(dict(wavenumber_mid=GRID,
                      alpha_measured_pos=np.where(mk[0], x[0] * sc / k, np.nan),
                      alpha_measured_neg=np.where(mk[1], x[1] * sc / k, np.nan),
                      alpha_denoised_pos=np.where(mk[0], y[0] * sc / k, np.nan),
                      alpha_denoised_neg=np.where(mk[1], y[1] * sc / k, np.nan))
                 ).to_csv(f"{OUT}/{s.device}/{s.pair}.csv", index=False)

# ------------------------------------------------------------ alpha at each peak
rows = []
for p in peaks.itertuples():
    i = idx_of[(p.device, p.pair)]
    s = S[i]
    k = 2 * k0(s.node_a, s.node_b)
    j = int(np.argmin(np.abs(GRID - p.wavenumber)))
    meas = np.abs(X[i][:, j]) * SC[i]
    if p.confidence == "confirmed":
        m = np.sqrt(meas[0] * meas[1])
    else:
        m = meas[0] if "+V" in p.confidence else meas[1]
    rows.append(dict(pair_length_mm=pair_length(s.node_a, s.node_b) * 10, diagonal=is_diagonal(s.node_a, s.node_b),
                     K0=k / 2, alpha_denoised=p.strength / k, alpha_measured=m / k))
pa = pd.concat([peaks, pd.DataFrame(rows)], axis=1)
pa.to_csv(f"{OUT}/peaks_alpha.csv", index=False)

# ------------------------------------------------------------ device statistics
summary = {}
for dev in ("pure", "impure"):
    idx = [i for i, s in enumerate(S) if s.device == dev and s.pair not in DUP]
    hc = pa[(pa.device == dev) & ~pa.duplicated_across_devices & ~pa.confidence.str.startswith("tentative")]
    spec = []
    for i in idx:
        ow, o = own_axis(i)[:2]
        spec.append(np.abs(o[(ow >= 200) & (ow <= 3648)]))
    spec = np.concatenate(spec)
    den_all = np.concatenate([np.abs(D[i][MASK[i]] * SC[i] / (2 * k0(S[i].node_a, S[i].node_b))) for i in idx])
    summary[dev] = dict(
        n_pairs=len(idx), n_peaks_high=len(hc),
        alpha_peak_denoised_median=float(hc.alpha_denoised.median()),
        alpha_peak_denoised_iqr=[float(v) for v in hc.alpha_denoised.quantile([0.25, 0.75])],
        alpha_peak_measured_median=float(hc.alpha_measured.median()),
        alpha_spectrum_original_median=float(np.median(spec)),
        alpha_spectrum_original_p99=float(np.percentile(spec, 99)),
        alpha_denoised_median=float(np.median(den_all)),
        alpha_max_original=float(spec.max()))
summary["ratio_impure_to_pure_peak_alpha"] = (summary["impure"]["alpha_peak_denoised_median"]
                                              / summary["pure"]["alpha_peak_denoised_median"])
summary["K0_straight"] = k0((1, 1), (1, 2))
summary["K0_diagonal"] = k0((1, 1), (2, 2))
correction = {}
for dev in ("pure", "impure"):
    hc = pa[(pa.device == dev) & ~pa.confidence.str.startswith("tentative")]
    # size of the first-order theta1 correction, 3 theta1 alpha V_DS, at the peaks (theta1 = 0.5 V^-1)
    correction[dev] = float(np.median(3 * 0.5 * hc.alpha_denoised * hc.wavenumber / 8065.54))
summary["theta1_correction_median"] = correction
json.dump(summary, open(f"{OUT}/alpha_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# ------------------------------------------------------------ check against the MATLAB output
chk = []
for dev, folder in MATLAB.items():          # skipped when the MATLAB output is not present
    for f in sorted(glob.glob(os.path.join(folder, "*_alpha_new.csv"))):
        pair = os.path.basename(f).split("_alpha")[0]
        raw = pd.read_csv(f)
        k0_m = raw.K0_used.dropna().iloc[0]
        raw = raw.dropna(subset=[raw.columns[0]])
        s = S[idx_of[(dev, pair)]]
        w = raw.iloc[:, 0].values
        ok = (w >= 200) & (w <= 3648)
        ours = alpha(read_csv(f).d2_ss.values, s.node_a, s.node_b)
        v = raw.Alpha.values[ok] / ours[ok]
        chk.append(dict(device=dev, pair=pair, K0_matlab=k0_m, K0_here=k0(s.node_a, s.node_b),
                        median_ratio=float(np.median(v)), max_ratio=float(np.max(v)),
                        matlab_dV_mV=float(np.median(np.diff(raw.iloc[:, 1].values)) * 1e3)))
chk = pd.DataFrame(chk)
if len(chk):
    chk.to_csv(f"{OUT}/matlab_check.csv", index=False)
    off = chk[(chk.median_ratio - 1).abs() > 0.01]
    print(f"MATLAB alpha vs here: K0 max rel. diff {((chk.K0_matlab / chk.K0_here) - 1).abs().max():.1e}; "
          f"{len(chk) - len(off)}/{len(chk)} files agree within 1%; differing: "
          + ", ".join(f"{r.device} {r.pair} (x{r.median_ratio:.3f}, MATLAB dV {r.matlab_dV_mV:.2f} mV)" for r in off.itertuples()))

# ------------------------------------------------------------ alpha spectra, every pair
LS = {"c": "-", "s": "--", "t": ":"}
for dev in ("impure", "pure"):
    idx = [i for i, s in enumerate(S) if s.device == dev]
    with PdfPages(f"{OUT}/alpha_spectra_{dev}.pdf") as pdf:
        for kk in range(0, len(idx), 6):
            fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)
            for ax, i in zip(axes, idx[kk:kk + 6]):
                ow, o, dw, d, gw, bg = own_axis(i)
                ax.plot(ow, o * 1e3, color=C_POS_L, lw=0.8, label="original α (+V, measured)")
                ax.plot(dw, d * 1e3, color=C_POS, lw=1.1, label="denoised α (+V, background restored)")
                ax.plot(gw, bg * 1e3, color="0.45", lw=0.7, ls="--", label="background removed before denoising")
                ax.axhline(0, color="k", lw=0.3)
                win = (ow >= 200) & (ow <= 3648)
                lim = np.abs(o[win]).max() * 1e3 * 1.1
                ax.set_ylim(-lim, lim)
                for p in peaks[(peaks.device == dev) & (peaks.pair == S[i].pair)].itertuples():
                    if p.confidence == "confirmed" or "+V" in p.confidence:
                        ax.axvline(p.wavenumber_pos_axis, color="k", lw=0.6, ls=LS[p.confidence[0]], alpha=0.5)
                s = S[i]
                ax.set_ylabel("α (10$^{-3}$)")
                ax.set_title(f"{dev} {s.pair}  L = {pair_length(s.node_a, s.node_b) * 10:.2f} mm, "
                             f"K0 = {k0(s.node_a, s.node_b):.3g} A V$^{{-2}}$", fontsize=8, loc="left")
            for ax in axes[len(idx[kk:kk + 6]):]:
                ax.axis("off")
            axes[0].legend(fontsize=7, loc="lower left", ncol=3)
            axes[-1].set_xlabel("wavenumber, +V axis (cm$^{-1}$)")
            axes[-1].set_xlim(100, 3900)
            fig.suptitle(f"{dev} device: α = (d²I/dV²)/(2K₀), original and denoised. Lines: solid confirmed, "
                         "dashed single-half (+V), dotted tentative (+V)", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
print("wrote", OUT)
