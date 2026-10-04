"""Main-text and Supplementary figures for the manuscript (outputs/paper_figures/).

    .venv/bin/python make_paper_figures.py

Run after train.py, evaluate.py, apply.py, make_report_figures.py and polarity_check.py.
"""
from __future__ import annotations

import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import binomtest

from iets.analysis import load_model, savgol, unet
from iets.data import duplicated_pairs, read_csv, DEVICES
from iets.pipeline import prepare_all
from iets.preprocess import GRID, detrend

FIG = "outputs/paper_figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "font.family": "sans-serif", "pdf.fonttype": 42})
C_POS, C_NEG, C_POS_L, C_NEG_L = "#2a6cb3", "#c8562a", "#a3bfe0", "#eab49c"
DEV_COL = {"impure": C_NEG, "pure": C_POS}
GOLD, GREY = "#a07d12", "0.55"
CM_PER_V = 8065.54


def panel(ax, letter, title=""):
    ax.set_title(f"$\\bf{{{letter}}}$  {title}", loc="left")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=200)
    plt.close(fig)


S, X, MASK, SC, A = prepare_all()
model2 = load_model("models/unet_2branch.pt", 2)
D = unet(model2, X) * MASK
peaks = pd.read_csv("outputs/peaks.csv")
peaks["tier"] = peaks.confidence.str.split(" ").str[0]
metrics = json.load(open("outputs/metrics.json"))
pol = json.load(open("outputs/polarity/polarity_summary.json"))
pc = np.load("outputs/polarity/polarity_curves.npz")
DUP = duplicated_pairs(S)
idx_of = {(s.device, s.pair): i for i, s in enumerate(S)}

# =============================================================== Figure 1: both polarities
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw=dict(hspace=0.62, wspace=0.3))
ex_dev = "pure"
shift = pol[ex_dev]["alpha_full"]["best_shift_cm"]
# example: the pure-device pair whose alpha correlates best at the device-level shift
best, ex_pair = -2, None
for s in S:
    if s.device != ex_dev:
        continue
    df = read_csv(os.path.join(DEVICES[ex_dev], f"{s.pair}_alpha_new.csv"))
    w, a = df.wavenumber.values, (df.d2_ss / df.d1_s).values
    p = detrend(np.interp(GRID, w[w > 0], a[w > 0]))
    m = detrend(np.interp(GRID, -w[w < 0][::-1], a[w < 0][::-1]))
    pw = np.interp(GRID + shift, GRID, p, left=np.nan, right=np.nan)
    ok = ~np.isnan(pw)
    r = np.corrcoef(pw[ok], m[ok])[0, 1]
    if r > best:
        best, ex_pair, ex = r, s.pair, (df, p, m)
df, p, m = ex
ax = axes[0, 0]
ax.plot(df.voltage, df.d2_ss * 1e9, color="0.2", lw=0.8)
for lo, hi in ((-0.6, -3648 / CM_PER_V), (3648 / CM_PER_V, 0.6), (-200 / CM_PER_V, 200 / CM_PER_V)):
    ax.axvspan(lo, hi, color="0.9", lw=0)
ax.axhline(0, color="0.3", lw=0.4)
ax.set_xlim(-0.5, 0.5)
win = (np.abs(df.voltage) > 200 / CM_PER_V) & (np.abs(df.voltage) < 3648 / CM_PER_V)
yl = np.abs(df.d2_ss[win]).max() * 1e9 * 1.25
ax.set_ylim(-yl, yl)
ax.set_xlabel("bias V (V)")
ax.set_ylabel("d$^2$I/dV$^2$ (nA V$^{-2}$)")
ax.text(-0.25, 0.03, "−V half", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=C_NEG)
ax.text(0.25, 0.03, "+V half", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=C_POS)
sec = ax.secondary_xaxis("top", functions=(lambda v: v * CM_PER_V / 1000, lambda k: k * 1000 / CM_PER_V))
sec.set_xlabel("eV as wavenumber (10$^3$ cm$^{-1}$)", fontsize=6.5)
sec.tick_params(labelsize=6)
panel(ax, "a", f"one full sweep ({ex_dev} device, pair {ex_pair})")

ax = axes[0, 1]
sc = 1.4826 * np.median(np.abs(np.concatenate([p, m])))
off = 9
ax.plot(GRID, p / sc + off, color=C_POS, lw=0.8, label="+V")
ax.plot(GRID, m / sc + off, color=C_NEG, lw=0.8, label="−V, mirrored")
ax.plot(GRID, p / sc - off, color=C_POS, lw=0.8)
ax.plot(GRID + shift, m / sc - off, color=C_NEG, lw=0.8)
ax.text(3650, off + 3.2, "as measured", ha="right", fontsize=6.5)
ax.text(3650, -off + 3.2, f"−V displaced by {shift:+.0f} cm$^{{-1}}$", ha="right", fontsize=6.5)
ax.set_yticks([])
ax.set_xlabel("|eV| as wavenumber (cm$^{-1}$)")
ax.set_ylabel("normalised signal (d$^2$I/dV$^2$)/(dI/dV)")
ax.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1, -0.02))
ax.set_ylim(-off - 8, off + 9)
panel(ax, "b", "positive vs mirrored negative bias")

ax = axes[1, 0]
sh = pc["shifts"]
for dev in ("pure", "impure"):
    ax.fill_between(sh, pc[f"{dev}_alpha_full_lo"], pc[f"{dev}_alpha_full_hi"], color=DEV_COL[dev], alpha=0.13, lw=0)
    ax.plot(sh, pc[f"{dev}_alpha_full_mean"], color=DEV_COL[dev], lw=1.4, label=f"{dev} device")
    ax.plot(sh, pc[f"{dev}_alpha_high_mean"], color=DEV_COL[dev], lw=0.8, ls="--")
    s_ = pol[dev]["alpha_full"]
    ax.annotate(f"{s_['best_shift_cm']:+.0f} cm$^{{-1}}$", (s_["best_shift_cm"], s_["mean_r_at_shift"]),
                xytext=(12 if dev == "impure" else -12, 2), textcoords="offset points",
                ha="left" if dev == "impure" else "right", fontsize=6.5)
ax.plot([], [], color="0.3", lw=0.8, ls="--", label="> 800 cm$^{-1}$ only")
ax.fill_between([], [], [], color="0.6", alpha=0.3, label="99% chance band")
ax.axvline(0, color="0.3", lw=0.5, ls=":")
ax.axhline(0, color="0.3", lw=0.4)
ax.set_xlabel("displacement of −V relative to +V (cm$^{-1}$)")
ax.set_ylabel("mean ±V correlation (55 pairs)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6)
panel(ax, "c", "same electrode pair vs unrelated pairs")

ax = axes[1, 1]
bins = np.arange(-600, 601, 50)
for dev in ("pure", "impure"):
    ax.hist(pc[f"{dev}_alpha_full_best_each"], bins, color=DEV_COL[dev], alpha=0.55, label=f"{dev} device",
            edgecolor="white", lw=0.4)
ax.axvline(0, color="0.3", lw=0.5, ls=":")
ax.set_xlabel("best displacement per electrode pair (cm$^{-1}$)")
ax.set_ylabel("electrode pairs")
ax.legend(frameon=False, loc="upper left")
panel(ax, "d", "per electrode pair")
save(fig, "fig_polarity")

# =============================================================== Figure 2: benchmark
syn = np.load("models/synth_test.npz")
Xs, Ys = syn["X"][1000:1400], syn["Y"][1000:1400]
Ds = unet(model2, Xs)
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.6, wspace=0.28)
ax = fig.add_subplot(gs[0, :])
k = 3
ax.plot(GRID, Xs[k, 0], color=GREY, lw=0.7, label="noisy input (one half)")
ax.plot(GRID, savgol(Xs[k:k + 1, 0])[0], color=GOLD, lw=0.9, label="Savitzky–Golay")
ax.plot(GRID, Ys[k, 0], color="k", lw=2.4, alpha=0.25, label="ground truth")
ax.plot(GRID, Ds[k, 0], color=C_POS, lw=1.0, label="U-Net (both halves)")
ax.axhline(0, color="0.3", lw=0.4)
ax.set_xlabel("wavenumber (cm$^{-1}$)")
ax.set_ylabel("signal (noise units)")
ax.legend(frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.08))
panel(ax, "a", "held-out simulated spectrum")

names = ["raw", "savitzky-golay", "gaussian", "branch-average", "unet-1branch", "unet-2branch"]
labels = ["raw", "Savitzky\n–Golay", "Gauss-\nian", "±V\naverage", "U-Net\n(1 half)", "U-Net\n(both)"]
ax = fig.add_subplot(gs[1, 0])
f1s = [metrics["synthetic"][n]["f1"] for n in names]
b = ax.bar(labels, f1s, color=["0.72"] * 4 + [C_POS_L, C_POS], width=0.7)
ax.bar_label(b, fmt="%.2f", fontsize=6)
ax.set_ylim(0, 1)
ax.set_ylabel("peak-detection F$_1$")
ax.tick_params(axis="x", labelsize=6)
panel(ax, "b", "simulated test set (ground truth known)")

ax = fig.add_subplot(gs[1, 1])
mt = metrics["matched"]
style = {"raw": (GREY, ":", "raw"), "savitzky-golay": (GOLD, "--", "Savitzky–Golay"),
         "gaussian": ("0.3", "-.", "Gaussian"), "unet-1branch": (C_POS, "-", "U-Net (1 half)")}
for key, (c, ls, lab) in style.items():
    xs = [int(t) for t in mt[key]]
    ax.plot(xs, [mt[key][t] for t in mt[key]], color=c, ls=ls, marker="o", ms=3, lw=1, label=lab)
ax.set_xlabel("peaks reported per half")
ax.set_xticks([2, 3, 4, 5, 6])
ax.set_ylabel("±V peak agreement above chance")
ax.legend(frameon=False)
n_al = sum(a.accepted for a in A)
panel(ax, "c", f"measured spectra ({n_al} aligned)")
save(fig, "fig_benchmark")

# =============================================================== Figure 3: before / after
def showcase(dev, n=3):
    sub = peaks[(peaks.device == dev) & (peaks.tier == "confirmed") & ~peaks.duplicated_across_devices]
    return list(sub.groupby("pair").snr.sum().sort_values(ascending=False).index[:n])


rows = [("pure", p) for p in showcase("pure")] + [("impure", p) for p in showcase("impure")]
fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 1.25 * len(rows) + 0.8), sharex=True,
                         gridspec_kw=dict(hspace=0.25, wspace=0.12))
LS = {"c": "-", "s": "--", "t": ":"}
for row, (dev, pair) in zip(axes, rows):
    i = idx_of[(dev, pair)]
    sc, mk = SC[i] * 1e9, MASK[i]
    for ax, Z, cols in ((row[0], X[i], (C_POS_L, C_NEG_L)), (row[1], D[i], (C_POS, C_NEG))):
        ax.plot(GRID, np.where(mk[0], Z[0] * sc, np.nan), color=cols[0], lw=0.8)
        ax.plot(GRID, np.where(mk[1], Z[1] * sc, np.nan), color=cols[1], lw=0.8)
        ax.axhline(0, color="0.3", lw=0.3)
    lim = np.abs(X[i][mk]).max() * sc * 1.1
    for ax in row:
        ax.set_ylim(-lim, lim)
    for pk in peaks[(peaks.device == dev) & (peaks.pair == pair) & (peaks.tier != "tentative")].itertuples():
        for ax in row:
            ax.axvline(pk.wavenumber, color="k", lw=0.5, ls=LS[pk.tier[0]], alpha=0.55)
        row[1].text(pk.wavenumber, lim * 0.96, f" {pk.wavenumber:.0f}", fontsize=5.5, va="top", rotation=90)
    row[0].set_ylabel(f"{dev}\n{pair}", fontsize=6.5)
    row[0].text(0.995, 0.04, "aligned" if A[i].accepted else "not aligned", transform=row[0].transAxes,
                ha="right", fontsize=5.5, color="0.4")
axes[0, 0].set_title("$\\bf{a}$  before: measured (detrended)", loc="left")
axes[0, 1].set_title("$\\bf{b}$  after: U-Net denoised", loc="left")
axes[0, 1].plot([], [], color=C_POS, label="+V")
axes[0, 1].plot([], [], color=C_NEG, label="−V, mirrored")
axes[0, 1].plot([], [], color="k", lw=0.6, label="confirmed (both halves)")
axes[0, 1].plot([], [], color="k", lw=0.6, ls="--", label="single half, ≥6σ")
fig.legend(*axes[0, 1].get_legend_handles_labels(), frameon=False, fontsize=6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
for ax in axes[-1]:
    ax.set_xlabel("wavenumber, common axis (cm$^{-1}$)")
fig.text(0.0, 0.5, "d$^2$I/dV$^2$ (nA V$^{-2}$)", rotation=90, va="center", fontsize=7)
save(fig, "fig_before_after")

# =============================================================== Figure 4: signed band maps
MAP_BANDS = [("C=C / amide C=O", 1600, 1695), ("N=C=S", 1990, 2140),
             ("S–C≡N", 2140, 2175), ("carboxylic O–H", 2500, 3300)]
xi, yi = np.meshgrid(np.linspace(1, 5, 120), np.linspace(1, 4, 90))
fig, axes = plt.subplots(len(MAP_BANDS), 2, figsize=(5.4, 1.7 * len(MAP_BANDS) + 0.4),
                         gridspec_kw=dict(hspace=0.3, wspace=0.08))
map_stats = {}
for r, (name, lo, hi) in enumerate(MAP_BANDS):
    m = (GRID >= lo) & (GRID <= hi)
    vals, agree, mids = {}, {}, {}
    for dev in ("pure", "impure"):
        idx = [i for i, s in enumerate(S) if s.device == dev]
        v, ag = [], []
        for i in idx:
            y = D[i][:, m]
            avg = (y[0] + y[1]) / 2                          # mirrored halves, even convention
            j = int(np.argmax(np.abs(avg)))
            v.append(avg[j] * SC[i] * 1e9)
            ag.append(np.sign(y[0, j]) == np.sign(y[1, j]) and abs(avg[j]) > 0)
        vals[dev], agree[dev], mids[dev] = np.array(v), np.array(ag), np.array([S[i].midpoint for i in idx])
        ok = agree[dev] & ~np.isin([S[i].pair for i in idx], list(DUP))
        npos, nneg = int((vals[dev][ok] > 0).sum()), int((vals[dev][ok] < 0).sum())
        map_stats[f"{dev}/{name}"] = dict(
            n=len(idx), n_sign_agree=int(ok.sum()), pos_agree=npos, neg_agree=nneg,
            p_sign=float(binomtest(npos, npos + nneg).pvalue) if npos + nneg else 1.0,
            median_nA=float(np.median(vals[dev])))
    vmax = max(np.percentile(np.abs(vals[d]), 95) for d in vals)
    for c, dev in enumerate(("pure", "impure")):
        ax = axes[r, c]
        zi = griddata(mids[dev], vals[dev], (xi, yi), method="cubic")
        cf = ax.contourf(xi, yi, np.clip(zi, -vmax, vmax), np.linspace(-vmax, vmax, 17), cmap="RdBu_r", extend="both")
        good = agree[dev]
        ax.plot(*mids[dev][good].T, "o", ms=2.2, color="k", mew=0)
        ax.plot(*mids[dev][~good].T, "x", ms=3.2, color="k", mew=0.6)
        ax.set_aspect("equal")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4])
        if c == 1:
            ax.set_yticklabels([])
        if r == len(MAP_BANDS) - 1:
            ax.set_xlabel("X")
        if c == 0:
            ax.set_ylabel("Y")
        if r == 0:
            ax.set_title(f"{dev} device", fontsize=7.5, fontweight="bold")
        if c == 0:
            ax.text(-0.32, 0.5, f"{name}\n{lo}–{hi} cm$^{{-1}}$", transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=6.5)
    cb = fig.colorbar(cf, ax=axes[r, :], shrink=0.9, pad=0.02)
    cb.set_label("nA V$^{-2}$", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
fig.text(0.45, 0.04, "●  both bias halves agree in sign    ×  halves disagree (sign not reproducible)",
         ha="center", fontsize=6.5)
save(fig, "fig_maps")
json.dump(map_stats, open("outputs/report/map_sign_stats.json", "w"), indent=1)

# =============================================================== Supplementary figures (copied)
for src, dst in [("outputs/report/fig01_raw_examples.png", "S_raw_examples.png"),
                 ("outputs/report/fig02_alignment.png", "S_alignment.png"),
                 ("outputs/report/fig03_alignment_params.png", "S_alignment_params.png"),
                 ("outputs/report/fig04_synthetic_example.png", "S_synthetic_example.png"),
                 ("outputs/report/fig06_before_after_pure.png", "S_before_after_pure.png"),
                 ("outputs/report/fig06_before_after_impure.png", "S_before_after_impure.png"),
                 ("outputs/report/fig07_heatmaps_pure.png", "S_heatmaps_pure.png"),
                 ("outputs/report/fig07_heatmaps_impure.png", "S_heatmaps_impure.png"),
                 ("outputs/report/fig08_regions.png", "S_regions.png"),
                 ("outputs/report/fig09_bond_families.png", "S_bond_families.png"),
                 ("outputs/report/fig10_spatial_peaks.png", "S_spatial_peaks.png")]:
    shutil.copy(src, f"{FIG}/{dst}")

shutil.copy(f"{FIG}/fig_polarity.png", "outputs/report/fig13_polarity.png")   # for the Word report

# noise statistics figure for the SI
from iets.pipeline import residual_noise
N = residual_noise(X, A)
ac = np.mean([np.correlate(n - n.mean(), n - n.mean(), "full")[len(n) - 1:] / np.dot(n - n.mean(), n - n.mean())
              for n in N], 0)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
lag = np.arange(len(ac)) * (GRID[1] - GRID[0])
axes[0].plot(lag[:60], ac[:60], color=C_POS, lw=1.2)
axes[0].axhline(0, color="0.3", lw=0.4)
axes[0].axhline(0.5, color="0.5", lw=0.4, ls=":")
hw = lag[np.argmax(ac < 0.5)]
axes[0].text(hw + 15, 0.55, f"half-width ≈ {hw:.0f} cm$^{{-1}}$", fontsize=6.5)
axes[0].set_xlabel("lag (cm$^{-1}$)")
axes[0].set_ylabel("autocorrelation")
panel(axes[0], "a", f"residual noise ({len(N)} aligned spectra)")
f = np.fft.rfftfreq(len(GRID), d=GRID[1] - GRID[0])
psd = np.mean(np.abs(np.fft.rfft(N, axis=1)) ** 2, 0)
axes[1].loglog(f[1:], psd[1:] / psd[1:].max(), color=C_POS, lw=1.0, label="residual noise in d$^2$I/dV$^2$")
f_lo, f_hi = np.sqrt(np.log(2)) / (2 * np.pi * 70), np.sqrt(np.log(2)) / (2 * np.pi * 12)
axes[1].axvspan(f_lo, f_hi, color=GOLD, alpha=0.15, lw=0, label="half-power band of features, σ = 12–70 cm$^{-1}$")
axes[1].set_xlabel("spectral frequency (cycles per cm$^{-1}$)")
axes[1].set_ylabel("normalised power")
axes[1].legend(frameon=False)
panel(axes[1], "b", "noise power spectrum")
fig.tight_layout()
save(fig, "S_noise")
print("wrote", FIG, sorted(os.listdir(FIG)))
print(json.dumps(map_stats, indent=1))
