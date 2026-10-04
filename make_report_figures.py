"""Figures and bond summary tables for the project report.

    .venv/bin/python make_report_figures.py     # -> outputs/report/

Run after train.py, evaluate.py and apply.py.
"""
from __future__ import annotations

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from iets.align import Alignment
from iets.analysis import bond_table, candidates, gauss, load_model, savgol, unet
from iets.bands import BANDS
from iets.data import duplicated_pairs
from iets.pipeline import prepare_all
from iets.preprocess import GRID, branches, robust_scale

OUT = "outputs/report"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
C_POS, C_NEG, C_POS_L, C_NEG_L = "#2a6cb3", "#c8562a", "#a3bfe0", "#eab49c"
DEV_COL = {"impure": "#c8562a", "pure": "#2a6cb3"}
TIER_LS = {"c": "-", "s": "--", "t": ":"}

# Coarse spectral regions: robust to the position uncertainty and to overlapping FTIR bands.
REGIONS = [
    ("< 400", 200, 400, "Below the FTIR table: lattice / metal–oxide modes"),
    ("400–900", 400, 900, "M–O stretch, C–halogen stretch, aromatic C–H out-of-plane bend"),
    ("900–1350", 900, 1350, "Fingerprint: C–O, C–N, C–F, S=O stretch; C=C bend"),
    ("1350–1500", 1350, 1500, "C–H and O–H bending; N–O stretch"),
    ("1500–1800", 1500, 1800, "C=C, C=O (incl. amide, carboxylic) stretch; N–H bend"),
    ("1800–2300", 1800, 2300, "Triple / cumulated bonds: C≡N, C≡C, N=C=S, S–C≡N, N=N=N"),
    ("2300–2500", 2300, 2500, "No entry in the FTIR table (CO₂ ≈ 2349 is one candidate)"),
    ("2500–3000", 2500, 3000, "Carboxylic O–H, C–H (alkane, aldehyde) and S–H stretch"),
    ("3000–3650", 3000, 3650, "O–H, N–H and sp²/sp C–H stretch"),
]

S, X, MASK, SC, A = prepare_all()
model2 = load_model("models/unet_2branch.pt", 2)
D = unet(model2, X) * MASK
peaks = pd.read_csv("outputs/peaks.csv")
metrics = json.load(open("outputs/metrics.json"))
align_cache = json.load(open("outputs/alignment.json"))
DUP = duplicated_pairs(S)
idx_of = {(s.device, s.pair): i for i, s in enumerate(S)}
peaks["tier"] = peaks.confidence.str.split(" ").str[0]
peaks["high_conf"] = peaks.tier.isin(["confirmed", "single-half"])


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------- 1. raw data
fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True)
for ax, (dev, pair) in zip(axes.flat, [("impure", "11_12"), ("impure", "14_23"),
                                       ("pure", "11_12"), ("pure", "12_21")]):
    s = S[idx_of[(dev, pair)]]
    ax.plot(s.wavenumber, s.d2, color=DEV_COL[dev], lw=0.8)
    for lo, hi in ((-4100, -3648), (3648, 4100), (-200, 200)):
        ax.axvspan(lo, hi, color="0.85")
    ax.set_title(f"{dev} device, pair {pair}")
    ax.set_xlim(-4100, 4100)
    ax.axhline(0, color="k", lw=0.3)
for ax in axes[1]:
    ax.set_xlabel("Wvnmbr (cm$^{-1}$)  = 8066 × bias (V)")
for ax in axes[:, 0]:
    ax.set_ylabel("DerivativeY2_2 (A/V²)")
fig.suptitle("Raw measurements. Grey: excluded regions (sweep-edge artefacts and zero bias)", fontsize=10)
fig.tight_layout()
save(fig, "fig01_raw_examples")

# ---------------------------------------------------------------- 2. misalignment
fig = plt.figure(figsize=(10, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
i = idx_of[("impure", "14_23")]
s, a = S[i], A[i]
p, m = branches(s.wavenumber, s.d2)
sc = robust_scale(np.concatenate([p, m]))
ax = fig.add_subplot(gs[0, :])
ax.plot(GRID, p / sc, color=C_POS, lw=1, label="+V half")
ax.plot(GRID, m / sc, color=C_NEG, lw=1, label="−V half (mirrored)")
ax.set_title(f"impure 14_23 as measured: the same feature sits ≈{abs(a.c + (a.k - 1) * 2400):.0f} cm$^{{-1}}$ apart "
             f"in the two halves (r = {a.r0:+.2f})", fontsize=9)
ax.legend(fontsize=8)
ax.set_ylabel("normalised")
ax = fig.add_subplot(gs[1, :2])
ax.plot(GRID, X[i, 0], color=C_POS, lw=1, label="+V half")
ax.plot(GRID, X[i, 1], color=C_NEG, lw=1, label="−V half")
ax.set_title(f"after alignment onto the common axis (−V ≈ +V at {a.k:.2f}·w {a.c:+.0f} cm$^{{-1}}$, r = {a.r:+.2f})",
             fontsize=9)
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("normalised")
ax = fig.add_subplot(gs[1, 2])
null = np.array(align_cache["null"])
best = np.array([abs(x.r) for x in A])
bins = np.linspace(0.2, 1, 33)
ax.hist(null, bins, color="0.7", density=True, label="unrelated pairs (chance)")
ax.hist(best, bins, color=C_POS, alpha=0.7, density=True, label="same pair, ±V")
ax.axvline(align_cache["threshold"], color="k", ls="--", lw=0.8)
ax.text(align_cache["threshold"], ax.get_ylim()[1] * 0.95, " accept", fontsize=7, va="top")
ax.set_xlabel("best |r| after alignment")
ax.set_title("Alignment vs chance", fontsize=9)
ax.legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
save(fig, "fig02_alignment")

acc = [x for x in A if x.accepted]
fig, ax = plt.subplots(figsize=(5, 3.4))
for dev in ("impure", "pure"):
    aa = [x for x, s in zip(A, S) if x.accepted and s.device == dev]
    ax.scatter([x.k for x in aa], [x.c for x in aa], s=18, color=DEV_COL[dev], label=f"{dev} ({len(aa)})")
ax.set_xlabel("stretch k")
ax.set_ylabel("shift c (cm$^{-1}$)")
ax.set_title("Fitted mapping −V(w) ≈ ±(+V)(k·w + c), accepted spectra", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig03_alignment_params")

# ---------------------------------------------------------------- 3. synthetic example
syn = np.load("models/synth_test.npz")
Xs, Ys = syn["X"][1000:1400], syn["Y"][1000:1400]
Ds = unet(model2, Xs)
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, k in zip(axes, (3, 11)):
    ax.plot(GRID, Xs[k, 0], color="0.7", lw=0.8, label="noisy input (+V half)")
    ax.plot(GRID, savgol(Xs[k:k + 1, 0])[0], color="#c9a227", lw=1, label="Savitzky–Golay")
    ax.plot(GRID, Ys[k, 0], color="k", lw=2.2, alpha=0.3, label="true clean signal")
    ax.plot(GRID, Ds[k, 0], color=C_POS, lw=1.2, label="U-Net (both halves)")
    ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("normalised")
axes[0].legend(fontsize=7, ncol=4, loc="upper left")
axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
fig.suptitle("Held-out simulated spectra, where the true signal is known", fontsize=10)
fig.tight_layout()
save(fig, "fig04_synthetic_example")

# ---------------------------------------------------------------- 4. benchmark
names = ["raw", "savitzky-golay", "gaussian", "branch-average", "unet-1branch", "unet-2branch"]
labels = ["Raw", "Savitzky–\nGolay", "Gaussian", "Branch\naverage", "U-Net\n1 half", "U-Net\nboth halves"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
f1s = [metrics["synthetic"][n]["f1"] for n in names]
b = axes[0].bar(labels, f1s, color=["0.65"] * 4 + [C_POS_L, C_POS])
axes[0].bar_label(b, fmt="%.2f", fontsize=8)
axes[0].set_ylabel("peak-detection F1")
axes[0].set_title("Simulated test set (ground truth known)")
axes[0].set_ylim(0, 1)
mt = metrics["matched"]
style = {"raw": ("0.6", ":"), "savitzky-golay": ("#c9a227", "--"), "gaussian": ("0.3", "-."),
         "unet-1branch": (C_POS, "-")}
for k, (c, ls) in style.items():
    xs = [int(t) for t in mt[k]]
    axes[1].plot(xs, [mt[k][t] for t in mt[k]], color=c, ls=ls, marker="o", ms=4,
                 label={"raw": "Raw", "savitzky-golay": "Savitzky–Golay", "gaussian": "Gaussian",
                        "unet-1branch": "U-Net (1 half)"}[k])
axes[1].set_xlabel("peaks reported per half (operating point)")
axes[1].set_ylabel("±V peak agreement above chance")
axes[1].set_title(f"Real data, {sum(x.accepted for x in A)} aligned spectra")
axes[1].legend(fontsize=7.5)
fig.tight_layout()
save(fig, "fig05_benchmark")


# ---------------------------------------------------------------- 5. before / after
def before_after(dev, pairs, name):
    fig, axes = plt.subplots(len(pairs), 2, figsize=(11, 2.15 * len(pairs)), sharex=True)
    for row, pair in zip(axes, pairs):
        i = idx_of[(dev, pair)]
        sc, a, mk = SC[i] * 1e9, A[i], MASK[i]
        for ax, Z, cols in ((row[0], X[i], (C_POS_L, C_NEG_L)), (row[1], D[i], (C_POS, C_NEG))):
            ax.plot(GRID, np.where(mk[0], Z[0] * sc, np.nan), color=cols[0], lw=0.9, label="+V")
            ax.plot(GRID, np.where(mk[1], Z[1] * sc, np.nan), color=cols[1], lw=0.9, label="−V (mirrored)")
            ax.axhline(0, color="k", lw=0.3)
        lim = np.abs(X[i][mk]).max() * sc * 1.1
        for ax in row:
            ax.set_ylim(-lim, lim)
        for p in peaks[(peaks.device == dev) & (peaks.pair == pair)].itertuples():
            ls = TIER_LS[p.tier[0]]
            for ax in row:
                ax.axvline(p.wavenumber, color="k", lw=0.6, ls=ls, alpha=0.6)
            if p.tier != "tentative":
                row[1].text(p.wavenumber, lim * 0.97, f" {p.wavenumber:.0f}", fontsize=6.5, va="top", rotation=90)
        tag = "aligned" if a.accepted else "not aligned"
        row[0].set_ylabel(f"{pair} ({tag})\nnA/V²", fontsize=7.5)
    axes[0, 0].set_title("Before: raw (detrended)")
    axes[0, 1].set_title("After: U-Net denoised")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("Wavenumber, common axis (cm$^{-1}$)")
    fig.suptitle(f"{dev} device — lines: solid confirmed, dashed single-half, dotted tentative", fontsize=10)
    fig.tight_layout()
    save(fig, name)


showcase = {}
for dev in ("impure", "pure"):
    sub = peaks[(peaks.device == dev) & (peaks.tier == "confirmed") & ~peaks.duplicated_across_devices]
    top = list(sub.groupby("pair").snr.sum().sort_values(ascending=False).index[:3])
    single = peaks[(peaks.device == dev) & (peaks.tier == "single-half") & ~peaks.pair.isin(top)
                   & ~peaks.duplicated_across_devices]
    extra = [single.sort_values("snr").pair.iloc[-1]] if len(single) else []
    flat = ["11_12"] if dev == "pure" else ["13_22"]
    showcase[dev] = top + extra + flat
    before_after(dev, showcase[dev], f"fig06_before_after_{dev}")

# ---------------------------------------------------------------- 6. heatmaps before / after
xi, yi = np.meshgrid(np.linspace(1, 5, 100), np.linspace(1, 4, 80))
hm_bands = ["C=C stretch", "Isothiocyanate N=C=S", "O-H (H-bonded)"]
for dev in ("impure", "pure"):
    idx = [i for i, s in enumerate(S) if s.device == dev]
    mid = np.array([S[i].midpoint for i in idx])
    fig, axes = plt.subplots(len(hm_bands), 2, figsize=(8, 3.1 * len(hm_bands)))
    for row, band in zip(axes, hm_bands):
        lo, hi = BANDS[band]
        m = (GRID >= lo) & (GRID <= hi)
        for ax, Z, lab in ((row[0], X, "before (raw)"), (row[1], D, "after (denoised)")):
            v = np.array([np.max(np.abs(Z[i][:, m] * MASK[i][:, m]).mean(0)) * SC[i] for i in idx]) * 1e9
            zi = np.clip(griddata(mid, v, (xi, yi), method="cubic"), 0, None)
            cf = ax.contourf(xi, yi, zi, 16, cmap="magma_r")
            fig.colorbar(cf, ax=ax, shrink=0.85, label="nA/V²")
            ax.plot(mid[:, 0], mid[:, 1], ".", color="0.4", ms=2)
            ax.set_title(f"{band} {lo}–{hi} cm$^{{-1}}$\n{lab}", fontsize=8)
            ax.set_aspect("equal")
    fig.suptitle(f"{dev} device: band magnitude maps before and after denoising", fontsize=10)
    fig.tight_layout()
    save(fig, f"fig07_heatmaps_{dev}")

# ---------------------------------------------------------------- 7. region summary (non-duplicated pairs)
uniq = peaks[~peaks.duplicated_across_devices]
n_uniq = {d: sum(1 for s in S if s.device == d and s.pair not in DUP) for d in ("impure", "pure")}
reg_rows = []
for name, lo, hi, desc in REGIONS:
    r = dict(region=name, lo=lo, hi=hi, description=desc)
    for dev in ("impure", "pure"):
        sub = uniq[(uniq.device == dev) & uniq.wavenumber.between(lo, hi, inclusive="left")]
        r[f"{dev}_high"] = int(sub[sub.high_conf].pair.nunique())
        r[f"{dev}_any"] = int(sub.pair.nunique())
    dsub = peaks[peaks.duplicated_across_devices & (peaks.device == "impure")
                 & peaks.wavenumber.between(lo, hi, inclusive="left")]
    r["dup_high"] = int(dsub[dsub.high_conf].pair.nunique())
    r["dup_any"] = int(dsub.pair.nunique())
    reg_rows.append(r)
reg = pd.DataFrame(reg_rows)
# device difference per region: Fisher exact test on pairs with / without a high-confidence peak,
# Holm-corrected over the regions
from scipy.stats import fisher_exact
reg["p"] = [fisher_exact([[r.impure_high, n_uniq["impure"] - r.impure_high],
                          [r.pure_high, n_uniq["pure"] - r.pure_high]])[1] for r in reg.itertuples()]
order = np.argsort(reg.p.values)
holm, run = np.empty(len(reg)), 0.0
for k, i in enumerate(order):
    run = max(run, (len(reg) - k) * reg.p.values[i])
    holm[i] = min(1.0, run)
reg["p_holm"] = holm
reg.to_csv(f"{OUT}/region_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 3.6))
xx = np.arange(len(reg))
for j, dev in enumerate(("impure", "pure")):
    hi_ = 100 * reg[f"{dev}_high"] / n_uniq[dev]
    any_ = 100 * reg[f"{dev}_any"] / n_uniq[dev]
    ax.bar(xx + (j - 0.5) * 0.38, any_, 0.38, color=DEV_COL[dev], alpha=0.3)
    ax.bar(xx + (j - 0.5) * 0.38, hi_, 0.38, color=DEV_COL[dev], label=f"{dev}")
ax.set_xticks(xx, [r.replace("–", "–\n") for r in reg.region], fontsize=8)
ax.set_xlabel("spectral region (cm$^{-1}$)")
ax.set_ylabel(f"% of the {n_uniq['impure']} non-duplicated pairs")
ax.set_title("Pairs with a peak in each region. Solid: confirmed or single-half; pale: including tentative",
             fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig08_regions")

# ---------------------------------------------------------------- 8. bond families (best match)
table = bond_table()


def family(v):
    v = re.sub(r"\s*\(.*?\)", "", v)
    return v.replace("out-of-plane ", "").replace("=C-H", "C-H").replace("≡C-H", "C-H").strip()


rows = []
for p in peaks.itertuples():
    unc = 0.0 if np.isnan(p.position_uncertainty) else p.position_uncertainty
    exact = candidates(p.wavenumber, 0.0, table)
    wide = candidates(p.wavenumber, unc, table)
    best = exact.assign(w=exact.wn_high - exact.wn_low).sort_values("w").head(1)
    rows.append(dict(device=p.device, pair=p.pair, x=p.x, y=p.y, wavenumber=p.wavenumber,
                     position_uncertainty=p.position_uncertainty, aligned=p.aligned, tier=p.tier,
                     confidence=p.confidence, strength_nA_V2=p.strength * 1e9, snr=p.snr, symmetry=p.symmetry,
                     duplicated_across_devices=p.duplicated_across_devices,
                     region=next((r[0] for r in REGIONS if r[1] <= p.wavenumber < r[2]), ""),
                     best_match=(f"{best.vibration.iloc[0]} — {best.group.iloc[0]} "
                                 f"({best.wn_low.iloc[0]}–{best.wn_high.iloc[0]})") if len(best) else "no match in table",
                     best_family=family(best.vibration.iloc[0]) if len(best) else "no match in table",
                     families_within_uncertainty="; ".join(sorted(set(wide.vibration.map(family))))))
bp = pd.DataFrame(rows).sort_values(["device", "pair", "wavenumber"])
bp.to_csv(f"{OUT}/peak_bond_assignments.csv", index=False)

hc = bp[bp.tier.isin(["confirmed", "single-half"])]
fam_rows = []
for fam_name in sorted(hc.best_family.unique()):
    g = hc[hc.best_family == fam_name]
    r = dict(family=fam_name)
    for dev in ("impure", "pure"):
        gd = g[(g.device == dev) & ~g.duplicated_across_devices]
        r[f"{dev}_pairs"] = int(gd.pair.nunique())
        r[f"{dev}_wavenumbers"] = ", ".join(f"{w:.0f}" for w in sorted(gd.wavenumber.unique()))
    gdup = g[g.duplicated_across_devices & (g.device == "impure")]
    r["dup_pairs"] = int(gdup.pair.nunique())
    fam_rows.append(r)
fam = pd.DataFrame(fam_rows)
fam["total"] = fam.impure_pairs + fam.pure_pairs + fam.dup_pairs
fam = fam.sort_values("total", ascending=False)
fam.to_csv(f"{OUT}/bond_family_summary.csv", index=False)

piv = fam.set_index("family")[["impure_pairs", "pure_pairs"]]
piv = piv[piv.sum(axis=1) > 0].iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 0.3 * len(piv) + 1.3))
yy = np.arange(len(piv))
for j, dev in enumerate(("impure", "pure")):
    ax.barh(yy + (j - 0.5) * 0.4, piv[f"{dev}_pairs"], 0.4, color=DEV_COL[dev], label=f"{dev} device")
ax.set_yticks(yy, piv.index, fontsize=7.5)
ax.set_xlabel(f"non-duplicated electrode pairs (of {n_uniq['impure']}) with a high-confidence peak")
ax.legend(fontsize=8)
ax.set_title("Best-matching FTIR vibration for confirmed and single-half peaks", fontsize=9)
fig.tight_layout()
save(fig, "fig09_bond_families")

# ---------------------------------------------------------------- 9. spatial peak map
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
cmap = plt.cm.viridis
for ax, dev in zip(axes, ("impure", "pure")):
    allp = [(s.pair, s.midpoint) for s in S if s.device == dev]
    for pair, (px, py) in allp:
        ax.plot(px, py, "s" if pair in DUP else "o", mfc="none", mec="0.6" if pair in DUP else "0.8",
                ms=13 if pair in DUP else 11, mew=1.2 if pair in DUP else 0.8)
    sub = bp[(bp.device == dev) & bp.tier.isin(["confirmed", "single-half"])]
    for (px, py), g in sub.groupby(["x", "y"]):
        ws = sorted(zip(g.wavenumber, g.tier))
        n = len(ws)
        for k, (w, t) in enumerate(ws):
            ang = 2 * np.pi * k / n
            r = 0.0 if n == 1 else 0.1
            ax.plot(px + r * np.cos(ang), py + r * np.sin(ang), "o" if t == "confirmed" else "^",
                    color=cmap((w - GRID[0]) / (GRID[-1] - GRID[0])), ms=5.5 if t == "confirmed" else 5)
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0.7, 4.3)
    ax.set_aspect("equal")
    ax.set_title(f"{dev}: {sub.pair.nunique()} of 55 pairs with a confirmed or single-half peak", fontsize=9)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(GRID[0], GRID[-1]))
fig.colorbar(sm, ax=axes, label="peak wavenumber (cm$^{-1}$)", shrink=0.85)
fig.text(0.02, 0.01, "● confirmed   ▲ single-half   □ pair with nearly identical data in both device folders",
         fontsize=8)
save(fig, "fig10_spatial_peaks")

# ---------------------------------------------------------------- summary numbers for the report
stats = {}
for dev in ("impure", "pure"):
    d_all = bp[bp.device == dev]
    d_u = d_all[~d_all.duplicated_across_devices]
    stats[dev] = dict(
        n_peaks=len(d_all), n_confirmed=int((d_all.tier == "confirmed").sum()),
        n_single=int((d_all.tier == "single-half").sum()), n_tentative=int((d_all.tier == "tentative").sum()),
        pairs_high=int(d_all[d_all.tier != "tentative"].pair.nunique()),
        pairs_high_unique=int(d_u[d_u.tier != "tentative"].pair.nunique()),
        aligned=int(sum(a.accepted for a, s in zip(A, S) if s.device == dev)),
        aligned_unique=int(sum(a.accepted for a, s in zip(A, S) if s.device == dev and s.pair not in DUP)),
        k_median=float(np.median([a.k for a, s in zip(A, S) if s.device == dev and a.accepted])),
        c_median=float(np.median([a.c for a, s in zip(A, S) if s.device == dev and a.accepted])),
        unc_median=float(d_all.position_uncertainty.median()),
        odd=int((d_all.symmetry == "odd").sum()), even=int((d_all.symmetry == "even").sum()))
json.dump(dict(showcase=showcase, stats=stats, n_unique=n_uniq, duplicated=sorted(DUP),
               threshold=align_cache["threshold"], metrics=metrics,
               regions=reg.to_dict("records"), families=fam.to_dict("records"),
               peaks=bp.replace({np.nan: None}).to_dict("records")),
          open(f"{OUT}/report_data.json", "w"), indent=1, default=float)
print(json.dumps(stats, indent=1))
print(reg.to_string())
print(fam.head(20).to_string())
