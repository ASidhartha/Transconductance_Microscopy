"""Main-text Figs. 3-5 and Supplementary Note 7 figures for the ALD MoS2 data.

    .venv/bin/python make_mos2_figures.py        (after mos2_analysis.py)

Writes PDFs into outputs/paper_figures and the bodies of Supplementary Tables 7.1-7.4 into
outputs/mos2/mos2_tables.tex (pasted into Supplementary_Information.tex, which must stay one file).
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FIG = "outputs/paper_figures"
TEX = "outputs/mos2/mos2_tables.tex"
OUT = "outputs/mos2"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "font.family": "sans-serif", "pdf.fonttype": 42})
# validated categorical order (dataviz validator, light surface): blue, orange, purple
BLUE, ORANGE, PURPLE, GREY = "#2a6cb3", "#c8562a", "#7b4fa8", "0.55"
BLUE_L, ORANGE_L = "#a3bfe0", "#eab49c"
CM_PER_V = 8065.54

S = json.load(open(f"{OUT}/summary.json"))
NZ = pd.read_csv(f"{OUT}/noise.csv")
TR = pd.read_csv(f"{OUT}/transient.csv")
FRC = pd.read_csv(f"{OUT}/frc.csv")
CAND = pd.read_csv(f"{OUT}/candidates.csv")
BEN = pd.read_csv(f"{OUT}/benchmark.csv")
D = np.load(f"{OUT}/spectra.npz", allow_pickle=True)
NAMES = [str(n) for n in D["names"]]
THERMAL_V = S["thermal_fwhm_mV"] / 1e3


def short(n):
    k, loc = n.split("/")
    return ("T1 " if k == "type1" else "") + loc.replace("pos", "P").replace("centre", "C")


def colour(n):
    """Type 1 (0 to +0.5 V, 5 mV/s) orange; Type 2 periphery blue; Type 2 centre purple."""
    if n.startswith("type1"):
        return ORANGE
    return PURPLE if "centre" in n else BLUE


def panel(ax, letter, title=""):
    ax.set_title(f"$\\bf{{{letter}}}$  {title}", loc="left")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


def get(key, n):
    return D[f"{key}__{n}"]


def tstat(n):
    """Same pseudo-t as iets.features.Traces.tstat (variance smoothed along bias)."""
    from scipy.ndimage import gaussian_filter1d
    x, V, valid = get("x", n), get("V", n), get("valid", n)
    fw = NZ.set_index("location").loc[n, "kernel_fwhm_mV"] / 1e3
    var = x.var(0, ddof=1)
    sig = 2 * fw / abs(np.mean(np.diff(V)))
    w = valid.astype(float)
    v = gaussian_filter1d(var * w, sig, mode="nearest") / np.maximum(gaussian_filter1d(w, sig, mode="nearest"), 1e-6)
    return x.mean(0) / np.sqrt(v / x.shape[0])


def relax_at_1s(n):
    """Voltage-equivalent current deficit 1 s after the bias step that starts the sweep (mV)."""
    Ic, Vn, tn = get("Iclean", n).mean(0), get("V", n), get("t", n)
    late = tn > 60
    g, c = np.polyfit(Vn[late], Ic[late], 1)
    y = (Ic - g * Vn - c) / g * 1e3
    return float(-np.median(y[(tn > 0.7) & (tn < 1.3)]))


def cm_axis(ax):
    sec = ax.secondary_xaxis("top", functions=(lambda v: v * CM_PER_V, lambda w: w / CM_PER_V))
    sec.set_xlabel("$\\tilde\\nu$ (cm$^{-1}$)", labelpad=2)
    sec.tick_params(labelsize=6)
    return sec


legend_types = [Line2D([], [], color=ORANGE, lw=1, label="Type 1, 0→0.5 V, 5 mV s$^{-1}$"),
                Line2D([], [], color=BLUE, lw=1, label="Type 2 periphery, 6–10 mV s$^{-1}$"),
                Line2D([], [], color=PURPLE, lw=1, label="Type 2 centre")]

# =============================================================== Fig. 3: time-locked artefacts
EX = "type2/pos2"
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(hspace=0.75, wspace=0.55))
Iraw, t = get("Iraw", EX), get("t", EX)
V = get("V", EX)
res = (Iraw - Iraw.mean(0)) * 1e12
a = ax[0, 0]
win = (t > 60) & (t < 100)
for j, k in enumerate((0, 4, 8)):
    y = res[k][win] - np.polyval(np.polyfit(t[win], res[k][win], 3), t[win])  # slow drift removed
    a.plot(t[win], y + 25 * (1 - j), lw=0.6, color=(BLUE, ORANGE, PURPLE)[j])
    a.text(101.5, 25 * (1 - j), f"rep. {k + 1}", fontsize=5.8, va="center", color=(BLUE, ORANGE, PURPLE)[j], clip_on=False)
a.set_xlim(60, 100)
a.set_ylim(-40, 40)
a.set_xlabel("time since sweep start (s)")
a.set_ylabel("$I_k - \\bar I$, drift removed (pA)")
panel(a, "a", "Repeat-to-repeat fluctuations")

a = ax[0, 1]
psd = get("psd", EX)
f, Praw, Pcl = psd
a.loglog(f[1:], Praw[1:] * 1e24, color=GREY, lw=0.9, label="measured")
a.loglog(f[1:], Pcl[1:] * 1e24, color=BLUE, lw=0.9, label="lines removed")
fl = NZ.set_index("location").loc[EX, "f_line_median_Hz"]
a.annotate(f"{fl:.2f} Hz", (fl, Praw[np.argmin(abs(f - fl))] * 1e24), xytext=(4, 0), textcoords="offset points",
           fontsize=6, va="center")
a.set_xlabel("frequency (Hz)")
a.set_ylabel("PSD (pA$^2$ Hz$^{-1}$)")
a.legend(frameon=False, loc="lower left")
panel(a, "b", "Noise spectrum (median of repeats)")

a = ax[0, 2]
leg, legl, valid = get("legacy", EX), get("leglines", EX), get("valid", EX)
m = valid
a.plot(V[m], leg[m], color="0.35", lw=0.9, label="conventional (SG)")
a.plot(V[m], legl[m], color=BLUE, lw=0.7, alpha=0.85, label="interference alone")
r = NZ.set_index("location").loc[EX, "legacy_corr_lines"]
a.text(0.02, 0.97, f"$r$ = {r:.2f}", transform=a.transAxes, va="top", fontsize=6.5)
a.set_xlabel("bias $V$ (V)")
a.set_ylabel("$(d^2I/dV^2)/(dI/dV)$ (V$^{-1}$)")
a.legend(frameon=False, loc="upper center", fontsize=5.6, ncol=2, bbox_to_anchor=(0.5, -0.32))
cm_axis(a)
panel(a, "c", "")
a.set_title("$\\bf{c}$  Conventional spectrum", loc="left", pad=16)

for k, (a, coord) in enumerate(((ax[1, 0], "V"), (ax[1, 1], "t"))):
    for n in NAMES:
        Ic = get("Iclean", n).mean(0)
        Vn, tn = get("V", n), get("t", n)
        late = tn > 60
        p = np.polyfit(Vn[late], Ic[late], 1)
        y = (Ic - np.polyval(p, Vn)) / p[0] * 1e3
        a.plot(Vn if coord == "V" else tn, y, color=colour(n), lw=0.7)
    a.set_ylim(-260, 60)
    a.set_ylabel("$(\\bar I - G V - c)/G$ (mV)")
    if coord == "V":
        a.set_xlabel("bias $V$ (V)")
        panel(a, "d", "Background against bias")
    else:
        a.set_xlabel("time since sweep start (s)")
        a.set_xlim(0, 100)
        panel(a, "e", "Same, against time")
ax[1, 1].legend(handles=legend_types, frameon=False, loc="lower right", fontsize=5.6, handlelength=1.2)

a = ax[1, 2]
nz = NZ.set_index("location").loc[NAMES]
y = np.arange(len(NAMES))
a.barh(y + 0.18, nz.legacy_var_frac_lines * 100, height=0.34, color=BLUE, label="interference lines")
a.barh(y - 0.18, nz.legacy_var_frac_glitches * 100, height=0.34, color=ORANGE, label="glitches")
a.set_yticks(y)
a.set_yticklabels([short(n) for n in NAMES], fontsize=6)
a.set_xlim(0, 100)
a.set_xlabel("variance of conventional spectrum (%)")
a.legend(frameon=False, loc="lower center", fontsize=5.6, ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1)
a.invert_yaxis()
a.set_title("$\\bf{f}$  Share due to artefacts", loc="left", pad=12)
save(fig, "fig_mos2_artefacts")

# =============================================================== SI: all tests on the measured spectra
fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 3, hspace=0.8, wspace=0.45)
a = fig.add_subplot(gs[0, 0])
for n in NAMES:
    d = FRC[FRC.location == n]
    a.semilogx(d.f_per_V, d.frc, color=colour(n), lw=0.7, alpha=0.9)
a.axhline(1 / 3, color="k", lw=0.5, ls="--")
a.axvline(1 / THERMAL_V, color=GREY, lw=0.6, ls=":")
a.text(1 / THERMAL_V * 0.92, -0.5, "1/(thermal FWHM)", fontsize=5.5, color=GREY, ha="right")
a.set_ylim(-0.6, 1.05)
a.set_xlim(1, 250)
a.set_xlabel("bias frequency (cycles V$^{-1}$)")
a.set_ylabel("half-data correlation")
panel(a, "a", "Resolution from the repeats")

a = fig.add_subplot(gs[0, 1:])
t2 = [n for n in NAMES if n.startswith("type2")]
off = 0
for n in t2:
    tt, Vn, valid = tstat(n), get("V", n), get("valid", n)
    thr = NZ.set_index("location").loc[n, "maxt_threshold"]
    a.plot(Vn[valid], tt[valid] + off, color=colour(n), lw=0.7)
    a.fill_between(Vn[valid], off - thr, off + thr, color="0.92", lw=0, zorder=0)
    a.text(0.505, off, short(n), fontsize=6, va="center")
    off -= 16
k = CAND[(CAND.location == "type2/pos4")].iloc[0]
a.annotate(f"{k.V * 1e3:.0f} mV, $p_{{\\rm FWER}}$ = {k.p_fwer:.2f}", (k.V, k.tstat - 16 * (len(t2) - 1)),
           xytext=(12, -6), textcoords="offset points", fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5))
a.set_xlim(-0.32, 0.55)
a.set_yticks([])
a.spines["left"].set_visible(False)
a.set_xlabel("bias $V$ (V)")
cm_axis(a)
a.set_title("$\\bf{b}$  Pseudo-$t$ per location; grey: family-wise 5% threshold", loc="left", pad=16)

a = fig.add_subplot(gs[1, 0])
grid = D["grid"]
for mode, c, lab in (("None", BLUE, "+V, all locations"), ("odd", ORANGE, "+ mirrored −V (odd)"),
                     ("even", PURPLE, "+ mirrored −V (even)")):
    Z = D[f"cons_{mode}"]
    sel = grid > 0
    a.plot(grid[sel], Z[sel], color=c, lw=0.7, label=lab)
    a.axhline(S["consensus"][mode]["thr"], color=c, lw=0.4, ls="--")
    a.axhline(-S["consensus"][mode]["thr"], color=c, lw=0.4, ls="--")
a.set_xlabel("bias $V$ (V)")
a.set_ylabel("Stouffer $Z$")
a.set_ylim(-6, 6)
a.legend(frameon=False, fontsize=5.5, loc="lower left", ncol=1)
panel(a, "c", "Consensus over locations")

a = fig.add_subplot(gs[1, 1])
lags, ps, pc = D["lags"] * 1e3, D["pol_same"], D["pol_cross"]
lo, hi = np.nanpercentile(pc, [0.5, 99.5], axis=0)
a.fill_between(lags, np.nanmean(pc, 0) - 2.58 * np.nanstd(pc, 0) / np.sqrt(len(ps)),
               np.nanmean(pc, 0) + 2.58 * np.nanstd(pc, 0) / np.sqrt(len(ps)), color="0.88", lw=0,
               label="unrelated pairs, 99%")
a.plot(lags, np.nanmean(ps, 0), color=BLUE, lw=0.9, label="same location")
a.axhline(0, color="k", lw=0.4)
a.set_xlabel("displacement of −V half (mV)")
a.set_ylabel("$r$ (+V, mirrored −V)")
a.legend(frameon=False, fontsize=5.6, loc="lower left")
panel(a, "d", "Polarity test, Type 2")

a = fig.add_subplot(gs[1, 2])
tru = BEN[BEN.kind == "truth"].copy()
bins = np.array([1, 2, 5, 10, 20, 40])
tru["ab"] = pd.cut(np.abs(tru.amp), bins)
for meth, c, lab in (("pipeline", BLUE, "this work"), ("no-cleaning", PURPLE, "no artefact removal"),
                     ("legacy SG", ORANGE, "SG before/after")):
    g = tru[tru.method == meth].groupby("ab", observed=False).detected.mean()
    a.plot(np.sqrt(bins[:-1] * bins[1:]), g.values * 100, "o-", color=c, ms=3, lw=0.9, label=lab)
a.set_xscale("log")
a.set_xticks([1, 2, 5, 10, 20, 40])
a.set_xticklabels(["1", "2", "5", "10", "20", "40"])
a.minorticks_off()
a.set_xlabel("injected amplitude (V$^{-1}$)")
a.set_ylabel("detected (%)")
a.set_ylim(0, 100)
a.legend(frameon=False, fontsize=5.6, loc="upper left")
panel(a, "e", "Benchmark at 5% FWER")
save(fig, "S_mos2_tests")

# =============================================================== SI: physical limits and relaxation summary
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.5, height_ratios=[1, 1.1])
a = fig.add_subplot(gs[0, :])
bands = [("MoS$_2$ lattice (Raman/IR)", 227, 470), ("sapphire phonons", 378, 751),
         ("MoO$_3$ (Mo–O–Mo, Mo=O)", 666, 995), ("sulfate (ν$_1$, ν$_3$)", 981, 1104),
         ("H$_2$O bend", 1600, 1660), ("CO$_2$ ν$_3$", 2330, 2365), ("S–H stretch", 2550, 2626),
         ("C–H stretch", 2850, 2960), ("O–H stretch", 3000, 3600)]
for i, (lab, lo_, hi_) in enumerate(bands):
    a.plot([lo_, hi_], [i, i], color="k", lw=3, solid_capstyle="butt")
    a.text(hi_ + 40, i, lab, fontsize=6, va="center")
th = S["thermal_fwhm_cm1"]
y0 = len(bands) + 0.6
a.plot([200, 200 + th], [y0, y0], color=ORANGE, lw=3, solid_capstyle="butt")
a.text(200 + th + 40, y0, f"room-temperature inelastic line width 5.4$k_BT$ ≈ {th:.0f} cm$^{{-1}}$",
       fontsize=6, va="center", color=ORANGE)
res = NZ.resolution_mV.values / 1e3 * CM_PER_V
a.plot([200, 200 + np.median(res)], [y0 + 1, y0 + 1], color=BLUE, lw=3, solid_capstyle="butt")
a.text(200 + np.median(res) + 40, y0 + 1,
       f"resolution supported by the MoS$_2$ data, median {np.median(res):.0f} cm$^{{-1}}$ "
       f"(range {res.min():.0f}–{res.max():.0f})", fontsize=6, va="center", color=BLUE)
a.set_xlim(0, 4100)
a.set_ylim(-0.8, y0 + 1.8)
a.set_yticks([])
a.spines["left"].set_visible(False)
a.set_xlabel("$\\tilde\\nu$ (cm$^{-1}$)")
panel(a, "a", "Reference positions against the achievable line width")

a = fig.add_subplot(gs[1, 0])
XY = {"pos4": (0.17, 0.82), "pos1": (0.83, 0.85), "centre1": (0.5, 0.67), "centre2": (0.52, 0.32),
      "pos3": (0.18, 0.24), "pos2": (0.85, 0.22)}
a.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=0.6))
tr = TR.set_index("location")
amp = {k: relax_at_1s(f"type2/{k}") for k in XY}
vmax = max(amp.values())
for k, (x, y) in XY.items():
    sc = a.scatter([x], [y], s=20 + 140 * amp[k] / vmax, color=PURPLE if "centre" in k else BLUE, alpha=0.8,
                   edgecolor="white", lw=0.5)
    a.text(x, y - 0.13, f"{short('type2/' + k)}\n{amp[k]:.0f} mV", fontsize=5.5, ha="center", va="top")
a.set_xlim(-0.05, 1.05)
a.set_ylim(-0.12, 1.05)
a.set_aspect("equal")
a.axis("off")
panel(a, "b", "Relaxation at 1 s, Type 2")

a = fig.add_subplot(gs[1, 1])
for n in NAMES:
    r = tr.loc[n]
    a.scatter(r.tau1_s, r.tau2_s, color=colour(n), s=14, edgecolor="white", lw=0.4)
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(0.5, 4)
a.set_ylim(4, 120)
a.set_xticks([0.5, 1, 2, 4])
a.set_xticklabels(["0.5", "1", "2", "4"])
a.set_yticks([5, 10, 20, 50, 100])
a.set_yticklabels(["5", "10", "20", "50", "100"])
a.minorticks_off()
a.set_xlabel("$\\tau_1$ (s)")
a.set_ylabel("$\\tau_2$ (s)")
a.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=3.5, label=l) for c, l in
                  ((ORANGE, "Type 1"), (BLUE, "Type 2 periphery"), (PURPLE, "Type 2 centre"))],
         frameon=False, fontsize=5.6, loc="lower right")
panel(a, "c", "Two trap time constants")

a = fig.add_subplot(gs[1, 2])
n = "type2/pos4"
Ic, Vn = get("Iclean", n), get("V", n)
m = (Vn > -0.05) & (Vn < 0.25)
for k in range(Ic.shape[0]):
    p = np.polyfit(Vn[m], Ic[k][m], 1)
    a.plot(Vn[m], (Ic[k][m] - np.polyval(p, Vn[m])) * 1e12, lw=0.6, color=BLUE if k else BLUE, alpha=0.4 + 0.12 * k)
a.axvline(0.1025, color=GREY, lw=0.5, ls=":")
a.set_xlabel("bias $V$ (V)")
a.set_ylabel("$I_k$ − linear (pA)")
panel(a, "d", "Kink at P4, five repeats")
save(fig, "S_mos2_limits")


# =============================================================== Fig. 4: resolution and sensitivity
TRP = pd.read_csv(f"{OUT}/traps.csv").set_index("location")
TRS = np.load(f"{OUT}/traps_series.npz")
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 3, hspace=0.8, wspace=0.45)
a = fig.add_subplot(gs[0, 0])
for n in NAMES:
    d = FRC[FRC.location == n]
    a.semilogx(d.f_per_V, d.frc, color=colour(n), lw=0.7, alpha=0.9)
a.axhline(1 / 3, color="k", lw=0.5, ls="--")
a.set_ylim(-0.6, 1.05)
a.set_xlim(1, 250)
a.set_xlabel("bias frequency (cycles V$^{-1}$)")
a.set_ylabel("half-data correlation")
a.legend(handles=[Line2D([], [], color=c, lw=1, label=l) for c, l in
                  ((ORANGE, "Type 1"), (BLUE, "Type 2 periphery"), (PURPLE, "Type 2 centre"))],
         frameon=False, fontsize=5.6, loc="upper right")
panel(a, "a", "Resolution from the repeats")

a = fig.add_subplot(gs[0, 1:])
n = "type2/pos2"
x, Vn, valid, leg = get("x", n), get("V", n), get("valid", n), get("legacy", n)
mu, se = x.mean(0), x.std(0, ddof=1) / np.sqrt(x.shape[0])
a.plot(Vn[valid], leg[valid] - np.median(leg[valid]), color=GREY, lw=0.6, label="conventional (SG)")
a.fill_between(Vn[valid], (mu - 2 * se)[valid], (mu + 2 * se)[valid], color=BLUE, alpha=0.25, lw=0)
a.plot(Vn[valid], mu[valid], color=BLUE, lw=1.0, label="this work, mean ± 2 s.e. of repeats")
a.axhline(0, color="k", lw=0.4)
a.set_xlabel("bias $V$ (V)")
a.set_ylabel("$(d^2I/dV^2)/(dI/dV)$ (V$^{-1}$)")
a.legend(frameon=False, fontsize=6, loc="lower right", ncol=2)
cm_axis(a)
a.set_title("$\\bf{b}$  Location P2: conventional spectrum and artefact-free estimate", loc="left", pad=16)

tru = BEN[BEN.kind == "truth"].copy()
for k, (col, bins, lab, sel) in enumerate((
        ("amp", np.array([1, 2, 5, 10, 20, 40]), "injected amplitude (V$^{-1}$)", None),
        ("fwhm", np.array([0.02, 0.04, 0.07, 0.1, 0.15]), "injected FWHM (mV), amplitude ≥ 5 V$^{-1}$", 5))):
    a = fig.add_subplot(gs[1, k])
    tt = tru if sel is None else tru[np.abs(tru.amp) >= sel]
    xb = pd.cut(np.abs(tt[col]), bins)
    for meth, c, lb in (("pipeline", BLUE, "this work"), ("no-cleaning", PURPLE, "no artefact removal"),
                        ("legacy SG", ORANGE, "SG before/after")):
        g = tt[tt.method == meth].groupby(xb[tt.method == meth], observed=False).detected.mean()
        xc = np.sqrt(bins[:-1] * bins[1:]) * (1e3 if col == "fwhm" else 1)
        a.plot(xc, g.values * 100, "o-", color=c, ms=3, lw=0.9, label=lb)
    a.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 40] if col == "amp" else [20, 40, 70, 100, 150]
    a.set_xticks(ticks)
    a.set_xticklabels([str(v) for v in ticks])
    a.minorticks_off()
    a.set_xlabel(lab)
    a.set_ylabel("detected (%)")
    a.set_ylim(0, 100)
    if k == 0:
        a.legend(frameon=False, fontsize=5.6, loc="upper left")
    panel(a, "cd"[k], "Sensitivity at 5% FWER" if k == 0 else "")
a = fig.add_subplot(gs[1, 2])
b = S["benchmark"]
meths = [("pipeline", BLUE, "this\nwork"), ("no-cleaning", PURPLE, "no artefact\nremoval"), ("legacy SG", ORANGE, "SG before/\nafter")]
a.bar(range(3), [100 * b[m]["sens_amp_ge10"] for m, _, _ in meths], color=[c for _, c, _ in meths], width=0.6)
for i, (m, _, _) in enumerate(meths):
    a.text(i, 100 * b[m]["sens_amp_ge10"] + 2, f"{100 * b[m]['sens_amp_ge10']:.0f}%", ha="center", fontsize=6)
a.set_xticks(range(3))
a.set_xticklabels([l for _, _, l in meths], fontsize=5.8)
a.set_ylim(0, 100)
a.set_ylabel("detected, amplitude ≥ 10 V$^{-1}$ (%)")
panel(a, "e", "")
save(fig, "fig_mos2_statistics")

# =============================================================== Fig. 5: charge-trap dynamics
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.5)
sub = gs[0, :2].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.08)
a = fig.add_subplot(sub[0])
n = "type2/pos2"
t_, V_, I_, M_, rep_, ok_ = TRS[n.replace("/", "__")]
sel = rep_ < 3
a.plot(t_[sel], I_[sel] * 1e9, color="0.6", lw=2.2, label="measured")
a.plot(t_[sel], M_[sel] * 1e9, color=BLUE, lw=0.7, label="conductance + trap branches")
a.set_ylabel("$I$ (nA)")
plt.setp(a.get_xticklabels(), visible=False)
a.legend(frameon=False, fontsize=6, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.98))
a2 = fig.add_subplot(sub[1], sharex=a)
a2.plot(t_[sel], (I_[sel] - M_[sel]) * 1e12, color=GREY, lw=0.4)
a2.axhline(0, color="k", lw=0.4)
a2.set_ylabel("resid. (pA)", fontsize=6)
a2.set_xlabel("time in the measurement session (s)")
a.set_title("$\\bf{a}$  Location P2, first three repeats", loc="left", pad=14)

a = fig.add_subplot(gs[0, 2])
order = [n for n in NAMES if n.startswith("type2")]  # Type 1 sweeps: Supplementary Table 7.5
y = np.arange(len(order))
vals = [TRP.loc[n, "r2_out_of_sample"] for n in order]
a.barh(y, np.clip(vals, 0, 1), color=[colour(n) for n in order], height=0.6)
for yy, v in zip(y, vals):
    if v < 0:
        a.text(0.02, yy, f"{v:.1f}", va="center", fontsize=5.5)
a.set_yticks(y)
a.set_yticklabels([short(n) for n in order], fontsize=6)
a.set_xlim(0, 1)
a.invert_yaxis()
a.set_xlabel("$R^2$, second half predicted\nfrom a fit to the first half")
panel(a, "b", "Prediction of held-out repeats")

t2 = [n for n in NAMES if n.startswith("type2")]
a = fig.add_subplot(gs[1, 0])
for n in t2:
    r = TRP.loc[n]
    a.scatter([r.tau1_s], [r.C1_nF], color=colour(n), marker="o", s=16, edgecolor="white", lw=0.4)
    a.scatter([r.tau2_s], [r.C2_nF], color=colour(n), marker="s", s=16, edgecolor="white", lw=0.4)
    dy = {"P1": -4, "P2": 4}.get(short(n), 0)
    a.annotate(short(n), (r.tau2_s, r.C2_nF), xytext=(4, dy), textcoords="offset points", fontsize=5.3, va="center")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(0.5, 120)
a.set_ylim(0.08, 40)
a.set_xticks([1, 3, 10, 30, 100])
a.set_xticklabels(["1", "3", "10", "30", "100"])
a.minorticks_off()
a.set_xlabel("trap time constant τ (s)")
a.set_ylabel("trapped charge per volt, $C$ (nF)")
a.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=3.5, label="fast population"),
                  Line2D([], [], marker="s", ls="", color=GREY, ms=3.5, label="slow population")],
         frameon=False, fontsize=5.6, loc="upper left")
sec = a.secondary_yaxis("right", functions=(lambda c: c * 1e-9 / 1.602e-19 / 1e9, lambda nn: nn * 1e9 * 1.602e-19 / 1e-9))
sec.set_ylabel("carriers per volt (10$^9$)", fontsize=6)
sec.tick_params(labelsize=5.5)
panel(a, "c", "Two trap populations")

for k, (col, lab) in enumerate((("C1_nF", "fast traps, $C_1$"), ("C2_nF", "slow traps, $C_2$"))):
    a = fig.add_subplot(gs[1, 1 + k])
    XY = {"pos4": (0.17, 0.82), "pos1": (0.83, 0.85), "centre1": (0.5, 0.67), "centre2": (0.52, 0.32),
          "pos3": (0.18, 0.24), "pos2": (0.85, 0.22)}
    a.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=0.6))
    vals = {kk: TRP.loc[f"type2/{kk}", col] for kk in XY}
    vmax = max(vals.values())
    for kk, (xx, yy) in XY.items():
        a.scatter([xx], [yy], s=10 + 130 * vals[kk] / vmax, color=PURPLE if "centre" in kk else BLUE, alpha=0.8,
                  edgecolor="white", lw=0.5)
        a.text(xx + 0.09, yy, f"{short('type2/' + kk)}\n{vals[kk]:.2f} nF" if vals[kk] < 1 else
               f"{short('type2/' + kk)}\n{vals[kk]:.1f} nF", fontsize=5.3, ha="left", va="center")
    a.set_xlim(-0.05, 1.05)
    a.set_ylim(-0.15, 1.05)
    a.set_aspect("equal")
    a.axis("off")
    panel(a, "de"[k], lab)
save(fig, "fig_mos2_traps")

# =============================================================== SI: trap fits of every location
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.75, wspace=0.45))
for a, n in zip(axs.ravel(), NAMES):
    t_, V_, I_, M_, rep_, ok_ = TRS[n.replace("/", "__")]
    a.plot(t_, (I_ - M_) * 1e12, color=GREY, lw=0.3)
    r = TRP.loc[n]
    a.set_title(f"{short(n)}  $R^2$ = {r.r2_in:.2f} / {r.r2_out_of_sample:.2f}", fontsize=6.3)
    a.set_xlabel("session time (s)", fontsize=6)
    a.set_ylabel("residual (pA)", fontsize=6)
axs.ravel()[-1].axis("off")
save(fig, "S_mos2_traps")

# =============================================================== SI: pooled test for a feature common to the film
PO = pd.read_csv(f"{OUT}/pooled.csv")
POS = json.load(open(f"{OUT}/pooled_summary.json"))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.4))
for a, (w, c) in zip(axs, ((30, BLUE), (60, PURPLE), (140, ORANGE))):
    d = PO[(PO.width_mV.round() == w) & (PO.n_locations >= 3)]
    a.plot(d.V0, d.amplitude / d.se, color=c, lw=0.8)
    thr = POS[f"{w}mV"]["main"]["z_threshold_fwer05"]
    a.axhline(thr, color="k", lw=0.4, ls="--")
    a.axhline(-thr, color="k", lw=0.4, ls="--")
    a.axvspan(-0.25, -0.10, color="0.92", lw=0)
    a.set_xlabel("$V_0$ (V)")
    a.set_ylabel("pooled $z$")
    a.set_title(f"width {w} mV", fontsize=6.5)
save(fig, "S_mos2_pooled")

# =============================================================== Supplementary figures
# S7.1 raw I-V of every repeat
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.7, wspace=0.45))
for a, n in zip(axs.ravel(), NAMES):
    I, Vn = get("Iraw", n), get("V", n)
    for k in range(I.shape[0]):
        a.plot(Vn, I[k] * 1e9, lw=0.35, color=colour(n), alpha=0.6)
    g = get("glitch", n)
    kk, ii = np.where(g)
    a.plot(Vn[ii], I[kk, ii] * 1e9, "x", color="k", ms=2, mew=0.4)
    a.set_title(f"{short(n)}  ({I.shape[0]} repeats)", fontsize=6.5)
    a.set_xlabel("$V$ (V)", fontsize=6)
    a.set_ylabel("$I$ (nA)", fontsize=6)
axs.ravel()[-1].axis("off")
axs.ravel()[-1].legend(handles=legend_types + [Line2D([], [], marker="x", ls="", color="k", label="flagged glitch")],
                       frameon=False, fontsize=6, loc="center")
save(fig, "S_mos2_raw")

# S7.2 PSDs
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.7, wspace=0.45))
for a, n in zip(axs.ravel(), NAMES):
    f, Pr, Pc = get("psd", n)
    a.loglog(f[1:], Pr[1:] * 1e24, color=GREY, lw=0.7)
    a.loglog(f[1:], Pc[1:] * 1e24, color=colour(n), lw=0.7)
    r = NZ.set_index("location").loc[n]
    a.set_title(f"{short(n)}  γ = {r.psd_gamma:.1f}", fontsize=6.5)
    a.set_xlabel("Hz", fontsize=6)
    a.set_ylabel("pA$^2$ Hz$^{-1}$", fontsize=6)
axs.ravel()[-1].axis("off")
axs.ravel()[-1].legend(handles=[Line2D([], [], color=GREY, label="measured"),
                                Line2D([], [], color=BLUE, label="glitches and lines removed")],
                       frameon=False, fontsize=6, loc="center")
save(fig, "S_mos2_psd")

# S7.3 FRC per location, raw vs cleaned
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.7, wspace=0.45))
for a, n in zip(axs.ravel(), NAMES):
    d = FRC[FRC.location == n]
    a.semilogx(d.f_per_V, d.frc_raw, color=GREY, lw=0.7)
    a.semilogx(d.f_per_V, d.frc, color=colour(n), lw=0.8)
    a.semilogx(d.f_per_V, 3 / np.sqrt(d.n_coef), color="k", lw=0.4, ls=":")
    a.axhline(1 / 3, color="k", lw=0.4, ls="--")
    fc = NZ.set_index("location").loc[n, "frc_cutoff_per_V"]
    a.axvline(fc, color=colour(n), lw=0.5, ls="--")
    a.set_ylim(-0.7, 1.05)
    a.set_title(f"{short(n)}  cut-off {fc:.0f} V$^{{-1}}$", fontsize=6.5)
    a.set_xlabel("cycles V$^{-1}$", fontsize=6)
axs.ravel()[-1].axis("off")
axs.ravel()[-1].legend(handles=[Line2D([], [], color=GREY, label="measured repeats"),
                                Line2D([], [], color=BLUE, label="after artefact removal"),
                                Line2D([], [], color="k", ls="--", lw=0.5, label="FRC = 1/3 (cut-off)"),
                                Line2D([], [], color="k", ls=":", lw=0.5, label="3σ of noise")],
                       frameon=False, fontsize=6, loc="center")
save(fig, "S_mos2_frc")

# S7.4 spectra: conventional vs this work, with +-2 SE
fig, axs = plt.subplots(6, 2, figsize=(7.2, 8.6), gridspec_kw=dict(hspace=0.75, wspace=0.3))
for a, n in zip(axs.ravel(), NAMES):
    x, Vn, valid, leg = get("x", n), get("V", n), get("valid", n), get("legacy", n)
    mu, se = x.mean(0), x.std(0, ddof=1) / np.sqrt(x.shape[0])
    a.plot(Vn[valid], leg[valid] - np.median(leg[valid]), color=GREY, lw=0.5, label="conventional SG")
    a.fill_between(Vn[valid], (mu - 2 * se)[valid], (mu + 2 * se)[valid], color=colour(n), alpha=0.25, lw=0)
    a.plot(Vn[valid], mu[valid], color=colour(n), lw=0.8, label="this work ±2 s.e.")
    lim = np.nanpercentile(np.abs(np.r_[leg[valid] - np.median(leg[valid]), mu[valid]]), 99.5) * 1.2
    a.set_ylim(-lim, lim)
    a.set_title(short(n), fontsize=6.5, loc="left")
    a.set_ylabel("V$^{-1}$", fontsize=6)
axs.ravel()[-1].axis("off")
axs.ravel()[-1].legend(handles=[Line2D([], [], color=GREY, label="conventional: SG, two gradients, SG"),
                                Line2D([], [], color=BLUE, label="this work, mean ± 2 s.e. of repeats")],
                       frameon=False, fontsize=6.5, loc="center")
for a in axs[-1]:
    a.set_xlabel("bias $V$ (V)", fontsize=6)
save(fig, "S_mos2_spectra")

# S7.5 transient fits in time
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.7, wspace=0.5))
for a, n in zip(axs.ravel(), NAMES):
    Ic, Vn, tn = get("Iclean", n).mean(0) * 1e12, get("V", n), get("t", n)
    r = tr.loc[n]
    late = tn > 60
    g, c = np.polyfit(Vn[late], Ic[late], 1)
    a.plot(tn, Ic - g * Vn - c, color=colour(n), lw=0.6)
    a.set_xscale("log")
    a.set_xlim(0.15, tn.max())
    a.set_title(f"{short(n)}  τ = {r.tau1_s:.1f}, {r.tau2_s:.0f} s", fontsize=6.5)
    a.set_xlabel("t (s)", fontsize=6)
    a.set_ylabel("pA", fontsize=6)
axs.ravel()[-1].axis("off")
save(fig, "S_mos2_transient")

# S7.6 benchmark details
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.45))
wb = np.array([0.02, 0.04, 0.07, 0.1, 0.15])
tru["wb"] = pd.cut(tru.fwhm, wb)
for meth, c, lab in (("pipeline", BLUE, "this work"), ("no-cleaning", PURPLE, "no artefact removal"),
                     ("legacy SG", ORANGE, "SG before/after")):
    tm = tru[(tru.method == meth) & (np.abs(tru.amp) >= 5)]
    g = tm.groupby("wb", observed=False).detected.mean()
    axs[0].plot(np.sqrt(wb[:-1] * wb[1:]) * 1e3, g.values * 100, "o-", color=c, ms=3, lw=0.9, label=lab)
    pe = tru[(tru.method == meth) & tru.detected].pos_err * 1e3
    axs[1].hist(pe, bins=np.linspace(0, 40, 21), histtype="step", color=c, lw=0.9, density=True)
axs[0].set_xlabel("injected FWHM (mV), amplitude ≥ 5 V$^{-1}$")
axs[0].set_ylabel("detected (%)")
axs[0].set_ylim(0, 100)
axs[0].legend(frameon=False, fontsize=5.6)
panel(axs[0], "a", "")
axs[1].set_xlabel("position error (mV)")
axs[1].set_ylabel("density")
panel(axs[1], "b", "")
fw = S["signflip_fwer_by_location"]
axs[2].bar(range(len(NAMES)), [100 * fw[n] for n in NAMES], color=[colour(n) for n in NAMES])
axs[2].axhline(5, color="k", lw=0.5, ls="--")
axs[2].set_xticks(range(len(NAMES)))
axs[2].set_xticklabels([short(n) for n in NAMES], rotation=90, fontsize=5.5)
axs[2].set_ylabel("feature-free sets with a\nfalse feature (%)")
panel(axs[2], "c", "")
save(fig, "S_mos2_benchmark")


# =============================================================== LaTeX tables for the SI
def fmt(x, d=1):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


rows = []
for n in NAMES:
    r = NZ.set_index("location").loc[n]
    t_ = tr.loc[n]
    rows.append(f"{short(n)} & {int(r.repeats)} & {r.step_mV:.2f} & {r.rate_mV_per_s:.1f} & {r.G_nS:.2f} & "
                f"{fmt(r.G_drift_pct_per_h, 0)} & {fmt(r.V_zero_current_mV, 0)} & {r.lsb_pA:g} & "
                f"{r.glitch_per_1000:.1f} & {int(r.repeats_with_line)}/{int(r.repeats)} & "
                f"{fmt(r.f_line_median_Hz, 3)} & {fmt(r.line_amp_pA, 1)} & {r.fluct_rms_pA:.1f} & "
                f"{r.psd_gamma:.1f} \\\\")
tab1 = "\n".join(rows)
rows = []
for n in NAMES:
    r = NZ.set_index("location").loc[n]
    t_ = tr.loc[n]
    rows.append(f"{short(n)} & {relax_at_1s(n):.0f} & {t_.tau1_s:.1f} & {t_.tau2_s:.0f} & "
                f"{t_.rms_resid_pA:.2f} & {r.frc_cutoff_per_V:.1f} & {r.resolution_mV:.0f} & "
                f"{r.resolution_mV * CM_PER_V / 1e3:.0f} & {r.kernel_fwhm_mV:.0f} & {r.maxt_threshold:.1f} & "
                f"{100 * r.legacy_var_frac_lines:.0f} & {100 * r.legacy_var_frac_glitches:.0f} \\\\")
tab2 = "\n".join(rows)
rows = []
for _, c in CAND.iterrows():
    pol = c.polarity if isinstance(c.polarity, str) and c.polarity != "n/a" else "--"
    if pol in ("odd", "even"):
        pol += f" ({c.p_polarity:.2f})"
    rows.append(f"{short(c.location)} & {c.V * 1e3:.0f} & {c.wavenumber_cm1:.0f} & {c.sigma_V_mV:.1f} "
                f"({c.sigma_V_mV * CM_PER_V / 1e3:.0f}) & {c.fwhm_mV:.0f} & {c.amplitude_per_V:.2f} & "
                f"{c.tstat:.1f} & {c.p_fwer:.2f} & {pol} & {int(c.n_other_locations)}/{int(c.n_tested)} & "
                f"{c.q_recurrence:.2f} \\\\")
tab3 = "\n".join(rows)
b = S["benchmark"]
tab4 = "\n".join(
    f"{lab} & {100 * b[m]['sensitivity']:.0f} & {100 * b[m]['sens_amp_ge5']:.0f} & {100 * b[m]['sens_amp_ge10']:.0f} & "
    f"{b[m]['pos_err_mV']:.1f} & {100 * b[m]['spectra_with_false']:.1f} \\\\"
    for m, lab in (("pipeline", "This work"), ("no-cleaning", "Same smoothing, no artefact removal"),
                   ("legacy SG", "Savitzky--Golay before and after differentiation")))
with open(TEX, "w") as fh:
    fh.write("% generated by make_mos2_figures.py -- do not edit by hand\n")
    for name, body in (("MOSTABNOISE", tab1), ("MOSTABRES", tab2), ("MOSTABCAND", tab3), ("MOSTABBENCH", tab4)):
        fh.write(f"\\newcommand{{\\{name}}}{{%\n{body}\n}}\n")
print("wrote", sorted(f for f in os.listdir(FIG) if "mos2" in f), TEX)
