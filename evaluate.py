"""Benchmark the U-Net denoisers against classical smoothing.

1. Synthetic test set (ground truth known): MSE and peak-detection F1 per branch, plus
   recall of large features and the precision of each confidence tier.
2. Real data, cross-branch agreement: on spectra whose branches could be aligned, each
   branch is denoised *independently* and we measure how well peaks found in one branch
   reproduce in the other, minus the chance level from randomly paired spectra.
3. Real data, retention of obvious features: how many prominent features of the raw
   (lightly smoothed) data survive denoising.

    .venv/bin/python evaluate.py
"""
from __future__ import annotations

import json

import numpy as np

from iets.analysis import branch_average, classify_peaks, detect, f1, gauss, load_model, match, savgol, unet
from iets.pipeline import prepare_all

THRESHOLDS = np.arange(0.4, 6.01, 0.2)
SINGLE_THRESHOLDS = np.arange(1.0, 12.01, 0.5)
OBVIOUS = 4.0   # prominence (noise units) of an "obvious" raw feature
rng = np.random.default_rng(0)

d = np.load("models/synth_test.npz")
X, Y = d["X"][:2000], d["Y"][:2000]
u1, u2 = load_model("models/unet_1branch.pt", 1), load_model("models/unet_2branch.pt", 2)

single = {
    "raw": lambda Z: Z,
    "savitzky-golay": savgol,
    "gaussian": gauss,
    "unet-1branch": lambda Z: unet(u1, Z),
}
both = {
    "branch-average": branch_average,
    "unet-2branch": lambda Z: unet(u2, Z),
}

truth = [[detect(y, 1.0) for y in yy] for yy in Y]
half = len(X) // 2
tune, test = range(0, half, 2), range(half, len(X))

print("== Synthetic test set (1000 held-out spectra, both branches scored)")
res = {}
outs = {k: np.concatenate([fn(X[:, :1]), fn(X[:, 1:])], axis=1) for k, fn in single.items()}
outs.update({k: fn(X) for k, fn in both.items()})


def mean_f1(P, idx, thr):
    return np.mean([f1(detect(P[i, c], thr), truth[i][c])[0] for i in idx for c in (0, 1)])


for k, P in outs.items():
    best = max(THRESHOLDS, key=lambda t: mean_f1(P, tune, t))
    r = [f1(detect(P[i, c], best), truth[i][c]) for i in test for c in (0, 1)]
    big = [(len(match([t for t in truth[i][c] if abs(t[1]) >= 5], detect(P[i, c], best), 24)),
            sum(abs(t[1]) >= 5 for t in truth[i][c])) for i in test for c in (0, 1)]
    res[k] = dict(mse=float(np.mean((P[half:] - Y[half:]) ** 2)), threshold=float(best),
                  f1=float(np.mean([a[0] for a in r])), precision=float(np.mean([a[1] for a in r])),
                  recall=float(np.mean([a[2] for a in r])),
                  recall_large=float(sum(a for a, _ in big) / max(1, sum(b for _, b in big))))
    print(f"  {k:16s} MSE {res[k]['mse']:.3f}  F1 {res[k]['f1']:.3f}  P {res[k]['precision']:.3f}  "
          f"R {res[k]['recall']:.3f}  R(large, >=5x noise) {res[k]['recall_large']:.3f}  (thr {best:.1f})")

# confidence tiers of the final model
thr = res["unet-2branch"]["threshold"]
P2 = outs["unet-2branch"]


def tier_precision(single_thr, idx):
    conf, sing = [0, 0], [0, 0]
    for i in idx:
        for p in classify_peaks(P2[i], thr, single_thr):
            if p["confidence"] == "confirmed":
                ok = any(abs(p["wavenumber"] - t[0]) <= 32 for c in (0, 1) for t in truth[i][c])
                conf[0] += ok
                conf[1] += 1
            else:
                c = 0 if "+V" in p["confidence"] else 1
                ok = any(abs(p["wavenumber"] - t[0]) <= 24 for t in truth[i][c])
                sing[0] += ok
                sing[1] += 1
    return conf[0] / max(1, conf[1]), sing[0] / max(1, sing[1]), conf[1], sing[1]


single_thr = next((t for t in SINGLE_THRESHOLDS if tier_precision(t, tune)[1] >= 0.9), SINGLE_THRESHOLDS[-1])
TENTATIVE = 2.0
pc, ps, nc, ns = tier_precision(single_thr, test)
_, pt_all, _, nt_all = tier_precision(TENTATIVE, test)
pt = (pt_all * nt_all - ps * ns) / max(1, nt_all - ns)   # precision of the 2..single_thr band alone
res["tiers"] = dict(threshold=thr, single_threshold=float(single_thr), tentative_threshold=TENTATIVE,
                    confirmed_precision=pc, single_precision=ps, tentative_precision=pt,
                    n_confirmed=nc, n_single=ns, n_tentative=nt_all - ns)
print(f"  tiers: confirmed precision {pc:.3f} (n={nc}); single-half {ps:.3f} (n={ns}, >= {single_thr:.1f}); "
      f"tentative {pt:.3f} (n={nt_all - ns}, {TENTATIVE:.1f}-{single_thr:.1f})")

# ---------------------------------------------------------------- real data
S, R, Mk, SC, A = prepare_all()
dev = np.array([s.device for s in S])
acc = np.array([a.accepted for a in A])
print(f"\n== Real data: cross-branch peak agreement on the {acc.sum()} aligned spectra "
      "(each branch denoised independently)")
real = {}
ai = np.where(acc)[0]
for k, fn in single.items():
    P = np.concatenate([fn(R[:, :1]), fn(R[:, 1:])], axis=1)
    t = res[k]["threshold"]
    pk = {i: [detect(P[i, c], t) for c in (0, 1)] for i in ai}
    same = np.array([f1(*pk[i], tol=32)[0] for i in ai])
    null = [f1(pk[i][0], pk[j][1], tol=32)[0] for i, j in rng.choice(ai, (2000, 2)) if i != j]
    real[k] = {dv: float(same[dev[ai] == dv].mean() - np.mean(null)) for dv in ("impure", "pure")}
    real[k].update(same=float(same.mean()), null=float(np.mean(null)),
                   peaks_per_branch=float(np.mean([len(a) + len(b) for a, b in pk.values()]) / 2))
    print(f"  {k:16s} same-pair {same.mean():.3f}  random-pair {np.mean(null):.3f}  "
          f"excess {same.mean() - np.mean(null):+.3f}  peaks/branch {real[k]['peaks_per_branch']:.1f}")

print("\n== Real data: agreement at matched operating points (equal peaks per branch)")
matched = {}
for k, fn in single.items():
    P = np.concatenate([fn(R[:, :1]), fn(R[:, 1:])], axis=1)
    cache = {}
    for t in np.arange(0.4, 12.0, 0.1):
        pk = {i: [detect(P[i, c], t) for c in (0, 1)] for i in ai}
        cache[round(float(t), 1)] = (np.mean([len(a) + len(b) for a, b in pk.values()]) / 2, pk)
    matched[k] = {}
    for target in (2, 3, 4, 5, 6):
        n, pk = min(cache.values(), key=lambda v: abs(v[0] - target))
        same = np.mean([f1(*pk[i], tol=32)[0] for i in ai])
        null = np.mean([f1(pk[i][0], pk[j][1], tol=32)[0] for i, j in rng.choice(ai, (1500, 2)) if i != j])
        matched[k][target] = float(same - null)
    print(f"  {k:16s} " + "  ".join(f"{t} pk: {v:+.3f}" for t, v in matched[k].items()))

print(f"\n== Real data: retention of obvious raw features (prominence >= {OBVIOUS:.0f}x noise "
      "after light Gaussian smoothing)")
D = unet(u2, R)
G = gauss(R)
kept = {"all": [0, 0], "both-branch": [0, 0]}
for i in range(len(R)):
    peaks = classify_peaks(D[i], thr, single_thr, TENTATIVE)
    reported = [p["wavenumber"] for p in peaks]
    obv = [detect(G[i, c], OBVIOUS) for c in (0, 1)]
    shared = [w for w, _ in obv[0] if any(abs(w - w2) <= 32 for w2, _ in obv[1])]
    for c in (0, 1):
        for w, _ in obv[c]:
            hit = any(abs(w - r) <= 32 for r in reported)
            kept["all"][0] += hit
            kept["all"][1] += 1
    for w in shared:
        kept["both-branch"][0] += any(abs(w - r) <= 32 for r in reported)
        kept["both-branch"][1] += 1
for k2, (a, b) in kept.items():
    print(f"  {k2:12s} obvious features retained: {a}/{b} = {a / max(1, b):.2f}")
res["retention"] = {k2: dict(kept=a, total=b) for k2, (a, b) in kept.items()}

json.dump(dict(synthetic=res, real=real, matched=matched), open("outputs/metrics.json", "w"), indent=2)
