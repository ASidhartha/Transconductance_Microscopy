"""Can the purified and impure rGO films be told apart from their spectra alone?

    .venv/bin/python rgo_classifier.py       (after train.py / apply.py; ~2 min)

A linear classifier (standardise -> PCA -> L2 logistic regression) is evaluated by
leave-one-electrode-pair-out cross-validation: every fold holds out both films' spectra of one
electrode position, so the classifier has to separate two spectra recorded at the same place on
the grid. The null distribution comes from swapping the two labels within randomly chosen held-out
pairs. Controls: the zero-bias conductance alone, and spectra plus conductance against conductance
alone. Pair 53_54 (near-identical in both data sets) is excluded.

Writes outputs/rgo_classifier/summary.json and discriminant.csv.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iets.analysis import load_model, unet
from iets.data import duplicated_pairs
from iets.pipeline import prepare_all
from iets.preprocess import GRID

OUT = "outputs/rgo_classifier"
N_PERM = 500
N_PC = 10


def model(n_pc=N_PC):
    return make_pipeline(StandardScaler(), PCA(n_pc, random_state=0), LogisticRegression(C=1.0, max_iter=2000))


def cv_scores(F, y, groups, n_pc=N_PC, plain=False):
    """Held-out probability of 'impure' for every spectrum (leave one electrode position out)."""
    p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        m = (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)) if plain
             else model(min(n_pc, F.shape[1])))
        m.fit(F[~te], y[~te])
        p[te] = m.predict_proba(F[te])[:, 1]
    return p


def paired_accuracy(p, y, groups):
    """Fraction of electrode positions at which the impure spectrum gets the higher score."""
    ok = []
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        if len(i) == 2 and y[i].sum() == 1:
            ok.append(p[i][y[i] == 1][0] > p[i][y[i] == 0][0])
    return float(np.mean(ok))


def perm_null(F, y, groups, rng, plain=False):
    """AUC under random label swaps within electrode positions (both films kept balanced)."""
    out = []
    ug = np.unique(groups)
    for _ in range(N_PERM):
        yp = y.copy()
        for g in ug[rng.random(len(ug)) < 0.5]:
            i = np.where(groups == g)[0]
            yp[i] = yp[i][::-1]
        out.append(roc_auc_score(yp, cv_scores(F, yp, groups, plain=plain)))
    return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(11)
    S, X, MASK, SC, A = prepare_all()
    D = unet(load_model("models/unet_2branch.pt", 2), X) * MASK
    dup = duplicated_pairs(S)
    keep = np.array([s.pair not in dup for s in S])
    y = np.array([s.device == "impure" for s in S], int)[keep]
    groups = np.array([s.pair for s in S])[keep]
    # zero-bias conductance of the averaged sweep (slope within +-50 mV)
    G = []
    for s in S:
        m = np.abs(s.voltage) < 0.05
        G.append(np.polyfit(s.voltage[m], s.current[m], 1)[0])
    logG = np.log10(np.abs(np.array(G)))[keep][:, None]
    feats = {
        "measured spectra": X[keep].reshape(keep.sum(), -1),
        "denoised spectra": D[keep].reshape(keep.sum(), -1),
    }
    res = {}
    for name, F in feats.items():
        p = cv_scores(F, y, groups)
        auc = roc_auc_score(y, p)
        null = perm_null(F, y, groups, rng)
        res[name] = dict(auc=float(auc), paired_accuracy=paired_accuracy(p, y, groups),
                         p_perm=float((1 + np.sum(null >= auc)) / (1 + N_PERM)),
                         null_auc_95=float(np.quantile(null, 0.95)))
        print(name, res[name])
    # the paper's own summary statistic: presence of a high-confidence peak in each of the nine
    # pre-defined spectral regions (Supplementary Table 5.5)
    pk = pd.read_csv("outputs/report/peak_bond_assignments.csv")
    pk = pk[(~pk.duplicated_across_devices) & pk.tier.isin(["confirmed", "single-half"])]
    regs = sorted(pk.region.dropna().unique())
    devs = np.array([s.device for s in S])[keep]
    Rf = np.array([[((pk.device == d) & (pk.pair == g) & (pk.region == r)).any() for r in regs]
                   for d, g in zip(devs, groups)], float)
    p = cv_scores(Rf, y, groups, plain=True)
    auc = roc_auc_score(y, p)
    null = perm_null(Rf, y, groups, rng, plain=True)
    res["region peak presence (9 regions)"] = dict(
        auc=float(auc), paired_accuracy=paired_accuracy(p, y, groups),
        p_perm=float((1 + np.sum(null >= auc)) / (1 + N_PERM)), null_auc_95=float(np.quantile(null, 0.95)))
    print("regions", res["region peak presence (9 regions)"])
    # controls
    pG = cv_scores(logG, y, groups, plain=True)
    res["conductance only"] = dict(auc=float(roc_auc_score(y, pG)), paired_accuracy=paired_accuracy(pG, y, groups))
    # spectra that carry no conductance information: regress log G out of every spectral feature
    F = feats["denoised spectra"]
    beta = np.linalg.lstsq(np.c_[np.ones(len(F)), logG], F, rcond=None)[0]
    Fr = F - np.c_[np.ones(len(F)), logG] @ beta
    p = cv_scores(Fr, y, groups)
    null = perm_null(Fr, y, groups, rng)
    auc = roc_auc_score(y, p)
    res["denoised spectra, conductance regressed out"] = dict(
        auc=float(auc), paired_accuracy=paired_accuracy(p, y, groups),
        p_perm=float((1 + np.sum(null >= auc)) / (1 + N_PERM)), null_auc_95=float(np.quantile(null, 0.95)))
    # sensitivity: without the ten row-1 impure files whose voltage is quantised to 1 mV (and their pairs)
    row1 = np.array([g.startswith("1") for g in groups])
    p = cv_scores(F[~row1], y[~row1], groups[~row1])
    res["denoised spectra, row 1 excluded"] = dict(auc=float(roc_auc_score(y[~row1], p)),
                                                    paired_accuracy=paired_accuracy(p, y[~row1], groups[~row1]))
    print({k: v for k, v in res.items() if "conductance" in k or "row" in k})
    # discriminant spectrum: coefficients of the full-data model mapped back to wavenumber,
    # with a bootstrap over electrode positions
    def coef(Fx, yx):
        m = model()
        m.fit(Fx, yx)
        sc, pca, lr = m.named_steps.values()
        return (pca.components_.T @ lr.coef_[0]) / sc.scale_

    F = feats["denoised spectra"]
    w0 = coef(F, y)
    ug = np.unique(groups)
    boots = []
    for _ in range(300):
        gs = rng.choice(ug, len(ug))
        idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        boots.append(coef(F[idx], y[idx]))
    boots = np.array(boots)
    L = len(GRID)
    df = pd.DataFrame(dict(wavenumber=GRID, w_pos=w0[:L], w_neg=w0[L:],
                           z_pos=w0[:L] / boots[:, :L].std(0), z_neg=w0[L:] / boots[:, L:].std(0)))
    df.to_csv(f"{OUT}/discriminant.csv", index=False, float_format="%.4g")
    # strongest consistently discriminating windows (both halves agree in sign, |z| > 2)
    both = (np.sign(df.z_pos) == np.sign(df.z_neg)) & (np.abs(df.z_pos) > 2) & (np.abs(df.z_neg) > 2)
    res["discriminating_wavenumbers"] = df.wavenumber[both].tolist()
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "discriminating_wavenumbers"}, indent=1))
    print("consistent discriminating wavenumbers:", res["discriminating_wavenumbers"])


if __name__ == "__main__":
    main()
