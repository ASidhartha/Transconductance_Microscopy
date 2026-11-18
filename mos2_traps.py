"""Charge-trap model of the MoS2 sweeps (iets/traps.py).

    .venv/bin/python mos2_traps.py        (~1 min)

Fits conductance + J trap branches to the full time series of every location (J = 1, 2, 3;
the Bayesian information criterion picks J). The model is validated by fitting the first half
of the repeats and predicting the second half, and by comparing the zero-current offset of the
up-sweep that it predicts with the measured one. Writes outputs/mos2/traps.csv,
traps_summary.json and traps_series.npz (for the figures).
"""
import json

import numpy as np
import pandas as pd

from iets.sweeps import find_glitches, load_mos2
from iets.traps import E, fit

OUT = "outputs/mos2"


def zero_crossing(V, I, t, t_min=20.0):
    k = np.where((np.diff(np.sign(I)) > 0) & (t[:-1] > t_min))[0]
    if not len(k):
        return np.nan
    k = k[0]
    return float(np.interp(0, I[k:k + 2], V[k:k + 2]))


def main():
    rows, series = [], {}
    for s in load_mos2():
        g = find_glitches(s.I)
        fits = {J: fit(s, J, mask=g) for J in (1, 2, 3)}
        J = min(fits, key=lambda j: fits[j][0].bic)
        f, (t, V, I, rep, ok, model) = fits[J]
        h = s.R // 2
        f2, (_, _, _, _, _, model2) = fit(s, J, mask=g, use=np.arange(h))
        test = ok & (rep >= h)
        ohm = f2.G0 * (1 + f2.drift * t / 3600) * V + f2.I0
        non = (I - ohm)[test]
        r2_oos = 1 - np.sum((I - model2)[test] ** 2) / np.sum((non - non.mean()) ** 2)
        # zero-current offset of the up-sweep (Type 2), mean over repeats: measured and model
        n = len(s.V)
        V0m = V0p = np.nan
        if s.kind == "type2":
            Im = I.reshape(s.R, n).mean(0)
            Mm = model.reshape(s.R, n).mean(0)
            V0m, V0p = zero_crossing(s.V, Im, s.t[0]), zero_crossing(s.V, Mm, s.t[0])
        row = dict(location=s.name, kind=s.kind, J=J, G0_nS=f.G0 * 1e9, drift_pct_h=f.drift * 100,
                   rms_pA=f.rms * 1e12, r2_in=f.r2_relax, r2_out_of_sample=r2_oos,
                   V_zero_measured_mV=V0m * 1e3, V_zero_model_mV=V0p * 1e3)
        for j in range(3):
            row[f"tau{j + 1}_s"] = f.tau[j] if j < J else np.nan
            row[f"C{j + 1}_nF"] = f.C[j] * 1e9 if j < J else np.nan
            row[f"N{j + 1}_per_V"] = f.C[j] / E if j < J else np.nan
        rows.append(row)
        series[s.name] = np.vstack([t, V, I, model, rep, ok])
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/traps.csv", index=False, float_format="%.4g")
    t2 = df[df.kind == "type2"]
    summ = dict(
        type2_tau1_s=[float(t2.tau1_s.min()), float(t2.tau1_s.max())],
        type2_tau2_s=[float(t2.tau2_s.min()), float(t2.tau2_s.max())],
        type2_C1_nF=[float(t2.C1_nF.min()), float(t2.C1_nF.max())],
        type2_C2_nF=[float(t2.C2_nF.min()), float(t2.C2_nF.max())],
        type2_r2_in=[float(t2.r2_in.min()), float(t2.r2_in.max())],
        type2_r2_out_of_sample=[float(t2.r2_out_of_sample.min()), float(t2.r2_out_of_sample.max())],
        type1_r2_out_of_sample=[float(df[df.kind == "type1"].r2_out_of_sample.min()),
                                float(df[df.kind == "type1"].r2_out_of_sample.max())],
        V_zero_corr=float(np.corrcoef(t2.V_zero_measured_mV, t2.V_zero_model_mV)[0, 1]),
        V_zero_mae_mV=float(np.mean(np.abs(t2.V_zero_measured_mV - t2.V_zero_model_mV))),
    )
    json.dump(summ, open(f"{OUT}/traps_summary.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/traps_series.npz", **{k.replace("/", "__"): v for k, v in series.items()})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
