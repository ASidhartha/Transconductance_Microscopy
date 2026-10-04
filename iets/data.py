"""Robust loading of the per-electrode-pair IETS CSV files.

The raw files have inconsistent headers (wvnmbr / Wvnmbr / wvnm / vnmbr, Current /
current / currrent ...) and some contain an extra empty column (Var3 / null) that
shifts every derivative column by one. We therefore read by position, drop all-NaN
columns, and map positions onto canonical names.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

EV_TO_CM1 = 8065.54
DEVICES = {
    "impure": "data/impure",
    "pure": "data/pure",
}
CANON = ["wavenumber", "voltage", "current", "d1", "d1_s", "d2", "d2_s", "d2_ss", "alpha"]


@dataclass
class Spectrum:
    device: str
    pair: str            # e.g. "11_12"
    node_a: tuple        # (row, col)
    node_b: tuple
    wavenumber: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    d2: np.ndarray       # the "DerivativeY2_2" column (last 2nd-derivative column)

    @property
    def midpoint(self):
        return ((self.node_a[0] + self.node_b[0]) / 2, (self.node_a[1] + self.node_b[1]) / 2)


def read_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.loc[:, ~raw.isna().all(axis=0) | (raw.columns.str.lower().str.startswith("alpha"))]
    # drop placeholder columns (Var3/null) that are entirely empty
    raw = raw[[c for c in raw.columns if not (raw[c].isna().all() and not c.lower().startswith("alpha"))]]
    cols = list(raw.columns)
    has_alpha = cols[-1].lower().startswith("alpha")
    body = raw.iloc[:, :-1] if has_alpha else raw
    n = body.shape[1]
    names = CANON[:n] if n <= 8 else CANON[:8]
    df = body.iloc[:, : len(names)].copy()
    df.columns = names
    # files with the shifted layout lack DerivativeY2_2: last 2nd-deriv col is used
    if "d2_ss" not in df.columns:
        df["d2_ss"] = df["d2_s"]
    df = df.dropna(subset=["wavenumber", "voltage", "current"]).reset_index(drop=True).astype(float)
    # The derivative columns are derivatives per SAMPLE (MATLAB-style gradient with unit spacing),
    # not per volt: DerivativeY1 == gradient(Current) exactly. The voltage step differs between
    # files (1.0-1.25 mV), so convert to A/V and A/V^2 with the local step (median-smoothed, since
    # the recorded voltages are averages of repeated sweeps and jitter slightly).
    step = pd.Series(np.gradient(df.voltage.values)).rolling(21, center=True, min_periods=1).median().values
    for c in ("d1", "d1_s"):
        if c in df:
            df[c] = df[c] / step
    for c in ("d2", "d2_s", "d2_ss"):
        if c in df:
            df[c] = df[c] / step**2
    return df


def parse_pair(name: str):
    m = re.match(r"(\d)(\d)_(\d)(\d)", name)
    a = (int(m.group(1)), int(m.group(2)))
    b = (int(m.group(3)), int(m.group(4)))
    return a, b


def load_device(device: str, root: str = ".") -> list[Spectrum]:
    out = []
    for f in sorted(glob.glob(os.path.join(root, DEVICES[device], "*.csv"))):
        pair = os.path.basename(f).split("_alpha")[0]
        df = read_csv(f)
        a, b = parse_pair(pair)
        out.append(Spectrum(device, pair, a, b, df.wavenumber.values, df.voltage.values,
                            df.current.values, df.d2_ss.values))
    return out


def load_all(root: str = ".") -> list[Spectrum]:
    return [s for d in DEVICES for s in load_device(d, root)]


DUP_RTOL = 0.02  # max |I_a - I_b| / max|I_a|; distinct measurements of the same pair differ by >= 22%


def duplicated_pairs(spectra) -> set[str]:
    """Pairs whose current is (nearly) identical in both device folders, a data-provenance issue.

    A relative criterion is used because the currents are ~1e-7 A, far below np.allclose's
    default absolute tolerance. In the corrected data only 53_54 is flagged (0.7% max difference)."""
    by = {}
    for s in spectra:
        by.setdefault(s.pair, {})[s.device] = s
    out = set()
    for pair, d in by.items():
        if len(d) == 2:
            a, b = d.values()
            if a.current.shape == b.current.shape and \
                    np.max(np.abs(a.current - b.current)) <= DUP_RTOL * np.max(np.abs(a.current)):
                out.add(pair)
    return out
