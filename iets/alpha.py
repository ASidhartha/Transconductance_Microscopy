"""Local coupling coefficient alpha from the second-order transconductance.

Leading-order ISFET model (manuscript Eq. 3): d2I/dV2 = 2 K0 alpha, so alpha = (d2I/dV2) / (2 K0)
with K0 = mu0 * Cox * W / L. Constants and geometry follow matlab/alpha_calculation_updated.m
(2 mm electrode pitch; L = 2 mm for horizontal/vertical pairs and 2.83 mm for diagonal pairs).

The original MATLAB script converted the per-sample derivative with the median voltage step.
In the ten impure row-1 files the recorded voltage is quantised to 1 mV (steps 1, 1, 1, 2 mV), so
that median is 1.0 mV instead of the true mean step of 1.25 mV and its alpha was 1.5625x too
large there; the copy in matlab/ uses the mean step. Here d2I/dV2 comes from iets.data.read_csv, which uses the mean local step.
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-14   # vacuum permittivity (F/cm), CODATA 2018
MU0 = 5.0                 # low-field mobility of thermally reduced rGO films (cm^2/V s)
EPS_R = 10.0              # static permittivity of the GO functional layer
T_OX = 0.8e-7             # functional-layer thickness (cm) = 0.8 nm
COX = EPS0 * EPS_R / T_OX  # F/cm^2, ~1.11e-5
W = 0.1                   # effective channel width (cm)
PITCH = 0.2               # electrode pitch, rows and columns (cm) = 2 mm


def pair_length(node_a, node_b) -> float:
    """Centre-to-centre distance (cm) between two grid nodes (row, col)."""
    return float(np.hypot((node_a[0] - node_b[0]) * PITCH, (node_a[1] - node_b[1]) * PITCH))


def k0(node_a, node_b) -> float:
    """K0 = mu0 Cox W / L (A/V^2) for the strip geometry of the MATLAB script."""
    return MU0 * COX * W / pair_length(node_a, node_b)


def alpha(d2, node_a, node_b):
    """alpha (dimensionless) from d2I/dV2 (A/V^2)."""
    return np.asarray(d2) / (2 * k0(node_a, node_b))


def is_diagonal(node_a, node_b) -> bool:
    return node_a[0] != node_b[0] and node_a[1] != node_b[1]
