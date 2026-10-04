"""1D U-Net denoiser for IETS spectra."""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin, cout, k=7):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv1d(cout, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet1D(nn.Module):
    """in_ch=2 uses both bias branches (main model); in_ch=1 is the single-branch
    variant used for the held-out-branch validation on real data."""

    def __init__(self, in_ch: int = 2, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.inp = _block(in_ch, chs[0])
        self.down = nn.ModuleList(nn.Sequential(nn.MaxPool1d(2), _block(chs[i], chs[i + 1])) for i in range(depth))
        self.up = nn.ModuleList(nn.ConvTranspose1d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(_block(2 * chs[i], chs[i]) for i in reversed(range(depth)))
        self.out = nn.Conv1d(chs[0], in_ch, 1)

    def forward(self, x):
        skips = [self.inp(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], dim=1))
        return self.out(h)
