"""Train the IETS denoisers on synthetic spectra whose noise is estimated from the real data.

    .venv/bin/python train.py            # trains both models -> models/
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from iets.model import UNet1D
from iets.pipeline import prepare_all, residual_noise
from iets.synth import NoiseBank, make_batch


def real_noise():
    """Residual (unshared) noise from the spectra whose two branches could be aligned."""
    _, X, _, _, A = prepare_all()
    return residual_noise(X, A)


def build_dataset(bank, n, seed):
    rng = np.random.default_rng(seed)
    X, Y = zip(*(make_batch(rng, bank, 1000) for _ in range(n // 1000)))
    return torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(Y))


def train(in_ch, Xtr, Ytr, Xva, Yva, epochs, device, out):
    torch.manual_seed(in_ch)
    model = UNet1D(in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * (len(Xtr) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    sel = slice(0, 1) if in_ch == 1 else slice(0, 2)
    Xva, Yva = Xva[:, sel].to(device), Yva[:, sel].to(device)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        t0, tot = time.time(), 0.0
        for b in range(len(Xtr) // 128):
            i = perm[b * 128:(b + 1) * 128]
            x, y = Xtr[i].to(device), Ytr[i].to(device)
            if in_ch == 1:  # either branch can serve as a single-branch example
                c = torch.randint(0, 2, (1,)).item()
                x, y = x[:, c:c + 1], y[:, c:c + 1]
            else:  # branch order carries no meaning
                if torch.rand(1) < 0.5:
                    x, y = x.flip(1), y.flip(1)
            sgn = torch.where(torch.rand(len(x), 1, 1, device=device) < 0.5, -1.0, 1.0)
            x, y = x * sgn, y * sgn
            loss = F.huber_loss(model(x), y, delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            va = F.mse_loss(model(Xva), Yva).item()
            base = F.mse_loss(Xva, Yva).item()
        print(f"[in_ch={in_ch}] epoch {ep + 1}/{epochs} train {tot / (b + 1):.4f} "
              f"val MSE {va:.4f} (noisy input {base:.4f}) {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), out)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs("models", exist_ok=True)
    bank = NoiseBank(real_noise())
    t0 = time.time()
    Xtr, Ytr = build_dataset(bank, args.n_train, seed=1)
    Xva, Yva = build_dataset(bank, 4000, seed=2)
    np.savez_compressed("models/synth_test.npz", **dict(zip("XY", build_dataset(bank, 4000, seed=3))))
    print(f"synthetic data built in {time.time() - t0:.0f}s", flush=True)
    train(2, Xtr, Ytr, Xva, Yva, args.epochs, device, "models/unet_2branch.pt")
    train(1, Xtr, Ytr, Xva, Yva, args.epochs, device, "models/unet_1branch.pt")
