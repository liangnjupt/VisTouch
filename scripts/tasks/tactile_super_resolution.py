"""Task: tactile super-resolution / denoising.

Reconstructs a clean, full-rate (100Hz) tactile force curve from a
low-rate + noisy version of the same materialized 2 s curve. The corrupted input is
generated on-the-fly at training/eval time (downsample by SR_FACTOR, linear
upsample back, add Gaussian noise) purely as a self-supervised training
signal -- the ground truth is the released curve deterministically
interpolated from a real 0.5 s sensor window; this task never modifies the
released dataset itself.

Usage:
    python tactile_super_resolution.py [--epochs 40] [--sr-factor 5] [--noise-std 0.2]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from common import TACTILE_LEN, abspath, fixed_tactile_onset, load_samples, znorm

TASKS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(TASKS_DIR, "weights")
DOCS_DIR = os.path.join(TASKS_DIR, "..", "..", "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")


def corrupt(clean: np.ndarray, sr_factor: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    low = clean[::sr_factor]
    x_old = np.linspace(0, 1, num=len(low))
    x_new = np.linspace(0, 1, num=len(clean))
    up = np.interp(x_new, x_old, low)
    up = up + rng.normal(0.0, noise_std, size=up.shape).astype(np.float32)
    return up.astype(np.float32)


class TactileSRDataset(Dataset):
    def __init__(self, rows, sr_factor, noise_std, seed):
        self.rows = rows
        self.sr_factor = sr_factor
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _clean(self, idx):
        if idx not in self._cache:
            r = self.rows[idx]
            clean = znorm(fixed_tactile_onset(abspath(r["tactile_path"])))
            self._cache[idx] = clean
        return self._cache[idx]

    def __getitem__(self, idx):
        clean = self._clean(idx)
        noisy = corrupt(clean, self.sr_factor, self.noise_std, self.rng)
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0)


class SRNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden * 2, hidden, kernel_size=9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=9, padding=4),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader):
    model.eval()
    mse_model, mse_baseline, n = 0.0, 0.0, 0
    all_pred, all_true, all_noisy = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            pred = model(noisy)
            mse_model += torch.mean((pred - clean) ** 2).item() * len(clean)
            mse_baseline += torch.mean((noisy - clean) ** 2).item() * len(clean)
            n += len(clean)
            all_pred.append(pred.squeeze(1).numpy())
            all_true.append(clean.squeeze(1).numpy())
            all_noisy.append(noisy.squeeze(1).numpy())
    mse_model /= n
    mse_baseline /= n
    pred_cat = np.concatenate(all_pred)
    true_cat = np.concatenate(all_true)
    corr = float(np.corrcoef(pred_cat.ravel(), true_cat.ravel())[0, 1])
    return mse_model, mse_baseline, corr, (all_noisy, all_pred, all_true)


def make_gif(noisy, pred, true, out_path, n_frames=48):
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(true))
    ax.set_xlim(0, len(true))
    pad = 0.15 * (true.max() - true.min() + 1e-6)
    ax.set_ylim(min(true.min(), noisy.min(), pred.min()) - pad, max(true.max(), noisy.max(), pred.max()) + pad)
    ax.set_title("VisTouch tactile super-resolution / denoising")
    ax.set_xlabel("sample (100Hz)")
    ax.set_ylabel("force (z-normalized)")
    l_noisy, = ax.plot([], [], color="#c0392b", lw=1.0, alpha=0.6, label="low-rate + noisy input")
    l_pred, = ax.plot([], [], color="#2980b9", lw=2.0, label="model reconstruction")
    l_true, = ax.plot([], [], color="#27ae60", lw=1.4, ls="--", label="ground truth (real sensor)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    reveal_idx = np.linspace(1, len(true), n_frames).astype(int)

    def update(i):
        k = reveal_idx[i]
        l_noisy.set_data(t[:k], noisy[:k])
        l_pred.set_data(t[:k], pred[:k])
        l_true.set_data(t[:k], true[:k])
        return l_noisy, l_pred, l_true

    anim = FuncAnimation(fig, update, frames=len(reveal_idx), interval=60, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sr-factor", type=int, default=5, help="downsampling factor simulating a low-rate sensor")
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    train_rows = load_samples(splits=("train",))
    test_rows = load_samples(splits=("test",))
    print(f"tactile_super_resolution: {len(train_rows)} train / {len(test_rows)} test 2 s contact clips")

    train_ds = TactileSRDataset(train_rows, args.sr_factor, args.noise_std, seed=args.seed)
    test_ds = TactileSRDataset(test_rows, args.sr_factor, args.noise_std, seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SRNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for noisy, clean in train_loader:
            opt.zero_grad()
            pred = model(noisy)
            loss = torch.mean((pred - clean) ** 2)
            loss.backward()
            opt.step()
            total += loss.item() * len(clean)
        if epoch % 10 == 0 or epoch == args.epochs:
            mse_m, mse_b, corr, _ = evaluate(model, test_loader)
            print(f"epoch {epoch:3d} | train MSE {total/len(train_ds):.4f} | test MSE {mse_m:.4f} "
                  f"(baseline {mse_b:.4f}) | corr {corr:.3f}")

    mse_m, mse_b, corr, (noisy_all, pred_all, true_all) = evaluate(model, test_loader)
    improvement = 100.0 * (mse_b - mse_m) / mse_b

    weights_path = os.path.join(WEIGHTS_DIR, "tactile_sr.pt")
    torch.save(model.state_dict(), weights_path)

    # pick a representative test example (median baseline error) for the GIF
    errs = [float(np.mean((p - t) ** 2)) for p, t in zip(pred_all[0], true_all[0])]
    if len(errs) == 0:
        idx_batch, idx_in_batch = 0, 0
    else:
        order = np.argsort(errs)
        idx_in_batch = order[len(order) // 2]
    gif_path = os.path.join(ASSETS_DIR, "tactile_sr_demo.gif")
    make_gif(noisy_all[0][idx_in_batch], pred_all[0][idx_in_batch], true_all[0][idx_in_batch], gif_path)

    report_path = os.path.join(DOCS_DIR, "tactile_sr_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# VisTouch Task: Tactile Super-Resolution / Denoising\n\n"
            f"Model: small 1D CNN (4 conv layers) trained {args.epochs} epochs on {len(train_ds)} "
            f"materialized 2 s train tactile curves; evaluated on {len(test_ds)} held-out test curves "
            f"(9N split).\n\n"
            f"Corruption (train+eval input only, never written back to the dataset): downsample by "
            f"{args.sr_factor}x then linearly upsample, + Gaussian noise (std={args.noise_std}).\n\n"
            "| metric | low-rate+noisy input (baseline) | model reconstruction |\n"
            "|---|---|---|\n"
            f"| MSE (z-normalized) | {mse_b:.4f} | {mse_m:.4f} |\n\n"
            f"**MSE improvement over the naive upsampled input: {improvement:.1f}%.** "
            f"Pearson correlation between reconstruction and released ground truth: {corr:.3f}.\n\n"
            f"![demo]({os.path.relpath(gif_path, DOCS_DIR).replace(os.sep, '/')})\n"
        )
    print(f"\nMSE improvement: {improvement:.1f}% | corr={corr:.3f}")
    print(f"Saved weights to {weights_path}")
    print(f"Saved demo GIF to {gif_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
