"""Task: cross-modal generation (haptic signal recovery).

Generates the tactile force curve of a press-slide event directly from the
contact *sound* of the same real event -- the haptic signal recovery task
showcased in the companion paper ("Cross-Modal Semantic Communications",
IEEE WCM 2022), which evaluates recovered haptic signals with MAE. The
model maps the short-time log-energy contour of the audio to the 100Hz
force curve; contact/release timing and pressure profile are recovered
from sound alone.

Usage:
    python cross_modal_generation.py [--epochs 60] [--env-bins 100]
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

from common import TACTILE_LEN, abspath, aligned_fixed_pair, audio_energy_envelope, load_samples, znorm

TASKS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(TASKS_DIR, "weights")
DOCS_DIR = os.path.join(TASKS_DIR, "..", "..", "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")


def audio_input_signal(audio: np.ndarray, env_bins: int) -> np.ndarray:
    """Log short-time energy contour of the audio, z-normed and linearly
    upsampled onto the tactile time grid (TACTILE_LEN points) so input and
    target share the same time axis."""
    env = audio_energy_envelope(audio, env_bins)
    env = np.log(env + 1e-4)
    env = znorm(env)
    x_old = np.linspace(0.0, 1.0, num=len(env))
    x_new = np.linspace(0.0, 1.0, num=TACTILE_LEN)
    return np.interp(x_new, x_old, env).astype(np.float32)


class GenDataset(Dataset):
    def __init__(self, rows, env_bins):
        self.rows = rows
        self.env_bins = env_bins

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        tactile, audio = aligned_fixed_pair(abspath(r["audio_path"]), abspath(r["tactile_path"]))
        x = audio_input_signal(audio, self.env_bins)
        y = znorm(tactile)
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)


class AudioToTactile(nn.Module):
    """Same-length dilated 1D CNN: audio energy contour -> force curve."""

    def __init__(self, hidden=32):
        super().__init__()
        layers = []
        in_ch = 1
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4 * d, dilation=d), nn.ReLU()]
            in_ch = hidden
        layers += [nn.Conv1d(hidden, 1, kernel_size=9, padding=4)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mae, n = 0.0, 0
    corrs, preds, trues = [], [], []
    for x, y in loader:
        pred = model(x)
        mae += torch.mean(torch.abs(pred - y)).item() * len(y)
        n += len(y)
        for p, t in zip(pred.numpy(), y.numpy()):
            if t.std() > 1e-6 and p.std() > 1e-6:
                corrs.append(float(np.corrcoef(p, t)[0, 1]))
        preds.append(pred.numpy())
        trues.append(y.numpy())
    return mae / n, float(np.mean(corrs)), np.concatenate(preds), np.concatenate(trues)


def make_gif(audio, pred_force, true_force, out_path, n_frames=40):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    audio_plot = audio[:: max(1, len(audio) // 2000)]
    t_aud = np.linspace(0, len(true_force), num=len(audio_plot))
    t_force = np.arange(len(true_force))
    ax1.set_xlim(0, len(true_force))
    ax1.set_title("contact sound (input)")
    amp = float(np.max(np.abs(audio_plot))) + 1e-6
    ax1.set_ylim(-1.1 * amp, 1.1 * amp)
    lo = min(true_force.min(), pred_force.min())
    hi = max(true_force.max(), pred_force.max())
    pad = 0.15 * (hi - lo + 1e-6)
    ax2.set_xlim(0, len(true_force))
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_title("tactile force curve: generated vs real (haptic recovery)")
    l_aud, = ax1.plot([], [], color="#8e44ad", lw=0.6)
    l_pred, = ax2.plot([], [], color="#2980b9", lw=2, label="generated (from sound)")
    l_true, = ax2.plot([], [], color="#e67e22", lw=1.6, ls="--", label="real tactile signal")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    reveal_aud = np.linspace(1, len(audio_plot), n_frames).astype(int)
    reveal_force = np.linspace(1, len(true_force), n_frames).astype(int)

    def update(i):
        k1, k2 = reveal_aud[i], reveal_force[i]
        l_aud.set_data(t_aud[:k1], audio_plot[:k1])
        l_pred.set_data(t_force[:k2], pred_force[:k2])
        l_true.set_data(t_force[:k2], true_force[:k2])
        return l_aud, l_pred, l_true

    anim = FuncAnimation(fig, update, frames=n_frames, interval=70, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=14))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--env-bins", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    train_rows = load_samples(splits=("train",))
    test_rows = load_samples(splits=("test",))
    print(f"cross_modal_generation (sound -> tactile): {len(train_rows)} train / {len(test_rows)} test real samples")

    train_loader = DataLoader(GenDataset(train_rows, args.env_bins), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(GenDataset(test_rows, args.env_bins), batch_size=64, shuffle=False)

    model = AudioToTactile()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            opt.zero_grad()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        if epoch % 10 == 0 or epoch == args.epochs:
            mae, corr, _, _ = evaluate(model, test_loader)
            print(f"epoch {epoch:3d} | train MSE {total/len(train_rows):.4f} | test MAE {mae:.4f} | corr {corr:.3f}")

    mae, corr, preds, trues = evaluate(model, test_loader)
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "cross_modal_generation.pt"))

    # representative example for the GIF: 75th-percentile per-sample correlation
    # (a good-but-not-cherry-picked-best test case)
    per_corr = []
    for p, t in zip(preds, trues):
        c = float(np.corrcoef(p, t)[0, 1]) if (t.std() > 1e-6 and p.std() > 1e-6) else -1.0
        per_corr.append(c)
    ex_idx = int(np.argsort(per_corr)[int(len(per_corr) * 0.75)])
    example_row = test_rows[ex_idx]
    tactile_ex, audio_ex = aligned_fixed_pair(abspath(example_row["audio_path"]), abspath(example_row["tactile_path"]))
    x_ex = audio_input_signal(audio_ex, args.env_bins)
    with torch.no_grad():
        pred_ex = model(torch.from_numpy(x_ex).unsqueeze(0).unsqueeze(0)).squeeze(0).numpy()
    true_ex = znorm(tactile_ex)

    gif_path = os.path.join(ASSETS_DIR, "cross_modal_generation_demo.gif")
    make_gif(audio_ex, pred_ex, true_ex, gif_path)

    report_path = os.path.join(DOCS_DIR, "cross_modal_generation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# VisTouch Task: Cross-Modal Generation (Sound -> Tactile Force Curve)\n\n"
            "Haptic signal recovery -- the cross-modal generation task showcased in the companion paper "
            "(*Cross-Modal Semantic Communications*, IEEE WCM 2022), which evaluates recovered haptic "
            "signals with MAE. A small dilated 1D-CNN generates the 100Hz tactile force curve of a real "
            f"press-slide event from the short-time log-energy contour of its contact sound, trained "
            f"{args.epochs} epochs on {len(train_rows)} real train samples, evaluated on {len(test_rows)} "
            "held-out real test samples (9N split).\n\n"
            "| metric | value |\n|---|---|\n"
            f"| MAE (z-normalized force) | {mae:.4f} |\n"
            f"| Pearson correlation (generated vs real force curve) | {corr:.3f} |\n\n"
            f"![demo]({os.path.relpath(gif_path, DOCS_DIR).replace(os.sep, '/')})\n\n"
            "## Notes\n\n"
            "- Input and target are cropped from the same contact-onset-aware, time-aligned window of one "
            "genuine capture; real tactile data in the released dataset is never modified or replaced by "
            "this task.\n"
        )
    print(f"\nTest MAE={mae:.4f} (z-normalized force), correlation={corr:.3f}")
    print("Saved weights + demo GIF + report.")


if __name__ == "__main__":
    main()
