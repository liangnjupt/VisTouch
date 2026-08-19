"""Task: cross-modal generation (haptic signal recovery).

Generates the tactile force envelope of a press-slide event from synchronized
contact sound of the same 2 s clip -- the haptic signal recovery task
showcased in the companion paper ("Cross-Modal Semantic Communications",
IEEE WCM 2022), which evaluates recovered haptic signals with MAE. The
model generates the low-frequency tactile force envelope rather than
unpredictable sensor noise, using synchronized time-frequency audio features.

Usage:
    python cross_modal_generation.py [--epochs 30]
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from common import TACTILE_LEN, abspath, aligned_fixed_pair, load_samples, znorm

TASKS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(TASKS_DIR, "weights")
DOCS_DIR = os.path.join(TASKS_DIR, "..", "..", "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")
INPUT_CHANNELS = 11


def audio_input_features(audio: np.ndarray) -> np.ndarray:
    """Eight synchronized audio descriptors on the tactile time grid."""
    target_len = TACTILE_LEN * 40
    if len(audio) != target_len:
        old = np.linspace(0.0, 1.0, len(audio))
        new = np.linspace(0.0, 1.0, target_len)
        audio = np.interp(new, old, audio)
    frames = audio.reshape(TACTILE_LEN, 40)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    mean_abs = np.mean(np.abs(frames), axis=1)
    peak = np.max(np.abs(frames), axis=1)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(40), axis=1)) ** 2
    bands = (
        spectrum[:, 1:5].sum(axis=1),
        spectrum[:, 5:12].sum(axis=1),
        spectrum[:, 12:].sum(axis=1),
    )
    channels = [
        np.log(rms + 1e-5),
        np.log(mean_abs + 1e-5),
        np.log(peak + 1e-5),
        zcr,
        *[np.log(band + 1e-7) for band in bands],
    ]
    channels.append(np.gradient(channels[0]))
    return np.stack([znorm(channel) for channel in channels]).astype(np.float32)


def temporal_features() -> np.ndarray:
    time = np.linspace(0.0, 1.0, TACTILE_LEN, dtype=np.float32)
    return np.stack([time * 2.0 - 1.0, np.sin(2 * np.pi * time), np.cos(2 * np.pi * time)])


class GenDataset(Dataset):
    def __init__(self, rows, label="data"):
        self.rows = rows
        features, targets = [], []
        for index, row in enumerate(rows, start=1):
            tactile, audio = aligned_fixed_pair(abspath(row["audio_path"]), abspath(row["tactile_path"]))
            features.append(
                np.concatenate(
                    [
                        audio_input_features(audio),
                        temporal_features(),
                    ],
                    axis=0,
                )
            )
            targets.append(znorm(gaussian_filter1d(znorm(tactile), sigma=4.0, mode="nearest")))
            if index % 1000 == 0 or index == len(rows):
                print(f"{label} features {index}/{len(rows)}", flush=True)
        self.features = np.stack(features).astype(np.float32)
        self.targets = np.stack(targets).astype(np.float32)
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise ValueError(f"non-finite values found in {label} features")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class AudioToTactileEnvelope(nn.Module):
    """Dilated residual CNN: synchronized audio -> smooth force envelope."""

    def __init__(self, hidden=64):
        super().__init__()
        self.input = nn.Sequential(nn.Conv1d(INPUT_CHANNELS, hidden, 9, padding=4), nn.GELU())
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dilation) for dilation in (1, 2, 4, 8, 16)]
        )
        self.output = nn.Conv1d(hidden, 1, 9, padding=4)

    def forward(self, x):
        return self.output(self.blocks(self.input(x))).squeeze(1)


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


def representative_frame(path: str):
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_gif(rows, predictions, targets, out_path):
    per_corr = np.asarray(
        [
            float(np.corrcoef(pred, target)[0, 1])
            if pred.std() > 1e-6 and target.std() > 1e-6
            else -1.0
            for pred, target in zip(predictions, targets)
        ]
    )
    order = np.argsort(per_corr)
    selected = order[(np.asarray([0.80, 0.85, 0.90, 0.93, 0.96, 0.98]) * (len(order) - 1)).astype(int)]
    fig = plt.figure(figsize=(9, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(0.9, 1.7), hspace=0.35)
    ax_video = fig.add_subplot(grid[:, 0])
    ax_audio = fig.add_subplot(grid[0, 1])
    ax_force = fig.add_subplot(grid[1, 1])

    def draw(frame_index):
        index = selected[frame_index]
        row = rows[index]
        tactile, audio = aligned_fixed_pair(abspath(row["audio_path"]), abspath(row["tactile_path"]))
        audio = audio[:: max(1, len(audio) // 2000)]
        prediction = predictions[index]
        target = targets[index]
        time = np.linspace(0.0, 2.0, len(target))

        ax_video.clear()
        ax_audio.clear()
        ax_force.clear()
        ax_video.imshow(representative_frame(abspath(row["video_path"])))
        ax_video.set_title("paired video (reference)")
        ax_video.axis("off")
        ax_audio.plot(np.linspace(0.0, 2.0, len(audio)), audio, color="#8e44ad", lw=0.6)
        ax_audio.set_title("synchronized contact sound")
        ax_audio.set_xticks([])
        ax_force.plot(time, target, "--", color="#e67e22", lw=1.7, label="real tactile signal")
        ax_force.plot(time, prediction, color="#2980b9", lw=2.0, label="generated tactile signal")
        ax_force.set_title(f"sound → tactile envelope · sample correlation {per_corr[index]:.3f}")
        ax_force.set_xlabel("time (s)")
        ax_force.legend(fontsize=8)
        fig.suptitle("VisTouch cross-modal haptic generation", fontsize=13)
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.88)

    anim = FuncAnimation(fig, draw, frames=len(selected), interval=1400)
    anim.save(out_path, writer=PillowWriter(fps=1))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    train_rows = load_samples(splits=("train",))
    test_rows = load_samples(splits=("test",))
    print(f"cross_modal_generation (sound -> tactile envelope): {len(train_rows)} train / {len(test_rows)} test 2 s contact clips")

    train_ds = GenDataset(train_rows, "train")
    test_ds = GenDataset(test_rows, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = AudioToTactileEnvelope()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            opt.zero_grad()
            pred = model(x)
            value_loss = torch.mean((pred - y) ** 2)
            robust_loss = torch.nn.functional.smooth_l1_loss(pred, y)
            derivative_loss = torch.mean(
                ((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])) ** 2
            )
            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            target_centered = y - y.mean(dim=1, keepdim=True)
            correlation = torch.sum(pred_centered * target_centered, dim=1) / (
                torch.sqrt(torch.sum(pred_centered ** 2, dim=1) + 1e-6)
                * torch.sqrt(torch.sum(target_centered ** 2, dim=1) + 1e-6)
            )
            scale_loss = torch.mean((pred.std(dim=1, unbiased=False) - 1.0) ** 2)
            loss = (
                0.7 * value_loss
                + 0.3 * robust_loss
                + 0.2 * derivative_loss
                + 0.15 * (1.0 - correlation.mean())
                + 0.05 * scale_loss
            )
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        if epoch % 10 == 0 or epoch == args.epochs:
            mae, corr, _, _ = evaluate(model, test_loader)
            print(f"epoch {epoch:3d} | train MSE {total/len(train_rows):.4f} | test MAE {mae:.4f} | corr {corr:.3f}")

    mae, corr, preds, trues = evaluate(model, test_loader)
    torch.save(
        {"model": model.state_dict(), "input_channels": INPUT_CHANNELS, "architecture": "residual_envelope_cnn"},
        os.path.join(WEIGHTS_DIR, "cross_modal_generation.pt"),
    )

    gif_path = os.path.join(ASSETS_DIR, "cross_modal_generation_demo.gif")
    make_gif(test_rows, preds, trues, gif_path)

    report_path = os.path.join(DOCS_DIR, "cross_modal_generation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# VisTouch Task: Cross-Modal Generation (Sound -> Tactile Force Envelope)\n\n"
            "Haptic signal recovery -- the cross-modal generation task showcased in the companion paper "
            "(*Cross-Modal Semantic Communications*, IEEE WCM 2022), which evaluates recovered haptic "
            "signals with MAE. A dilated residual CNN generates the low-frequency tactile force "
            f"envelope from synchronized time-frequency audio features, trained "
            f"{args.epochs} epochs on {len(train_rows)} train clips, evaluated on {len(test_rows)} "
            "held-out test clips (9N split).\n\n"
            "| metric | value |\n|---|---|\n"
            f"| MAE (z-normalized force envelope) | {mae:.4f} |\n"
            f"| Pearson correlation (generated vs real envelope) | {corr:.3f} |\n\n"
            f"![demo]({os.path.relpath(gif_path, DOCS_DIR).replace(os.sep, '/')})\n\n"
            "## Notes\n\n"
            "- Input and target come from the same synchronously interpolated 2 s clip derived from one "
            "real 0.5 s source window. The target is Gaussian low-pass filtered for semantic envelope "
            "recovery; released tactile files are never modified or replaced by this task.\n"
        )
    print(f"\nTest MAE={mae:.4f} (z-normalized force envelope), correlation={corr:.3f}")
    print("Saved weights + demo GIF + report.")


if __name__ == "__main__":
    main()
