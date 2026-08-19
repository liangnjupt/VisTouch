"""Task: multimodal tactile missing-block completion.

A contiguous block of each tactile curve is hidden. A temporal CNN uses the
remaining tactile context together with synchronized audio and video motion
features to reconstruct the missing force values. Evaluation compares the
model against context-only linear interpolation on the held-out condition.
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import abspath, load_samples, load_tactile_series, load_wav_raw, znorm

TASKS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(TASKS_DIR, "weights")
DOCS_DIR = os.path.join(TASKS_DIR, "..", "..", "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")
STEPS = 200


def normalize_channel(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


def resample(x: np.ndarray, n: int = STEPS) -> np.ndarray:
    if len(x) == n:
        return x.astype(np.float32)
    old = np.linspace(0.0, 1.0, len(x))
    new = np.linspace(0.0, 1.0, n)
    return np.interp(new, old, x).astype(np.float32)


def audio_features(path: str) -> np.ndarray:
    audio, _ = load_wav_raw(path)
    target_len = STEPS * 160
    audio = resample(audio, target_len)
    frames = audio.reshape(STEPS, 160)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    mean_abs = np.mean(np.abs(frames), axis=1)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    return np.stack(
        [
            normalize_channel(np.log(rms + 1e-5)),
            normalize_channel(np.log(mean_abs + 1e-5)),
            normalize_channel(zcr),
        ]
    ).astype(np.float32)


def video_features(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    means, stds, motions = [], [], []
    previous = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 36)).astype(np.float32) / 255.0
        means.append(float(gray.mean()))
        stds.append(float(gray.std()))
        motions.append(0.0 if previous is None else float(np.abs(gray - previous).mean()))
        previous = gray
    cap.release()
    if not means:
        return np.zeros((3, STEPS), dtype=np.float32)
    channels = [resample(np.asarray(values), STEPS) for values in (means, stds, motions)]
    return np.stack([normalize_channel(channel) for channel in channels]).astype(np.float32)


def prepare(rows, label: str):
    tactile = np.empty((len(rows), STEPS), dtype=np.float32)
    audio = np.empty((len(rows), 3, STEPS), dtype=np.float32)
    video = np.empty((len(rows), 3, STEPS), dtype=np.float32)
    for index, row in enumerate(rows):
        tactile[index] = znorm(resample(load_tactile_series(abspath(row["tactile_path"])), STEPS))
        audio[index] = audio_features(abspath(row["audio_path"]))
        video[index] = video_features(abspath(row["video_path"]))
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            print(f"{label} features {index + 1}/{len(rows)}", flush=True)
    return tactile, audio, video


def mask_start(curve: np.ndarray, block: int, rng=None) -> int:
    max_start = len(curve) - block
    gradient = np.abs(np.gradient(curve))
    peak_start = int(np.clip(np.argmax(gradient) - block // 2, 0, max_start))
    if rng is None:
        return peak_start
    if rng.random() < 0.7:
        return int(np.clip(peak_start + rng.integers(-block // 3, block // 3 + 1), 0, max_start))
    return int(rng.integers(0, max_start + 1))


class CompletionDataset(Dataset):
    def __init__(self, tactile, audio, video, block, training, seed=0):
        self.tactile = tactile
        self.audio = audio
        self.video = video
        self.block = block
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tactile)

    def __getitem__(self, index):
        target = self.tactile[index]
        start = mask_start(target, self.block, self.rng if self.training else None)
        missing = np.zeros(STEPS, dtype=np.float32)
        missing[start:start + self.block] = 1.0
        observed = 1.0 - missing
        masked = target * observed
        features = np.concatenate(
            [
                masked[None, :],
                observed[None, :],
                self.audio[index],
                self.video[index],
            ],
            axis=0,
        )
        return (
            torch.from_numpy(features),
            torch.from_numpy(target),
            torch.from_numpy(missing),
            index,
        )


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


class TactileCompletionNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.input = nn.Sequential(nn.Conv1d(8, hidden, 9, padding=4), nn.GELU())
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dilation) for dilation in (1, 2, 4, 8, 16)]
        )
        self.output = nn.Conv1d(hidden, 1, 9, padding=4)

    def forward(self, features):
        return self.output(self.blocks(self.input(features))).squeeze(1)


def linear_completion(target: np.ndarray, missing: np.ndarray) -> np.ndarray:
    indexes = np.arange(len(target))
    observed = missing < 0.5
    output = target.copy()
    output[~observed] = np.interp(indexes[~observed], indexes[observed], target[observed])
    return output


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    model_values, baseline_values, target_values = [], [], []
    completed_all, baseline_all, target_all, missing_all, indices = [], [], [], [], []
    for features, target, missing, index in loader:
        prediction = model(features)
        completed = target * (1.0 - missing) + prediction * missing
        for t, m, c, idx in zip(target.numpy(), missing.numpy(), completed.numpy(), index.numpy()):
            baseline = linear_completion(t, m)
            mask = m > 0.5
            model_values.append(c[mask])
            baseline_values.append(baseline[mask])
            target_values.append(t[mask])
            completed_all.append(c)
            baseline_all.append(baseline)
            target_all.append(t)
            missing_all.append(m)
            indices.append(int(idx))

    model_values = np.concatenate(model_values)
    baseline_values = np.concatenate(baseline_values)
    target_values = np.concatenate(target_values)
    mse_model = float(np.mean((model_values - target_values) ** 2))
    mse_baseline = float(np.mean((baseline_values - target_values) ** 2))
    mae_model = float(np.mean(np.abs(model_values - target_values)))
    corr_model = float(np.corrcoef(model_values, target_values)[0, 1])
    return {
        "mse_model": mse_model,
        "mse_baseline": mse_baseline,
        "mae_model": mae_model,
        "corr_model": corr_model,
        "completed": np.asarray(completed_all),
        "baseline": np.asarray(baseline_all),
        "target": np.asarray(target_all),
        "missing": np.asarray(missing_all),
        "indices": np.asarray(indices),
    }


def representative_frame(path: str):
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_gif(rows, result, out_path):
    errors = np.mean((result["completed"] - result["target"]) ** 2 * result["missing"], axis=1)
    order = np.argsort(errors)
    chosen = order[np.linspace(len(order) // 4, 3 * len(order) // 4, 6).astype(int)]
    fig = plt.figure(figsize=(9, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(0.9, 1.7), hspace=0.35)
    ax_video = fig.add_subplot(grid[:, 0])
    ax_audio = fig.add_subplot(grid[0, 1])
    ax_force = fig.add_subplot(grid[1, 1])

    def draw(frame_index):
        local = chosen[frame_index]
        row = rows[result["indices"][local]]
        target = result["target"][local]
        completed = result["completed"][local]
        baseline = result["baseline"][local]
        missing = result["missing"][local] > 0.5
        audio, _ = load_wav_raw(abspath(row["audio_path"]))
        audio = audio[:: max(1, len(audio) // 2000)]
        time = np.linspace(0.0, 2.0, STEPS)

        ax_video.clear()
        ax_audio.clear()
        ax_force.clear()
        ax_video.imshow(representative_frame(abspath(row["video_path"])))
        ax_video.set_title("synchronized video")
        ax_video.axis("off")
        ax_audio.plot(np.linspace(0.0, 2.0, len(audio)), audio, color="#8e44ad", lw=0.55)
        ax_audio.set_title("synchronized audio")
        ax_audio.set_xticks([])
        ax_force.plot(time, target, "--", color="#27ae60", lw=1.5, label="ground truth")
        ax_force.plot(time, baseline, color="#c0392b", lw=1.3, alpha=0.8, label="context-only")
        ax_force.plot(time, completed, color="#2980b9", lw=2.0, label="multimodal completion")
        start, end = np.flatnonzero(missing)[[0, -1]]
        ax_force.axvspan(time[start], time[end], color="#f1c40f", alpha=0.18, label="missing block")
        ax_force.set_title("tactile missing-block completion")
        ax_force.set_xlabel("time")
        ax_force.legend(fontsize=7, loc="best")
        fig.suptitle("VisTouch multimodal tactile completion", fontsize=13)
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.88)

    animation = FuncAnimation(fig, draw, frames=len(chosen), interval=1400)
    animation.save(out_path, writer=PillowWriter(fps=1))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    train_rows = load_samples(slice_modes=("half",), splits=("train",))
    test_rows = load_samples(slice_modes=("half",), splits=("test",))
    print(f"multimodal_tactile_completion: {len(train_rows)} train / {len(test_rows)} test clips")
    train_t, train_a, train_v = prepare(train_rows, "train")
    test_t, test_a, test_v = prepare(test_rows, "test")
    train_loader = DataLoader(
        CompletionDataset(train_t, train_a, train_v, args.block_size, True, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        CompletionDataset(test_t, test_a, test_v, args.block_size, False, args.seed),
        batch_size=128,
        shuffle=False,
    )

    model = TactileCompletionNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for features, target, missing, _ in train_loader:
            optimizer.zero_grad()
            prediction = model(features)
            completed = target * (1.0 - missing) + prediction * missing
            value_loss = torch.sum((prediction - target) ** 2 * missing) / torch.sum(missing)
            diff_mask = torch.maximum(missing[:, 1:], missing[:, :-1])
            diff_loss = torch.sum(
                ((completed[:, 1:] - completed[:, :-1]) - (target[:, 1:] - target[:, :-1])) ** 2
                * diff_mask
            ) / torch.sum(diff_mask)
            loss = value_loss + 0.2 * diff_loss
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
        if epoch % 5 == 0 or epoch == args.epochs:
            result = evaluate(model, test_loader)
            print(
                f"epoch {epoch:3d} | train loss {total/len(train_rows):.4f} | "
                f"masked MSE {result['mse_model']:.4f} "
                f"(linear {result['mse_baseline']:.4f}) | corr {result['corr_model']:.3f}",
                flush=True,
            )

    result = evaluate(model, test_loader)
    improvement = 100.0 * (result["mse_baseline"] - result["mse_model"]) / result["mse_baseline"]
    weights_path = os.path.join(WEIGHTS_DIR, "multimodal_tactile_completion.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "block_size": args.block_size,
            "input_channels": 8,
        },
        weights_path,
    )
    gif_path = os.path.join(ASSETS_DIR, "multimodal_tactile_completion_demo.gif")
    make_gif(test_rows, result, gif_path)
    report_path = os.path.join(DOCS_DIR, "multimodal_tactile_completion_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# VisTouch Task: Multimodal Tactile Completion\n\n"
            "A contiguous tactile block is hidden and reconstructed from the "
            "remaining tactile context plus synchronized audio and video features. "
            "Metrics are calculated only inside the missing block.\n\n"
            "| metric | context-only linear interpolation | multimodal model |\n"
            "|---|---:|---:|\n"
            f"| MSE | {result['mse_baseline']:.4f} | {result['mse_model']:.4f} |\n"
            f"| MAE | — | {result['mae_model']:.4f} |\n"
            f"| Correlation | — | {result['corr_model']:.3f} |\n\n"
            f"**MSE improvement over context-only interpolation: {improvement:.1f}%.**\n\n"
            f"![demo]({os.path.relpath(gif_path, DOCS_DIR).replace(os.sep, '/')})\n"
        )
    print(
        f"masked MSE={result['mse_model']:.4f} vs linear={result['mse_baseline']:.4f} "
        f"({improvement:.1f}% improvement), MAE={result['mae_model']:.4f}, "
        f"corr={result['corr_model']:.3f}"
    )
    print("saved weights + report + demo GIF")


if __name__ == "__main__":
    main()
