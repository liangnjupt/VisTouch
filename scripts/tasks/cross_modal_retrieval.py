"""Task: cross-modal retrieval.

Learns a shared embedding space (contrastive / InfoNCE) between audio and
tactile encoders so that a query clip in one modality retrieves the
matching clip (same real press-slide event) in the other modality.

Uses the contact `half` segments (one half of a press-slide envelope
oscillation per sample; strictly non-overlapping) and excludes the `idle`
filler segments, which contain no contact signal to match on.

Usage:
    python cross_modal_retrieval.py [--epochs 60] [--embed-dim 64]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from common import abspath, aligned_fixed_pair, load_samples, znorm

TASKS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(TASKS_DIR, "weights")
DOCS_DIR = os.path.join(TASKS_DIR, "..", "..", "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        tactile, audio = aligned_fixed_pair(abspath(r["audio_path"]), abspath(r["tactile_path"]))
        return torch.from_numpy(znorm(audio)).unsqueeze(0), torch.from_numpy(znorm(tactile)).unsqueeze(0)


def conv_block(in_ch, out_ch, k):
    return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=2, padding=k // 2), nn.BatchNorm1d(out_ch), nn.ReLU())


class Encoder(nn.Module):
    def __init__(self, embed_dim, k=15):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1, 16, k), conv_block(16, 32, k), conv_block(32, 64, k), conv_block(64, 64, k),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)


def info_nce(audio_z, tactile_z, temperature):
    logits = audio_z @ tactile_z.T / temperature
    labels = torch.arange(logits.shape[0])
    loss_a = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_t) / 2, logits


def recall_at_k(sim, k):
    n = sim.shape[0]
    topk = torch.topk(sim, k=min(k, n), dim=1).indices
    labels = torch.arange(n).unsqueeze(1)
    hit = (topk == labels).any(dim=1).float().mean().item()
    return hit


@torch.no_grad()
def embed_all(audio_enc, tactile_enc, rows):
    ds = PairDataset(rows)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    audio_zs, tactile_zs = [], []
    for a, t in loader:
        audio_zs.append(audio_enc(a))
        tactile_zs.append(tactile_enc(t))
    return torch.cat(audio_zs), torch.cat(tactile_zs)


def make_gif(rows, audio_raw, tactile_raw, sim, out_path, n_queries=5, top_k=3):
    """Two-column layout: the tactile query fills the left column; the top_k
    ranked audio candidates are stacked in the right column, whose combined
    size matches the query panel."""
    order = np.random.default_rng(0).choice(len(rows), size=min(n_queries, len(rows)), replace=False)
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(top_k, 2, hspace=0.35, wspace=0.12,
                          left=0.04, right=0.96, top=0.86, bottom=0.04)
    ax_query = fig.add_subplot(gs[:, 0])
    ax_cands = [fig.add_subplot(gs[j, 1]) for j in range(top_k)]
    fig.suptitle("VisTouch cross-modal retrieval: tactile query -> ranked audio candidates")

    def draw(frame_idx):
        ax_query.clear()
        for ax in ax_cands:
            ax.clear()
        q = order[frame_idx]
        ax_query.plot(tactile_raw[q], color="#8e44ad", lw=1.6)
        ax_query.set_title(f"query #{q} (tactile)", fontsize=10)
        ax_query.set_xticks([]); ax_query.set_yticks([])
        ranked = torch.topk(sim[q], k=top_k).indices.tolist()
        for j, cand in enumerate(ranked):
            ax = ax_cands[j]
            ax.plot(audio_raw[cand], color="#2980b9", linewidth=0.6)
            correct = cand == q
            ax.set_title(f"rank {j+1}{' ✓' if correct else ''}", fontsize=9,
                         color="#27ae60" if correct else "#333333")
            for spine in ax.spines.values():
                spine.set_edgecolor("#27ae60" if correct else "#cccccc")
                spine.set_linewidth(3 if correct else 1)
            ax.set_xticks([]); ax.set_yticks([])

    anim = FuncAnimation(fig, draw, frames=len(order), interval=1400)
    anim.save(out_path, writer=PillowWriter(fps=1 / 1.4))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    train_rows = load_samples(slice_modes=("half",), splits=("train",))
    test_rows = load_samples(slice_modes=("half",), splits=("test",))
    print(f"cross_modal_retrieval: {len(train_rows)} train / {len(test_rows)} test real half-wave segments")

    train_loader = DataLoader(PairDataset(train_rows), batch_size=args.batch_size, shuffle=True, drop_last=True)

    audio_enc = Encoder(args.embed_dim, k=15)
    tactile_enc = Encoder(args.embed_dim, k=9)
    params = list(audio_enc.parameters()) + list(tactile_enc.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        audio_enc.train(); tactile_enc.train()
        total = 0.0
        for a, t in train_loader:
            opt.zero_grad()
            az, tz = audio_enc(a), tactile_enc(t)
            loss, _ = info_nce(az, tz, args.temperature)
            loss.backward()
            opt.step()
            total += loss.item() * len(a)
        if epoch % 10 == 0 or epoch == args.epochs:
            audio_enc.eval(); tactile_enc.eval()
            az, tz = embed_all(audio_enc, tactile_enc, test_rows)
            sim = az @ tz.T
            r1 = recall_at_k(sim, 1)
            print(f"epoch {epoch:3d} | train loss {total/len(train_rows):.3f} | test R@1(audio->tactile) {r1:.3f}")

    audio_enc.eval(); tactile_enc.eval()
    az, tz = embed_all(audio_enc, tactile_enc, test_rows)
    sim_a2t = az @ tz.T
    sim_t2a = tz @ az.T
    r1_a2t, r5_a2t = recall_at_k(sim_a2t, 1), recall_at_k(sim_a2t, 5)
    r1_t2a, r5_t2a = recall_at_k(sim_t2a, 1), recall_at_k(sim_t2a, 5)
    chance1, chance5 = 1.0 / len(test_rows), min(5, len(test_rows)) / len(test_rows)

    torch.save({"audio_encoder": audio_enc.state_dict(), "tactile_encoder": tactile_enc.state_dict()},
               os.path.join(WEIGHTS_DIR, "cross_modal_retrieval.pt"))

    pairs = [aligned_fixed_pair(abspath(r["audio_path"]), abspath(r["tactile_path"])) for r in test_rows]
    audio_raw = [znorm(a) for _, a in pairs]
    tactile_raw = [znorm(t) for t, _ in pairs]
    gif_path = os.path.join(ASSETS_DIR, "cross_modal_retrieval_demo.gif")
    make_gif(test_rows, audio_raw, tactile_raw, sim_t2a, gif_path)

    report_path = os.path.join(DOCS_DIR, "cross_modal_retrieval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# VisTouch Task: Cross-Modal Retrieval\n\n"
            f"Two small 1D-CNN encoders (audio, tactile) trained with symmetric InfoNCE contrastive loss "
            f"for {args.epochs} epochs on {len(train_rows)} real train `half` segments; evaluated on "
            f"{len(test_rows)} held-out real test `half` segments (9N split).\n\n"
            "| query -> gallery | Recall@1 | Recall@5 | chance@1 | chance@5 |\n"
            "|---|---|---|---|---|\n"
            f"| audio -> tactile | {r1_a2t:.3f} | {r5_a2t:.3f} | {chance1:.3f} | {chance5:.3f} |\n"
            f"| tactile -> audio | {r1_t2a:.3f} | {r5_t2a:.3f} | {chance1:.3f} | {chance5:.3f} |\n\n"
            f"![demo]({os.path.relpath(gif_path, DOCS_DIR).replace(os.sep, '/')})\n"
        )
    print(f"\naudio->tactile R@1={r1_a2t:.3f} R@5={r5_a2t:.3f} | tactile->audio R@1={r1_t2a:.3f} R@5={r5_t2a:.3f} "
          f"(chance@1={chance1:.3f}, chance@5={chance5:.3f})")
    print(f"Saved weights + demo GIF + report.")


if __name__ == "__main__":
    main()
