"""VisTouch dataset loader.

A thin, dependency-light `torch.utils.data.Dataset` over the exported
VisTouch/ tree (metadata/samples.csv + audio/tactile/video files), letting
users select:
  - which material class(es) to load (`classes=["silk", "stone"]`, or None
    for all 8 classes)
  - which modality/modalities to load (`modalities=("audio", "tactile")`)
  - which split (`split="train" | "test" | "val" | "all"`)
  - which slice mode (`slice_modes=("half",)` for the contact half-wave
    segments used by the benchmarks, or `"idle"` for the non-contact filler
    segments; all segments are strictly non-overlapping and tile the valid
    tri-modal timeline of the 24 raw sessions)

For denser training views, `metadata/clips_index.csv` (0.5 s non-overlapping
clips) and `metadata/frames_index.csv` (per-video-frame tri-modal alignment)
index into the same segment files by time offset.

Every sample in VisTouch is a genuine camera/microphone/tactile-sensor
capture -- there is no synthetic/generated data in this release.

Example (library use):

    from dataloader import VisTouchDataset, get_dataloader

    ds = VisTouchDataset(classes=["silk", "stone"], modalities=("audio", "tactile"))
    loader = get_dataloader(ds, batch_size=8, shuffle=True)

Example (CLI):

    python dataloader.py --classes silk stone --modalities audio tactile --split train
"""
from __future__ import annotations

import argparse
import csv
import os
import wave
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - allows read-only inspection without torch
    torch = None
    Dataset = object
    DataLoader = None

from vistouch_common import CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS, SAMPLES_CSV, VISTOUCH_ROOT

ALL_MODALITIES = ("audio", "tactile", "video")
ALL_SPLITS = ("train", "val", "test")


def _normalize_classes(classes: Optional[Sequence[str]]):
    if classes is None:
        return None
    norm = []
    for c in classes:
        c = str(c).strip().lower()
        if c.isdigit():
            c = ID_TO_CLASS[int(c)]
        if c not in CLASS_TO_ID:
            raise ValueError(f"Unknown VisTouch class '{c}'. Valid classes: {CLASS_NAMES}")
        norm.append(c)
    return set(norm)


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        sampwidth = w.getsampwidth()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    return np.frombuffer(raw, dtype=dtype).astype(np.float32) / np.iinfo(dtype).max


def load_tactile(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                vals.append(float(row[1]))
            except ValueError:
                continue
    return np.asarray(vals, dtype=np.float32)


def load_video(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


class VisTouchDataset(Dataset):
    """PyTorch Dataset over the VisTouch benchmark.

    Args:
        root: path to the VisTouch/ release directory (defaults to the
            directory this script's build pipeline wrote into).
        classes: iterable of material names (English, case-insensitive) or
            class ids to include. None means all 8 classes.
        modalities: subset of ("audio", "tactile", "video") to actually
            load in __getitem__ (unselected modalities are set to None,
            saving I/O).
        split: "train", "val", "test", or "all".
        slice_modes: subset of {"half", "idle"}. "half" (default) are the
            contact half-wave segments (rise/fall phase of one press-slide
            envelope oscillation) used by the benchmarks; "idle" are the
            non-contact filler segments (useful as negatives / for contact
            detection). Segments never overlap.
        video_max_frames: optionally cap the number of video frames loaded
            per sample (keeps memory bounded for quick experiments).
    """

    def __init__(
        self,
        root: str = VISTOUCH_ROOT,
        classes: Optional[Sequence[str]] = None,
        modalities: Sequence[str] = ALL_MODALITIES,
        split: str = "all",
        slice_modes: Sequence[str] = ("half",),
        video_max_frames: Optional[int] = None,
    ):
        self.root = root
        self.modalities = tuple(modalities)
        for m in self.modalities:
            if m not in ALL_MODALITIES:
                raise ValueError(f"Unknown modality '{m}', expected subset of {ALL_MODALITIES}")
        self.video_max_frames = video_max_frames

        wanted_classes = _normalize_classes(classes)
        wanted_slice_modes = set(slice_modes)

        samples_csv = os.path.join(root, "metadata", "samples.csv")
        with open(samples_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        def keep(r):
            if wanted_classes is not None and r["material_english"] not in wanted_classes:
                return False
            if split != "all" and r["split"] != split:
                return False
            if r["slice_mode"] not in wanted_slice_modes:
                return False
            return True

        self.samples = [r for r in rows if keep(r)]
        if not self.samples:
            raise ValueError(
                f"No samples matched classes={classes}, split={split}, slice_modes={slice_modes}. "
                f"Valid classes: {CLASS_NAMES}"
            )

    def __len__(self):
        return len(self.samples)

    def classes_present(self):
        return sorted({r["material_english"] for r in self.samples})

    def __getitem__(self, idx):
        r = self.samples[idx]
        item = {
            "sample_id": r["sample_id"],
            "material": r["material_english"],
            "label": CLASS_TO_ID[r["material_english"]],
            "force_n": int(r["force_n"]),
            "split": r["split"],
        }
        if "audio" in self.modalities:
            item["audio"] = load_wav(os.path.join(self.root, *r["audio_path"].split("/")))
        if "tactile" in self.modalities:
            item["tactile"] = load_tactile(os.path.join(self.root, *r["tactile_path"].split("/")))
        if "video" in self.modalities:
            item["video"] = load_video(os.path.join(self.root, *r["video_path"].split("/")), self.video_max_frames)
        return item


def collate_variable_length(batch):
    """Default collate_fn: stacks scalar fields, keeps variable-length
    audio/tactile/video sequences as a list (they differ by a few samples
    per segment since raw cycle durations aren't perfectly identical)."""
    out = {
        "sample_id": [b["sample_id"] for b in batch],
        "material": [b["material"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long) if torch is not None else [b["label"] for b in batch],
        "force_n": [b["force_n"] for b in batch],
        "split": [b["split"] for b in batch],
    }
    for key in ("audio", "tactile", "video"):
        if key in batch[0]:
            out[key] = [b[key] for b in batch]
    return out


def get_dataloader(dataset: "VisTouchDataset", batch_size: int = 4, shuffle: bool = False, **kwargs):
    if DataLoader is None:
        raise ImportError("PyTorch is required for get_dataloader(); VisTouchDataset itself works without it.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable_length, **kwargs)


def _main():
    parser = argparse.ArgumentParser(description="Inspect the VisTouch dataset via VisTouchDataset.")
    parser.add_argument("--classes", nargs="*", default=None, help=f"subset of {CLASS_NAMES} (default: all)")
    parser.add_argument("--modalities", nargs="*", default=list(ALL_MODALITIES), choices=ALL_MODALITIES)
    parser.add_argument("--split", default="all", choices=("all", "train", "val", "test"))
    parser.add_argument("--slice-modes", nargs="*", default=["half"], choices=("half", "idle"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    ds = VisTouchDataset(
        classes=args.classes,
        modalities=tuple(args.modalities),
        split=args.split,
        slice_modes=tuple(args.slice_modes),
    )
    print(f"Loaded {len(ds)} samples. Classes present: {ds.classes_present()}")

    if torch is not None:
        loader = get_dataloader(ds, batch_size=args.batch_size, shuffle=True)
        batch = next(iter(loader))
        print("First batch:")
        print("  sample_id:", batch["sample_id"])
        print("  material :", batch["material"])
        print("  label    :", batch["label"])
        if "audio" in batch:
            print("  audio shapes  :", [a.shape for a in batch["audio"]])
        if "tactile" in batch:
            print("  tactile shapes:", [t.shape for t in batch["tactile"]])
        if "video" in batch:
            print("  video shapes  :", [v.shape for v in batch["video"]])
    else:
        print("(torch not installed: showing item 0 only)")
        item = ds[0]
        for k, v in item.items():
            print(f"  {k}: {getattr(v, 'shape', v)}")


if __name__ == "__main__":
    _main()
