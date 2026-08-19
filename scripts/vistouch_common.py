"""Shared constants and helpers for the VisTouch dataset.

This module backs the user-facing scripts (`dataloader.py`,
`classify_baseline.py`, and any task scripts under `scripts/tasks/`):
material name mapping, class ids, and canonical paths into the released
`VisTouch/` tree.

The original dataset-curation pipeline (raw file discovery, timestamp
alignment, envelope segmentation, export, and validation) was used once to
build this release and has been removed from the repo to keep it lean; the
alignment/segmentation methodology it implemented is documented in
`docs/alignment_report.md` and `docs/data_quality_report.md` for full
transparency.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
VISTOUCH_ROOT = os.path.dirname(SCRIPTS_DIR)  # .../VisTouch

OUT_METADATA_DIR = os.path.join(VISTOUCH_ROOT, "metadata")
OUT_DOCS_DIR = os.path.join(VISTOUCH_ROOT, "docs")
OUT_DATASET_DIR = os.path.join(VISTOUCH_ROOT, "dataset")
OUT_AUDIO_DIR = os.path.join(OUT_DATASET_DIR, "audio")
OUT_TACTILE_DIR = os.path.join(OUT_DATASET_DIR, "tactile")
OUT_VIDEO_DIR = os.path.join(OUT_DATASET_DIR, "video")

# The 312 source segments are archived outside the public dataset directory.
ARCHIVE_DATASET_DIR = os.path.abspath(
    os.path.join(VISTOUCH_ROOT, "..", "dataset20260818")
)

SAMPLES_CSV = os.path.join(OUT_METADATA_DIR, "samples.csv")
SESSIONS_CSV = os.path.join(OUT_METADATA_DIR, "sessions.csv")
CLASSES_JSON = os.path.join(OUT_METADATA_DIR, "classes.json")
CLIPS_CSV = os.path.join(OUT_METADATA_DIR, "clips_index.csv")     # 10,498 materialized 2s micro-clips
FRAMES_CSV = os.path.join(OUT_METADATA_DIR, "frames_index.csv")   # per-video-frame tri-modal index

DATASET_NAME = "VisTouch"

# ---------------------------------------------------------------------------
# Material mapping: pinyin (raw) -> English slug (release) -> Chinese
# ---------------------------------------------------------------------------
# class_id assigned alphabetically by English slug for a stable label space.
MATERIALS = [
    # pinyin,     english,     chinese
    ("anlun",     "spandex",   "氨纶"),
    ("baizhi",    "paper",     "白纸"),
    ("dilun",     "polyester", "涤纶"),
    ("huangtong", "brass",     "黄铜"),
    ("mabu",      "linen",     "麻布"),
    ("muban",     "wood",      "木板"),
    ("shitou",    "stone",     "石头"),
    ("sichou",    "silk",      "丝绸"),
]

PINYIN_TO_ENGLISH = {p: e for p, e, _ in MATERIALS}
ENGLISH_TO_PINYIN = {e: p for p, e, _ in MATERIALS}
ENGLISH_TO_CHINESE = {e: c for _, e, c in MATERIALS}

# stable class ids, sorted by english name
CLASS_NAMES = sorted(PINYIN_TO_ENGLISH.values())
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}

FORCE_LEVELS = [3, 6, 9]  # Newton


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
