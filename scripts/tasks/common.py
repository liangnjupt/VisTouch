"""Shared data-loading / preprocessing utilities for the VisTouch downstream
task scripts (cross_modal_generation.py, cross_modal_retrieval.py,
tactile_super_resolution.py).

All three tasks work off fixed-length, time-aligned audio+tactile windows
cropped from the *same* real sample. The crop is contact-onset aware: the
tactile force curve is scanned for the first sustained deviation from its
resting baseline, and the FIXED_SECONDS window starts shortly before that
onset (falling back to the sample start when no onset is detectable), so
windows are dominated by actual contact rather than pre-contact idle time.
The same time offset is applied to both modalities, so a tactile curve and
its paired audio waveform always correspond to the same physical time
window of one genuine press-slide event. No synthetic/fake samples are
introduced anywhere in this module -- only real captures are loaded; any
on-the-fly corruption (see tactile_super_resolution.py) is purely a
training-time input transform for a self-supervised task, never written
back to the released dataset.
"""
from __future__ import annotations

import csv
import os
import sys
import wave

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from vistouch_common import CLASS_NAMES, CLASS_TO_ID, SAMPLES_CSV, VISTOUCH_ROOT  # noqa: E402

FIXED_SECONDS = 5.0
AUDIO_DECIMATE = 4  # 16kHz raw -> 4kHz, enough for envelope/energy-level tasks, much faster to train on
TACTILE_FS = 100.0
TACTILE_LEN = int(FIXED_SECONDS * TACTILE_FS)  # 500


def abspath(rel_path: str) -> str:
    return os.path.join(VISTOUCH_ROOT, *rel_path.split("/"))


def load_samples(slice_modes=("cycle", "sliding"), splits=("train", "test")):
    with open(SAMPLES_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["slice_mode"] in slice_modes and r["split"] in splits]


def load_wav_raw(path: str):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        rate = w.getframerate()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data, rate


def fixed_audio(path: str, offset_s: float = 0.0) -> np.ndarray:
    """Real captured audio, decimated to ~4kHz and cropped/zero-padded to a
    fixed FIXED_SECONDS-long window starting at offset_s."""
    data, rate = load_wav_raw(path)
    data = data[::AUDIO_DECIMATE]
    eff_rate = rate / AUDIO_DECIMATE
    start = int(round(offset_s * eff_rate))
    n = int(round(FIXED_SECONDS * eff_rate))
    data = data[start:start + n]
    if len(data) < n:
        data = np.pad(data, (0, n - len(data)))
    return data.astype(np.float32)


def load_tactile_series(path: str) -> np.ndarray:
    """Full real tactile force series (already resampled to 100Hz)."""
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


def onset_offset_s(vals: np.ndarray, lead_s: float = 1.0) -> float:
    """Window start (seconds) so the FIXED_SECONDS crop is dominated by
    actual contact: find the first sustained deviation of the tactile curve
    from its resting baseline and start lead_s before it. Returns 0.0 when
    no onset is detectable (e.g. contact from the very first sample)."""
    if len(vals) < 60:
        return 0.0
    n_base = min(50, len(vals) // 4)
    baseline = float(np.median(vals[:n_base]))
    resid = np.abs(vals - baseline)
    sigma = 1.4826 * float(np.median(resid[:n_base])) + 1e-6
    thresh = max(5.0 * sigma, 0.1)
    active = resid > thresh
    # sustained: >=20 active samples within a 30-sample lookahead
    kernel = np.ones(30)
    counts = np.convolve(active.astype(np.float32), kernel, mode="full")[:len(active)]
    hits = np.flatnonzero(counts >= 20)
    if len(hits) == 0:
        return 0.0
    onset_s = max(0.0, hits[0] / TACTILE_FS - lead_s)
    max_offset = max(0.0, len(vals) / TACTILE_FS - FIXED_SECONDS)
    return min(onset_s, max_offset)


def fixed_tactile(path: str, offset_s: float = 0.0) -> np.ndarray:
    """Real captured tactile force curve, cropped/zero-padded to TACTILE_LEN
    samples starting at offset_s (same window convention as fixed_audio)."""
    vals = load_tactile_series(path)
    start = int(round(offset_s * TACTILE_FS))
    vals = vals[start:start + TACTILE_LEN]
    if len(vals) < TACTILE_LEN:
        vals = np.pad(vals, (0, TACTILE_LEN - len(vals)))
    return vals


def fixed_tactile_onset(path: str) -> np.ndarray:
    """Contact-onset-aware fixed tactile crop (tactile-only tasks)."""
    vals = load_tactile_series(path)
    return fixed_tactile(path, offset_s=onset_offset_s(vals))


def aligned_fixed_pair(audio_path: str, tactile_path: str):
    """Contact-onset-aware, time-aligned (tactile, audio) crops of the same
    real event: the shared offset is derived from the tactile onset and
    applied identically to both modalities."""
    vals = load_tactile_series(tactile_path)
    offset = onset_offset_s(vals)
    return fixed_tactile(tactile_path, offset), fixed_audio(audio_path, offset)


def znorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def audio_energy_envelope(audio: np.ndarray, n_bins: int = 100) -> np.ndarray:
    """Short-time RMS energy envelope of a fixed-length audio window,
    downsampled to n_bins values -- used by cross_modal_generation.py as
    the audio-side input representation for haptic signal recovery."""
    n = len(audio)
    bounds = np.linspace(0, n, n_bins + 1).astype(int)
    env = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        seg = audio[bounds[i]:bounds[i + 1]]
        env[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    return env
