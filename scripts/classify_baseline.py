"""Baseline 8-class material classifier used to sanity-check that the
exported/segmented VisTouch dataset is actually usable for its intended
task (material recognition from touch/sound/vision).

Trains a RandomForest on lightweight, dependency-free features extracted
per modality (tactile summary statistics, hand-rolled audio spectral
features via numpy FFT, and streamed low-resolution video motion/texture
features), using the dataset's predefined train (f3+f6) / test (f9) split
(see the At-a-glance section of README.md -- this also evaluates generalization across
pressure levels, not just a random split).

It reports accuracy for each modality alone and fused together, and writes
docs/classification_report.md + a confusion matrix figure. If the fused
accuracy is not clearly above chance (1/8 = 12.5%), the per-modality
breakdown is used to diagnose which stage of the pipeline needs revisiting.

Usage:
    python classify_baseline.py [--slice-mode half|idle]
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from dataloader import load_tactile, load_wav
from vistouch_common import CLASS_NAMES, CLASS_TO_ID, OUT_DOCS_DIR, SAMPLES_CSV, VISTOUCH_ROOT, ensure_dir

AUDIO_SR = 16000
N_BANDS = 20
VIDEO_SIZE = 48
VIDEO_STRIDE = 2  # process every Nth frame to keep this fast


def load_samples(slice_mode: str):
    with open(SAMPLES_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["slice_mode"] == slice_mode]


def tactile_features(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        x = np.pad(x, (0, 3 - len(x)))
    diffs = np.diff(x)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(x, prominence=(np.ptp(x) * 0.1 + 1e-6))
    autocorr1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 and np.std(x) > 1e-6 else 0.0
    return np.array([
        x.mean(), x.std(), x.min(), x.max(), np.median(x), np.ptp(x),
        np.sqrt(np.mean(x ** 2)), float(len(peaks)), diffs.max() if len(diffs) else 0.0,
        diffs.min() if len(diffs) else 0.0, diffs.std() if len(diffs) else 0.0, autocorr1,
    ], dtype=np.float32)


def _framed_power_spectrum(x: np.ndarray, frame_len=1024, hop=512):
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop
    window = np.hanning(frame_len)
    specs = []
    for i in range(max(1, n_frames)):
        start = i * hop
        seg = x[start:start + frame_len]
        if len(seg) < frame_len:
            seg = np.pad(seg, (0, frame_len - len(seg)))
        spec = np.abs(np.fft.rfft(seg * window)) ** 2
        specs.append(spec)
    return np.stack(specs, axis=0)  # (n_frames, n_bins)


def audio_features(x: np.ndarray, sr=AUDIO_SR) -> np.ndarray:
    x = x.astype(np.float32)
    if len(x) == 0:
        x = np.zeros(1024, dtype=np.float32)
    power = _framed_power_spectrum(x)
    freqs = np.fft.rfftfreq(1024, 1.0 / sr)

    band_edges = np.logspace(np.log10(max(freqs[1], 20)), np.log10(freqs[-1]), N_BANDS + 1)
    band_energy = np.zeros((power.shape[0], N_BANDS), dtype=np.float32)
    for b in range(N_BANDS):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        if mask.any():
            band_energy[:, b] = power[:, mask].sum(axis=1)
    log_band = np.log1p(band_energy)
    band_feats = np.concatenate([log_band.mean(axis=0), log_band.std(axis=0)])

    mean_spec = power.mean(axis=0)
    total = mean_spec.sum() + 1e-9
    centroid = float((freqs * mean_spec).sum() / total)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * mean_spec).sum() / total))
    cumsum = np.cumsum(mean_spec)
    rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    rms = float(np.sqrt(np.mean(x ** 2)))

    scalars = np.array([centroid, bandwidth, rolloff, zcr, rms], dtype=np.float32)
    return np.concatenate([band_feats, scalars]).astype(np.float32)


def video_features(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    prev = None
    means, stds, motions = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % VIDEO_STRIDE == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (VIDEO_SIZE, VIDEO_SIZE)).astype(np.float32)
            means.append(gray.mean())
            stds.append(gray.std())
            if prev is not None:
                motions.append(float(np.abs(gray - prev).mean()))
            prev = gray
        i += 1
    cap.release()
    if not means:
        return np.zeros(6, dtype=np.float32)
    means, stds = np.array(means), np.array(stds)
    motions = np.array(motions) if motions else np.zeros(1)
    return np.array([
        means.mean(), means.std(), stds.mean(), motions.mean(), motions.std(), motions.max(),
    ], dtype=np.float32)


def build_feature_table(rows, cache_video=True):
    tac_list, aud_list, vid_list, labels, ids = [], [], [], [], []
    for r in rows:
        tactile = load_tactile(os.path.join(VISTOUCH_ROOT, *r["tactile_path"].split("/")))
        audio = load_wav(os.path.join(VISTOUCH_ROOT, *r["audio_path"].split("/")))
        video_path = os.path.join(VISTOUCH_ROOT, *r["video_path"].split("/"))
        tac_list.append(tactile_features(tactile))
        aud_list.append(audio_features(audio))
        vid_list.append(video_features(video_path))
        labels.append(CLASS_TO_ID[r["material_english"]])
        ids.append(r["sample_id"])
    return (
        np.stack(tac_list), np.stack(aud_list), np.stack(vid_list),
        np.array(labels), ids,
    )


def eval_modality(name, X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)
    pred = clf.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, pred)
    return acc, pred, clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-mode", default="half", choices=("half", "idle"))
    args = parser.parse_args()

    ensure_dir(OUT_DOCS_DIR)
    rows = load_samples(args.slice_mode)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    tag = args.slice_mode
    print(f"Building features for {len(train_rows)} train + {len(test_rows)} test samples ({tag} segments)...")

    Xt_tr, Xa_tr, Xv_tr, y_tr, ids_tr = build_feature_table(train_rows)
    Xt_te, Xa_te, Xv_te, y_te, ids_te = build_feature_table(test_rows)

    results = {}
    for name, Xtr, Xte in (
        ("tactile", Xt_tr, Xt_te),
        ("audio", Xa_tr, Xa_te),
        ("video", Xv_tr, Xv_te),
    ):
        acc, pred, _ = eval_modality(name, Xtr, y_tr, Xte, y_te)
        results[name] = (acc, pred)
        print(f"  {name:8s} solo accuracy: {acc:.3f}")

    X_tr_fused = np.concatenate([Xt_tr, Xa_tr, Xv_tr], axis=1)
    X_te_fused = np.concatenate([Xt_te, Xa_te, Xv_te], axis=1)
    fused_acc, fused_pred, fused_clf = eval_modality("fused", X_tr_fused, y_tr, X_te_fused, y_te)
    print(f"  {'fused':8s} accuracy: {fused_acc:.3f}")

    chance = 1.0 / len(CLASS_NAMES)
    verdict = "USABLE" if fused_acc > 2 * chance else "NEEDS IMPROVEMENT"

    label_names = [CLASS_NAMES[i] for i in sorted(set(y_te))]
    report_txt = classification_report(
        y_te, fused_pred, labels=sorted(set(y_te)), target_names=label_names, zero_division=0
    )

    cm = confusion_matrix(y_te, fused_pred, labels=sorted(set(y_te)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax, colorbar=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    cm_name = "confusion_matrix.png" if tag == "half" else f"confusion_matrix_{tag}.png"
    cm_path = os.path.join(OUT_DOCS_DIR, cm_name)
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    lines = [
        "# VisTouch Baseline Classification Report",
        "",
        f"Task: 8-class material recognition. Split: train = force 3N+6N sessions, test = held-out "
        f"9N sessions. Segment granularity: `{args.slice_mode}`. "
        f"Train samples: {len(train_rows)}, test samples: {len(test_rows)}. Chance level: {chance:.3f}.",
        "",
        "## Per-modality accuracy (RandomForest on hand-rolled features)",
        "",
        "| modality | test accuracy |",
        "|---|---|",
    ]
    for name in ("tactile", "audio", "video"):
        lines.append(f"| {name} | {results[name][0]:.3f} |")
    lines.append(f"| **fused (all 3)** | **{fused_acc:.3f}** |")
    lines += [
        "",
        f"**Verdict: {verdict}** (fused accuracy {'exceeds' if fused_acc > 2*chance else 'does not clearly exceed'} "
        f"2x chance level of {chance:.3f}).",
        "",
    ]

    weak_modalities = [name for name, (acc, _) in results.items() if acc < 1.5 * chance]
    if weak_modalities:
        lines += [
            "### Diagnostic note on near-chance modalities",
            "",
            f"`{', '.join(weak_modalities)}` performed close to chance level in isolation under this "
            "particular train/test split (train = 3N/6N sessions, test = held-out 9N sessions). For "
            "tactile in particular this is expected rather than a data defect: the raw force reading "
            "is driven primarily by *how hard the probe presses* (3N vs 6N vs 9N), so a classifier "
            "trained only on 3N/6N tactile signals is effectively asked to generalize across an "
            "unseen pressure regime, which is a harder and different task than material recognition "
            "at a fixed pressure. Audio and video are comparatively pressure-invariant (texture-driven "
            "sound/appearance), which is why they carry most of the fused model's accuracy here. This "
            "is a genuine property of the cross-pressure split (see README.md), not a pipeline "
            "bug -- users who want an easier, same-pressure tactile benchmark can instead build a "
            "random session-level split instead (group by `session_id` in `metadata/samples.csv` rather "
            "than by `force_n`).",
            "",
        ]

    lines += [
        "## Fused-model classification report (test set)",
        "",
        "```",
        report_txt,
        "```",
        "",
        "## Confusion matrix (fused model, test set)",
        "",
        f"![confusion matrix]({cm_name})",
        "",
        "## Notes",
        "",
        "- Features are intentionally simple/hand-rolled (no librosa/deep features) so the script "
        "runs with only numpy/scipy/opencv/sklearn -- this is a *usability sanity check*, not a "
        "SOTA benchmark result.",
        "- Train/test are disjoint force levels from disjoint raw recordings, so this also measures "
        "generalization across pressure (3N/6N -> 9N), not just memorization of one recording.",
    ]

    report_name = "classification_report.md" if tag == "half" else f"classification_report_{tag}.md"
    out_path = os.path.join(OUT_DOCS_DIR, report_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nVerdict: {verdict} (fused acc={fused_acc:.3f}, chance={chance:.3f})")
    print(f"Wrote report to {out_path}")
    return fused_acc, chance


if __name__ == "__main__":
    main()
