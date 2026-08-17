<p align="center">
  <img src="docs/assets/vistouch_logo.png" alt="VisTouch" width="680">
</p>

<h3 align="center">A large-scale synchronized vision–touch–audio dataset of robotic sliding contact</h3>

<p align="center">
  <img alt="full corpus observations" src="https://img.shields.io/badge/full_corpus-million--scale_observations-ff7b00">
  <img alt="full corpus materials" src="https://img.shields.io/badge/full_corpus-47_materials-a371f7">
  <img alt="modalities" src="https://img.shields.io/badge/modalities-video_·_audio_·_haptic-red">
  <img alt="public release" src="https://img.shields.io/badge/public_release-51K_aligned_frames_·_8_classes-6e7681">
  <img alt="paper" src="https://img.shields.io/badge/IEEE_WCM_2022-10.1109%2FMWC.008.2200180-blue">
  <img alt="license" src="https://img.shields.io/badge/license-CC--BY--4.0_%2F_MIT-green">
</p>

---

## 🎬 What you can do with VisTouch

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/assets/classification_demo.gif" width="100%"><br/>
      <b>🏷️ Material Recognition</b><br/>
      <sub>fused audio+tactile+video · <b>72.5%</b> test acc (chance 12.5%)</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/assets/tactile_sr_demo.gif" width="100%"><br/>
      <b>🧵 Tactile Super-Resolution</b><br/>
      <sub>clean 100Hz force curve from noisy low-rate input · <b>−93% MSE</b>, 0.999 corr</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/assets/cross_modal_retrieval_demo.gif" width="100%"><br/>
      <b>🔍 Cross-Modal Retrieval</b><br/>
      <sub>touch query → matching sound clip · <b>3× over chance</b> Recall@1/@5</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/assets/cross_modal_generation_demo.gif" width="100%"><br/>
      <b>🔄 Cross-Modal Generation</b><br/>
      <sub>haptic recovery: tactile force curve generated from sound alone · <b>0.93</b> corr</sub>
    </td>
  </tr>
</table>

## 📖 About

**VisTouch** was constructed for [*Cross-Modal Semantic Communications*
(IEEE Wireless Communications, 2022)](https://doi.org/10.1109/MWC.008.2200180)
by controlling a robot arm (UR3 + RH56BF3 dexterous hand) to press and
slide across everyday materials while a camera, a microphone, and a tactile
force sensor record the same contact event simultaneously.

The **full VisTouch research corpus spans 47 material categories and
contains million-scale raw sensor observations** across video frames,
audio waveform samples, and tactile measurements. The companion paper
reports **1000+ synchronized video–audio–haptic signal pairs over all 47
categories**, supporting cross-modal semantic encoding, retrieval, and
haptic signal recovery research.

This repository is the first curated public benchmark release, covering
**8 representative material categories** used in the paper's evaluation —
brass · linen · paper · polyester · silk · spandex · stone · wood. The
release ships **51,177 frame-level tri-modal aligned samples** and
**3,209 aligned 0.5 s clips**, organized as **312 strictly non-overlapping,
phase-labeled segments** (240 contact half-waves + 72 non-contact idle
segments) that tile 24 continuous recording sessions, with a predefined
cross-pressure train/test split.

To the best of our knowledge, this makes VisTouch **the largest public
dataset of same-event synchronized video–audio–force-tactile material
sliding contact** — related resources either pair static surface imagery
with audio/tactile streams, or omit contact audio entirely. All released
views are non-overlapping cuts and time indexes of the 24 raw sessions
(full provenance in `metadata/`). **Every released signal is a genuine
sensor capture; no synthetic data is included.** Future updates will
progressively release more of the 47-category corpus, additional paths,
views, and benchmark tasks.

<p align="center">
  <img src="docs/assets/device_setup.png" alt="capture rig" width="440"><br/>
  <sub>Capture rig: dexterous hand + microphone + tactile sensor, fixed camera view.</sub>
</p>

## 📥 Download

The data files are hosted externally — this repository ships with an empty
`dataset/` folder:

| Source | Link |
|---|---|
| 🌐 Google Drive | [drive.google.com/drive/folders/1U2qW1Oqbkj-...](https://drive.google.com/drive/folders/1U2qW1Oqbkj-47fNWz6BvVEuHI9LAK40j?usp=drive_link) |
| ☁️ Baidu Netdisk | [pan.baidu.com/s/111Nqcwbk30hZRAOpxANKQg](https://pan.baidu.com/s/111Nqcwbk30hZRAOpxANKQg) · extraction code `1234` |

After downloading, place the contents inside the `dataset/` folder at the
repository root:

```
VisTouch/
└── dataset/
    ├── audio/      # per-material subfolders of .wav files
    ├── tactile/    # per-material subfolders of .csv files
    └── video/      # per-material subfolders of .avi files
```

All file paths in `metadata/samples.csv` (and every script) resolve
relative to this layout, so no further configuration is needed.

## 📊 At a glance

| | |
|---|---|
| Modalities | Audio 16kHz WAV · Tactile force 100Hz CSV · Video 640×480@30fps AVI |
| Materials | 8 classes (47 in the full corpus) |
| Contact forces | 3N / 6N / 9N constant normal force |
| Frame samples | **51,177** per-frame tri-modal aligned samples (`metadata/frames_index.csv`) — train 34,202 · test 16,975 |
| Clips | **3,209** non-overlapping 0.5 s aligned clips (`metadata/clips_index.csv`) — train 2,139 · test 1,070 |
| Segments | **312** phase-labeled physical segments: 240 contact half-waves (`segNNa/b`, rise/fall of one press-slide envelope) + 72 idle (`idleNN`) — train 208 · test 104 |
| Sessions | 24 continuous recordings (~70 s each, ~1,706 s total); segments tile each session with **zero overlap** |
| Split | train = 3N+6N · test = held-out 9N — measures cross-pressure generalization |
| License | data CC-BY-4.0 · code MIT |

All views are derived from the same 24 raw sessions: segments are physical
files, clips and frames are non-overlapping time indexes into them.

## 🚀 Quick start

```
VisTouch/
├── dataset/            # audio/ tactile/ video/  (empty in this repo — see Download above)
├── metadata/           # samples.csv · sessions.csv · clips_index.csv · frames_index.csv · classes.json
├── scripts/            # dataloader.py · classify_baseline.py · tasks/
└── docs/               # reports, alignment/quality docs, demo assets, logs/
```

```python
from dataloader import VisTouchDataset, get_dataloader

ds = VisTouchDataset(classes=["silk", "stone"],          # any subset of the 8 materials
                     modalities=("audio", "tactile"),    # load only what you need
                     split="train")
loader = get_dataloader(ds, batch_size=8, shuffle=True)
```

```bash
cd scripts
python dataloader.py --classes silk stone --modalities audio tactile --split train
python classify_baseline.py --slice-mode half           # reproduce the classification baseline
python tasks/tactile_super_resolution.py                # reproduce any task baseline
```

Files follow `VisTouch_{material}_f{force}_r{path}_v{view}_{segment}.{ext}`
(e.g. `VisTouch_silk_f6_r1_v1_seg03a.wav` — `a`/`b` are the rise/fall
half-waves of press-slide cycle 03; `idleNN` are non-contact segments).
Sample-level metadata, labels, and file paths live in
`metadata/samples.csv`; clip/frame views in `metadata/clips_index.csv`
and `metadata/frames_index.csv`.

## 🏆 Benchmarks

All baselines run on the 240 contact half-wave segments with the
cross-pressure split (train 3N+6N, test held-out 9N):

| Task | Model | Metric | Result | Details | Log |
|---|---|---|---|---|---|
| Material recognition | RandomForest, fused 3 modalities | accuracy | **72.5%** (chance 12.5%) | [report](docs/classification_report.md) | [log](docs/logs/classify_half.log) |
| Tactile super-resolution | 1D CNN | MSE vs. naive input | **−93.3%** · corr 0.999 | [report](docs/tactile_sr_report.md) | [log](docs/logs/tactile_super_resolution.log) |
| Cross-modal retrieval | dual 1D-CNN + InfoNCE | Recall@1 / @5 | **3.8% / 16–20%** (chance 1.3% / 6.2%) | [report](docs/cross_modal_retrieval_report.md) | [log](docs/logs/cross_modal_retrieval.log) |
| Cross-modal generation (haptic recovery) | dilated 1D CNN, sound → tactile force curve | MAE / correlation | **0.26 / 0.93** | [report](docs/cross_modal_generation_report.md) | [log](docs/logs/cross_modal_generation.log) |

All baselines are intentionally lightweight (CPU-trainable in minutes) and
serve as usability floors, not state-of-the-art. Trained weights ship in
`scripts/tasks/weights/`; full training/test console records live in
[`docs/logs/`](docs/logs/).

Every released signal passed an automated anomaly screen: rare, physically
impossible ADC glitches in the tactile streams and isolated electrical pops
in the audio were detected and repaired (67 tactile points and 2 audio
samples across all 24 sessions — everything else is a faithful cut of the
raw sensor recordings). Full findings: [`docs/logs/anomaly_check.log`](docs/logs/anomaly_check.log).

## 🗺️ Roadmap

- [ ] Release the remaining material categories (47 total)
- [ ] Additional slide paths (`r2`, `r3`) and camera views (`v2`, `v3`)
- [ ] Deep baselines (spectrogram CNNs, video transformers, full waveform generation)

## 📄 Citation

If you use VisTouch, please cite:

```bibtex
@article{li2022crossmodal,
  title   = {Cross-Modal Semantic Communications},
  author  = {Li, Ang and Wei, Xin and Wu, Dan and Zhou, Liang},
  journal = {IEEE Wireless Communications},
  volume  = {29},
  number  = {6},
  pages   = {144--151},
  year    = {2022},
  doi     = {10.1109/MWC.008.2200180}
}
```

## ⚖️ License

Dataset files are released under **CC-BY-4.0** (`DATA_LICENSE`); all code
under **MIT** (`LICENSE`).
