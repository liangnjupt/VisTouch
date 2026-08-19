# VisTouch Data Quality Report

Validated 312 non-overlapping, phase-labeled physical source segments (240
contact half-waves + 72 non-contact idle segments) that tile the valid
tri-modal timeline of 24 continuous recording sessions, plus 10,498
materialized 2 s micro-clips and the frame-level index view (51,177
samples). Source segments are real sensor captures; micro-clips are
deterministic interpolations of synchronized real windows.

## 1. Segmentation integrity

- Segments per session are strictly consecutive: **0 overlaps** and
  **0 gaps > 20 ms** across all 24 sessions.
- Each press-slide envelope cycle is split at its smoothed tactile-envelope
  peak into a `rise` (a) and a `fall` (b) half-wave: 120 + 120 halves,
  durations 3.7–6.6 s (mean 5.1 s).
- The remaining timeline (lead-in, tail) is tiled into 72 `idle` segments
  (max 10 s each) so the release covers ~1,706 s of continuous tri-modal
  recording with nothing hidden or discarded.
- All clip/frame index rows resolve to an existing segment file
  (0 orphan references).
- Micro-clips use a 0.5 s source window and 0.15 s stride, creating an
  intentional 0.35 s (70%) overlap. Every source window is fully contained
  within its parent segment (0 out-of-range windows); no clip crosses a
  phase or train/test boundary.

## 2. Tactile sample-rate standardization

Each session's raw integer-second tactile stream (~30–140 Hz, see
`metadata/sessions.csv` `tactile_est_fs_hz`) was interpolated onto a fixed
100 Hz epoch grid before cutting, so every source tactile CSV is uniformly
sampled. Source `Timestamp` is a float Unix epoch (3 decimals); interpolated
micro-clips use local `Time_s` from 0.00 to 1.99 s.

## 3. File integrity & cross-modality duration check

Spot and batch checks after the rebuild: every source segment has non-empty,
openable audio/tactile/video files whose durations match the metadata to
within one video frame (33 ms). The micro-clip tier contains exactly 10,498
files per modality. Sampled clips at the beginning, middle, and end of the
index each contain exactly 60 frames at 30 fps, 32,000 audio samples at
16 kHz, and 200 tactile values at 100 Hz — exactly 2.0 s per modality.

## 4. Filename convention check

Pattern: `VisTouch_{material}_f{force}_r{path}_v{view}_{label}.{ext}` with
`label` in `segNNa` / `segNNb` (contact half-waves) or `idleNN`
(non-contact). Micro-clips use concise, paired modality names:
`{material}_{video|audio|tactile}_{sequence}.{ext}`. All released filenames
match their tier's convention.

## 5. Class / split balance

| material | train | test | total |
|---|---|---|---|
| brass | 26 | 13 | 39 |
| linen | 26 | 13 | 39 |
| paper | 26 | 13 | 39 |
| polyester | 26 | 13 | 39 |
| silk | 26 | 13 | 39 |
| spandex | 26 | 13 | 39 |
| stone | 26 | 13 | 39 |
| wood | 26 | 13 | 39 |

**Total: 208 train / 104 test segments (312 overall, 8 classes — exactly
balanced).** Derived views inherit the same split: micro-clips 7,019 /
3,479; frames 34,202 / 16,975. The split is by force level (train = 3N+6N
sessions, test = held-out 9N sessions), so no raw recording ever spans
both sides.

## 6. Signal-level anomaly screen & repair

Each session waveform/series was screened before cutting:

- **Tactile**: points whose residual against a 5-tap median exceeds half
  the file's dynamic range AND 50 raw units — unmistakable ADC glitches —
  were replaced by the local median. **67 points repaired.**
- **Audio**: isolated electrical pops (>15 robust sigmas vs a 5-tap
  median, >10x file RMS, >25% full scale) were interpolated.
  **2 samples repaired.**

All other content is a faithful cut of the raw sensor recordings. Full
findings: `docs/logs/anomaly_check.log`.

## 7. Micro-clip materialization (synchronized 4x interpolation)

The 10,498 micro-clips in `dataset/{audio,tactile,video}/` are physically materialized
from synchronized 0.5 s source windows sampled every 0.15 s. Each triplet
is stretched by the same 4x temporal mapping to exactly 2.0 s:

- **video**: linear temporal frame interpolation to 60 frames at 30 fps.
- **audio**: linear waveform interpolation to 32,000 samples at 16 kHz.
- **tactile**: linear force-curve interpolation to 200 values at 100 Hz.

Interpolation creates derived values, so micro-clips must not be described
as additional independent physical captures. They are deterministic,
traceable augmentations of real sensor data, not simulated or
model-generated recordings. `metadata/clips_index.csv` preserves each
clip's parent segment, source epoch pair, source duration, stride, and
stretch factor. Full build log: `docs/logs/microclip_build.log`.

## 8. Known, accepted limitations (see also docs/alignment_report.md)

- Video frame -> epoch alignment is a linear approximation (no independent
  video timestamp source was recorded); acceptable for segment-level work
  but not guaranteed frame-accurate.
- The captured video resolution is 640x480 @ 30fps.
