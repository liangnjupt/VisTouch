# VisTouch Data Quality Report

Validated the full non-overlapping release: 312 phase-labeled physical
segments (240 contact half-waves + 72 non-contact idle segments) that tile
the valid tri-modal timeline of the 24 continuous recording sessions, plus
the two derived index views (3,209 aligned 0.5 s clips and 51,177
frame-level tri-modal samples). All signals are real sensor captures; no
synthetic data.

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

## 2. Tactile sample-rate standardization

Each session's raw integer-second tactile stream (~30–140 Hz, see
`metadata/sessions.csv` `tactile_est_fs_hz`) was interpolated onto a fixed
100 Hz epoch grid before cutting, so every tactile CSV in the release is
uniformly sampled. `Timestamp` is a float Unix epoch (3 decimals).

## 3. File integrity & cross-modality duration check

Spot and batch checks after the rebuild: every segment has non-empty,
openable audio/tactile/video files whose durations match the metadata to
within one video frame (33 ms).

## 4. Filename convention check

Pattern: `VisTouch_{material}_f{force}_r{path}_v{view}_{label}.{ext}` with
`label` in `segNNa` / `segNNb` (contact half-waves) or `idleNN`
(non-contact). All released filenames match.

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
balanced).** Derived views inherit the same split: clips 2,139 / 1,070;
frames 34,202 / 16,975. The split is by force level (train = 3N+6N
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

## 7. Known, accepted limitations (see also docs/alignment_report.md)

- Video frame -> epoch alignment is a linear approximation (no independent
  video timestamp source was recorded); acceptable for segment-level work
  but not guaranteed frame-accurate.
- The captured video resolution is 640x480 @ 30fps.
