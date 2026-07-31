# VisTouch Data Quality Report

Validated 2000 exported samples (all real sensor captures: 120 single-cycle
segments + 1880 multi-scale sliding-window segments cut from the raw
recordings at distinct, partly phase-shifted/jittered start timestamps; no
synthetic data). Every sample triplet was checked for file existence and
non-zero size after the final expansion pass; tactile standardization below
applies to all 2000 tactile CSVs.

## 1. Tactile sample-rate standardization

All exported tactile CSVs were resampled (linear interpolation) onto a fixed 100 Hz grid, replacing the raw ~30-140 Hz inconsistency documented in `metadata/sessions.csv` (`tactile_est_fs_hz`). `Timestamp` is now a float Unix epoch (3 decimal places) rather than the raw integer-second value.

Example row-count changes (sample_id, original_rows, resampled_rows):

- VisTouch_brass_f3_r1_v1_seg01: 316 -> 1101
- VisTouch_brass_f3_r1_v1_seg02: 319 -> 1001
- VisTouch_brass_f3_r1_v1_seg03: 322 -> 1001
- VisTouch_brass_f3_r1_v1_seg04: 319 -> 1101
- VisTouch_brass_f3_r1_v1_seg05: 318 -> 1001
- VisTouch_brass_f3_r1_v1_wins01: 225 -> 701
- VisTouch_brass_f3_r1_v1_wins02: 219 -> 701
- VisTouch_brass_f3_r1_v1_wins03: 214 -> 701
- VisTouch_brass_f3_r1_v1_wins04: 215 -> 701
- VisTouch_brass_f3_r1_v1_wins05: 214 -> 701
- ... (all tactile CSVs resampled in total)

## 2. File integrity & cross-modality duration check

Samples with at least one issue: 0 / 2000

No issues found: every sample has non-empty, openable audio/tactile/video files with consistent durations across modalities.

## 3. Filename convention check

Pattern: `VisTouch_{material}_f{force}_r{path}_v{view}_{label}.{ext}`

All exported filenames match the VisTouch naming convention.

## 4. Class / split balance

| material | train | test | val | total |
|---|---|---|---|---|
| brass | 168 | 82 | 0 | 250 |
| linen | 177 | 89 | 0 | 266 |
| paper | 164 | 83 | 0 | 247 |
| polyester | 168 | 85 | 0 | 253 |
| silk | 169 | 84 | 0 | 253 |
| spandex | 159 | 80 | 0 | 239 |
| stone | 163 | 83 | 0 | 246 |
| wood | 164 | 82 | 0 | 246 |

**Total: 1332 train samples, 668 test samples (2000 overall, 8 classes).**

## 5. Signal-level anomaly screen & repair

Every released tactile CSV and audio WAV was screened for recording
glitches after regeneration from the raw sources:

- **Tactile**: isolated jumps whose residual against an 11-point rolling
  median exceeds half the file's dynamic range (p99-p1) and 50 raw sensor
  units — physically impossible within 0.11s, i.e. unmistakable ADC
  glitches (e.g. single-sample spikes to ~1790 in sessions peaking at
  ~220). 824 points across 404 files were replaced by the local rolling
  median. Genuine press ramps and stick-slip transients were verified
  visually to sit far below this bar and are untouched.
- **Audio**: isolated electrical pops (>15 robust sigmas vs a 7-point
  rolling median, >10x file RMS, >25% full scale). 18 samples across 18
  files were interpolated.

All other content is byte-identical to the raw sensor recordings. The full
per-file findings are in `docs/logs/anomaly_check.log`.

## 6. Known, accepted limitations (see also docs/alignment_report.md)

- Video frame -> epoch alignment is a linear approximation (no independent video timestamp source was recorded); acceptable for cycle-level segmentation but not frame-accurate.
- The captured video resolution is 640x480 @ 30fps.
