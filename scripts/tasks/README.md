# VisTouch downstream application scripts

Three extended-application baselines built on top of VisTouch (see the
Benchmarks section of the main `README.md` for results, demo GIFs, and
logs):

- `common.py` — shared loading utilities used by all three scripts:
  fixed-length 2 s, time-aligned audio/tactile clips loaded directly from
  the materialized public tier. Only contact clips are selected; every
  pair shares the same source epoch and 4x interpolation mapping.
- `tactile_super_resolution.py` — reconstruct a clean, high-rate tactile
  force curve from a low-rate / noisy input.
- `multimodal_tactile_completion.py` — reconstruct a contiguous missing
  tactile block from synchronized audio, video, and surrounding touch.
- `cross_modal_generation.py` — generate the low-frequency tactile force envelope
  from the contact sound (the haptic signal recovery task showcased in the
  companion paper, evaluated with MAE).

Each script trains a small CPU-friendly model, saves weights under
`scripts/tasks/weights/`, and writes a report + demo GIF under `docs/` /
`docs/assets/` that's linked from the main README. Training/test console
records are archived in `docs/logs/`.

Windows note: running these alongside opencv can trigger an OpenMP
duplicate-runtime error; set `KMP_DUPLICATE_LIB_OK=TRUE` before running if
you hit it.
