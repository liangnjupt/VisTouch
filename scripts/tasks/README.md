# VisTouch downstream application scripts

Three extended-application baselines built on top of VisTouch (see the
Benchmarks section of the main `README.md` for results, demo GIFs, and
logs):

- `common.py` — shared loading utilities used by all three scripts:
  fixed-length (5s), contact-onset-aware, time-aligned audio/tactile
  windows cropped from the same real sample (the window starts shortly
  before the first sustained tactile contact, so crops are dominated by
  actual contact rather than pre-contact idle time; no synthetic data is
  introduced anywhere here).
- `tactile_super_resolution.py` — reconstruct a clean, high-rate tactile
  force curve from a low-rate / noisy input.
- `cross_modal_retrieval.py` — contrastive audio↔tactile embedding +
  retrieval.
- `cross_modal_generation.py` — generate the tactile force curve directly
  from the contact sound (the haptic signal recovery task showcased in the
  companion paper, evaluated with MAE).

Each script trains a small CPU-friendly model, saves weights under
`scripts/tasks/weights/`, and writes a report + demo GIF under `docs/` /
`docs/assets/` that's linked from the main README. Training/test console
records are archived in `docs/logs/`.

Windows note: running these alongside opencv can trigger an OpenMP
duplicate-runtime error; set `KMP_DUPLICATE_LIB_OK=TRUE` before running if
you hit it.
