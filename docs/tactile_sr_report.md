# VisTouch Task: Tactile Super-Resolution / Denoising

Model: small 1D CNN (4 conv layers) trained 40 epochs on 5049 materialized 2 s train tactile curves; evaluated on 2506 held-out test curves (9N split).

Corruption (train+eval input only, never written back to the dataset): downsample by 5x then linearly upsample, + Gaussian noise (std=0.2).

| metric | low-rate+noisy input (baseline) | model reconstruction |
|---|---|---|
| MSE (z-normalized) | 0.0512 | 0.0077 |

**MSE improvement over the naive upsampled input: 84.9%.** Pearson correlation between reconstruction and released ground truth: 0.993.

![demo](assets/tactile_sr_demo.gif)
