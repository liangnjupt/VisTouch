# VisTouch Task: Tactile Super-Resolution / Denoising

Model: small 1D CNN (4 conv layers) trained 40 epochs on 1332 real train tactile curves; evaluated on 668 held-out real test curves (9N split).

Corruption (train+eval input only, never written back to the dataset): downsample by 5x then linearly upsample, + Gaussian noise (std=0.2).

| metric | low-rate+noisy input (baseline) | model reconstruction |
|---|---|---|
| MSE (z-normalized) | 0.0487 | 0.0074 |

**MSE improvement over the naive upsampled input: 84.9%.** Pearson correlation between reconstruction and real ground truth: 0.996.

![demo](assets/tactile_sr_demo.gif)
