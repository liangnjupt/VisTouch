# VisTouch Task: Cross-Modal Generation (Sound -> Tactile Force Envelope)

Haptic signal recovery -- the cross-modal generation task showcased in the companion paper (*Cross-Modal Semantic Communications*, IEEE WCM 2022), which evaluates recovered haptic signals with MAE. A dilated residual CNN generates the low-frequency tactile force envelope from synchronized time-frequency audio features, trained 20 epochs on 5049 train clips, evaluated on 2506 held-out test clips (9N split).

| metric | value |
|---|---|
| MAE (z-normalized force envelope) | 0.3378 |
| Pearson correlation (generated vs real envelope) | 0.790 |

![demo](assets/cross_modal_generation_demo.gif)

## Notes

- Input and target come from the same synchronously interpolated 2 s clip derived from one real 0.5 s source window. The target is Gaussian low-pass filtered for semantic envelope recovery; released tactile files are never modified or replaced by this task.
