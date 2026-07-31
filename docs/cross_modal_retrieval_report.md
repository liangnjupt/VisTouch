# VisTouch Task: Cross-Modal Retrieval

Two small 1D-CNN encoders (audio, tactile) trained with symmetric InfoNCE contrastive loss for 60 epochs on 80 real train `cycle` samples; evaluated on 40 held-out real test `cycle` samples (9N split).

| query -> gallery | Recall@1 | Recall@5 | chance@1 | chance@5 |
|---|---|---|---|---|
| audio -> tactile | 0.050 | 0.250 | 0.025 | 0.125 |
| tactile -> audio | 0.050 | 0.175 | 0.025 | 0.125 |

![demo](assets/cross_modal_retrieval_demo.gif)
