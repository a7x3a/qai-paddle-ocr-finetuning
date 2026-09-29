# PaddleOCR Kurdish Recognition Fine-Tuning Leaderboard

Comparative evaluation of un-finetuned foundation models against domain-adapted checkpoints.

| Model / Checkpoint | CER (Micro) | WER (Micro) | Exact Match Acc | Avg Latency (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `iter_epoch_5.pdparams` | **0.4159** | 0.2476 | 0.7525 | 3.21 | 311.9 |
| `latest.pdparams` | **0.4159** | 0.2476 | 0.7525 | 3.20 | 312.4 |
| `best_accuracy.pdparams` | 0.4202 | **0.2462** | **0.7535** | 3.32 | 300.9 |
| `iter_epoch_1.pdparams` | 0.4261 | 0.2568 | 0.7429 | 3.25 | 307.8 |
| `iter_epoch_3.pdparams` | 0.4267 | 0.2629 | 0.7373 | 3.26 | 306.9 |
| `iter_epoch_2.pdparams` | 0.4381 | 0.2663 | 0.7336 | 3.27 | 305.7 |
| `iter_epoch_4.pdparams` | 0.5307 | 0.3804 | 0.6290 | 3.29 | 303.9 |

## Metric Definitions
- **CER (Micro)**: Total character Levenshtein edit distance divided by total reference character count.
- **WER (Micro)**: Total word Levenshtein edit distance divided by total reference word count.
- **Exact Match Acc**: Proportion of model predictions exactly matching target strings.
- **Latency (ms)**: Wall-clock duration per single image recognition request.
- **Throughput (FPS)**: Inference throughput in frames processed per second.
