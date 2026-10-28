# Qasd — fine-tuning results

Banking77, 9,233 training / 770 validation / 3,080 test messages, 77 intents. Base model `distilbert-base-uncased`. Hardware: GitHub Actions runners (4-core CPU, no GPU), one per method. Reproduce with `python -m scripts.train`.

| Method | Test accuracy | Macro-F1 | Trained weights | Training time | Peak memory | Per-task checkpoint |
|---|---|---|---|---|---|---|
| Full fine-tuning | 91.3% | 91.3% | 67.01M (100.00%) | 61.1 min | 3.18 GB | 268.1 MB |
| **LoRA (r=8, attention q/v)** (deployed) | 92.0% | 91.9% | 0.80M (1.19%) | 73.1 min | 2.28 GB | 3.2 MB |
| Frozen encoder, train classifier only | 82.0% | 82.0% | 0.65M (0.97%) | 44.5 min | 1.57 GB | 2.6 MB |

## Deployed model (ONNX)

| | Test accuracy | Agreement with PyTorch | Size | Latency (1 query, 1 thread) |
|---|---|---|---|---|
| fp32 | 92.0% | 100.0% | 268.2 MB | 14.26 ms |
| int8 (served) | 91.0% | 96.8% | 67.6 MB | 3.33 ms |

## Findings

- LoRA trained 1.2% of the weights and reached 92.0% test accuracy, 0.7 points above full fine-tuning (91.3%).
- Each extra task costs a 3.2 MB adapter instead of a 268 MB model copy (84× smaller).
- Peak training memory was 2.28 GB with LoRA vs 3.18 GB (28% less); one LoRA epoch took 0.80× the time of a full fine-tuning epoch on CPU.
- Training only the classifier on a frozen encoder (1.0% of the weights, a similar budget) reached 82.0%: adapting the attention layers is what closes the gap.
- int8 quantisation shrank the deployed model from 268 MB to 68 MB; accuracy changed by -1.0 points and 96.8% of predictions match PyTorch.

Hardest intents for the deployed model (F1): balance not updated after bank transfer (77%), topping up by card (78%), compromised card (81%), pending transfer (82%), transfer not received by recipient (83%).
