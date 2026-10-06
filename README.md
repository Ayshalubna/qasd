# Qasd قصد — LoRA fine-tuning for banking intent routing

[![CI](https://github.com/Ayshalubna/qasd/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayshalubna/qasd/actions/workflows/ci.yml)
[![Train](https://github.com/Ayshalubna/qasd/actions/workflows/train.yml/badge.svg)](https://github.com/Ayshalubna/qasd/actions/workflows/train.yml)
**[Live demo ↗](https://lubna777-qasd.static.hf.space)** · MIT

A bank's contact centre receives thousands of messages a day ("my card hasn't arrived", "I was charged twice", "someone
is using my card"). Qasd reads each message, recognises which of 77 banking requests it is, and routes it to the team
that handles it. Fraud reports are flagged urgent, and low-confidence messages go to a human.

The core is a **parameter-efficient fine-tuning study**. The same pretrained DistilBERT is adapted three ways and
compared on accuracy, trained weights, training time, memory and checkpoint size:

1. **Full fine-tuning**: all 66M weights are updated.
2. **LoRA**: rank-8 adapters on the attention query/value projections plus a new classifier. About 1% of the weights are trained.
3. **Frozen encoder**: only the classifier is trained, as a baseline with a similar parameter budget.

The LoRA model is merged, exported to ONNX and quantised to int8. Two checks run before it ships: ONNX logits must
match PyTorch, and test accuracy is measured again after quantisation. The same file serves the FastAPI service
(ONNX Runtime, no PyTorch) and the in-browser demo (transformers.js).

## Results

All numbers come from the CI training job. It trains every method from scratch on a GitHub Actions CPU runner and
publishes the model, together with [`eval/RESULTS.md`](https://github.com/Ayshalubna/qasd/blob/model/eval/RESULTS.md),
to the `model` branch. The live demo's results page reads the same file.

<!-- RESULTS-TABLE -->

## How it is built

| | |
|---|---|
| **Data** | [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, CC BY 4.0): 10,003 training and 3,080 test messages across 77 intents. The data is checksum-verified. The official test set is left untouched, and 10 messages per intent are held out for validation. |
| **Training** | Plain PyTorch loop: AdamW, linear warm-up and decay, gradient clipping, best epoch chosen on validation. Each method runs in its own process, so peak memory is measured per method. |
| **LoRA** | `peft`, r = 8, α = 16, dropout 0.1, on `q_lin`/`v_lin`. `pre_classifier` and `classifier` are trained in full. A test checks that the merged model gives the same logits as model + adapter. |
| **Tracking** | MLflow logs parameters, learning curves, test metrics, time, memory and checkpoint size for every run. The CI job uploads the MLflow store as an artifact. |
| **Export** | `torch.onnx` (opset 17), with a check that logits match PyTorch for several batch sizes and paddings. Then dynamic int8 quantisation, and test accuracy and agreement with PyTorch are measured again. |
| **Serving** | FastAPI `POST /api/route` returns the team, the urgency and the top-k intents with probabilities. Below 50% confidence the message goes to a human triage queue. Built on ONNX Runtime and `tokenizers`, with no PyTorch in the image. |
| **In the browser** | transformers.js runs the same int8 ONNX file. Messages typed into the demo never leave the browser. |
| **Quality gates** | Training fails if LoRA accuracy drops below 85%, if LoRA trains more than 2% of the weights, or if int8 predictions agree with PyTorch less than 98% of the time. |
| **Tests** | A tiny DistilBERT is built from the training texts, so the whole pipeline (train → merge → ONNX → int8 → API) is tested in CI in minutes, with no downloads. |

## Run it

```bash
# serve the published model (no PyTorch needed)
pip install -r requirements.txt
python -m scripts.fetch_model && python -m scripts.serve      # http://localhost:8000, API docs at /docs

curl -s localhost:8000/api/route -H 'content-type: application/json' -d '{"text": "someone has used my card without permission"}'

# train everything yourself (CPU is fine; about an hour)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
python -m scripts.train
```

Docker: `docker build -t qasd . && docker run -p 8000:8000 qasd`. Tests: `pip install -r requirements-dev.txt && pytest -q`.

## Limits

- The model covers English only and only these 77 intents. A real deployment needs the bank's own labelled messages, including Arabic.
- Banking77 messages are short, single requests; real messages often mix several. The confidence threshold and human queue are the safety net.
- 6 of the 3,080 test messages also appear word for word in the training data (a known property of Banking77). They were left in, so the published test set stays comparable with other work.

---

Built by Ayesha Lubna.
