"""Export the chosen fine-tuned model to ONNX, quantise it to int8, and check nothing was lost.

    python -m qasd.export --run runs/lora
Writes export/ in the layout transformers.js expects (config.json, tokenizer files, onnx/model_quantized.onnx),
so the same files serve the FastAPI service (onnxruntime) and the in-browser demo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .config import EXPORT, MAX_LEN, RUNS
from .data import splits


def to_onnx(model_dir: Path, out: Path) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    model.config.return_dict = False
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = tok(["how do i activate my card", "where is my money"], padding=True, return_tensors="pt")
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    path = out / "onnx" / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(path), input_names=["input_ids", "attention_mask"],
                          output_names=["logits"], opset_version=17, dynamo=False,
                          dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
                                        "logits": {0: "batch"}})
    model.config.save_pretrained(out)
    tok.save_pretrained(out)
    # the traced graph must give the same logits as PyTorch for other batch sizes and lengths (padding included)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    texts = [t for t, _ in splits()["test"][:48]]
    worst = 0.0
    for batch in ([texts[0]], texts[:7], texts[:48]):
        e = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            ref = model(e["input_ids"], attention_mask=e["attention_mask"])[0].numpy()
        got = sess.run(["logits"], {"input_ids": e["input_ids"].numpy(), "attention_mask": e["attention_mask"].numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    if worst > 1e-3:
        raise RuntimeError(f"ONNX export differs from PyTorch (max |diff| {worst:.2e})")
    print(f"ONNX matches PyTorch: max |logit diff| {worst:.1e}", flush=True)
    return path


# int8 recipes tried in order; the one that agrees best with PyTorch is shipped (all are reported)
RECIPES = {
    "per_channel_head_fp32": {"per_channel": True, "keep_head": True},   # classifier layers stay in full precision
    "per_channel": {"per_channel": True, "keep_head": False},
    "per_tensor": {"per_channel": False, "keep_head": False},
}


def quantise(fp32: Path, recipe: str = "per_channel_head_fp32", dest: Path | None = None) -> Path:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    r = RECIPES[recipe]
    exclude = []
    if r["keep_head"]:          # the last two matrix multiplications are pre_classifier and classifier
        mm = [n.name for n in onnx.load(str(fp32), load_external_data=False).graph.node if n.op_type in ("MatMul", "Gemm")]
        exclude = mm[-2:]
    q = dest or fp32.with_name("model_quantized.onnx")
    quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8, per_channel=r["per_channel"], nodes_to_exclude=exclude)
    return q


class OnnxClassifier:
    """Minimal inference used by the API and the checks: tokenizers + onnxruntime, no PyTorch."""

    def __init__(self, folder: Path, file: str = "model_quantized.onnx", threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        pad = self.tok.token_to_id("[PAD]") or 0
        self.tok.enable_padding(pad_id=pad, pad_token="[PAD]")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(folder / "onnx" / file), so, providers=["CPUExecutionProvider"])
        cfg = json.loads((folder / "config.json").read_text())
        self.labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]

    def logits(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        return self.sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]


def check(folder: Path, run_dir: Path) -> dict:
    """Accuracy of the fp32 and int8 ONNX models on the test set, agreement with PyTorch, size and latency."""
    test = splits()["test"]
    texts, y = [t for t, _ in test], np.array([c for _, c in test])
    torch_pred = np.load(run_dir / "test_pred.npy")
    out = {}
    for name, file in (("onnx_fp32", "model.onnx"), ("onnx_int8", "model_quantized.onnx")):
        clf = OnnxClassifier(folder, file)
        pred = np.concatenate([clf.logits(texts[i:i + 64]).argmax(1) for i in range(0, len(texts), 64)])
        t0 = time.perf_counter()
        for t in texts[:200]:
            clf.logits([t])
        n = len(torch_pred)
        out[name] = {"test_accuracy": float((pred == y).mean()), "agreement_with_pytorch": float((pred[:n] == torch_pred).mean()),
                     "size_mb": round((folder / "onnx" / file).stat().st_size / 1e6, 1),
                     "latency_ms_single_query_1_thread": round((time.perf_counter() - t0) / 200 * 1000, 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUNS / "lora"))
    ap.add_argument("--out", default=str(EXPORT))
    ap.add_argument("--keep-fp32", action="store_true", help="keep onnx/model.onnx (not needed for serving)")
    a = ap.parse_args()
    run_dir, out = Path(a.run), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    fp32 = to_onnx(run_dir / "model", out)
    trials = {}
    for recipe in RECIPES:
        quantise(fp32, recipe)
        trials[recipe] = check(out, run_dir)["onnx_int8"]
        print(f"int8 recipe {recipe}: {trials[recipe]}", flush=True)
    best = max(trials, key=lambda k: (trials[k]["agreement_with_pytorch"], -trials[k]["size_mb"]))
    quantise(fp32, best)
    result = check(out, run_dir)
    result["int8_recipe"] = best
    result["int8_recipes_tried"] = trials
    if not a.keep_fp32:
        fp32.unlink()
    (out / "export_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
