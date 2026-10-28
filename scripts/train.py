"""Train all three methods, export the deployed one, and write the results.

    python -m scripts.train                       # DistilBERT, as in CI (about 40-60 minutes on a 4-core CPU)
    python -m scripts.train --base /tmp/tiny --epochs 1 --limit 600     # quick offline check with a tiny model
    python -m scripts.train --skip-training       # CI: each method trains in its own parallel job, this step reports
Each method runs in its own process so peak memory is measured per method.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from qasd.config import BASE_MODEL, EXPORT, METHODS, ROOT, RUNS

DEPLOY = "lora"
GATES = {"lora_min_accuracy": 0.85, "lora_max_trainable_pct": 2.0, "int8_max_accuracy_drop": 0.015}


def findings(m: dict, ex: dict) -> list[str]:
    full, lora, head = m.get("full"), m["lora"], m.get("head")
    out = []
    if full:
        gap = 100 * (full["test_accuracy"] - lora["test_accuracy"])
        rel = "below" if gap > 0 else "above"
        out.append(f"LoRA trained {lora['trainable_pct']:.1f}% of the weights and reached {100 * lora['test_accuracy']:.1f}% test accuracy, "
                   f"{abs(gap):.1f} points {rel} full fine-tuning ({100 * full['test_accuracy']:.1f}%).")
        out.append(f"Each extra task costs a {lora['task_checkpoint_mb']:.1f} MB adapter instead of a {full['task_checkpoint_mb']:.0f} MB model copy "
                   f"({full['task_checkpoint_mb'] / lora['task_checkpoint_mb']:.0f}× smaller).")
        mem = (full["peak_memory_mb"] - lora["peak_memory_mb"]) / full["peak_memory_mb"]
        tm = (full["train_seconds"] / full["config"]["epochs"]) / (lora["train_seconds"] / lora["config"]["epochs"])
        out.append(f"Peak training memory was {lora['peak_memory_mb'] / 1024:.2f} GB with LoRA vs {full['peak_memory_mb'] / 1024:.2f} GB "
                   f"({'%.0f%% less' % (100 * mem) if mem > 0 else 'no saving'}); one LoRA epoch took {1 / tm:.2f}× the time of a full fine-tuning epoch on CPU.")
    if head:
        out.append(f"Training only the classifier on a frozen encoder ({head['trainable_pct']:.1f}% of the weights, a similar budget) reached "
                   f"{100 * head['test_accuracy']:.1f}%: {'adapting the attention layers is what closes the gap' if lora['test_accuracy'] - head['test_accuracy'] > 0.03 else 'on this task a frozen encoder is nearly as good'}.")
    if ex.get("onnx_int8") and ex.get("onnx_fp32"):
        d = 100 * (ex["onnx_int8"]["test_accuracy"] - ex["onnx_fp32"]["test_accuracy"])
        out.append(f"int8 quantisation shrank the deployed model from {ex['onnx_fp32']['size_mb']:.0f} MB to {ex['onnx_int8']['size_mb']:.0f} MB; "
                   f"accuracy changed by {d:+.1f} points and {100 * ex['onnx_int8']['agreement_with_pytorch']:.1f}% of predictions match PyTorch.")
    return out


def report(res: dict) -> str:
    m, ex = res["methods"], res["export"]
    lines = ["# Qasd — fine-tuning results", "",
             f"Banking77, {res['data']['train']:,} training / {res['data']['valid']} validation / {res['data']['test']:,} test messages, "
             f"77 intents. Base model `{res['base_model']}`. Hardware: {res['hardware']}. Reproduce with `python -m scripts.train`.", "",
             "| Method | Test accuracy | Macro-F1 | Trained weights | Training time | Peak memory | Per-task checkpoint |",
             "|---|---|---|---|---|---|---|"]
    for k in ("full", "lora", "head"):
        if k in m:
            r = m[k]
            lines.append(f"| {'**' + r['label'] + '** (deployed)' if k == res['deployed'] else r['label']} | {100 * r['test_accuracy']:.1f}% | "
                         f"{100 * r['test_macro_f1']:.1f}% | {r['trainable_params'] / 1e6:.2f}M ({r['trainable_pct']:.2f}%) | "
                         f"{r['train_seconds'] / 60:.1f} min | {r['peak_memory_mb'] / 1024:.2f} GB | {r['task_checkpoint_mb']:.1f} MB |")
    if ex:
        lines += ["", "## Deployed model (ONNX)", "", "| | Test accuracy | Agreement with PyTorch | Size | Latency (1 query, 1 thread) |", "|---|---|---|---|---|"]
        for k, lab in (("onnx_fp32", "fp32"), ("onnx_int8", "int8 (served)")):
            e = ex[k]
            lines.append(f"| {lab} | {100 * e['test_accuracy']:.1f}% | {100 * e['agreement_with_pytorch']:.1f}% | {e['size_mb']} MB | {e['latency_ms_single_query_1_thread']} ms |")
    lines += ["", "## Findings", ""] + [f"- {f}" for f in res["findings"]]
    worst = sorted(m[res["deployed"]]["per_class_f1"].items(), key=lambda x: x[1])[:5]
    lines += ["", "Hardest intents for the deployed model (F1): " + ", ".join(f"{k.replace('_', ' ')} ({100 * v:.0f}%)" for k, v in worst) + "."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--methods", default="head,lora,full")
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--skip-training", action="store_true", help="only export and report (runs/ already filled, e.g. by parallel CI jobs)")
    a = ap.parse_args()
    for method in ([] if a.skip_training else a.methods.split(",")):
        cmd = [sys.executable, "-m", "qasd.train", "--method", method, "--base", a.base]
        if a.epochs:
            cmd += ["--epochs", str(a.epochs)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "-m", "qasd.export", "--run", str(RUNS / DEPLOY), "--out", str(EXPORT)], check=True)
    methods = {k: json.loads((RUNS / k / "metrics.json").read_text()) for k in METHODS if (RUNS / k / "metrics.json").exists()}
    ex = json.loads((EXPORT / "export_check.json").read_text())
    any_run = next(iter(methods.values()))
    hw = os.getenv("QASD_HARDWARE") or f"{platform.machine()} CPU, {os.cpu_count()} cores (PyTorch threads: {any_run['threads']})"
    res = {"base_model": a.base if "/" not in a.base or not Path(a.base).exists() else "tiny test model", "deployed": DEPLOY,
           "hardware": hw, "data": {"train": any_run["n_train"], "valid": 770 if not a.limit else a.limit, "test": any_run["n_test"]},
           "methods": methods, "export": ex}
    res["findings"] = findings(methods, ex)
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "results.json").write_text(json.dumps(res, indent=1))
    (ROOT / "eval" / "RESULTS.md").write_text(report(res))
    (EXPORT / "results.json").write_text(json.dumps(res))       # published with the model for the web demo
    print(report(res))
    if a.no_gates:
        return 0
    lora = methods[DEPLOY]
    fails = []
    if lora["test_accuracy"] < GATES["lora_min_accuracy"]:
        fails.append(f"LoRA accuracy {lora['test_accuracy']:.3f} < {GATES['lora_min_accuracy']}")
    if lora["trainable_pct"] > GATES["lora_max_trainable_pct"]:
        fails.append(f"LoRA trains {lora['trainable_pct']:.2f}% of weights")
    drop = lora["test_accuracy"] - ex["onnx_int8"]["test_accuracy"]
    if drop > GATES["int8_max_accuracy_drop"]:
        fails.append(f"int8 model loses {100 * drop:.1f} accuracy points (limit {100 * GATES['int8_max_accuracy_drop']:.1f})")
    if fails:
        print("GATE FAILED:\n  " + "\n  ".join(fails))
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
