"""Fine-tune one adaptation method and record what it cost.

    python -m qasd.train --method lora            # or: full, head
Runs as its own process so peak memory is measured per method (scripts/train.py runs all three).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import BASE_MODEL, MAX_LEN, METHODS, MLFLOW_URI, RUNS, SEED
from .data import labels, splits


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(method: str, base: str, labs: list[str], cfg: dict):
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labs), id2label=dict(enumerate(labs)), label2id={c: i for i, c in enumerate(labs)})
    model.base_param_count = sum(p.numel() for p in model.parameters())
    if method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        lc = LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                        target_modules=["q_lin", "v_lin"], modules_to_save=["pre_classifier", "classifier"])
        n = model.base_param_count
        model = get_peft_model(model, lc)
        model.base_param_count = n
    elif method == "head":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("pre_classifier", "classifier"))
    return model


def count(model) -> tuple[int, int]:
    """Trainable parameters, and the size of the underlying pretrained model (LoRA's frozen copies not double-counted)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, model.base_param_count


def batches(data, tok, size, shuffle, rng=None):
    order = list(range(len(data)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [data[j] for j in order[i:i + size]]
        enc = tok([t for t, _ in chunk], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        yield enc, torch.tensor([y for _, y in chunk])


@torch.no_grad()
def predict(model, tok, data, size=128) -> np.ndarray:
    model.eval()
    out = []
    for enc, _ in batches(data, tok, size, shuffle=False):
        out.append(model(**enc).logits.float().numpy())
    return np.concatenate(out)


def dir_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)


def run(method: str, base: str = BASE_MODEL, out_root: Path = RUNS, epochs: int | None = None, limit: int | None = None,
        use_mlflow: bool = True) -> dict:
    cfg = dict(METHODS[method])
    if epochs:
        cfg["epochs"] = epochs
    seed_all()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    labs = labels()
    data = splits()
    if limit:                                                   # quick runs in tests
        data = {k: v[:limit] for k, v in data.items()}
    tok = AutoTokenizer.from_pretrained(base)
    model = build(method, base, labs, cfg)
    trainable, total = count(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    steps = cfg["epochs"] * math.ceil(len(data["train"]) / cfg["batch"])
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    out = out_root / method
    out.mkdir(parents=True, exist_ok=True)

    mlf = None
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("qasd-banking77")
        mlf = mlflow.start_run(run_name=method)
        mlflow.log_params({"method": method, "base_model": base, "trainable_params": trainable, "total_params": total,
                           "max_len": MAX_LEN, **{k: v for k, v in cfg.items() if k != "label"}})

    rng = random.Random(SEED)
    history, best, best_state = [], -1.0, None
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for enc, y in batches(data["train"], tok, cfg["batch"], shuffle=True, rng=rng):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        val_acc = float((predict(model, tok, data["valid"]).argmax(1) == np.array([y for _, y in data["valid"]])).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_accuracy": val_acc,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[{method}] epoch {epoch}: loss {np.mean(losses):.4f}  valid acc {val_acc:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        if mlf:
            import mlflow
            mlflow.log_metrics({"train_loss": float(np.mean(losses)), "valid_accuracy": val_acc}, step=epoch)
        if val_acc > best:                                      # keep the best epoch on the validation set
            best = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)

    y_test = np.array([y for _, y in data["test"]])
    t1 = time.perf_counter()
    pred = predict(model, tok, data["test"]).argmax(1)
    infer_ms = (time.perf_counter() - t1) / len(y_test) * 1000

    # what has to be stored per task
    if method == "lora":
        model.save_pretrained(out / "adapter")
        task_mb = dir_mb(out / "adapter")
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "model")
    elif method == "head":
        model.save_pretrained(out / "model")
        torch.save({k: v for k, v in model.state_dict().items() if k.startswith(("pre_classifier", "classifier"))}, out / "head.pt")
        task_mb = round((out / "head.pt").stat().st_size / 1e6, 2)
    else:
        model.save_pretrained(out / "model")
        task_mb = dir_mb(out / "model")
    tok.save_pretrained(out / "model")

    f1 = f1_score(y_test, pred, average=None, labels=list(range(len(labs))), zero_division=0)
    result = {
        "method": method, "label": cfg["label"], "base_model": base, "config": {k: v for k, v in cfg.items() if k != "label"},
        "trainable_params": trainable, "total_params": total, "trainable_pct": 100 * trainable / total,
        "test_accuracy": float((pred == y_test).mean()), "test_macro_f1": float(f1.mean()),
        "best_valid_accuracy": best, "train_seconds": round(train_seconds, 1),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "task_checkpoint_mb": task_mb, "cpu_inference_ms_per_query": round(infer_ms, 2), "history": history,
        "per_class_f1": {labs[i]: float(v) for i, v in enumerate(f1)}, "n_train": len(data["train"]), "n_test": len(y_test),
        "threads": torch.get_num_threads(),
    }
    np.save(out / "test_pred.npy", pred)
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    if mlf:
        import mlflow
        mlflow.log_metrics({k: result[k] for k in ("test_accuracy", "test_macro_f1", "train_seconds", "peak_memory_mb",
                                                   "task_checkpoint_mb", "trainable_pct")})
        mlflow.log_artifact(str(out / "metrics.json"))
        mlflow.end_run()
    print(json.dumps({k: v for k, v in result.items() if k not in ("history", "per_class_f1")}), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHODS), required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    run(a.method, a.base, Path(a.runs), a.epochs, a.limit, not a.no_mlflow)


if __name__ == "__main__":
    main()
