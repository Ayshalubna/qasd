"""Fast tests with a tiny randomly initialised DistilBERT built from the data (no downloads)."""
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from qasd.data import labels, splits
from qasd.routing import ROUTE, TEAMS


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    from scripts.make_tiny_model import main
    return main(str(tmp_path_factory.mktemp("tiny")))


@pytest.fixture(scope="session")
def trained(tiny, tmp_path_factory):
    from qasd.train import run
    runs = tmp_path_factory.mktemp("runs")
    res = {m: run(m, base=str(tiny), out_root=runs, epochs=1, limit=400, use_mlflow=False) for m in ("lora", "head", "full")}
    return runs, res


def test_data_splits():
    s = splits()
    assert (len(s["train"]), len(s["valid"]), len(s["test"])) == (9233, 770, 3080)
    assert len({t for t, _ in s["train"]} | {t for t, _ in s["valid"]}) > 9900
    counts = np.bincount([y for _, y in s["valid"]], minlength=77)
    assert (counts == 10).all()


def test_every_intent_has_a_team():
    assert set(ROUTE) == set(labels()) and len(ROUTE) == 77
    assert sum(len(v[1]) for v in TEAMS.values()) == 77


def test_parameter_budgets(trained):
    _, r = trained
    assert r["full"]["trainable_pct"] == 100.0
    assert r["lora"]["trainable_params"] < 0.2 * r["full"]["trainable_params"]
    assert r["lora"]["total_params"] == r["full"]["total_params"]           # same base model, not double-counted
    assert r["head"]["trainable_params"] < r["lora"]["trainable_params"]
    assert r["lora"]["task_checkpoint_mb"] < r["full"]["task_checkpoint_mb"]


def test_merged_lora_matches_adapter(tiny):
    from transformers import AutoTokenizer

    from qasd.config import METHODS
    from qasd.train import build
    torch.manual_seed(0)
    model = build("lora", str(tiny), labels(), METHODS["lora"]).eval()
    for n, p in model.named_parameters():                   # make the adapter non-trivial
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    tok = AutoTokenizer.from_pretrained(tiny)
    enc = tok(["my card has not arrived", "why was i charged a fee"], padding=True, return_tensors="pt")
    with torch.no_grad():
        a = model(**enc).logits
        b = model.merge_and_unload()(**enc).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_export_onnx_int8_and_api(trained, tmp_path):
    from qasd.api import create_app
    from qasd.export import OnnxClassifier, check, quantise, to_onnx
    runs, _ = trained
    fp32 = to_onnx(runs / "lora" / "model", tmp_path)                # raises if ONNX differs from PyTorch
    quantise(fp32)
    res = check(tmp_path, runs / "lora")
    assert res["onnx_fp32"]["agreement_with_pytorch"] > 0.999
    assert res["onnx_int8"]["size_mb"] < res["onnx_fp32"]["size_mb"]
    clf = OnnxClassifier(tmp_path)
    assert clf.logits(["hello", "a much longer message about my transfer"]).shape == (2, 77)
    client = TestClient(create_app(tmp_path))
    r = client.post("/api/route", json={"text": "I lost my card", "top_k": 3}).json()
    assert len(r["intents"]) == 3 and abs(sum(i["probability"] for i in r["intents"])) <= 1.0001
    assert r["route"]["team"] in set(TEAMS) | {"Human triage queue"}
    assert client.post("/api/route", json={"text": ""}).status_code == 422
    assert client.get("/health").json()["status"] == "ok"


def test_api_without_model(tmp_path):
    from qasd.api import create_app
    client = TestClient(create_app(tmp_path))
    assert client.post("/api/route", json={"text": "hello there"}).status_code == 503


def test_findings_wording():
    from scripts.train import findings
    m = {k: {"test_accuracy": a, "trainable_pct": p, "task_checkpoint_mb": mb, "peak_memory_mb": mem, "train_seconds": s,
             "config": {"epochs": e}} for k, a, p, mb, mem, s, e in
         [("full", 0.93, 100, 268, 3000, 900, 4), ("lora", 0.92, 1.2, 3.1, 2000, 1000, 6), ("head", 0.70, 0.9, 2.4, 1500, 300, 8)]}
    f = findings(m, {})
    assert "1.0 points below full fine-tuning" in f[0] and "86× smaller" in f[1] and "33% less" in f[2]
    assert "adapting the attention layers" in f[3]


def test_routing_json_matches_python():
    from qasd.config import ROOT
    web = json.loads((ROOT / "web" / "routing.json").read_text())
    assert web == {k: list(v) for k, v in ROUTE.items()}
