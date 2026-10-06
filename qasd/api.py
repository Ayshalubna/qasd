"""Routing API: ONNX Runtime + tokenizers, no PyTorch.  Run: python -m scripts.serve  (docs at /docs)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import EXPORT, ROOT
from .routing import ROUTE, pretty

CONFIDENT = 0.5


class RouteRequest(BaseModel):
    text: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(3, ge=1, le=10)


def create_app(export_dir: Path = EXPORT) -> FastAPI:
    app = FastAPI(title="Qasd — banking intent routing", version="1.0.0",
                  description="Routes bank customer messages to the right team with a LoRA-fine-tuned DistilBERT (int8 ONNX).")
    state: dict = {}

    def clf():
        if "clf" not in state:
            if not (export_dir / "onnx" / "model_quantized.onnx").exists():
                raise HTTPException(503, "Model not found. Run `python -m scripts.fetch_model` or `python -m scripts.train`.")
            from .export import OnnxClassifier
            state["clf"] = OnnxClassifier(export_dir)
        return state["clf"]

    @app.get("/health")
    def health():
        ok = (export_dir / "onnx" / "model_quantized.onnx").exists()
        return {"status": "ok" if ok else "model missing", "model": "distilbert+lora int8 onnx"}

    @app.post("/api/route")
    def route(body: RouteRequest):
        c = clf()
        t0 = time.perf_counter()
        logits = c.logits([body.text.strip()])[0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(-p)[: body.top_k]
        intents = [{"intent": c.labels[i], "label": pretty(c.labels[i]), "probability": round(float(p[i]), 4),
                    "team": ROUTE[c.labels[i]][0]} for i in top]
        best = intents[0]
        confident = best["probability"] >= CONFIDENT
        team, urgency = ROUTE[best["intent"]] if confident else ("Human triage queue", "normal")
        return {"route": {"team": team, "urgency": urgency, "confident": confident}, "intents": intents,
                "ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.get("/api/results")
    def results():
        f = ROOT / "eval" / "results.json"
        if not f.exists():
            raise HTTPException(404, "no results yet")
        return json.loads(f.read_text())

    web = ROOT / "web"
    if web.exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


api = create_app()
