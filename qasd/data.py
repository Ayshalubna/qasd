"""Banking77 (PolyAI, CC BY 4.0): 13,083 customer messages labelled with 77 banking intents."""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict

from .config import CHECKSUMS, DATA, SEED


def _read(name: str) -> list[tuple[str, str]]:
    path = DATA / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKSUMS[name]:
        raise ValueError(f"{name}: unexpected checksum")
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["text"].strip(), r["category"]) for r in csv.DictReader(f)]


def labels() -> list[str]:
    return sorted({c for _, c in _read("train.csv")})


def splits(valid_per_class: int = 10) -> dict[str, list[tuple[str, int]]]:
    """Official test set untouched; a stratified validation set (10 per intent) carved from the training data."""
    labs = labels()
    idx = {c: i for i, c in enumerate(labs)}
    by_class = defaultdict(list)
    for t, c in _read("train.csv"):
        by_class[c].append(t)
    rng = random.Random(SEED)
    train, valid = [], []
    for c in labs:
        items = by_class[c][:]
        rng.shuffle(items)
        valid += [(t, idx[c]) for t in items[:valid_per_class]]
        train += [(t, idx[c]) for t in items[valid_per_class:]]
    rng.shuffle(train)
    test = [(t, idx[c]) for t, c in _read("test.csv")]
    return {"train": train, "valid": valid, "test": test}
