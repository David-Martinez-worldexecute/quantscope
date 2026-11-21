"""The toy model format, and the feature/logit functions shared by run and verify.

The format is deliberately boring: ``config.json`` plus ``weights.npz`` holding
``W`` (n_labels x n_buckets) and ``b`` (n_labels,). A prompt becomes a hashed
bag of unigrams and bigrams, then a linear map produces the four choice logits.

This is a pipeline demonstrator, not a language model. It exists so that a fresh
clone can run the whole measurement chain — prompt, logits, prediction, pairs,
manifest, verify — without downloading weights or trusting a benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np

TOY_KIND = "toy-hashed-softmax"
N_LABELS = 4
LABELS = ("A", "B", "C", "D")
DEFAULT_BUCKETS = 256

_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list:
    """Split into ASCII words / numbers and single CJK characters."""
    return _TOKEN.findall(str(text).lower())


def featurize(prompt: str, n_buckets: int = DEFAULT_BUCKETS) -> np.ndarray:
    """Hash unigrams and bigrams into a fixed-width, L2-normalised vector."""
    tokens = tokenize(prompt)
    vector = np.zeros(int(n_buckets), dtype=np.float32)
    for token in tokens:
        vector[zlib.crc32(token.encode("utf-8")) % n_buckets] += 1.0
    for left, right in zip(tokens, tokens[1:]):
        pair = left + "\x00" + right
        vector[zlib.crc32(pair.encode("utf-8")) % n_buckets] += 1.0
    vector = np.sqrt(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / np.float32(norm)
    return vector.astype(np.float32)


def logits(weights: np.ndarray, bias: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Four choice logits for one prompt; identical arithmetic in run and verify."""
    return (np.asarray(weights, dtype=np.float32) @ np.asarray(features, dtype=np.float32)
            + np.asarray(bias, dtype=np.float32)).astype(np.float32)


def fit(prompts, answers, n_buckets: int = DEFAULT_BUCKETS, seed: int = 0,
        epochs: int = 400, lr: float = 0.5):
    """Train the toy head with full-batch gradient descent (deterministic)."""
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must have the same length")
    if len(prompts) == 0:
        raise ValueError("cannot fit on zero samples")
    features = np.stack([featurize(p, n_buckets) for p in prompts]).astype(np.float32)
    targets = np.zeros((len(prompts), N_LABELS), dtype=np.float32)
    for row, answer in enumerate(answers):
        targets[row, LABELS.index(answer)] = 1.0

    rng = np.random.default_rng(int(seed))
    weights = (rng.standard_normal((N_LABELS, int(n_buckets))) * 0.01).astype(np.float32)
    bias = np.zeros(N_LABELS, dtype=np.float32)

    for _ in range(int(epochs)):
        scores = features @ weights.T + bias
        scores = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        gradient = (probabilities - targets) / np.float32(len(features))
        weights = (weights - np.float32(lr) * (gradient.T @ features)).astype(np.float32)
        bias = (bias - np.float32(lr) * gradient.sum(axis=0)).astype(np.float32)
    return weights, bias


def save_model(path, weights, bias, n_buckets: int = DEFAULT_BUCKETS, seed: int = 0) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "weights.npz",
             W=np.asarray(weights, dtype=np.float32),
             b=np.asarray(bias, dtype=np.float32))
    config = {
        "kind": TOY_KIND,
        "n_buckets": int(n_buckets),
        "n_labels": N_LABELS,
        "seed": int(seed),
        "features": "hashed unigrams + bigrams, sqrt counts, L2 normalised",
    }
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return directory


def load_model(path):
    """Load a toy model directory and return weights, bias and config."""
    directory = Path(path)
    config_path = directory / "config.json"
    weights_path = directory / "weights.npz"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"{directory} does not look like a model directory "
            "(expected config.json and weights.npz)")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("kind") != TOY_KIND:
        raise ValueError(
            f"unsupported model kind {config.get('kind')!r}; this version loads {TOY_KIND!r}")
    with np.load(weights_path) as archive:
        weights = archive["W"].astype(np.float32)
        bias = archive["b"].astype(np.float32)
    expected = (N_LABELS, int(config["n_buckets"]))
    if weights.shape != expected or bias.shape != (N_LABELS,):
        raise ValueError(f"weights shape {weights.shape} does not match config {expected}")
    return {"dir": directory, "config": config, "weights": weights, "bias": bias}


def file_manifest(path) -> list:
    """Record every model file with its size and sha256, sorted by name."""
    directory = Path(path)
    rows = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if not entry.is_file():
            continue
        digest = hashlib.sha256()
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({
            "path": entry.name,
            "bytes": entry.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return rows
