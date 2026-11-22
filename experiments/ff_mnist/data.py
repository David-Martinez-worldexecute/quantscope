"""Read MNIST idx files without torchvision, and hand back a small subset.

Why not ``torchvision.datasets.MNIST``: this experiment is meant to run inside a
CPU-only CI job that installs exactly one extra dependency (torch). torchvision
would drag in PIL and a pile of transforms for something that is, at heart, four
``struct.unpack`` calls.

The four official files are accepted either plain or gzipped, and looked up
directly in ``base`` or in ``base/raw`` (the layout ``torchvision`` uses).
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

# Two public mirrors of the same four gzipped files. The first is the one the
# PyTorch project uses; the second is Google's long-standing copy, kept as a
# fallback because a single mirror is a single point of failure for CI.
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
BASE_URL = MIRRORS[0]

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}

PIXELS = 28 * 28


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def read_idx_images(path: Path) -> np.ndarray:
    """Return ``(n, 784)`` uint8 images."""
    with _open(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: expected magic 2051, got {magic}")
        if rows * cols != PIXELS:
            raise ValueError(f"{path}: expected 28x28 pixels, got {rows}x{cols}")
        payload = handle.read(count * PIXELS)
    if len(payload) != count * PIXELS:
        raise ValueError(f"{path}: truncated, got {len(payload)} of {count * PIXELS} bytes")
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, PIXELS)


def read_idx_labels(path: Path) -> np.ndarray:
    with _open(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: expected magic 2049, got {magic}")
        payload = handle.read(count)
    if len(payload) != count:
        raise ValueError(f"{path}: truncated, got {len(payload)} of {count} labels")
    return np.frombuffer(payload, dtype=np.uint8).copy()


def find(base: Path, stem: str) -> Path:
    """Locate ``stem`` (optionally ``.gz``) under ``base`` or ``base/raw``."""
    for candidate in (
        base / stem,
        base / f"{stem}.gz",
        base / "raw" / stem,
        base / "raw" / f"{stem}.gz",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{stem} not found under {base} (looked for plain and .gz, with and without raw/)"
    )


def download_file(base: Path, stem: str) -> Path:
    """Fetch ``<stem>.gz`` into ``base``, trying each mirror in turn.

    Streamed in 1 MiB chunks with progress, because the training images are
    47 MB and a single ``read()`` gives no sign of life for minutes on a slow
    link. A mirror that stalls or fails is skipped; if all fail the error lists
    every attempt instead of only the last one.
    """
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{stem}.gz"
    if target.is_file() and target.stat().st_size > 0:
        return target

    tmp = target.with_suffix(".gz.part")
    failures: list[str] = []
    for mirror in MIRRORS:
        url = f"{mirror}{stem}.gz"
        print(f"[data] downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as response, tmp.open("wb") as fh:
                expected = int(response.headers.get("content-length") or 0)
                written = 0
                next_report = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    if written >= next_report:
                        total = f"/{expected / 1e6:.1f}" if expected else ""
                        print(f"[data]   {stem} {written / 1e6:.1f}{total} MB")
                        next_report = written + 8 * (1 << 20)
            if written == 0:
                raise ValueError("server returned an empty body")
            if expected and written != expected:
                raise ValueError(f"short read: {written} of {expected} bytes")
            tmp.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001 - report and try the next mirror
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            tmp.unlink(missing_ok=True)
            print(f"[data] mirror failed, trying the next one")
    raise RuntimeError(
        "could not download " + f"{stem}.gz from any mirror. Attempts: "
        + " | ".join(failures))


def download(base: Path) -> Path:
    """Fetch the four gzipped files into ``base``. Returns ``base``."""
    for stem in FILES.values():
        download_file(base, stem)
    return base


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with _open(path) as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(base: Path, train_n: int, test_n: int, seed: int = 0) -> dict:
    """Load a deterministic subset.

    The subset indices depend on ``seed`` only, so two runs with different
    quantisation settings but the same seed see *exactly* the same images —
    which is what makes the aligned-vs-row comparison paired rather than
    "two similar-looking runs".
    """
    paths = {key: find(base, stem) for key, stem in FILES.items()}
    x_train_all = read_idx_images(paths["train_images"])
    y_train_all = read_idx_labels(paths["train_labels"])
    x_test_all = read_idx_images(paths["test_images"])
    y_test_all = read_idx_labels(paths["test_labels"])

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(x_train_all))[:train_n]
    test_idx = rng.permutation(len(x_test_all))[:test_n]

    return {
        "x_train": x_train_all[train_idx],
        "y_train": y_train_all[train_idx],
        "x_test": x_test_all[test_idx],
        "y_test": y_test_all[test_idx],
        "source": {
            "train_images": paths["train_images"].name,
            "test_images": paths["test_images"].name,
            "sha256_train_images": sha256_of(paths["train_images"]),
            "sha256_test_images": sha256_of(paths["test_images"]),
        },
    }
