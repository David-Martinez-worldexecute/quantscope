"""Re-check a finished run against its own manifest.

The verifier deliberately trusts nothing but the recorded bytes:

* the manifest must carry ``complete: true`` (written last by ``run``);
* every prediction must equal the argmax of its stored logits;
* every ``correct`` flag must follow from that prediction and the recorded answer;
* every per-mode count, accuracy, pair count and delta must be recomputed;
* when the model directory or the question file is still on disk, their
  sha256 values must match what the manifest recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import protocol
from .run import pair_counts, sha256_file

ACCURACY_TOLERANCE = 1e-9
DELTA_TOLERANCE = 1e-3


def verify(run_dir, model_dir=None) -> dict:
    issues = []
    checks = {
        "manifest_read": False,
        "complete": False,
        "protocol": False,
        "items_checked": 0,
        "model_hashes": "skipped",
        "data_hash": "skipped",
    }
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "fail",
            "issues": [f"manifest not found: {manifest_path}"],
            "checks": checks,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks["manifest_read"] = True

    if manifest.get("complete") is not True:
        issues.append("manifest.complete is not true: the run is truncated or was edited")
    else:
        checks["complete"] = True

    if manifest.get("protocol_id") != protocol.PROTOCOL_ID:
        issues.append(
            f"protocol_id {manifest.get('protocol_id')!r} does not match {protocol.PROTOCOL_ID!r}")
    else:
        checks["protocol"] = True

    quant_block = manifest.get("quant") or {}
    modes = list(quant_block.get("modes") or [])
    baseline = quant_block.get("baseline")
    if not modes:
        issues.append("manifest.quant.modes is empty")
    if baseline not in modes:
        issues.append(f"baseline {baseline!r} is not one of the recorded modes")

    items = manifest.get("items") or {}
    accuracies = {}
    for mode in modes:
        records = items.get(mode)
        if not records:
            issues.append(f"no per-item records for mode {mode!r}")
            continue
        correct = 0
        for record in records:
            label = f"{mode}/{record.get('id')}"
            logits = record.get("logits")
            if not isinstance(logits, list) or len(logits) != len(protocol.LABELS):
                issues.append(f"{label}: expected {len(protocol.LABELS)} logits")
                continue
            derived = protocol.argmax_label(logits)
            if derived != record.get("pred"):
                issues.append(f"{label}: pred {record.get('pred')!r} != argmax {derived!r}")
            expected = bool(derived == record.get("answer"))
            if bool(record.get("correct")) != expected:
                issues.append(f"{label}: correct flag disagrees with pred and answer")
            correct += int(expected)
        accuracy = correct / len(records)
        accuracies[mode] = accuracy
        summary = (manifest.get("summary") or {}).get(mode) or {}
        if summary.get("correct") != correct:
            issues.append(
                f"{mode}: summary.correct {summary.get('correct')!r} != recomputed {correct}")
        if abs(float(summary.get("accuracy", -1.0)) - accuracy) > ACCURACY_TOLERANCE:
            issues.append(
                f"{mode}: summary.accuracy {summary.get('accuracy')!r} != recomputed {accuracy}")

    for mode in modes:
        if mode == baseline or mode not in accuracies or baseline not in accuracies:
            continue
        reference = [record["correct"] for record in items[baseline]]
        candidate = [record["correct"] for record in items[mode]]
        counts = pair_counts(reference, candidate)
        summary = (manifest.get("summary") or {}).get(mode) or {}
        if summary.get("pairs") != counts:
            issues.append(f"{mode}: pair counts {summary.get('pairs')!r} != recomputed {counts}")
        delta = round(100.0 * (accuracies[mode] - accuracies[baseline]), 4)
        if abs(float(summary.get("delta_pp", 0.0)) - delta) > DELTA_TOLERANCE:
            issues.append(
                f"{mode}: delta_pp {summary.get('delta_pp')!r} != recomputed {delta}")

    checks["items_checked"] = sum(len(records) for records in items.values())

    model_block = manifest.get("model") or {}
    target = Path(model_dir) if model_dir else Path(str(model_block.get("dir", "")))
    files = model_block.get("files") or []
    if files and target.is_dir():
        for row in files:
            path = target / row["path"]
            if not path.is_file():
                issues.append(f"model file missing: {path}")
                continue
            if path.stat().st_size != row["bytes"]:
                issues.append(f"model file size changed: {row['path']}")
            if sha256_file(path) != row["sha256"]:
                issues.append(f"model file hash changed: {row['path']}")
        checks["model_hashes"] = f"verified ({len(files)} files)"
    elif files:
        checks["model_hashes"] = f"skipped (model directory not found: {target})"

    data_block = manifest.get("data") or {}
    data_path = Path(str(data_block.get("path", "")))
    if data_block.get("sha256") and data_path.is_file():
        if sha256_file(data_path) != data_block["sha256"]:
            issues.append(f"question file hash changed: {data_path}")
        else:
            checks["data_hash"] = "verified"
    elif data_block.get("sha256"):
        checks["data_hash"] = f"skipped (question file not found: {data_path})"

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "checks": checks,
    }
