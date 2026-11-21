"""Run a quantization comparison and write a manifest that can be re-checked."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import model as model_mod
from . import protocol, quant

SCHEMA = "quantscope-run-v2"
CLI_VERSION = "0.1.0"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path) -> list:
    """Read JSONL questions, reject malformed records, keep file order."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise protocol.ProtocolError(
                    f"{path}:{line_no} is not valid JSON ({error.msg})") from None
            rows.append(protocol.validate_question(record))
    if not rows:
        raise protocol.ProtocolError(f"{path} contains no questions")
    return rows


def pair_counts(reference_correct, candidate_correct) -> dict:
    """Count prediction flips between two modes.

    Keys read as ``nXY`` where X is the reference verdict and Y the candidate
    verdict (1 = correct). ``n10`` is "reference right, candidate wrong" and
    ``n01`` the reverse, so ``delta_pp = 100 * (n01 - n10) / n`` matches the
    accuracy difference exactly.
    """
    counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    reference = list(reference_correct)
    candidate = list(candidate_correct)
    if len(reference) != len(candidate):
        raise protocol.ProtocolError("pair counting needs two equal-length prediction lists")
    for ref, cand in zip(reference, candidate):
        counts[f"n{int(bool(ref))}{int(bool(cand))}"] += 1
    return counts


def _parse_modes(modes) -> list:
    if isinstance(modes, str):
        modes = [part.strip() for part in modes.split(",") if part.strip()]
    modes = list(modes)
    if not modes:
        raise protocol.ProtocolError("at least one mode is required")
    for mode in modes:
        if mode not in quant.MODES:
            raise protocol.ProtocolError(f"unknown mode {mode!r}; expected one of {quant.MODES}")
    if len(set(modes)) != len(modes):
        raise protocol.ProtocolError("modes must not repeat")
    return modes


def run(model_dir, data_path, n, modes=("fp32", "int4", "ternary"), group_size: int = 64,
        out_dir="runs/run", source_repo=None, source_revision=None):
    """Evaluate every mode on the same questions and return (manifest, output dir)."""
    modes = _parse_modes(modes)
    questions = load_questions(data_path)
    protocol.require_sample_count(questions, int(n))
    questions = questions[:int(n)]

    loaded = model_mod.load_model(model_dir)
    source = protocol.validate_source(source_repo, source_revision)
    buckets = int(loaded["config"]["n_buckets"])
    baseline = modes[0]

    items_by_mode = {}
    summaries = {}
    for mode in modes:
        weights = quant.quantize(loaded["weights"], mode, group_size)
        items = []
        for question in questions:
            prompt = protocol.build_prompt(question["question"], question["options"])
            features = model_mod.featurize(prompt, buckets)
            scores = model_mod.logits(weights, loaded["bias"], features)
            prediction = protocol.argmax_label(scores)
            items.append({
                "id": question["id"],
                "answer": question["answer"],
                "pred": prediction,
                "correct": bool(prediction == question["answer"]),
                "logits": [round(float(value), 6) for value in scores],
            })
        items_by_mode[mode] = items
        correct = sum(1 for item in items if item["correct"])
        summaries[mode] = {
            "correct": correct,
            "n": len(items),
            "accuracy": correct / len(items),
            "weights": quant.describe(mode, group_size),
        }

    for mode in modes:
        if mode == baseline:
            continue
        counts = pair_counts(
            [item["correct"] for item in items_by_mode[baseline]],
            [item["correct"] for item in items_by_mode[mode]],
        )
        summaries[mode]["vs"] = baseline
        summaries[mode]["pairs"] = counts
        summaries[mode]["delta_pp"] = round(
            100.0 * (summaries[mode]["accuracy"] - summaries[baseline]["accuracy"]), 4)

    manifest = {
        "schema": SCHEMA,
        "protocol_id": protocol.PROTOCOL_ID,
        "cli_version": CLI_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "dir": str(model_dir),
            "kind": loaded["config"].get("kind"),
            "files": model_mod.file_manifest(loaded["dir"]),
        },
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "n": len(questions),
        },
        "quant": {
            "modes": modes,
            "baseline": baseline,
            "group_size": int(group_size),
        },
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "source": source,
        "items": items_by_mode,
        "summary": summaries,
    }
    # Written last on purpose: a manifest without this flag is treated as
    # truncated by verify, so a crash mid-write can never look like a result.
    manifest["complete"] = True

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for mode, items in items_by_mode.items():
        with (out / f"items_{mode}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest, out


def summary_table(manifest) -> str:
    """Render the summary as fixed-width text for the terminal."""
    modes = manifest["quant"]["modes"]
    baseline = manifest["quant"]["baseline"]
    rows = [("mode", "correct", "n", "accuracy", "vs", "n01", "n10", "delta_pp")]
    for mode in modes:
        entry = manifest["summary"][mode]
        pairs = entry.get("pairs", {})
        rows.append((
            mode,
            str(entry["correct"]),
            str(entry["n"]),
            f"{100.0 * entry['accuracy']:.2f}%",
            entry.get("vs", "-"),
            str(pairs.get("n01", "-")),
            str(pairs.get("n10", "-")),
            f"{entry['delta_pp']:+.2f}" if "delta_pp" in entry else "-",
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(widths))))
    del baseline
    return "\n".join(lines)
