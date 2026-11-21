"""Refuse to publish a repo that carries lab leftovers.

Three separate mistakes have burned this project before, and each one is silent
until a stranger clones the repo:

* a local absolute path (drive letter, ``/home/...``, or a ``file:`` URL) leaking
  from the lab checkout into a public document;
* a personal or third-party artefact (a model directory, a persona blob) being
  copied in "just to make it work";
* a README whose first screen no longer contains a runnable command, because
  someone rewrote the opening paragraph.

The check is deliberately literal and has no exceptions to grant: exit 0 means
clean, exit 1 means fix the listed lines. URLs are stripped before the path
check, because ``https://`` contains text that looks like a drive letter.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    "quantscope/__init__.py",
    "quantscope/cli.py",
    "quantscope/protocol.py",
    "quantscope/quant.py",
    "quantscope/model.py",
    "quantscope/run.py",
    "quantscope/verify.py",
    "quantscope/report.py",
    "data/sample_questions.jsonl",
    "tests/support.py",
    "tests/test_protocol.py",
    "tests/test_pairs.py",
    "tests/test_manifest.py",
    "tests/test_tiny_model.py",
    "docs/reproduce.md",
    "docs/forward-forward-ternary.md",
    "docs/evidence/ff_controlled_evidence.json",
    "docs/evidence/ff_evidence.json",
    "docs/evidence/qwen35_forward.json",
    "docs/real-model-qwen3.5-0.8b.md",
    "scripts/sync_from_lab.py",
    "scripts/check_public.py",
]

SCAN_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml", ".json", ".jsonl", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".tmp", "runs", "build", "dist", ".venv", ".github"}

# The two tooling scripts name the private lab by construction: one holds the
# copy map, the other holds the patterns this very check enforces. Scanning them
# would flag the rule definition as if it were leaked content, so they are the
# only exempt paths -- and they are the two files a human reads before trusting
# the check at all.
EXEMPT_FILES = {"scripts/check_public.py", "scripts/sync_from_lab.py"}

URL_RE = re.compile(r"https?://\S+")
LOCAL_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/|file://")
# Words that should not appear in a repo whose only job is measurement.
PRIVATE_RE = re.compile(r"\.ollama|nagisa|acceptance_runs/|portfolio/", re.IGNORECASE)
SECRET_RE = re.compile(r"(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{8,}", re.IGNORECASE)

README_MIN_CHARS = 2000
README_MUST_CONTAIN = ("pip install", "quantscope run")


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in SCAN_SUFFIXES:
            yield path


def strip_urls(text: str) -> str:
    return URL_RE.sub("<url>", text)


def evidence_bundle_issues(payload: dict) -> list[str]:
    """The published bundle must carry the numbers the docs quote, and the two
    implementations of the blockwise first layer must agree item by item."""
    issues: list[str] = []
    summary = (payload.get("precision_matrix") or {}).get("summary") or {}
    for mode in ("fp32", "ternary", "first_fp32", "second_fp32"):
        if mode not in summary:
            issues.append(f"ff_evidence: missing precision_matrix.summary.{mode}")
    mechanism = payload.get("mechanism") or {}
    for key in (
        "blockwise_in_training",
        "two_bit_layer0",
        "two_bit_both",
        "label_at_second_layer",
    ):
        if key not in mechanism:
            issues.append(f"ff_evidence: missing mechanism.{key}")
    rows = ((mechanism.get("reproduction") or {}).get("rows")) or []
    if len(rows) != 3:
        issues.append("ff_evidence: reproduction should list three seeds")
    for row in rows:
        if row.get("same_prediction") != row.get("count"):
            issues.append(f"ff_evidence: reproduction seed {row.get('seed')} is not identical")
    group_scale = (payload.get("group_scale") or {}).get("variants") or {}
    for name in ("aligned_g11", "unaligned_g11", "aligned_g14"):
        if name not in group_scale:
            issues.append(f"ff_evidence: missing group_scale.variants.{name}")
    if "aligned_g11" in group_scale and "unaligned_g11" in group_scale:
        aligned = group_scale["aligned_g11"]
        unaligned = group_scale["unaligned_g11"]
        if aligned.get("groups") != unaligned.get("groups"):
            issues.append("ff_evidence: aligned/unaligned do not share a group count")
        if 784 not in {span[0] for span in (aligned.get("spans") or [])[1:]}:
            issues.append("ff_evidence: aligned_g11 has no span boundary at the label boundary")
        if 784 in {span[0] for span in (unaligned.get("spans") or [])[1:]}:
            issues.append("ff_evidence: unaligned_g11 has a span boundary at the label boundary")
    cifar = (payload.get("cifar") or {}).get("variants") or {}
    for name in ("fp32", "ternary_row", "aligned_g11", "unaligned_g11"):
        if name not in cifar:
            issues.append(f"ff_evidence: missing cifar.variants.{name}")
    if "aligned_g11" in cifar and "unaligned_g11" in cifar:
        definitions = (payload.get("cifar") or {}).get("group_definitions") or {}
        aligned = definitions.get("aligned_g11") or []
        unaligned = definitions.get("unaligned_g11") or []
        if not aligned or len(aligned) != len(unaligned):
            issues.append("ff_evidence: cifar aligned/unaligned do not share a group count")
        elif 3072 not in {span[0] for span in aligned[1:]}:
            issues.append("ff_evidence: cifar aligned_g11 has no boundary at the label boundary")
        elif 3072 in {span[0] for span in unaligned[1:]}:
            issues.append("ff_evidence: cifar unaligned_g11 has a boundary at the label boundary")
        for name in ("aligned_g11", "unaligned_g11"):
            if not cifar[name].get("quantisers"):
                issues.append(f"ff_evidence: cifar.{name} does not name its quantiser")
    two_layers = (payload.get("group_two_layers") or {}).get("runs") or {}
    if not two_layers:
        issues.append("ff_evidence: missing group_two_layers.runs")
    for run_name, run in two_layers.items():
        if run.get("layer1_input_dim") != 500:
            issues.append(f"ff_evidence: group_two_layers.{run_name} does not record a 500-wide layer 1")
        if run.get("status") != "complete" and not run.get("variants"):
            issues.append(f"ff_evidence: group_two_layers.{run_name} is unfinished and empty")
    headline = payload.get("headline_pp") or {}
    for key in (
        "fp32_to_ternary",
        "blockwise_recovery",
        "blockwise_share_of_gap",
        "cifar_aligned_share_of_gap",
    ):
        if key not in headline:
            issues.append(f"ff_evidence: missing headline_pp.{key}")
    return issues


def qwen35_forward_issues(payload: dict) -> list[str]:
    """Check the Qwen3.5-0.8B forward measurement before it is published."""
    issues: list[str] = []
    if payload.get("schema") != "quantscope-qwen35-forward-v1":
        issues.append("qwen35: unexpected schema")
    files = payload.get("weight_files") or []
    if len(files) != 2:
        issues.append(f"qwen35: expected 2 weight files, found {len(files)}")
    total = 0
    for entry in files:
        name = entry.get("name") or "?"
        sha = entry.get("sha256") or ""
        size = entry.get("bytes")
        if not isinstance(size, int) or size <= 0:
            issues.append(f"qwen35: {name} has no byte size")
        else:
            total += size
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            issues.append(f"qwen35: {name} has no 64-hex sha256")
    if total and payload.get("weight_bytes_total") != total:
        issues.append("qwen35: weight_bytes_total does not match the file sizes")
    load = payload.get("load") or {}
    if not isinstance(load.get("seconds"), (int, float)) or load.get("seconds", 0) <= 0:
        issues.append("qwen35: load.seconds missing")
    forward = payload.get("forward") or {}
    if forward.get("logits_finite") is not True:
        issues.append("qwen35: logits are not reported finite")
    shape = forward.get("logits_shape") or []
    if len(shape) != 3 or shape[1] != forward.get("prompt_tokens"):
        issues.append("qwen35: logits_shape does not match prompt_tokens")
    seconds = forward.get("seconds") or []
    if len(seconds) != 3 or sorted(seconds)[1] != forward.get("seconds_median"):
        issues.append("qwen35: forward.seconds_median is not the middle measurement")
    peak = payload.get("peak_working_set_gib") or {}
    before = peak.get("before_load")
    after_load = peak.get("after_load")
    after_forward = peak.get("after_forward")
    if None in (before, after_load, after_forward):
        issues.append("qwen35: peak memory triple is incomplete")
    elif not (before < after_load <= after_forward):
        issues.append("qwen35: peak memory does not grow with load and forward")
    params = payload.get("parameters") or {}
    text = params.get("bytes_text")
    visual = params.get("bytes_visual")
    total_bytes = params.get("bytes_total")
    if None in (text, visual, total_bytes):
        issues.append("qwen35: parameter byte split is incomplete")
    elif total_bytes < text + visual:
        issues.append("qwen35: bytes_total is smaller than text plus visual")
    if not params.get("note"):
        issues.append("qwen35: the tied-embedding double count is not explained")
    if len(payload.get("limits") or []) < 3:
        issues.append("qwen35: fewer than three stated limits")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT), help="repo root to check")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    issues: list[str] = []

    for rel in REQUIRED:
        if not (root / rel).is_file():
            issues.append(f"missing required file: {rel}")

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in EXEMPT_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"{rel}: not UTF-8 text")
            continue
        scrubbed = strip_urls(text)
        for match in LOCAL_PATH_RE.finditer(scrubbed):
            line = scrubbed.count("\n", 0, match.start()) + 1
            issues.append(f"{rel}:{line}: local absolute path")
        for match in PRIVATE_RE.finditer(scrubbed):
            line = scrubbed.count("\n", 0, match.start()) + 1
            issues.append(f"{rel}:{line}: lab-only reference ({match.group(0)})")
        for match in SECRET_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            issues.append(f"{rel}:{line}: possible secret")

    readme = root / "README.md"
    if readme.is_file():
        head = readme.read_text(encoding="utf-8")[:README_MIN_CHARS]
        for needle in README_MUST_CONTAIN:
            if needle not in head:
                issues.append(f"README.md: first {README_MIN_CHARS} chars lack {needle!r}")

    evidence = root / "docs" / "evidence" / "ff_controlled_evidence.json"
    if evidence.is_file():
        try:
            payload = json.loads(evidence.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"docs/evidence/ff_controlled_evidence.json: {exc}")
        else:
            summary = payload.get("summary") or {}
            for mode in ("fp32", "ternary", "first_fp32", "second_fp32"):
                if mode not in summary:
                    issues.append(f"evidence: missing summary.{mode}")

    bundle = root / "docs" / "evidence" / "ff_evidence.json"
    if bundle.is_file():
        try:
            payload = json.loads(bundle.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"docs/evidence/ff_evidence.json: {exc}")
        else:
            issues.extend(evidence_bundle_issues(payload))

    forward_bundle = root / "docs" / "evidence" / "qwen35_forward.json"
    if forward_bundle.is_file():
        try:
            payload = json.loads(forward_bundle.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"docs/evidence/qwen35_forward.json: {exc}")
        else:
            issues.extend(qwen35_forward_issues(payload))

    if issues:
        for line in issues:
            print(f"ERROR {line}", file=sys.stderr)
        print(f"\n{len(issues)} issue(s)", file=sys.stderr)
        return 1
    print("public check: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
