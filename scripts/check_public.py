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

    if issues:
        for line in issues:
            print(f"ERROR {line}", file=sys.stderr)
        print(f"\n{len(issues)} issue(s)", file=sys.stderr)
        return 1
    print("public check: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
