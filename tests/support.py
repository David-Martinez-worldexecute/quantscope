"""Shared helpers for the test suite.

Fixtures are created under ``tests/.tmp`` inside the repository and removed in
``tearDown``.

Two platform facts shaped this file:

* the system temp directory is not writable under the local sandbox, so a
  fixture built there fails before any assertion runs;
* ``tempfile.mkdtemp`` creates the leaf directory with mode ``0o700``, and a
  directory created that way rejects every ``mkdir`` inside it with
  ``PermissionError: [WinError 5]`` on this machine. A plain ``mkdir`` inherits
  the workspace ACL and works, so ``new_dir`` avoids ``mkdtemp`` entirely.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "sample_questions.jsonl"
TMP_ROOT = ROOT / "tests" / ".tmp"


def new_dir(prefix: str = "case-") -> Path:
    """Create a fresh, unique directory under ``tests/.tmp``."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"{prefix}{uuid.uuid4().hex[:12]}"
    path.mkdir()
    return path


@contextmanager
def temp_dir(prefix: str = "case-"):
    path = new_dir(prefix)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
