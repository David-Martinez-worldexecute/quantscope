"""The prompt protocol: one template, one label set, one extraction rule.

Every manifest records ``PROTOCOL_ID``. Changing the template, the label set or
the extraction rule requires a new id, because runs with different ids are not
comparable and must never be averaged together.
"""

from __future__ import annotations

import re

import numpy as np

PROTOCOL_ID = "quantscope-choice-logits-v2"
LABELS = ("A", "B", "C", "D")
PROMPT_TAIL = "Answer:"

_SYSTEM = (
    "You are answering a multiple-choice question. "
    "Reply with one letter and nothing else."
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ProtocolError(ValueError):
    """Raised when data or run parameters violate the protocol."""


def _normalise_options(options) -> dict:
    """Accept a list of four texts or a mapping from label to text.

    Both shapes appear in practice: a JSONL file written by hand uses a list,
    while a converter that already knows the labels uses a mapping. Normalising
    here keeps one prompt template for every caller.
    """
    if isinstance(options, dict):
        if set(options) != set(LABELS):
            raise ProtocolError(f"options must have exactly the keys {LABELS}")
        items = {label: str(options[label]).strip() for label in LABELS}
    elif isinstance(options, (list, tuple)):
        if len(options) != len(LABELS):
            raise ProtocolError(f"options must hold exactly {len(LABELS)} entries")
        items = {label: str(text).strip() for label, text in zip(LABELS, options)}
    else:
        raise ProtocolError(
            "options must be a list of four texts or a mapping from label to text"
        )
    for label, text in items.items():
        if not text:
            raise ProtocolError(f"option {label} is empty")
    return items


def build_prompt(question: str, options) -> str:
    """Return the exact prompt used for every mode of a run.

    The template is fixed on purpose: no few-shot examples, no thinking mode,
    no per-mode variation. Otherwise a difference between modes could come from
    the prompt instead of the weights.
    """
    options = _normalise_options(options)
    lines = [_SYSTEM, "", "Question: " + str(question).strip(), ""]
    for label in LABELS:
        lines.append(f"{label}. {options[label]}")
    lines.append("")
    lines.append(PROMPT_TAIL)
    return "\n".join(lines)


def validate_question(record: dict) -> dict:
    """Check one JSONL record and return it unchanged."""
    if not isinstance(record, dict):
        raise ProtocolError("question record must be an object")
    for key in ("id", "question", "options", "answer"):
        if key not in record:
            raise ProtocolError(f"question record is missing {key!r}")
    if record["answer"] not in LABELS:
        raise ProtocolError(f"answer must be one of {LABELS}, got {record['answer']!r}")
    if not str(record["id"]).strip():
        raise ProtocolError("question id must not be empty")
    build_prompt(record["question"], record["options"])
    return record


def require_sample_count(items, requested: int) -> None:
    """Refuse to run when fewer samples exist than were requested.

    Silently shrinking the evaluation set is how two runs end up with different
    denominators while looking comparable.
    """
    if requested <= 0:
        raise ProtocolError("--n must be a positive integer")
    if len(items) < requested:
        raise ProtocolError(
            f"requested {requested} samples but only {len(items)} are available"
        )


def validate_source(repo, revision):
    """Validate an optional declared model source.

    A revision must be a full 40-character lowercase hex commit id. A branch
    name or a short hash is rejected: neither pins the bytes that were measured.
    """
    if repo is None and revision is None:
        return None
    if not repo or not revision:
        raise ProtocolError("--source-repo and --source-revision must be given together")
    if not _REVISION.match(str(revision)):
        raise ProtocolError("--source-revision must be 40 lowercase hex characters")
    return {"repo": str(repo), "revision": str(revision)}


def argmax_label(logits) -> str:
    """Map four choice logits to a label."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(LABELS),):
        raise ProtocolError(f"expected {len(LABELS)} logits, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ProtocolError("logits contain a non-finite value")
    return LABELS[int(np.argmax(values))]
