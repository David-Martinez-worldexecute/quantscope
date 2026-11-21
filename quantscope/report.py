"""Rebuild a self-contained HTML report from a run manifest.

The output has no external stylesheet, script, font or image, so it opens the
same way offline and inside a network-restricted CI job.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
body { font-family: -apple-system, "Segoe UI", "Noto Sans", sans-serif; margin: 2rem auto;
       max-width: 60rem; line-height: 1.55; color: #1c1c1c; padding: 0 1rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
table { border-collapse: collapse; width: 100%; margin: 0.6rem 0 1rem; font-size: 0.92rem; }
th, td { border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }
th { background: #f5f5f5; font-weight: 600; }
code { background: #f2f2f2; padding: 0.05rem 0.3rem; border-radius: 3px; }
.muted { color: #666; font-size: 0.9rem; }
.note { background: #fff8e1; border-left: 3px solid #e0b400; padding: 0.6rem 0.8rem; margin: 1rem 0; }
"""


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _table(headers, rows) -> str:
    head = "".join(f"<th>{_escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html(manifest) -> str:
    modes = manifest["quant"]["modes"]
    baseline = manifest["quant"]["baseline"]
    summary_rows = []
    for mode in modes:
        entry = manifest["summary"][mode]
        pairs = entry.get("pairs") or {}
        summary_rows.append([
            mode,
            entry["correct"],
            entry["n"],
            f"{100.0 * entry['accuracy']:.2f}%",
            entry.get("vs", "-"),
            pairs.get("n01", "-"),
            pairs.get("n10", "-"),
            f"{entry['delta_pp']:+.2f}" if "delta_pp" in entry else "-",
        ])

    details = []
    for mode in modes:
        rows = [
            [item["id"], item["answer"], item["pred"], "yes" if item["correct"] else "no"]
            for item in manifest["items"][mode]
        ]
        details.append(f"<h2>Per-item predictions: {_escape(mode)}</h2>")
        details.append(_table(["id", "answer", "predicted", "correct"], rows))

    meta = [
        ["protocol id", manifest["protocol_id"]],
        ["cli version", manifest["cli_version"]],
        ["created (UTC)", manifest["created_utc"]],
        ["model kind", manifest["model"]["kind"]],
        ["model dir", manifest["model"]["dir"]],
        ["question file", f"{manifest['data']['path']} (sha256 {manifest['data']['sha256'][:16]}…)"],
        ["samples", manifest["data"]["n"]],
        ["baseline", baseline],
        ["group size", manifest["quant"]["group_size"]],
        ["python", manifest["env"]["python"]],
        ["numpy", manifest["env"]["numpy"]],
        ["device", manifest["env"]["device"]],
    ]
    source = manifest.get("source")
    if source:
        meta.append(["declared source", f"{source['repo']} @ {source['revision']}"])

    return "".join([
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>QuantScope report</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>QuantScope report</h1>",
        f'<p class="muted">Run manifest: <code>manifest.json</code>, {_escape(manifest["schema"])}</p>',
        '<div class="note">Accuracy is measured on the recorded questions only. '
        "This page reports what the run recorded; it does not claim a benchmark position.</div>",
        "<h2>Run</h2>",
        _table(["field", "value"], meta),
        "<h2>Accuracy by weight format</h2>",
        _table(["mode", "correct", "n", "accuracy", "vs", "n01", "n10", "delta_pp"], summary_rows),
        '<p class="muted">n01 = baseline wrong, comparison right. n10 = baseline right, comparison wrong. '
        "delta_pp is the accuracy difference in percentage points relative to the baseline.</p>",
        *details,
        "</body></html>",
    ])


def write_report(run_dir, out_path=None) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    target = Path(out_path) if out_path else run_dir / "report.html"
    target.write_text(build_html(manifest), encoding="utf-8", newline="\n")
    return target
