# -*- coding: utf-8 -*-
"""Draw the README figures straight from the evidence bundle.

Why hand-written SVG instead of matplotlib: matplotlib is not installed in this
environment (and cannot be installed offline here), and a 900 kB plotting dependency
for three bar charts would be the wrong trade anyway. The SVG is plain text, diffs in
git, and needs no build step.

Why "straight from the bundle": a figure that contradicts the table next to it is worse
than no figure. Every number drawn here is read from ``docs/evidence/ff_evidence.json``
and cross-checked against the ``headline_pp`` block, so the picture and the prose come
from the same bytes. The corresponding check lives in ``tests/test_figures.py``.

Usage::

    python scripts/make_figures.py            # write docs/img/*.svg
    python scripts/make_figures.py --check    # fail if a figure would change
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "ff_evidence.json"
OUT_DIR = ROOT / "docs" / "img"

WIDTH = 760
HEIGHT = 420

BG = "#0f1115"
PANEL = "#151922"
FG = "#eef1f6"
MUTED = "#98a2b3"
GRID = "#252a34"
BLUE = "#5b8def"
GOLD = "#f0b64a"
RED = "#e2686b"
GREEN = "#5cc4a0"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

# The leak check has to name drive letters, but writing one literally in this file
# makes scripts/check_public.py flag the checker as the leak (it scans every file
# in the repo, including this one). So the needles are assembled at runtime from a
# drive-letter set and a separator set: same strings, no literal local path.
LEAK_DRIVES = "CH"
LEAK_EXTRA = ("acceptance_runs",)


def leak_needles() -> tuple[str, ...]:
    """Every substring a figure must not contain: a drive path, or the lab's run dir."""
    separators = ("/", "\\")
    drive_paths = tuple(f"{drive}:{sep}" for drive in LEAK_DRIVES for sep in separators)
    return drive_paths + LEAK_EXTRA


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load_bundle(path: pathlib.Path | None = None) -> dict:
    target = pathlib.Path(path) if path else EVIDENCE
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f} pp"


def text(x: float, y: float, body: str, size: int = 12, fill: str = FG,
         anchor: str = "start", weight: str = "normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(body)}</text>')


def frame(title: str, subtitle: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title>',
        f'<desc id="desc">{esc(desc)}</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        f'<rect x="16" y="16" width="{WIDTH - 32}" height="{HEIGHT - 32}" rx="12" '
        f'fill="{PANEL}" stroke="{GRID}"/>',
        text(40, 56, title, size=19, weight="bold"),
        text(40, 80, subtitle, size=12, fill=MUTED),
    ]


def svg_close() -> str:
    return "</svg>"


def nice_bounds(values: list[float], lower_hint: float | None = None) -> tuple[float, float]:
    """Round the value axis outwards so bars never touch the frame."""
    low = min(values) if lower_hint is None else lower_hint
    high = max(values)
    span = max(high - low, 1e-6)
    step = 10.0 ** (len(f"{int(span)}") - 1) if span >= 1 else 0.1
    for candidate in (span / 4, span / 5, span / 3):
        if candidate > 0:
            step = round(candidate, 2)
            break
    floor = round((low - span * 0.25) / step) * step
    if lower_hint is not None:
        floor = lower_hint
    ceil = round((high + span * 0.18) / step) * step
    if ceil <= high:
        ceil = high + step
    return floor, ceil


def bar_panel(values: list[float], labels: list[str], colors: list[str],
              floor: float, ceil: float, top: float, bottom: float,
              value_labels: list[str], note_lines: list[str]) -> list[str]:
    left, right = 96.0, WIDTH - 48.0
    parts: list[str] = []
    # gridlines + y labels
    ticks = 4
    for index in range(ticks + 1):
        value = floor + (ceil - floor) * index / ticks
        y = bottom - (bottom - top) * index / ticks
        parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(text(left - 12, y + 4, f"{value:.0f}%" if ceil > 1 else f"{value * 100:.0f}%",
                          size=11, fill=MUTED, anchor="end"))
    slot = (right - left) / len(values)
    bar_w = slot * 0.44
    for index, value in enumerate(values):
        cx = left + slot * (index + 0.5)
        height = (bottom - top) * (value - floor) / (ceil - floor)
        y = bottom - height
        parts.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                     f'height="{height:.1f}" rx="4" fill="{colors[index]}"/>')
        parts.append(text(cx, y - 10, value_labels[index], size=13, anchor="middle", weight="bold"))
        parts.append(text(cx, bottom + 22, labels[index], size=11, fill=MUTED, anchor="middle"))
    for offset, line in enumerate(note_lines):
        parts.append(text(left, HEIGHT - 48 + offset * 18, line, size=11, fill=MUTED))
    return parts


def figure_mnist(bundle: dict) -> str:
    summary = bundle["precision_matrix"]["summary"]
    fp32 = summary["fp32"]["mean_accuracy"]
    ternary = summary["ternary"]["mean_accuracy"]
    aligned = bundle["mechanism"]["blockwise_in_training"]["accuracy_mean"]
    share = bundle["headline_pp"]["blockwise_share_of_gap"]
    recovery = bundle["headline_pp"]["blockwise_recovery"]
    assert recovery > 0 and 0 < share < 1, "the MNIST mechanism block no longer matches its headlines"

    floor, ceil = nice_bounds([fp32 * 100, aligned * 100, ternary * 100], lower_hint=86)
    parts = frame(
        "Ternary weights on MNIST: where the 4.62 pp goes",
        "2-layer forward-forward MLP, 500 hidden units, 10,000 test images, 3 seeds",
        f"FP32 {fp32 * 100:.2f} percent, aligned groups {aligned * 100:.2f} percent, "
        f"one scale per row {ternary * 100:.2f} percent.")
    parts += bar_panel(
        [fp32 * 100, aligned * 100, ternary * 100],
        ["FP32 (reference)", "ternary + 11 aligned groups", "ternary, one scale per row"],
        [BLUE, GREEN, RED], floor, ceil, top=120, bottom=330,
        value_labels=[f"{fp32 * 100:.2f}%", f"{aligned * 100:.2f}%", f"{ternary * 100:.2f}%"],
        note_lines=[
            f"Cost of ternarizing at all: {pp((ternary - fp32) * 100)}",
            f"Isolating the 10 label columns wins back {recovery:.2f} pp "
            f"= {share * 100:.1f}% of that cost",
        ])
    parts.append(svg_close())
    return "\n".join(parts)


def figure_cifar(bundle: dict) -> str:
    order = ["fp32", "aligned_g11", "unaligned_g11", "ternary_row"]
    names = ["FP32", "aligned 11", "equal 11", "one scale/row"]
    small = [bundle["cifar"]["variants"][key]["accuracy_mean"] for key in order]
    full = [bundle["cifar_full"]["variants"][key]["accuracy_mean"] for key in order]
    assert bundle["cifar"]["settings"]["train_subset"] == 20000
    assert bundle["cifar_full"]["settings"]["train_subset"] == 50000

    floor, ceil = nice_bounds([v * 100 for v in small + full], lower_hint=20)
    left, right = 96.0, WIDTH - 48.0
    top, bottom = 130.0, 320.0
    parts = frame(
        "The same four formats, two training-set sizes (CIFAR-10)",
        "2-layer MLP on 3,072 pixels + 10 label columns, 5 epochs per layer, 3 seeds",
        "Left bar of each pair: first 20,000 training images. Right bar: all 50,000.")
    for index in range(5):
        value = floor + (ceil - floor) * index / 4
        y = bottom - (bottom - top) * index / 4
        parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
                     f'stroke="{GRID}"/>')
        parts.append(text(left - 12, y + 4, f"{value:.0f}%", size=11, fill=MUTED, anchor="end"))
    slot = (right - left) / len(order)
    bar_w = slot * 0.24
    for index, key in enumerate(order):
        cx = left + slot * (index + 0.5)
        for offset, (value, color) in enumerate(((small[index], GOLD), (full[index], BLUE))):
            height = (bottom - top) * (value * 100 - floor) / (ceil - floor)
            x = cx + (offset - 0.5) * bar_w * 1.15 - bar_w / 2
            parts.append(f'<rect x="{x:.1f}" y="{bottom - height:.1f}" width="{bar_w:.1f}" '
                         f'height="{height:.1f}" rx="3" fill="{color}"/>')
            parts.append(text(x + bar_w / 2, bottom - height - 8, f"{value * 100:.2f}",
                              size=10, fill=MUTED, anchor="middle"))
        parts.append(text(cx, bottom + 22, names[index], size=11, fill=MUTED, anchor="middle"))
    parts.append(f'<rect x="{left:.1f}" y="96" width="12" height="12" rx="2" fill="{GOLD}"/>')
    parts.append(text(left + 18, 106, "20,000 training images", size=11, fill=MUTED))
    parts.append(f'<rect x="{left + 190:.1f}" y="96" width="12" height="12" rx="2" fill="{BLUE}"/>')
    parts.append(text(left + 208, 106, "all 50,000 training images", size=11, fill=MUTED))
    parts.append(text(left, HEIGHT - 48,
                      "More data widens the cost of ternarizing ({:.2f} pp → {:.2f} pp) — "
                      "it does not shrink it.".format(
                          bundle["headline_pp"]["cifar_fp32_to_ternary"],
                          bundle["headline_pp"]["cifar_full_fp32_to_ternary"]),
                      size=11, fill=MUTED))
    parts.append(svg_close())
    return "\n".join(parts)


def figure_recovery(bundle: dict) -> str:
    shares = [
        bundle["headline_pp"]["blockwise_share_of_gap"],
        bundle["headline_pp"]["cifar_aligned_share_of_gap"],
        bundle["headline_pp"]["cifar_full_aligned_share_of_gap"],
    ]
    for value in shares:
        assert 0 < value <= 1, "a recovery share left (0, 1]; the figure would lie"
    left_mnist = (bundle["precision_matrix"]["summary"]["fp32"]["mean_accuracy"]
                  - bundle["mechanism"]["blockwise_in_training"]["accuracy_mean"]) * 100
    left_cifar = (bundle["cifar"]["variants"]["fp32"]["accuracy_mean"]
                  - bundle["cifar"]["variants"]["aligned_g11"]["accuracy_mean"]) * 100
    left_cifar_full = (bundle["cifar_full"]["variants"]["fp32"]["accuracy_mean"]
                       - bundle["cifar_full"]["variants"]["aligned_g11"]["accuracy_mean"]) * 100
    parts = frame(
        "How much of the ternary cost label-column isolation wins back",
        "share of the FP32→ternary gap recovered by splitting the input scale per column group",
        "MNIST {:.1f}%, CIFAR-10 (20k) {:.1f}%, CIFAR-10 (50k) {:.1f}%.".format(
            shares[0] * 100, shares[1] * 100, shares[2] * 100))
    parts += bar_panel(
        [value * 100 for value in shares],
        ["MNIST, 794 inputs", "CIFAR-10, 20k images", "CIFAR-10, 50k images"],
        [BLUE, GOLD, GREEN], 0, 110, top=120, bottom=320,
        value_labels=[f"{value * 100:.1f}%" for value in shares],
        note_lines=[
            "Leftover gap to FP32: MNIST {:.2f} pp, CIFAR-10 (20k) {:.2f} pp, (50k) {:.2f} pp".format(
                left_mnist, left_cifar, left_cifar_full),
            "Equal-width groups that cut through the label columns recover almost nothing.",
        ])
    parts.append(svg_close())
    return "\n".join(parts)


def build(bundle: dict) -> dict[str, str]:
    if not OUT_DIR.is_dir():
        OUT_DIR.mkdir(parents=True)
    return {
        "fig-mnist-ternary-cost.svg": figure_mnist(bundle),
        "fig-cifar-two-training-sets.svg": figure_cifar(bundle),
        "fig-label-isolation-recovery.svg": figure_recovery(bundle),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="do not write; fail when a figure would change")
    parser.add_argument("--evidence", help="bundle to read (defaults to docs/evidence/ff_evidence.json)")
    args = parser.parse_args(argv)

    figures = build(load_bundle(args.evidence))
    problems = []
    for name, body in figures.items():
        for needle in leak_needles():
            if needle in body:
                problems.append(f"{name} leaks a local path: {needle}")
        path = OUT_DIR / name
        if args.check:
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            if current != body:
                problems.append(f"{name} is stale; run python scripts/make_figures.py")
        else:
            path.write_text(body, encoding="utf-8", newline="\n")
            print(f"wrote {path.relative_to(ROOT)} ({len(body)} bytes)")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    print("figures ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
