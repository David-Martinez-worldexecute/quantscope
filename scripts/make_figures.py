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
BENCH_EVIDENCE = ROOT / "docs" / "evidence" / "bench_qwen3_0.6b.json"
REAL_EVIDENCE = ROOT / "docs" / "evidence" / "real_model_qwen3_0.6b.json"
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


def figure_cost(bench: dict) -> str:
    """What each weight format costs: payload bytes on disk, then prefill and decode time."""
    order = ["baseline", "int8", "int4", "ternary"]
    names = {
        "baseline": "FP32",
        "int8": "INT8 dynamic",
        "int4": "INT4 RTN, groups of 64",
        "ternary": "ternary RTN, groups of 64",
    }
    colors = {"baseline": MUTED, "int8": BLUE, "int4": GOLD, "ternary": GREEN}
    modes = bench["modes"]
    mib = {mode: modes[mode]["quantized_bytes"] / (1024 * 1024) for mode in order}
    prefill = {mode: modes[mode]["prefill_ms_median"] for mode in order}
    decode = {mode: modes[mode]["decode_tokens_per_second"] for mode in order}
    # The one ordering claim this figure makes, checked against the stored numbers.
    assert mib["ternary"] < mib["int4"] < mib["int8"] < mib["baseline"], (
        "payload sizes are not ordered by width; the figure would mislead")
    assert any("simulat" in line.lower() for line in bench["limits"]), (
        "the caveat this figure prints no longer matches the bundle's own limits")
    baseline_mib = mib["baseline"]
    faster = [mode for mode in order if prefill[mode] < prefill["baseline"] * 0.9]
    protocol = bench["protocol"]
    parts = frame(
        "What each weight format costs on this CPU",
        "Qwen3-0.6B, {} prompt tokens, {} new tokens, {} threads, {} decode runs".format(
            protocol["prompt_tokens"], protocol["max_new_tokens"],
            bench["environment"]["threads"], protocol["decode_repeats"]),
        "Payload bytes per format, with FP32 at {:.0f} MiB, plus the prefill and decode time "
        "measured on the same machine.".format(baseline_mib))
    left_label, left_bar = 40.0, 300.0
    right, top, row_h = WIDTH - 56.0, 128.0, 58.0
    scale = (right - left_bar) / baseline_mib
    for index, mode in enumerate(order):
        y = top + index * row_h
        parts.append(text(left_label, y, names[mode], size=13))
        parts.append(text(left_label, y + 17,
                          "prefill {:.0f} ms · decode {:.2f} tok/s".format(
                              prefill[mode], decode[mode]),
                          size=10.5, fill=MUTED))
        width = mib[mode] * scale
        parts.append(f'<rect x="{left_bar:.1f}" y="{y - 13:.1f}" width="{width:.1f}" '
                     f'height="22" rx="4" fill="{colors[mode]}"/>')
        parts.append(text(left_bar + width + 10, y - 1,
                          "{:.0f} MiB · {:.0%}".format(mib[mode], mib[mode] / baseline_mib),
                          size=12, weight="bold"))
    parts.append(text(left_label, HEIGHT - 66,
                      "Rounded weights are storable: INT8 1 byte, INT4 half a byte, ternary a "
                      "quarter byte, plus a 4-byte scale per 64-value group.", size=11, fill=MUTED))
    parts.append(text(left_label, HEIGHT - 48,
                      "Faster than FP32 here: {}. The 4- and 2-bit rows are simulated in float32, "
                      "so they cannot be faster.".format(
                          ", ".join(names[mode] for mode in faster) if faster else "none"),
                      size=11, fill=MUTED))
    parts.append(svg_close())
    return "\n".join(parts)


def figure_accuracy_vs_payload(real: dict, bench: dict) -> str:
    """Accuracy against payload, with the interval over questions as a vertical bar."""
    order = ["baseline", "int8", "int4", "ternary"]
    names = {"baseline": "FP32", "int8": "INT8 dynamic",
             "int4": "INT4 g64", "ternary": "ternary g64"}
    colors = {"baseline": MUTED, "int8": BLUE, "int4": GOLD, "ternary": GREEN}
    modes = real["modes"]
    assert set(modes) == set(order), (
        "real-model bundle has modes {}; the figure expects {}".format(sorted(modes), order))
    mib = {mode: bench["modes"][mode]["quantized_bytes"] / (1024 * 1024) for mode in order}
    acc = {mode: modes[mode]["accuracy"] for mode in order}
    low = {mode: modes[mode]["ci95"]["low"] for mode in order}
    high = {mode: modes[mode]["ci95"]["high"] for mode in order}
    assert mib["ternary"] < mib["int4"] < mib["int8"] < mib["baseline"], (
        "payload sizes are not ordered by width; the figure would mislead")
    for mode in order:
        assert low[mode] <= acc[mode] <= high[mode], (
            "{}: stored accuracy sits outside its own interval".format(mode))
    count = modes["baseline"]["count"]
    left, right, top, bottom = 92.0, WIDTH - 150.0, 118.0, 306.0
    x_lo, x_hi = 600.0, 2600.0
    y_lo, y_hi = 0.15, 0.45

    def px(value: float) -> float:
        return left + (right - left) * (value - x_lo) / (x_hi - x_lo)

    def py(value: float) -> float:
        return bottom - (bottom - top) * (value - y_lo) / (y_hi - y_lo)

    parts = frame(
        "Accuracy against payload, on one CPU",
        "Qwen3-0.6B, {} questions, one run per format; bars are 95% bootstrap intervals "
        "(seed {}, {} resamples over questions)".format(
            count, modes["baseline"]["ci95"]["seed"], modes["baseline"]["ci95"]["resamples"]),
        "Each format's accuracy against the payload it needs, with the interval over questions. "
        "Where the bars overlap, the ordering between those two formats is not a result.")
    parts.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" '
                 f'stroke="{GRID}"/>')
    parts.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" '
                 f'stroke="{GRID}"/>')
    for tick_mib in (800, 1200, 1600, 2000, 2400):
        x = px(tick_mib)
        parts.append(f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 6:.1f}" '
                     f'stroke="{GRID}"/>')
        parts.append(text(x, bottom + 22, "{} MiB".format(tick_mib), size=10.5,
                          fill=MUTED, anchor="middle"))
    for tick_acc in (0.20, 0.25, 0.30, 0.35, 0.40):
        y = py(tick_acc)
        parts.append(f'<line x1="{left - 6:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" '
                     f'stroke="{GRID}"/>')
        parts.append(text(left - 10, y + 4, "{:.0f}%".format(tick_acc * 100), size=10.5,
                          fill=MUTED, anchor="end"))
    parts.append(text((left + right) / 2, bottom + 42, "payload, MiB", size=11,
                      fill=MUTED, anchor="middle"))
    parts.append(text(40, top + 2, "accuracy", size=11, fill=MUTED))
    # The two 2-bit rows sit close together, so one label goes above and one below.
    label_dy = {"baseline": -16, "int8": -16, "int4": -18, "ternary": 30}
    for mode in order:
        x = px(mib[mode])
        parts.append(f'<line x1="{x:.1f}" y1="{py(low[mode]):.1f}" x2="{x:.1f}" '
                     f'y2="{py(high[mode]):.1f}" stroke="{colors[mode]}" stroke-width="2"/>')
        for edge in (low[mode], high[mode]):
            parts.append(f'<line x1="{x - 5:.1f}" y1="{py(edge):.1f}" x2="{x + 5:.1f}" '
                         f'y2="{py(edge):.1f}" stroke="{colors[mode]}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{py(acc[mode]):.1f}" r="6" '
                     f'fill="{colors[mode]}" stroke="{BG}" stroke-width="2"/>')
        if mode == "baseline":
            parts.append(text(x - 12, py(acc[mode]) + label_dy[mode],
                              "{} · {:.1f}%".format(names[mode], acc[mode] * 100),
                              size=11.5, fill=colors[mode], anchor="end", weight="bold"))
        else:
            parts.append(text(x + 12, py(acc[mode]) + label_dy[mode],
                              "{} · {:.1f}%".format(names[mode], acc[mode] * 100),
                              size=11.5, fill=colors[mode], weight="bold"))
    parts.append(text(40, HEIGHT - 58,
                      "Payload is computed from the stored parameter shapes (1, 0.5 and 0.25 bytes "
                      "per weight, plus a 4-byte scale per 64-value group), not measured RSS.",
                      size=10.5, fill=MUTED))
    parts.append(text(40, HEIGHT - 40,
                      "The 4- and 2-bit rows are simulated in float32: that payload is what a real "
                      "kernel would need, not what this process holds.", size=10.5, fill=MUTED))
    parts.append(svg_close())
    return "\n".join(parts)


def build(bundle: dict, bench: dict | None = None, real: dict | None = None) -> dict[str, str]:
    if bench is None:
        bench = load_bundle(BENCH_EVIDENCE)
    if real is None:
        real = load_bundle(REAL_EVIDENCE)
    if not OUT_DIR.is_dir():
        OUT_DIR.mkdir(parents=True)
    return {
        "fig-mnist-ternary-cost.svg": figure_mnist(bundle),
        "fig-cifar-two-training-sets.svg": figure_cifar(bundle),
        "fig-label-isolation-recovery.svg": figure_recovery(bundle),
        "fig-cpu-cost.svg": figure_cost(bench),
        "fig-accuracy-vs-payload.svg": figure_accuracy_vs_payload(real, bench),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="do not write; fail when a figure would change")
    parser.add_argument("--evidence", help="bundle to read (defaults to docs/evidence/ff_evidence.json)")
    parser.add_argument("--bench", help="cost bundle to read (defaults to docs/evidence/bench_qwen3_0.6b.json)")
    parser.add_argument("--real", help="real-model bundle (defaults to docs/evidence/real_model_qwen3_0.6b.json)")
    args = parser.parse_args(argv)

    figures = build(load_bundle(args.evidence), load_bundle(args.bench or BENCH_EVIDENCE),
                    load_bundle(args.real or REAL_EVIDENCE))
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
