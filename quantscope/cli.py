"""Command line entry point.

Exit codes are part of the contract: 0 success, 1 verification failure, 2 bad
input or protocol violation. A CI job or a reader can rely on them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, model as model_mod, protocol, reproduce as reproduce_mod
from .report import write_report
from .run import load_questions, run as run_pipeline, summary_table
from .verify import verify

DEFAULT_DATA = "data/sample_questions.jsonl"


def _demo(args) -> int:
    data = Path(args.data)
    questions = load_questions(data)
    prompts = [protocol.build_prompt(row["question"], row["options"]) for row in questions]
    weights, bias = model_mod.fit(
        prompts,
        [row["answer"] for row in questions],
        n_buckets=args.buckets,
        seed=args.seed,
        epochs=args.epochs,
    )
    out = Path(args.out)
    model_dir = model_mod.save_model(out / "model", weights, bias, args.buckets, args.seed)
    manifest, run_dir = run_pipeline(
        model_dir,
        data,
        len(questions),
        args.modes,
        args.group_size,
        out / "run",
        args.source_repo,
        args.source_revision,
    )
    print(summary_table(manifest))
    print()
    print("The demo model is fitted and evaluated on the same questions, so these numbers")
    print("exercise the pipeline; they are not a claim about model quality or about LLMs.")
    print(f"manifest: {run_dir / 'manifest.json'}")
    print(f"report:   {write_report(run_dir)}")
    return 0


def _run(args) -> int:
    manifest, run_dir = run_pipeline(
        args.model,
        args.data,
        args.n,
        args.modes,
        args.group_size,
        args.out,
        args.source_repo,
        args.source_revision,
    )
    print(summary_table(manifest))
    print()
    print(f"manifest: {run_dir / 'manifest.json'}")
    return 0


def _verify(args) -> int:
    result = verify(args.run, args.model)
    print(result["status"].upper())
    for key, value in result["checks"].items():
        print(f"  {key}: {value}")
    for issue in result["issues"]:
        print(f"  - {issue}")
    return 0 if result["status"] == "pass" else 1


def _report(args) -> int:
    print(write_report(args.run, args.out))
    return 0


def _reproduce(args) -> int:
    evidence = Path(args.evidence) if args.evidence else None
    report = reproduce_mod.reproduce(evidence)
    for check in report["checks"]:
        if not check["ok"]:
            print(f"MISMATCH {check['id']} stored={check['stored']!r} "
                  f"recomputed={check['recomputed']!r} tolerance={check['tolerance']!r}")
    counts = report["counts"]
    where = " + ".join(
        f"{item['name']} ({item['bytes']} bytes, sha256 {item['sha256'][:16]}.., "
        f"{item['checks']} numbers)"
        for item in report.get("files", [])
    ) or f"{report['evidence']} ({report['evidence_bytes']} bytes)"
    print(f"reproduce {report['status']}: {counts['checked']} published numbers recomputed from "
          f"{where}, {counts['failed']} mismatched")
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"REPORT {out}")
    reproduce_mod.raise_for_failures(report)
    return 0


def _add_source_arguments(parser) -> None:
    parser.add_argument("--source-repo", help="declared model repository, for example Qwen/Qwen3-0.6B")
    parser.add_argument("--source-revision", help="full 40-character lowercase hex commit id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantscope",
        description="Compare weight formats on the same questions and keep a verifiable receipt.",
    )
    parser.add_argument("--version", action="version", version=f"quantscope {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="fit a toy model, evaluate three formats, write a report")
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--data", default=DEFAULT_DATA)
    demo.add_argument("--modes", default="fp32,int4,ternary")
    demo.add_argument("--group-size", type=int, default=64)
    demo.add_argument("--buckets", type=int, default=model_mod.DEFAULT_BUCKETS)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--epochs", type=int, default=400)
    _add_source_arguments(demo)
    demo.set_defaults(func=_demo)

    run = sub.add_parser("run", help="evaluate a model directory and write a manifest")
    run.add_argument("--model", required=True)
    run.add_argument("--data", default=DEFAULT_DATA)
    run.add_argument("--n", type=int, default=20)
    run.add_argument("--modes", default="fp32,int4,ternary")
    run.add_argument("--group-size", type=int, default=64)
    run.add_argument("--out", default="runs/run")
    _add_source_arguments(run)
    run.set_defaults(func=_run)

    check = sub.add_parser("verify", help="recheck a finished run")
    check.add_argument("--run", required=True)
    check.add_argument("--model", help="model directory to hash; defaults to the recorded path")
    check.set_defaults(func=_verify)

    report = sub.add_parser("report", help="rebuild the self-contained HTML report")
    report.add_argument("--run", required=True)
    report.add_argument("--out")
    report.set_defaults(func=_report)

    repro = sub.add_parser(
        "reproduce",
        help="recompute every published headline number from the evidence bundle")
    repro.add_argument(
        "--evidence",
        help="bundle to recompute; defaults to docs/evidence/ff_evidence.json in this checkout")
    repro.add_argument("--out", help="write the full check-by-check report as JSON")
    repro.set_defaults(func=_reproduce)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (protocol.ProtocolError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
