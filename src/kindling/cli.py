"""The ``kindling`` command — a thin CLI over the engine, eval harness, and server.

Subcommands
-----------
* ``kindling bench``    fit + realistic-tier eval (by warmth) on a dataset or CSV
* ``kindling fit``      fit an engine on a CSV and save it for serving
* ``kindling serve``    serve a saved engine over HTTP (needs the ``serve`` extra)
* ``kindling version``  print the installed version

Run ``kindling <subcommand> --help`` for the per-command options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble engine constructor kwargs from the common flags + JSON escape hatch."""
    kw: dict[str, Any] = {}
    if args.ease_lambda is not None:
        kw["ease_lambda"] = args.ease_lambda
    if args.cold_slots is not None:
        kw["cold_slots"] = args.cold_slots
    if args.retrieval_budget is not None:
        kw["retrieval_budget"] = args.retrieval_budget
    if args.no_open_catalog:
        kw["open_catalog"] = False
    if args.engine_json:
        kw.update(json.loads(args.engine_json))
    return kw


def _add_engine_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("engine (most knobs auto-gate; override only if needed)")
    g.add_argument("--ease-lambda", type=float, default=None, help="EASE ridge strength.")
    g.add_argument("--cold-slots", type=int, default=None, help="Reserved cold-item slots.")
    g.add_argument("--retrieval-budget", type=int, default=None, help="Candidate pool size.")
    g.add_argument("--no-open-catalog", action="store_true", help="Disable metadata-only items.")
    g.add_argument("--engine-json", default=None, help="Extra kwargs as JSON, e.g. '{\"k\":5}'.")


def _cmd_bench(args: argparse.Namespace) -> int:
    from kindling.harness import evaluate, format_report
    from kindling.harness.baselines import available_baselines
    from kindling.harness.data import resolve_dataset

    source = args.data or args.dataset
    if not source:
        print("error: pass --dataset NAME or --data CSV", file=sys.stderr)
        return 2

    split = resolve_dataset(source, test_fraction=args.test_fraction, metadata=args.metadata)
    if args.all_baselines:
        baselines = available_baselines()
    else:
        baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    report = evaluate(
        split.train,
        split.test,
        split.items,
        dataset=split.name,
        k=args.k,
        engine_kwargs=_engine_kwargs(args),
        baselines=baselines,
        max_users=args.max_users,
        seed=args.seed,
        log=(lambda m: print(f"  {m}", file=sys.stderr)) if not args.quiet else None,
    )
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from kindling import Engine
    from kindling.harness.data import load_interactions_csv, read_csv_aliased

    interactions = load_interactions_csv(args.data)
    items = read_csv_aliased(args.metadata) if args.metadata else None
    engine = Engine(**_engine_kwargs(args))
    engine.fit(interactions, item_metadata=items)
    engine.save(args.out)
    plan = engine.activation_plan
    print(
        f"fitted {len(interactions):,} interactions  base={plan.base_scorer}  "
        f"channels={plan.active_channels}\nsaved → {args.out}",
        file=sys.stderr,
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from kindling.serving import serve
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not Path(args.model).exists():
        print(f"error: model file not found: {args.model}", file=sys.stderr)
        return 2
    print(f"serving {args.model} on http://{args.host}:{args.port}", file=sys.stderr)
    serve(args.model, host=args.host, port=args.port)
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from kindling import __version__

    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kindling", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="fit + realistic-tier eval by user warmth")
    src = b.add_argument_group("data source (one required)")
    src.add_argument("--dataset", help="built-in name, e.g. synthetic-grocery / movielens-1m")
    src.add_argument("--data", help="path to an interaction-log CSV (entity_id,item_id,...)")
    b.add_argument("--metadata", help="item metadata CSV (enables the cold-slot path).")
    b.add_argument("--test-fraction", type=float, default=0.1)
    b.add_argument("--k", type=int, default=10)
    b.add_argument(
        "--baselines",
        default="popularity",
        help="comma list: popularity,item-knn,als,bpr (trained ones need [baselines]).",
    )
    b.add_argument("--all-baselines", action="store_true", help="every available baseline.")
    b.add_argument("--max-users", type=int, default=2000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--json", help="also write the full report to this path.")
    b.add_argument("--quiet", action="store_true")
    _add_engine_flags(b)
    b.set_defaults(func=_cmd_bench)

    f = sub.add_parser("fit", help="fit an engine on a CSV and save it for serving")
    f.add_argument("--data", required=True, help="interaction-log CSV.")
    f.add_argument("--metadata", help="item metadata CSV.")
    f.add_argument("--out", required=True, help="output path for the saved engine.")
    _add_engine_flags(f)
    f.set_defaults(func=_cmd_fit)

    s = sub.add_parser("serve", help="serve a saved engine over HTTP (needs [serve])")
    s.add_argument("--model", required=True, help="path to an engine saved with .save().")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    v = sub.add_parser("version", help="print the installed kindling version")
    v.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Any = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
