"""CLI wiring: argument parsing and the bench / fit / serve / version commands."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kindling.cli import build_parser, main
from kindling.loaders import synthetic


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    split = synthetic.make_ratings(n_entities=120, n_items=80, ratings_per_entity=20, seed=0)
    df = pd.concat([split.train, split.test], ignore_index=True)
    path = tmp_path / "interactions.csv"
    df.to_csv(path, index=False)
    return path


def test_parser_exposes_all_subcommands() -> None:
    parser = build_parser()
    for cmd in ("bench", "fit", "serve", "version"):
        args = parser.parse_args([cmd] if cmd in ("version",) else _min_args(cmd))
        assert args.command == cmd


def _min_args(cmd: str) -> list[str]:
    return {
        "bench": ["bench", "--dataset", "x"],
        "fit": ["fit", "--data", "x", "--out", "y"],
        "serve": ["serve", "--model", "m"],
    }[cmd]


def test_version_prints_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from kindling import __version__

    rc = main(["version"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == __version__


def test_bench_on_csv_prints_table(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["bench", "--data", str(csv_path), "--max-users", "100", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NDCG@10" in out and "kindling" in out


def test_bench_writes_json_report(csv_path: Path, tmp_path: Path) -> None:
    out_json = tmp_path / "report.json"
    rc = main(
        ["bench", "--data", str(csv_path), "--max-users", "100", "--quiet", "--json", str(out_json)]
    )
    assert rc == 0
    assert out_json.exists()
    import json

    data = json.loads(out_json.read_text())
    assert data["models"][0] == "kindling"


def test_fit_saves_loadable_engine(csv_path: Path, tmp_path: Path) -> None:
    model = tmp_path / "engine.kindling"
    rc = main(["fit", "--data", str(csv_path), "--out", str(model)])
    assert rc == 0 and model.exists()
    from kindling import Engine

    eng = Engine.load(model)
    assert eng._state is not None


def test_serve_missing_model_returns_error(tmp_path: Path) -> None:
    rc = main(["serve", "--model", str(tmp_path / "nope.kindling")])
    assert rc == 2
