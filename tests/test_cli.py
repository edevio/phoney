from pathlib import Path

import pytest
from typer.testing import CliRunner

from phoney.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _chdir_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run every test from a fresh cwd so default output/ and prompts/generations/ land in tmp."""
    monkeypatch.chdir(tmp_path)


def _write_csv(path: Path) -> None:
    path.write_text(
        "category,rating,label,text_\n"
        "A,5.0,CG,first\n"
        "A,4.0,OR,second\n"
        "B,3.0,CG,third\n"
        "B,2.0,OR,fourth\n",
        encoding="utf-8",
    )


def _setup_run(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    csv_path = tmp_path / "reviews.csv"
    _write_csv(csv_path)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("classify the review", encoding="utf-8")
    # Tests chdir to tmp_path, so the implicit output/ and prompts/generations/
    # all land under tmp_path automatically. Return them for assertions only.
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "output"
    return csv_path, prompt_path, results_dir, output_dir


def _classify_args(
    csv_path: Path,
    prompt_path: Path,
    *extra: str,
) -> list[str]:
    return [
        "classify",
        "--dataset", str(csv_path),
        "--prompt", str(prompt_path),
        "--provider", "fake",
        "--model", "fake",
        *extra,
    ]


def test_default_limit_lands_in_results(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir, output_dir = _setup_run(tmp_path)

    result = runner.invoke(app, _classify_args(csv_path, prompt_path))

    assert result.exit_code == 0, result.output
    assert list(results_dir.glob("*.csv")), "canonical CSV should be in results/"
    assert not list(output_dir.glob("*.csv")), "nothing in output/ for canonical run"


def test_non_default_limit_lands_in_output(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir, output_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        _classify_args(csv_path, prompt_path, "--limit", "2"),
    )

    assert result.exit_code == 0, result.output
    assert not list(results_dir.glob("*.csv"))
    csvs = list(output_dir.glob("*.csv"))
    assert len(csvs) == 1
    assert "pct" in csvs[0].name, "non-canonical filename should include accuracy"


def test_snapshot_forces_output_even_at_default(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir, output_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        _classify_args(csv_path, prompt_path, "--snapshot"),
    )

    assert result.exit_code == 0, result.output
    assert not list(results_dir.glob("*.csv"))
    assert list(output_dir.glob("*.csv"))


def test_all_rows_lands_in_results_with_full_suffix(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir, output_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        _classify_args(csv_path, prompt_path, "--all"),
    )

    import csv as _csv

    assert result.exit_code == 0, result.output
    csvs = list(results_dir.glob("*_full.csv"))
    assert len(csvs) == 1
    with csvs[0].open(encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 4


def test_save_prompt_writes_sidecar_to_output(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir, output_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        _classify_args(
            csv_path, prompt_path,
            "--limit", "2",
            "--save-prompt",
        ),
    )

    assert result.exit_code == 0, result.output
    sidecars = list(output_dir.glob("*_prompts.txt"))
    assert len(sidecars) == 1

    content = sidecars[0].read_text(encoding="utf-8")
    assert content.count("=== row_id=") == 2
    assert "classify the review" in content
    assert "CG or OR on the first line" in content


def test_save_prompt_sidecar_in_output_even_for_canonical_run(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir, output_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        _classify_args(csv_path, prompt_path, "--all", "--save-prompt"),
    )

    assert result.exit_code == 0, result.output
    assert list(results_dir.glob("*_full.csv"))
    assert list(output_dir.glob("*_prompts.txt"))
    assert not list(results_dir.glob("*_prompts.txt"))
