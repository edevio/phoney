from pathlib import Path

from typer.testing import CliRunner

from fake_review_classifier.cli import app

runner = CliRunner()


def _write_csv(path: Path) -> None:
    path.write_text(
        "category,rating,label,text_\n"
        "A,5.0,CG,first\n"
        "A,4.0,OR,second\n"
        "B,3.0,CG,third\n"
        "B,2.0,OR,fourth\n",
        encoding="utf-8",
    )


def test_preview_prints_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "reviews.csv"
    _write_csv(csv_path)

    result = runner.invoke(
        app, ["preview", "--dataset", str(csv_path), "--limit", "2"]
    )

    assert result.exit_code == 0, result.output
    assert "row_id" in result.output
    assert "category" in result.output
    assert "Reviews (2)" in result.output


def _setup_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    csv_path = tmp_path / "reviews.csv"
    _write_csv(csv_path)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("classify the review", encoding="utf-8")
    results_dir = tmp_path / "results"
    return csv_path, prompt_path, results_dir


def test_run_without_save_prompt_has_no_sidecar(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset", str(csv_path),
            "--prompt", str(prompt_path),
            "--provider", "fake",
            "--model", "fake",
            "--limit", "2",
            "--results-dir", str(results_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert not list(results_dir.glob("*_prompts.txt"))
    assert list(results_dir.glob("*.csv"))


def test_run_all_rows_overrides_limit(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset", str(csv_path),
            "--prompt", str(prompt_path),
            "--provider", "fake",
            "--model", "fake",
            "--limit", "1",
            "--all",
            "--results-dir", str(results_dir),
        ],
    )

    import csv as _csv

    assert result.exit_code == 0, result.output
    csvs = list(results_dir.glob("*.csv"))
    assert len(csvs) == 1
    with csvs[0].open(encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    # The fixture has 4 rows, so --all should yield 4 result rows.
    assert len(rows) == 4


def test_run_with_save_prompt_writes_sidecar(tmp_path: Path) -> None:
    csv_path, prompt_path, results_dir = _setup_run(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset", str(csv_path),
            "--prompt", str(prompt_path),
            "--provider", "fake",
            "--model", "fake",
            "--limit", "2",
            "--results-dir", str(results_dir),
            "--save-prompt",
        ],
    )

    assert result.exit_code == 0, result.output
    sidecars = list(results_dir.glob("*_prompts.txt"))
    assert len(sidecars) == 1

    content = sidecars[0].read_text(encoding="utf-8")
    # Two entries with delimiter headers
    assert content.count("=== row_id=") == 2
    # The instruction is present in every rendered prompt
    assert "classify the review" in content
    # The output contract appears in every rendered prompt
    assert "CG or OR on the first line" in content
    # Per-row data is interpolated
    assert "<text>first</text>" in content or "<text>second</text>" in content
