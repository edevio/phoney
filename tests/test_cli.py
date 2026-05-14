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
