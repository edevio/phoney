import csv
from pathlib import Path

from phoney.models import Review
from phoney.providers.fake import FakeProvider
from phoney.runner import RESULTS_FIELDS, render_prompt, run


def _reviews() -> list[Review]:
    return [
        Review(row_id=0, category="A", rating=5.0, label="CG", text="first review"),
        Review(row_id=1, category="B", rating=3.0, label="OR", text="second review"),
    ]


def test_render_prompt_contains_all_sections() -> None:
    review = _reviews()[0]
    rendered = render_prompt("INSTRUCTION", review)
    assert "INSTRUCTION" in rendered
    assert "<review>" in rendered
    assert "<category>A</category>" in rendered
    assert "<text>first review</text>" in rendered
    assert "CG or OR on the first line" in rendered


def test_run_writes_csv_with_expected_schema(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    run(
        _reviews(),
        instruction="anything",
        provider=FakeProvider(),
        output_path=out,
        total=2,
    )
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert list(rows[0].keys()) == RESULTS_FIELDS
    assert rows[0]["row_id"] == "0"
    assert rows[0]["true_label"] == "CG"
    assert rows[0]["model_label"] in {"CG", "OR"}
    assert rows[0]["model_raw"].strip() != ""
