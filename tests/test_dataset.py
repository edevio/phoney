from collections import Counter
from pathlib import Path

import pytest

from fake_review_classifier.dataset import (
    load_reviews,
    read_all,
    stratified_sample,
)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "reviews.csv"
    path.write_text(
        "category,rating,label,text_\n"
        "A,5.0,CG,first\n"
        "A,4.0,OR,second\n"
        "B,3.0,CG,third\n"
        "B,2.0,OR,fourth\n"
        "A,1.0,CG,fifth\n"
        "B,5.0,OR,sixth\n",
        encoding="utf-8",
    )
    return path


def test_read_all_returns_every_row(tiny_csv: Path) -> None:
    reviews = read_all(tiny_csv)
    assert len(reviews) == 6
    assert reviews[0].row_id == 0
    assert reviews[0].text == "first"
    assert reviews[-1].row_id == 5


def test_load_reviews_no_limit_returns_all(tiny_csv: Path) -> None:
    assert len(load_reviews(tiny_csv)) == 6


def test_load_reviews_respects_limit(tiny_csv: Path) -> None:
    assert len(load_reviews(tiny_csv, limit=4)) == 4


def test_stratified_sample_is_balanced(tiny_csv: Path) -> None:
    reviews = read_all(tiny_csv)
    sample = stratified_sample(reviews, limit=4, seed=42)
    counts = Counter(r.label for r in sample)
    assert counts["CG"] == 2
    assert counts["OR"] == 2


def test_seed_is_deterministic(tiny_csv: Path) -> None:
    reviews = read_all(tiny_csv)
    a = stratified_sample(reviews, limit=4, seed=42)
    b = stratified_sample(reviews, limit=4, seed=42)
    assert [r.row_id for r in a] == [r.row_id for r in b]


def test_different_seeds_can_differ(tiny_csv: Path) -> None:
    reviews = read_all(tiny_csv)
    a = stratified_sample(reviews, limit=4, seed=1)
    b = stratified_sample(reviews, limit=4, seed=2)
    assert sorted(r.row_id for r in a) != sorted(r.row_id for r in b) or a != b


def test_limit_zero_returns_empty(tiny_csv: Path) -> None:
    assert load_reviews(tiny_csv, limit=0) == []
