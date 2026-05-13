import csv
import random
from pathlib import Path

from .models import Review

DEFAULT_SEED = 42


def read_all(path: Path) -> list[Review]:
    """Read every row from the source CSV into Review objects."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            Review(
                row_id=i,
                category=row["category"],
                rating=float(row["rating"]),
                label=row["label"],
                text=row["text_"],
            )
            for i, row in enumerate(reader)
        ]


def stratified_sample(
    reviews: list[Review], limit: int, seed: int
) -> list[Review]:
    """Return up to `limit` reviews, split evenly across labels.

    Deterministic given the same seed.
    """
    if limit <= 0:
        return []

    rng = random.Random(seed)
    by_label: dict[str, list[Review]] = {}
    for r in reviews:
        by_label.setdefault(r.label, []).append(r)

    per_label = limit // len(by_label)
    remainder = limit % len(by_label)

    picked: list[Review] = []
    for i, label in enumerate(sorted(by_label)):
        n = per_label + (1 if i < remainder else 0)
        pool = by_label[label]
        picked.extend(rng.sample(pool, min(n, len(pool))))

    rng.shuffle(picked)
    return picked


def load_reviews(
    path: Path, limit: int | None = None, seed: int = DEFAULT_SEED
) -> list[Review]:
    """Load reviews from disk, optionally sampled and stratified by label."""
    reviews = read_all(path)
    if limit is None:
        return reviews
    return stratified_sample(reviews, limit, seed)
