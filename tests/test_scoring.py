import pandas as pd

from phoney.scoring import score


def _df(rows: list[dict]) -> pd.DataFrame:
    base = {"row_id": 0, "category": "X", "reasoning": ""}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_perfect_run_is_100_percent() -> None:
    df = _df(
        [
            {"true_label": "CG", "model_label": "CG"},
            {"true_label": "OR", "model_label": "OR"},
        ]
    )
    s = score(df)
    assert s.total == 2
    assert s.answered == 2
    assert s.correct == 2
    assert s.unparseable == 0
    assert s.accuracy == 1.0


def test_unparseable_excluded_from_accuracy() -> None:
    df = _df(
        [
            {"true_label": "CG", "model_label": "CG"},
            {"true_label": "OR", "model_label": "OR"},
            {"true_label": "CG", "model_label": "UNPARSEABLE"},
        ]
    )
    s = score(df)
    assert s.total == 3
    assert s.answered == 2
    assert s.correct == 2
    assert s.unparseable == 1
    assert s.accuracy == 1.0


def test_misclassified_captured() -> None:
    df = _df(
        [
            {"row_id": 1, "true_label": "CG", "model_label": "OR"},
            {"row_id": 2, "true_label": "OR", "model_label": "OR"},
        ]
    )
    s = score(df)
    assert s.accuracy == 0.5
    assert list(s.misclassified["row_id"]) == [1]


def test_confusion_matrix_shape() -> None:
    df = _df(
        [
            {"true_label": "CG", "model_label": "CG"},
            {"true_label": "CG", "model_label": "OR"},
            {"true_label": "OR", "model_label": "OR"},
        ]
    )
    s = score(df)
    # classes sorted: [CG, OR]
    assert s.classes == ["CG", "OR"]
    assert s.confusion_matrix == [[1, 1], [0, 1]]


def test_per_category_groups() -> None:
    df = _df(
        [
            {"category": "A", "true_label": "CG", "model_label": "CG"},
            {"category": "A", "true_label": "OR", "model_label": "OR"},
            {"category": "B", "true_label": "CG", "model_label": "OR"},
        ]
    )
    s = score(df)
    assert int(s.per_category.loc["A", "count"]) == 2
    assert s.per_category.loc["A", "accuracy"] == 1.0
    assert int(s.per_category.loc["B", "count"]) == 1
    assert s.per_category.loc["B", "accuracy"] == 0.0


def test_all_unparseable_does_not_crash() -> None:
    df = _df(
        [
            {"true_label": "CG", "model_label": "UNPARSEABLE"},
            {"true_label": "OR", "model_label": "UNPARSEABLE"},
        ]
    )
    s = score(df)
    assert s.answered == 0
    assert s.accuracy == 0.0
    assert s.unparseable == 2
