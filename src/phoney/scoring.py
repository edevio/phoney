from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .parsing import UNPARSEABLE, VALID_LABELS

LABELS = sorted(VALID_LABELS)


@dataclass
class Scores:
    total: int
    answered: int
    correct: int
    unparseable: int
    accuracy: float
    classes: list[str]
    classification_report_text: str
    confusion_matrix: list[list[int]]
    per_category: pd.DataFrame
    misclassified: pd.DataFrame


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def score(results: pd.DataFrame) -> Scores:
    """Compute accuracy and classification metrics from a results frame.

    UNPARSEABLE rows are excluded from the accuracy denominator and the
    classification report; they are reported separately as parse failures.
    """
    total = len(results)
    unparseable = int((results["model_label"] == UNPARSEABLE).sum())

    answered_df = results[results["model_label"] != UNPARSEABLE]
    answered = len(answered_df)

    if answered:
        correct = int((answered_df["true_label"] == answered_df["model_label"]).sum())
        accuracy = correct / answered
        report_text = classification_report(
            answered_df["true_label"],
            answered_df["model_label"],
            labels=LABELS,
            zero_division=0,
        )
        cm = confusion_matrix(
            answered_df["true_label"],
            answered_df["model_label"],
            labels=LABELS,
        ).tolist()
    else:
        correct = 0
        accuracy = 0.0
        report_text = "No parseable predictions."
        cm = [[0] * len(LABELS) for _ in LABELS]

    if answered:
        per_category = (
            answered_df.assign(correct=lambda d: d["true_label"] == d["model_label"])
            .groupby("category")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy"})
            .sort_values("count", ascending=False)
        )
    else:
        per_category = pd.DataFrame(columns=["accuracy", "count"])

    misclassified = answered_df[
        answered_df["true_label"] != answered_df["model_label"]
    ][["row_id", "category", "true_label", "model_label", "reasoning"]]

    return Scores(
        total=total,
        answered=answered,
        correct=correct,
        unparseable=unparseable,
        accuracy=accuracy,
        classes=LABELS,
        classification_report_text=report_text,
        confusion_matrix=cm,
        per_category=per_category,
        misclassified=misclassified,
    )
