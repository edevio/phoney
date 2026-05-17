from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .scoring import Scores


def _color_pct(value: float) -> str:
    if value >= 0.9:
        colour = "green"
    elif value >= 0.7:
        colour = "yellow"
    else:
        colour = "red"
    return f"[{colour}]{value * 100:.1f}%[/]"


def render_scores(scores: Scores, console: Console, *, verbose: bool = False) -> None:
    _render_headline(scores, console)
    _render_confusion(scores, console)
    _render_classification_report(scores, console)
    _render_per_category(scores, console)
    if verbose:
        _render_misclassified(scores, console)
    if scores.unparseable:
        console.print(
            f"\n[dim]Note: {scores.unparseable} of {scores.total} rows had "
            f"unparseable responses and were not scored.[/]"
        )


def _render_headline(scores: Scores, console: Console) -> None:
    table = Table(
        title="Headline",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Accuracy", _color_pct(scores.accuracy))
    table.add_row("Correct", f"{scores.correct} / {scores.answered}")
    console.print(table)


def _render_confusion(scores: Scores, console: Console) -> None:
    table = Table(
        title="Confusion matrix",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
    )
    table.add_column("truth ╲ pred")
    for cls in scores.classes:
        table.add_column(cls, justify="right")
    for i, cls in enumerate(scores.classes):
        row = [cls]
        for j in range(len(scores.classes)):
            value = scores.confusion_matrix[i][j]
            if i == j:
                cell = f"[green]{value}[/]"
            elif value:
                cell = f"[red]{value}[/]"
            else:
                cell = "0"
            row.append(cell)
        table.add_row(*row)
    console.print(table)


def _render_classification_report(scores: Scores, console: Console) -> None:
    console.print(
        Panel(
            scores.classification_report_text,
            title="Classification report",
            border_style="cyan",
            expand=False,
        )
    )


def _render_per_category(scores: Scores, console: Console) -> None:
    table = Table(
        title="Per-category accuracy",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
    )
    table.add_column("Category")
    table.add_column("Accuracy", justify="right")
    table.add_column("Count", justify="right")
    for category, row in scores.per_category.iterrows():
        table.add_row(str(category), _color_pct(row["accuracy"]), str(int(row["count"])))
    console.print(table)


def _render_misclassified(scores: Scores, console: Console) -> None:
    if scores.misclassified.empty:
        console.print("[green]No misclassified rows.[/]")
        return
    table = Table(
        title=f"Misclassified ({len(scores.misclassified)} rows)",
        title_style="bold red",
        box=box.SIMPLE_HEAVY,
        border_style="red",
    )
    table.add_column("row_id", justify="right")
    table.add_column("category")
    table.add_column("true")
    table.add_column("pred")
    table.add_column("reasoning")
    for _, row in scores.misclassified.iterrows():
        reasoning = str(row["reasoning"])
        if len(reasoning) > 120:
            reasoning = reasoning[:119] + "…"
        table.add_row(
            str(row["row_id"]),
            str(row["category"]),
            str(row["true_label"]),
            str(row["model_label"]),
            reasoning,
        )
    console.print(table)
