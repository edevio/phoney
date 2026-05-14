from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .dataset import DEFAULT_SEED, load_reviews
from .models import Review

app = typer.Typer(help="Fake review classifier.", no_args_is_help=True)
console = Console()


@app.callback()
def _root() -> None:
    """Keeps subcommands explicit even when only one exists."""

DEFAULT_DATASET = Path("data/fake-reviews-dataset.csv")
PREVIEW_TEXT_CHARS = 120


def build_preview_table(reviews: list[Review]) -> Table:
    table = Table(title=f"Reviews ({len(reviews)})", show_lines=False)
    table.add_column("row_id", justify="right", style="cyan")
    table.add_column("category", style="magenta")
    table.add_column("rating", justify="right")
    table.add_column("label", style="green")
    table.add_column("text")

    for r in reviews:
        text = r.text if len(r.text) <= PREVIEW_TEXT_CHARS else r.text[: PREVIEW_TEXT_CHARS - 1] + "…"
        table.add_row(str(r.row_id), r.category, f"{r.rating:g}", r.label, text)
    return table


@app.command()
def preview(
    dataset: Path = typer.Option(DEFAULT_DATASET, help="Path to the source CSV."),
    limit: int = typer.Option(10, min=1, help="Number of rows to sample."),
    seed: int = typer.Option(DEFAULT_SEED, help="Sampling seed."),
) -> None:
    """Sample rows from the dataset and print them as a Rich table."""
    reviews = load_reviews(dataset, limit=limit, seed=seed)
    console.print(build_preview_table(reviews))
