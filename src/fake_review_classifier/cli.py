from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .dataset import DEFAULT_SEED, load_reviews
from .models import Review
from .prompt import load_prompt, result_path
from .providers.base import Provider
from .providers.fake import FakeProvider
from .providers.ollama import OllamaProvider
from .runner import run as run_eval

app = typer.Typer(help="Fake review classifier.", no_args_is_help=True)
console = Console()

DEFAULT_DATASET = Path("data/fake-reviews-dataset.csv")
DEFAULT_PROMPT = Path("prompts/prompt.txt")
DEFAULT_RESULTS_DIR = Path("results")
PREVIEW_TEXT_CHARS = 120


@app.callback()
def _root() -> None:
    """Keeps subcommands explicit even when only one exists."""


def build_preview_table(reviews: list[Review]) -> Table:
    table = Table(title=f"Reviews ({len(reviews)})", show_lines=False)
    table.add_column("row_id", justify="right", style="cyan")
    table.add_column("category", style="magenta")
    table.add_column("rating", justify="right")
    table.add_column("label", style="green")
    table.add_column("text")

    for r in reviews:
        text = (
            r.text
            if len(r.text) <= PREVIEW_TEXT_CHARS
            else r.text[: PREVIEW_TEXT_CHARS - 1] + "…"
        )
        table.add_row(str(r.row_id), r.category, f"{r.rating:g}", r.label, text)
    return table


def get_provider(name: str, model: str) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "ollama":
        return OllamaProvider(model=model)
    raise typer.BadParameter(f"Unknown provider: {name}")


@app.command()
def preview(
    dataset: Path = typer.Option(DEFAULT_DATASET, help="Path to the source CSV."),
    limit: int = typer.Option(10, min=1, help="Number of rows to sample."),
    seed: int = typer.Option(DEFAULT_SEED, help="Sampling seed."),
) -> None:
    """Sample rows from the dataset and print them as a Rich table."""
    reviews = load_reviews(dataset, limit=limit, seed=seed)
    console.print(build_preview_table(reviews))


@app.command()
def run(
    prompt: Path = typer.Option(DEFAULT_PROMPT, help="Prompt file."),
    dataset: Path = typer.Option(DEFAULT_DATASET, help="Path to the source CSV."),
    provider: str = typer.Option("ollama", help="Provider name (fake, ollama)."),
    model: str = typer.Option("qwen3:14b", help="Model identifier."),
    limit: int = typer.Option(200, min=1, help="Number of rows to sample."),
    seed: int = typer.Option(DEFAULT_SEED, help="Sampling seed."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, help="Results directory."),
    save_prompt: bool = typer.Option(
        False, "--save-prompt", help="Also write the prompt next to the results CSV."
    ),
) -> None:
    """Classify a sample of reviews using the chosen provider."""
    instruction, prompt_digest = load_prompt(prompt)
    output_path = result_path(model, prompt_digest, results_dir)

    reviews = load_reviews(dataset, limit=limit, seed=seed)
    provider_impl = get_provider(provider, model)

    written = run_eval(
        reviews,
        instruction=instruction,
        provider=provider_impl,
        output_path=output_path,
        total=len(reviews),
        console=console,
    )
    console.print(f"[green]Wrote {len(reviews)} rows to[/green] {written}")

    if save_prompt:
        sidecar = written.with_suffix(".prompt.txt")
        sidecar.write_text(instruction, encoding="utf-8")
        console.print(f"[green]Saved prompt to[/green] {sidecar}")
