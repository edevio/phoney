from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from .dataset import DEFAULT_SEED, load_reviews
from .models import Review
from .paths import (
    DEFAULT_LIMIT,
    canonical_csv_path,
    is_canonical_run,
    output_csv_path,
    output_stem,
    sidecar_path,
)
from .prompt import load_or_resolve_prompt
from .providers.base import Provider
from .providers.fake import FakeProvider
from .providers.ollama import OllamaProvider
from .render import render_scores
from .runner import run as run_eval
from .scoring import load_results, score as compute_scores

app = typer.Typer(help="Fake review classifier.", no_args_is_help=True)
console = Console()

DEFAULT_DATASET = Path("data/fake-reviews-dataset.csv")
DEFAULT_PROMPT = Path("prompts/prompt.txt")
DEFAULT_GENERATIONS_DIR = Path("prompts/generations")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("output")


@app.callback()
def _root() -> None:
    """Keeps subcommands explicit even when only one exists."""


def get_provider(name: str, model: str) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "ollama":
        return OllamaProvider(model=model)
    raise typer.BadParameter(f"Unknown provider: {name}")


@app.command()
def classify(
    prompt: Path = typer.Option(DEFAULT_PROMPT, help="Prompt file."),
    hash_val: str = typer.Option(
        None,
        "--hash",
        help="Re-run with a historical prompt by hash from prompts/generations/.",
    ),
    dataset: Path = typer.Option(DEFAULT_DATASET, help="Path to the source CSV."),
    provider: str = typer.Option("ollama", help="Provider name (fake, ollama)."),
    model: str = typer.Option("qwen3:14b", help="Model identifier."),
    limit: int = typer.Option(DEFAULT_LIMIT, min=1, help="Number of rows to sample."),
    all_rows: bool = typer.Option(
        False,
        "--all",
        help="Run against every row in the dataset; overrides --limit.",
    ),
    snapshot: bool = typer.Option(
        False,
        "--snapshot",
        help="Force this run into output/ even when it would otherwise be canonical.",
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="Sampling seed."),
    save_prompt: bool = typer.Option(
        False,
        "--save-prompt",
        help="Also write every rendered prompt to a sidecar file in output/.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Also print misclassified rows after scoring."
    ),
) -> None:
    """Classify a sample of reviews and print the score."""
    try:
        instruction, prompt_digest, snapshotted = load_or_resolve_prompt(
            generations_dir=DEFAULT_GENERATIONS_DIR,
            fallback_prompt_path=prompt,
            hash_val=hash_val,
        )
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e))

    if snapshotted:
        console.print(
            f"[green]Snapshotted prompt to[/green] "
            f"{DEFAULT_GENERATIONS_DIR}/{prompt_digest}.txt"
        )

    canonical = is_canonical_run(limit, all_rows, snapshot)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stem = output_stem(model, prompt_digest, limit, all_rows, timestamp)

    if canonical:
        initial_path = canonical_csv_path(model, prompt_digest, all_rows, DEFAULT_RESULTS_DIR)
    else:
        initial_path = output_csv_path(stem, None, DEFAULT_OUTPUT_DIR)

    sample_limit = None if all_rows else limit
    reviews = load_reviews(dataset, limit=sample_limit, seed=seed)
    provider_impl = get_provider(provider, model)

    collected: list[tuple[Review, str]] = []
    on_prompt = (
        (lambda rendered, review: collected.append((review, rendered)))
        if save_prompt
        else None
    )

    written = run_eval(
        reviews,
        instruction=instruction,
        provider=provider_impl,
        output_path=initial_path,
        total=len(reviews),
        console=console,
        on_prompt=on_prompt,
    )

    scores = compute_scores(load_results(written))

    if not canonical:
        final_path = output_csv_path(stem, scores.accuracy, DEFAULT_OUTPUT_DIR)
        written.rename(final_path)
        written = final_path

    console.print(f"[green]Wrote {len(reviews)} rows to[/green] {written}")

    if save_prompt:
        sidecar = write_prompts_sidecar(
            sidecar_path(stem, DEFAULT_OUTPUT_DIR), collected, model, prompt_digest
        )
        console.print(f"[green]Saved prompts to[/green] {sidecar}")

    render_scores(scores, console, verbose=verbose)


@app.command()
def score(
    results: Path = typer.Argument(..., help="Path to a results CSV from `phoney run`."),
    verbose: bool = typer.Option(False, "--verbose", help="Also print misclassified rows."),
) -> None:
    """Print accuracy and classification metrics for a results CSV."""
    df = load_results(results)
    render_scores(compute_scores(df), console, verbose=verbose)


def write_prompts_sidecar(
    sidecar: Path,
    entries: list[tuple[Review, str]],
    model: str,
    prompt_digest: str,
) -> Path:
    """Write every rendered prompt to one file, one entry per row."""
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as f:
        for i, (review, rendered) in enumerate(entries):
            if i > 0:
                f.write("\n")
            f.write(
                f"=== row_id={review.row_id} "
                f"category={review.category} "
                f"true_label={review.label} "
                f"model={model} "
                f"hash={prompt_digest} ===\n"
            )
            f.write(rendered)
            if not rendered.endswith("\n"):
                f.write("\n")
    return sidecar
