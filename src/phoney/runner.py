import csv
from pathlib import Path
from typing import Callable, Iterable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .models import Review
from .parsing import parse_response
from .providers.base import Provider

RESULTS_FIELDS = [
    "row_id",
    "category",
    "true_label",
    "model_label",
    "reasoning",
    "model_raw",
    "latency_ms",
]

OUTPUT_CONTRACT = (
    "Respond with CG or OR on the first line, "
    "then one or two sentences of reasoning on the next line. "
    "Output nothing else."
)


def render_prompt(instruction: str, review: Review) -> str:
    """Combine user instruction, the row data block, and the output contract."""
    data_block = (
        "<review>\n"
        f"  <category>{review.category}</category>\n"
        f"  <rating>{review.rating}</rating>\n"
        f"  <text>{review.text}</text>\n"
        "</review>"
    )
    return f"{instruction.rstrip()}\n\n{data_block}\n\n{OUTPUT_CONTRACT}\n"


def run(
    reviews: Iterable[Review],
    *,
    instruction: str,
    provider: Provider,
    output_path: Path,
    total: int | None = None,
    console: Console | None = None,
    on_prompt: Callable[[str, Review], None] | None = None,
) -> Path:
    """Iterate reviews, classify with the provider, write results CSV.

    If `on_prompt` is given, it is called with the rendered prompt and the
    row after the prompt is built and before the provider is called.
    """
    console = console or Console()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("classifying", total=total)
            for review in reviews:
                prompt = render_prompt(instruction, review)
                if on_prompt is not None:
                    on_prompt(prompt, review)
                response = provider.classify(prompt)
                label, reasoning = parse_response(response)
                writer.writerow(
                    {
                        "row_id": review.row_id,
                        "category": review.category,
                        "true_label": review.label,
                        "model_label": label,
                        "reasoning": reasoning,
                        "model_raw": response.text,
                        "latency_ms": response.latency_ms,
                    }
                )
                progress.advance(task)

    return output_path
