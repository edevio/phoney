from pathlib import Path


def sanitise_model(name: str) -> str:
    """Make a model identifier safe for a filename."""
    return name.replace("/", "_").replace(":", "_")


def result_path(model: str, prompt_digest: str, results_dir: Path) -> Path:
    """Build the CSV path for a model + prompt combination."""
    return results_dir / f"{sanitise_model(model)}_{prompt_digest}.csv"
