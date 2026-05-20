from pathlib import Path

DEFAULT_LIMIT = 100


def sanitise_model(name: str) -> str:
    """Make a model identifier safe for a filename."""
    return name.replace("/", "_").replace(":", "_")


def result_path(model: str, prompt_digest: str, results_dir: Path) -> Path:
    """Build the CSV path for a model + prompt combination."""
    return results_dir / f"{sanitise_model(model)}_{prompt_digest}.csv"


def is_canonical_run(
    limit: int, all_rows: bool, snapshot: bool, default_limit: int = DEFAULT_LIMIT
) -> bool:
    """A run is canonical when it represents one of the project's reference shapes."""
    if snapshot:
        return False
    if all_rows:
        return True
    return limit == default_limit


def canonical_csv_path(
    model: str, prompt_hash: str, all_rows: bool, results_dir: Path
) -> Path:
    """results/<model>_<hash>[_full].csv — overwritten on each canonical run."""
    suffix = "_full" if all_rows else ""
    return results_dir / f"{sanitise_model(model)}_{prompt_hash}{suffix}.csv"


def output_stem(
    model: str, prompt_hash: str, limit: int, all_rows: bool, timestamp: str
) -> str:
    """Shared filename prefix for any non-canonical run artefact."""
    base = f"run_{sanitise_model(model)}_{prompt_hash}_{timestamp}"
    scope = "full" if all_rows else f"limit_{limit}"
    return f"{base}_{scope}"


def output_csv_path(stem: str, accuracy: float | None, output_dir: Path) -> Path:
    """output/<stem>[_<acc>pct].csv — versioned, never overwrites."""
    if accuracy is None:
        return output_dir / f"{stem}.csv"
    return output_dir / f"{stem}_{round(accuracy * 100)}pct.csv"


def sidecar_path(stem: str, output_dir: Path) -> Path:
    """output/<stem>_prompts.txt — always lives in output, regardless of CSV destination."""
    return output_dir / f"{stem}_prompts.txt"
