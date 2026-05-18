import hashlib
from pathlib import Path

HASH_LENGTH = 8


def load_prompt(path: Path) -> tuple[str, str]:
    """Read a prompt file, return its text and short content hash."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:HASH_LENGTH]
    return text, digest


def save_prompt_generation(text: str, digest: str, generations_dir: Path) -> Path:
    """Snapshot a prompt to the generations archive. Idempotent."""
    generations_dir.mkdir(parents=True, exist_ok=True)
    path = generations_dir / f"{digest}.txt"
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path


def load_or_resolve_prompt(
    generations_dir: Path,
    fallback_prompt_path: Path,
    hash_val: str | None,
) -> tuple[str, str, bool]:
    """Resolve a prompt by historical hash, or load + snapshot the current one.

    Returns (text, digest, snapshotted) where `snapshotted` is True only when
    a new generation file was just written.
    """
    if hash_val:
        path = generations_dir / f"{hash_val}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"No prompt generation found for hash '{hash_val}' at {path}"
            )
        return path.read_text(encoding="utf-8"), hash_val, False

    text, digest = load_prompt(fallback_prompt_path)
    target = generations_dir / f"{digest}.txt"
    snapshotted = not target.exists()
    save_prompt_generation(text, digest, generations_dir)
    return text, digest, snapshotted


def sanitise_model(name: str) -> str:
    """Make a model identifier safe for a filename."""
    return name.replace("/", "_").replace(":", "_")


def result_path(model: str, prompt_digest: str, results_dir: Path) -> Path:
    """Build the CSV path for a model + prompt combination."""
    return results_dir / f"{sanitise_model(model)}_{prompt_digest}.csv"
