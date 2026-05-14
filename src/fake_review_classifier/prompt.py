import hashlib
from pathlib import Path

HASH_LENGTH = 8


def load_prompt(path: Path) -> str:
    """Read a prompt file from disk."""
    return path.read_text(encoding="utf-8")


def prompt_hash(text: str) -> str:
    """Short, stable hash of the prompt contents."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


def sanitise_model(name: str) -> str:
    """Make a model identifier safe for a filename."""
    return name.replace("/", "_").replace(":", "_")


def result_path(model: str, prompt_text: str, results_dir: Path) -> Path:
    """Build the CSV path for a model + prompt combination."""
    return results_dir / f"{sanitise_model(model)}_{prompt_hash(prompt_text)}.csv"
