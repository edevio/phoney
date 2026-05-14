import hashlib
from pathlib import Path

HASH_LENGTH = 8


def load_prompt(path: Path) -> tuple[str, str]:
    """Read a prompt file, return its text and short content hash."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:HASH_LENGTH]
    return text, digest


def sanitise_model(name: str) -> str:
    """Make a model identifier safe for a filename."""
    return name.replace("/", "_").replace(":", "_")


def result_path(model: str, prompt_digest: str, results_dir: Path) -> Path:
    """Build the CSV path for a model + prompt combination."""
    return results_dir / f"{sanitise_model(model)}_{prompt_digest}.csv"
