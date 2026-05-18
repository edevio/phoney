from pathlib import Path

import pytest

from phoney.prompt import (
    HASH_LENGTH,
    load_or_resolve_prompt,
    load_prompt,
    result_path,
    sanitise_model,
    save_prompt_generation,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_prompt_returns_text_and_hash(tmp_path: Path) -> None:
    path = _write(tmp_path / "p.txt", "hello")
    text, digest = load_prompt(path)
    assert text == "hello"
    assert len(digest) == HASH_LENGTH


def test_load_prompt_hash_is_stable(tmp_path: Path) -> None:
    path = _write(tmp_path / "p.txt", "hello")
    _, a = load_prompt(path)
    _, b = load_prompt(path)
    assert a == b


def test_load_prompt_hash_changes_with_content(tmp_path: Path) -> None:
    _, a = load_prompt(_write(tmp_path / "a.txt", "one"))
    _, b = load_prompt(_write(tmp_path / "b.txt", "two"))
    assert a != b


def test_sanitise_model_replaces_slash_and_colon() -> None:
    assert sanitise_model("qwen3:14b") == "qwen3_14b"
    assert sanitise_model("org/model:tag") == "org_model_tag"


def test_result_path_combines_model_and_hash(tmp_path: Path) -> None:
    p = result_path("qwen3:14b", "abcdef12", tmp_path)
    assert p == tmp_path / "qwen3_14b_abcdef12.csv"


def test_save_prompt_generation_writes_file(tmp_path: Path) -> None:
    generations = tmp_path / "generations"
    path = save_prompt_generation("hello", "abcdef12", generations)
    assert path == generations / "abcdef12.txt"
    assert path.read_text(encoding="utf-8") == "hello"


def test_save_prompt_generation_is_idempotent(tmp_path: Path) -> None:
    generations = tmp_path / "generations"
    save_prompt_generation("first", "abc", generations)
    save_prompt_generation("ignored", "abc", generations)
    assert (generations / "abc.txt").read_text(encoding="utf-8") == "first"


def test_load_or_resolve_snapshots_current_when_no_hash(tmp_path: Path) -> None:
    prompt = _write(tmp_path / "p.txt", "current")
    generations = tmp_path / "generations"

    text, digest, snapshotted = load_or_resolve_prompt(generations, prompt, None)

    assert text == "current"
    assert snapshotted is True
    assert (generations / f"{digest}.txt").read_text(encoding="utf-8") == "current"


def test_load_or_resolve_does_not_resnapshot(tmp_path: Path) -> None:
    prompt = _write(tmp_path / "p.txt", "current")
    generations = tmp_path / "generations"

    load_or_resolve_prompt(generations, prompt, None)
    _, _, snapshotted = load_or_resolve_prompt(generations, prompt, None)

    assert snapshotted is False


def test_load_or_resolve_returns_historical_prompt(tmp_path: Path) -> None:
    generations = tmp_path / "generations"
    generations.mkdir()
    (generations / "deadbeef.txt").write_text("historical", encoding="utf-8")

    text, digest, snapshotted = load_or_resolve_prompt(
        generations, tmp_path / "missing.txt", "deadbeef"
    )

    assert text == "historical"
    assert digest == "deadbeef"
    assert snapshotted is False


def test_load_or_resolve_raises_for_missing_hash(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_or_resolve_prompt(tmp_path / "generations", tmp_path / "p.txt", "nope")
