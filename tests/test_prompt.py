from pathlib import Path

from phoney.prompt import (
    HASH_LENGTH,
    load_prompt,
    result_path,
    sanitise_model,
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
