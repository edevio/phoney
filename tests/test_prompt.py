from pathlib import Path

from fake_review_classifier.prompt import (
    HASH_LENGTH,
    load_prompt,
    prompt_hash,
    result_path,
    sanitise_model,
)


def test_load_prompt_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "p.txt"
    path.write_text("hello", encoding="utf-8")
    assert load_prompt(path) == "hello"


def test_prompt_hash_is_stable() -> None:
    assert prompt_hash("hello") == prompt_hash("hello")


def test_prompt_hash_is_short() -> None:
    assert len(prompt_hash("hello")) == HASH_LENGTH


def test_prompt_hash_changes_with_content() -> None:
    assert prompt_hash("a") != prompt_hash("b")


def test_sanitise_model_replaces_slash_and_colon() -> None:
    assert sanitise_model("qwen3:14b") == "qwen3_14b"
    assert sanitise_model("org/model:tag") == "org_model_tag"


def test_result_path_combines_model_and_hash(tmp_path: Path) -> None:
    p = result_path("qwen3:14b", "hello", tmp_path)
    assert p.parent == tmp_path
    assert p.name.startswith("qwen3_14b_")
    assert p.suffix == ".csv"
    assert len(p.stem.split("_")[-1]) == HASH_LENGTH


def test_result_path_changes_with_prompt(tmp_path: Path) -> None:
    a = result_path("m", "prompt one", tmp_path)
    b = result_path("m", "prompt two", tmp_path)
    assert a != b
