from pathlib import Path

from phoney.paths import result_path, sanitise_model


def test_sanitise_model_replaces_slash_and_colon() -> None:
    assert sanitise_model("qwen3:14b") == "qwen3_14b"
    assert sanitise_model("org/model:tag") == "org_model_tag"


def test_result_path_combines_model_and_hash(tmp_path: Path) -> None:
    p = result_path("qwen3:14b", "abcdef12", tmp_path)
    assert p == tmp_path / "qwen3_14b_abcdef12.csv"
