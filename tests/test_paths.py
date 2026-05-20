from pathlib import Path

import pytest

from phoney.paths import (
    canonical_csv_path,
    is_canonical_run,
    output_csv_path,
    output_stem,
    result_path,
    sanitise_model,
    sidecar_path,
)


def test_sanitise_model_replaces_slash_and_colon() -> None:
    assert sanitise_model("qwen3:14b") == "qwen3_14b"
    assert sanitise_model("org/model:tag") == "org_model_tag"


def test_result_path_combines_model_and_hash(tmp_path: Path) -> None:
    p = result_path("qwen3:14b", "abcdef12", tmp_path)
    assert p == tmp_path / "qwen3_14b_abcdef12.csv"


@pytest.mark.parametrize(
    "limit, all_rows, snapshot, expected",
    [
        (100, False, False, True),    # default sample, canonical
        (100, False, True, False),    # snapshot overrides
        (50, False, False, False),    # non-default limit
        (1000, False, False, False),  # non-default limit
        (100, True, False, True),     # --all is canonical
        (100, True, True, False),     # --all + --snapshot
    ],
)
def test_is_canonical_run(
    limit: int, all_rows: bool, snapshot: bool, expected: bool
) -> None:
    assert is_canonical_run(limit, all_rows, snapshot) is expected


def test_canonical_csv_path_default(tmp_path: Path) -> None:
    p = canonical_csv_path("qwen3:14b", "abcdef12", all_rows=False, results_dir=tmp_path)
    assert p == tmp_path / "qwen3_14b_abcdef12.csv"


def test_canonical_csv_path_full(tmp_path: Path) -> None:
    p = canonical_csv_path("qwen3:14b", "abcdef12", all_rows=True, results_dir=tmp_path)
    assert p == tmp_path / "qwen3_14b_abcdef12_full.csv"


def test_output_stem_limit() -> None:
    assert (
        output_stem("qwen3:14b", "abcdef12", 50, False, "20260520T120000")
        == "run_qwen3_14b_abcdef12_20260520T120000_limit_50"
    )


def test_output_stem_full() -> None:
    assert (
        output_stem("qwen3:14b", "abcdef12", 100, True, "20260520T120000")
        == "run_qwen3_14b_abcdef12_20260520T120000_full"
    )


def test_output_csv_path_no_accuracy(tmp_path: Path) -> None:
    p = output_csv_path("run_x_y_t_limit_50", None, tmp_path)
    assert p == tmp_path / "run_x_y_t_limit_50.csv"


def test_output_csv_path_with_accuracy(tmp_path: Path) -> None:
    p = output_csv_path("run_x_y_t_limit_50", 0.8567, tmp_path)
    assert p == tmp_path / "run_x_y_t_limit_50_86pct.csv"


def test_sidecar_path(tmp_path: Path) -> None:
    p = sidecar_path("run_x_y_t_full", tmp_path)
    assert p == tmp_path / "run_x_y_t_full_prompts.txt"
