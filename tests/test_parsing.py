from fake_review_classifier.models import ProviderResponse
from fake_review_classifier.parsing import UNPARSEABLE, parse_response


def _resp(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, latency_ms=0)


def test_clean_two_line_response() -> None:
    label, reasoning = parse_response(_resp("CG\nGeneric praise."))
    assert label == "CG"
    assert reasoning == "Generic praise."


def test_or_label() -> None:
    label, _ = parse_response(_resp("OR\nSpecific complaint about colour."))
    assert label == "OR"


def test_lowercase_label_is_normalised() -> None:
    label, _ = parse_response(_resp("cg\nbecause"))
    assert label == "CG"


def test_punctuation_around_label() -> None:
    label, _ = parse_response(_resp("**CG**\nbecause"))
    assert label == "CG"


def test_leading_whitespace_ignored() -> None:
    label, _ = parse_response(_resp("\n  \nOR\nthe handle is at an angle"))
    assert label == "OR"


def test_garbage_first_line() -> None:
    label, reasoning = parse_response(_resp("I think this is fake.\nReasoning..."))
    assert label == UNPARSEABLE
    assert reasoning == "Reasoning..."


def test_empty_response() -> None:
    label, reasoning = parse_response(_resp(""))
    assert label == UNPARSEABLE
    assert reasoning == ""


def test_reasoning_joins_multiple_lines() -> None:
    _, reasoning = parse_response(_resp("CG\nfirst line.\nsecond line."))
    assert reasoning == "first line. second line."
