from fake_review_classifier.providers.base import Provider
from fake_review_classifier.providers.fake import FakeProvider


def test_fake_provider_conforms_to_protocol() -> None:
    assert isinstance(FakeProvider(), Provider)


def test_fake_provider_is_deterministic() -> None:
    a = FakeProvider().classify("prompt", "some review text")
    b = FakeProvider().classify("prompt", "some review text")
    assert a == b


def test_fake_provider_emits_known_label_on_line_one() -> None:
    response = FakeProvider().classify("prompt", "some review text")
    first_line = response.text.splitlines()[0]
    assert first_line in {"CG", "OR"}


def test_fake_provider_has_reasoning_line() -> None:
    response = FakeProvider().classify("prompt", "some review text")
    lines = response.text.splitlines()
    assert len(lines) >= 2
    assert lines[1].strip() != ""


def test_fake_provider_distinguishes_inputs() -> None:
    """Across enough inputs the fake should not collapse to a single label."""
    labels = {
        FakeProvider().classify("p", f"review {i}").text.splitlines()[0]
        for i in range(50)
    }
    assert labels == {"CG", "OR"}
