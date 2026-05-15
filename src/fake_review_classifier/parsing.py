from .models import ProviderResponse

UNPARSEABLE = "UNPARSEABLE"
VALID_LABELS = {"CG", "OR"}


def parse_response(response: ProviderResponse) -> tuple[str, str]:
    """Return (label, reasoning) extracted from a provider response.

    Lenient: case-insensitive, ignores leading whitespace and stray
    punctuation. If the first non-empty line does not start with CG or OR,
    label is UNPARSEABLE.
    """
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    if not lines:
        return UNPARSEABLE, ""

    first = lines[0].upper().lstrip("`*-_ ").rstrip("`*-_.,: ")
    label = first if first in VALID_LABELS else UNPARSEABLE

    reasoning = " ".join(lines[1:]) if len(lines) > 1 else ""
    return label, reasoning
