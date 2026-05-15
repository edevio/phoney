import hashlib

from ..models import ProviderResponse


class FakeProvider:
    """Deterministic offline provider for tests and end-to-end plumbing.

    Label is decided by hashing the review text, so the same review always
    gets the same label.
    """

    name = "fake"

    def classify(self, prompt: str, review_text: str) -> ProviderResponse:
        del prompt  # ignored; the fake doesn't follow instructions
        label = "CG" if self._hash_bit(review_text) else "OR"
        text = (
            f"{label}\n"
            f"Deterministic fake response based on a hash of the review text."
        )
        return ProviderResponse(text=text, latency_ms=0)

    @staticmethod
    def _hash_bit(text: str) -> int:
        return hashlib.sha256(text.encode("utf-8")).digest()[0] & 1
