import hashlib

from ..models import ProviderResponse


class FakeProvider:
    """Deterministic offline provider for tests and end-to-end plumbing.

    Decides the label from a hash of the rendered prompt, so the same prompt
    always gets the same answer.
    """

    name = "fake"

    def classify(self, prompt: str) -> ProviderResponse:
        label = "CG" if self._hash_bit(prompt) else "OR"
        text = (
            f"{label}\n"
            f"Deterministic fake response based on a hash of the prompt."
        )
        return ProviderResponse(text=text, latency_ms=0)

    @staticmethod
    def _hash_bit(text: str) -> int:
        return hashlib.sha256(text.encode("utf-8")).digest()[0] & 1
