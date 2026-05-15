from typing import Protocol, runtime_checkable

from ..models import ProviderResponse


@runtime_checkable
class Provider(Protocol):
    """Anything that classifies a review given a prompt."""

    name: str

    def classify(self, prompt: str, review_text: str) -> ProviderResponse: ...
