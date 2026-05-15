from typing import Protocol, runtime_checkable

from ..models import ProviderResponse


@runtime_checkable
class Provider(Protocol):
    """Anything that takes a fully-rendered prompt and returns a response."""

    name: str

    def classify(self, prompt: str) -> ProviderResponse: ...
