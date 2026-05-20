import time

import ollama

from ..models import ProviderResponse


class OllamaProvider:
    """Local Ollama provider.

    Assumes the Ollama daemon is running on the host and the model is pulled.
    """

    def __init__(self, model: str, host: str | None = None, temperature: float = 0.0) -> None:
        self.name = "ollama"
        self.model = model
        self.temperature = temperature
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def classify(self, prompt: str) -> ProviderResponse:
        start = time.perf_counter()
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature},
            stream=False,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ProviderResponse(text=response.response, latency_ms=latency_ms)
