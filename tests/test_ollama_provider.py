from types import SimpleNamespace
from unittest.mock import MagicMock

from phoney.providers.base import Provider
from phoney.providers.ollama import OllamaProvider


def _provider_with_mock(response_text: str) -> tuple[OllamaProvider, MagicMock]:
    provider = OllamaProvider(model="test-model")
    mock_client = MagicMock()
    mock_client.generate.return_value = SimpleNamespace(response=response_text)
    provider._client = mock_client
    return provider, mock_client


def test_ollama_provider_conforms_to_protocol() -> None:
    provider, _ = _provider_with_mock("CG\nbecause.")
    assert isinstance(provider, Provider)


def test_ollama_provider_returns_model_text() -> None:
    provider, _ = _provider_with_mock("CG\nbecause.")
    result = provider.classify("rendered prompt")
    assert result.text == "CG\nbecause."
    assert result.latency_ms >= 0


def test_ollama_provider_passes_prompt_and_model() -> None:
    provider, client = _provider_with_mock("OR\nfine.")
    provider.classify("rendered prompt")

    client.generate.assert_called_once()
    kwargs = client.generate.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["prompt"] == "rendered prompt"
    assert kwargs["stream"] is False
    assert "temperature" in kwargs["options"]
