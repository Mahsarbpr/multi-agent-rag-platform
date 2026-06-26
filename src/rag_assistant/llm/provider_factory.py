from rag_assistant.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_MODEL,
    CLAUDE_MODEL,
)

from rag_assistant.llm.base_provider import LLMProvider
from rag_assistant.llm.ollama_provider import OllamaProvider
from rag_assistant.llm.openai_provider import OpenAIProvider
from rag_assistant.llm.claude_provider import ClaudeProvider


_PROVIDER_FACTORIES = {
    "ollama": lambda: OllamaProvider(
        model_name=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    ),
    "openai": lambda: OpenAIProvider(
        model_name=OPENAI_MODEL,
    ),
    "claude": lambda: ClaudeProvider(
        model_name=CLAUDE_MODEL,
    ),
}


def create_llm_provider() -> LLMProvider:
    """Create the configured LLM provider."""

    provider_name = LLM_PROVIDER.lower()

    try:
        return _PROVIDER_FACTORIES[provider_name]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}"
        ) from exc