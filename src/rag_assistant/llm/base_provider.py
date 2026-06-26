from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass