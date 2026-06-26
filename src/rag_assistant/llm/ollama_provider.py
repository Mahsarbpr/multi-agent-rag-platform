from langchain_ollama import OllamaLLM

from rag_assistant.llm.base_provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str, base_url: str):
        self.llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
        )

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)