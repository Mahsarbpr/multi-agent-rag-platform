import os

from langchain_anthropic import ChatAnthropic

from rag_assistant.llm.base_provider import LLMProvider


class ClaudeProvider(LLMProvider):
    def __init__(self, model_name: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

        self.llm = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=0,
        )

    def invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content