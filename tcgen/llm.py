"""Thin wrapper around Groq so the pipeline only depends on one small interface."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def generate(self, system: str, user: str, schema: type[T]) -> T: ...


class GroqLLM:
    def __init__(self, api_key: str, model: str = config.DEFAULT_MODEL, temperature: float = 0.2):
        from langchain_groq import ChatGroq

        # max_retries lets the Groq client back off and retry on 429 rate-limit responses,
        # which the free tier hits quickly on larger documents.
        self._chat = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=6,
            timeout=120,
        )
        self.model = model

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        structured = self._chat.with_structured_output(schema, method="json_schema", strict=True)
        result = structured.invoke([("system", system), ("human", user)])
        if not isinstance(result, schema):
            result = schema.model_validate(result)
        return result
