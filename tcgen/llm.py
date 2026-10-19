"""Thin wrapper around Groq so the pipeline only depends on one small interface."""

import time
from typing import Protocol, TypeVar

from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)

# Groq returns HTTP 400 with these codes when the model's output didn't match the schema
# (including the occasional empty generation). Asking again usually succeeds.
_BAD_GENERATION = ("json_validate_failed", "Failed to validate JSON", "tool_use_failed")


class StructuredLLM(Protocol):
    def generate(self, system: str, user: str, schema: type[T]) -> T: ...


def is_bad_generation(exc: Exception) -> bool:
    return any(marker in str(exc) for marker in _BAD_GENERATION)


class GroqLLM:
    attempts = 3

    def __init__(self, api_key: str, model: str = config.DEFAULT_MODEL, temperature: float = 0.2):
        from langchain_groq import ChatGroq

        # max_retries lets the Groq client back off and retry on 429 rate-limit responses,
        # which the free tier hits quickly on larger documents.
        # max_tokens is generous because GPT-OSS spends part of its budget on reasoning
        # before writing the JSON; too small a budget yields empty or cut-off output.
        self._chat = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=16384,
            max_retries=6,
            timeout=180,
        )
        self.model = model

    def _runnable(self, schema: type[T], attempt: int):
        if attempt < self.attempts:
            return self._chat.with_structured_output(schema, method="json_schema", strict=True)
        # Last try: tool calling, a different path on Groq's side that often succeeds
        # when strict JSON mode keeps failing for a given input.
        return self._chat.with_structured_output(schema, method="function_calling")

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        for attempt in range(1, self.attempts + 1):
            try:
                result = self._runnable(schema, attempt).invoke([("system", system), ("human", user)])
                if result is None:
                    raise ValueError("json_validate_failed: empty structured output")
                return result if isinstance(result, schema) else schema.model_validate(result)
            except Exception as exc:
                if attempt == self.attempts or not is_bad_generation(exc):
                    raise
                time.sleep(1.5 * attempt)
        raise RuntimeError("unreachable")
