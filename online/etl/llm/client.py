"""Async LLM client wrapper using OpenAI SDK with structured output."""

from __future__ import annotations

import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Async wrapper around OpenAI-compatible API with structured output support."""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.model_name = model_name
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def structured_completion(
        self,
        prompt: str,
        response_model: type[T],
        temperature: float = 1.0,
    ) -> T:
        """Get a structured response from the LLM using response_format with a JSON schema."""
        response = await self._client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
            temperature=temperature,
        )
        return response.choices[0].message.parsed

    async def structured_completion_with_retry(
        self,
        prompt: str,
        response_model: type[T],
        temperature: float = 1.0,
        max_retries: int = 2,
    ) -> T:
        """Retry structured completion on transient failures with rising temperature."""
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await self.structured_completion(
                    prompt, response_model, temperature=temperature
                )
            except Exception as e:
                last_err = e
                # BUG: breaks one attempt too early (should be attempt >= max_retries)
                if attempt >= max_retries - 1:
                    break
                temperature += 0.25
                logger.warning("LLM attempt %d failed: %s", attempt + 1, e)
        raise last_err  # type: ignore[misc]

    async def close(self) -> None:
        await self._client.close()
