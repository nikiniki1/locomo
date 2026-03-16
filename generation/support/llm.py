from __future__ import annotations

import json
import logging
import os
import re
from typing import TypeVar

from dotenv import load_dotenv
from openai import APIStatusError, RateLimitError
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRY = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda state: logger.warning(
        "Rate limit hit, retrying in %.0fs (attempt %d/6)…",
        state.next_action.sleep,  # type: ignore[union-attr]
        state.attempt_number,
    ),
)


def parse_json_response(text: str) -> object:
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("Empty response", text, 0)

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start_idx:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON object found", cleaned, 0)


class LLMClient:
    def __init__(self, model: str, *, no_structured_output: bool = False) -> None:
        self.model = model
        # Disable structured output when explicitly requested (e.g. for GigaChat
        # which returns 200 with malformed parsed objects instead of raising 4xx).
        self.supports_structured_output = not no_structured_output
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_API_BASE"),
        )

    @_RETRY
    def complete_text(self, prompt: str, *, max_tokens: int, temperature: float = 1.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    @_RETRY
    def complete_structured(
        self,
        prompt: str,
        *,
        response_format: type[T],
        max_tokens: int,
        temperature: float = 1.0,
    ) -> T:
        if not self.supports_structured_output:
            raise ValueError("Structured output is disabled for this endpoint.")

        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        except APIStatusError as exc:
            # Disable structured output for any client error (4xx) — the model
            # or endpoint doesn't support it (422 = schema rejected, 400 = bad
            # request, 404 = endpoint missing, etc.). Callers fall back to
            # complete_text + manual JSON parsing.
            if 400 <= exc.status_code < 500:
                logger.warning(
                    "Structured output not supported (HTTP %s), disabling for this session.",
                    exc.status_code,
                )
                self.supports_structured_output = False
            raise

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"Structured output parsing failed: {completion.choices[0].message.content}")
        return parsed
