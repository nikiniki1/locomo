from __future__ import annotations

import json
import os
import re
from typing import TypeVar

from dotenv import load_dotenv
from openai import APIStatusError
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


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
    def __init__(self, model: str) -> None:
        self.model = model
        self.supports_structured_output = True
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_API_BASE"),
        )

    def complete_text(self, prompt: str, *, max_tokens: int, temperature: float = 1.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

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
            if exc.status_code == 422:
                self.supports_structured_output = False
            raise

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"Structured output parsing failed: {completion.choices[0].message.content}")
        return parsed
