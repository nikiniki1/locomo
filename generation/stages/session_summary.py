from __future__ import annotations

from generation.schemas.generation import SessionRecord
from generation.support.formatting import session_transcript
from generation.support.llm import LLMClient


SESSION_SUMMARY_PROMPT_RU = """
Напиши краткое резюме следующего разговора.
- Упомяни обоих собеседников по имени.
- Сохрани важные факты и временные указания.
- Описывай только эту сессию.
- Не больше 140 слов.
- Пиши по-русски.

ДИАЛОГ:
{conversation}
"""


class SessionSummaryGenerator:
    def __init__(self, llm: LLMClient, language: str = "ru") -> None:
        self.llm = llm
        self.language = language

    def generate(self, session: SessionRecord) -> str:
        prompt = SESSION_SUMMARY_PROMPT_RU.format(conversation=session_transcript(session))
        return self.llm.complete_text(prompt, max_tokens=220, temperature=0.4).strip()
