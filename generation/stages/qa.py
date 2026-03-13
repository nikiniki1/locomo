from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from generation.config import BenchmarkConfig
from generation.schemas.benchmark import BenchmarkSample, QAPair
from generation.support.formatting import full_conversation_transcript
from generation.support.llm import LLMClient, parse_json_response


QA_PROMPT_RU = """
Создай аннотации вопрос-ответ для бенчмарка по длинному диалогу.
- Верни только JSON.
- Используй форму {{"qa": [{{"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 1}}]}}.
- Вопросы должны быть строго отвечаемы по диалогу.
- `evidence` должен ссылаться на реальные `dia_id` из диалога.
- Смешай вопросы на прямое воспоминание фактов, временные связи и multi-hop между сессиями.
- Избегай дубликатов и вопросов да/нет.
- Сгенерируй ровно {num_questions} вопросов.
- Пиши вопросы и ответы по-русски.
- Категории:
  1 = прямое воспоминание факта из одной сессии
  2 = временное рассуждение
  3 = кросс-сессионная или multi-hop память

ДИАЛОГ:
{conversation}
"""


class QAEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str | int | float | bool
    evidence: list[str] = Field(default_factory=list)
    category: int


class QAPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    qa: list[QAEntry] = Field(default_factory=list)


class QAGenerator:
    def __init__(self, config: BenchmarkConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    def generate(self, sample: BenchmarkSample) -> list[QAPair]:
        prompt = QA_PROMPT_RU.format(
            num_questions=self.config.qa_per_sample,
            conversation=full_conversation_transcript(sample.conversation),
        )
        try:
            payload = self.llm.complete_structured(prompt, response_format=QAPayload, max_tokens=2200)
        except Exception:
            raw = self.llm.complete_text(prompt, max_tokens=2200, temperature=0.5)
            payload = QAPayload.model_validate(parse_json_response(raw))

        questions: list[QAPair] = []
        for item in payload.qa:
            question = item.question.strip()
            evidence = self._normalize_evidence(item.evidence)
            if not question or not evidence:
                continue
            questions.append(
                QAPair(
                    question=question,
                    answer=item.answer,
                    evidence=evidence,
                    category=max(1, min(3, item.category)),
                )
            )
        return questions

    def _normalize_evidence(self, raw_evidence: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in raw_evidence:
            text = value.strip()
            if not text:
                continue
            matches = re.findall(r"D\d+:\d+", text)
            candidates = matches if matches else [text.rstrip(".,;: ")]
            for candidate in candidates:
                cleaned = candidate.strip().rstrip(".,;: ")
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                normalized.append(cleaned)
        return normalized
