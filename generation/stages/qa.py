from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from generation.config import BenchmarkConfig
from generation.schemas.benchmark import BenchmarkSample, QAPair
from generation.support.formatting import full_conversation_transcript
from generation.support.llm import LLMClient, parse_json_response

logger = logging.getLogger(__name__)

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
    category: int = 1


class QAPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    qa: list[QAEntry] = Field(default_factory=list)


class QAGenerator:
    def __init__(self, config: BenchmarkConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    def generate(self, sample: BenchmarkSample) -> list[QAPair]:
        target = self.config.qa_per_sample

        # For models without structured output (e.g. GigaChat), generate in
        # per-session batches to avoid truncation on large JSON outputs.
        if not self.llm.supports_structured_output:
            return self._generate_batched(sample, target)

        return self._generate_full(sample, target)

    # ── single-call path (Gemini, GPT, etc.) ──────────────────────────────

    def _generate_full(self, sample: BenchmarkSample, target: int) -> list[QAPair]:
        prompt = QA_PROMPT_RU.format(
            num_questions=target,
            conversation=full_conversation_transcript(sample.conversation),
        )
        max_tokens = max(2200, target * 150)
        try:
            payload = self.llm.complete_structured(prompt, response_format=QAPayload, max_tokens=max_tokens)
        except Exception:
            raw = self.llm.complete_text(prompt, max_tokens=max_tokens, temperature=0.5)
            payload = self._safe_parse(raw)
        return self._to_qa_pairs(payload.qa)

    # ── batched path (GigaChat / no-structured-output models) ─────────────

    def _generate_batched(self, sample: BenchmarkSample, target: int) -> list[QAPair]:
        """Generate QA in per-session batches, aggregate unique questions."""
        session_keys = sorted(sample.conversation.sessions.keys())
        num_sessions = len(session_keys)
        if num_sessions == 0:
            return []

        per_session = max(3, (target + num_sessions - 1) // num_sessions)
        # ~150 tokens per pair, minimum 512
        max_tokens = max(512, per_session * 150)

        all_pairs: list[QAPair] = []
        seen_questions: set[str] = set()

        for idx in session_keys:
            if len(all_pairs) >= target:
                break
            turns = sample.conversation.sessions[idx]
            date_time = sample.conversation.session_date_times.get(idx, "")
            turn_lines = "\n".join(
                f"[{t.dia_id}] {t.speaker} сказал(а): \"{t.clean_text or t.text}\""
                for t in turns
            )
            conv_text = f"Сессия {idx}\n{date_time}\n{turn_lines}"

            prompt = QA_PROMPT_RU.format(
                num_questions=per_session,
                conversation=conv_text,
            )
            try:
                raw = self.llm.complete_text(prompt, max_tokens=max_tokens, temperature=0.5)
                payload = self._safe_parse(raw)
            except Exception as exc:
                logger.warning("QA batch for session %s failed: %s", idx, exc)
                continue

            for pair in self._to_qa_pairs(payload.qa):
                norm_q = pair.question.strip().lower()
                if norm_q and norm_q not in seen_questions:
                    seen_questions.add(norm_q)
                    all_pairs.append(pair)
                    if len(all_pairs) >= target:
                        break

        logger.info("Batched QA: generated %d/%d pairs across %d sessions",
                    len(all_pairs), target, num_sessions)
        return all_pairs

    # ── helpers ───────────────────────────────────────────────────────────

    def _safe_parse(self, raw: str) -> QAPayload:
        try:
            return QAPayload.model_validate(parse_json_response(raw))
        except Exception as exc:
            logger.warning("QA JSON parse failed (%s). Raw: %s", exc, raw[:200])
            return QAPayload()

    def _to_qa_pairs(self, entries: list[QAEntry]) -> list[QAPair]:
        pairs: list[QAPair] = []
        for item in entries:
            question = item.question.strip()
            if not question:
                continue
            evidence = self._normalize_evidence(item.evidence)
            # Don't require evidence — GigaChat may omit D1:3 format
            pairs.append(
                QAPair(
                    question=question,
                    answer=item.answer,
                    evidence=evidence,
                    category=max(1, min(3, item.category)),
                )
            )
        return pairs

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
