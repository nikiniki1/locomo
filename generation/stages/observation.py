from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from generation.schemas.benchmark import ObservationItem, SessionObservation
from generation.schemas.generation import SessionRecord
from generation.support.formatting import session_transcript
from generation.support.llm import LLMClient, parse_json_response


OBSERVATION_PROMPT_RU = """
Составь базу фактических наблюдений о двух собеседниках по приведённому ниже диалогу.
- Верни только JSON.
- Используй форму {{"speaker_facts": {{"Имя": [{{"fact": "...", "evidence": "D1:3"}}]}}}}.
- Каждый факт должен быть объективным, конкретным и опираться на диалог.
- В `evidence` укажи один `dia_id`, который подтверждает факт.
- Не включай абстрактные суждения о динамике отношений.
- Не более 8 фактов на каждого собеседника.
- Поля `fact` пиши по-русски.

ДИАЛОГ:
{conversation}
"""


class ObservationEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fact: str
    evidence: str


class ObservationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker_facts: dict[str, list[ObservationEntry]] = Field(default_factory=dict)


class ObservationGenerator:
    def __init__(self, llm: LLMClient, language: str = "ru") -> None:
        self.llm = llm
        self.language = language

    def generate(self, session: SessionRecord, speakers: list[str]) -> SessionObservation:
        prompt = OBSERVATION_PROMPT_RU.format(conversation=session_transcript(session))
        try:
            payload = self.llm.complete_structured(prompt, response_format=ObservationPayload, max_tokens=700)
        except Exception:
            raw = self.llm.complete_text(prompt, max_tokens=700, temperature=0.4)
            payload = ObservationPayload.model_validate(parse_json_response(raw))

        speaker_facts: dict[str, list[ObservationItem]] = {}
        for speaker in speakers:
            items = payload.speaker_facts.get(speaker, [])
            speaker_facts[speaker] = [
                ObservationItem(fact=item.fact.strip(), evidence=item.evidence.strip())
                for item in items
                if item.fact.strip() and item.evidence.strip()
            ]
        return SessionObservation(speaker_facts=speaker_facts)
