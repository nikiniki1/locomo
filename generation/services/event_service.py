from __future__ import annotations

import json
from datetime import date, datetime, timedelta
import random
from pathlib import Path

import tiktoken
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from generation.schemas.generation import Event, GenerationConfig, PersonaProfile
from generation.support.llm import LLMClient, parse_json_response


EVENT_INIT_PROMPT_EN = """
Let's write a graph representing sub-events that occur in a person's life based on a short summary of their personality. Nodes represent sub-events and edges represent the influence of past sub-events on a current sub-event.
- The graph is represented as a json object with a single key "events".
- The value of "events" is a json list.
- Each entry is a dictionary containing the following keys: "sub-event", "time", "caused_by", "id".
- The "sub-event" field contains a short description of the sub-event.
- The "time" field contains a date.
- The "id" field contains a unique identifier for the sub-event.
- The "caused_by" field represents edges and is a list of "id" of existing sub-events that have caused this sub-event. Sub-events in the "caused_by" field should occur on dates before the sub-event they have caused. Generate as many causal connections as possible.
- Sub-events can be positive or negative life events.

For example,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

For the following input personality, generate three independent sub-events E1, E2 and E3 aligned with their personality. Sub-events can be positive or negative life events and should reflect evolution in the person's relationships, state of mind, personality etc.

PERSONALITY: %s
OUTPUT:
"""


EVENT_CONTINUE_PROMPT_EN = """
Let's write a graph representing sub-events that occur in a person's life based on a short summary of their personality. Nodes represent sub-events and edges represent the influence of past sub-events on a current sub-event.
- The graph is represented as a json object with a single key "events".
- The value of "events" is a json list.
- Each entry is a dictionary containing the following keys: "sub-event", "time", "caused_by", "id".
- The "sub-event" field contains a short description of the sub-event.
- The "time" field contains a date.
- The "id" field contains a unique identifier for the sub-event.
- The "caused_by" field represents edges and is a list of "id" of existing sub-events that have caused this sub-event. Sub-events in the "caused_by" field should occur on dates before the sub-event they have caused. Generate as many causal connections as possible.
- Sub-events can be positive or negative life events.

For example,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

For the following input personality, generate new sub-events %s that are caused by one or more EXISTING sub-events. Sub-events can be positive or negative life events and should reflect evolution in the person's relationships, state of mind, personality etc. Do not repeat existing sub-events.

PERSONALITY: %s
EXISTING: %s
OUTPUT:
"""

EVENT_INIT_PROMPT_RU = """
Построй граф подсобытий из жизни человека на основе краткого описания его личности. Узлы графа это подсобытия, а рёбра отражают влияние прошлых подсобытий на текущее.
- Граф должен быть представлен как JSON-объект с одним ключом "events".
- Значение "events" должно быть JSON-списком.
- Каждый элемент списка должен содержать ключи "sub-event", "time", "caused_by", "id".
- Поле "sub-event" должно содержать короткое описание подсобытия на русском языке.
- Поле "time" должно содержать дату.
- Поле "id" должно содержать уникальный идентификатор подсобытия.
- Поле "caused_by" должно быть списком "id" уже существующих подсобытий, которые вызвали текущее подсобытие. Подсобытия в "caused_by" должны происходить раньше вызванного ими события. Добавляй как можно больше причинно-следственных связей.
- Подсобытия могут быть как позитивными, так и негативными жизненными событиями.

Пример,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

Для следующего описания личности сгенерируй три независимых подсобытия E1, E2 и E3, соответствующих этой личности. Подсобытия должны отражать развитие отношений, эмоционального состояния, характера и жизненных обстоятельств.

PERSONALITY: %s
OUTPUT:
"""


EVENT_CONTINUE_PROMPT_RU = """
Построй граф подсобытий из жизни человека на основе краткого описания его личности. Узлы графа это подсобытия, а рёбра отражают влияние прошлых подсобытий на текущее.
- Граф должен быть представлен как JSON-объект с одним ключом "events".
- Значение "events" должно быть JSON-списком.
- Каждый элемент списка должен содержать ключи "sub-event", "time", "caused_by", "id".
- Поле "sub-event" должно содержать короткое описание подсобытия на русском языке.
- Поле "time" должно содержать дату.
- Поле "id" должно содержать уникальный идентификатор подсобытия.
- Поле "caused_by" должно быть списком "id" уже существующих подсобытий, которые вызвали текущее подсобытие. Подсобытия в "caused_by" должны происходить раньше вызванного ими события. Добавляй как можно больше причинно-следственных связей.
- Подсобытия могут быть как позитивными, так и негативными жизненными событиями.

Пример,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

Для следующего описания личности сгенерируй новые подсобытия %s, которые вызваны одним или несколькими УЖЕ СУЩЕСТВУЮЩИМИ подсобытиями. Не повторяй уже существующие подсобытия. Пиши "sub-event" по-русски.

PERSONALITY: %s
EXISTING: %s
OUTPUT:
"""


class EventNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    date: str = Field(validation_alias=AliasChoices("date", "time"))
    sub_event: str = Field(validation_alias=AliasChoices("sub-event", "sub_event"))
    caused_by: list[str] = Field(default_factory=list)

    def to_event(self) -> Event:
        return Event(id=self.id, date=self.date, **{"sub-event": self.sub_event}, caused_by=self.caused_by)


class EventGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    events: list[EventNode]

    def to_events(self) -> list[Event]:
        return [item.to_event() for item in self.events]


def _num_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding_name = "cl100k_base" if model_name in {"gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"} else "p50k_base"
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def _load_graph_example(prompt_dir: Path) -> tuple[str, str]:
    event_examples = json.loads((prompt_dir / "event_generation_examples.json").read_text(encoding="utf-8"))
    graph_examples = json.loads((prompt_dir / "graph_generation_examples.json").read_text(encoding="utf-8"))
    persona_example = event_examples["examples"][0]["input"] + "\nGenerate events between 1 January, 2020 and 30 April, 2020."
    graph_example = json.dumps({"events": graph_examples["examples"][0]["output"][:12]}, indent=2)
    return persona_example, graph_example


def random_date_range(num_days: int) -> tuple[str, str]:
    start = date(2022, 1, 1) + timedelta(days=random.randint(1, (date(2023, 6, 1) - date(2022, 1, 1)).days - 1))
    end = start + timedelta(days=num_days)
    return start.strftime("%d %B, %Y"), end.strftime("%d %B, %Y")


def _normalize_event_payload(payload: object) -> list[Event]:
    if isinstance(payload, EventGraph):
        return payload.to_events()
    if isinstance(payload, str):
        payload = parse_json_response(payload)
    if isinstance(payload, dict):
        if "events" in payload:
            events_payload = payload["events"]
            if not isinstance(events_payload, list):
                raise ValueError(f"Unsupported event payload: {payload}")
            return [EventNode.model_validate(item).to_event() for item in events_payload]
        values = list(payload.values())
        if "id" in payload:
            payload = [payload]
        elif values:
            for value in values:
                try:
                    return _normalize_event_payload(value)
                except Exception:
                    continue
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported event payload: {payload}")

    normalized: list[Event] = []
    for item in payload:
        if isinstance(item, str):
            item = parse_json_response(item)
        if isinstance(item, list):
            normalized.extend(_normalize_event_payload(item))
            continue
        normalized.append(EventNode.model_validate(item).to_event())
    return normalized


def _event_context(events: list[Event], *, limit: int = 12) -> str:
    return json.dumps({"events": [event.to_legacy_dict() for event in events[-limit:]]}, indent=2)


def _generate_events_text_fallback(
    llm: LLMClient,
    prompt: str,
    *,
    max_tokens: int,
) -> list[Event]:
    fallback_prompt = (
        prompt
        + '\n\nReturn valid JSON only. Do not include explanations, markdown, or prose. '
        + 'Use the schema {"events": [...]} exactly.'
    )
    raw = llm.complete_text(fallback_prompt, max_tokens=max_tokens, temperature=0.7)
    return _normalize_event_payload(raw)


def generate_events(
    llm: LLMClient,
    config: GenerationConfig,
    persona: PersonaProfile,
    start_date: str,
    end_date: str,
) -> list[Event]:
    persona_example, graph_example = _load_graph_example(config.paths.prompt_dir)
    if config.language.lower().startswith("ru"):
        prompt_persona = persona.persona + f"\nРаспредели даты между {start_date} и {end_date}. Пиши события на русском языке."
        init_prompt = EVENT_INIT_PROMPT_RU % (persona_example, graph_example, prompt_persona)
    else:
        prompt_persona = persona.persona + f"\nAssign dates between {start_date} and {end_date}."
        init_prompt = EVENT_INIT_PROMPT_EN % (persona_example, graph_example, prompt_persona)

    try:
        events = llm.complete_structured(init_prompt, response_format=EventGraph, max_tokens=512).to_events()
    except Exception:
        events = _generate_events_text_fallback(llm, init_prompt, max_tokens=512)

    while len(events) < config.num_events:
        last_id = int(events[-1].id[1:])
        remaining = config.num_events - len(events)
        batch_size = min(2, remaining)
        next_ids = [f"E{i}" for i in range(last_id + 1, last_id + batch_size + 1)]
        requested = ", ".join(next_ids) if len(next_ids) > 1 else next_ids[0]
        if config.language.lower().startswith("ru"):
            continue_prompt = EVENT_CONTINUE_PROMPT_RU % (
                persona_example,
                graph_example,
                requested,
                prompt_persona,
                _event_context(events),
            )
        else:
            continue_prompt = EVENT_CONTINUE_PROMPT_EN % (
                persona_example,
                graph_example,
                requested,
                prompt_persona,
                _event_context(events),
            )
        prompt_tokens = _num_tokens(continue_prompt)
        max_tokens = max(256, min(768, 4096 - prompt_tokens - 128))
        try:
            new_events = llm.complete_structured(
                continue_prompt,
                response_format=EventGraph,
                max_tokens=max_tokens,
            ).to_events()
        except Exception:
            new_events = _generate_events_text_fallback(llm, continue_prompt, max_tokens=max_tokens)

        seen = {event.id for event in events}
        events.extend([event for event in new_events if event.id not in seen])
        events = filter_events(events)
    return events[: config.num_events]


def filter_events(events: list[Event]) -> list[Event]:
    id_to_event = {event.id: event for event in events}
    remove_ids: set[str] = set()
    for event in events:
        if event.caused_by:
            continue
        has_child = any(event.id in other.caused_by for other in events)
        if not has_child:
            remove_ids.add(event.id)
    return [event for event in events if event.id not in remove_ids]
