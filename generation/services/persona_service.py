from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from generation.schemas.generation import GenerationConfig, PersonaProfile, SpeakerSelection
from generation.support.llm import LLMClient, parse_json_response


PERSONA_PROMPT_EN = (
    "Let's write speaker descriptions from a given set of life attributes. Example:\n\n"
    "%s\n\n"
    "Note: Add crucial details in the persona about the person such as their name, age, marital status, "
    "gender, job etc. Add additional details like names of family/friends or specific activities, likes and "
    "dislikes, experiences when appropriate.\n\n"
    "For the following attributes, write a persona. Output a json file with the keys 'persona' and 'name'.\n\n"
    "%s\n\nStart your answer with a curly bracket.\n"
)

PERSONA_PROMPT_RU = (
    "Напиши описание персонажа на основе набора жизненных атрибутов. Пример:\n\n"
    "%s\n\n"
    "Добавь важные детали о человеке: имя, возраст, семейное положение, пол, работу и другие уместные подробности. "
    "При необходимости добавь имена родственников и друзей, конкретные занятия, предпочтения, опыт и привычки.\n\n"
    "Для следующих атрибутов напиши персону. Верни JSON с ключами 'persona' и 'name'. Текст поля 'persona' должен быть на русском языке.\n\n"
    "%s\n\nНачни ответ с символа '{'.\n"
)


class PersonaResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    persona: str


def _load_persona_examples(prompt_dir: Path) -> list[str]:
    task = json.loads((prompt_dir / "persona_generation_examples.json").read_text(encoding="utf-8"))
    return [
        task["input_prefix"] + json.dumps(example["input"], indent=2) + "\n" + task["output_prefix"] + example["output"]
        for example in task["examples"]
    ]


def load_speaker_pair(config: GenerationConfig) -> SpeakerSelection:
    payload = json.loads(config.paths.msc_personas_file.read_text(encoding="utf-8"))
    split = "train"
    candidates = [
        (idx, entry)
        for idx, entry in enumerate(payload[split])
        if not entry.get("in_dataset")
    ]
    if not candidates:
        raise ValueError("No unused MSC personas available in the train split.")
    source_index, selected = random.choice(candidates)
    if config.mark_persona_as_used:
        payload[split][source_index]["in_dataset"] = 1
        config.paths.msc_personas_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return SpeakerSelection(
        speaker_1=selected["Speaker 1"],
        speaker_2=selected["Speaker 2"],
        source_split=split,
        source_index=source_index,
    )


def generate_persona(
    llm: LLMClient,
    prompt_dir: Path,
    attributes: list[str] | dict[str, object],
    language: str = "ru",
) -> PersonaProfile:
    examples = _load_persona_examples(prompt_dir)
    input_string = "Input:\n" + json.dumps(attributes, indent=2)
    prompt_template = PERSONA_PROMPT_RU if language.lower().startswith("ru") else PERSONA_PROMPT_EN
    prompt = prompt_template % (examples, input_string)

    try:
        parsed = llm.complete_structured(prompt, response_format=PersonaResponse, max_tokens=1000)
    except Exception:
        raw = llm.complete_text(prompt, max_tokens=1000)
        parsed = PersonaResponse.model_validate(parse_json_response(raw))

    return PersonaProfile(name=parsed.name, persona=parsed.persona, msc_prompt=attributes)
