from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PersonaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    persona: str
    msc_prompt: list[str] | dict[str, object] | None = None


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    date: str
    sub_event: str = Field(alias="sub-event")
    caused_by: list[str] = Field(default_factory=list)

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "date": self.date,
            "sub-event": self.sub_event,
            "caused_by": list(self.caused_by),
        }


class SessionTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dia_id: str
    speaker: str
    text: str
    clean_text: str

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "dia_id": self.dia_id,
            "speaker": self.speaker,
            "text": self.text,
            "clean_text": self.clean_text,
        }


class SessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    date_time: str
    events: list[Event] = Field(default_factory=list)
    turns: list[SessionTurn] = Field(default_factory=list)
    summary: str | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: PersonaProfile
    events: list[Event] = Field(default_factory=list)
    sessions: list[SessionRecord] = Field(default_factory=list)

    def to_legacy_dict(self) -> dict[str, object]:
        payload = {
            "name": self.persona.name,
            "persona_summary": self.persona.persona,
            "msc_prompt": self.persona.msc_prompt,
            "graph": [event.to_legacy_dict() for event in self.events],
        }
        for session in self.sessions:
            payload[f"session_{session.index}_date_time"] = session.date_time
            payload[f"events_session_{session.index}"] = [event.to_legacy_dict() for event in session.events]
            payload[f"session_{session.index}"] = [turn.to_legacy_dict() for turn in session.turns]
            if session.summary is not None:
                payload[f"session_{session.index}_summary"] = session.summary
        return {k: v for k, v in payload.items() if v is not None}


class ConversationArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_a: AgentState
    agent_b: AgentState
    events_start_date: str | None = None
    events_end_date: str | None = None


class GenerationPaths(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    out_dir: Path
    prompt_dir: Path
    msc_personas_file: Path = Path("data/msc_personas_all.json")


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    paths: GenerationPaths
    language: str = "ru"
    num_days: int = 240
    num_events: int = 15
    num_sessions: int = 20
    start_session: int = 1
    max_turns_per_session: int = 20
    num_events_per_session: int = 1
    model: str = "gpt-4o-mini"
    overwrite_persona: bool = False
    overwrite_events: bool = False
    overwrite_session: bool = False
    mark_persona_as_used: bool = False


class SpeakerSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_1: list[str] | dict[str, object]
    speaker_2: list[str] | dict[str, object]
    source_split: Literal["train", "valid", "test"]
    source_index: int
