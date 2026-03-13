from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from generation.schemas.generation import AgentState, SessionTurn


ScalarAnswer = str | int | float | bool


class ObservationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact: str
    evidence: str

    def to_legacy_list(self) -> list[str]:
        return [self.fact, self.evidence]


class SessionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_facts: dict[str, list[ObservationItem]]

    def to_legacy_dict(self) -> dict[str, list[list[str]]]:
        return {
            speaker: [item.to_legacy_list() for item in items]
            for speaker, items in self.speaker_facts.items()
        }


class SessionEventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    speaker_events: dict[str, list[str]]

    def to_legacy_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {speaker: list(events) for speaker, events in self.speaker_events.items()}
        payload["date"] = self.date
        return payload


class QAPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: ScalarAnswer
    evidence: list[str] = Field(default_factory=list)
    category: int

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": list(self.evidence),
            "category": self.category,
        }


class BenchmarkConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_a: str
    speaker_b: str
    session_date_times: dict[int, str]
    sessions: dict[int, list[SessionTurn]]

    def to_legacy_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "speaker_a": self.speaker_a,
            "speaker_b": self.speaker_b,
        }
        for index in sorted(self.sessions):
            payload[f"session_{index}_date_time"] = self.session_date_times[index]
            payload[f"session_{index}"] = [turn.to_legacy_dict() for turn in self.sessions[index]]
        return payload


class BenchmarkSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    conversation: BenchmarkConversation
    observation: dict[int, SessionObservation]
    session_summary: dict[int, str]
    event_summary: dict[int, SessionEventSummary]
    qa: list[QAPair]

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "conversation": self.conversation.to_legacy_dict(),
            "observation": {
                f"session_{index}_observation": observation.to_legacy_dict()
                for index, observation in sorted(self.observation.items())
            },
            "session_summary": {
                f"session_{index}_summary": summary
                for index, summary in sorted(self.session_summary.items())
            },
            "event_summary": {
                f"events_session_{index}": summary.to_legacy_dict()
                for index, summary in sorted(self.event_summary.items())
            },
            "qa": [item.to_legacy_dict() for item in self.qa],
        }


class BenchmarkRunArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    agent_a: AgentState
    agent_b: AgentState
    sample: BenchmarkSample | None = None
