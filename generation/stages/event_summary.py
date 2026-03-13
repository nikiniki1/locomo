from __future__ import annotations

from generation.schemas.benchmark import SessionEventSummary
from generation.schemas.generation import AgentState, SessionRecord
from generation.support.formatting import session_date_label


class EventSummaryGenerator:
    def generate(self, session_a: SessionRecord, session_b: SessionRecord, speaker_a: AgentState, speaker_b: AgentState) -> SessionEventSummary:
        speaker_events = {
            speaker_a.persona.name: [event.sub_event for event in session_a.events],
            speaker_b.persona.name: [event.sub_event for event in session_b.events],
        }
        return SessionEventSummary(
            date=session_date_label(session_a.date_time),
            speaker_events=speaker_events,
        )
