from __future__ import annotations

from generation.schemas.benchmark import BenchmarkConversation, BenchmarkSample
from generation.schemas.generation import AgentState


class BenchmarkAssembler:
    def build_empty_sample(self, sample_id: str, agent_a: AgentState, agent_b: AgentState) -> BenchmarkSample:
        conversation = BenchmarkConversation(
            speaker_a=agent_a.persona.name,
            speaker_b=agent_b.persona.name,
            session_date_times={session.index: session.date_time for session in agent_a.sessions},
            sessions={session.index: session.turns for session in agent_a.sessions},
        )
        return BenchmarkSample(
            sample_id=sample_id,
            conversation=conversation,
            observation={},
            session_summary={},
            event_summary={},
            qa=[],
        )
