from __future__ import annotations

from generation.config import BenchmarkConfig
from generation.schemas.generation import AgentState, GenerationConfig, GenerationPaths
from generation.services.session_service import generate_session
from generation.support.llm import LLMClient
from generation.support.storage import save_agent_state


class SessionGenerator:
    def __init__(self, config: BenchmarkConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    def generate(self, sample_dir, agent_a: AgentState, agent_b: AgentState) -> tuple[AgentState, AgentState]:
        if not agent_a.events or not agent_b.events:
            raise ValueError("Session generation requires event graphs for both agents.")

        generation_config = GenerationConfig(
            paths=GenerationPaths(
                out_dir=sample_dir,
                prompt_dir=self.config.paths.prompt_dir,
                msc_personas_file=self.config.paths.msc_personas_file,
            ),
            language=self.config.language,
            num_days=self.config.num_days,
            num_events=self.config.num_events,
            num_sessions=self.config.num_sessions,
            start_session=self.config.start_session,
            max_turns_per_session=self.config.max_turns_per_session,
            num_events_per_session=self.config.num_events_per_session,
            model=self.config.model,
            overwrite_persona=self.config.overwrite_persona,
            overwrite_events=self.config.overwrite_events,
            overwrite_session=self.config.overwrite_session,
            mark_persona_as_used=self.config.mark_persona_as_used,
        )

        for session_index in range(self.config.start_session, self.config.num_sessions + 1):
            needs_session = (
                self.config.overwrite_session
                or len(agent_a.sessions) < session_index
                or len(agent_b.sessions) < session_index
            )
            if not needs_session:
                continue
            generated = generate_session(
                self.llm,
                generation_config,
                agent_a=agent_a,
                agent_b=agent_b,
                session_index=session_index,
            )
            if generated is None:
                break
            session_a, session_b = generated
            if len(agent_a.sessions) >= session_index:
                agent_a.sessions[session_index - 1] = session_a
                agent_b.sessions[session_index - 1] = session_b
            else:
                agent_a.sessions.append(session_a)
                agent_b.sessions.append(session_b)
            if self.config.save_intermediate_agents:
                save_agent_state(sample_dir, "agent_a", agent_a)
                save_agent_state(sample_dir, "agent_b", agent_b)
        return agent_a, agent_b
