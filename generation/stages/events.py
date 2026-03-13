from __future__ import annotations

from generation.config import BenchmarkConfig
from generation.schemas.generation import AgentState, GenerationConfig, GenerationPaths
from generation.services.event_service import generate_events, random_date_range
from generation.support.llm import LLMClient
from generation.support.storage import save_agent_state


class EventGraphGenerator:
    def __init__(self, config: BenchmarkConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    def generate(self, sample_dir, agent_a: AgentState, agent_b: AgentState) -> tuple[AgentState, AgentState]:
        if not self.config.overwrite_events and agent_a.events and agent_b.events:
            return agent_a, agent_b

        start_date, end_date = random_date_range(self.config.num_days)
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
        agent_a.events = generate_events(self.llm, generation_config, agent_a.persona, start_date, end_date)
        agent_b.events = generate_events(self.llm, generation_config, agent_b.persona, start_date, end_date)
        if self.config.save_intermediate_agents:
            save_agent_state(sample_dir, "agent_a", agent_a)
            save_agent_state(sample_dir, "agent_b", agent_b)
        return agent_a, agent_b
