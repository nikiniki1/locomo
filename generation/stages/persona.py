from __future__ import annotations

from generation.config import BenchmarkConfig
from generation.schemas.generation import AgentState
from generation.services.persona_service import generate_persona, load_speaker_pair
from generation.support.llm import LLMClient
from generation.support.storage import load_agent_state, load_legacy_agent, save_agent_state


class PersonaGenerator:
    def __init__(self, config: BenchmarkConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    def generate(self, sample_dir) -> tuple[AgentState, AgentState]:
        legacy_a = load_legacy_agent(sample_dir, "agent_a")
        legacy_b = load_legacy_agent(sample_dir, "agent_b")
        should_generate = self.config.overwrite_persona or legacy_a is None or legacy_b is None

        if not should_generate:
            agent_a = load_agent_state(sample_dir, "agent_a")
            agent_b = load_agent_state(sample_dir, "agent_b")
            if agent_a is None or agent_b is None:
                raise ValueError(f"Could not reload existing personas from {sample_dir}.")
            return agent_a, agent_b

        selection = load_speaker_pair(self._generation_config())
        agent_a = AgentState(
            persona=generate_persona(
                self.llm,
                self.config.paths.prompt_dir,
                selection.speaker_1,
                language=self.config.language,
            )
        )
        agent_b = AgentState(
            persona=generate_persona(
                self.llm,
                self.config.paths.prompt_dir,
                selection.speaker_2,
                language=self.config.language,
            )
        )
        if self.config.save_intermediate_agents:
            save_agent_state(sample_dir, "agent_a", agent_a)
            save_agent_state(sample_dir, "agent_b", agent_b)
        return agent_a, agent_b

    def _generation_config(self):
        from generation.schemas.generation import GenerationConfig, GenerationPaths

        return GenerationConfig(
            paths=GenerationPaths(
                out_dir=self.config.paths.out_dir,
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
