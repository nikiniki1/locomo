from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkPaths(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    out_dir: Path
    prompt_dir: Path = Path("prompt_examples")
    msc_personas_file: Path = Path("data/msc_personas_all.json")
    dataset_file: Path | None = None


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    paths: BenchmarkPaths
    model: str = "gpt-4o-mini"
    language: str = "ru"
    num_samples: int = 1
    num_days: int = 240
    num_events: int = 15
    num_sessions: int = 20
    start_session: int = 1
    max_turns_per_session: int = 20
    num_events_per_session: int = 1
    qa_per_sample: int = 24
    overwrite_persona: bool = False
    overwrite_events: bool = False
    overwrite_session: bool = False
    overwrite_annotations: bool = False
    mark_persona_as_used: bool = False
    save_intermediate_agents: bool = True
    sample_prefix: str = "sample"
    sample_id_prefix: str = "S"
    fail_fast: bool = True
    no_structured_output: bool = False

    @property
    def dataset_file(self) -> Path:
        if self.paths.dataset_file is not None:
            return self.paths.dataset_file
        return self.paths.out_dir / "benchmark.json"

    def sample_dir(self, sample_index: int) -> Path:
        return self.paths.out_dir / f"{self.sample_prefix}_{sample_index:02d}"

    def sample_id(self, sample_index: int) -> str:
        return f"{self.sample_id_prefix}{sample_index}"
