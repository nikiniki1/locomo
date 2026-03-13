from __future__ import annotations

import json
import logging
from pathlib import Path

from generation.config import BenchmarkConfig
from generation.schemas.benchmark import BenchmarkSample
from generation.stages import (
    BenchmarkAssembler,
    BenchmarkValidator,
    EventGraphGenerator,
    EventSummaryGenerator,
    ObservationGenerator,
    PersonaGenerator,
    QAGenerator,
    SessionGenerator,
    SessionSummaryGenerator,
)
from generation.support.llm import LLMClient
from generation.support.storage import ensure_output_dir, save_agent_state
from tqdm import tqdm


logger = logging.getLogger(__name__)


class GenerateBenchmark:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._configure_logging()
        ensure_output_dir(self.config.paths.out_dir)
        self.llm = LLMClient(model=self.config.model)

        self.persona_generator = PersonaGenerator(config, self.llm)
        self.event_generator = EventGraphGenerator(config, self.llm)
        self.session_generator = SessionGenerator(config, self.llm)
        self.observation_generator = ObservationGenerator(self.llm, config.language)
        self.session_summary_generator = SessionSummaryGenerator(self.llm, config.language)
        self.event_summary_generator = EventSummaryGenerator()
        self.assembler = BenchmarkAssembler()
        self.qa_generator = QAGenerator(config, self.llm)
        self.validator = BenchmarkValidator()

    def run(self) -> list[BenchmarkSample]:
        logger.info(
            "Starting benchmark generation: samples=%s model=%s language=%s out_dir=%s",
            self.config.num_samples,
            self.config.model,
            self.config.language,
            self.config.paths.out_dir,
        )
        samples: list[BenchmarkSample] = []
        with tqdm(
            total=self.config.num_samples,
            desc="benchmark",
            unit="sample",
        ) as progress:
            for sample_index in range(1, self.config.num_samples + 1):
                sample = self._run_single_sample(sample_index)
                samples.append(sample)
                progress.update(1)
                progress.set_postfix_str(sample.sample_id)
        self._save_dataset(samples)
        logger.info("Finished benchmark generation: dataset=%s", self.config.dataset_file)
        return samples

    def _run_single_sample(self, sample_index: int) -> BenchmarkSample:
        sample_dir = self.config.sample_dir(sample_index)
        ensure_output_dir(sample_dir)
        sample_id = self.config.sample_id(sample_index)
        logger.info("Generating %s in %s", sample_id, sample_dir)

        stage_names = [
            "persona",
            "events",
            "sessions",
            "assemble",
            "annotations",
            "validate",
            "save",
        ]
        with tqdm(total=len(stage_names), desc=sample_id, unit="stage", leave=False) as progress:
            progress.set_postfix_str("persona")
            agent_a, agent_b = self.persona_generator.generate(sample_dir)
            progress.update(1)

            progress.set_postfix_str("events")
            agent_a, agent_b = self.event_generator.generate(sample_dir, agent_a, agent_b)
            progress.update(1)

            progress.set_postfix_str("sessions")
            agent_a, agent_b = self.session_generator.generate(sample_dir, agent_a, agent_b)
            progress.update(1)

            progress.set_postfix_str("assemble")
            sample = self.assembler.build_empty_sample(sample_id, agent_a, agent_b)
            progress.update(1)

            progress.set_postfix_str("annotations")
            self._generate_annotations(sample, agent_a, agent_b, sample_id)
            progress.update(1)

            progress.set_postfix_str("validate")
            self.validator.validate(sample)
            progress.update(1)

            progress.set_postfix_str("save")
            self._save_intermediate_sample(sample_dir, sample, agent_a, agent_b)
            progress.update(1)

        logger.info(
            "Completed %s: sessions=%s qa=%s",
            sample_id,
            len(sample.conversation.sessions),
            len(sample.qa),
        )
        return sample

    def _generate_annotations(self, sample: BenchmarkSample, agent_a, agent_b, sample_id: str) -> None:
        speakers = [agent_a.persona.name, agent_b.persona.name]
        session_count = len(agent_a.sessions)
        logger.info("Generating annotations for %s: sessions=%s", sample_id, session_count)
        with tqdm(total=session_count + 1, desc=f"{sample_id}:annotations", unit="step", leave=False) as progress:
            for session_a, session_b in zip(agent_a.sessions, agent_b.sessions, strict=True):
                progress.set_postfix_str(f"session_{session_a.index}")
                sample.observation[session_a.index] = self.observation_generator.generate(session_a, speakers)
                sample.session_summary[session_a.index] = self.session_summary_generator.generate(session_a)
                sample.event_summary[session_a.index] = self.event_summary_generator.generate(
                    session_a,
                    session_b,
                    agent_a,
                    agent_b,
                )
                progress.update(1)

            progress.set_postfix_str("qa")
            sample.qa = self.qa_generator.generate(sample)
            progress.update(1)

    def _save_intermediate_sample(self, sample_dir: Path, sample: BenchmarkSample, agent_a, agent_b) -> None:
        if self.config.save_intermediate_agents:
            save_agent_state(sample_dir, "agent_a", agent_a)
            save_agent_state(sample_dir, "agent_b", agent_b)
        benchmark_path = sample_dir / "benchmark_sample.json"
        benchmark_path.write_text(json.dumps(sample.to_legacy_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved sample artifact: %s", benchmark_path)

    def _save_dataset(self, samples: list[BenchmarkSample]) -> None:
        payload = [sample.to_legacy_dict() for sample in samples]
        self.config.dataset_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved dataset with %s samples to %s", len(samples), self.config.dataset_file)

    def _configure_logging(self) -> None:
        root_logger = logging.getLogger()
        if root_logger.handlers:
            return
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
