from __future__ import annotations

import argparse

from generation.config import BenchmarkConfig, BenchmarkPaths
from generation.pipeline import GenerateBenchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt-dir", default="prompt_examples")
    parser.add_argument("--msc-personas-file", default="data/msc_personas_all.json")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-days", type=int, default=240)
    parser.add_argument("--num-events", type=int, default=15)
    parser.add_argument("--start-session", type=int, default=1)
    parser.add_argument("--num-sessions", type=int, default=20)
    parser.add_argument("--max-turns-per-session", type=int, default=20)
    parser.add_argument("--num-events-per-session", type=int, default=1)
    parser.add_argument("--qa-per-sample", type=int, default=24)
    parser.add_argument("--overwrite-persona", action="store_true")
    parser.add_argument("--overwrite-events", action="store_true")
    parser.add_argument("--overwrite-session", action="store_true")
    parser.add_argument("--overwrite-annotations", action="store_true")
    parser.add_argument("--mark-persona-as-used", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error (default: False)")
    parser.add_argument("--no-save-intermediate", action="store_true", help="Skip saving per-sample agent state files")
    parser.add_argument("--no-structured-output", action="store_true",
                        help="Disable structured output (use plain text fallback). Required for GigaChat.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        paths=BenchmarkPaths(
            out_dir=args.out_dir,
            prompt_dir=args.prompt_dir,
            msc_personas_file=args.msc_personas_file,
            dataset_file=args.dataset_file,
        ),
        model=args.model,
        language=args.language,
        num_samples=args.num_samples,
        num_days=args.num_days,
        num_events=args.num_events,
        start_session=args.start_session,
        num_sessions=args.num_sessions,
        max_turns_per_session=args.max_turns_per_session,
        num_events_per_session=args.num_events_per_session,
        qa_per_sample=args.qa_per_sample,
        overwrite_persona=args.overwrite_persona,
        overwrite_events=args.overwrite_events,
        overwrite_session=args.overwrite_session,
        overwrite_annotations=args.overwrite_annotations,
        mark_persona_as_used=args.mark_persona_as_used,
        fail_fast=args.fail_fast,
        save_intermediate_agents=not args.no_save_intermediate,
        no_structured_output=args.no_structured_output,
    )
    pipeline = GenerateBenchmark(config)
    pipeline.run()


if __name__ == "__main__":
    main()
