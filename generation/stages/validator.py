from __future__ import annotations

from generation.schemas.benchmark import BenchmarkSample


class BenchmarkValidator:
    def validate(self, sample: BenchmarkSample) -> None:
        dialog_ids = {
            turn.dia_id
            for turns in sample.conversation.sessions.values()
            for turn in turns
        }

        if not sample.conversation.sessions:
            raise ValueError(f"Sample {sample.sample_id} has no generated sessions.")
        if not sample.qa:
            raise ValueError(f"Sample {sample.sample_id} has no QA annotations.")

        for session_index in sample.conversation.sessions:
            if session_index not in sample.observation:
                raise ValueError(f"Sample {sample.sample_id} is missing observation for session {session_index}.")
            if session_index not in sample.session_summary:
                raise ValueError(f"Sample {sample.sample_id} is missing summary for session {session_index}.")
            if session_index not in sample.event_summary:
                raise ValueError(f"Sample {sample.sample_id} is missing event summary for session {session_index}.")

        for qa in sample.qa:
            if not qa.evidence:
                raise ValueError(f"Sample {sample.sample_id} has QA without evidence: {qa.question}")
            for evidence in qa.evidence:
                if evidence not in dialog_ids:
                    raise ValueError(
                        f"Sample {sample.sample_id} has unknown evidence {evidence!r} for question {qa.question!r}."
                    )
