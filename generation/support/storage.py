from __future__ import annotations

import json
from pathlib import Path

from generation.schemas.generation import AgentState, Event, SessionRecord, SessionTurn


def ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def agent_file(out_dir: Path, agent_key: str) -> Path:
    return out_dir / f"{agent_key}.json"


def save_agent_state(out_dir: Path, agent_key: str, agent_state: AgentState) -> None:
    ensure_output_dir(out_dir)
    agent_file(out_dir, agent_key).write_text(
        json.dumps(agent_state.to_legacy_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_legacy_agent(out_dir: Path, agent_key: str) -> dict[str, object] | None:
    path = agent_file(out_dir, agent_key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_agent_state(out_dir: Path, agent_key: str) -> AgentState | None:
    legacy = load_legacy_agent(out_dir, agent_key)
    if legacy is None:
        return None

    session_indices = sorted(
        {
            int(key.split("_")[1])
            for key in legacy
            if key.startswith("session_") and key.endswith("_date_time")
        }
    )
    sessions: list[SessionRecord] = []
    for index in session_indices:
        turns = [
            SessionTurn.model_validate(turn)
            for turn in legacy.get(f"session_{index}", [])
        ]
        events = [
            Event.model_validate(event)
            for event in legacy.get(f"events_session_{index}", [])
        ]
        sessions.append(
            SessionRecord(
                index=index,
                date_time=legacy[f"session_{index}_date_time"],
                turns=turns,
                events=events,
                summary=legacy.get(f"session_{index}_summary"),
            )
        )

    return AgentState.model_validate(
        {
            "persona": {
                "name": legacy["name"],
                "persona": legacy["persona_summary"],
                "msc_prompt": legacy.get("msc_prompt"),
            },
            "events": legacy.get("graph", []),
            "sessions": sessions,
        }
    )
