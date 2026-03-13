from __future__ import annotations

from generation.schemas.benchmark import BenchmarkConversation
from generation.schemas.generation import SessionRecord
from generation.services.session_service import parse_session_datetime


def session_date_label(date_time: str) -> str:
    parsed = parse_session_datetime(date_time)
    return parsed.strftime("%d %B, %Y").lstrip("0")


def session_lines(session: SessionRecord) -> list[str]:
    lines = [session.date_time]
    for turn in session.turns:
        lines.append(f"[{turn.dia_id}] {turn.speaker} сказал(а): \"{turn.clean_text or turn.text}\"")
    return lines


def session_transcript(session: SessionRecord) -> str:
    return "\n".join(session_lines(session))


def full_conversation_transcript(conversation: BenchmarkConversation) -> str:
    lines: list[str] = []
    for index in sorted(conversation.sessions):
        lines.append(f"Сессия {index}")
        lines.append(conversation.session_date_times[index])
        for turn in conversation.sessions[index]:
            lines.append(f"[{turn.dia_id}] {turn.speaker} сказал(а): \"{turn.clean_text or turn.text}\"")
        lines.append("")
    return "\n".join(lines).strip()
