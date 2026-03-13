from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from generation.schemas.generation import AgentState, Event, GenerationConfig, SessionRecord, SessionTurn
from generation.support.llm import LLMClient


SESSION_FIRST_PROMPT_EN = """
Use the following PERSONALITY to write the next line in a conversation.
- Start naturally and warmly.
- Keep replies under 20 words.
- Be personal and specific.
- Mention real-life details, time references, people, and places when relevant.
- Do not mention that you are following a prompt.

PERSONALITY: {persona}

{speaker} is meeting {other} for the first time. Today is {date_time}.
Recent events in {speaker}'s life:
{events}

{stop_instruction}
"""


SESSION_CONT_PROMPT_EN = """
Use the following PERSONALITY to write the next line in a conversation.
- Keep replies under 20 words.
- Be personal and emotionally specific.
- Ask natural follow-up questions when useful.
- Talk only about facts this speaker would reasonably know.
- Do not repeat the same information.

PERSONALITY: {persona}

{speaker} last talked to {other} on {prev_date_time}. Today is {date_time}.
Summary of the previous conversation:
{previous_summary}

Recent events in {speaker}'s life since the last conversation:
{events}

{stop_instruction}
"""


SUMMARY_INIT_PROMPT_EN = """
Write a concise summary containing key facts mentioned about {speaker_1} and {speaker_2} on {date_time} in the following conversation:

{conversation}
"""


SUMMARY_CONT_PROMPT_EN = """
Previous conversations between {speaker_1} and {speaker_2} can be summarized as follows:
{previous_summary}

The current conversation on {date_time} is:
{conversation}

Summarize the previous and current conversations between {speaker_1} and {speaker_2} in 150 words or less. Include key facts and time references.
"""


CASUAL_DIALOG_PROMPT_EN = """
Make the sentence short, natural, and casual.

Input: {text}
Output:
"""

SESSION_FIRST_PROMPT_RU = """
Используй следующую ЛИЧНОСТЬ, чтобы написать следующую реплику в разговоре.
- Начинай естественно и дружелюбно.
- Держи реплики короче 20 слов.
- Пиши лично и конкретно.
- При необходимости упоминай детали реальной жизни, время, людей и места.
- Не говори, что следуешь инструкции.
- Пиши по-русски.

ЛИЧНОСТЬ: {persona}

{speaker} впервые встречается с {other}. Сегодня {date_time}.
Недавние события в жизни {speaker}:
{events}

{stop_instruction}
"""


SESSION_CONT_PROMPT_RU = """
Используй следующую ЛИЧНОСТЬ, чтобы написать следующую реплику в разговоре.
- Держи реплики короче 20 слов.
- Пиши лично и эмоционально конкретно.
- При необходимости задавай естественные уточняющие вопросы.
- Говори только о том, что этот персонаж разумно мог бы знать.
- Не повторяй одну и ту же информацию.
- Пиши по-русски.

ЛИЧНОСТЬ: {persona}

{speaker} в последний раз разговаривал(а) с {other} {prev_date_time}. Сегодня {date_time}.
Краткое содержание предыдущего разговора:
{previous_summary}

Недавние события в жизни {speaker} после прошлого разговора:
{events}

{stop_instruction}
"""


SUMMARY_INIT_PROMPT_RU = """
Напиши краткое резюме с ключевыми фактами о {speaker_1} и {speaker_2}, которые были упомянуты {date_time} в следующем разговоре. Пиши по-русски:

{conversation}
"""


SUMMARY_CONT_PROMPT_RU = """
Предыдущие разговоры между {speaker_1} и {speaker_2} можно кратко описать так:
{previous_summary}

Текущий разговор {date_time}:
{conversation}

Суммируй предыдущие и текущий разговоры между {speaker_1} и {speaker_2} не более чем в 150 словах. Включай ключевые факты и временные указания. Пиши по-русски.
"""


CASUAL_DIALOG_PROMPT_RU = """
Сделай фразу короткой, естественной и разговорной.

Вход: {text}
Выход:
"""


def parse_human_date(date_str: str) -> datetime:
    date_formats = (
        "%d %B, %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%B %d %Y",
    )
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {date_str}")


def parse_session_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")


def format_session_datetime(value: datetime) -> str:
    return value.strftime("%I:%M %p on %d %B, %Y").lower()


def random_time() -> timedelta:
    start_seconds = 9 * 3600
    end_seconds = (21 * 3600) + (59 * 60) + 59
    random_seconds = random.randint(start_seconds, end_seconds)
    return timedelta(seconds=random_seconds)


def sort_events(events: list[Event]) -> list[Event]:
    return sorted(events, key=lambda event: parse_human_date(event.date))


def relevant_events(events: list[Event], current_date: datetime, previous_date: datetime | None) -> list[Event]:
    relevant: list[Event] = []
    for event in sort_events(events):
        event_date = parse_human_date(event.date)
        if event_date > current_date:
            continue
        if previous_date is not None and event_date <= previous_date:
            continue
        relevant.append(event)
    return relevant


def next_session_date(
    agent_a_events: list[Event],
    agent_b_events: list[Event],
    *,
    previous_date: datetime | None,
    num_events_per_session: int,
) -> datetime:
    def cutoff(events: list[Event]) -> datetime:
        count = 0
        selected = sort_events(events)
        last_seen = parse_human_date(selected[-1].date)
        for event in selected:
            event_date = parse_human_date(event.date)
            if previous_date is not None and event_date < previous_date:
                continue
            count += 1
            last_seen = event_date
            if count == num_events_per_session:
                return last_seen
        return last_seen

    return min(cutoff(agent_a_events), cutoff(agent_b_events)) + timedelta(days=random.choice([1, 2]))


def event_string(session_events: list[Event], all_events: list[Event]) -> str:
    event_by_id = {event.id: event for event in all_events}
    parts: list[str] = []
    for event in session_events:
        text = f"On {event.date}, {event.sub_event}"
        if event.caused_by:
            caused = ", ".join(
                f"{event_by_id[event_id].sub_event} ({event_by_id[event_id].date})"
                for event_id in event.caused_by
                if event_id in event_by_id
            )
            if caused:
                text += f". This followed from: {caused}"
        parts.append(text)
    return "\n".join(parts) if parts else "No major new events."


def event_string_ru(session_events: list[Event], all_events: list[Event]) -> str:
    event_by_id = {event.id: event for event in all_events}
    parts: list[str] = []
    for event in session_events:
        text = f"{event.date}: {event.sub_event}"
        if event.caused_by:
            caused = ", ".join(
                f"{event_by_id[event_id].sub_event} ({event_by_id[event_id].date})"
                for event_id in event.caused_by
                if event_id in event_by_id
            )
            if caused:
                text += f". Это произошло вследствие: {caused}"
        parts.append(text)
    return "\n".join(parts) if parts else "Нет значимых новых событий."


def build_prompt(
    config: GenerationConfig,
    agent: AgentState,
    other: AgentState,
    session_index: int,
    date_time: str,
    session_events: list[Event],
    previous_summary: str | None,
    previous_date_time: str | None,
    *,
    instruct_stop: bool,
) -> str:
    is_ru = config.language.lower().startswith("ru")
    stop_instruction = (
        "Чтобы завершить разговор, напиши [END] в конце реплики." if instruct_stop and is_ru
        else "To end the conversation, write [END] at the end of the line." if instruct_stop
        else ""
    )
    events_text = event_string_ru(session_events, agent.events) if is_ru else event_string(session_events, agent.events)
    if session_index == 1 or previous_summary is None or previous_date_time is None:
        prompt = SESSION_FIRST_PROMPT_RU if is_ru else SESSION_FIRST_PROMPT_EN
        return prompt.format(
            persona=agent.persona.persona,
            speaker=agent.persona.name,
            other=other.persona.name,
            date_time=date_time,
            events=events_text,
            stop_instruction=stop_instruction,
        )
    prompt = SESSION_CONT_PROMPT_RU if is_ru else SESSION_CONT_PROMPT_EN
    return prompt.format(
        persona=agent.persona.persona,
        speaker=agent.persona.name,
        other=other.persona.name,
        prev_date_time=previous_date_time,
        date_time=date_time,
        previous_summary=previous_summary,
        events=events_text,
        stop_instruction=stop_instruction,
    )


def clean_dialog_line(raw_text: str, speaker_name: str) -> str:
    text = raw_text.strip().split("\n")[0]
    if text.startswith(speaker_name):
        text = text[len(speaker_name):].lstrip(": ").strip()
    return text


def casualize(llm: LLMClient, text: str, language: str = "ru") -> str:
    if not text.strip():
        return ""
    if "[END]" in text:
        return text.replace("[END]", "").strip() + " [END]"
    prompt = CASUAL_DIALOG_PROMPT_RU if language.lower().startswith("ru") else CASUAL_DIALOG_PROMPT_EN
    return llm.complete_text(prompt.format(text=text), max_tokens=80, temperature=0.7).strip()


def generate_session(
    llm: LLMClient,
    config: GenerationConfig,
    *,
    agent_a: AgentState,
    agent_b: AgentState,
    session_index: int,
) -> tuple[SessionRecord, SessionRecord] | None:
    previous_session = agent_a.sessions[-1] if agent_a.sessions else None
    previous_date = parse_session_datetime(previous_session.date_time) if previous_session else None
    previous_date_time = previous_session.date_time if previous_session else None
    previous_summary = previous_session.summary if previous_session else None

    current_date = next_session_date(
        agent_a.events,
        agent_b.events,
        previous_date=previous_date,
        num_events_per_session=config.num_events_per_session,
    )
    current_date_time = current_date + random_time()
    current_date_time_str = format_session_datetime(current_date_time)

    session_events_a = relevant_events(agent_a.events, current_date_time, previous_date)
    session_events_b = relevant_events(agent_b.events, current_date_time, previous_date)
    if not session_events_a and not session_events_b:
        return None

    turns: list[SessionTurn] = []
    current_speaker = 0 if random.random() < 0.5 else 1
    transcript = f"{agent_a.persona.name if current_speaker == 0 else agent_b.persona.name}: "
    stop_after = (
        config.max_turns_per_session
        if config.max_turns_per_session <= 10
        else random.choice(range(10, config.max_turns_per_session + 1))
    )
    break_after = {"a": False, "b": False}

    for turn_index in range(config.max_turns_per_session):
        if break_after["a"] and break_after["b"]:
            break

        speaker = agent_a if current_speaker == 0 else agent_b
        other = agent_b if current_speaker == 0 else agent_a
        session_events = session_events_a if current_speaker == 0 else session_events_b
        prompt = build_prompt(
            config,
            speaker,
            other,
            session_index,
            current_date_time_str,
            session_events,
            previous_summary,
            previous_date_time,
            instruct_stop=turn_index >= stop_after,
        )
        conversation_label = "\nДИАЛОГ:\n\n" if config.language.lower().startswith("ru") else "\nCONVERSATION:\n\n"
        raw = llm.complete_text(prompt + conversation_label + transcript, max_tokens=100, temperature=1.0)
        text = clean_dialog_line(raw, speaker.persona.name)
        clean_text = casualize(llm, text, config.language)
        turn = SessionTurn(
            dia_id=f"D{session_index}:{turn_index + 1}",
            speaker=speaker.persona.name,
            text=text,
            clean_text=clean_text.replace("[END]", "").strip(),
        )
        turns.append(turn)
        if text.endswith("[END]") or clean_text.endswith("[END]"):
            break_after["a" if current_speaker == 0 else "b"] = True

        transcript += turn.clean_text + "\n\n" + f"{other.persona.name}: "
        current_speaker = int(not current_speaker)

    return SessionRecord(
        index=session_index,
        date_time=current_date_time_str,
        events=session_events_a,
        turns=turns,
        summary=None,
    ), SessionRecord(
        index=session_index,
        date_time=current_date_time_str,
        events=session_events_b,
        turns=turns,
        summary=None,
    )


def generate_summary(
    llm: LLMClient,
    *,
    session: SessionRecord,
    speaker_1: AgentState,
    speaker_2: AgentState,
    previous_summary: str | None,
) -> str:
    conversation = "\n".join(f"{turn.speaker}: {turn.text}" for turn in session.turns)
    use_ru = any("\u0400" <= char <= "\u04FF" for char in (previous_summary or conversation or ""))
    if previous_summary:
        prompt_template = SUMMARY_CONT_PROMPT_RU if use_ru else SUMMARY_CONT_PROMPT_EN
        prompt = prompt_template.format(
            speaker_1=speaker_1.persona.name,
            speaker_2=speaker_2.persona.name,
            previous_summary=previous_summary,
            date_time=session.date_time,
            conversation=conversation,
        )
    else:
        prompt_template = SUMMARY_INIT_PROMPT_RU if use_ru else SUMMARY_INIT_PROMPT_EN
        prompt = prompt_template.format(
            speaker_1=speaker_1.persona.name,
            speaker_2=speaker_2.persona.name,
            date_time=session.date_time,
            conversation=conversation,
        )
    return llm.complete_text(prompt, max_tokens=180, temperature=0.5).strip()
