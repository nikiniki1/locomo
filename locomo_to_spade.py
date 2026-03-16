#!/usr/bin/env python3
"""Convert locomo benchmark JSON → spade_llm YAML scenario files.

Usage:
    python locomo_to_spade.py data/locomo10.json --out-dir ../locomo_scenarios
    python locomo_to_spade.py benchmark/benchmark.json --out-dir scenarios --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


CATEGORY_MAP: dict[int, str] = {
    1: "factual",
    2: "temporal",
    3: "multi_evidence",
    4: "multi_evidence",
    5: "adversarial",
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_sessions(conversation: dict) -> list[tuple[int, str, list[dict]]]:
    """Extract ordered (session_idx, date_time, turns) from conversation dict."""
    sessions = []
    idx = 1
    while True:
        turns_key = f"session_{idx}"
        if turns_key not in conversation:
            break
        date_time = conversation.get(f"session_{idx}_date_time", "")
        sessions.append((idx, date_time, conversation[turns_key]))
        idx += 1
    return sessions


def sample_to_scenario(sample: dict, sample_index: int) -> dict:
    conv = sample["conversation"]
    speaker_a: str = conv["speaker_a"]
    speaker_b: str = conv["speaker_b"]

    def make_email(name: str) -> str:
        return f"{slugify(name)}@locomo.test"

    sessions = parse_sessions(conv)

    emails = []
    msg_counter = 1

    for session_idx, date_time, turns in sessions:
        subject = f"Session {session_idx} [{date_time}]" if date_time else f"Session {session_idx}"
        session_first_id: str | None = None

        for turn in turns:
            speaker: str = turn["speaker"]
            other = speaker_b if speaker == speaker_a else speaker_a
            body: str = (turn.get("clean_text") or turn.get("text") or "").strip()
            if not body:
                continue

            email_id = f"msg{msg_counter:03d}"
            email_entry: dict = {
                "id": email_id,
                "sender": make_email(speaker),
                "recipient": make_email(other),
                "subject": subject,
                "body": body,
                "delay": 0,
            }
            if session_first_id is not None:
                email_entry["reply_to"] = session_first_id
            else:
                session_first_id = email_id

            emails.append(email_entry)
            msg_counter += 1

    questions = []
    for qa in sample.get("qa", []):
        question: str = qa.get("question", "").strip()
        if not question:
            continue

        # Category 5 (adversarial/unanswerable) uses adversarial_answer
        answer = qa.get("answer") if "answer" in qa else qa.get("adversarial_answer", "I don't know")
        category_int: int = qa.get("category", 1)
        category_str: str = CATEGORY_MAP.get(category_int, "factual")
        evidence: list = qa.get("evidence", [])

        questions.append({
            "question": question,
            "expected_answer": str(answer),
            "ask_after": 1,
            "category": category_str,
            "evidence": evidence,
        })

    name = f"locomo_sample_{sample_index}_{slugify(speaker_a)}_{slugify(speaker_b)}"
    return {
        "name": name,
        "speakers": {"a": speaker_a, "b": speaker_b},
        "emails": emails,
        "questions": questions,
    }


def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Use literal block style (|) for multi-line strings, plain for single-line."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def build_dumper() -> type[yaml.Dumper]:
    dumper = yaml.Dumper
    dumper.add_representer(str, _str_representer)
    return dumper


def convert(input_path: Path, out_dir: Path, max_samples: int | None) -> None:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    samples = data[:max_samples] if max_samples is not None else data
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        scenario = sample_to_scenario(sample, idx)
        out_path = out_dir / f"locomo_scenario_{idx}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(scenario, f, Dumper=build_dumper(), allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
        n_emails = len(scenario["emails"])
        n_questions = len(scenario["questions"])
        print(f"[{idx + 1}/{len(samples)}] {out_path.name}  ({n_emails} emails, {n_questions} questions)")

    print(f"\nDone: {len(samples)} scenario(s) → {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert locomo benchmark JSON → spade_llm YAML scenario files"
    )
    parser.add_argument("input", help="Path to benchmark JSON (locomo10.json or benchmark.json)")
    parser.add_argument("--out-dir", default="locomo_scenarios", help="Output directory (default: locomo_scenarios)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to convert")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.out_dir), args.max_samples)


if __name__ == "__main__":
    main()
