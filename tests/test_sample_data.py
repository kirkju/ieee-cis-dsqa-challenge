"""Shape of the organizers' sample question set.

The local runner and the evaluation tools read sample/sample_qa.json. If an
upstream update adds a field, a domain or a question type, it should be
noticed here rather than as questions silently dropped from a score.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = json.loads((ROOT / "sample" / "sample_qa.json").read_text(encoding="utf-8"))

FIELDS = {"id", "domain", "question", "question_type", "answer"}
DOMAINS = {"DONES", "AI Papers"}
QUESTION_TYPES = {"factual", "comparative", "procedural", "causal", "predictive"}


def test_every_item_has_exactly_the_known_fields():
    unexpected = [item.get("id") for item in SAMPLE if set(item) != FIELDS]
    assert not unexpected


def test_ids_are_unique():
    ids = [item["id"] for item in SAMPLE]
    assert len(ids) == len(set(ids))


def test_questions_and_answers_are_non_empty_text():
    for item in SAMPLE:
        assert isinstance(item["question"], str) and item["question"].strip(), item["id"]
        assert isinstance(item["answer"], str) and item["answer"].strip(), item["id"]


def test_domains_are_the_two_scored_ones():
    # The ranking metric is accuracy balanced across exactly these two.
    assert {item["domain"] for item in SAMPLE} == DOMAINS


def test_question_types_are_known():
    assert {item["question_type"] for item in SAMPLE} <= QUESTION_TYPES
