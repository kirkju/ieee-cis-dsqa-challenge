"""plain on the first entry: every question asked with the organizers'
instruction and turn over this entry's context, the reply returned as the
model wrote it; with dones-plain too, the questions on IFMIF-DONES exactly
as dones-plain alone asks them."""

import hashlib
import json

import pymupdf
import pytest
from dsqa import config, plain, prompt
from dsqa.answering import Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf

DONES_QUESTION = "Where is the lithium loop of the IFMIF-DONES plant?"
AI_QUESTION = "Which learning rate does the neural transformer use?"
CONTEXT = "Document: Notes on the nitrogen trap\n\nThe trap is in the dump tank."


class Engine:
    def __init__(self, reply="It works off-line.\nFinal answer: In the dump tank."):
        self.reply, self.conversations = reply, []

    def count(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, max_seconds=None):
        self.conversations.append(messages)
        return self.reply

    def release(self):
        pass


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    paper_pdf(tmp_path / "b-loop.pdf", BODY)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        document.save(tmp_path / "a-trap.pdf")
    monkeypatch.setattr(config, "CHUNK_TOKENS", 24)
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 80)
    return tmp_path


def ask(corpus, names, question, monkeypatch, reply=None):
    monkeypatch.setenv("DSQA_VARIANT", names)
    engine = Engine() if reply is None else Engine(reply)
    answered = Answerer(corpus, engine, embedder=Letters(), say=lambda line: None).answer(question)
    return answered, engine.conversations


def fingerprint(conversations) -> str:
    return hashlib.sha256(json.dumps(conversations, sort_keys=True).encode()).hexdigest()


def test_the_turn_is_the_rag_examples_over_this_entrys_context(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "plain")
    system, user = prompt.messages(" Where is the trap? ", CONTEXT)
    assert system == {"role": "system", "content": plain.SYSTEM_PROMPT}
    assert user["content"] == (
        f"{plain.CONTEXT_OPENING}\n\n{CONTEXT}\n\n---\n\nWhere is the trap?"
    )
    assert prompt.messages("Where is the trap?")[1]["content"] == "Where is the trap?"
    monkeypatch.setenv("DSQA_VARIANT", "")
    assert prompt.messages("Where is the trap?", CONTEXT)[0]["content"] == prompt.SYSTEM


@pytest.mark.parametrize("reply", ["It works off-line, in the dump tank.", "In the tank."])
def test_the_reply_is_returned_as_it_is_with_nothing_asked_apart(corpus, monkeypatch, reply):
    answered, shown = ask(corpus, "plain", AI_QUESTION, monkeypatch, reply=f" {reply} ")
    assert answered == reply
    (conversation,) = shown  # no answer asked for apart
    assert conversation[0]["content"] == plain.SYSTEM_PROMPT
    assert conversation[1]["content"].startswith(plain.CONTEXT_OPENING)
    assert conversation[1]["content"].endswith(f"---\n\n{AI_QUESTION}")
    # A line "Final answer:" is not looked for: what was written is returned.
    answered, _ = ask(corpus, "plain", AI_QUESTION, monkeypatch)
    assert answered == "It works off-line.\nFinal answer: In the dump tank."


def test_with_dones_plain_the_dones_questions_are_asked_as_dones_plain_alone(
    corpus, monkeypatch
):
    _, alone = ask(corpus, "dones-plain", DONES_QUESTION, monkeypatch)
    _, together = ask(corpus, "dones-plain,plain", DONES_QUESTION, monkeypatch)
    assert fingerprint(together) == fingerprint(alone)
    # And the questions on AI papers with the examples' instruction over our context.
    _, shown = ask(corpus, "dones-plain,plain", AI_QUESTION, monkeypatch)
    assert shown[0][0]["content"] == plain.SYSTEM_PROMPT
    assert "Document: " in shown[0][1]["content"]
