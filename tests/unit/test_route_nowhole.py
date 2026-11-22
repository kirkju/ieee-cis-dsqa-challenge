"""dones-nowhole: a question on IFMIF-DONES answered the entry's own way,
told only that no document is to be placed whole in its context; every
other question exactly as without it."""

import hashlib
import json

import pymupdf
import pytest
from dsqa import config
from dsqa.answering import Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf

DONES_QUESTION = "Where is the lithium loop of the IFMIF-DONES plant?"
AI_QUESTION = "Which learning rate does the neural transformer use?"


class Engine:
    def __init__(self, reply="Final answer: In the dump tank."):
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
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 400)
    return tmp_path


def asked(corpus, names, question, monkeypatch):
    """What the engine was shown for a question, and the share a whole
    document could take in the context it was given."""
    monkeypatch.setenv("DSQA_VARIANT", names)
    engine, said = Engine(), []
    answerer = Answerer(corpus, engine, embedder=Letters(), say=said.append)
    given = []
    context = answerer.index.context

    def recorded(*arguments, **options):
        given.append(options.get("whole"))
        return context(*arguments, **options)

    monkeypatch.setattr(answerer.index, "context", recorded)
    reply = answerer.answer(question)
    return reply, engine.conversations, given, said


def fingerprint(conversations) -> str:
    return hashlib.sha256(json.dumps(conversations, sort_keys=True).encode()).hexdigest()


def test_a_dones_question_is_answered_the_entrys_way_with_no_document_whole(corpus, monkeypatch):
    plain_reply, plain_shown, plain_given, _ = asked(corpus, "", DONES_QUESTION, monkeypatch)
    reply, shown, given, said = asked(corpus, "dones-nowhole", DONES_QUESTION, monkeypatch)
    assert (plain_given, given) == ([None], [0.0])
    assert reply == plain_reply == "In the dump tank."
    # The same instructions and the same turn around the passages: only which
    # passages are placed may differ.
    assert shown[0][0] == plain_shown[0][0]
    assert shown[0][1]["content"].startswith("Passages retrieved from the document collection:")
    assert any("the dones-nowhole way, none whole" in line for line in said)


def test_a_question_on_ai_papers_is_asked_exactly_as_without_it(corpus, monkeypatch):
    _, plain_shown, plain_given, _ = asked(corpus, "", AI_QUESTION, monkeypatch)
    _, shown, given, said = asked(corpus, "dones-nowhole", AI_QUESTION, monkeypatch)
    assert given == plain_given == [None]
    assert fingerprint(shown) == fingerprint(plain_shown)
    assert not any("the dones-nowhole way" in line for line in said)


def test_the_share_is_told_afresh_for_every_question(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-nowhole")
    answerer = Answerer(corpus, Engine(), embedder=Letters(), say=lambda line: None)
    answerer.answer(DONES_QUESTION)
    assert answerer.whole == 0.0
    answerer.answer(AI_QUESTION)
    assert answerer.whole is None
