"""llm-proof (route.py): every question answered as the entry answers it, but
that a question on IFMIF-DONES gets the fixed sentence for as long as no
question of the run has had its ranking scored in full. With the candidate's
adjustments (the stack of 2026-11-15, llm-one and llm-proof) and an engine
that scores, every conversation is the stack's, by hash; with one that
cannot score, or whose scores rank nothing, the questions on AI papers are
still the stack's and those on IFMIF-DONES get the sentence; and once one
question was scored in full, a later failure is answered as the stack
answers it. The engine is the scripted one of test_route_relevance.py, with
first() beside score()."""

import types

import pymupdf
import pytest
from dsqa import config, route
from dsqa.answering import Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf
from tests.unit.test_route import AI_QUESTION, DONES_QUESTION, Engine, fingerprint
from tests.unit.test_route_relevance import FAILURES, REPLY, STACK, Scoring

CANDIDATE = STACK + ",llm-one,llm-proof"
OTHER_DONES = "How is the lithium of the IFMIF-DONES loop carried?"
QUESTIONS = [AI_QUESTION, DONES_QUESTION, OTHER_DONES]


class OneByOne(Scoring):
    """The scripted engine with first(), which gives for one conversation
    what score() gives it; and scoring that fails for the questions named in
    `failing`, as if the GPU ran out of memory."""

    def __init__(self, failing=(), **kwargs):
        super().__init__(**kwargs)
        self.failing = set(failing)

    def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
        asked = [conversation[1]["content"] for conversation in conversations]
        if any(f"Question: {question}\n" in text for question in self.failing for text in asked):
            self.scored.extend(conversations)
            return None
        return super().score(conversations, labels, most_tokens)

    def first(self, conversation, labels=("Yes", "No")):
        rows = self.score([conversation], labels)
        return None if rows is None else rows[0]


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


def asked(corpus, monkeypatch, variants, engine, questions=QUESTIONS):
    """The replies to the questions, in one run, and what was said."""
    monkeypatch.setenv("DSQA_VARIANT", variants)
    said = []
    built = Answerer(corpus, engine, embedder=Letters(), say=said.append)
    return [built.answer(question) for question in questions], said


def proof_lines(said):
    return [line for line in said if ": llm-proof: " in line]


def test_proving_is_off_without_it_and_with_llm_acted(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", STACK)
    assert not route.proving()
    monkeypatch.setenv("DSQA_VARIANT", CANDIDATE)
    assert route.proving()
    monkeypatch.setenv("DSQA_VARIANT", CANDIDATE + ",llm-acted")
    assert not route.proving() and route.measuring()
    monkeypatch.setenv("DSQA_VARIANT", "no-such-adjustment")
    assert not route.proving()


def test_with_a_scoring_engine_every_question_is_asked_as_the_stack_asks_it(corpus, monkeypatch):
    stack = OneByOne(reply=REPLY)
    replies, _ = asked(corpus, monkeypatch, STACK, stack)
    engine = OneByOne(reply=REPLY)
    mine, said = asked(corpus, monkeypatch, CANDIDATE, engine)
    assert mine == replies and route.UNANSWERED not in mine
    assert fingerprint(engine.conversations) == fingerprint(stack.conversations)
    assert engine.scored and fingerprint(engine.scored) == fingerprint(stack.scored)
    assert proof_lines(said) == [
        "[dsqa] question 2: llm-proof: scored in full, the first: answered",
        "[dsqa] question 3: llm-proof: scored in full, answered",
    ]
    assert sum(" one by one in " in line for line in said) == 2


@pytest.mark.parametrize("how", ["cannot score", *sorted(FAILURES)])
def test_without_scores_that_rank_dones_gets_the_sentence_and_ai_is_the_stack(
    corpus, monkeypatch, how
):
    def engine():
        if how == "cannot score":
            return Engine(reply=REPLY)
        failing, _ = FAILURES[how]
        return OneByOne(reply=REPLY, **failing)

    stack = engine()
    replies, _ = asked(corpus, monkeypatch, STACK, stack, [AI_QUESTION])
    mine = engine()
    got, said = asked(corpus, monkeypatch, CANDIDATE, mine)
    assert got == [replies[0], route.UNANSWERED, route.UNANSWERED]
    # The question on AI papers asked as the stack asks it, and nothing more.
    assert fingerprint(mine.conversations) == fingerprint(stack.conversations)
    lines = proof_lines(said)
    assert len(lines) == 2 and all(
        line.endswith(", and never yet, the fixed sentence") for line in lines
    )
    why = "the engine cannot score one by one" if how == "cannot score" else FAILURES[how][1]
    assert f"llm-proof: not scored ({why}" in lines[0]


def test_once_proved_a_later_failure_is_answered_as_the_stack_answers_it(corpus, monkeypatch):
    # The stack, with the second question on IFMIF-DONES not scored: the
    # ranking as it was, answered the dones-plain way.
    stack = OneByOne(reply=REPLY, failing=[OTHER_DONES])
    replies, _ = asked(corpus, monkeypatch, STACK, stack)
    engine = OneByOne(reply=REPLY, failing=[OTHER_DONES])
    mine, said = asked(corpus, monkeypatch, CANDIDATE, engine)
    assert mine == replies and route.UNANSWERED not in mine
    assert fingerprint(engine.conversations) == fingerprint(stack.conversations)
    assert proof_lines(said) == [
        "[dsqa] question 2: llm-proof: scored in full, the first: answered",
        "[dsqa] question 3: llm-proof: not scored (no scores came back, out of memory), "
        "but proved before: answered",
    ]


def test_before_the_first_success_the_sentence_and_from_it_on_the_stack(corpus, monkeypatch):
    order = [DONES_QUESTION, OTHER_DONES, DONES_QUESTION, AI_QUESTION]
    engine = OneByOne(reply=REPLY, failing=[DONES_QUESTION])
    mine, said = asked(corpus, monkeypatch, CANDIDATE, engine, order)
    # The first fails before any success; the second proves; the third fails
    # again and is answered over the ranking as it was; AI as always.
    assert mine[0] == route.UNANSWERED
    assert route.UNANSWERED not in mine[1:]
    assert [line.split(": llm-proof: ")[1] for line in proof_lines(said)] == [
        "not scored (no scores came back, out of memory), and never yet, the fixed sentence",
        "scored in full, the first: answered",
        "not scored (no scores came back, out of memory), but proved before: answered",
    ]


def test_a_scoring_cut_for_time_counts_and_is_answered_over_what_it_scored(corpus, monkeypatch):
    # 20 s a slice of two, and 30 s for the scoring: two chunks scored.
    monkeypatch.setattr(config, "RELEVANCE_SLICE", 2)
    clock = types.SimpleNamespace(now=0.0)

    def apart(passage):
        given.append(passage)
        return [len(given) - 1.5, 0.0]

    def stepped(**kwargs):
        return OneByOne(reply=REPLY, clock=clock, step=20.0, gives=apart, **kwargs)

    given = []
    stack = stepped()
    monkeypatch.setenv("DSQA_VARIANT", STACK)
    said_stack = []
    built = Answerer(
        corpus,
        stack,
        embedder=Letters(),
        say=said_stack.append,
        clock=lambda: clock.now,
        started=0.0,
    )
    replies = [built.answer(DONES_QUESTION)]
    given.clear()
    clock.now = 0.0
    engine = stepped()
    monkeypatch.setenv("DSQA_VARIANT", CANDIDATE)
    said = []
    built = Answerer(
        corpus, engine, embedder=Letters(), say=said.append, clock=lambda: clock.now, started=0.0
    )
    assert [built.answer(DONES_QUESTION)] == replies != [route.UNANSWERED]
    assert len(engine.scored) == len(stack.scored) == 2
    assert fingerprint(engine.conversations) == fingerprint(stack.conversations)
    (line,) = proof_lines(said)
    assert line.endswith(": llm-proof: scored 2 of 15, stopped for time, the first: answered")
