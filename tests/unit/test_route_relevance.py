"""The base model as a judge of relevance in the dones-plain way (route.py,
judged_order(), and relevance.py): it acts on a question answered that way
alone, and only with an llm- adjustment chosen; the log says, question by
question, what it did or why it did not; and nothing that goes wrong in it
keeps the question from being answered the dones-plain way. A question on
AI papers is asked exactly what it is asked without it. The engine is the
scripted one of test_route.py, with a score() of its own."""

import math
import types

import pymupdf
import pytest
from dsqa import config, plain, relevance, route
from dsqa.answering import Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf
from tests.unit.test_route import AI_QUESTION, DONES_QUESTION, Budget, Engine, fingerprint

REPLY = "It is in the plant."


class Scoring(Engine):
    """The scripted engine with a score() that likes the chunks of the
    nitrogen trap, which the ranking puts below those of the loop. It may
    raise, give nothing back, or give each passage the logits `gives` makes
    of it, and each call may take seconds of a clock."""

    def __init__(
        self, fails_with=None, gives_none=False, clock=None, step=0.0, gives=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.fails_with, self.gives_none, self.gives = fails_with, gives_none, gives
        self.clock, self.step = clock, step
        self.scored: list[list[dict]] = []

    def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
        self.scored.extend(conversations)
        if self.clock is not None:
            self.clock.now += self.step
        if self.fails_with is not None:
            raise self.fails_with
        if self.gives_none:
            return None
        rows = []
        for conversation in conversations:
            passage = conversation[1]["content"].split("\n\nQuestion:")[0]
            if self.gives is not None:
                rows.append(self.gives(passage))
            else:
                rows.append([4.0, 0.0] if "trap" in passage else [-4.0, 0.0])
        return rows


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


def answerer(corpus, engine, said=None, clock=None):
    say = said.append if said is not None else (lambda line: None)
    extra = {} if clock is None else {"clock": clock, "started": 0.0}
    return Answerer(corpus, engine, embedder=Letters(), say=say, **extra)


def relevance_lines(said):
    return [line for line in said if "relevance:" in line]


def test_without_an_llm_adjustment_a_dones_plain_question_is_asked_as_it_was(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain")
    before = Engine(reply=REPLY)
    answerer(corpus, before).answer(DONES_QUESTION)
    engine, said = Scoring(reply=REPLY), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == REPLY
    assert fingerprint(engine.conversations) == fingerprint(before.conversations)
    assert [limits[0] for limits in engine.limits] == [plain.MAX_NEW_TOKENS]
    assert engine.scored == []
    (line,) = relevance_lines(said)
    assert line.startswith(
        "[dsqa] question 1: relevance: not applied (no llm- adjustment chosen), context of "
    )
    assert line.endswith(" fragments")


def test_a_question_on_ai_papers_is_asked_exactly_as_without_the_filter(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "")
    before = Engine()
    reply = answerer(corpus, before).answer(AI_QUESTION)
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain,llm-rank,llm-gate,llm-wide")
    engine, said = Scoring(), []
    assert answerer(corpus, engine, said).answer(AI_QUESTION) == reply
    assert fingerprint(engine.conversations) == fingerprint(before.conversations)
    assert engine.scored == [] and relevance_lines(said) == []


def test_llm_rank_moves_up_what_the_model_likes_and_says_so(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain,llm-rank")
    engine, said = Scoring(reply=REPLY), []
    built = answerer(corpus, engine, said)
    order = built.index.rank(DONES_QUESTION)
    chunks = built.index.chunks
    trap = [place for place, number in enumerate(order) if "trap" in chunks[int(number)]]
    assert trap and trap[0] > 1  # the ranking puts the trap below the loop
    assert built.answer(DONES_QUESTION) == REPLY
    n = min(len(order), config.RELEVANCE_N)
    assert len(engine.scored) == n
    expected, _ = relevance.judged(Scoring(), DONES_QUESTION, built.index, order)
    moved = [place for place, number in enumerate(expected) if "trap" in chunks[int(number)]]
    assert moved[0] < trap[0]
    ((_, user),) = engine.conversations
    fragments = plain.fragments_of(expected, chunks, engine.count)
    assert user == plain.rag_messages(DONES_QUESTION, fragments)[1]
    (line,) = relevance_lines(said)
    assert line.startswith(f"[dsqa] question 1: relevance: llm-rank scored {n} of {n} in ")
    assert ", first document " in line
    assert line.endswith(
        f", context of {sum(map(engine.count, fragments))} tokens in {len(fragments)} fragments"
    )
    assert any("answered the dones-plain way" in line for line in said)


def test_the_gate_the_wide_scoring_and_the_tokens_are_said(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain,llm-rank,llm-gate,llm-wide")
    engine, said = Scoring(reply=REPLY), []
    built = answerer(corpus, engine, said)
    order = built.index.rank(DONES_QUESTION)
    built.answer(DONES_QUESTION)
    n = min(len(order), config.RELEVANCE_WIDE_N)
    assert len(engine.scored) == n
    ((_, user),) = engine.conversations
    shown = user["content"].count("[Fragment ")
    # What the model turns down is out, but for the best RELEVANCE_KEEP.
    trap = sum(1 for number in order[:n] if "trap" in built.index.chunks[int(number)])
    kept = max(trap, min(config.RELEVANCE_KEEP, n))
    assert shown == kept < len(order)
    (line,) = relevance_lines(said)
    assert f"relevance: llm-rank+llm-gate scored {n} of {n} in " in line
    assert f", the gate kept {kept} of {n}, first document " in line
    assert line.endswith(f" tokens in {shown} fragments")


def test_an_engine_that_cannot_score_leaves_the_dones_plain_question_as_it_was(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain")
    before = Engine(reply=REPLY)
    answerer(corpus, before).answer(DONES_QUESTION)
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain,llm-rank,llm-gate")
    engine, said = Engine(reply=REPLY), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == REPLY
    assert fingerprint(engine.conversations) == fingerprint(before.conversations)
    (line,) = relevance_lines(said)
    assert "relevance: not applied (the engine cannot score), context of " in line


FAILURES = {
    "raises": ({"fails_with": ValueError("a template it could not read")}, "the scoring failed: "),
    "raises out of memory": (
        {"fails_with": RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")},
        "the scoring failed: RuntimeError: CUDA out of memory",
    ),
    "gives nothing": ({"gives_none": True}, "no scores came back, out of memory"),
    # Scores that come back and rank nothing: the trap would go first with
    # real ones, so a ranking left as it was is the guard's doing.
    "gives no numbers": ({"gives": lambda passage: [math.nan, 0.0]}, "the scores rank nothing: "),
    "gives one no number": (
        {"gives": lambda passage: [math.nan, 0.0] if "trap" in passage else [-4.0, 0.0]},
        "the scores rank nothing: 1 of ",
    ),
    "gives all alike": ({"gives": lambda passage: [2.0, 0.0]}, "the scores rank nothing: all "),
}
UNUSABLE = ("gives all alike", "gives no numbers", "gives one no number")


@pytest.mark.parametrize("how", sorted(FAILURES))
def test_a_scoring_that_fails_leaves_the_question_to_the_dones_plain_way(corpus, monkeypatch, how):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain")
    before = Engine(reply=REPLY)
    answerer(corpus, before).answer(DONES_QUESTION)
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain,llm-rank,llm-gate,llm-wide")
    failing, why = FAILURES[how]
    engine, said = Scoring(reply=REPLY, **failing), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == REPLY
    assert fingerprint(engine.conversations) == fingerprint(before.conversations)
    assert any("answered the dones-plain way" in line for line in said)
    assert not any("way failed" in line for line in said)
    (line,) = relevance_lines(said)
    assert f"relevance: not applied ({why}" in line


# The adjustments of the upload of 2026-11-15, written out: what the
# branch ships as DEFAULT may add others.
STACK = "doc-focus,doc-name,doc-reserve,dones-plain,llm-rank,llm-wide"


@pytest.mark.parametrize("how", UNUSABLE)
def test_scores_that_rank_nothing_leave_the_stack_as_dones_plain_alone(corpus, monkeypatch, how):
    # With one score that is not a number, the fusion would put that chunk
    # last on nothing the model read.
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain")
    before = Engine(reply=REPLY)
    answerer(corpus, before).answer(DONES_QUESTION)
    monkeypatch.setenv("DSQA_VARIANT", STACK)
    failing, why = FAILURES[how]
    engine, said = Scoring(reply=REPLY, **failing), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == REPLY
    assert engine.scored  # the model was asked, and its scores were not used
    assert fingerprint(engine.conversations) == fingerprint(before.conversations)
    (line,) = relevance_lines(said)
    assert f"relevance: not applied ({why}" in line


def test_without_a_ranking_the_filter_says_so_and_the_question_is_answered(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain,llm-rank")
    said = []
    asked = types.SimpleNamespace(
        engine=Scoring(reply=REPLY),
        index=None,
        budget=Budget(0.0),
        asked=1,
        say=said.append,
        clock=lambda: 0.0,
        whole=None,
    )
    assert route.answered(asked, DONES_QUESTION, 0.0, 50.0) == REPLY
    (line,) = relevance_lines(said)
    assert line == (
        "[dsqa] question 1: relevance: not applied (no ranking), context of 0 tokens in 0 fragments"
    )
    ((_, user),) = asked.engine.conversations
    assert user["content"] == DONES_QUESTION


def test_the_seconds_the_scoring_takes_are_taken_from_the_reply(corpus, monkeypatch):
    hour = config.HOUR_SECONDS - config.MARGIN_SECONDS
    emergency = hour - (config.QUESTIONS_AN_HOUR - 1) * route.RESERVE_SECONDS
    clock = types.SimpleNamespace(now=0.0)

    def tick():
        return clock.now

    monkeypatch.setenv("DSQA_VARIANT", "dones-plain")
    engine = Scoring(reply=REPLY, clock=clock, step=7.0)
    answerer(corpus, engine, clock=tick).answer(DONES_QUESTION)
    ((_, seconds),) = engine.limits
    assert seconds == pytest.approx(emergency)  # nothing scored, nothing taken
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain,llm-rank")
    clock.now = 0.0
    said = []
    engine = Scoring(reply=REPLY, clock=clock, step=7.0)
    answerer(corpus, engine, said, clock=tick).answer(DONES_QUESTION)
    ((_, seconds),) = engine.limits
    assert seconds == pytest.approx(emergency - 7.0)
    (line,) = relevance_lines(said)
    assert " in 7.0 s, first document " in line
