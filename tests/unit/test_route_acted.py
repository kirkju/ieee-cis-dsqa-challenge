"""llm-acted (route.py): the entry as a measurement of whether the base
model's scoring of passages runs where it is evaluated. A question on
IFMIF-DONES whose ranking the model scored in full is answered exactly as
without llm-acted, by hash; every other question gets the fixed sentence and
the model is asked for no reply: a question on AI papers, and one on
IFMIF-DONES whose scoring could not be had, was cut short or ranked nothing,
or whose dones-plain way failed; never the entry's own way, and never the
last resort of main.py. The engine is the scripted one of
test_route_relevance.py."""

import importlib.util
import types
from pathlib import Path

import pymupdf
import pytest
from dsqa import config, route
from dsqa.answering import Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf
from tests.unit.test_route import AI_QUESTION, DONES_QUESTION, Engine, fingerprint
from tests.unit.test_route_relevance import FAILURES, REPLY, STACK, Scoring

ACTED = STACK + ",llm-acted"
MAIN = Path(__file__).resolve().parents[2] / "03_dsqa" / "main.py"


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


def acted_lines(said):
    return [line for line in said if ": llm-acted: " in line]


def test_the_sentence_and_when_the_entry_is_measuring(monkeypatch):
    assert route.UNANSWERED == "This entry does not answer this question."
    monkeypatch.setenv("DSQA_VARIANT", STACK)
    assert not route.measuring()
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    assert route.measuring()
    monkeypatch.setenv("DSQA_VARIANT", "no-such-adjustment")
    assert not route.measuring()  # and it does not raise


def test_without_llm_acted_the_entry_answers_as_the_stack(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", STACK)
    engine, said = Scoring(reply=REPLY), []
    built = answerer(corpus, engine, said)
    assert built.answer(DONES_QUESTION) == REPLY
    assert len(engine.conversations) == 1
    assert built.answer(AI_QUESTION) not in ("", route.UNANSWERED)
    assert len(engine.conversations) > 1 and acted_lines(said) == []


def test_a_question_on_dones_scored_in_full_is_answered_exactly_as_the_stack(corpus, monkeypatch):
    def still():
        return 0.0

    monkeypatch.setenv("DSQA_VARIANT", STACK)
    before = Scoring(reply=REPLY)
    assert answerer(corpus, before, clock=still).answer(DONES_QUESTION) == REPLY
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    engine, said = Scoring(reply=REPLY), []
    assert answerer(corpus, engine, said, clock=still).answer(DONES_QUESTION) == REPLY
    # The same passages scored, the same conversation, the same limits.
    assert engine.scored and fingerprint(engine.scored) == fingerprint(before.scored)
    assert fingerprint(engine.conversations) == fingerprint(before.conversations)
    assert engine.limits == before.limits
    assert acted_lines(said) == ["[dsqa] question 1: llm-acted: scored in full, answered"]
    assert any(": relevance: llm-rank scored " in line for line in said)


def test_a_question_not_on_dones_gets_the_sentence_and_nothing_of_the_model(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    engine, said = Scoring(reply=REPLY), []
    assert answerer(corpus, engine, said).answer(AI_QUESTION) == route.UNANSWERED
    assert engine.conversations == [] and engine.scored == []
    assert acted_lines(said) == [
        "[dsqa] question 1: llm-acted: not told on IFMIF-DONES, the model not asked, "
        "the fixed sentence"
    ]


def test_an_engine_that_cannot_score_gets_the_sentence_and_no_reply(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    engine, said = Engine(reply=REPLY), []
    built = answerer(corpus, engine, said)
    assert built.answer(DONES_QUESTION) == built.answer(AI_QUESTION) == route.UNANSWERED
    assert engine.conversations == []
    assert acted_lines(said)[0] == (
        "[dsqa] question 1: llm-acted: not scored in full (the engine cannot score), "
        "the fixed sentence"
    )


@pytest.mark.parametrize("how", sorted(FAILURES))
def test_a_scoring_that_fails_or_ranks_nothing_gets_the_sentence(corpus, monkeypatch, how):
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    failing, why = FAILURES[how]
    engine, said = Scoring(reply=REPLY, **failing), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == route.UNANSWERED
    assert engine.scored and engine.conversations == []
    (line,) = acted_lines(said)
    assert line.startswith(f"[dsqa] question 1: llm-acted: not scored in full ({why}")
    assert line.endswith("), the fixed sentence")


def test_a_scoring_cut_short_gets_the_sentence_and_says_how_short(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    monkeypatch.setattr(config, "RELEVANCE_SLICE", 2)
    clock = types.SimpleNamespace(now=0.0)

    given = []

    def apart(passage):
        # A score of its own for each passage scored, so that the two rank.
        given.append(passage)
        return [len(given) - 1.5, 0.0]

    engine, said = Scoring(reply=REPLY, clock=clock, step=20.0, gives=apart), []
    built = answerer(corpus, engine, said, clock=lambda: clock.now)
    asked = min(len(built.index.chunks), config.RELEVANCE_WIDE_N)
    assert asked > 2
    assert built.answer(DONES_QUESTION) == route.UNANSWERED
    # 20 s a slice of two: a second would end past the 30 s the scoring has.
    assert len(engine.scored) == 2 and engine.conversations == []
    assert acted_lines(said) == [
        f"[dsqa] question 1: llm-acted: not scored in full (2 of {asked} scored, stopped for "
        "time), the fixed sentence"
    ]


def test_a_reply_that_fails_after_the_scoring_gets_the_sentence_and_no_other_way(
    corpus, monkeypatch
):
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    engine, said = Scoring(reply=REPLY, raises=[RuntimeError("the GPU fell over")]), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == route.UNANSWERED
    # One reply asked for, the dones-plain one: the entry's own way is not tried.
    assert engine.scored and len(engine.conversations) == 1
    assert acted_lines(said) == [
        "[dsqa] question 1: llm-acted: the dones-plain way failed (RuntimeError), "
        "the fixed sentence"
    ]
    engine, said = Scoring(reply="  "), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == route.UNANSWERED
    assert len(engine.conversations) == 1
    assert acted_lines(said) == [
        "[dsqa] question 1: llm-acted: scored in full, and the dones-plain way gave no "
        "answer, the fixed sentence"
    ]


def test_whatever_raises_gets_the_sentence(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    engine, said = Scoring(reply=REPLY), []
    built = answerer(corpus, engine, said)

    def broken(question):
        raise ValueError("no ranking today")

    built.index.rank = broken
    assert built.answer(DONES_QUESTION) == route.UNANSWERED
    assert engine.conversations == [] and engine.scored == []
    assert acted_lines(said) == [
        "[dsqa] question 1: llm-acted: the dones-plain way failed (ValueError), the fixed sentence"
    ]


def test_without_a_model_or_an_answer_of_route_py_the_sentence(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", ACTED)
    said = []
    assert answerer(corpus, None, said).answer(DONES_QUESTION) == route.UNANSWERED
    assert acted_lines(said) == ["[dsqa] question 1: llm-acted: no model, the fixed sentence"]
    # Whatever route.py gives back, the entry's own way does not answer.
    engine, said = Scoring(reply=REPLY), []
    built = answerer(corpus, engine, said)
    monkeypatch.setattr(route, "answered", lambda *args: None)
    assert built.answer(AI_QUESTION) == route.UNANSWERED
    assert engine.conversations == []
    assert acted_lines(said) == [
        "[dsqa] question 1: llm-acted: no answer came from route.py, the fixed sentence"
    ]


def test_another_way_for_dones_is_not_measured(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-zero,llm-rank,llm-acted")
    engine, said = Scoring(reply=REPLY), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == route.UNANSWERED
    assert engine.conversations == [] and engine.scored == []
    assert acted_lines(said) == [
        "[dsqa] question 1: llm-acted: dones-zero chosen, and only dones-plain is measured, "
        "the fixed sentence"
    ]


def main_module():
    spec = importlib.util.spec_from_file_location("acted_main", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_gives_the_sentence_and_never_its_last_resort(monkeypatch):
    main = main_module()
    asked = []

    class Model:
        def generate(self, messages, max_new_tokens, max_seconds=None):
            asked.append(messages)
            return "the reply of the model alone"

    class Breaks:
        def answer(self, question):
            raise RuntimeError("the answerer fell over")

    main._kept["engine"] = Model()
    for answerer_kept in (None, Breaks()):
        main._kept["answerer"] = answerer_kept
        monkeypatch.setenv("DSQA_VARIANT", ACTED)
        assert main.answer("Where is the trap?", "nowhere", "the/model") == route.UNANSWERED
        assert asked == []
        # Without llm-acted, the last resort is the model alone, as it was.
        monkeypatch.setenv("DSQA_VARIANT", STACK)
        assert main.answer("Where is the trap?", "nowhere", "the/model") == (
            "the reply of the model alone"
        )
        assert len(asked) == 1
        asked.clear()
