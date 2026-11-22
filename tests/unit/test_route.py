"""dsqa/route.py and dsqa/plain.py: IFMIF-DONES questions answered as the
organizers' examples answer them, and every other question as before.

The engine is scripted as in test_answering.py; the corpus is two PDFs made
on the spot, one on the lithium loop (on IFMIF-DONES by its words) and one
on a trap (on neither). What matters most is held to by hash: a question on
AI papers is asked exactly what it is asked without the adjustments, so that
the two runs differ on the IFMIF-DONES questions alone.
"""

import hashlib
import json

import numpy as np
import pymupdf
import pytest
from dsqa import config, domain, plain, route
from dsqa.answering import Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf

DONES_QUESTION = "Where is the lithium loop of the IFMIF-DONES plant?"
AI_QUESTION = "Which learning rate does the neural transformer use?"


class Engine:
    """Replies what it is told, after raising what it is told; keeps every
    conversation it was shown and the limits it was given."""

    def __init__(self, reply="Final answer: In the dump tank.", raises=()):
        self.reply, self.raises = reply, list(raises)
        self.conversations: list[list[dict]] = []
        self.limits: list[tuple] = []
        self.released = 0

    def count(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, max_seconds=None):
        self.conversations.append(messages)
        self.limits.append((max_new_tokens, max_seconds))
        if self.raises:
            raise self.raises.pop(0)
        return self.reply

    def release(self):
        self.released += 1


def fingerprint(conversations) -> str:
    return hashlib.sha256(json.dumps(conversations, sort_keys=True).encode()).hexdigest()


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


def answerer(corpus, engine, said=None):
    say = said.append if said is not None else (lambda line: None)
    return Answerer(corpus, engine, embedder=Letters(), say=say)


class Index:
    """What route.py needs of an index: its chunks, the document of each, and
    a ranking that is the order given."""

    def __init__(self, chunks, documents, order):
        self.chunks, self.documents, self.order = chunks, np.asarray(documents), order

    def rank(self, question):
        return np.asarray(self.order)


ON_DONES = "The lithium target of the IFMIF-DONES neutron source and its deuteron beam."
ON_AI = "A neural network trained on a benchmark dataset with a transformer encoder."


def test_the_domain_is_told_always_and_the_side_of_doubt_is_the_entrys_own():
    assert route.decided(DONES_QUESTION) == domain.DONES
    assert route.decided(AI_QUESTION) == domain.AI
    # No word of either list, and no documents: the entry's own way.
    assert route.decided("What is the volume of the main building?") == domain.AI
    # One word on each side weighs the same.
    assert route.decided("The lithium of the neural plant?") == domain.AI


def test_the_documents_ranked_first_tell_a_question_without_words():
    index = Index([ON_DONES, ON_DONES, ON_AI, ON_AI], [0, 0, 1, 1], [0, 1, 2, 3])
    assert list(route.document_domains(index)) == [True, False]
    question = "What is the volume of the main building?"
    assert route.decided(question, index) == domain.DONES
    assert route.decided(question, index, order=[2, 3, 0, 1]) == domain.AI
    # The first counts 1 and the second 1/2: the words of the question
    # outweigh the documents when there are two of them.
    assert route.decided("The neural dataset of the benchmark?", index) == domain.AI
    assert route.first_documents([3, 2, 1, 0], index.documents) == [1, 0]


# An AI paper that speaks of radiologists, as CheXNet does: dense in words of
# the IFMIF-DONES list, and as dense in those of the AI list.
ON_RADIOLOGY = (
    "The neural network detects pneumonia from chest images better than the radiologists; "
    "the convolutional model was trained on a radiology dataset and a radiology benchmark."
)


def test_an_ai_paper_as_dense_in_words_of_its_own_list_is_not_on_dones():
    index = Index([ON_RADIOLOGY, ON_DONES], [0, 1], [0, 1])
    on_dones, on_ai = domain.counted(ON_RADIOLOGY)
    assert on_dones * 1000 >= route.DONES_PER_THOUSAND * len(ON_RADIOLOGY.split())
    assert on_dones < route.DONES_OVER_AI * on_ai
    assert list(route.document_domains(index)) == [False, True]
    question = "How does the F1 score of the model compare with that of the radiologists?"
    assert route.decided(question, index) == domain.AI


def test_a_question_whose_first_document_is_not_on_dones_is_the_entrys_own():
    chunks = [ON_AI, ON_DONES, ON_DONES, ON_DONES, ON_DONES]
    index = Index(chunks, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    question = "The lithium of the deuteron beam?"  # two words of the IFMIF-DONES list
    assert route.decided(question, index) == domain.AI
    assert route.decided(question, index, order=[1, 0, 2, 3, 4]) == domain.DONES


def test_the_domains_are_told_when_the_answerer_is_built(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "")
    assert getattr(answerer(corpus, Engine()).index, "_route_dones", None) is None
    monkeypatch.setenv("DSQA_VARIANT", "dones-zero")
    assert getattr(answerer(corpus, Engine()).index, "_route_dones", None) is not None
    route.prepared(None)  # nothing to tell, and nothing raised


def test_a_documents_domain_is_worked_out_once_and_kept():
    index = Index([ON_DONES, ON_AI], [0, 1], [0, 1])
    first = route.document_domains(index)
    index.chunks = [ON_AI, ON_AI]
    assert route.document_domains(index) is first


def test_without_an_adjustment_nothing_is_routed(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "")
    engine = Engine()
    asked = answerer(corpus, engine)
    assert route.answered(asked, DONES_QUESTION, 0.0, 50.0) is None
    assert engine.conversations == []


@pytest.mark.parametrize("way", ["dones-zero", "dones-plain"])
def test_a_question_on_ai_papers_is_asked_exactly_as_without_the_adjustments(
    corpus, monkeypatch, way
):
    monkeypatch.setenv("DSQA_VARIANT", "")
    before = Engine()
    reply = answerer(corpus, before).answer(AI_QUESTION)
    monkeypatch.setenv("DSQA_VARIANT", way)
    after = Engine()
    said = []
    assert answerer(corpus, after, said).answer(AI_QUESTION) == reply == "In the dump tank."
    assert fingerprint(after.conversations) == fingerprint(before.conversations)
    assert [limits[0] for limits in after.limits] == [config.ANSWER_TOKENS]
    assert not any("way" in line for line in said)


def test_dones_zero_asks_the_question_alone_as_the_zero_shot_example(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-zero")
    engine = Engine(reply="  The loop is in the lithium systems building.  ")
    said = []
    reply = answerer(corpus, engine, said).answer(DONES_QUESTION)
    assert reply == "The loop is in the lithium systems building."
    (conversation,) = engine.conversations
    assert conversation == [
        {"role": "system", "content": plain.SYSTEM_PROMPT},
        {"role": "user", "content": DONES_QUESTION},
    ]
    ((tokens, seconds),) = engine.limits
    assert tokens == plain.MAX_NEW_TOKENS
    # Not the seconds a question of the entry may take: what is left of the
    # hour once the 49 questions after it are kept their few seconds.
    hour = config.HOUR_SECONDS - config.MARGIN_SECONDS
    emergency = hour - (config.QUESTIONS_AN_HOUR - 1) * route.RESERVE_SECONDS
    assert emergency - 1 < seconds <= emergency
    assert seconds > config.QUESTION_SECONDS
    assert engine.released == 1
    assert any("answered the dones-zero way" in line for line in said)


def test_dones_plain_gives_our_chunks_as_the_rag_examples_fragments(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain")
    engine = Engine(reply="It is in the plant.")
    asked = answerer(corpus, engine)
    assert asked.answer(DONES_QUESTION) == "It is in the plant."
    ((system, user),) = engine.conversations
    assert system == {"role": "system", "content": plain.SYSTEM_PROMPT}
    text = user["content"]
    assert text.startswith(f"{plain.CONTEXT_OPENING}\n\n[Fragment 1]\n")
    assert text.endswith(f"\n\n---\n\n{DONES_QUESTION}")
    # The fragments are the chunks of the ranking, best first, as they are.
    order = asked.index.rank(DONES_QUESTION)
    fragments = text.split("\n\n---\n\n")[0].split("[Fragment ")[1:]
    for place, fragment in enumerate(fragments, 1):
        number, _, chunk = fragment.partition("]\n")
        assert int(number) == place
        assert chunk.strip() == asked.index.chunks[int(order[place - 1])].strip()


def test_the_entrys_own_way_answers_when_the_dones_way_fails(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-zero")
    engine = Engine(raises=[RuntimeError("the GPU fell over")])
    said = []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == "In the dump tank."
    first, second = engine.conversations
    assert first[1]["content"] == DONES_QUESTION  # the zero-shot way, which failed
    assert "Passages retrieved" in second[1]["content"]  # then the entry's own
    assert any("the dones-zero way failed" in line for line in said)

    # A blank reply is no answer either.
    engine = Engine(reply="   ")
    said = []
    reply = answerer(corpus, engine, said).answer(DONES_QUESTION)
    assert reply  # what the entry's own way falls back on, never blank
    assert any("dones-zero gave no answer" in line for line in said)


def test_with_no_time_left_the_dones_way_is_not_asked(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-zero")
    engine = Engine()
    asked = answerer(corpus, engine)
    assert route.answered(asked, DONES_QUESTION, began=0.0, allowance=1.0) is None
    assert engine.conversations == []


def test_without_a_model_the_dones_way_is_not_taken(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-plain")
    asked = answerer(corpus, None)
    assert route.answered(asked, DONES_QUESTION, 0.0, 50.0) is None


def test_the_fragments_are_taken_as_the_rag_example_takes_them():
    chunks = ["one two three", "four five", "six seven eight nine", "ten"]

    def count(text):
        return len(text.split())

    assert plain.fragments_of([2, 0, 1, 3], chunks, count, budget=8) == chunks[2:3] + chunks[:1]
    # Not even the first fits: a slice of it, the share the budget is of it.
    assert plain.fragments_of([2], chunks, count, budget=2) == [chunks[2][:10]]
    assert plain.context_of([]) == ""
    assert plain.rag_messages("Why?", [])[1]["content"] == "Why?"
    assert plain.rag_messages("Why?", ["a", "b"])[1]["content"] == (
        f"{plain.CONTEXT_OPENING}\n\n[Fragment 1]\na\n\n[Fragment 2]\nb\n\n---\n\nWhy?"
    )


def test_more_than_one_way_chosen_takes_the_first_of_them(corpus, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-zero,dones-plain")
    assert route.chosen_ways() == ["dones-plain", "dones-zero"]
    assert route.chosen_way() == "dones-plain"
    engine, said = Engine(reply="It is in the plant."), []
    answerer(corpus, engine, said).answer(DONES_QUESTION)
    assert any("dones-plain chosen over dones-zero" in line for line in said)
    monkeypatch.setenv("DSQA_VARIANT", "")
    assert route.chosen_way() is None


class Budget:
    def __init__(self, elapsed):
        self.spent = elapsed

    def elapsed(self):
        return self.spent


def test_the_emergency_limit_keeps_a_few_seconds_for_every_later_question():
    hour = config.HOUR_SECONDS - config.MARGIN_SECONDS
    reserve = route.RESERVE_SECONDS
    assert route.emergency_seconds(Budget(300), 0) == hour - 300 - 49 * reserve
    assert route.emergency_seconds(Budget(2000), 49) == hour - 2000
    # The 51st question is the first of the second hour.
    assert route.emergency_seconds(Budget(3500), 50) == 2 * config.HOUR_SECONDS - 360 - 3500 - (
        49 * reserve
    )


class Example:
    """A stand-in for example.py, the RAG example itself."""

    def __init__(self, ready=True, reply="Off-line, at 500 to 600 degrees.", seconds=False):
        self.is_ready, self.reply, self.calls = ready, reply, []
        if seconds:
            self.answer = self._timed

    def ready(self):
        return self.is_ready

    def answer(self, question, corpus_dir, model_id, engine):
        self.calls.append((question, corpus_dir, model_id, engine))
        return self.reply

    def _timed(self, question, corpus_dir, model_id, engine, seconds=None):
        self.calls.append((question, corpus_dir, model_id, engine, seconds))
        return self.reply


@pytest.fixture
def example_way(monkeypatch):
    """dones-example known as an adjustment, as example.py's branch makes it."""
    monkeypatch.setitem(
        route.variant.KNOWN, "dones-example", "the RAG example itself (example.py)"
    )

    def put(module):
        import sys

        monkeypatch.setitem(sys.modules, "dsqa.example", module)
        return module

    return put


def test_dones_example_asks_example_py_with_the_corpus_and_the_engine(
    corpus, monkeypatch, example_way
):
    monkeypatch.setenv("DSQA_VARIANT", "dones-example,dones-plain")
    example = example_way(Example())
    engine, said = Engine(), []
    reply = answerer(corpus, engine, said).answer(DONES_QUESTION)
    assert reply == "Off-line, at 500 to 600 degrees."
    ((question, corpus_dir, model_id, given),) = example.calls
    assert (question, corpus_dir, model_id, given) == (DONES_QUESTION, corpus, None, engine)
    assert engine.conversations == []
    assert any("answered the dones-example way" in line for line in said)
    assert any("dones-example chosen over dones-plain" in line for line in said)
    # A question on AI papers is not its.
    answerer(corpus, Engine()).answer(AI_QUESTION)
    assert len(example.calls) == 1

    # Its answer() is told the seconds when it takes them.
    timed = example_way(Example(seconds=True))
    answerer(corpus, Engine()).answer(DONES_QUESTION)
    assert timed.calls[0][-1] > config.QUESTION_SECONDS


@pytest.mark.parametrize("module", ["missing", "not ready", "no answer"])
def test_without_the_example_the_entrys_own_way_answers(corpus, monkeypatch, example_way, module):
    monkeypatch.setenv("DSQA_VARIANT", "dones-example")
    if module == "not ready":
        example_way(Example(ready=False))
    elif module == "no answer":
        example_way(Example(reply=None))
    engine, said = Engine(), []
    assert answerer(corpus, engine, said).answer(DONES_QUESTION) == "In the dump tank."
    (conversation,) = engine.conversations
    assert "Passages retrieved" in conversation[1]["content"]
    assert any("dones-example gave no answer" in line for line in said)


def test_with_plain_too_the_dones_questions_still_go_to_the_example(
    corpus, monkeypatch, example_way
):
    # plain changes the entry's own way only: a question on IFMIF-DONES is
    # still the example's, and one on AI papers is asked as plain asks it.
    monkeypatch.setenv("DSQA_VARIANT", "dones-example,plain")
    example = example_way(Example())
    engine, said = Engine(reply="In the dump tank, off-line."), []
    asked = answerer(corpus, engine, said)
    assert asked.answer(DONES_QUESTION) == "Off-line, at 500 to 600 degrees."
    assert len(example.calls) == 1 and engine.conversations == []
    assert asked.answer(AI_QUESTION) == "In the dump tank, off-line."
    (conversation,) = engine.conversations
    assert conversation[0]["content"] == plain.SYSTEM_PROMPT
    assert conversation[1]["content"].startswith(plain.CONTEXT_OPENING)
    assert len(example.calls) == 1
    # When the example gives nothing, the question on IFMIF-DONES is the
    # entry's own way's, which with plain asks as plain asks.
    example_way(Example(reply=None))
    engine = Engine(reply="From the passages.")
    assert answerer(corpus, engine).answer(DONES_QUESTION) == "From the passages."
    assert engine.conversations[0][0]["content"] == plain.SYSTEM_PROMPT
