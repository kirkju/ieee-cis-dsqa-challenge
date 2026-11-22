"""dsqa/answering.py: a question answered, and what is said when a part fails.

The engine is scripted in the test: what it replies, and what it raises on
which call. The corpus is two PDFs made on the spot, and the embedder the
stand-in that counts letters.
"""

import pymupdf
import pytest
from dsqa import answering, config, prompt, retrieval
from dsqa.answering import NO_ANSWER, Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf


class OutOfMemoryError(RuntimeError):
    """Named as torch names it."""


class Engine:
    """An engine that replies what it is told to, after raising what it is
    told to on its first calls. It keeps the conversations it was shown."""

    def __init__(self, reply="Final answer: 97.5 litres per second.", raises=()):
        self.reply, self.raises = reply, list(raises)
        self.shown: list[str] = []
        self.limits: list[tuple] = []
        self.released = 0

    def count(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, max_seconds=None):
        self.shown.append(messages[-1]["content"])
        self.limits.append((max_new_tokens, max_seconds))
        if self.raises:
            raise self.raises.pop(0)
        return self.reply

    def release(self):
        self.released += 1


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two papers, and the entry's numbers cut down to their size."""
    paper_pdf(tmp_path / "b-loop.pdf", BODY)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        document.save(tmp_path / "a-trap.pdf")
    monkeypatch.setattr(config, "CHUNK_TOKENS", 24)
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 80)
    monkeypatch.setattr(config, "LEAST_CONTEXT_TOKENS", 20)
    return tmp_path


# The trap's paper is short enough to be one chunk. Without the title every
# chunk is ranked under, it is the document's text from its start: the title
# the PDF tells, the heading as the page has it, and the text.
PASSAGE = "\n\n".join(["Notes on the nitrogen trap"] * 2 + ["The trap is in the dump tank."])


def answerer(corpus, engine, **options):
    options.setdefault("embedder", Letters())
    options.setdefault("say", lambda line: None)
    return Answerer(corpus, engine, **options)


def passages(shown: str) -> int:
    """How many words of passages a conversation shown to the engine has."""
    if "Passages retrieved" not in shown:
        return 0
    return len(shown.split("---")[0].split())


def test_the_model_is_shown_the_passages_and_its_reply_is_the_answer(corpus):
    engine = Engine()
    asked = answerer(corpus, engine)
    assert asked.answer("Where is the nitrogen trap?") == "97.5 litres per second."
    (shown,) = engine.shown
    assert shown.endswith("---\n\nQuestion: Where is the nitrogen trap?")
    assert f"Document: Notes on the nitrogen trap\n\n{PASSAGE}\n\n---" in shown
    # The other paper's chunks are together under its title, said once over them.
    assert shown.count("Document: ") == 2
    assert shown.count("Document: A study of the lithium loop of the plant\n\n") == 1
    # The answer is bounded in tokens and in time, and the GPU is given back.
    # The time is what the question has left once its passages were found,
    # less what is kept back to ask for an answer that the reply did not
    # state: on a clock finer than Windows's, a little under that.
    ((tokens, seconds),) = engine.limits
    assert tokens == config.ANSWER_TOKENS
    longest = config.QUESTION_SECONDS - config.FINAL_SECONDS
    assert longest - 1 < seconds <= longest
    assert engine.released >= 1
    assert asked.asked == 1


def test_without_a_model_the_best_passage_is_the_answer(corpus):
    reply = answerer(corpus, None).answer("Where is the nitrogen trap?")
    assert reply == PASSAGE


def test_without_a_corpus_the_model_answers_from_what_it_knows(tmp_path):
    engine = Engine("Final answer: In the dump tank.")
    asked = answerer(tmp_path / "nowhere", engine)
    assert asked.index is None
    assert asked.answer("Where is the nitrogen trap?") == "In the dump tank."
    assert engine.shown == ["Where is the nitrogen trap?"]
    # And with neither, there is nothing to say but that.
    assert answerer(tmp_path / "nowhere", None).answer("Where?") == NO_ANSWER


def test_out_of_memory_halves_the_context_for_this_question_and_the_next(corpus):
    engine = Engine(raises=[OutOfMemoryError("CUDA out of memory"), RuntimeError("out of memory")])
    asked = answerer(corpus, engine)
    assert asked.answer("How is the lithium carried?") == "97.5 litres per second."
    first, second, third = (passages(shown) for shown in engine.shown)
    assert first > second > third > 0
    assert asked.context_tokens == 20
    # The next question starts from the smaller context.
    asked.answer("And where is the trap?")
    assert passages(engine.shown[-1]) <= third + 10


def test_out_of_memory_on_the_smallest_context_leaves_the_passages_out(corpus, monkeypatch):
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 20)
    engine = Engine("Final answer: From memory.", raises=[OutOfMemoryError("CUDA out of memory")])
    assert answerer(corpus, engine).answer("How is the lithium carried?") == "From memory."
    assert [passages(shown) > 0 for shown in engine.shown] == [True, False]


def test_another_failure_of_the_model_asks_it_again_without_passages(corpus):
    engine = Engine("Final answer: From memory.", raises=[ValueError("a bad prompt")])
    asked = answerer(corpus, engine)
    assert asked.answer("How is the lithium carried?") == "From memory."
    assert engine.shown[1] == "How is the lithium carried?"
    assert asked.context_tokens == 80  # not a matter of memory: the next one gets its passages


def test_a_model_that_fails_alone_too_leaves_the_best_passage(corpus):
    engine = Engine(raises=[ValueError("no"), ValueError("still no")])
    reply = answerer(corpus, engine).answer("Where is the nitrogen trap?")
    assert reply == PASSAGE
    assert len(engine.shown) == 2


def test_a_blank_reply_is_not_an_answer(corpus, tmp_path):
    assert answerer(corpus, Engine("  \n")).answer("Where is the nitrogen trap?") == PASSAGE
    assert answerer(tmp_path / "nowhere", Engine("")).answer("Where?") == NO_ANSWER
    assert answerer(corpus, Engine(None)).answer("Where is the nitrogen trap?") != ""


def test_with_no_time_left_the_model_is_not_asked(corpus):
    clock = Clock()
    engine = Engine()
    asked = answerer(corpus, engine, clock=clock, started=0.0)
    clock.now = 3600.0  # the hour is up
    assert asked.answer("Where is the nitrogen trap?") == PASSAGE
    assert engine.shown == []
    # With a little left, the model has its share of it and no longer: 300
    # seconds for the 49 questions the hour may still have.
    clock.now = 3600 - 360 - 300
    asked.answer("Where is the nitrogen trap?")
    assert engine.limits == [(config.ANSWER_TOKENS, pytest.approx(300 / 49))]


def test_the_time_a_failed_try_took_is_time_the_next_try_does_not_have(corpus):
    clock = Clock()

    class Slow(Engine):
        def generate(self, messages, max_new_tokens, max_seconds=None):
            clock.now += 48  # nearly all a question may take
            return super().generate(messages, max_new_tokens, max_seconds)

    engine = Slow(raises=[ValueError("no")])
    asked = answerer(corpus, engine, clock=clock, started=0.0)
    assert asked.answer("Where is the nitrogen trap?") == PASSAGE
    assert len(engine.shown) == 1  # two seconds were left: not enough to ask again


def test_without_the_embedder_the_corpus_is_ranked_by_words(corpus, monkeypatch):
    def fail(device=None):
        raise OSError("no embed_model folder in the zip")

    said = []
    monkeypatch.setattr(retrieval, "load_embedder", fail)
    asked = Answerer(corpus, Engine(), say=said.append)  # the embedder it ships with
    assert asked.index is not None and asked.index.embeddings is None
    assert said[0].startswith("[dsqa] no embedder, ranking by words alone: OSError")
    assert asked.answer("Where is the nitrogen trap?") == "97.5 litres per second."


def test_a_corpus_that_cannot_be_indexed_is_done_without(corpus, monkeypatch):
    def fail(*arguments, **options):
        raise MemoryError("too many chunks")

    said = []
    monkeypatch.setattr(retrieval, "build", fail)
    engine = Engine("Final answer: From memory.")
    asked = answerer(corpus, engine, say=said.append)
    assert asked.index is None
    assert any(line.startswith("[dsqa] no index, answering without passages") for line in said)
    assert asked.answer("Where is the nitrogen trap?") == "From memory."


def test_a_ranking_that_fails_on_a_question_costs_only_its_passages(corpus):
    engine = Engine("Final answer: From memory.")
    asked = answerer(corpus, engine)
    asked.index.rank = lambda question: 1 / 0
    assert asked.answer("Where is the nitrogen trap?") == "From memory."
    assert engine.shown == ["Where is the nitrogen trap?"]
    assert answering.out_of_memory(OutOfMemoryError("anything"))
    assert not answering.out_of_memory(ValueError("a bad prompt"))


class Scripted(Engine):
    """An engine that replies one thing after another, and raises what is an
    error among them."""

    def __init__(self, *replies):
        super().__init__()
        self.replies = list(replies)

    def generate(self, messages, max_new_tokens, max_seconds=None):
        self.reply = self.replies.pop(0)
        if isinstance(self.reply, Exception):
            self.raises = [self.reply]
        return super().generate(messages, max_new_tokens, max_seconds)


WORKING = "Table 1 gives the flow of the main loop."
LITHIUM = "How is the lithium carried?"


def test_the_answer_a_reply_states_last_is_returned_first(corpus):
    engine = Engine(f"{WORKING}\nFinal answer: 97.5 litres per second.")
    assert answerer(corpus, engine).answer(LITHIUM) == f"97.5 litres per second.\n\n{WORKING}"
    assert len(engine.shown) == 1  # it stated its answer: it is not asked again


def test_a_reply_that_states_no_answer_is_asked_for_it(corpus):
    engine = Scripted("Table 1 gives the flow of the main", "**Final answer:** 97.5 l/s.")
    said = []
    reply = answerer(corpus, engine, say=said.append).answer(LITHIUM)
    assert reply == "97.5 l/s.\n\nTable 1 gives the flow of the main"
    # The model was shown what it had written and asked for the answer alone,
    # in few tokens.
    assert engine.shown[1] == prompt.CLOSE
    assert engine.limits[1][0] == config.FINAL_TOKENS
    assert any(line.endswith("the answer was asked for apart") for line in said)
    # An answer given without the line it was asked to begin with is taken.
    engine = Scripted("Table 1 gives the flow", "97.5 litres per second.")
    reply = answerer(corpus, engine).answer(LITHIUM)
    assert reply == "97.5 litres per second.\n\nTable 1 gives the flow"


def test_time_is_kept_back_to_ask_a_reply_cut_by_the_clock_for_its_answer(corpus):
    clock = Clock()

    class Cut(Scripted):
        def generate(self, messages, max_new_tokens, max_seconds=None):
            clock.now += max_seconds  # all the time it was given
            return super().generate(messages, max_new_tokens, max_seconds)

    engine = Cut("Table 1 gives the flow of the main", "Final answer: 97.5 l/s.")
    asked = answerer(corpus, engine, clock=clock, started=0.0)
    assert asked.answer(LITHIUM).startswith("97.5 l/s.")
    (_, working), (_, stating) = engine.limits
    assert working == pytest.approx(config.QUESTION_SECONDS - config.FINAL_SECONDS)
    assert stating == pytest.approx(config.FINAL_SECONDS)

    # With little time for a question, all of it goes to the working: a reply
    # cut then is returned as it is.
    engine = Cut("Table 1 gives the flow of the main")
    said = []
    asked = answerer(corpus, engine, clock=clock, started=0.0, say=said.append)
    clock.now = 3600 - 360 - 300
    assert asked.answer(LITHIUM) == "Table 1 gives the flow of the main"
    assert engine.limits == [(config.ANSWER_TOKENS, pytest.approx(300 / 50))]
    assert any("no time to ask for it" in line for line in said)


def test_a_reply_is_kept_when_asking_for_its_answer_comes_to_nothing(corpus):
    for second in (ValueError("no"), OutOfMemoryError("CUDA out of memory"), "  ", None):
        engine = Scripted("Table 1 gives the flow", second)
        asked = answerer(corpus, engine)
        assert asked.answer(LITHIUM) == "Table 1 gives the flow"
        assert len(engine.shown) == 2
        # A failure there is not the context's: the next question has all of it.
        assert asked.context_tokens == 80
