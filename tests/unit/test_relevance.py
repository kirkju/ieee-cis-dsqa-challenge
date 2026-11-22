"""dsqa/relevance.py: the base model as a judge of relevance. The engine is a
stand-in whose score() reads a passage's words, and the index a small one of
chunks made on the spot; no model is loaded."""

import types

import numpy as np
import pytest
from dsqa import config, relevance

# Six chunks of three documents, each opening with its document's title.
CHUNKS = [
    "Lithium loop\n\nthe loop carries lithium at 250 C",
    "Lithium loop\n\nthe pump drives the lithium",
    "Nitrogen trap\n\nthe trap holds nitrogen",
    "Nitrogen trap\n\nthe trap is cold",
    "Beam dump\n\nthe beam stops in the dump",
    "Beam dump\n\nlithium is not here",
]
DOCUMENTS = np.asarray([0, 0, 1, 1, 2, 2])


def index_of(chunks=CHUNKS, documents=DOCUMENTS, tables=None):
    return types.SimpleNamespace(chunks=list(chunks), documents=documents, tables=tables or {})


class Judge:
    """Scores Yes as many times as the passage has a word of `liked`, and the
    document a question is about by the word `about`; notes what it read."""

    def __init__(self, liked=("lithium",), about="lithium", fails=False):
        self.liked, self.about, self.fails = liked, about, fails
        self.read: list[list[dict]] = []
        self.labels: list[tuple] = []

    def count(self, text):
        return len(text.split())

    def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
        self.read.extend(conversations)
        self.labels.append(tuple(labels))
        if self.fails:
            return None
        if tuple(labels) == ("Yes", "No"):
            rows = []
            for conversation in conversations:
                passage = conversation[1]["content"].split("\n\nQuestion:")[0]
                liked = sum(passage.count(word) for word in self.liked)
                rows.append([float(liked), 1.0])
            return rows
        rows = []
        for conversation in conversations:
            asked = conversation[1]["content"]
            row = []
            for letter in labels:
                start = asked.find(f"Document {letter}: ")
                end = asked.find("\n\nDocument ", start + 1)
                card = asked[start : end if end > 0 else None]
                row.append(5.0 if self.about in card.split("\n")[0].lower() else 0.0)
            rows.append(row)
        return rows


def test_a_passage_is_scored_before_its_question_as_its_document_names_it():
    asked = relevance.conversation("How hot?", "Lithium loop", "the loop is hot")
    assert asked[0] == {"role": "system", "content": relevance.SYSTEM}
    text = asked[1]["content"]
    assert text.startswith('Passage from the document "Lithium loop":\nthe loop is hot')
    assert text.index("the loop is hot") < text.index("Question: How hot?")
    assert text.endswith("Answer Yes or No.")


def test_the_first_n_are_scored_by_p_yes():
    judge = Judge()
    scores = relevance.scored(judge, "Where is the lithium?", index_of(), [0, 4, 2, 5], n=3)
    assert len(judge.read) == 3 and judge.labels == [("Yes", "No")]
    # P(Yes) / (P(Yes) + P(No)) is the sigmoid of the difference of the
    # logits; the judge likes "lithium" once in the body of chunk 0.
    expected = [1 / (1 + np.exp(-(liked - 1.0))) for liked in (1, 0, 0)]
    assert scores == pytest.approx(expected)
    assert relevance.scored(judge, "q", index_of(), [], n=3) == []


def test_without_a_score_the_ranking_stands():
    plain = types.SimpleNamespace(count=len)
    assert relevance.scored(plain, "q", index_of(), [0, 1]) is None
    assert relevance.scored(Judge(fails=True), "q", index_of(), [0, 1]) is None


def test_a_table_shown_whole_is_scored_whole_and_a_long_passage_cut(monkeypatch):
    whole = "Lithium loop\n\nTable 1. Flows.\n| a | 1 |\n| b | 2 |"
    tabled = index_of(tables={1: (1, 2, whole)})
    monkeypatch.setattr(config, "TABLE_WHOLE_TOKENS", 500, raising=False)
    judge = Judge()
    relevance.scored(judge, "q", tabled, [1], n=1)
    assert "Table 1. Flows." in judge.read[0][1]["content"]
    # A version whose context shows no table whole scores the chunk as it is.
    monkeypatch.setattr(config, "TABLE_WHOLE_TOKENS", 0)
    judge = Judge()
    relevance.scored(judge, "q", tabled, [1], n=1)
    assert "the pump drives the lithium" in judge.read[0][1]["content"]
    assert "Table 1." not in judge.read[0][1]["content"]
    long = ["Lithium loop\n\n" + "word " * 2000]
    judge = Judge()
    relevance.scored(judge, "q", index_of(long, np.asarray([0])), [0], n=1)
    passage = judge.read[0][1]["content"].split("\n\nQuestion:")[0]
    assert len(passage.split()) <= config.RELEVANCE_BODY_TOKENS + 10


def test_llm_rank_fuses_the_order_with_the_scores_and_keeps_the_rest_behind():
    order = np.asarray([3, 0, 5, 1, 2, 4])
    # The scores of the first four: the last of them scores best.
    fused = relevance.reranked(order, [0.1, 0.2, 0.3, 0.9])
    assert sorted(fused[:4].tolist()) == [0, 1, 3, 5] and fused[4:].tolist() == [2, 4]
    # Fused, not replaced: the first of the order and the best score lead.
    assert set(fused[:2].tolist()) == {3, 1}
    assert relevance.reranked(order, []).tolist() == order.tolist()


def test_llm_gate_leaves_out_the_low_keeps_the_best_and_is_never_empty():
    scores_of = {3: 0.05, 0: 0.9, 5: 0.1, 1: 0.5}
    order = [3, 0, 5, 1, 2, 4]
    # Under 0.2: 3 and 5 out; 2 and 4 were not scored, out too.
    assert relevance.gated(order, scores_of, 0.2, keep=1).tolist() == [0, 1]
    # The best k kept whatever their score.
    assert relevance.gated(order, scores_of, 0.95, keep=2).tolist() == [0, 1]
    assert relevance.gated(order, scores_of, 0.95, keep=0).tolist() == []
    assert relevance.gated(order, {}, 0.2, keep=3).tolist() == []


def test_llm_doc_takes_the_document_whose_best_chunks_score_most():
    scores_of = {0: 0.9, 1: 0.8, 2: 0.95, 4: 0.1}
    order = [2, 0, 1, 4]
    # Document 0: 0.9 + 0.8; document 1: 0.95 alone.
    assert relevance.document_of(order, scores_of, DOCUMENTS, top=3) == 0
    assert relevance.document_of(order, scores_of, DOCUMENTS, top=1) == 1
    # A tie goes to the document ranked first.
    assert relevance.document_of([4, 2], {2: 0.5, 4: 0.5}, DOCUMENTS, top=1) == 2
    assert relevance.document_of(order, {}, DOCUMENTS, top=3) is None
    put = relevance.put_first([2, 0, 4, 1, 5], DOCUMENTS, 0)
    assert put.tolist() == [0, 1, 2, 4, 5]
    assert relevance.put_first([2, 0], DOCUMENTS, None).tolist() == [2, 0]


def test_llm_pick_asks_only_when_the_first_two_are_close(monkeypatch):
    index = index_of()
    judge = Judge(about="nitrogen")
    # Document 0 well ahead: not asked.
    far = {0: 0.9, 1: 0.9, 2: 0.1, 4: 0.1}
    assert relevance.picked(judge, "q", index, [0, 1, 2, 4], far) is None
    assert judge.read == []
    # Close: asked twice, in both orders, and the document the question is about chosen.
    close = {0: 0.6, 2: 0.55, 4: 0.5}
    monkeypatch.setattr(config, "RELEVANCE_PICK_MARGIN", 0.5)
    assert relevance.picked(judge, "What is cold?", index, [0, 2, 4], close) == 1
    assert len(judge.read) == 2 and judge.labels == [("A", "B", "C")]
    first, second = (asked[1]["content"] for asked in judge.read)
    assert first.index("Lithium loop") < first.index("Beam dump")
    assert second.index("Beam dump") < second.index("Lithium loop")
    assert first.rstrip().endswith("Answer with one letter.")
    assert relevance.picked(Judge(fails=True), "q", index, [0, 2, 4], close) is None
    assert relevance.picked(types.SimpleNamespace(count=len), "q", index, [0, 2], close) is None


def test_a_position_alone_does_not_choose_the_document(monkeypatch):
    class First(Judge):
        """Says the first letter, whatever is shown."""

        def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
            self.read.extend(conversations)
            return [[5.0] + [0.0] * (len(labels) - 1) for _ in conversations]

    monkeypatch.setattr(config, "RELEVANCE_PICK_MARGIN", 1.0)
    close = {0: 0.6, 2: 0.55, 4: 0.5}
    # Shown first once and last once, the first and the third tie; the tie
    # goes to the document that was first by its scores.
    assert relevance.picked(First(), "q", index_of(), [0, 2, 4], close) == 0


def test_without_an_adjustment_chosen_the_ranking_is_the_one_it_was(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "")
    order = np.asarray([2, 0, 1])
    judge = Judge()
    assert relevance.adjusted(judge, "q", index_of(), order) is order
    assert judge.read == []


def test_the_adjustments_chosen_make_the_ranking(monkeypatch):
    order = np.asarray([2, 4, 0, 3, 1, 5])
    question = "Where is the lithium?"
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank")
    ranked = relevance.adjusted(Judge(), question, index_of(), order)
    assert sorted(ranked.tolist()) == sorted(order.tolist()) and ranked.tolist() != order.tolist()
    monkeypatch.setenv("DSQA_VARIANT", "llm-gate")
    gated = relevance.adjusted(Judge(), question, index_of(), order)
    assert set(gated.tolist()) <= set(order.tolist()) and len(gated) >= config.RELEVANCE_KEEP
    # llm-doc: the lithium loop's chunks first, in the order they had.
    monkeypatch.setenv("DSQA_VARIANT", "llm-doc")
    assert relevance.adjusted(Judge(), question, index_of(), order).tolist()[:2] == [0, 1]


def test_with_too_little_time_or_a_failure_the_ranking_stands(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank")
    order = np.asarray([2, 4, 0])
    judge = Judge()
    seconds = config.RELEVANCE_SECONDS - 1
    assert relevance.adjusted(judge, "q", index_of(), order, seconds=seconds) is order
    assert judge.read == []

    class Breaks(Judge):
        def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
            raise RuntimeError("the GPU fell over")

    said = []
    assert relevance.adjusted(Breaks(), "q", index_of(), order, say=said.append) is order
    assert said == ["[dsqa] the passages were not scored: RuntimeError: the GPU fell over"]
    assert relevance.adjusted(types.SimpleNamespace(count=len), "q", index_of(), order) is order


class Clock:
    """A clock that only the judge below moves."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Slow(Judge):
    """A judge each call of whose score() takes `step` seconds of a clock."""

    def __init__(self, clock, step, **kwargs):
        super().__init__(**kwargs)
        self.clock, self.step, self.calls = clock, step, 0

    def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
        self.calls += 1
        self.clock.now += self.step
        return super().score(conversations, labels, most_tokens)


class Picky(Judge):
    """Yes to a passage with "lithium" in it, and a clear No to any other."""

    def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
        self.read.extend(conversations)
        rows = []
        for conversation in conversations:
            passage = conversation[1]["content"].split("\n\nQuestion:")[0]
            rows.append([3.0, 0.0] if "lithium" in passage else [0.0, 3.0])
        return rows


def many_chunks(count=100):
    """`count` chunks of four to a document; every third speaks of lithium."""
    chunks = [
        f"Doc {number // 4}\n\n{'lithium' if number % 3 == 0 else 'other'} {number}"
        for number in range(count)
    ]
    return index_of(chunks, np.asarray([number // 4 for number in range(count)]))


def test_llm_wide_scores_twice_as_many(monkeypatch):
    index, order = many_chunks(), np.arange(100)
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank")
    _, report = relevance.judged(Judge(), "q", index, order)
    assert (report.asked, report.scored) == (config.RELEVANCE_N, config.RELEVANCE_N) == (40, 40)
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-wide")
    _, report = relevance.judged(Judge(), "q", index, order)
    assert (report.asked, report.scored) == (config.RELEVANCE_WIDE_N, 80) == (80, 80)
    # llm-wide alone ranks nothing.
    monkeypatch.setenv("DSQA_VARIANT", "llm-wide")
    same, report = relevance.judged(Judge(), "q", index, order)
    assert same is order and report.line() == "not applied (llm-wide alone changes nothing)"


def test_the_scoring_stops_for_time_and_what_was_scored_is_used(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-wide")
    monkeypatch.setattr(config, "RELEVANCE_SLICE", 20)
    monkeypatch.setattr(config, "RELEVANCE_MOST_SECONDS", 30)
    index, order = many_chunks(), np.arange(100)
    clock = Clock()
    judge = Slow(clock, step=12.0)
    new, report = relevance.judged(judge, "q", index, order, clock=clock)
    # 12 s a slice of 20: after two a third would end at 36 s, past the 30.
    assert judge.calls == 2 and (report.scored, report.asked) == (40, 80)
    assert report.stopped == "time" and report.why is None
    assert sorted(new.tolist()) == list(range(100))  # nothing is lost
    assert new[40:].tolist() == list(range(40, 100))  # and the unscored keep their place
    assert report.line().startswith("llm-rank scored 40 of 80 in 24.0 s (stopped for time), ")
    # Half of what is left of the question's time is the most it may take.
    clock.now = 0.0
    judge = Slow(clock, step=12.0)
    _, report = relevance.judged(judge, "q", index, order, seconds=40, clock=clock)
    assert judge.calls == 1 and report.scored == 20
    # The first slice is asked whatever it takes.
    clock.now = 0.0
    judge = Slow(clock, step=100.0)
    _, report = relevance.judged(judge, "q", index, order, clock=clock)
    assert judge.calls == 1 and report.scored == 20 and report.stopped == "time"


def test_the_gate_leaves_out_none_of_the_first_n_that_time_left_unscored(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-gate,llm-wide")
    monkeypatch.setattr(config, "RELEVANCE_SLICE", 20)
    index, order = many_chunks(), np.arange(100)
    lithium = [number for number in range(100) if number % 3 == 0]
    # All 80 scored: what the model turns down and all after the 80 are out.
    new, report = relevance.judged(Picky(), "q", index, order)
    assert sorted(new.tolist()) == [number for number in lithium if number < 80]
    assert (report.kept, report.spared, report.scored) == (27, 0, 80)
    assert "the gate kept 27 of 80, first document 0 as before" in report.line()

    class Timed(Picky):
        def __init__(self, clock):
            super().__init__()
            self.clock = clock

        def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
            self.clock.now += 12.0
            return super().score(conversations, labels, most_tokens)

    clock = Clock()
    new, report = relevance.judged(Timed(clock), "q", index, order, clock=clock)
    # 40 scored, 14 of them kept; the 40 the time left unscored keep their place.
    assert sorted(new.tolist()[:14]) == [number for number in lithium if number < 40]
    assert new.tolist()[14:] == list(range(40, 80))
    assert (report.kept, report.spared) == (14, 40)
    assert "the gate kept 14 of 40 and 40 unscored" in report.line()


def test_the_report_says_why_the_ranking_was_left_as_it_was(monkeypatch):
    index, order = index_of(), np.asarray([2, 4, 0, 3, 1, 5])
    monkeypatch.setenv("DSQA_VARIANT", "")
    assert relevance.judged(Judge(), "q", index, order)[1].line() == (
        "not applied (no llm- adjustment chosen)"
    )
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank")
    cannot = types.SimpleNamespace(count=len)
    assert relevance.judged(cannot, "q", index, order)[1].line() == (
        "not applied (the engine cannot score)"
    )
    assert relevance.judged(Judge(), "q", index, None) == (
        None,
        relevance.Judged(("llm-rank",), why="no ranking"),
    )
    assert relevance.judged(Judge(), "q", index, order, seconds=10)[1].line() == (
        "not applied (10 s left, under 15)"
    )
    assert relevance.judged(Judge(fails=True), "q", index, order)[1].line() == (
        "not applied (no scores came back, out of memory)"
    )
    new, report = relevance.judged(Judge(), "Where is the lithium?", index, order)
    assert report.line().startswith("llm-rank scored 6 of 6 in 0.0 s, first document ")
    assert (report.first_before, report.first_after) == (1, int(DOCUMENTS[new[0]]))


class Gives(Judge):
    """Gives every passage the logits `row` makes of its number in the
    ranking it was asked for, counted over the calls."""

    def __init__(self, row):
        super().__init__()
        self.row, self.asked = row, 0

    def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
        self.read.extend(conversations)
        rows = [self.row(self.asked + at) for at in range(len(conversations))]
        self.asked += len(conversations)
        return rows


def test_scores_that_rank_nothing_are_told_apart_from_those_that_rank():
    assert relevance.unusable([0.2, 0.9, 0.5]) is None
    assert relevance.unusable([0.5]) is None  # a lone score: nothing to put out of place
    assert relevance.unusable([0.5, 0.5]) == "the scores rank nothing: all 2 are 0.5000"
    assert relevance.unusable([0.1, float("nan"), 0.3]) == (
        "the scores rank nothing: 1 of 3 not finite"
    )
    assert relevance.unusable([float("inf"), float("-inf")]) == (
        "the scores rank nothing: 2 of 2 not finite"
    )
    # Infinite logits are scores of 0 and 1, which rank; NaN logits are not.
    assert relevance._sigmoid(float("inf")) == 1.0 and relevance._sigmoid(float("-inf")) == 0.0
    assert np.isnan(relevance._sigmoid(float("nan")))


def test_scores_that_rank_nothing_leave_the_ranking_as_it_was_and_say_why(monkeypatch):
    # Every adjustment that makes a ranking of the scores, llm-gate among
    # them, which would otherwise keep only the best few by their place.
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-gate,llm-doc,llm-wide")
    index, order = many_chunks(), np.arange(100)
    nan = float("nan")
    cases = {
        "the scores rank nothing: 80 of 80 not finite": lambda at: [nan, 0.0],
        "the scores rank nothing: all 80 are 0.8808": lambda at: [2.0, 0.0],
        "the scores rank nothing: 1 of 80 not finite": lambda at: (
            [nan, 0.0] if at == 57 else [3.0 if at % 3 == 0 else -3.0, 0.0]
        ),
        "the scores rank nothing: 2 of 80 not finite": lambda at: (
            [float("inf"), float("inf")] if at in (0, 79) else [float(at % 5), 0.0]
        ),
    }
    for why, row in cases.items():
        judge = Gives(row)
        same, report = relevance.judged(judge, "q", index, order)
        assert same is order and len(judge.read) == 80
        assert (report.why, report.asked, report.scored) == (why, 80, 80)
        assert report.kept is None and report.first_after is None
        assert report.line().startswith(f"not applied ({why})")
    # The same scores without the NaN are real ones, and they rank.
    real = Gives(lambda at: [3.0 if at % 3 == 0 else -3.0, 0.0])
    new, report = relevance.judged(real, "q", index, order)
    assert report.why is None and report.kept == 27
    assert new.tolist() != order.tolist()


def test_judged_never_raises(monkeypatch):
    # Chunks 0 and 2 score apart, so the scoring goes on to llm-doc.
    order = np.asarray([0, 2])
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-doc")
    no_documents = types.SimpleNamespace(chunks=CHUNKS)
    same, report = relevance.judged(Judge(), "q", no_documents, order)
    assert same is order and report.why.startswith("the scoring failed: AttributeError")
    monkeypatch.setenv("DSQA_VARIANT", "no-such-adjustment")
    same, report = relevance.judged(Judge(), "q", index_of(), order)
    assert same is order and report.why.startswith("the adjustments could not be read")


class OneByOne(Judge):
    """The judge with first() beside score(): one conversation at a time,
    the same logits as score() gives it; None once it is told to run out."""

    def __init__(self, room=None, **kwargs):
        super().__init__(**kwargs)
        self.room, self.firsts, self.batches = room, 0, 0

    def score(self, conversations, labels=("Yes", "No"), most_tokens=8000):
        self.batches += 1
        return super().score(conversations, labels, most_tokens)

    def first(self, conversation, labels=("Yes", "No")):
        self.firsts += 1
        if self.room is not None and self.firsts > self.room:
            return None
        (row,) = Judge.score(self, [conversation], labels)
        return row


def test_llm_one_scores_each_chunk_alone_and_ranks_as_the_batches_do(monkeypatch):
    index, order = many_chunks(), np.arange(100)
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-wide")
    batched, report = relevance.judged(OneByOne(), "q", index, order)
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-wide,llm-one")
    judge = OneByOne()
    alone, alone_report = relevance.judged(judge, "q", index, order)
    assert alone.tolist() == batched.tolist()
    assert (judge.firsts, judge.batches) == (80, 0)
    assert alone_report.alone and not report.alone
    assert alone_report.line().startswith("llm-rank scored 80 of 80 one by one in ")
    assert report.line().startswith("llm-rank scored 80 of 80 in ")


def test_llm_one_without_first_or_out_of_memory_leaves_the_ranking(monkeypatch):
    index, order = many_chunks(), np.arange(100)
    monkeypatch.setenv("DSQA_VARIANT", "llm-rank,llm-wide,llm-one")
    same, report = relevance.judged(Judge(), "q", index, order)
    assert same is order and report.why == "the engine cannot score one by one"
    # Out of memory in the first slice: nothing scored; in the second: what was.
    same, report = relevance.judged(OneByOne(room=5), "q", index, order)
    assert same is order and report.why == "no scores came back, out of memory"
    monkeypatch.setattr(config, "RELEVANCE_SLICE", 20)
    _, report = relevance.judged(OneByOne(room=30), "q", index, order)
    assert (report.scored, report.stopped, report.why) == (20, "memory", None)
    # llm-one alone, as llm-wide alone, ranks nothing.
    monkeypatch.setenv("DSQA_VARIANT", "llm-one,llm-wide")
    same, report = relevance.judged(OneByOne(), "q", index, order)
    assert same is order and report.line() == "not applied (llm-one+llm-wide alone changes nothing)"
