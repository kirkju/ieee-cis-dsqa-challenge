"""The base model as a judge of relevance: each of the first chunks ranked for
a question read by the model in a prompt of its own, the passage before the
question, and whether it helps answer the question read from the logits of
"Yes" and "No" at the first token, nothing written (engine.score()). To read
costs the model little, and it reads a passage better than MiniLM or BM25.

What is made of the scores, each an adjustment of its own (variant.py):

    llm-rank   the first N chunks ranked by fusing, by reciprocal rank, the
               order they came in and the order of their scores: a score
               alone ranks no better than BM25 in what has been published,
               and fused with the first stage it does
    llm-gate   with llm-rank, those of the first N the model gives less than
               config.RELEVANCE_GATE of P(Yes) left out, and nothing after the
               first N: the context may be shorter than its budget. The best
               config.RELEVANCE_KEEP are kept whatever their score, since the
               model is known to turn good passages down
    llm-doc    the document of the question taken to be the one whose best
               config.RELEVANCE_DOC_CHUNKS chunks score most, and its chunks
               put first in the order they had
    llm-pick   with llm-doc, when the first two documents by that are within
               config.RELEVANCE_PICK_MARGIN, the model shown the first three
               side by side and asked which one the question is about. It is
               asked twice, the second time in the opposite order, and the two
               are averaged: the place a document is shown in does not choose
               it. Scores read apart have no zero common to two documents, and
               to tell sibling documents apart they have to be seen together.
    llm-wide   with any of them, N is config.RELEVANCE_WIDE_N and not
               config.RELEVANCE_N: in 10,000 tokens about 36 chunks fit, so
               scoring 40 reorders what is read more than it changes it
    llm-one    with any of them, each chunk scored alone through generate()
               (engine.first()), the way every reply is read, and not in the
               batches padded on the left of engine.score(): the same scores
               but for the rounding, by a path the evaluation has run with
               every answer, at the cost of a pass for each chunk

The chunks are scored in slices of config.RELEVANCE_SLICE, in the order of
the ranking, within config.RELEVANCE_MOST_SECONDS and half of what is left of
the question's time: a slice is not begun when the one before it says it would
end past that. What was scored when the time runs out is used, and the rest
keeps its place (llm-gate leaves out none of the first N that were not
scored). A question with less than config.RELEVANCE_SECONDS left is not
scored at all.

    judged()      the ranking with the adjustments chosen, and a report of
                  what was done or why nothing was, for the log
    adjusted()    the ranking alone, as judged() makes it
    scored()      P(Yes) for each of the first N chunks of a ranking
    unusable()    why scores that came back rank nothing
    reranked()    llm-rank
    gated()       llm-gate
    document_of() the document of llm-doc
    picked()      the document of llm-pick
    put_first()   a ranking with a document's chunks first

An engine without score() leaves every one of these without effect, and a
score that cannot be had (score() gives None) leaves the ranking as it was.
So do scores that came back but rank nothing (unusable()): one that is not
a finite number, or all of them the same. Fused with the ranking, they give
it back as it was while the log says it was scored, and llm-gate and llm-doc
would keep chunks and choose a document by where they were ranked; so they
are taken for a scoring that failed, and the log says which they were.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from . import config, prompt, variant
from .rank import fuse

SYSTEM = (
    "You are a careful scientific reader. You decide whether a passage contains information "
    "that helps answer a question. Answer with one word: Yes or No."
)
ASKED = (
    'Passage from the document "{title}":\n{body}\n\nQuestion: {question}\n\n'
    "Does this passage contain information that helps answer the question? Answer Yes or No."
)
PICK_SYSTEM = (
    "You match a question to the document it was written from. Several documents are shown, "
    "each with its title and the passages of it that are closest to the question. Exactly one "
    "of them is the document the question is about."
)
PICK_ASKED = "{documents}\n\nQuestion: {question}\n\nWhich document is the question about? " + (
    "Answer with one letter."
)
LETTERS = "ABC"


WAYS = ("llm-rank", "llm-gate", "llm-doc", "llm-pick", "llm-wide", "llm-one")
# The ones that make a ranking; llm-wide only says how many are scored, and
# llm-one how.
ACTING = ("llm-rank", "llm-gate", "llm-doc", "llm-pick")


def chosen() -> set[str]:
    """The adjustments of this module the entry runs with. A name is looked
    for among those chosen, not asked of variant.on(): a version that does
    not know one of them has it off."""
    picked = variant.chosen()
    return {name for name in WAYS if name in picked}


def many(names: set[str] | None = None) -> int:
    """How many of the first chunks of a ranking are scored."""
    names = chosen() if names is None else names
    return config.RELEVANCE_WIDE_N if "llm-wide" in names else config.RELEVANCE_N


@dataclass
class Judged:
    """What the judge of relevance did for one question, as the log says it."""

    acting: tuple[str, ...] = ()
    asked: int = 0  # the chunks it was to score
    scored: int = 0
    seconds: float = 0.0
    stopped: str | None = None  # "time" or "memory" when it scored fewer than asked
    kept: int | None = None  # with llm-gate, the scored chunks it kept
    spared: int = 0  # with llm-gate, the first ones left unscored, kept as they were
    first_before: int | None = None  # the document of the first chunk, before and after
    first_after: int | None = None
    why: str | None = None  # why the ranking is the one it was, when it is
    alone: bool = False  # with llm-one: each chunk scored alone, through generate()

    def line(self) -> str:
        """The report in one line: what was done, or why nothing was."""
        if self.why is not None:
            done = f"not applied ({self.why})"
            if self.seconds >= 0.05:
                done += f" after {self.seconds:.1f} s"
            return done
        stopped = {"time": " (stopped for time)", "memory": " (stopped, out of memory)"}
        how = " one by one" if self.alone else ""
        parts = [
            f"{'+'.join(self.acting)} scored {self.scored} of {self.asked}{how} in "
            f"{self.seconds:.1f} s{stopped.get(self.stopped, '')}"
        ]
        if self.kept is not None:
            spared = f" and {self.spared} unscored" if self.spared else ""
            parts.append(f"the gate kept {self.kept} of {self.scored}{spared}")
        if self.first_before == self.first_after:
            parts.append(f"first document {self.first_after} as before")
        else:
            parts.append(f"first document {self.first_before} -> {self.first_after}")
        return ", ".join(parts)


def judged(
    engine,
    question: str,
    index,
    order,
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple:
    """The ranking a question's context is packed from, with the adjustments
    of this module that are chosen, and what was done (Judged). The ranking
    given, as it was, when none is chosen, when there is none, when the engine
    cannot score, when there are fewer than config.RELEVANCE_SECONDS of the
    question's time to score in, when no score comes back, when the scores
    that come back rank nothing (unusable()), or when anything fails; the
    report says which. This never raises."""
    began = clock()
    try:
        names = chosen()
        acting = tuple(name for name in ACTING if name in names)
    except Exception as error:
        return order, Judged(why=f"the adjustments could not be read: {_named(error)}")
    if not acting:
        alone = f"{'+'.join(sorted(names))} alone changes nothing"
        return order, Judged(why=alone if names else "no llm- adjustment chosen")
    try:
        one = "llm-one" in names
        if order is None or not len(order):
            return order, Judged(acting, why="no ranking")
        if not callable(getattr(engine, "first" if one else "score", None)):
            cannot = "the engine cannot score one by one" if one else "the engine cannot score"
            return order, Judged(acting, why=cannot)
        if seconds is not None and seconds < config.RELEVANCE_SECONDS:
            return order, Judged(
                acting, why=f"{seconds:.0f} s left, under {config.RELEVANCE_SECONDS}"
            )
        n = many(names)
        budget = config.RELEVANCE_MOST_SECONDS
        if seconds is not None:
            budget = min(budget, seconds / 2)
        scores, stopped = scored_within(engine, question, index, order, n, budget, clock, one)
        asked = min(n, len(order))
        if not scores:
            took = clock() - began
            why = "no scores came back, out of memory"
            return order, Judged(acting, asked, seconds=took, why=why, alone=one)
        why = unusable(scores)
        if why is not None:
            took = clock() - began
            return order, Judged(acting, asked, len(scores), took, stopped, why=why, alone=one)
        new = np.asarray(order, dtype=np.intp)
        head = [int(number) for number in new[:n]]
        scores_of = {head[at]: score for at, score in enumerate(scores)}
        if "llm-rank" in names or "llm-gate" in names:
            new = reranked(new, scores)
        kept, spared = None, 0
        if "llm-gate" in names:
            spare = set(head[len(scores) :])
            new = gated(new, scores_of, config.RELEVANCE_GATE, config.RELEVANCE_KEEP, spare)
            kept = sum(1 for number in new if int(number) in scores_of)
            spared = len(spare)
        if "llm-doc" in names or "llm-pick" in names:
            document = document_of(new, scores_of, index.documents, config.RELEVANCE_DOC_CHUNKS)
            if "llm-pick" in names and stopped is None and clock() - began < budget:
                other = picked(engine, question, index, new, scores_of)
                document = document if other is None else other
            new = put_first(new, index.documents, document)
        took = clock() - began
        if not len(new):
            return order, Judged(acting, asked, len(scores), took, why="the gate left nothing")
        documents = index.documents
        return new, Judged(
            acting,
            asked,
            len(scores),
            took,
            stopped,
            kept,
            spared,
            int(documents[int(order[0])]),
            int(documents[int(new[0])]),
            alone=one,
        )
    except Exception as error:
        return order, Judged(
            acting, seconds=clock() - began, why=f"the scoring failed: {_named(error)}"
        )


def adjusted(engine, question: str, index, order, seconds=None, say=print) -> np.ndarray:
    """The ranking a question's context is packed from, as judged() makes it;
    a scoring that failed is said. This never raises."""
    new, report = judged(engine, question, index, order, seconds)
    if report.why is not None and report.why.startswith("the scoring failed: "):
        try:
            say(f"[dsqa] the passages were not scored: {report.why[20:]}"[:300])
        except Exception:
            pass
    return new


def cut(text: str, tokens: int, count) -> str:
    """A text cut to about `tokens` tokens as `count` counts them."""
    had = count(text)
    if had <= tokens or not text:
        return text
    return text[: max(1, int(len(text) * tokens / had))]


def shown(index, number: int) -> tuple[str, str]:
    """The title of a chunk's document and the body it is scored by: the
    table it is a part of when the context would show that table whole,
    as the context shows it."""
    text = index.chunks[number]
    table = (getattr(index, "tables", None) or {}).get(number)
    if table is not None and getattr(config, "TABLE_WHOLE_TOKENS", 0):
        text = table[2]
    return prompt.title_and_body(text)


def conversation(question: str, title: str, body: str) -> list[dict]:
    """The prompt a passage is scored with."""
    asked = ASKED.format(title=title, body=body, question=question)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": asked}]


def scored(
    engine, question: str, index, order, n: int | None = None, one: bool = False
) -> list[float] | None:
    """P(Yes) / (P(Yes) + P(No)) for each of the first `n` chunks of a
    ranking, config.RELEVANCE_N unless given: in one call of score(), or,
    `one` (llm-one), in a call of first() for each; None when the engine
    cannot score that way or the scoring failed."""
    if not callable(getattr(engine, "first" if one else "score", None)):
        return None
    numbers = [int(number) for number in list(order)[: config.RELEVANCE_N if n is None else n]]
    if not numbers:
        return []
    conversations = []
    for number in numbers:
        title, body = shown(index, number)
        body = cut(body, config.RELEVANCE_BODY_TOKENS, engine.count)
        conversations.append(conversation(question, title, body))
    if one:
        logits = []
        for asked in conversations:
            row = engine.first(asked, ("Yes", "No"))
            if row is None:
                return None
            logits.append(row)
    else:
        logits = engine.score(conversations, ("Yes", "No"), config.SCORE_TOKENS)
    if logits is None:
        return None
    return [_sigmoid(yes - no) for yes, no in logits]


def scored_within(
    engine,
    question: str,
    index,
    order,
    n: int,
    seconds: float,
    clock: Callable[[], float] = time.monotonic,
    one: bool = False,
) -> tuple[list[float], str | None]:
    """P(Yes) for as many of the first `n` chunks of a ranking as are scored
    in `seconds`, slice by slice of config.RELEVANCE_SLICE in the order of the
    ranking, and why fewer than `n` were, when they were: "time" when a slice
    was not begun because the one before it says it would end past `seconds`,
    "memory" when a slice came back with nothing. The first slice is always
    asked. `one` scores each chunk alone (scored())."""
    numbers = [int(number) for number in list(order)[:n]]
    size = max(1, int(config.RELEVANCE_SLICE))
    found: list[float] = []
    began, last = clock(), 0.0
    for start in range(0, len(numbers), size):
        if start and clock() - began + last > seconds:
            return found, "time"
        before = clock()
        part = scored(engine, question, index, numbers[start : start + size], n=size, one=one)
        last = clock() - before
        if not part:
            return found, "memory"
        found.extend(part)
    return found, None


def unusable(scores: Sequence[float]) -> str | None:
    """Why scores that came back rank nothing, or None when they rank: one
    that is not a finite number cannot be put in order, and when two or more
    are all the same nothing is told apart by them. A lone score is let be:
    there is nothing beside it to put out of place."""
    numbers = [float(score) for score in scores]
    bad = sum(1 for number in numbers if not math.isfinite(number))
    if bad:
        return f"the scores rank nothing: {bad} of {len(numbers)} not finite"
    if len(numbers) > 1 and max(numbers) == min(numbers):
        return f"the scores rank nothing: all {len(numbers)} are {numbers[0]:.4f}"
    return None


def reranked(order, scores: Sequence[float]) -> np.ndarray:
    """The first len(scores) of a ranking by the fusion, by reciprocal rank,
    of the order they came in and the order of their scores; the rest after
    them, as they were."""
    order = np.asarray(order, dtype=np.intp)
    n = len(scores)
    if n == 0:
        return order
    head = order[:n]
    by_score = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    fused = fuse([np.arange(n), by_score])
    return np.concatenate([head[fused], order[n:]])


def gated(
    order, scores_of: dict[int, float], threshold: float, keep: int, spare=frozenset()
) -> np.ndarray:
    """Of a ranking of the scored chunks, those scored under `threshold`
    left out, the `keep` best scored kept whatever their score; the chunks
    not scored left out too, but for those in `spare`, which keep their
    place. Never empty when anything was scored."""
    best = sorted(scores_of, key=lambda number: -scores_of[number])[:keep]
    kept = [
        int(number)
        for number in order
        if (
            int(number) in scores_of
            and (scores_of[int(number)] >= threshold or int(number) in best)
        )
        or int(number) in spare
    ]
    return np.asarray(kept, dtype=np.intp)


def document_scores(scores_of: dict[int, float], documents, top: int) -> dict[int, float]:
    """For each document with a scored chunk, the sum of its `top` best scores."""
    by_document: dict[int, list[float]] = {}
    for number, score in scores_of.items():
        by_document.setdefault(int(documents[number]), []).append(score)
    return {doc: sum(sorted(found, reverse=True)[:top]) for doc, found in by_document.items()}


def document_of(order, scores_of: dict[int, float], documents, top: int) -> int | None:
    """The document whose best `top` chunks score most; between two that
    score the same, the one ranked first."""
    totals = document_scores(scores_of, documents, top)
    if not totals:
        return None
    first = {}
    for place, number in enumerate(order):
        first.setdefault(int(documents[int(number)]), place)
    return max(totals, key=lambda doc: (totals[doc], -first.get(doc, len(first))))


def put_first(order, documents, document: int | None) -> np.ndarray:
    """A ranking with the chunks of a document first, in the order they had,
    and the others after them, in theirs."""
    order = np.asarray(order, dtype=np.intp)
    if document is None or not len(order):
        return order
    mine = np.asarray(documents)[order] == document
    return np.concatenate([order[mine], order[~mine]])


def picked(engine, question: str, index, order, scores_of: dict[int, float]) -> int | None:
    """The document of the question as the model picks it among the first
    three by their chunks' scores, when the first two are within
    config.RELEVANCE_PICK_MARGIN; None when they are not, when there are
    fewer than two, or when the engine cannot tell."""
    if not callable(getattr(engine, "score", None)):
        return None
    totals = document_scores(scores_of, index.documents, config.RELEVANCE_DOC_CHUNKS)
    first = document_of(order, scores_of, index.documents, config.RELEVANCE_DOC_CHUNKS)
    ranked = sorted(totals, key=lambda doc: (doc != first, -totals[doc]))[: len(LETTERS)]
    if len(ranked) < 2 or totals[ranked[0]] - totals[ranked[1]] >= config.RELEVANCE_PICK_MARGIN:
        return None
    cards = [_card(index, doc, scores_of, engine.count) for doc in ranked]
    letters = LETTERS[: len(ranked)]
    shown_in = [list(range(len(ranked))), list(reversed(range(len(ranked))))]
    conversations = [_pick_conversation(question, [cards[at] for at in way]) for way in shown_in]
    logits = engine.score(conversations, tuple(letters), config.SCORE_TOKENS)
    if logits is None:
        return None
    chances = np.zeros(len(ranked))
    for way, row in zip(shown_in, logits, strict=True):
        probabilities = _softmax(row)
        for place, at in enumerate(way):
            chances[at] += probabilities[place] / len(shown_in)
    return int(ranked[int(np.argmax(chances))])


def _card(index, document: int, scores_of: dict[int, float], count) -> str:
    """A document as llm-pick shows it: its title, and its two best scored
    passages of up to config.RELEVANCE_CARD_TOKENS tokens each."""
    mine = [n for n in scores_of if int(index.documents[n]) == document]
    best = sorted(mine, key=lambda number: -scores_of[number])[:2]
    title = shown(index, best[0])[0] if best else ""
    passages = [
        cut(shown(index, number)[1], config.RELEVANCE_CARD_TOKENS, count) for number in best
    ]
    return "\n".join([title, *passages])


def _pick_conversation(question: str, cards: list[str]) -> list[dict]:
    documents = "\n\n".join(
        f"Document {letter}: {card}" for letter, card in zip(LETTERS, cards, strict=False)
    )
    asked = PICK_ASKED.format(documents=documents, question=question)
    return [{"role": "system", "content": PICK_SYSTEM}, {"role": "user", "content": asked}]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    small = math.exp(value)
    return small / (1.0 + small)


def _softmax(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    exponents = np.exp(values - values.max())
    return exponents / exponents.sum()


def _named(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:200]
