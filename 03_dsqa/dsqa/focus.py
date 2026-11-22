"""The document a question is about, and a context made of it (doc-focus).

Every question of the challenge is written from one document. The entry's
context is cut from a ranking of chunks, and so it holds pieces of many
documents: on the night's line of the entry, eight on average in a context
of 10,000 tokens, of which the question's own documents are 58% on the 98
tuning questions. A paper of 20,000 tokens, the median length of the AI
papers, is then shown in the pieces that look most like the question,
without the table or the section that answers it.

The documents are ranked for a question two ways, from what the index has
of it already: by their best chunk in the entry's ranking, which fuses
meaning and words, and by the sum of their three best chunks in the ranking
by words alone. When the two put the same document first, the question is
taken to be about it, and with doc-focus (variant.py) its context is that
document alone: whole when it fits, else its chunks by their rank, each
followed by the one after it and the one before, so that a passage cut at a
chunk's edge is read on.

A question the routing tells to be on IFMIF-DONES (route.decided) is not
focused: the plant's documents describe the same systems at other stages of
its design, and on the expert-style questions the document of the best
chunk is the wrong one of the candidates one time in three.

This is the night's focus.py cut down to that one case, over the first
entry's index and its context.pack(), which counts each chunk with the
title it opens with: a document goes in whole less often than on the
night's line, which counts what is printed.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from . import domain, prompt, route, variant
from .context import pack

# What a question's context is made of.
ONE, AS_BEFORE = "one document", "as before"
# How many of a document's best chunks in the ranking by words are summed to
# rank it by words, and how many documents of the ranking of chunks are
# candidates besides the first by words.
WORDS_CHUNKS = 3
FIRST_DOCUMENTS = 2


@dataclass
class Focus:
    """What a question's context is made of: ONE or AS_BEFORE; the documents
    it is made of; whether the two rankings put the same document first; the
    domain the routing told; why the context is as before, when it is; and,
    for ONE, whether the document went in whole (placed() says)."""

    how: str
    documents: list[int] = field(default_factory=list)
    agreed: bool = False
    told: str | None = None
    why: str = ""
    whole: bool | None = None


def chosen() -> bool:
    """Whether the entry makes a context of the document a question is about."""
    return variant.on("doc-focus")


def top_sum(scores: np.ndarray, documents: np.ndarray, count: int, k: int) -> np.ndarray:
    """For every document, the sum of the k best scores of its chunks."""
    numbers = np.lexsort((-scores, documents))
    owners = documents[numbers]
    starts = np.r_[0, np.flatnonzero(np.diff(owners)) + 1]
    sizes = np.diff(np.r_[starts, len(numbers)])
    within = np.arange(len(numbers)) - np.repeat(starts, sizes)
    kept = within < k
    return np.bincount(owners[kept], weights=scores[numbers][kept], minlength=count)


def candidates(order, documents: np.ndarray, words: np.ndarray) -> tuple[list[int], bool]:
    """The documents a question may be about, the likeliest first: the first
    FIRST_DOCUMENTS of the ranking of chunks, and the first by words when it
    is not one of them. And whether the two rankings put the same one first."""
    first: list[int] = []
    for number in order:
        document = int(documents[number])
        if document not in first:
            first.append(document)
            if len(first) == FIRST_DOCUMENTS:
                break
    count = int(documents.max()) + 1
    summed = top_sum(np.asarray(words, float), documents, count, WORDS_CHUNKS)
    # BM25's scores are never below zero (rank.Bm25: the rarity of a word is
    # a logarithm of more than one), so a best of zero is no word in common.
    best = summed.max()
    if not best > 0:
        return first, False
    leaders = np.flatnonzero(summed == best)
    by_words = int(leaders[0])
    # Two documents first by words together single out neither of them.
    agreed = bool(first) and len(leaders) == 1 and by_words == first[0]
    return first + ([by_words] if by_words not in first else []), agreed


def decided(index, question: str, order) -> Focus:
    """What the context of a question is to be made of. `index` is the
    entry's (retrieval.Index) and `order` its ranking of the chunks for the
    question."""
    if not len(order):
        return Focus(AS_BEFORE, why="nothing ranked")
    if getattr(index, "embeddings", None) is None:
        # Without the embedder the chunks are ranked by their words alone,
        # and the two rankings are one: their agreeing would say nothing.
        return Focus(AS_BEFORE, why="no embedder, so one ranking only")
    told = route.decided(question, index, order)
    if told == domain.DONES:
        return Focus(AS_BEFORE, told=told, why="told to be on IFMIF-DONES")
    documents, agreed = candidates(order, index.documents, index.bm25.scores(question))
    if agreed:
        if named():
            names = vetoed(index, question, documents, documents[0])
            if names:
                why = f"the head of document {documents[0]} does not say {', '.join(names)}"
                return Focus(AS_BEFORE, [], agreed, told, why=why)
        return Focus(ONE, documents[:1], agreed, told)
    return Focus(AS_BEFORE, [], agreed, told, why="the two rankings put different documents first")


# --- doc-name: the name a question gives, and the head of a document --------
#
# A question is focused on the document both rankings put first; when the
# question gives a name (a method, a model, a tool), that document's head is
# to say it. On the official sample, a question on SimCLR was focused on
# another paper that compares itself with SimCLR all through its text, and
# lost its passage: SimCLR's own paper names it in its abstract and not in
# its title. The rule was written before it was measured, and not moved:
#
#   name        a word of the question as written, of letters and digits
#               with inner hyphens, two characters or more, with two
#               capitals or more, or a capital after a small letter, or
#               letters and digits together; and rare: said, as written and
#               as a whole word, in at most NAME_RARE_SHARE of the
#               documents, over their text as the index holds it
#   head        a document's title and the first HEAD_CHARACTERS of its text
#   counted     the names that the head of one of the candidates says
#   the veto    a question with counted names is not focused on a document
#               whose head says none of them: its context is as before
#
# Measured without a model on v1's index: on the tuning and control sets it
# stops 1 of 66 right bets and neither of the 2 wrong ones; on the official
# sample it stops the SimCLR question and nothing else, and no passage of
# the 87 questions on AI papers is lost.

NAME_WORD = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?")
NAME_RARE_SHARE = 0.10
HEAD_CHARACTERS = 2_000


def named() -> bool:
    """Whether a question is not focused on a document whose head does not
    say the name it gives (doc-name)."""
    return variant.on("doc-name")


def shaped(word: str) -> bool:
    """Whether a word has the shape of a name."""
    capitals = sum(character.isupper() for character in word)
    inner = re.search(r"[a-z][A-Z]", word) is not None
    letters = re.search(r"[A-Za-z]", word) is not None
    digits = re.search(r"[0-9]", word) is not None
    return len(word) >= 2 and (capitals >= 2 or inner or (letters and digits))


def says(text: str, word: str) -> bool:
    """Whether a text says a word, as written and as a whole word."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text) is not None


class Heads:
    """What doc-name needs of an index: each document's text as the index
    holds it (its chunks without the title they open with), its head, and
    how many documents say a word. Made once an index (heads_of)."""

    def __init__(self, index):
        documents = index.documents
        self.count = int(documents.max()) + 1 if len(documents) else 0
        texts: dict[int, list[str]] = {}
        titles: dict[int, str] = {}
        for number, chunk in enumerate(index.chunks):
            document = int(documents[number])
            title, body = prompt.title_and_body(chunk)
            titles.setdefault(document, title)
            texts.setdefault(document, []).append(body or chunk)
        self.texts = {document: "\n".join(parts) for document, parts in texts.items()}
        self.heads = {
            document: f"{titles[document]}\n{text[:HEAD_CHARACTERS]}"
            for document, text in self.texts.items()
        }
        self.frequencies: dict[str, int] = {}

    def frequency(self, word: str) -> int:
        """How many documents say a word."""
        if word not in self.frequencies:
            self.frequencies[word] = sum(says(text, word) for text in self.texts.values())
        return self.frequencies[word]

    def names(self, question: str) -> list[str]:
        """The names a question gives, in their order."""
        found: list[str] = []
        for word in NAME_WORD.findall(question):
            word = word.strip("-")
            if shaped(word) and word not in found:
                if self.frequency(word) <= NAME_RARE_SHARE * self.count:
                    found.append(word)
        return found

    def said(self, document: int, names: list[str]) -> bool:
        """Whether a document's head says one of the names."""
        head = self.heads.get(document, "")
        return any(says(head, name) for name in names)


def heads_of(index) -> Heads:
    """The Heads of an index, made the first time they are asked for."""
    kept = getattr(index, "_focus_heads", None)
    if kept is None:
        kept = Heads(index)
        index._focus_heads = kept
    return kept


def vetoed(index, question: str, candidates_, document: int) -> list[str]:
    """The names that count for a question (those the head of one of the
    candidates says) when the head of `document` says none of them; an
    empty list when the question may be focused on it."""
    heads = heads_of(index)
    counted = [
        name for name in heads.names(question) if any(heads.said(d, [name]) for d in candidates_)
    ]
    if counted and not heads.said(document, counted):
        return counted
    return []


def _read(document: int, order, chunks, documents) -> list[int]:
    """The chunks of a document by their rank, each followed by the one after
    it and the one before."""
    read: list[int] = []
    seen: set[int] = set()
    for number in order:
        if int(documents[number]) != document:
            continue
        for beside in (int(number), int(number) + 1, int(number) - 1):
            if 0 <= beside < len(chunks) and int(documents[beside]) == document:
                if beside not in seen:
                    seen.add(beside)
                    read.append(beside)
    return read


def placed(focus: Focus, order, chunks, documents: np.ndarray, count, budget: int):
    """The chunks of a context made of the one document of `focus`, each with
    its number, and the tokens they take as `count` counts them: the document
    whole when it fits the budget; when it does not, its chunks by their
    rank, each followed by the one after it and the one before. Notes on
    `focus` whether it went in whole.

    With doc-reserve, RESERVE of the budget is kept first for the best chunks
    of the ranking, of whatever document, and the document has the rest:
    when the ranking holds pieces of another document at its top, they stay,
    and a question focused on the wrong document keeps them. The document
    comes first among the chunks placed, so that prompt.context_of() puts it
    last, next to the question."""
    document = focus.documents[0]
    every = int(np.count_nonzero(np.asarray(documents) == document))
    read = _read(document, order, chunks, documents)
    if not reserved():
        context, used = pack(read, chunks, count, budget, documents, 1.0)
        focus.whole = len(context) == every and all(text == chunks[n] for n, text in context)
        return context, used
    top, kept = pack([int(number) for number in order], chunks, count, int(budget * RESERVE))
    taken = {number for number, _ in top}
    mine, used = pack([n for n in read if n not in taken], chunks, count, budget - kept)
    context = mine + top
    whole = {number for number, text in context if int(documents[number]) == document}
    focus.whole = len(whole) == every and all(text == chunks[n] for n, text in context)
    return context, used + kept


# With doc-reserve, the share of a focused context kept for the best chunks
# of the ranking, of whatever document. Set once, before it was measured,
# and not tuned: a quarter.
RESERVE = 0.25


def reserved() -> bool:
    """Whether a focused context keeps RESERVE of it for the ranking's best."""
    return variant.on("doc-reserve")


def said(made: Focus | None, used: int, chunks, documents, asked: int) -> str:
    """What the log says of the context of the question asked last: the
    document it was made of and how, in how many tokens, or why it is the
    context as before."""
    if made is None:
        return f"[dsqa] question {asked}: not focused: the context as it was, after an error"
    if made.how != ONE:
        return f"[dsqa] question {asked}: not focused: {made.why}, {used} tokens"
    document = made.documents[0]
    first = int(np.flatnonzero(np.asarray(documents) == document)[0])
    title = prompt.title_and_body(chunks[first])[0][:80]
    how = "whole" if made.whole else "its best chunks with their neighbours"
    line = f"[dsqa] question {asked}: focused on document {document} ({title}), {how}"
    return f"{line}, {used} tokens"
