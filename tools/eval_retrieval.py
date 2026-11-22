"""Retrieval measured without a model: does a passage that answers each
question reach the context?

A retrieval system is run over the development corpus and the sample
questions. Each reference passage (dev/questions/sample_evidence.json) is
then followed through the system's stages, so that one that does not arrive
is charged to the stage that lost it:

    parsed   the document's parsed text holds the passage
    chunked  so do its chunks, read one after the other
    ranked   how far down the ranking one has to read to have it
    context  it is in the context the model would be given

    python -m tools.eval_retrieval --model-id <hf repo id>
    python -m tools.eval_retrieval --model-id <id> --ids 3,8
    python -m tools.eval_retrieval --model-id <id> --record --dataset sample-25 --corpus dev-150
    python -m tools.eval_retrieval --model-id <id> --system lab:ranker=bm25
    python -m tools.eval_retrieval --model-id <id> --system 03_dsqa

The model id names the tokenizer the context budget is counted with; no model
is loaded. Without --system, what is measured is the organizers' RAG example
as it is; "lab:" and options give it with stages swapped
(tools/retrieval_lab.py); 03_dsqa is our entry, through the code its answer()
runs (tools/entry_system.py).

A question counts as covered when a passage that answers it is whole in the
context. Passages annotated with the same "group" answer it only together,
and then all of them have to be there; a passage without a group answers by
itself. The annotations of the organizers' sample have no groups: they do not
say which passages are needed together and which state the same fact twice,
so for the sample any one passage counts, which is the lenient reading. The
passages are also counted one by one.

How a passage is recognized. A quote was cut from the text PyMuPDF extracts
from a page. Another parser spaces, hyphenates, orders and marks up the same
text differently, so a quote cannot be searched for as a string. Both texts
are reduced to their letters and digits, and a character of the quote counts
as found when it lies in a run of at least SHINGLE characters that the text
also has. The coverage of a passage by a text is the fraction of its quote
found there: the passage is "whole" from WHOLE up and "in part" from PART up.

A passage cut from a table is recognized by its words instead. A table has no
one order to be read in: PyMuPDF gives its cells as they come in the file, row
by row or column by column, and a parser that sets them in rows gives another
order of the same table. So the quote of a table is taken as its words, each
reduced to its letters and digits and counted as often as the quote has it,
and its coverage by a text is the fraction of them found there, in whatever
order.

What counts is what one place holds. Phrases of a quote turn up all over the
document it comes from, and added together they can look like the passage
without it being anywhere; so a text holds a passage as far as a single
stretch of it does, about twice the quote's length. Chunks that follow one
another in a document are read as one text: a passage the chunker cut in two
is there when both halves were retrieved. The lines a chunk opens with that
the chunk before it has too, as the title of the document, the heading of the
section, the caption of a table cut in parts or its row of headings, are read
once: repeated before every part, they would push the parts of one passage
out of a single stretch. A passage is looked for only in the document its
annotation names.

Parsing is the slow stage and gives the same text every time, so the text of
each document is kept under data/cache/, by parser and by the hash of the
file, together with the time the parsing took.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pymupdf

from tools import corpus, evidence, ledger, run_local

REPO_ROOT = ledger.REPO_ROOT
CACHE_DIR = REPO_ROOT / "data" / "cache" / "parse"
# A recorded run keeps its per-question results in the repository, so that a
# later variant can be compared with it question by question.
SCRATCH_OUT = REPO_ROOT / "runs"
RECORDED_OUT = REPO_ROOT / "results" / "runs"

# Length of a run of characters, and how much of a quote has to be found for
# the passage to count as whole or as in part. Set on the development corpus
# (journal, 2026-10-08): with runs of 12 every quote is covered 1.00 by its own
# page and at most 0.45 by another page of its document.
SHINGLE = 12
WHOLE = 0.8
PART = 0.5
# A chunk is looked at for a passage of a table when it has this share of the
# passage's words: any text has a few of them.
SHARED = 0.05
# Names the rules a passage is recognized by, for the ledger. 2: a passage of
# a table by its words, and what consecutive chunks open with read once.
RULES = 2

# How far down the ranking coverage is reported.
DEPTHS = (1, 5, 10, 20, 50)


def squash(text: str) -> str:
    """The letters and digits of a text, case folded and with compatibility
    characters unfolded: what is left once spacing, hyphenation, punctuation
    and markup are taken away."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def spell(text: str) -> tuple[str, ...]:
    """The words of a text, each reduced to its letters and digits. What has
    neither, as the bars of a Markdown table, is not a word."""
    return tuple(word for word in map(squash, text.split()) if word)


def continuation(before: str, chunk: str) -> str:
    """What a chunk adds when it is read after the chunk before it: its text
    without the lines it opens with that the other has too, whole or as the
    start of a longer line. A part of a table may open with the title of the
    document, the first sentence of the caption and the row of headings."""
    had = [line for line in before.split("\n") if line.strip()]
    lines = chunk.split("\n")
    new = 0
    while new < len(lines) and (
        not lines[new].strip() or any(line.startswith(lines[new]) for line in had)
    ):
        new += 1
    return "\n".join(lines[new:])


class Passage:
    """A reference passage of running text, in the form it is looked for.

    read() gives the form of a text in which the passage can be looked for,
    and together() that of several read one after the other. The texts the
    other methods are given have been through read() already.
    """

    form = "letters"
    read = staticmethod(squash)
    together = staticmethod("".join)

    def __init__(self, doc_id: str, page: int, kind: str, quote: str, shingle: int = SHINGLE):
        self.doc_id, self.page, self.kind = doc_id, page, kind
        self.letters = squash(quote)
        if not self.letters:
            raise ValueError(f"a quote without letters or digits cannot be looked for: {quote!r}")
        self.size = len(self.letters)
        self.shingle = min(shingle, len(self.letters))
        # Where in the quote each run starts; a run can occur more than once.
        self._starts: dict[str, list[int]] = {}
        for start in range(len(self.letters) - self.shingle + 1):
            self._starts.setdefault(self.letters[start : start + self.shingle], []).append(start)

    def _runs(self, text: str) -> tuple[list[int], list[int]]:
        """The runs text shares with the quote: where each starts in text and
        where in the quote."""
        in_text, in_quote = [], []
        for at in range(len(text) - self.shingle + 1):
            starts = self._starts.get(text[at : at + self.shingle])
            if starts:
                in_text.extend([at] * len(starts))
                in_quote.extend(starts)
        return in_text, in_quote

    def _marks(self, in_quote) -> np.ndarray:
        """The characters of the quote that lie in one of the runs found."""
        starts = np.zeros(len(self.letters) - self.shingle + 1, dtype=np.int32)
        starts[np.asarray(in_quote, dtype=np.intp)] = 1
        return np.convolve(starts, np.ones(self.shingle, dtype=np.int32)) > 0

    def found(self, text: str) -> np.ndarray:
        """Which characters of the quote text holds, one flag per character."""
        return self._marks(self._runs(text)[1])

    def shares(self, text: str) -> bool:
        """Whether text has any of the quote."""
        return bool(self._runs(text)[0])

    def best_stretch(self, text: str, lengths: int = 2) -> float:
        """The coverage of the quote by the stretch of text that holds most of it.

        A stretch is `lengths` times as long as the quote. Runs scattered over
        a long document do not add up to the passage being there, so what
        counts is what a single stretch holds; stretches are tried a quarter of
        their length apart.
        """
        in_text, in_quote = self._runs(text)
        if not in_text:
            return 0.0
        in_text, in_quote = np.asarray(in_text), np.asarray(in_quote)
        width = lengths * len(self.letters)
        last = max(len(text) - width, 0)
        best = 0.0
        for start in [*range(0, last, max(width // 4, 1)), last]:
            first, end = np.searchsorted(in_text, [start, start + width - self.shingle + 1])
            if end > first:
                best = max(best, float(self._marks(in_quote[first:end]).mean()))
        return best


class TablePassage:
    """A reference passage cut from a table, looked for by its words.

    It answers as a Passage does. A text is read as its words, and holds the
    passage as far as one stretch of it has the words of the quote, each as
    often as the quote has it and in whatever order.
    """

    form = "words"
    read = staticmethod(spell)

    def __init__(self, doc_id: str, page: int, kind: str, quote: str):
        self.doc_id, self.page, self.kind = doc_id, page, kind
        words = spell(quote)
        if not words:
            raise ValueError(f"a quote without letters or digits cannot be looked for: {quote!r}")
        self.size = len(words)
        self._number = {word: number for number, word in enumerate(dict.fromkeys(words))}
        self._times = np.bincount([self._number[word] for word in words])

    @staticmethod
    def together(texts: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
        return tuple(word for text in texts for word in text)

    def _numbers(self, text: tuple[str, ...]) -> np.ndarray:
        """The words of a text by their number in the quote; -1 for the others."""
        numbers = (self._number.get(word, -1) for word in text)
        return np.fromiter(numbers, dtype=np.int64, count=len(text))

    def _share(self, numbers: np.ndarray) -> float:
        """How much of the quote these words are: none counts more often than
        the quote has it."""
        had = np.bincount(numbers[numbers >= 0], minlength=len(self._times))
        return float(np.minimum(had, self._times).sum()) / self.size

    def shares(self, text: tuple[str, ...]) -> bool:
        """Whether text has enough of the quote's words to be worth reading."""
        return self._share(self._numbers(text)) >= SHARED

    def best_stretch(self, text: tuple[str, ...], lengths: int = 2) -> float:
        """The coverage of the quote by the stretch of text that holds most of
        its words, a stretch being `lengths` times as long as the quote."""
        numbers = self._numbers(text)
        if not (numbers >= 0).any():
            return 0.0
        width = lengths * self.size
        last = max(len(numbers) - width, 0)
        starts = [*range(0, last, max(width // 4, 1)), last]
        return max(self._share(numbers[start : start + width]) for start in starts)


def passage_of(entry: dict) -> Passage | TablePassage:
    """A passage as its annotation gives it, in the form its kind is looked for."""
    kind = TablePassage if entry["kind"] == "table" else Passage
    return kind(entry["doc_id"], entry["page"], entry["kind"], entry["quote"])


@dataclass
class Retrieved:
    """What a system retrieved for one question."""

    order: Sequence[int]  # chunks by their number, best first
    context: list[tuple[int, str]]  # the chunks placed in the context, with the text each gives
    tokens: int = 0  # as the system counts them


class System(Protocol):
    """A retrieval system, as far as this tool needs to know one.

    chunk() is called once per document, in the order the documents were
    given, and index() once with all the chunks: a chunk's number is its
    place in that list. A system may also have verify(question, retrieved),
    called outside the timing, to check what it reported.
    """

    name: str
    parse_key: str  # names the parser and its settings; parsed text is kept under it
    config: dict  # what was run, for the ledger

    def parse(self, pdf: Path) -> str: ...

    def chunk(self, text: str) -> list[str]: ...

    def index(self, chunks: list[str]) -> None: ...

    def retrieve(self, question: str) -> Retrieved: ...


@dataclass
class Document:
    """A PDF of the corpus and what the parser made of it."""

    doc_id: str
    sha256: str
    pages: int
    text: str | None  # None when the parser failed on it
    seconds: float  # that the parsing took, now or when the text was kept
    cached: bool
    error: str = ""


def _pages(path: Path) -> int:
    try:
        with pymupdf.open(path) as document:
            return document.page_count
    except Exception:
        return 0


def read_corpus(
    system: System,
    pdfs: Iterable[Path],
    cache_dir: Path | None = CACHE_DIR,
    *,
    reparse: bool = False,
    clock=time.perf_counter,
    say=print,
) -> list[Document]:
    """Parse every PDF, or read back the text kept from an earlier run.

    A document the parser fails on is kept without text and the run goes on,
    as it would in a submission: one bad PDF must not take the others down.
    """
    pdfs = list(pdfs)
    documents = []
    for number, path in enumerate(pdfs, 1):
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        kept = cache_dir / system.parse_key / f"{sha256}.json" if cache_dir else None
        if kept is not None and kept.exists() and not reparse:
            stored = json.loads(kept.read_text(encoding="utf-8"))
            document = Document(
                path.stem, sha256, stored["pages"], stored["text"], stored["seconds"], True
            )
        else:
            started = clock()
            try:
                text, error = system.parse(path), ""
            except Exception as e:
                text, error = None, f"{type(e).__name__}: {e}"
                say(f"[retrieval] {path.name}: the parser failed: {error}")
            seconds = clock() - started
            document = Document(path.stem, sha256, _pages(path), text, seconds, False, error)
            if text is not None and kept is not None:
                kept.parent.mkdir(parents=True, exist_ok=True)
                partial = kept.with_suffix(".part")
                stored = {"pages": document.pages, "seconds": document.seconds, "text": text}
                partial.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
                os.replace(partial, kept)
        documents.append(document)
        if number % 25 == 0 or number == len(pdfs):
            fresh = [document for document in documents if not document.cached]
            say(
                f"[retrieval] {number} of {len(pdfs)} documents read, {len(fresh)} of them "
                f"parsed now in {sum(document.seconds for document in fresh):.0f}s"
            )
    return documents


def fingerprint(documents: list[Document]) -> str:
    """A short name for exactly this set of files."""
    lines = sorted(f"{document.doc_id}:{document.sha256}" for document in documents)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def manifest_differences(documents: list[Document], manifest: list[dict]) -> dict[str, list[str]]:
    """How the files that were read differ from what the manifest describes."""
    listed = {row["doc_id"]: row["sha256"] for row in manifest}
    read = {document.doc_id: document.sha256 for document in documents}
    return {
        "not_in_the_manifest": sorted(set(read) - set(listed)),
        "not_the_file_described": sorted(
            doc_id for doc_id in set(read) & set(listed) if read[doc_id] != listed[doc_id]
        ),
        "not_in_the_folder": sorted(set(listed) - set(read)),
    }


@dataclass
class Located:
    """Where a passage is in what a system made of its document."""

    passage: Passage
    parsed: float  # how much of it one place in the parsed text holds
    chunked: float  # the same in the document's chunks, read one after the other
    one_chunk: float  # how much of it a single chunk holds
    shares: list[int]  # the chunks that share a run with it
    group: str | None = None  # the passages it is needed together with, if any


def _held(passage, letters: dict, added: dict | None = None) -> float:
    """How much of a passage one place holds, in chunks given by their numbers.

    Chunks that follow one another in their document are read as one text, so
    a passage the chunker cut in two counts when both halves are there. Chunks
    apart are read apart: what each shares with the quote does not add up.

    `added` has, for a chunk, what it adds to the chunk before it. A chunk
    read after that one is read so, without the opening the two share.
    """
    added = added or {}
    numbers = sorted(letters)
    best, start = 0.0, 0
    for end in range(1, len(numbers) + 1):
        if end == len(numbers) or numbers[end] != numbers[end - 1] + 1:
            first, *following = numbers[start:end]
            read = [letters[first], *(added.get(number, letters[number]) for number in following)]
            best = max(best, passage.best_stretch(passage.together(read)))
            start = end
    return best


def _rank(located: Located, position: dict[int, int], letters: dict, added: dict) -> int | None:
    """How many chunks down the ranking one has to read to have the passage.

    The chunks that share text with the quote are taken as they were ranked,
    each read together with the chunks next to it in the document that were
    ranked above it.
    """
    passage = located.passage
    reach = 2 * passage.size
    for number in sorted((n for n in located.shares if n in position), key=position.get):
        depth = position[number]
        read = {number: letters[number]}
        for step in (-1, 1):
            beside, length = number + step, 0
            while beside in letters and position.get(beside, depth + 1) <= depth and length < reach:
                read[beside] = letters[beside]
                length += len(added.get(beside, letters[beside]))
                beside += step
        if _held(passage, read, added) >= WHOLE:
            return depth
    return None


def _follow(
    located: Located, retrieved: Retrieved, position: dict, letters: dict, added: dict, chunks
) -> dict:
    """One passage through the stages, for one question. letters holds the
    chunks of the passage's document by their numbers, in the form the
    passage is looked for in, and added what each adds to the one before it."""
    passage = located.passage
    placed, adding = {}, {}
    for number, text in retrieved.context:
        if number not in letters:
            continue
        if text == chunks[number]:
            placed[number] = letters[number]
            if number in added:
                adding[number] = added[number]
        else:
            # A system may cut or rewrite a chunk: what counts is what it placed.
            placed[number] = passage.read(text)
    result = {
        "doc_id": passage.doc_id,
        "page": passage.page,
        "kind": passage.kind,
        "parsed": round(located.parsed, 3),
        "chunked": round(located.chunked, 3),
        "one_chunk": round(located.one_chunk, 3),
        "rank": _rank(located, position, letters, added),
        "context": round(_held(passage, placed, adding), 3),
    }
    result["lost"] = _lost(result)
    if located.group is not None:
        result["group"] = located.group
    return result


# The stages a passage can be lost in, in the order it goes through them.
STAGES = ("parsing", "chunking", "ranking")


def _lost(measured: dict) -> str | None:
    """The stage that lost a passage, or None when it is whole in the context."""
    if measured["context"] >= WHOLE:
        return None
    if measured["parsed"] < WHOLE:
        return "parsing"
    if measured["chunked"] < WHOLE:
        return "chunking"
    return "ranking"


def _ways(passages: list[dict]) -> list[list[dict]]:
    """The ways a question can be answered from its passages: by any one that
    stands alone, or by all the passages of a group together."""
    alone, together = [], {}
    for passage in passages:
        if passage.get("group") is None:
            alone.append([passage])
        else:
            together.setdefault(passage["group"], []).append(passage)
    return [*alone, *together.values()]


def _question(passages: list[dict]) -> dict:
    """What a question's passages amount to: the best of the ways to answer
    it, each way being as good as the worst of the passages it needs."""
    names = ("parsed", "chunked", "one_chunk", "context")
    if not passages:
        return {"lost": None, **dict.fromkeys(names), "rank": None}
    ways = _ways(passages)

    def reached(way: list[dict]) -> int:
        # A way is lost at the first stage that loses one of its passages.
        lost = [STAGES.index(passage["lost"]) for passage in way if passage["lost"] is not None]
        return min(lost, default=len(STAGES))

    furthest = max(reached(way) for way in ways)
    ranks = [
        max(passage["rank"] for passage in way)
        for way in ways
        if all(passage["rank"] is not None for passage in way)
    ]
    return {
        "lost": None if furthest == len(STAGES) else STAGES[furthest],
        **{name: max(min(passage[name] for passage in way) for way in ways) for name in names},
        "rank": min(ranks, default=None),
    }


def measure(
    system: System,
    questions: list[dict],
    items: list[dict],
    pdfs: Iterable[Path],
    *,
    cache_dir: Path | None = CACHE_DIR,
    reparse: bool = False,
    clock=time.perf_counter,
    say=print,
) -> dict:
    """Run a system over the corpus and the questions and follow every passage.

    Returns the documents, one row per question, the number of chunks and the
    time each stage took.
    """
    started = clock()
    documents = read_corpus(system, pdfs, cache_dir, reparse=reparse, clock=clock, say=say)
    texts = {document.doc_id: document.text for document in documents}

    chunking = clock()
    chunks: list[str] = []
    owner: list[str] = []  # the document each chunk comes from
    for document in documents:
        if document.text is not None:
            pieces = system.chunk(document.text)
            chunks.extend(pieces)
            owner.extend([document.doc_id] * len(pieces))
    chunk_s = clock() - chunking
    of_document: dict[str, list[int]] = {}
    for number, doc_id in enumerate(owner):
        of_document.setdefault(doc_id, []).append(number)
    say(f"[retrieval] {len(documents)} documents in {len(chunks)} chunks; indexing them")

    indexing = clock()
    system.index(chunks)
    index_s = clock() - indexing

    # Every passage against its own document: the parsed text, the chunks and
    # what each chunk adds to the one before it, in the form the passage is
    # looked for in.
    read: dict[tuple[str, str], tuple] = {}
    located: dict = {}
    for item in items:
        located[item["id"]] = mine = []
        for entry in item.get("evidence", []):
            passage = passage_of(entry)
            doc_id = passage.doc_id
            if (doc_id, passage.form) not in read:
                numbers = of_document.get(doc_id, [])
                read[doc_id, passage.form] = (
                    passage.read(texts.get(doc_id) or ""),
                    {number: passage.read(chunks[number]) for number in numbers},
                    {
                        number: passage.read(continuation(chunks[number - 1], chunks[number]))
                        for number in numbers[1:]
                    },
                )
            parsed, letters, added = read[doc_id, passage.form]
            shares = [number for number, text in letters.items() if passage.shares(text)]
            mine.append(
                Located(
                    passage,
                    parsed=passage.best_stretch(parsed),
                    chunked=_held(passage, letters, added),
                    one_chunk=max((passage.best_stretch(letters[n]) for n in shares), default=0.0),
                    shares=shares,
                    group=entry.get("group"),
                )
            )

    statuses = {item["id"]: item.get("status") for item in items}
    rows, retrieve_s = [], []
    for question in questions:
        asked = clock()
        retrieved = system.retrieve(question["question"])
        retrieve_s.append(clock() - asked)
        if hasattr(system, "verify"):
            system.verify(question["question"], retrieved)

        position = {number: rank for rank, number in enumerate(retrieved.order, 1)}
        passages = [
            _follow(
                found,
                retrieved,
                position,
                *read[found.passage.doc_id, found.passage.form][1:],
                chunks,
            )
            for found in located.get(question.get("id"), [])
        ]
        placed = [number for number, _ in retrieved.context]
        sources = {passage["doc_id"] for passage in passages}
        ranks = [
            position[number]
            for doc_id in sources
            for number in of_document.get(doc_id, [])
            if number in position
        ]
        row = {
            "id": question.get("id"),
            "domain": question.get("domain"),
            "question_type": question.get("question_type"),
            "status": statuses.get(question.get("id")),
            "passages": passages,
            # Where it was lost, and how it does at each stage.
            **_question(passages),
            # The first chunk of a document that holds one of its passages.
            "document_rank": min(ranks, default=None),
            "document_in_context": any(owner[number] in sources for number in placed),
            "context_chunks": len(placed),
            "context_tokens": retrieved.tokens,
            "context_documents": len({owner[number] for number in placed}),
            "top_documents": list(dict.fromkeys(owner[number] for number in retrieved.order))[:5],
            "seconds": round(retrieve_s[-1], 4),
        }
        rows.append(row)

    fresh = [document for document in documents if not document.cached]
    pages = sum(document.pages for document in documents)
    parse_s = sum(document.seconds for document in documents)
    timings = {
        # Parsing as it was timed when each text was produced, in this run or
        # in the one that kept it.
        "parse_s": parse_s,
        "parse_s_per_page": parse_s / pages if pages else 0.0,
        "parsed_now": len(fresh),
        "chunk_s": chunk_s,
        "index_s": index_s,
        "retrieve_s_per_question": sum(retrieve_s) / len(retrieve_s) if retrieve_s else 0.0,
        "total_s": clock() - started,
    }
    return {
        "documents": documents,
        "rows": rows,
        "chunks": len(chunks),
        "timings": {name: round(value, 4) for name, value in timings.items()},
    }


def _slug(name: str) -> str:
    return "_".join(str(name).lower().split())


def _groups(rows: list[dict], field_name: str) -> dict[str, dict[str, int]]:
    """Per value of a field: questions with a passage, and how many are covered."""
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["passages"]:
            group = groups.setdefault(str(row[field_name]), {"with_evidence": 0, "in_context": 0})
            group["with_evidence"] += 1
            group["in_context"] += row["lost"] is None
    return dict(sorted(groups.items()))


def summarize(rows: list[dict]) -> dict:
    """The run in numbers: how many questions get through each stage.

    The stages are nested: a question counted at one stage is counted at all
    the stages before it.
    """
    answerable = [row for row in rows if row["passages"]]
    passages = [passage for row in rows for passage in row["passages"]]

    def past(stage: str) -> int:
        return sum(
            row["lost"] is None or STAGES.index(row["lost"]) > STAGES.index(stage)
            for row in answerable
        )

    def within(depth: int) -> int:
        return sum(row["rank"] is not None and row["rank"] <= depth for row in answerable)

    metrics = {
        "questions": len(rows),
        "with_evidence": len(answerable),
        "passages": len(passages),
        "parsed": past("parsing"),
        "chunked": past("chunking"),
        "one_chunk": sum(row["one_chunk"] >= WHOLE for row in answerable),
        **{f"top_{depth}": within(depth) for depth in DEPTHS},
        "in_context": past("ranking"),
        "in_context_part": sum(PART <= row["context"] < WHOLE for row in answerable),
        "document_in_context": sum(row["document_in_context"] for row in answerable),
        "passages_in_context": sum(passage["lost"] is None for passage in passages),
    }
    domains = _groups(rows, "domain")
    for domain, group in domains.items():
        metrics[f"{_slug(domain)}_in_context"] = group["in_context"]
        metrics[f"{_slug(domain)}_with_evidence"] = group["with_evidence"]
    if domains:
        # The challenge weighs the two domains equally, whatever their sizes.
        rates = [group["in_context"] / group["with_evidence"] for group in domains.values()]
        metrics["balanced"] = round(sum(rates) / len(rates), 4)
    if rows:
        metrics["context_chunks"] = round(sum(row["context_chunks"] for row in rows) / len(rows), 1)
        metrics["context_tokens"] = round(sum(row["context_tokens"] for row in rows) / len(rows), 1)
    return metrics


def _number(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def report(rows: list[dict], metrics: dict, say=print) -> None:
    say("[retrieval]  id  domain     type         passages  parsed  1 chunk   rank  context")
    for row in rows:
        head = f"[retrieval] {row['id']!s:>3}  {row['domain']!s:<9}  {row['question_type']!s:<11}"
        if not row["passages"]:
            say(f"{head}         -  no passage in the corpus")
            continue
        outcome = "" if row["lost"] is None else f"  lost in {row['lost']}"
        if row["lost"] is not None and row["context"] >= PART:
            outcome += ", in part in the context"
        rank = "-" if row["rank"] is None else row["rank"]
        say(
            f"{head}  {len(row['passages']):>8}   {_number(row['parsed'])}    "
            f"{_number(row['one_chunk'])}  {rank!s:>5}    {_number(row['context'])}{outcome}"
        )

    answerable = metrics["with_evidence"]
    say(
        f"[retrieval] {answerable} of {metrics['questions']} questions have a passage in the "
        f"corpus ({metrics['passages']} passages)"
    )
    if not answerable:
        return
    say(f"[retrieval]   kept by the parser    {metrics['parsed']:>3}")
    say(
        f"[retrieval]   held by the chunks    {metrics['chunked']:>3}   "
        f"({metrics['one_chunk']} whole in a single chunk)"
    )
    depths = " / ".join(str(depth) for depth in DEPTHS)
    counts = " / ".join(str(metrics[f"top_{depth}"]) for depth in DEPTHS)
    say(f"[retrieval]   within the first {depths} chunks: {counts}")
    say(
        f"[retrieval]   in the context        {metrics['in_context']:>3}   "
        f"({metrics['in_context_part']} more in part; {metrics['passages_in_context']} of "
        f"{metrics['passages']} passages)"
    )
    for field_name, label in (("domain", "by domain"), ("question_type", "by question type")):
        groups = _groups(rows, field_name)
        parts = [f"{name} {g['in_context']} of {g['with_evidence']}" for name, g in groups.items()]
        tail = f"; balanced {metrics['balanced']:.3f}" if field_name == "domain" else ""
        say(f"[retrieval]   {label}: {', '.join(parts)}{tail}")
    say(
        f"[retrieval]   a chunk of a document that holds the answer is in the context for "
        f"{metrics['document_in_context']} of {answerable}"
    )
    say(
        f"[retrieval] a context holds {metrics['context_chunks']} chunks and "
        f"{metrics['context_tokens']} tokens on average"
    )


def run(
    system: System,
    questions: list[dict],
    items: list[dict],
    pdfs: Iterable[Path],
    *,
    out_dir: Path = SCRATCH_OUT,
    manifest: list[dict] | None = None,
    cache_dir: Path | None = CACHE_DIR,
    reparse: bool = False,
    clock=time.perf_counter,
    say=print,
) -> dict:
    """Measure a system and write questions.jsonl (one row per question) and
    summary.json into a new folder of out_dir named after the run. Returns the
    summary."""
    started = datetime.now(timezone.utc)
    git = ledger.git_state()
    run_id = ledger.new_run_id(started, git["commit"])

    measured = measure(
        system, questions, items, pdfs, cache_dir=cache_dir, reparse=reparse, clock=clock, say=say
    )
    documents, rows = measured["documents"], measured["rows"]
    summary = {
        "run_id": run_id,
        "utc": started.isoformat(timespec="seconds"),
        "git": git,
        "system": system.name,
        "config": system.config,
        "matching": {"rules": RULES, "shingle": SHINGLE, "whole": WHOLE, "part": PART},
        "corpus": {
            "documents": len(documents),
            "pages": sum(document.pages for document in documents),
            "not_parsed": sorted(d.doc_id for d in documents if d.text is None),
            "fingerprint": fingerprint(documents),
        },
        "chunks": measured["chunks"],
        "metrics": summarize(rows),
        "by_domain": _groups(rows, "domain"),
        "by_question_type": _groups(rows, "question_type"),
        "timings": measured["timings"],
        "problems": [],
    }
    if manifest is not None:
        differences = manifest_differences(documents, manifest)
        summary["corpus"]["differs_from_manifest"] = {
            name: doc_ids for name, doc_ids in differences.items() if doc_ids
        }
    read = {document.doc_id for document in documents if document.text is not None}
    wanted = {passage["doc_id"] for row in rows for passage in row["passages"]}
    if wanted - read:
        summary["problems"].append(
            "documents that hold a passage were not read: " + ", ".join(sorted(wanted - read))
        )
    # The tree may have changed under a long run; then the commit recorded at
    # the start no longer describes everything that ran.
    if ledger.git_state() != git:
        summary["git"] = {**git, "dirty": True}
        summary["problems"].append("the working tree changed while the run was in progress")

    run_dir = out_dir.resolve() / run_id
    run_dir.mkdir(parents=True)
    with open(run_dir / "questions.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(run_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report(rows, summary["metrics"], say)
    return summary


# The systems that can be measured, by name: what builds each from a model id
# and from the options that follow its name. Their modules are imported when
# asked for: they need this one for what a system returns.
def _rag_example(model_id: str, options: str) -> System:
    from tools import rag_example

    if options:
        raise ValueError("the RAG example is measured as it is: it takes no options")
    return rag_example.RagExample.load(REPO_ROOT / "02_rag", model_id)


def _lab(model_id: str, options: str) -> System:
    from tools import retrieval_lab

    return retrieval_lab.Lab.load(options, model_id)


def _entry(model_id: str, options: str) -> System:
    from tools import entry_system

    if options:
        raise ValueError("the entry is measured as it is: it takes no options")
    return entry_system.Entry.load(model_id)


SYSTEMS = {"02_rag": _rag_example, "lab": _lab, "03_dsqa": _entry}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a retrieval system against the reference passages, without a model."
    )
    parser.add_argument(
        "--system",
        default="02_rag",
        metavar="NAME[:OPTIONS]",
        help="02_rag, 03_dsqa, or lab:option=value,... (see tools/retrieval_lab.py)",
    )
    parser.add_argument("--model-id", required=True, help="the base model; only its tokenizer")
    parser.add_argument("--corpus-dir", type=Path, default=corpus.CORPUS_DIR)
    parser.add_argument("--questions", type=Path, default=evidence.SAMPLE)
    parser.add_argument("--evidence", type=Path, default=evidence.EVIDENCE)
    parser.add_argument("--manifest", type=Path, default=corpus.MANIFEST)
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N questions")
    parser.add_argument("--ids", help="only the questions with these ids, comma-separated")
    parser.add_argument("--cache", type=Path, default=CACHE_DIR, help="where parsed text is kept")
    parser.add_argument("--reparse", action="store_true", help="parse again what is kept")
    parser.add_argument("--out", type=Path, help="where run folders go")
    record = parser.add_argument_group("recording the run in the ledger")
    record.add_argument("--record", action="store_true", help="append a row to the ledger")
    record.add_argument("--dataset", help="name of the question set (default: its file name)")
    record.add_argument("--corpus", help="name of the corpus (default: its folder name)")
    record.add_argument("--machine", help="where this runs, to compare timings across machines")
    record.add_argument("--note", default="", help="free text stored with the row")
    record.add_argument("--ledger", type=Path, default=ledger.LEDGER_PATH)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    try:
        questions = run_local.load_questions(args.questions)
    except (OSError, ValueError) as e:
        parser.error(str(e))
    questions = run_local.select(questions, args.limit, args.ids.split(",") if args.ids else None)
    if not questions:
        parser.error("no questions selected")
    pdfs = sorted(args.corpus_dir.glob("*.pdf"))
    if not pdfs:
        parser.error(f"no PDFs in {args.corpus_dir}")
    out_dir = args.out or (RECORDED_OUT if args.record else SCRATCH_OUT)
    name, _, options = args.system.partition(":")
    if name not in SYSTEMS:
        parser.error(f"no such system: {name!r} (there are {', '.join(sorted(SYSTEMS))})")

    # As in the evaluation, nothing is downloaded: what a system needs has to
    # be on this machine already.
    with run_local._environment(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"):
        try:
            system = SYSTEMS[name](args.model_id, options)
        except ValueError as e:
            parser.error(str(e))
        summary = run(
            system,
            questions,
            evidence.load(args.evidence),
            pdfs,
            out_dir=out_dir,
            manifest=corpus.load_manifest(args.manifest),
            cache_dir=args.cache,
            reparse=args.reparse,
        )
    for problem in summary["problems"]:
        print(f"[retrieval] PROBLEM: {problem}")
    differences = summary["corpus"].get("differs_from_manifest", {})
    if differences:
        counts = [f"{len(ids)} {name.replace('_', ' ')}" for name, ids in differences.items()]
        print(f"[retrieval] the corpus folder is not the manifest's: {', '.join(counts)}")
    timings = summary["timings"]
    print(
        f"[retrieval] parsing {timings['parse_s']:.0f}s ({timings['parse_s_per_page']:.2f} s/page; "
        f"{timings['parsed_now']} of {summary['corpus']['documents']} documents parsed now), "
        f"indexing {timings['index_s']:.0f}s, "
        f"{timings['retrieve_s_per_question']:.3f}s per question"
    )
    print(f"[retrieval] results in {(out_dir.resolve() / summary['run_id']).as_posix()}")

    if args.record:
        ledger.record(
            "retrieval",
            summary["metrics"],
            dataset=args.dataset or args.questions.stem,
            corpus=args.corpus or args.corpus_dir.resolve().name,
            engine="none",
            config={
                **summary["config"],
                "matching": summary["matching"],
                "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                "evidence_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
                "corpus_documents": summary["corpus"]["documents"],
                "corpus_pages": summary["corpus"]["pages"],
                "corpus_fingerprint": summary["corpus"]["fingerprint"],
                "corpus_is_the_manifest": not differences,
                "chunks": summary["chunks"],
                "limit": args.limit,
                "ids": args.ids,
            },
            timings=timings,
            notes=args.note,
            # Timings only compare across runs on the same hardware.
            machine=args.machine or run_local._gpu_name(),
            run_id=summary["run_id"],
            state=summary["git"],
            path=args.ledger,
        )
        print(f"[retrieval] recorded in {args.ledger.name} as {summary['run_id']}")
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
