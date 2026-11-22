"""Retrieval variants for the lab.

A lab system is the organizers' RAG example with stages swapped, so that what
one change buys can be measured with tools/eval_retrieval.py:

    python -m tools.eval_retrieval --model-id <id> --system lab:ranker=bm25
    python -m tools.eval_retrieval --model-id <id> --system lab:chunker=fitted,ranker=rrf

With no options a lab system is the example again, put together from the
same parts: it has to give the example's numbers, and it is checked against
the example's own context on every question.

Options, each with the example's behaviour as its default:

    parser        docling   the example's Docling conversion to Markdown
                  pymupdf   PyMuPDF's text blocks, one paragraph each
                  pymupdf_titled      the same, opening with the document's title as
                            the PDF tells it (its metadata, or its largest type)
                  pymupdf_tables      the titled text, with each table that is ruled and
                            captioned read as rows of cells under its caption
                  docling_or_pymupdf  Docling's Markdown, and for a document Docling
                            could not read the titled text of PyMuPDF
    chunker       example   paragraphs packed to about 800 characters, never split
                  fitted    no chunk longer than chunk_tokens of the embedder's tokens
                  sections  the same, along the structure of the Markdown: a chunk
                            stays inside a section and starts with its heading, and
                            a table goes with its caption, repeated on each part
    chunk_tokens  254       what all-MiniLM-L6-v2 reads of a chunk, less its two markers
    table_tokens  0         a table goes into chunks of at most this many tokens, longer
                            than the embedder reads, and is cut between its rows: the
                            first part has the caption, the others its start and the
                            table's first row (sections); 0: cut as any text is
    title         0         1: every chunk starts with its document's title (sections)
    references    1         0: the reference list is left out (sections)
    ranker        dense     cosine similarity of the example's embedder
                  bm25      words in common, weighted by how rare they are
                  rrf       the two rankings fused by reciprocal rank
    packer        prefix    the ranking from the top until a chunk does not fit
                  fill      the same, passing over the chunks that do not fit
    whole         0         the document of the best chunk goes in whole, ahead of the
                            ranking, when it takes no more than this percent of the budget
    beside        0         when it does not go in whole, each chunk of that document
                            brings this many chunks on either side of it
    budget        10000     tokens of context, counted with the base model's tokenizer
    device        cpu       where the embedder runs (cuda: faster, last digits may differ)

Nothing here ships in a submission. What proved its worth is the entry's own
code by now, and the lab takes it from there (tools/entry.py): PyMuPDF's
text with its title and its tables, the sections chunker, BM25 and the
fusion of rankings. What stayed behind is here only: the example's stages,
the fitted chunker, the filling packer and the chunks beside.
"""

import hashlib
import re
from pathlib import Path

import numpy as np

from tools import entry
from tools.eval_retrieval import CACHE_DIR, REPO_ROOT, Retrieved, squash
from tools.rag_example import RagExample

EMBEDDINGS_DIR = CACHE_DIR.parent / "embed"

DEFAULTS = {
    "parser": "docling",
    "chunker": "example",
    "chunk_tokens": 254,
    "table_tokens": 0,
    "title": 0,
    "references": 1,
    "ranker": "dense",
    "packer": "prefix",
    "whole": 0,
    "beside": 0,
    "budget": 10_000,
    "device": "cpu",
}
CHOICES = {
    "parser": {"docling", "pymupdf", "pymupdf_titled", "pymupdf_tables", "docling_or_pymupdf"},
    "chunker": {"example", "fitted", "sections"},
    "ranker": {"dense", "bm25", "rrf"},
    "packer": {"prefix", "fill"},
    "device": {"cpu", "cuda"},
}

# How far down a ranking the filling packer looks for chunks that still fit.
LOOK = 200

# Docling could not read a document when its text has less than this share of
# the letters and digits PyMuPDF's has. On the development corpus the share is
# 0.03 to 0.13 for the four documents it cannot read and at least 0.87 for the
# other 146 (journal, 2026-10-10).
READ_SHARE = 0.5


def parse_options(spec: str) -> dict:
    """The options of a lab system from "name=value,name=value"; what is not
    named keeps the example's behaviour."""
    options = dict(DEFAULTS)
    for part in filter(None, (part.strip() for part in spec.split(","))):
        name, _, value = part.partition("=")
        if name not in DEFAULTS:
            raise ValueError(f"no such option: {name!r} (there are {', '.join(DEFAULTS)})")
        if isinstance(DEFAULTS[name], int):
            if not value.isdigit():
                raise ValueError(f"{name} takes a number, not {value!r}")
            options[name] = int(value)
        elif value not in CHOICES[name]:
            raise ValueError(f"{name} is one of {', '.join(sorted(CHOICES[name]))}, not {value!r}")
        else:
            options[name] = value
    if options["whole"] > 100:
        raise ValueError(f"whole is a percent of the budget, not {options['whole']}")
    return options


def could_read(text: str | None, other: str) -> bool:
    """Whether a parser read a document, going by another parser's text of it:
    it has at least READ_SHARE of the letters and digits the other has."""
    return text is not None and len(squash(text)) >= READ_SHARE * len(squash(other))


def fit(text: str, tokenizer, limit: int) -> list[str]:
    """Chunks of at most `limit` tokens of a tokenizer.

    Paragraphs are packed together while they fit. A paragraph too long by
    itself, as a table or a reference list is, is cut where the limit falls:
    at the last sentence end before it, or the last clause, or the last space.
    Nothing is repeated from one chunk to the next.
    """
    chunks: list[str] = []
    current, used = [], 0
    for paragraph in (part.strip() for part in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        for piece, tokens in entry.chunk.pieces(paragraph, tokenizer, limit):
            if current and used + tokens > limit:
                chunks.append("\n\n".join(current))
                current, used = [], 0
            current.append(piece)
            used += tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def pack(
    order,
    chunks: list[str],
    count,
    budget: int,
    fill: bool = False,
    *,
    documents=None,
    whole: float = 0.0,
    beside: int = 0,
) -> tuple[list, int]:
    """The chunks that go into a context of `budget` tokens, and the tokens used.

    Taken from the top of the ranking until one does not fit, as the example
    does: if that is the very first, a slice of it is given. With fill, a chunk
    that does not fit is passed over and the ones after it get their chance.

    With `documents`, which says for every chunk the document it is of, the
    document of the best chunk can be favoured. If all of it takes no more
    than the share `whole` of the budget, all of it goes in, in its own order
    and ahead of the ranking: an answer that needs two passages of a short
    paper then has both. If it does not go in whole, each chunk of it that is
    taken brings along the `beside` chunks on either side of it that still
    fit, nearest first.
    """
    context, used, taken = [], 0, set()

    def place(number: int, tokens: int) -> None:
        nonlocal used
        context.append((number, chunks[number]))
        used += tokens
        taken.add(number)

    favoured = None
    if documents is not None and len(order) and (whole or beside):
        first = documents[order[0]]
        mine = [int(number) for number in np.flatnonzero(np.asarray(documents) == first)]
        costs = [count(chunks[number]) for number in mine] if whole else []
        if whole and sum(costs) <= whole * budget:
            for number, tokens in zip(mine, costs, strict=True):
                place(number, tokens)
        elif beside:
            favoured = first

    for number in order[: LOOK if fill else len(order)]:
        number = int(number)
        if number in taken:
            continue
        chunk = chunks[number]
        tokens = count(chunk)
        if used + tokens > budget:
            if fill:
                continue
            if not context and tokens > 0:
                context.append((number, chunk[: int(len(chunk) * (budget / tokens))]))
            break
        place(number, tokens)
        if favoured is None or documents[number] != favoured:
            continue
        for distance in range(1, beside + 1):
            for near in (number - distance, number + distance):
                if 0 <= near < len(chunks) and near not in taken and documents[near] == favoured:
                    tokens = count(chunks[near])
                    if used + tokens <= budget:
                        place(near, tokens)
    return context, used


class Lab:
    """The RAG example with some of its stages swapped; see the module's notes."""

    name = "lab"

    def __init__(self, example: RagExample, options: dict, embeddings_dir: Path | None = None):
        self.example, self.options = example, options
        self.embeddings_dir = embeddings_dir
        self.chunks: list[str] = []
        self.embeddings = None
        self.bm25 = None
        # How many chunks each document gave, in the order they were chunked,
        # and from that the document of every chunk that was indexed.
        self.sizes: list[int] = []
        self.documents = None
        self._tokens: dict[str, int] = {}

    @classmethod
    def load(cls, spec: str, model_id: str) -> "Lab":
        options = parse_options(spec)
        return cls(RagExample.load(REPO_ROOT / "02_rag", model_id), options, EMBEDDINGS_DIR)

    @property
    def is_the_example(self) -> bool:
        return self.options == DEFAULTS

    @property
    def parse_key(self) -> str:
        parser = self.options["parser"]
        if parser == "docling":
            return self.example.parse_key  # the same text, kept once
        # PyMuPDF's text is read by the entry's code, and kept under the name
        # of that code as it is now: a change to it reads the corpus again.
        if parser == "pymupdf":
            return entry.parse_key("blocks")
        if parser == "pymupdf_titled":
            return entry.parse_key("titled")
        if parser == "pymupdf_tables":
            return entry.parse_key("tables")
        return f"{self.example.parse_key}-or-{entry.parse_key('titled')}-under{READ_SHARE}"

    @property
    def config(self) -> dict:
        return {**self.example.config, "system": self.name, **self.options}

    def _embedder(self):
        if self.example.embedder is None:
            from sentence_transformers import SentenceTransformer

            path = self.example.main.EMBED_MODEL_PATH
            self.example.embedder = SentenceTransformer(path, device=self.options["device"])
        return self.example.embedder

    def parse(self, pdf: Path) -> str:
        parser = self.options["parser"]
        if parser == "docling":
            return self.example.parse(pdf)
        if parser != "docling_or_pymupdf":
            titled, tabled = parser != "pymupdf", parser == "pymupdf_tables"
            return entry.parse.read(pdf, title=titled, tables=tabled)
        other = entry.parse.read(pdf, title=True, tables=False)
        try:
            text = self.example.parse(pdf)
        except Exception:
            text = None  # a document Docling fails on is one it could not read
        return text if could_read(text, other) else other

    def chunk(self, text: str) -> list[str]:
        chunks = self._chunk(text)
        self.sizes.append(len(chunks))
        return chunks

    def _chunk(self, text: str) -> list[str]:
        options = self.options
        if options["chunker"] == "example":
            return self.example.chunk(text)
        tokenizer, limit = self._embedder().tokenizer, options["chunk_tokens"]
        if options["chunker"] == "fitted":
            return fit(text, tokenizer, limit)
        return entry.chunk.sectioned(
            text,
            tokenizer,
            limit,
            title=bool(options["title"]),
            references=bool(options["references"]),
            table_tokens=options["table_tokens"],
        )

    def _embed(self, chunks: list[str]) -> np.ndarray:
        """The embedding of every chunk, read back when these very chunks were
        embedded before by the same embedder on the same kind of device."""
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(hashlib.sha256(chunk.encode("utf-8")).digest())
        kept = None
        if self.embeddings_dir is not None:
            source = self.example.source_sha256[:8]
            folder = self.embeddings_dir / f"02_rag-{source}-{self.options['device']}"
            kept = folder / f"{digest.hexdigest()}.npy"
            if kept.exists():
                return np.load(kept)
        embeddings = self._embedder().encode(
            chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if kept is not None and len(chunks):
            kept.parent.mkdir(parents=True, exist_ok=True)
            np.save(kept, embeddings)
        return embeddings

    def index(self, chunks: list[str]) -> None:
        self.chunks = chunks
        # The chunks are the documents' one after the other, as chunk() gave
        # them. Indexed otherwise, nothing says which document a chunk is of.
        self.documents = None
        if sum(self.sizes) == len(chunks):
            self.documents = np.repeat(np.arange(len(self.sizes)), self.sizes)
        elif self.options["whole"] or self.options["beside"]:
            raise ValueError("whole and beside need the chunks as chunk() gave them, by document")
        if self.options["ranker"] in ("dense", "rrf"):
            self.embeddings = self._embed(chunks)
        if self.options["ranker"] in ("bm25", "rrf"):
            self.bm25 = entry.rank.Bm25(chunks)

    def rank(self, question: str) -> np.ndarray:
        orders = []
        if self.embeddings is not None:
            query = self._embedder().encode(
                [question], normalize_embeddings=True, show_progress_bar=False
            )
            # The example's own expression, so that equal scores fall as its do.
            orders.append(np.argsort(-(self.embeddings @ query[0])))
        if self.bm25 is not None:
            orders.append(entry.rank.best_first(self.bm25.scores(question)))
        return orders[0] if len(orders) == 1 else entry.rank.fuse(orders)

    def retrieve(self, question: str) -> Retrieved:
        if not self.chunks:
            return Retrieved([], [])
        order = self.rank(question)

        def count(text: str) -> int:
            # A chunk is counted once: a document that may go in whole has all
            # of its chunks counted, question after question.
            if text not in self._tokens:
                self._tokens[text] = len(self.example.tokenizer.encode(text))
            return self._tokens[text]

        options = self.options
        context, used = pack(
            order,
            self.chunks,
            count,
            options["budget"],
            options["packer"] == "fill",
            documents=self.documents,
            whole=options["whole"] / 100,
            beside=options["beside"],
        )
        return Retrieved([int(number) for number in order], context, used)

    def verify(self, question: str, retrieved: Retrieved) -> None:
        """With no options changed, this is the example and has to pack what it packs."""
        if self.is_the_example and self.chunks:
            self.example.chunks, self.example.embeddings = self.chunks, self.embeddings
            self.example.verify(question, retrieved)
