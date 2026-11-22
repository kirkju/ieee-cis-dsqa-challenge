"""From a folder of PDFs to the context of a question.

    read_corpus   every PDF of the folder as text, the ones that cannot be read left out
    chunks_of     a document's text in chunks, with the entry's numbers
    Index         the chunks of the corpus, ranked for a question two ways and fused
    build         the three together: what answer() calls once, on its first question

The corpus is read once a run and kept in memory: it is local to the
evaluation, so nothing of it can be shipped ready-made.
"""

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import config, focus, parse
from .chunk import Words, sectioned
from .context import pack
from .rank import Bm25, best_first, fuse


def read_corpus(
    corpus_dir: Path | str,
    say: Callable[[str], None] = print,
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """The text of every PDF in a folder, in the order of their names. A PDF
    that cannot be read is left out and said: one bad file must not take the
    others down. After `seconds` the reading stops, and what was read stands:
    a corpus larger than planned for must not cost the run."""
    started = clock()
    pdfs = sorted(Path(corpus_dir).glob("*.pdf"))
    texts = []
    for number, pdf in enumerate(pdfs):
        if seconds is not None and clock() - started > seconds:
            left = len(pdfs) - number
            say(f"[dsqa] out of time for reading: {left} of {len(pdfs)} PDFs left out")
            break
        try:
            text = parse.read(pdf)
        except Exception as error:
            say(f"[dsqa] not read: {pdf.name}: {type(error).__name__}: {error}")
            continue
        if text:
            texts.append(text)
    return texts


def chunks_of(text: str, tokenizer) -> list[str]:
    """A document's text as the chunks that are ranked: no longer than the
    embedder reads, each under the document's title, and a table in chunks
    of its own under its caption. `tokenizer` is the embedder's."""
    return sectioned(
        text, tokenizer, config.CHUNK_TOKENS, title=True, table_tokens=config.TABLE_TOKENS
    )


def load_embedder(device: str | None = None):
    """The embedding model shipped with the entry, on the GPU when there is one."""
    if not config.EMBEDDER.is_dir():
        raise FileNotFoundError(f"no embedding model at {config.EMBEDDER}")
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(str(config.EMBEDDER), device=device)


class Index:
    """The chunks of a corpus, ranked for a question.

    Two rankings are fused by rank: the embedder's, by cosine similarity, and
    BM25's, by the words a chunk shares with the question. `documents` says
    for every chunk the document it is of. Without an embedder the ranking is
    BM25's alone.
    """

    def __init__(self, chunks: Sequence[str], documents: Sequence[int], embedder=None):
        if len(chunks) != len(documents):
            raise ValueError(f"{len(chunks)} chunks and {len(documents)} documents of them")
        self.chunks = list(chunks)
        self.documents = np.asarray(documents, dtype=np.intp)
        self.embedder = embedder
        self.embeddings = None
        if embedder is not None and self.chunks:
            embeddings = embedder.encode(
                self.chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=64
            )
            self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.bm25 = Bm25(self.chunks)
        self._tokens: dict[str, int] = {}
        # What the context of the question asked last was made of (focus.py).
        self.focused = None

    def rank(self, question: str) -> np.ndarray:
        """The chunks by their numbers, the best for the question first."""
        by_words = best_first(self.bm25.scores(question))
        if self.embeddings is None:
            return by_words
        query = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        by_meaning = np.argsort(-(self.embeddings @ query[0]))
        return fuse([by_meaning, by_words])

    def context(
        self,
        question: str,
        count: Callable[[str], int],
        budget: int | None = None,
        whole: float | None = None,
    ) -> tuple[list[tuple[int, str]], int, np.ndarray]:
        """The chunks that go into the context of a question, each with its
        number, the tokens they take as `count` counts them, and the ranking
        they were taken from. The budget and the share a whole document may
        take are the entry's unless given.

        With doc-focus (focus.py) the context may be made of the one document
        the question is about instead, within the same budget; what it was
        made of is kept as `focused`, for the question asked last."""
        budget = config.CONTEXT_TOKENS if budget is None else budget
        whole = config.WHOLE if whole is None else whole
        self.focused = None
        if not self.chunks:
            return [], 0, np.zeros(0, dtype=np.intp)
        order = self.rank(question)

        def counted(text: str) -> int:
            # A chunk is counted once: the document that may go in whole has
            # all of its chunks counted, question after question.
            if text not in self._tokens:
                self._tokens[text] = count(text)
            return self._tokens[text]

        if focus.chosen():
            try:
                self.focused = focus.decided(self, question, order)
                if self.focused.how == focus.ONE:
                    placed, used = focus.placed(
                        self.focused, order, self.chunks, self.documents, counted, budget
                    )
                    return placed, used, order
            except Exception as error:
                # What cannot be made of the document identified is made as
                # it was: the question keeps the passages it would have had.
                print(f"[dsqa] the context as it was: {type(error).__name__}: {error}"[:300])
                self.focused = None
        placed, used = pack(order, self.chunks, counted, budget, self.documents, whole)
        return placed, used, order


def index_of(texts: Sequence[str], embedder, tokenizer=None) -> Index:
    """The index of a corpus given as the texts of its documents. The chunks
    are cut by the embedder's tokenizer unless another one is given; without
    either, by words."""
    if tokenizer is None:
        tokenizer = Words() if embedder is None else embedder.tokenizer
    chunks: list[str] = []
    documents: list[int] = []
    for number, text in enumerate(texts):
        mine = chunks_of(text, tokenizer)
        chunks.extend(mine)
        documents.extend([number] * len(mine))
    return Index(chunks, documents, embedder)


def build(
    corpus_dir: Path | str,
    embedder,
    say: Callable[[str], None] = print,
    seconds: float | None = None,
) -> Index:
    """Read, chunk and index every PDF of a folder; the reading for no more
    than `seconds`. Without an embedder the index ranks by words alone."""
    texts = read_corpus(corpus_dir, say, seconds)
    index = index_of(texts, embedder)
    # The documents' texts, kept for a way of route.py that cuts them its
    # own way (dones-lite) without reading the PDFs again.
    index.texts = texts
    say(f"[dsqa] {len(texts)} documents in {len(index.chunks)} chunks")
    return index
