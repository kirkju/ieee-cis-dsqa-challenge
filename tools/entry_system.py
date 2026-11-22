"""Our entry, as a system tools/eval_retrieval.py can measure.

    python -m tools.eval_retrieval --model-id <id> --system 03_dsqa

Every stage is done by the entry's own code, the functions its answer() goes
through: parse.read() for a PDF, retrieval.chunks_of() for its chunks,
retrieval.Index for the ranking and the context. Entry adds what the tool
needs and the entry does not keep: how many chunks each document gave, from
which the index is told the document of every chunk.

Nothing of the base model is loaded but its tokenizer, which the budget of
the context is counted with. The embedder is the one shipped with the entry,
loaded as the entry loads it.
"""

import importlib.metadata
from pathlib import Path

import numpy as np

from tools import entry
from tools.eval_retrieval import Retrieved


class Entry:
    """The entry's retrieval, stage by stage."""

    name = "03_dsqa"

    def __init__(self, tokenizer, embedder=None, model_id: str = ""):
        self.tokenizer, self.embedder, self.model_id = tokenizer, embedder, model_id
        # How many chunks each document gave, in the order they were chunked.
        self.sizes: list[int] = []
        self.built = None

    @classmethod
    def load(cls, model_id: str) -> "Entry":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model_id), model_id=model_id)

    @property
    def parse_key(self) -> str:
        return entry.parse_key("tables")  # the lab's text with tables is this same text

    @property
    def config(self) -> dict:
        modules = [entry.text, entry.tables, entry.parse, entry.chunk, entry.rank]
        modules += [entry.context, entry.retrieval, entry.config]
        return {
            "system": self.name,
            "code": entry.source_hash(*modules),
            "pymupdf": importlib.metadata.version("pymupdf"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "chunk_tokens": entry.config.CHUNK_TOKENS,
            "table_tokens": entry.config.TABLE_TOKENS,
            "context_tokens": entry.config.CONTEXT_TOKENS,
            "whole": entry.config.WHOLE,
            "tokenizer": self.model_id,
        }

    def _embedder(self):
        if self.embedder is None:
            self.embedder = entry.retrieval.load_embedder()
        return self.embedder

    def parse(self, pdf: Path) -> str:
        return entry.parse.read(pdf)

    def chunk(self, text: str) -> list[str]:
        chunks = entry.retrieval.chunks_of(text, self._embedder().tokenizer)
        self.sizes.append(len(chunks))
        return chunks

    def index(self, chunks: list[str]) -> None:
        if sum(self.sizes) != len(chunks):
            raise ValueError("the entry's index needs the chunks as chunk() gave them, by document")
        documents = np.repeat(np.arange(len(self.sizes)), self.sizes)
        self.built = entry.retrieval.Index(chunks, documents, self._embedder())

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def retrieve(self, question: str) -> Retrieved:
        placed, used, order = self.built.context(question, self.count)
        return Retrieved([int(number) for number in order], placed, used)
