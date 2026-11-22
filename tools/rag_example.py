"""The organizers' RAG example, as a system tools/eval_retrieval.py can measure.

02_rag/main.py stays as it came, so that upstream updates merge cleanly. It is
loaded from its file and its own code does every stage; RagExample says what
is added around it.
"""

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np

from tools.eval_retrieval import Retrieved


class ExampleChanged(Exception):
    """The RAG example no longer builds its context the way RagExample reports it."""


class RagExample:
    """The organizers' RAG example, measured as it is.

    Each stage is done by the example's own code, loaded from its main.py:
    its Docling converter, its chunker, its embedder and its ranking. Two
    things are added, because the example does not keep them: which document
    each chunk came from, and which chunks went into the context. The second
    repeats the example's packing loop, and verify() checks the result
    against the context the example itself builds.

    Nothing of the base model is loaded but its tokenizer, which the example
    counts its context budget with.
    """

    name = "02_rag"

    def __init__(self, main, tokenizer, embedder=None, source_sha256: str = "", model_id: str = ""):
        self.main, self.tokenizer, self.embedder = main, tokenizer, embedder
        self.source_sha256, self.model_id = source_sha256, model_id
        self._converter = None
        self.chunks: list[str] = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)

    @classmethod
    def load(cls, folder: Path, model_id: str) -> "RagExample":
        source = folder / "main.py"
        spec = importlib.util.spec_from_file_location("rag_example_main", source)
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        return cls(main, tokenizer, source_sha256=sha256, model_id=model_id)

    @property
    def parse_key(self) -> str:
        return f"02_rag-{self.source_sha256[:8]}-docling-{importlib.metadata.version('docling')}"

    @property
    def config(self) -> dict:
        return {
            "system": self.name,
            "main_sha256": self.source_sha256,
            "docling": importlib.metadata.version("docling"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "chunk_size": self.main.CHUNK_SIZE,
            "chunk_overlap": self.main.CHUNK_OVERLAP,
            "context_tokens": self.main.MAX_CONTEXT_TOKENS,
            "tokenizer": self.model_id,
        }

    def parse(self, pdf: Path) -> str:
        if self._converter is None:
            self._converter = self.main._docling_converter()
        return self._converter.convert(str(pdf)).document.export_to_markdown()

    def chunk(self, text: str) -> list[str]:
        return self.main._chunk_markdown(text)

    def index(self, chunks: list[str]) -> None:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.main.EMBED_MODEL_PATH, device="cpu")
        self.chunks = chunks
        embeddings = self.embedder.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def retrieve(self, question: str) -> Retrieved:
        if not self.chunks:
            return Retrieved([], [])
        query = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        order = np.argsort(-(self.embeddings @ query[0]))
        budget = self.main.MAX_CONTEXT_TOKENS
        context, used = [], 0
        for number in order:
            chunk = self.chunks[number]
            tokens = len(self.tokenizer.encode(chunk))
            if used + tokens > budget:
                if not context and tokens > 0:
                    # Not even the best chunk fits: the example gives a slice of it.
                    ratio = budget / tokens
                    context.append((int(number), chunk[: int(len(chunk) * ratio)]))
                break
            context.append((int(number), chunk))
            used += tokens
        return Retrieved([int(number) for number in order], context, used)

    def verify(self, question: str, retrieved: Retrieved) -> None:
        """The context the example builds is made of exactly the chunks reported.

        The example is asked twice: over every chunk, as in a run, and over
        the reported chunks alone with room for all of them. Both give the
        same text only if those are the chunks it packs, in that order.
        """
        if not self.chunks:
            return
        arguments = (self.embedder, self.tokenizer)
        real = self.main._build_context(question, self.chunks, self.embeddings, *arguments)
        numbers = [number for number, _ in retrieved.context]
        room = retrieved.tokens or self.main.MAX_CONTEXT_TOKENS
        alone = self.main._build_context(
            question, [self.chunks[n] for n in numbers], self.embeddings[numbers], *arguments, room
        )
        if real != alone or any(text not in real for _, text in retrieved.context):
            raise ExampleChanged(
                "the RAG example's context is not made of the chunks reported for it; "
                "RagExample.retrieve() no longer mirrors _build_context() in its main.py"
            )
