"""Stand-ins for what the RAG example brings: its main.py, an embedder and a
tokenizer, each with the part of the interface that tools/ uses and nothing
that needs a model. And a paper made on the spot, for the tests that read a
real PDF.
"""

import re
import types

import numpy as np
import pymupdf

BUDGET = 12


def tokens(text):
    """How many tokens Words makes of a text."""
    return len(text.split())


def paper_pdf(path, body):
    """A one-page paper: a title in large type over two lines, its authors
    and a body."""
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "A study of the lithium loop", fontsize=18)
        page.insert_text((72, 104), "of the plant", fontsize=18)
        page.insert_text((72, 130), "Some Author, Another Author", fontsize=10)
        page.insert_text((72, 160), body, fontsize=9)
        document.save(path)
    return path


BODY = "\n".join(["The main loop carries the liquid lithium to the target and back."] * 12)


class Words:
    """A tokenizer with a token per word."""

    def encode(self, text):
        return text.split()

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, verbose=False):
        spans = [match.span() for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


class Letters:
    """Embeds a text as the counts of its letters: a very small embedder. It
    notes how many texts it was asked to embed at each call."""

    def __init__(self):
        self.tokenizer = Words()
        self.calls: list[int] = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64):
        self.calls.append(len(texts))
        counts = [
            [text.lower().count(letter) for letter in "abcdefghijklmnopqrstuvwxyz"]
            for text in texts
        ]
        vectors = np.asarray(counts, dtype=np.float32).reshape(len(texts), 26)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def example_main(build_context=None):
    """What tools/rag_example.py uses of 02_rag/main.py, with a _build_context
    that ranks, packs and writes its context the way the example's does."""

    def _build_context(question, chunks, embeddings, embedder, tokenizer, max_tokens=BUDGET):
        query = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        selected, used = [], 0
        for i in np.argsort(-(embeddings @ query)):
            n = len(tokenizer.encode(chunks[i]))
            if used + n > max_tokens:
                if not selected and n > 0:
                    selected.append(chunks[i][: int(len(chunks[i]) * (max_tokens / n))])
                break
            selected.append(chunks[i])
            used += n
        if not selected:
            return ""
        body = "\n\n".join(f"[Fragment {i}]\n{c}" for i, c in enumerate(selected, 1))
        return f"Context:\n\n{body}\n\n---\n\n"

    built = []

    class Converter:
        def convert(self, path):
            markdown = f"# {path}"
            document = types.SimpleNamespace(export_to_markdown=lambda: markdown)
            return types.SimpleNamespace(document=document)

    def _docling_converter():
        built.append(1)
        return Converter()

    return types.SimpleNamespace(
        MAX_CONTEXT_TOKENS=BUDGET,
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=150,
        EMBED_MODEL_PATH="not used: the tests bring their own embedder",
        _build_context=build_context or _build_context,
        _chunk_markdown=lambda text: text.split("\n\n"),
        _docling_converter=_docling_converter,
        converters_built=built,
    )
