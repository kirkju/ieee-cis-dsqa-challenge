"""The organizers' RAG example's way over the entry's own text, for the
questions on IFMIF-DONES (the adjustment dones-lite).

The example itself (dones-example, example.py) reads the corpus with
Docling, in a process of its own and with 330 MB of models in the zip. This
way reads nothing: it takes the text the entry already read with PyMuPDF,
one text per PDF in the order of their names, as the example reads its
PDFs, and gives it to the example's own code, imported from its main.py as
it came (example_rag.py, held to 02_rag/main.py byte for byte):

    its _chunk_markdown()   chunks of about 800 characters, 150 again, on
                            paragraph boundaries
    its embedder            all-MiniLM-L6-v2 from the zip, on the CPU, with
                            the arguments its _build_index() passes
    its answer()            its _build_context() of 10,000 tokens ranked by
                            cosine alone, its prompt and its 2,048 greedy
                            tokens, the reply as it is

so that next to the example it differs in the text it starts from and in
what follows from it: where the paragraphs break, what a table looks like,
which PDFs a reader could not read. The base model is the engine's, put in
the example's _cache as example.py puts it; the example's module is loaded
apart from example.py's, so that the two never share a cache.

    prepare()   the chunks and their embeddings, made when the answerer is built
    ready()     whether they were made
    answer()    the example's answer to a question; None when it has none
"""

import importlib.util
import time
from collections.abc import Callable
from pathlib import Path

from . import config
from .example import _bounded

SOURCE = Path(__file__).resolve().parent.parent / "example_rag.py"
# The name the model is kept under in the example's _cache.
KEY = "dsqa-lite"

_kept: dict = {"main": None, "index": None, "say": print}


def _example():
    """The example's module, loaded from its file once, apart from the one
    example.py loads."""
    if _kept["main"] is None:
        spec = importlib.util.spec_from_file_location("dsqa_example_lite", SOURCE)
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        _kept["main"] = main
    return _kept["main"]


def prepare(
    texts, say: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic
) -> None:
    """The example's chunks of the entry's texts and their embeddings, made
    as its _build_index() makes them from Docling's: its _chunk_markdown()
    on each text in turn, and its embedder on the CPU. When anything fails,
    nothing is kept and that is said."""
    _kept.update(index=None, say=say)
    began = clock()
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        main = _example()
        chunks = []
        for text in texts:
            chunks.extend(main._chunk_markdown(text))
        embedder = SentenceTransformer(main.EMBED_MODEL_PATH, device="cpu")
        embeddings = embedder.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
        _kept["index"] = (chunks, np.asarray(embeddings, dtype=np.float32), embedder)
    except Exception as error:
        say(f"[dsqa] the lite index could not be made: {type(error).__name__}: {error}"[:300])
        return
    say(f"[dsqa] the lite index: {len(chunks)} chunks in {clock() - began:.0f} s")


def ready() -> bool:
    """Whether the chunks and their embeddings were made."""
    return _kept["index"] is not None


def answer(question: str, engine, seconds: float | None = None) -> str | None:
    """The example's answer to a question over the entry's text, as its own
    answer() returns it. None when there is no index, when the engine is not
    the base model through transformers, when there are not even
    config.LEAST_SECONDS to answer in, or when anything fails. `seconds` is
    the most the reply may take."""
    if not ready():
        return None
    model = getattr(engine, "model", None)
    tokenizer = getattr(engine, "tokenizer", None)
    if model is None or tokenizer is None:
        return None
    if seconds is not None and seconds < config.LEAST_SECONDS:
        return None
    try:
        main = _example()
        if KEY not in main._cache:
            chunks, embeddings, embedder = _kept["index"]
            main._cache[KEY] = (tokenizer, model, chunks, embeddings, embedder)
        with _bounded(model, seconds):
            reply = main.answer(question, "", KEY)
    except Exception as error:
        _kept["say"](f"[dsqa] the lite way could not answer: {type(error).__name__}: {error}"[:300])
        return None
    finally:
        try:
            engine.release()
        except Exception:
            pass
    return reply if isinstance(reply, str) else None
