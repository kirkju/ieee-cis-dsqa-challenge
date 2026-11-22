"""The organizers' RAG example asked one question, in one of two ways, and what
decides whether the two are the same printed as one line of JSON:

    original   02_rag/main.py by itself: its answer() reads the corpus, loads
               the model and answers, as when it is the submission
    replica    the entry with dones-example: main.py has the example read the
               corpus before the model is loaded, the entry's engine loads the
               model, and dsqa/example.py asks the example's answer()

    python tests/slow/example_run.py original|replica <corpus_dir> <model_id> <question> [<entry>]

Printed: the answer; the context the example gave the model, made again by
its own _build_context() from what is in its _cache; and the hashes of its
chunks and of their embeddings. One process a run, as a run of the
evaluation is: the caller sets HF_HUB_OFFLINE, and DSQA_VARIANT for the
replica.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original(corpus_dir: str, model_id: str, question: str):
    main = _module("main", ROOT / "02_rag" / "main.py")
    return main, main.answer(question, corpus_dir, model_id)


def replica(corpus_dir: str, model_id: str, question: str, folder: Path):
    sys.path.insert(0, str(folder))
    entry = _module("entry_main", folder / "main.py")
    entry._example_first(corpus_dir, model_id)
    engine = entry._engine(model_id)
    from dsqa import example

    if not example.ready():
        raise SystemExit("the example's index was not made")
    return example._example(), example.answer(question, corpus_dir, model_id, engine)


def main() -> int:
    way, corpus_dir, model_id, question = sys.argv[1:5]
    if way == "original":
        example, reply = original(corpus_dir, model_id, question)
    else:
        # The entry's folder, as its zip unpacks: 03_dsqa unless another is given.
        folder = Path(sys.argv[5]) if len(sys.argv) > 5 else ROOT / "03_dsqa"
        example, reply = replica(corpus_dir, model_id, question, folder)
    tokenizer, model, chunks, embeddings, embedder = example._cache[model_id]
    context = example._build_context(question, chunks, embeddings, embedder, tokenizer)
    pieces = hashlib.sha256()
    for chunk in chunks:
        pieces.update(hashlib.sha256(chunk.encode("utf-8")).digest())
    found = {
        "way": way,
        "answer": reply,
        "context": context,
        "chunks": len(chunks),
        "chunks_sha256": pieces.hexdigest(),
        "embeddings_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "model": type(model).__name__,
        "dtype": str(model.dtype),
    }
    print("RESULT " + json.dumps(found, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
