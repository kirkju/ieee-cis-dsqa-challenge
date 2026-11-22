"""The organizers' RAG example itself, for the questions on IFMIF-DONES (the
adjustment dones-example).

On the public leaderboard the organizers' RAG example scores what our first
entry does, and what we measured says that it gets there on IFMIF-DONES.
So a question the code tells is on IFMIF-DONES (route.py) may be answered by
the example itself: its main.py, copied as it came into the entry's folder
as example_rag.py, which a test holds to 02_rag/main.py byte for byte. Its
own code does all of it: Docling's reading without OCR or table structure,
its chunks of 800 characters with 150 again, MiniLM on the CPU, its ranking
by cosine, its context of 10,000 tokens, its prompt and its 2,048 greedy
tokens; and what its answer() returns is the answer, as it is.

    prepare()   the example's index of the corpus, made before the base model is loaded
    ready()     whether that index was made
    answer()    the example's answer to a question; None when it has none

Where this is not the example, and why:

- Its index is made by its own _build_index(), untouched, before the base
  model is loaded, as the example makes it; but in a process of its own,
  started by main.py and only with the adjustment. That process has until
  config.EXAMPLE_INDEX_SECONDS after our code started to finish the whole
  index, reading and embedding; then it is stopped, with whatever it
  started. Stopped, failed or dead (a PDF Docling cannot take, the memory),
  it leaves no index and the way is not available: the questions on
  IFMIF-DONES are answered the entry's way. A part of an index would answer
  worse than either, and is never read: the process writes its files under
  other names and gives them theirs when all are written, and only then
  marks them done.
- It sees what the example would: the same interpreter, the same
  environment, the same cores. Nothing here sets the threads of torch: with
  other threads MiniLM gives other last digits, and a question can rank its
  chunks otherwise (journal).
- The chunks and their embeddings come back as the process wrote them, as
  JSON and .npy, nothing lost; the embedder of the questions is the
  example's own, loaded here as its _build_index() loads it, MiniLM on the
  CPU.
- The base model is not loaded twice: the engine's, loaded as the example
  loads its own, is put where the example's _load() would keep its own, in
  its _cache under the model's name, with the example's index.
- A reply takes no more than the seconds it is given, an emergency limit
  (route.py) that 2,048 tokens do not reach in practice: it is set on the
  model's generation config for that one call, and taken off after it.
- Whatever fails is said and gives None: the question is then answered the
  entry's way. Nothing here raises.
"""

import gc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from . import config

# The example's main.py as it came, next to main.py, where the paths it
# finds its two models by, docling_models/ and embed_model/ beside it,
# resolve as they do in the example's own zip.
SOURCE = Path(__file__).resolve().parent.parent / "example_rag.py"

# What the process of the index runs: the example's module from its file, its
# _build_index() of the corpus, and the index written into a folder, each
# file under another name until it is whole, and the mark "done" last.
INDEX = r"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

source, corpus_dir, folder = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("example_rag", source)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
chunks, embeddings, embedder = main._build_index(corpus_dir)


def written(name, write):
    part = Path(folder) / f"{name}.part"
    with open(part, "wb") as file:
        write(file)
    os.replace(part, Path(folder) / name)


written("embeddings.npy", lambda file: np.save(file, np.asarray(embeddings, dtype=np.float32)))
written("chunks.json", lambda file: file.write(json.dumps(chunks).encode("utf-8")))
written("done", lambda file: file.write(b"done"))
"""


class OutOfTime(Exception):
    """The example's index ran out of its time."""


# What is kept between calls: the example's module, its index of the
# corpus (its chunks, their embeddings and its embedder), the model's name
# and what says what happens.
_kept: dict = {"main": None, "index": None, "model_id": None, "say": print}


def _example():
    """The example's module, loaded from its file once."""
    if _kept["main"] is None:
        spec = importlib.util.spec_from_file_location("dsqa_example_rag", SOURCE)
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        _kept["main"] = main
    return _kept["main"]


def _stop(process: subprocess.Popen) -> None:
    """A process stopped, with every process it started: on Linux its group,
    which it leads; on Windows its tree."""
    try:
        if os.name == "posix":
            import signal

            os.killpg(process.pid, signal.SIGKILL)
        else:
            stopping = ["taskkill", "/F", "/T", "/PID", str(process.pid)]
            subprocess.run(stopping, capture_output=True, timeout=60)
    except Exception:
        pass
    try:
        process.kill()
        process.wait(timeout=60)
    except Exception:
        pass


def _apart(command: list[str], seconds: float) -> int:
    """A command run in a process of its own, leading a group of its own,
    and its exit code. Past `seconds` it is stopped with every process it
    started, and OutOfTime is raised."""
    if seconds <= 0:
        raise OutOfTime
    own = {"start_new_session": True} if os.name == "posix" else {}
    if os.name != "posix":
        own["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(command, **own)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        _stop(process)
        raise OutOfTime from None
    except BaseException:
        _stop(process)
        raise


def _indexed(corpus_dir, seconds: float):
    """The example's index of a corpus, made in a process of its own that
    has `seconds`: its chunks and their embeddings as that process wrote
    them, and the example's embedder for the questions."""
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="dsqa-example-", ignore_cleanup_errors=True) as kept:
        folder = Path(kept)
        command = [sys.executable, "-c", INDEX, str(SOURCE), str(corpus_dir), str(folder)]
        code = _apart(command, seconds)
        if code != 0 or not (folder / "done").is_file():
            raise RuntimeError(f"its process ended with code {code} and did not finish it")
        chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        embeddings = np.load(folder / "embeddings.npy")
    main = _example()
    embedder = main.SentenceTransformer(main.EMBED_MODEL_PATH, device="cpu")
    return chunks, embeddings, embedder


def prepare(
    corpus_dir,
    say: Callable[[str], None] = print,
    *,
    started: float | None = None,
    model_id: str | None = None,
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Make the example's index of a corpus with its own code, in a process
    of its own, before the base model is loaded. `started` is when our code
    started, on `clock`; the index has until `seconds` after it: unless
    given, the seconds DSQA_EXAMPLE_SECONDS names, for a run that is to see
    the index stopped (the evaluation sets no such variable), or else
    config.EXAMPLE_INDEX_SECONDS. `model_id` is the name the model is kept
    under when no other is given to answer(). Past its time, or when
    anything fails, no index is kept and that is said."""
    _kept.update(index=None, model_id=model_id, say=say)
    began = clock()
    limit = seconds
    if limit is None:
        try:
            limit = float(os.environ.get("DSQA_EXAMPLE_SECONDS") or config.EXAMPLE_INDEX_SECONDS)
        except ValueError:
            limit = config.EXAMPLE_INDEX_SECONDS
    deadline = (began if started is None else started) + limit
    try:
        chunks, embeddings, embedder = _indexed(corpus_dir, deadline - clock())
    except OutOfTime:
        say(f"[dsqa] the example's index ran out of its {limit:.0f} s: not available")
        return
    except Exception as error:
        say(f"[dsqa] the example's index could not be made: {_named(error)}")
        return
    finally:
        _free()
    _kept["index"] = (chunks, embeddings, embedder)
    say(f"[dsqa] the example's index: {len(chunks)} chunks in {clock() - began:.0f} s")


def ready() -> bool:
    """Whether the example's index was made."""
    return _kept["index"] is not None


def answer(question: str, corpus_dir, model_id, engine, seconds: float | None = None) -> str | None:
    """The example's answer to a question, as its own answer() returns it.
    None when its index was not made, when the engine is not the base model
    through transformers, when there are not even config.LEAST_SECONDS to
    answer in, or when anything fails. `seconds` is the most the reply may
    take; without it, as long as the example takes."""
    if not ready():
        return None
    name = model_id or _kept["model_id"]
    model = getattr(engine, "model", None)
    tokenizer = getattr(engine, "tokenizer", None)
    if not name or model is None or tokenizer is None:
        return None
    if seconds is not None and seconds < config.LEAST_SECONDS:
        return None
    say = _kept["say"]
    try:
        main = _example()
        if name not in main._cache:
            chunks, embeddings, embedder = _kept["index"]
            main._cache[name] = (tokenizer, model, chunks, embeddings, embedder)
        with _bounded(model, seconds):
            reply = main.answer(question, str(corpus_dir or ""), name)
    except Exception as error:
        say(f"[dsqa] the example could not answer: {_named(error)}")
        return None
    finally:
        try:
            engine.release()
        except Exception:
            pass
    return reply if isinstance(reply, str) else None


class _bounded:
    """The model's generation config with a time limit for one call, and
    without it again after: what the config had is put back."""

    def __init__(self, model, seconds: float | None):
        self.config = getattr(model, "generation_config", None)
        self.seconds = seconds
        self.had = None

    def __enter__(self):
        if self.seconds is not None and self.config is not None:
            self.had = getattr(self.config, "max_time", None)
            self.config.max_time = float(self.seconds)
        return self

    def __exit__(self, *exc_info):
        if self.seconds is not None and self.config is not None:
            self.config.max_time = self.had
        return False


def _free() -> None:
    """What is left of the reading, given back before the model loads."""
    try:
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _named(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:300]
