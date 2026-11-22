"""dsqa/example.py: the organizers' RAG example itself, for the questions on
IFMIF-DONES. No model is loaded. The example's index is made by a process of
its own, as in a run, from a stand-in example written on the spot that has
the shape of 02_rag/main.py and does what a test tells it; the example's
answer() is a stand-in in this process, and so is the engine."""

import ctypes
import os
import sys
import time
import types

import numpy as np
import pytest
from dsqa import config
from dsqa import example as way

# A stand-in for 02_rag/main.py: what the process of the index and this
# process ask of it. What its _build_index() does is told by FAKE_EXAMPLE.
STAND_IN = """import os
import subprocess
import sys
import time

import numpy as np

EMBED_MODEL_PATH = "the embedder's folder"
_cache = {}


class SentenceTransformer:
    def __init__(self, path, device=None):
        self.path, self.device = path, device


def _build_index(corpus_dir):
    told = os.environ.get("FAKE_EXAMPLE", "")
    if told == "slow":
        time.sleep(60)
    if told == "dies":
        os._exit(3)
    if told == "raises":
        raise MemoryError("no room for Docling")
    if told == "grandchild":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open(os.environ["FAKE_PID_FILE"], "w") as noted:
            noted.write(str(child.pid))
        time.sleep(60)
    seen = f"{os.environ.get('FAKE_MARK')}|{os.environ.get('OMP_NUM_THREADS')}"
    chunks = [f"{name} {seen} \\u00b5Sv" for name in sorted(os.listdir(corpus_dir))]
    embeddings = np.arange(len(chunks) * 3, dtype=np.float32).reshape(len(chunks), 3) / 7
    return chunks, embeddings, "the embedder in the process"
"""


def alive(pid: int) -> bool:
    """Whether a process runs. On Linux one that was killed may wait as a
    zombie until it is reaped, and in a container nothing may reap it: a
    zombie does not run."""
    if os.name == "posix":
        stat = f"/proc/{pid}/stat"
        if os.path.exists(stat):
            with open(stat, encoding="utf-8") as status:
                return status.read().rsplit(")", 1)[-1].split()[0] not in ("Z", "X")
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return bool(ok) and code.value == 259


@pytest.fixture
def fresh(monkeypatch):
    """Nothing kept from before."""
    monkeypatch.setattr(way, "_kept", {"main": None, "index": None, "model_id": None, "say": print})


@pytest.fixture
def stand_in(fresh, tmp_path, monkeypatch):
    """The stand-in example where the entry looks for the example, and a
    corpus of two PDFs (by name: the stand-in does not read them)."""
    source = tmp_path / "example_rag.py"
    source.write_text(STAND_IN, encoding="utf-8")
    monkeypatch.setattr(way, "SOURCE", source)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (corpus / name).write_bytes(b"%PDF")
    monkeypatch.setenv("FAKE_EXAMPLE", "")
    return corpus


def test_the_index_is_made_apart_and_read_back_as_it_was_written(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_MARK", "seen")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    said = []
    way.prepare(stand_in, said.append, model_id="the/model")
    assert way.ready()
    chunks, embeddings, embedder = way._kept["index"]
    # Written by the other process as its _build_index() made them, and
    # read back whole; the environment it saw is this one, threads untold.
    assert chunks == ["a.pdf seen|None µSv", "b.pdf seen|None µSv"]
    assert embeddings.dtype == np.float32
    assert embeddings.tobytes() == (np.arange(6, dtype=np.float32).reshape(2, 3) / 7).tobytes()
    # The embedder of the questions is the example's, loaded here as it loads it.
    assert (embedder.path, embedder.device) == ("the embedder's folder", "cpu")
    assert said[-1].startswith("[dsqa] the example's index: 2 chunks in ")


def test_the_process_sees_the_environment_as_it_is(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_MARK", "other")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    way.prepare(stand_in, lambda line: None)
    assert way._kept["index"][0][0] == "a.pdf other|3 µSv"


def test_past_its_time_the_process_is_stopped_and_nothing_is_kept(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_EXAMPLE", "slow")
    said = []
    began = time.monotonic()
    way.prepare(stand_in, said.append, seconds=3)
    assert time.monotonic() - began < 30
    assert not way.ready()
    assert said == ["[dsqa] the example's index ran out of its 3 s: not available"]


def test_a_process_stopped_leaves_nothing_it_started(stand_in, monkeypatch, tmp_path):
    noted = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FAKE_EXAMPLE", "grandchild")
    monkeypatch.setenv("FAKE_PID_FILE", str(noted))
    way.prepare(stand_in, lambda line: None, seconds=8)
    assert not way.ready()
    pid = int(noted.read_text())
    for _ in range(50):
        if not alive(pid):
            break
        time.sleep(0.2)
    assert not alive(pid)


def test_a_run_may_be_told_a_shorter_time_by_the_environment(stand_in, monkeypatch):
    monkeypatch.setenv("FAKE_EXAMPLE", "slow")
    monkeypatch.setenv("DSQA_EXAMPLE_SECONDS", "2")
    said = []
    way.prepare(stand_in, said.append)
    assert not way.ready() and said == [
        "[dsqa] the example's index ran out of its 2 s: not available"
    ]
    # What is not a number is the entry's own time.
    monkeypatch.setenv("FAKE_EXAMPLE", "")
    monkeypatch.setenv("DSQA_EXAMPLE_SECONDS", "soon")
    way.prepare(stand_in, said.append)
    assert way.ready()


def test_the_time_counts_from_when_our_code_started(stand_in):
    said = []
    # Started an hour ago, with 30 minutes for the index: none is begun.
    way.prepare(stand_in, said.append, started=time.monotonic() - 3600, seconds=1800)
    assert not way.ready() and "ran out of its 1800 s" in said[0]
    assert config.EXAMPLE_INDEX_SECONDS > 0


@pytest.mark.parametrize(
    ("told", "said"),
    [
        ("dies", "its process ended with code 3 and did not finish it"),
        ("raises", "its process ended with code 1 and did not finish it"),
    ],
)
def test_a_process_that_dies_or_fails_leaves_the_way_closed(stand_in, monkeypatch, told, said):
    monkeypatch.setenv("FAKE_EXAMPLE", told)
    heard = []
    way.prepare(stand_in, heard.append)
    assert not way.ready()
    assert heard == [f"[dsqa] the example's index could not be made: RuntimeError: {said}"]


def test_without_a_folder_to_write_in_the_way_is_closed(stand_in, monkeypatch):
    def unwritable(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(way.tempfile, "TemporaryDirectory", unwritable)
    heard = []
    way.prepare(stand_in, heard.append)
    assert not way.ready() and "PermissionError: read-only file system" in heard[0]


def test_without_the_example_s_file_the_way_is_closed(fresh, monkeypatch, tmp_path):
    monkeypatch.setattr(way, "SOURCE", tmp_path / "example_rag.py")
    heard = []
    way.prepare(tmp_path, heard.append)
    assert not way.ready() and "did not finish it" in heard[0]
    assert (
        way.answer("q", tmp_path, "the/model", types.SimpleNamespace(model=1, tokenizer=2)) is None
    )


# The example's answer, in this process: a stand-in example module and engine.


def answering_example():
    main = types.ModuleType("example_like")
    main._cache = {}
    main.asked = []

    def answer(question, corpus_dir, model_id):
        tokenizer, model, chunks, embeddings, embedder = main._cache[model_id]
        main.asked.append((question, corpus_dir, model_id, model.generation_config.max_time))
        if question == "fail":
            raise RuntimeError("CUDA out of memory")
        return f"  {len(chunks)} chunks answer {question!r} with {model.name}  "

    main.answer = answer
    return main


class Engine:
    def __init__(self):
        self.model = types.SimpleNamespace(
            name="the model", generation_config=types.SimpleNamespace(max_time=None)
        )
        self.tokenizer = "the tokenizer"
        self.released = 0

    def release(self):
        self.released += 1


@pytest.fixture
def indexed(monkeypatch):
    """The example's index made, as prepare() keeps it, and its module."""
    main = answering_example()
    index = (["one", "two", "three"], np.ones((3, 3), dtype=np.float32), "the embedder")
    kept = {"main": main, "index": index, "model_id": "the/model", "say": print}
    monkeypatch.setattr(way, "_kept", kept)
    return main


def test_the_example_answers_with_the_engine_s_model_and_its_reply_is_as_it_is(indexed):
    engine = Engine()
    reply = way.answer("What is the flow?", "corpus", "the/model", engine)
    # As the example returns it: nothing taken off, nothing added.
    assert reply == "  3 chunks answer 'What is the flow?' with the model  "
    tokenizer, model, chunks, embeddings, embedder = indexed._cache["the/model"]
    assert (tokenizer, model) == ("the tokenizer", engine.model)
    assert len(chunks) == 3 and embeddings.shape == (3, 3) and embedder == "the embedder"
    # No limit without seconds, and the engine is asked to give back its memory.
    assert indexed.asked == [("What is the flow?", "corpus", "the/model", None)]
    assert engine.released == 1
    # Without a name, the one prepare() was given.
    assert way.answer("Again?", None, None, engine).startswith("  3 chunks answer 'Again?'")
    assert indexed.asked[-1][1:3] == ("", "the/model")


def test_a_reply_is_held_to_its_seconds_and_the_limit_is_taken_off_after(indexed):
    engine = Engine()
    engine.model.generation_config.max_time = 7.0  # what the config had
    way.answer("What is the flow?", "corpus", "the/model", engine, seconds=95)
    assert indexed.asked[-1][3] == 95.0
    assert engine.model.generation_config.max_time == 7.0
    # Also when the example fails on the question.
    said = []
    way._kept["say"] = said.append
    assert way.answer("fail", "corpus", "the/model", engine, seconds=95) is None
    assert engine.model.generation_config.max_time == 7.0
    assert said == ["[dsqa] the example could not answer: RuntimeError: CUDA out of memory"]
    # With too little time the example is not asked at all.
    asked = len(indexed.asked)
    assert way.answer("q", "corpus", "the/model", engine, seconds=config.LEAST_SECONDS - 1) is None
    assert len(indexed.asked) == asked


def test_no_answer_without_an_index_a_model_or_a_name(indexed):
    engine = Engine()
    way._kept["model_id"] = None
    assert way.answer("q", "corpus", None, engine) is None  # no name to keep the model under
    no_model = types.SimpleNamespace(count=len, release=lambda: None)
    assert way.answer("q", "corpus", "the/model", no_model) is None  # not transformers
    assert indexed.asked == []
    way._kept["index"] = None
    assert way.answer("q", "corpus", "the/model", engine) is None  # no index
    assert indexed.asked == []


def test_the_example_s_file_is_found_next_to_main_py():
    assert way.SOURCE.name == "example_rag.py"
    assert way.SOURCE.parent == config.EMBEDDER.parent
    assert sys.executable  # the process of the index runs on this interpreter
