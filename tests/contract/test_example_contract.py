"""The contract of answer() with dones-example: the organizers' RAG example
reads the corpus, in a process of its own, before the base model is loaded,
and whatever it does, the entry answers every question with a string.

Run as tests/contract/test_answer.py runs the entry: copied to a folder of
its own, by the local runner, with the fake engine in the base model's place.
The example is a stand-in written next to main.py as example_rag.py, with the
shape of 02_rag/main.py (its _cache, _docling_converter(), _build_index() and
answer()); which questions it is asked is route.py's, not this branch's, so
here only its index and its order are seen.
"""

import json
import shutil
from pathlib import Path

import pymupdf
import pytest

from tests.unit.stand_ins import BODY, paper_pdf
from tools import run_local

ROOT = Path(__file__).resolve().parents[2]
FAKE = Path(__file__).with_name("fake_engine.py")
LOADED = "[fake] the model is loaded"

EXAMPLE = '''"""A stand-in for the organizers' RAG example, with its shape."""
import os
from pathlib import Path

EMBED_MODEL_PATH = "the embedder's folder"
_cache = {}


class SentenceTransformer:
    def __init__(self, path, device=None):
        self.path, self.device = path, device


class _Read:
    def __init__(self, text):
        self.document = self
        self.text = text

    def export_to_markdown(self):
        return self.text


class _Converter:
    def convert(self, path):
        if os.environ.get("EXAMPLE_FAKE") == "unreadable":
            raise RuntimeError("Docling read nothing")
        return _Read(f"# {path}\\n\\nwhat it says")


def _docling_converter():
    return _Converter()


def _build_index(corpus_dir):
    if os.environ.get("EXAMPLE_FAKE") == "raises":
        raise MemoryError("no room for Docling")
    converter = _docling_converter()
    chunks = []
    for pdf in sorted(Path(corpus_dir).glob("*.pdf")):
        try:
            chunks.append(converter.convert(str(pdf)).document.export_to_markdown())
        except Exception as e:
            print(f"[rag] skipping {pdf.name}: {e}")
    print(f"[rag] parsed into {len(chunks)} chunks")
    return chunks, [[0.0]] * len(chunks), "the embedder"


def answer(question, corpus_dir, model_id):
    return "the example's answer"
'''


def asked(*questions):
    return [
        {"id": number, "domain": "DONES", "question_type": "factual", "question": question}
        for number, question in enumerate(questions, 1)
    ]


@pytest.fixture
def entry(tmp_path):
    """The entry as it would be unzipped, with the stand-in example next to main.py."""
    folder = tmp_path / "unzipped"
    folder.mkdir()
    shutil.copy(ROOT / "03_dsqa" / "main.py", folder / "main.py")
    shutil.copytree(
        ROOT / "03_dsqa" / "dsqa", folder / "dsqa", ignore=shutil.ignore_patterns("__pycache__")
    )
    (folder / "example_rag.py").write_text(EXAMPLE, encoding="utf-8")
    return folder


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    paper_pdf(folder / "b-loop.pdf", BODY)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        document.save(folder / "a-trap.pdf")
    return folder


@pytest.fixture
def run(entry, corpus, tmp_path, monkeypatch, capfd):
    """Run the entry over two questions with some adjustments and some
    behaviour of the stand-in example. Returns the rows of its answers and
    what was printed, by the entry and by the process the example's index is
    made in."""

    def run(variants="dones-example", example="", folder=entry, engine=FAKE):
        monkeypatch.setenv("DSQA_ENGINE", str(engine))
        monkeypatch.setenv("FAKE_ENGINE_FILE", str(FAKE))
        monkeypatch.setenv("DSQA_FAKE", "")
        monkeypatch.setenv("DSQA_VARIANT", variants)
        monkeypatch.setenv("EXAMPLE_FAKE", example)
        meta = run_local.run(
            folder,
            asked("Where is the nitrogen trap?", "How is the lithium carried?"),
            model_id="the/base-model",
            corpus_dir=corpus,
            out_dir=tmp_path / "runs",
            say=lambda line: None,
        )
        text = (tmp_path / "runs" / meta["run_id"] / "answers.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines()]
        assert meta["completed"] and not run_local.broke_the_contract(meta)
        assert all(row["status"] == "ok" and row["answer"].strip() for row in rows)
        return rows, capfd.readouterr().out

    return run


def test_the_example_reads_the_corpus_before_the_model_is_loaded(run):
    _, printed = run()
    made = printed.index("[dsqa] the example's index: 2 chunks in")
    assert printed.index("[rag] parsed into 2 chunks") < made < printed.index(LOADED)
    # Read once for the run, as the entry's own index is.
    assert printed.count("[rag] parsed into") == 1 and printed.count(LOADED) == 1


def test_without_the_adjustment_the_example_is_not_read(run):
    _, printed = run(variants="")
    assert "[rag]" not in printed and "the example's index" not in printed


FAILED = "the example's index could not be made: RuntimeError: its process ended with code 1"


@pytest.mark.parametrize(
    ("example", "said"),
    [("raises", FAILED), ("unreadable", "the example's index: 0 chunks in")],
)
def test_an_example_that_reads_nothing_leaves_the_entry_answering(run, example, said):
    _, printed = run(example=example)
    assert said in printed
    assert printed.index(said) < printed.index(LOADED)


def test_without_the_example_s_file_the_entry_answers_as_it_did(run, entry):
    (entry / "example_rag.py").unlink()
    _, printed = run()
    assert FAILED in printed


# The fake engine with what the example needs of the base model through
# transformers: a model with a generation config, and a tokenizer.
TRANSFORMERS_LIKE = """import importlib.util
import os
import types

spec = importlib.util.spec_from_file_location("fake_engine", os.environ["FAKE_ENGINE_FILE"])
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)


def load(model_id):
    engine = fake.load(model_id)
    engine.model = types.SimpleNamespace(generation_config=types.SimpleNamespace(max_time=None))
    engine.tokenizer = "the tokenizer"
    return engine
"""


def test_a_question_on_ifmif_dones_is_answered_by_the_example_as_it_answers(run, entry, tmp_path):
    if not (entry / "dsqa" / "route.py").is_file():
        pytest.skip("which questions the example answers is route.py's, not on this branch")
    engine = tmp_path / "transformers_like.py"
    engine.write_text(TRANSFORMERS_LIKE, encoding="utf-8")
    rows, printed = run(engine=engine)
    # The first is on the nitrogen trap, a document with no word of
    # IFMIF-DONES: the entry's own way. The second, on the lithium loop, is
    # the example's, and its reply is the answer as the example gave it.
    assert rows[0]["answer"].startswith("An answer with passages from the/base-model")
    assert rows[1]["answer"] == "the example's answer"
    assert "answered the dones-example way" in printed
    # With the fake engine alone there is no model for the example: the
    # entry's own way answers both.
    rows, _ = run()
    assert all(row["answer"].startswith("An answer") for row in rows)
