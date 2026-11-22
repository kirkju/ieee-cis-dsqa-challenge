"""tools/run_local.py: the local stand-in for the evaluator.

Submissions here are a few lines written into a temporary folder, so the
runner is exercised without any model.
"""

import itertools
import json
import os
import socket
import sys
import textwrap
import types
import zipfile

import pytest

from tools import ledger, run_local

QUESTIONS = [
    {"id": 1, "domain": "DONES", "question_type": "factual", "question": "first?"},
    {"id": 2, "domain": "AI Papers", "question_type": "causal", "question": "second?"},
    {"id": 3, "domain": "AI Papers", "question_type": "factual", "question": "third?"},
]

ECHO = """
    def answer(question, corpus_dir, model_id):
        return f"{model_id} heard: {question}"
"""


def write_submission(folder, source, **helpers):
    folder.mkdir(parents=True)
    (folder / "main.py").write_text(textwrap.dedent(source), encoding="utf-8")
    for name, text in helpers.items():
        (folder / f"{name}.py").write_text(textwrap.dedent(text), encoding="utf-8")
    return folder


def write_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, textwrap.dedent(text))
    return path


def quiet(line):
    pass


@pytest.fixture
def execute(tmp_path):
    """Run a submission given as source code. Returns the run's summary and
    the rows of its answers file."""
    numbers = itertools.count(1)

    def execute(source, questions=QUESTIONS, helpers=None, **options):
        folder = write_submission(tmp_path / f"entry{next(numbers)}", source, **(helpers or {}))
        meta = run_local.run(
            folder, questions, model_id="stand-in", out_dir=tmp_path / "runs", say=quiet, **options
        )
        return meta, read_answers(tmp_path / "runs" / meta["run_id"])

    return execute


def read_answers(run_dir):
    text = (run_dir / "answers.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_every_question_is_asked_in_order_and_its_answer_kept(execute):
    meta, rows = execute(ECHO)

    assert [row["answer"] for row in rows] == [
        "stand-in heard: first?",
        "stand-in heard: second?",
        "stand-in heard: third?",
    ]
    assert rows[0] == {
        "id": 1,
        "domain": "DONES",
        "question_type": "factual",
        "question": "first?",
        "status": "ok",
        "answer": "stand-in heard: first?",
        "seconds": rows[0]["seconds"],
        "error": None,
    }
    assert meta["counts"] == {"ok": 3, "empty": 0, "not_a_string": 0, "error": 0}
    assert meta["completed"] is True
    assert not run_local.broke_the_contract(meta)


def test_what_the_submission_keeps_in_memory_survives_between_calls(execute):
    _, rows = execute(
        """
        calls = []

        def answer(question, corpus_dir, model_id):
            calls.append(question)
            return str(len(calls))
        """
    )
    assert [row["answer"] for row in rows] == ["1", "2", "3"]


def test_an_exception_is_recorded_and_the_run_goes_on(execute):
    meta, rows = execute(
        """
        def answer(question, corpus_dir, model_id):
            if question == "second?":
                raise ValueError("no index for this one")
            return "fine"
        """
    )
    assert [row["status"] for row in rows] == ["ok", "error", "ok"]
    assert rows[1]["answer"] is None
    assert "ValueError: no index for this one" in rows[1]["error"]
    assert meta["counts"]["error"] == 1
    assert meta["completed"] is True
    assert run_local.broke_the_contract(meta)


@pytest.mark.parametrize(("returned", "type_name"), [("42", "int"), ("None", "NoneType")])
def test_a_return_that_is_not_a_string_is_no_answer(execute, returned, type_name):
    meta, rows = execute(
        f"""
        def answer(question, corpus_dir, model_id):
            return {returned}
        """,
        questions=QUESTIONS[:1],
    )
    assert rows[0]["status"] == "not_a_string"
    assert rows[0]["answer"] is None
    assert rows[0]["returned"] == type_name
    assert run_local.broke_the_contract(meta)


def test_a_blank_answer_is_counted_apart(execute):
    meta, rows = execute(
        """
        def answer(question, corpus_dir, model_id):
            return "   "
        """,
        questions=QUESTIONS[:1],
    )
    assert rows[0]["status"] == "empty"
    assert meta["counts"]["empty"] == 1
    # It is a string, so the contract holds; it is just not worth anything.
    assert not run_local.broke_the_contract(meta)


def test_a_module_next_to_main_is_importable(execute):
    _, rows = execute(
        """
        from helper import shout

        def answer(question, corpus_dir, model_id):
            return shout(question)
        """,
        helpers={"helper": "def shout(text):\n    return text.upper()\n"},
        questions=QUESTIONS[:1],
    )
    assert rows[0]["answer"] == "FIRST?"


def test_development_code_cannot_be_imported_by_a_submission(execute):
    meta, rows = execute(
        """
        import tools.ledger

        def answer(question, corpus_dir, model_id):
            return "leaned on development code"
        """
    )
    assert rows == []
    assert meta["completed"] is False
    assert "could not be imported" in meta["problems"][0]
    assert "No module named 'tools'" in meta["problems"][0]
    assert run_local.broke_the_contract(meta)
    # ... and it is back in place for the runner itself afterwards.
    assert sys.modules["tools.ledger"] is ledger


def test_the_entry_s_own_package_is_imported_from_the_submission_s_folder(execute, tmp_path):
    # A test session has the entry's package imported from the repository. A
    # submission that brings a package of that name has to get its own.
    import dsqa

    assert run_local._is_ours(dsqa.__file__)
    meta, rows = execute(
        """
        import dsqa

        def answer(question, corpus_dir, model_id):
            return dsqa.WHOSE
        """,
        questions=QUESTIONS[:1],
        helpers={"dsqa": "WHOSE = 'the submission'\n"},
    )
    assert [row["answer"] for row in rows] == ["the submission"]
    assert sys.modules["dsqa"] is dsqa  # and the repository's is back afterwards


def test_the_program_itself_stays_where_libraries_look_for_it(execute, monkeypatch):
    # Started as python -m tools.run_local, the program is a file of the
    # repository, and transformers looks its caller up as sys.modules["__main__"].
    program = types.ModuleType("__main__")
    program.__file__ = run_local.__file__
    monkeypatch.setitem(sys.modules, "__main__", program)
    meta, rows = execute(
        """
        import sys

        def answer(question, corpus_dir, model_id):
            return str(sys.modules["__main__"].__file__ is not None)
        """,
        questions=QUESTIONS[:1],
    )
    assert [row["answer"] for row in rows] == ["True"]


def test_the_interpreter_is_left_as_it_was(execute):
    path_before = sys.path[:]
    getaddrinfo_before = socket.getaddrinfo
    directory_before = os.getcwd()

    execute(ECHO, helpers={"helper": "VALUE = 1\n"})

    assert sys.path == path_before
    assert "main" not in sys.modules
    assert "helper" not in sys.modules
    assert socket.getaddrinfo is getaddrinfo_before
    assert os.getcwd() == directory_before


def test_main_without_an_answer_function_is_reported(execute):
    meta, rows = execute(
        """
        def respond(question, corpus_dir, model_id):
            return "wrong name"
        """
    )
    assert rows == []
    assert meta["problems"] == ["main.py does not define a function named answer"]
    assert run_local.broke_the_contract(meta)


def test_an_answer_that_takes_too_few_arguments_is_reported(execute):
    meta, rows = execute(
        """
        def answer(question, corpus_dir):
            return "two arguments only"
        """
    )
    assert any("three positional arguments" in problem for problem in meta["problems"])
    assert [row["status"] for row in rows] == ["error"] * 3


def test_argument_names_other_than_the_documented_ones_are_reported(execute):
    # The call below works by position; the evaluator may call by name.
    meta, rows = execute(
        """
        def answer(q, corpus, model):
            return "fine"
        """
    )
    assert [row["status"] for row in rows] == ["ok"] * 3
    assert any("by name" in problem for problem in meta["problems"])
    assert run_local.broke_the_contract(meta)


def test_the_corpus_folder_is_handed_over_as_given(execute, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("a.pdf", "b.pdf", "notes.txt"):
        (corpus / name).write_bytes(b"")

    meta, rows = execute(
        """
        def answer(question, corpus_dir, model_id):
            return corpus_dir
        """,
        questions=QUESTIONS[:1],
        corpus_dir=corpus,
    )
    assert rows[0]["answer"] == str(corpus.resolve())
    assert meta["corpus_pdfs"] == 2


def test_without_a_corpus_an_empty_folder_is_handed_over(execute):
    _, rows = execute(
        """
        import os

        def answer(question, corpus_dir, model_id):
            return f"{os.path.isdir(corpus_dir)} {os.listdir(corpus_dir)}"
        """,
        questions=QUESTIONS[:1],
    )
    assert rows[0]["answer"] == "True []"


def test_the_run_happens_in_an_empty_folder_and_what_is_left_there_is_listed(execute):
    meta, rows = execute(
        """
        import os

        def answer(question, corpus_dir, model_id):
            seen = sorted(os.listdir("."))
            open("scratch.txt", "w").close()
            return repr(seen)
        """,
        questions=QUESTIONS[:2],
    )
    assert [row["answer"] for row in rows] == ["[]", "['scratch.txt']"]
    assert meta["left_in_working_directory"] == ["scratch.txt"]


def test_the_offline_switches_are_set_for_the_run_only(execute, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    _, rows = execute(
        """
        import os

        def answer(question, corpus_dir, model_id):
            switches = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            return "".join(os.environ.get(name, "-") for name in switches)
        """,
        questions=QUESTIONS[:1],
    )
    assert rows[0]["answer"] == "11"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_a_host_name_outside_this_machine_is_refused_and_listed(execute):
    meta, rows = execute(
        """
        import socket

        def answer(question, corpus_dir, model_id):
            socket.create_connection(("example.com", 443), timeout=5)
            return "reached the internet"
        """,
        questions=QUESTIONS[:1],
    )
    assert rows[0]["status"] == "error"
    assert "no internet in the evaluation" in rows[0]["error"]
    assert meta["blocked_connections"] == ["example.com:443"]


def test_an_address_outside_this_machine_is_refused_and_listed(execute):
    # 203.0.113.0/24 is reserved for documentation: nothing answers there.
    meta, rows = execute(
        """
        import socket

        def answer(question, corpus_dir, model_id):
            with socket.socket() as connection:
                code = connection.connect_ex(("203.0.113.5", 80))
                connection.connect(("203.0.113.5", 80))
            return f"reached the internet ({code})"
        """,
        questions=QUESTIONS[:1],
    )
    assert rows[0]["status"] == "error"
    assert "ConnectionRefusedError" in rows[0]["error"]
    assert meta["blocked_connections"] == ["203.0.113.5:80"]


def test_this_machine_stays_reachable(execute):
    # A stand-in model served on localhost has to keep working.
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        meta, rows = execute(
            f"""
            import socket

            def answer(question, corpus_dir, model_id):
                with socket.create_connection(("127.0.0.1", {port}), timeout=5):
                    return "connected"
            """,
            questions=QUESTIONS[:1],
        )
    assert rows[0]["answer"] == "connected"
    assert meta["blocked_connections"] == []


def test_the_internet_is_left_alone_when_asked(execute):
    meta, rows = execute(
        """
        import os
        import socket

        def answer(question, corpus_dir, model_id):
            return socket.getaddrinfo.__name__ + os.environ.get("HF_HUB_OFFLINE", "-")
        """,
        questions=QUESTIONS[:1],
        allow_network=True,
    )
    assert rows[0]["answer"].startswith("getaddrinfo")
    assert "blocked_connections" not in meta


def test_an_interrupted_run_keeps_the_answers_it_had(execute, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        execute(
            """
            def answer(question, corpus_dir, model_id):
                if question == "second?":
                    raise KeyboardInterrupt
                return "fine"
            """
        )
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert [row["answer"] for row in read_answers(run_dir)] == ["fine"]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["completed"] is False
    assert run_local.broke_the_contract(meta)


def test_gpu_memory_is_noted_after_each_call(execute, monkeypatch):
    # Stands in for torch.cuda. It has no way to reset the peak counters: the
    # runner only reads, because a submission may be reading them too.
    gigabyte = 2**30
    cuda = types.SimpleNamespace(
        is_initialized=lambda: True,
        memory_allocated=lambda: 2 * gigabyte,
        memory_reserved=lambda: 3 * gigabyte,
        max_memory_allocated=lambda: 5 * gigabyte,
        max_memory_reserved=lambda: 6 * gigabyte,
        get_device_name=lambda index: "Test GPU",
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    meta, rows = execute(ECHO, questions=QUESTIONS[:2])

    memory = {
        "allocated_gb": 2.0,
        "reserved_gb": 3.0,
        "peak_allocated_gb": 5.0,
        "peak_reserved_gb": 6.0,
    }
    assert [row["gpu"] for row in rows] == [memory, memory]
    assert meta["gpu"] == {"name": "Test GPU", **memory}


def test_an_unused_gpu_is_left_alone(execute, monkeypatch):
    # Asking CUDA anything before the submission has used it would start it.
    def must_not_be_called(*args):
        raise AssertionError("the runner touched a GPU the submission had not used")

    cuda = types.SimpleNamespace(
        is_initialized=lambda: False,
        memory_allocated=must_not_be_called,
        get_device_name=must_not_be_called,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    meta, rows = execute(ECHO, questions=QUESTIONS[:1])

    assert "gpu" not in rows[0]
    assert meta["gpu"] == {}


def test_the_clock_is_read_around_the_import_and_around_each_call(execute):
    # A clock that advances one second every time it is read.
    ticks = itertools.count()
    meta, rows = execute(ECHO, clock=lambda: float(next(ticks)))

    assert [row["seconds"] for row in rows] == [1.0, 1.0, 1.0]
    assert meta["timings"] == run_local.summarize_timings(1.0, [1.0, 1.0, 1.0], 8.0)


def test_timings_separate_what_is_paid_once_from_what_is_paid_per_question():
    # 2 s to import, a first call of 100 s that loads everything, then 10 and 20 s.
    assert run_local.summarize_timings(2.0, [100.0, 10.0, 20.0], 132.0) == {
        "import_s": 2.0,
        "total_s": 132.0,
        "first_call_s": 100.0,
        "per_question_s": 15.0,
        "slowest_later_call_s": 20.0,
        "fixed_s": 87.0,  # the import, plus the first call beyond an ordinary question
        "projected_public_s": 87.0 + 50 * 15.0,
        "projected_private_s": 87.0 + 100 * 15.0,
    }


def test_a_single_question_gives_no_projection():
    assert run_local.summarize_timings(1.0, [5.0], 6.0) == {
        "import_s": 1.0,
        "total_s": 6.0,
        "first_call_s": 5.0,
    }


def test_a_zip_is_run_from_what_it_contains(tmp_path):
    archive = write_zip(
        tmp_path / "entry.zip",
        {
            "main.py": "from helper import ANSWER\n\n"
            "def answer(question, corpus_dir, model_id):\n    return ANSWER\n",
            "helper.py": "ANSWER = 'from inside the zip'\n",
        },
    )
    meta = run_local.run(
        archive, QUESTIONS[:1], model_id="stand-in", out_dir=tmp_path / "runs", say=quiet
    )
    rows = read_answers(tmp_path / "runs" / meta["run_id"])
    assert rows[0]["answer"] == "from inside the zip"
    assert meta["submission"] == "entry.zip"


def test_a_zip_with_main_inside_a_folder_is_refused(tmp_path):
    archive = write_zip(tmp_path / "entry.zip", {"entry/main.py": ECHO})
    with pytest.raises(run_local.SubmissionError, match="top level"):
        run_local.run(
            archive, QUESTIONS, model_id="stand-in", out_dir=tmp_path / "runs", say=quiet
        )


def test_a_folder_without_main_is_refused(tmp_path):
    folder = tmp_path / "entry"
    folder.mkdir()
    with pytest.raises(run_local.SubmissionError, match="no main.py"):
        run_local.run(folder, QUESTIONS, model_id="stand-in", out_dir=tmp_path / "runs", say=quiet)


def test_questions_load_from_a_json_list_or_from_json_lines(tmp_path):
    as_list = tmp_path / "questions.json"
    as_list.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    as_lines = tmp_path / "questions.jsonl"
    as_lines.write_text("\n".join(json.dumps(item) for item in QUESTIONS) + "\n", encoding="utf-8")

    assert run_local.load_questions(as_list) == QUESTIONS
    assert run_local.load_questions(as_lines) == QUESTIONS


def test_a_question_file_of_another_shape_is_refused(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": QUESTIONS}), encoding="utf-8")
    with pytest.raises(ValueError, match="list of objects"):
        run_local.load_questions(path)


def test_questions_are_picked_by_id_and_then_capped():
    assert run_local.select(QUESTIONS) == QUESTIONS
    assert run_local.select(QUESTIONS, limit=2) == QUESTIONS[:2]
    assert run_local.select(QUESTIONS, ids=["3", "1"]) == [QUESTIONS[0], QUESTIONS[2]]
    assert run_local.select(QUESTIONS, limit=1, ids=["2", "3"]) == [QUESTIONS[1]]


@pytest.fixture
def command_line(tmp_path):
    """Arguments for the command, minus the submission."""
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    out = tmp_path / "runs"
    return ["--model-id", "stand-in", "--questions", str(questions), "--out", str(out)]


def test_the_command_reports_the_run_and_can_record_it(tmp_path, command_line, capsys):
    folder = write_submission(tmp_path / "entry", ECHO)
    rows_file = tmp_path / "ledger.jsonl"

    status = run_local.main(
        [str(folder), *command_line, "--record", "--ledger", str(rows_file), "--note", "plumbing"]
    )

    assert status == 0
    assert "3 answered" in capsys.readouterr().out
    (row,) = ledger.read(rows_file)
    assert row["kind"] == "run"
    assert (row["dataset"], row["corpus"], row["model"]) == ("questions", "none", "stand-in")
    assert row["metrics"] == {"questions": 3, "ok": 3, "empty": 0, "not_a_string": 0, "error": 0}
    assert row["config"]["submission"] == "entry"
    assert row["notes"] == "plumbing"
    # The row and the folder with the answers go by the same name.
    assert (tmp_path / "runs" / row["run_id"] / "answers.jsonl").is_file()


def test_the_command_does_not_record_unless_asked(tmp_path, command_line):
    folder = write_submission(tmp_path / "entry", ECHO)
    rows_file = tmp_path / "ledger.jsonl"
    assert run_local.main([str(folder), *command_line, "--ledger", str(rows_file)]) == 0
    assert not rows_file.exists()


def test_the_exit_status_tells_a_broken_contract(tmp_path, command_line):
    folder = write_submission(
        tmp_path / "entry", "def answer(question, corpus_dir, model_id):\n    return None\n"
    )
    assert run_local.main([str(folder), *command_line]) == 1


def test_the_command_refuses_a_folder_without_main(tmp_path, command_line, capsys):
    (tmp_path / "entry").mkdir()
    assert run_local.main([str(tmp_path / "entry"), *command_line]) == 1
    assert "no main.py" in capsys.readouterr().out
