"""Local stand-in for the evaluator: run a submission over a question set.

Imports a submission's main.py the way the evaluation does, as far as that is
known, calls answer() once per question in a single process, and records each
answer, how long it took and anything that went wrong.

    python -m tools.run_local 01_zero_shot --model-id <hf repo id>
    python -m tools.run_local 02_rag --model-id <id> --corpus-dir data/corpus --record
    python -m tools.run_local some_submission.zip --model-id <id> --limit 5

What is reproduced, and why:

- One process, one import of main.py, one call per question, in order. What
  the submission keeps in memory (model, index) survives between calls.
- The clock starts before main.py is imported, because the time limit counts
  from when the submission's code starts running.
- No internet. HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are set, and Python
  connections to anything but this machine are refused and listed.
- Besides the installed packages, only the submission's own folder is
  importable: the repository root is taken off sys.path and this package is
  hidden, so a submission that leans on development code fails here.
- The working directory is an empty temporary folder. The evaluator's is not
  documented, so a submission must not depend on it. Files left there are
  listed.

An exception or a non-string return is recorded and the run goes on, to show
every problem in one pass. In the evaluation either can cost the question or
the whole submission, so the exit status is then 1.

What is not reproduced: the evaluation machine, its filesystem and its system
libraries. A clean run here is necessary, not sufficient.
"""

import argparse
import contextlib
import errno
import hashlib
import importlib
import inspect
import ipaddress
import json
import os
import platform
import socket
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from tools import ledger

REPO_ROOT = ledger.REPO_ROOT
DEFAULT_QUESTIONS = REPO_ROOT / "sample" / "sample_qa.json"
DEFAULT_OUT = REPO_ROOT / "runs"

# Number of questions and wall-clock limit, in seconds, of each leaderboard.
PHASES = {"public": (50, 3600), "private": (100, 7200)}


class SubmissionError(Exception):
    """The submission cannot be run at all, as opposed to failing on a question."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions(path: Path) -> list[dict]:
    """A JSON list, which is the organizers' sample format, or JSON Lines."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        items = json.loads(text)
    well_formed = isinstance(items, list) and all(
        isinstance(item, dict) and isinstance(item.get("question"), str) for item in items
    )
    if not well_formed:
        raise ValueError(f"{path}: expected a list of objects, each with a 'question' string")
    return items


def select(questions: list[dict], limit: int | None = None, ids: list[str] | None = None):
    """The questions to ask: those with the given ids, then at most `limit`."""
    if ids:
        wanted = set(ids)
        questions = [item for item in questions if str(item.get("id")) in wanted]
    return questions[:limit] if limit else questions


def _unpack(submission: Path, into: Path) -> Path:
    """The folder holding the submission's main.py. A zip is extracted first,
    so that what runs is what would be uploaded."""
    if submission.is_dir():
        folder = submission
    elif submission.is_file() and zipfile.is_zipfile(submission):
        with zipfile.ZipFile(submission) as archive:
            archive.extractall(into)
        folder = into
    else:
        raise SubmissionError(f"{submission} is neither a folder nor a zip file")
    if not (folder / "main.py").is_file():
        nested = sorted(path.parent.name for path in folder.glob("*/main.py"))
        hint = ""
        if nested:
            hint = f"; there is one inside '{nested[0]}/', it has to be at the top level"
        raise SubmissionError(f"no main.py at the root of {submission.name}{hint}")
    return folder


def _is_inside(path: str, folder: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(folder)
    except (OSError, ValueError):
        return False


def _is_ours(path: str) -> bool:
    """Whether a file or folder is part of this repository, the virtualenv
    that may live inside it apart: the installed packages are the evaluation's
    too, our own code is not."""
    return _is_inside(path, REPO_ROOT) and not _is_inside(path, Path(sys.prefix).resolve())


@contextlib.contextmanager
def _evaluation_imports(folder: Path):
    """Make the submission's folder importable, and nothing else of ours.

    What was imported from the repository before the run is hidden for its
    length, the entry's own package included: the submission has to import
    its copy, from its folder, as it will in the evaluation.
    """
    folder = folder.resolve()
    saved_path = sys.path[:]
    hidden = {}
    for name, module in sys.modules.items():
        origin = getattr(module, "__file__", None)
        # Not the program itself, whatever file it was started from: the
        # libraries a submission loads look their caller up by that name.
        if name in ("__main__", "__mp_main__"):
            continue
        if name.split(".")[0] == "tools" or (origin and _is_ours(origin)):
            hidden[name] = module
    saved_main = sys.modules.pop("main", None)
    hidden.pop("main", None)
    for name in hidden:
        del sys.modules[name]
    # An empty entry means the working directory, which is replaced during the run.
    kept = [entry for entry in saved_path if entry and not _is_ours(entry)]
    sys.path[:] = [str(folder), *kept]
    importlib.invalidate_caches()
    try:
        yield
    finally:
        # Unload whatever came from the submission, so another run in the same
        # process starts from nothing, as a new evaluation would.
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if origin and _is_inside(origin, folder):
                del sys.modules[name]
        sys.path[:] = saved_path
        sys.modules.update(hidden)
        if saved_main is not None:
            sys.modules["main"] = saved_main


@contextlib.contextmanager
def _environment(**values: str):
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _empty_working_directory(left_behind: list[str]):
    """Run in a new, empty folder and list the files found there afterwards."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="dsqa-cwd-", ignore_cleanup_errors=True) as scratch:
        os.chdir(scratch)
        try:
            yield
        finally:
            files = (path for path in Path(scratch).rglob("*") if path.is_file())
            left_behind.extend(sorted(path.relative_to(scratch).as_posix() for path in files))
            os.chdir(previous)


class _NoInternet:
    """Refuse connections from Python to anything but this machine, and keep
    a list of what was attempted.

    It patches the socket module, which requests, httpx and urllib all end up
    in. A native client that opens its own sockets is not seen.
    """

    def __init__(self):
        self.blocked: list[str] = []

    def _is_local(self, host) -> bool:
        if not host:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if host in ("localhost", self._hostname):
            return True
        try:
            address = ipaddress.ip_address(host.split("%")[0])
        except ValueError:
            return False
        # The unspecified address (0.0.0.0) reaches this machine too.
        return address.is_loopback or address.is_unspecified

    def _refuse(self, host, port) -> str:
        target = f"{host}:{port}" if port else str(host)
        self.blocked.append(target)
        return f"no internet in the evaluation: the local runner refused a connection to {target}"

    def __enter__(self):
        self._hostname = socket.gethostname()
        self._saved = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)
        connect, connect_ex, getaddrinfo = self._saved

        def is_remote(address) -> bool:
            # Only internet addresses are (host, port, ...) tuples; a Unix
            # socket path cannot leave the machine.
            internet_address = isinstance(address, tuple) and len(address) >= 2
            return internet_address and not self._is_local(address[0])

        def guarded_connect(sock, address):
            if is_remote(address):
                raise ConnectionRefusedError(self._refuse(address[0], address[1]))
            return connect(sock, address)

        def guarded_connect_ex(sock, address):
            if is_remote(address):
                self._refuse(address[0], address[1])
                return errno.ECONNREFUSED
            return connect_ex(sock, address)

        def guarded_getaddrinfo(host, port, *args, **kwargs):
            if not self._is_local(host):
                raise socket.gaierror(socket.EAI_NONAME, self._refuse(host, port))
            return getaddrinfo(host, port, *args, **kwargs)

        socket.socket.connect = guarded_connect
        socket.socket.connect_ex = guarded_connect_ex
        socket.getaddrinfo = guarded_getaddrinfo
        return self

    def __exit__(self, *exc_info):
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = self._saved


def signature_problems(answer) -> list[str]:
    """Ways answer() could not be called as answer(question, corpus_dir, model_id).

    Whether the evaluator passes the three arguments by position or by name is
    not documented, so both have to work.
    """
    try:
        signature = inspect.signature(answer)
    except (TypeError, ValueError):
        return []
    problems = []
    try:
        signature.bind("question", "corpus_dir", "model_id")
    except TypeError as e:
        problems.append(f"answer() does not take three positional arguments: {e}")
    try:
        signature.bind(question="q", corpus_dir="c", model_id="m")
    except TypeError as e:
        problems.append(f"answer() does not take question, corpus_dir and model_id by name: {e}")
    return problems


def summarize_timings(import_s: float, seconds: list[float], total_s: float) -> dict:
    """Split a run into what is paid once and what is paid per question.

    The first call carries the one-off costs (loading the model, building the
    index) and the later calls give the cost of a question. From the two, the
    total for each leaderboard is projected. A projection only means something
    on hardware like the evaluation machine's, running the base model.

    The split is a rough guide. A submission is a black box here, so the
    one-off part is taken as the import plus whatever the first call cost
    beyond an average later one; when questions differ a lot in cost and the
    first happens to be a cheap one, that comes out as the import alone.
    """
    timings = {"import_s": import_s, "total_s": total_s}
    if seconds:
        timings["first_call_s"] = seconds[0]
    later = seconds[1:]
    if later:
        per_question = sum(later) / len(later)
        fixed = import_s + max(seconds[0] - per_question, 0.0)
        timings.update(per_question_s=per_question, slowest_later_call_s=max(later), fixed_s=fixed)
        for phase, (count, _limit) in PHASES.items():
            timings[f"projected_{phase}_s"] = fixed + count * per_question
    return {name: round(value, 3) for name, value in timings.items()}


def _gigabytes(count: int) -> float:
    return round(count / 2**30, 2)


def _gpu_memory() -> dict:
    """GPU memory of this process right now, if the submission has used CUDA
    through torch; empty otherwise.

    Read-only on purpose. The runner does not import torch, does not touch an
    uninitialized CUDA (asking it anything would initialize it) and does not
    reset the peak counters, which a submission may be reading itself.
    """
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return {}
        return {
            "allocated_gb": _gigabytes(torch.cuda.memory_allocated()),
            "reserved_gb": _gigabytes(torch.cuda.memory_reserved()),
            "peak_allocated_gb": _gigabytes(torch.cuda.max_memory_allocated()),
            "peak_reserved_gb": _gigabytes(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return {}


def _gpu_name() -> str | None:
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _ask(answer, item: dict, corpus_dir: str, model_id: str, clock) -> dict:
    """One call to answer(), with everything worth knowing about it."""
    started = clock()
    try:
        returned, error = answer(item["question"], corpus_dir, model_id), None
    except Exception:
        returned, error = None, traceback.format_exc()
    seconds = clock() - started

    if error is not None:
        status = "error"
    elif not isinstance(returned, str):
        status = "not_a_string"
    elif not returned.strip():
        status = "empty"
    else:
        status = "ok"
    row = {
        "id": item.get("id"),
        "domain": item.get("domain"),
        "question_type": item.get("question_type"),
        "question": item["question"],
        "status": status,
        "answer": returned if isinstance(returned, str) else None,
        "seconds": round(seconds, 3),
        "error": error,
    }
    if status == "not_a_string":
        row["returned"] = type(returned).__name__
    # What the call left on the GPU, and the peak so far. Memory that grows
    # from one call to the next is what slows down or kills a later question.
    gpu = _gpu_memory()
    if gpu:
        row["gpu"] = gpu
    return row


def run(
    submission: Path,
    questions: list[dict],
    *,
    model_id: str,
    corpus_dir: Path | None = None,
    out_dir: Path = DEFAULT_OUT,
    allow_network: bool = False,
    clock=time.perf_counter,
    say=print,
) -> dict:
    """Run a submission over the questions and return the run's summary.

    Writes answers.jsonl (one row per question, flushed as it goes, so an
    interrupted run keeps what it had) and meta.json into a new folder of
    out_dir named after the run.
    """
    submission = submission.resolve()
    started = datetime.now(timezone.utc)
    git = ledger.git_state()
    run_id = ledger.new_run_id(started, git["commit"])
    run_dir = out_dir.resolve() / run_id
    run_dir.mkdir(parents=True)

    meta = {
        "run_id": run_id,
        "utc": started.isoformat(timespec="seconds"),
        "git": git,
        "submission": submission.name,
        "model_id": model_id,
        "questions": len(questions),
        "network_allowed": allow_network,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "completed": False,
        "problems": [],
        "counts": {"ok": 0, "empty": 0, "not_a_string": 0, "error": 0},
    }
    left_behind: list[str] = []
    internet = contextlib.nullcontext() if allow_network else _NoInternet()
    offline = {} if allow_network else {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    seconds: list[float] = []

    try:
        with contextlib.ExitStack() as stack:
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="dsqa-run-")))
            folder = _unpack(submission, scratch / "submission")
            meta["main_sha256"] = _sha256(folder / "main.py")
            if corpus_dir is None:
                corpus_dir = scratch / "empty_corpus"
                corpus_dir.mkdir()
            corpus_dir = corpus_dir.resolve()
            meta["corpus_dir"] = str(corpus_dir)
            meta["corpus_pdfs"] = len(list(corpus_dir.glob("*.pdf")))

            answers = stack.enter_context(
                open(run_dir / "answers.jsonl", "w", encoding="utf-8", newline="\n")
            )
            stack.enter_context(_empty_working_directory(left_behind))
            stack.enter_context(_environment(**offline))
            blocked = stack.enter_context(internet)
            stack.enter_context(_evaluation_imports(folder))

            say(
                f"[runner] {submission.name}: {len(questions)} questions, "
                f"{meta['corpus_pdfs']} PDFs in the corpus, model {model_id}"
            )
            clock_started = clock()
            try:
                answer = getattr(importlib.import_module("main"), "answer", None)
            except Exception:
                answer = None
                meta["problems"].append("main.py could not be imported:\n" + traceback.format_exc())
            else:
                if not callable(answer):
                    meta["problems"].append("main.py does not define a function named answer")
                else:
                    meta["problems"].extend(signature_problems(answer))
            import_s = clock() - clock_started

            if callable(answer):
                for number, item in enumerate(questions, 1):
                    row = _ask(answer, item, str(corpus_dir), model_id, clock)
                    answers.write(json.dumps(row, ensure_ascii=False) + "\n")
                    answers.flush()
                    seconds.append(row["seconds"])
                    meta["counts"][row["status"]] += 1
                    say(
                        f"[runner] {number}/{len(questions)}  id={row['id']}  "
                        f"{row['status']}  {row['seconds']:.1f}s"
                    )
                meta["completed"] = True
            meta["timings"] = summarize_timings(import_s, seconds, clock() - clock_started)
            memory = _gpu_memory()
            meta["gpu"] = {"name": _gpu_name(), **memory} if memory else {}
            if blocked is not None:
                meta["blocked_connections"] = sorted(set(blocked.blocked))
    finally:
        meta["left_in_working_directory"] = left_behind
        # The tree may have changed under a long run; then the commit recorded
        # at the start no longer describes everything that ran.
        if ledger.git_state() != git:
            meta["git"] = {**git, "dirty": True}
            meta["problems"].append("the working tree changed while the run was in progress")
        with open(run_dir / "meta.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return meta


def broke_the_contract(meta: dict) -> bool:
    """Whether the evaluation could have failed this submission or dropped answers."""
    counts = meta["counts"]
    structural = [p for p in meta["problems"] if not p.startswith("the working tree changed")]
    return bool(structural or counts["error"] or counts["not_a_string"] or not meta["completed"])


def _report(meta: dict, say=print) -> None:
    counts, timings = meta["counts"], meta.get("timings", {})
    say(
        f"[runner] {meta['submission']}: {counts['ok']} answered, {counts['empty']} empty, "
        f"{counts['not_a_string']} not a string, {counts['error']} raised "
        f"(of {meta['questions']})"
    )
    for problem in meta["problems"]:
        say(f"[runner] PROBLEM: {problem}")
    if "first_call_s" in timings:
        say(
            f"[runner] import {timings['import_s']:.1f}s, "
            f"first call {timings['first_call_s']:.1f}s, total {timings['total_s']:.1f}s"
        )
    if "per_question_s" in timings:
        say(
            f"[runner] after the first call: {timings['per_question_s']:.1f}s per question, "
            f"slowest {timings['slowest_later_call_s']:.1f}s"
        )
        for phase, (count, limit) in PHASES.items():
            say(
                f"[runner] projected for the {phase} leaderboard ({count} questions): "
                f"{timings[f'projected_{phase}_s']:.0f}s of {limit}s, on this machine and model"
            )
    if meta.get("gpu"):
        gpu = meta["gpu"]
        say(f"[runner] GPU {gpu['name']}: peak {gpu['peak_allocated_gb']} GB allocated")
    if meta.get("blocked_connections"):
        say(f"[runner] tried to reach: {', '.join(meta['blocked_connections'])}")
    if meta["left_in_working_directory"]:
        left = ", ".join(meta["left_in_working_directory"])
        say(f"[runner] left in the working directory: {left}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a submission over a question set the way the evaluation would."
    )
    parser.add_argument("submission", type=Path, help="folder with main.py, or a submission zip")
    parser.add_argument("--model-id", required=True, help="passed to answer() as model_id")
    parser.add_argument("--corpus-dir", type=Path, help="folder of PDFs (default: an empty one)")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N questions")
    parser.add_argument("--ids", help="only the questions with these ids, comma-separated")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where run folders go")
    parser.add_argument("--allow-network", action="store_true", help="do not cut the internet")
    record = parser.add_argument_group("recording the run in the ledger")
    record.add_argument("--record", action="store_true", help="append a row to the ledger")
    record.add_argument("--dataset", help="name of the question set (default: its file name)")
    record.add_argument("--corpus", help="name of the corpus (default: its folder name)")
    record.add_argument("--engine", default="transformers", help="what generates the text")
    record.add_argument("--machine", help="where this runs, to compare timings across machines")
    record.add_argument("--note", default="", help="free text stored with the row")
    record.add_argument("--ledger", type=Path, default=ledger.LEDGER_PATH)
    args = parser.parse_args(argv)

    # A submission's own prints may hold characters the Windows console encoding
    # lacks, and failing on one would be a fault of this terminal. Line
    # buffering keeps the progress visible when the output goes to a file.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError) as e:
        parser.error(str(e))
    questions = select(questions, args.limit, args.ids.split(",") if args.ids else None)
    if not questions:
        parser.error("no questions selected")
    try:
        meta = run(
            args.submission,
            questions,
            model_id=args.model_id,
            corpus_dir=args.corpus_dir,
            out_dir=args.out,
            allow_network=args.allow_network,
        )
    except SubmissionError as e:
        print(f"[runner] {e}")
        return 1
    except KeyboardInterrupt:
        print("[runner] interrupted; the answers given so far were kept")
        return 130
    _report(meta)
    print(f"[runner] answers in {(args.out.resolve() / meta['run_id']).as_posix()}")

    if args.record:
        gpu = meta.get("gpu", {})
        metrics = {"questions": meta["questions"], **meta["counts"]}
        if gpu:
            metrics["gpu_peak_allocated_gb"] = gpu["peak_allocated_gb"]
            metrics["gpu_peak_reserved_gb"] = gpu["peak_reserved_gb"]
        corpus = args.corpus or (args.corpus_dir.resolve().name if args.corpus_dir else "none")
        ledger.record(
            "run",
            metrics,
            dataset=args.dataset or args.questions.stem,
            corpus=corpus,
            engine=args.engine,
            model=args.model_id,
            config={
                "submission": meta["submission"],
                "main_sha256": meta["main_sha256"],
                "questions_sha256": _sha256(args.questions),
                "limit": args.limit,
                "ids": args.ids,
                "network_allowed": args.allow_network,
            },
            timings=meta.get("timings", {}),
            notes=args.note,
            # Timings only compare across runs on the same hardware.
            machine=args.machine or gpu.get("name"),
            run_id=meta["run_id"],
            state=meta["git"],
            path=args.ledger,
        )
        print(f"[runner] recorded in {args.ledger.name} as {meta['run_id']}")
    return 1 if broke_the_contract(meta) else 0


if __name__ == "__main__":
    sys.exit(main())
