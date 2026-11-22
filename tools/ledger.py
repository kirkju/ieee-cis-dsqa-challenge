"""Append-only ledger of measured runs: results/experiments.jsonl.

Every evaluation that produces a number worth keeping appends one row here,
stamped with the commit it ran on, so a figure quoted in the journal or in
the paper can be traced to the exact code that produced it. Rows are written
by tools, never by hand.

One JSON object per line rather than CSV, because rows are appended from
more than one machine and branch (the laptop, rented GPU pods): with the
"merge=union" attribute in .gitattributes concurrent appends merge without
conflicts, and each kind of run can carry its own metrics.

    python -m tools.ledger                        # list the recorded runs
    python -m tools.ledger --kind retrieval --last 10
"""

import argparse
import hashlib
import json
import platform
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "results" / "experiments.jsonl"

# Paths that only hold records of what was run. A change there does not alter
# what a run measures, so it does not make the tree dirty; otherwise appending
# a row would taint every run after it until the ledger was committed.
RECORD_PATHS = ("results/", "docs/")


def _git(repo: Path, *args: str) -> str | None:
    """Output of a git command run in repo, or None when it fails (git is
    missing, repo is not a repository, there are no commits yet)."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout


def _changed_paths(status: str) -> list[str]:
    """Paths named in the output of `git status --porcelain -z`."""
    entries = status.split("\0")
    paths = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if not entry:
            continue
        code, path = entry[:2], entry[3:]
        paths.append(path)
        # A rename or copy is followed by the path it came from.
        if ("R" in code or "C" in code) and i < len(entries):
            paths.append(entries[i])
            i += 1
    return paths


def git_state(repo: Path = REPO_ROOT) -> dict:
    """The commit a run will be attributed to, and whether the code on disk
    really is that commit."""
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(repo, "status", "--porcelain", "-z", "--untracked-files=all")
    if status is None:
        # Unknown is treated as dirty: a row is only quotable when the tree
        # was positively seen to be clean.
        dirty = True
    else:
        dirty = any(not path.startswith(RECORD_PATHS) for path in _changed_paths(status))
    return {"commit": commit.strip(), "branch": branch.strip() if branch else None, "dirty": dirty}


def config_hash(config: dict | None) -> str:
    """Short fingerprint of a run's configuration, independent of key order."""
    canonical = json.dumps(config or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _plain(value):
    """json.dumps fallback: numpy and torch scalars, and paths."""
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} cannot be stored in the ledger")


def new_run_id(now: datetime, commit: str | None) -> str:
    """When the run happened and on which commit, plus a few random characters
    so two runs started in the same second do not collide."""
    return f"{now:%Y%m%dT%H%M%SZ}-{(commit or 'nogit')[:7]}-{secrets.token_hex(2)}"


def _ends_without_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        f.seek(-1, 2)
        return f.read(1) != b"\n"


def record(
    kind: str,
    metrics: dict,
    *,
    dataset: str,
    corpus: str,
    engine: str,
    model: str = "",
    config: dict | None = None,
    timings: dict | None = None,
    notes: str = "",
    machine: str | None = None,
    run_id: str | None = None,
    state: dict | None = None,
    path: Path = LEDGER_PATH,
    repo: Path = REPO_ROOT,
) -> dict:
    """Append one run to the ledger and return the row that was written.

    kind     what was measured, e.g. "retrieval" or "end_to_end"
    dataset  the question set and its version, e.g. "sample-25"
    corpus   the corpus version that was searched, or "" when none was used
    engine   what generated the text: "none", a stand-in, or the base model
    machine  where it ran, when timings are to be compared across machines
    run_id   for a tool that already named its output after the run
    state    git_state() as it was when a long run started; by default the
             state at the moment of recording
    """
    now = datetime.now(timezone.utc)
    state = git_state(repo) if state is None else state
    row = {
        "run_id": run_id or new_run_id(now, state["commit"]),
        "utc": now.isoformat(timespec="seconds"),
        **state,
        "kind": kind,
        "dataset": dataset,
        "corpus": corpus,
        "engine": engine,
        "model": model,
        "machine": machine or platform.system().lower(),
        "config_hash": config_hash(config),
        "config": config or {},
        "metrics": metrics,
        "timings": timings or {},
        "notes": notes,
    }
    # allow_nan=False: NaN and Infinity are not JSON, and a metric that is not
    # a finite number is a failed measurement, not a result to keep. Serializing
    # before the file is opened means a refused row leaves no trace.
    line = json.dumps(row, ensure_ascii=False, allow_nan=False, default=_plain)
    # Return what a reader of the ledger will get back, not the objects passed in.
    row = json.loads(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    # A union merge or a hand edit can leave the file without a final newline;
    # appending straight after it would glue two rows into one line.
    prefix = "\n" if _ends_without_newline(path) else ""
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(prefix + line + "\n")
    return row


def read(path: Path = LEDGER_PATH) -> list[dict]:
    """Every recorded run, in file order."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {number} is not valid JSON ({e.msg})") from e
    return rows


def _format(row: dict) -> str:
    commit = (row.get("commit") or "-------")[:7] + ("*" if row.get("dirty") else " ")
    metrics = "  ".join(
        f"{name}={value:.4g}" if isinstance(value, float) else f"{name}={value}"
        for name, value in row.get("metrics", {}).items()
    )
    return (
        f"{row['utc'][:16]}  {commit}  {row['kind']:<12}  {row['dataset']:<14}  "
        f"{row['engine']:<12}  {metrics}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the runs recorded in the ledger.")
    parser.add_argument("--kind", help="only runs of this kind")
    parser.add_argument("--last", type=int, metavar="N", help="only the N most recent runs")
    parser.add_argument("--path", type=Path, default=LEDGER_PATH, help="ledger file to read")
    args = parser.parse_args(argv)

    # Rows merged from another branch are not in chronological order in the file.
    rows = sorted(read(args.path), key=lambda row: row["utc"])
    if args.kind:
        rows = [row for row in rows if row["kind"] == args.kind]
    if args.last:
        rows = rows[-args.last :]
    for row in rows:
        print(_format(row))
    if any(row.get("dirty") for row in rows):
        print("* ran on uncommitted code: not to be quoted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
