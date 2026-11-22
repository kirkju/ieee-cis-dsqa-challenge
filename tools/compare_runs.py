"""Compare two recorded runs question by question.

Two variants are not compared by their averages: an average hides that one
covers questions the other loses. This lists the questions on which the two
runs differ and says how likely a split like that is when neither is better.

    python -m tools.compare_runs RUN_A RUN_B            # two retrieval runs
    python -m tools.compare_runs RUN_A RUN_B --marks    # two runs of answers, by their marks
    python -m tools.compare_runs 20261009T040403Z-257534f-935a 20261009T05

A run is named by its id, or by the start of it, among the folders of
results/runs/ (or of --runs). Both have to be over the same questions.

Two retrieval runs are compared on the questions with a reference passage:
one counts for a run when its passages reached the context. Two runs of
answers are compared on the marks tools/judge.py left next to them: a
question counts for a run when its answer was marked correct. --judge names
the judge whose marks are read, for a run that has the marks of several.

The test is the exact sign test on the questions only one of the two covers
(McNemar's test in its exact form): were the runs equally good, each such
question would fall to either with probability one half.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from tools import eval_retrieval


def _folder(run: str, folder: Path) -> Path:
    matches = sorted(path for path in folder.glob(f"{run}*") if path.is_dir())
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise ValueError(f"{run!r} names {len(matches)} runs in {folder} ({found})")
    return matches[0]


def _rows(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["id"]: row for row in rows}


def load(run: str, folder: Path) -> tuple[str, dict]:
    """A run's id and its rows by question, the run given by the start of its id."""
    found = _folder(run, folder)
    return found.name, _rows(found / "questions.jsonl")


def load_marks(run: str, folder: Path, judge: str = "") -> tuple[str, dict]:
    """A run's id and the marks of its answers by question. The judge is
    named as its file is, judged-<judge>.jsonl, and has to be when the run
    was marked by more than one."""
    found = _folder(run, folder)
    files = sorted(found.glob(f"judged-{judge or '*'}.jsonl"))
    if len(files) != 1:
        names = ", ".join(path.name for path in files) or "none"
        raise ValueError(f"{found.name} has {len(files)} files of marks to read ({names})")
    return found.name, _rows(files[0])


def covered(row: dict) -> bool:
    return bool(row["passages"]) and row["lost"] is None


def right(row: dict) -> bool:
    return row["verdict"] == "correct"


def sign_test(only_a: int, only_b: int) -> float:
    """The probability of a split at least this uneven between two equally
    good runs, either way round."""
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def compare(a: dict, b: dict, marks: bool = False) -> dict:
    """The questions by which of the two runs they count for: those with a
    passage, by the run that covers them, or with marks every question, by
    the run whose answer is right."""
    if set(a) != set(b):
        raise ValueError("the two runs are not over the same questions")
    counts = right if marks else covered
    asked = [number for number in a if marks or a[number]["passages"]]
    split = {"both": [], "only_a": [], "only_b": [], "neither": []}
    for number in asked:
        in_a, in_b = counts(a[number]), counts(b[number])
        name = "both" if in_a and in_b else "only_a" if in_a else "only_b" if in_b else "neither"
        split[name].append(number)
    return {
        "questions": len(asked),
        **split,
        "p": sign_test(len(split["only_a"]), len(split["only_b"])),
    }


def _where(row: dict) -> str:
    return "in the context" if covered(row) else f"lost in {row['lost']} (rank {row['rank']})"


def _by_domain(numbers: list, rows: dict) -> dict[str, int]:
    count: dict[str, int] = {}
    for number in numbers:
        count[rows[number]["domain"]] = count.get(rows[number]["domain"], 0) + 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two recorded runs question by question.")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--marks", action="store_true", help="two runs of answers, by their marks")
    parser.add_argument("--judge", default="", help="whose marks, when a run has several judges'")
    parser.add_argument("--runs", type=Path, default=eval_retrieval.RECORDED_OUT)
    args = parser.parse_args(argv)
    try:
        if args.marks:
            name_a, a = load_marks(args.run_a, args.runs, args.judge)
            name_b, b = load_marks(args.run_b, args.runs, args.judge)
        else:
            name_a, a = load(args.run_a, args.runs)
            name_b, b = load(args.run_b, args.runs)
        result = compare(a, b, args.marks)
    except ValueError as e:
        parser.error(str(e))

    print(f"[compare] A {name_a}\n[compare] B {name_b}")
    both, neither = len(result["both"]), len(result["neither"])
    only_a, only_b = result["only_a"], result["only_b"]
    if args.marks:
        print(
            f"[compare] {result['questions']} questions: A is right on "
            f"{both + len(only_a)}, B on {both + len(only_b)}"
        )
    else:
        print(
            f"[compare] {result['questions']} questions with a passage: A covers "
            f"{both + len(only_a)}, B covers {both + len(only_b)}"
        )
    print(
        f"[compare] both {both}, only A {len(only_a)}, only B {len(only_b)}, neither {neither}; "
        f"a split this uneven or more has probability {result['p']:.3f} between equal runs"
    )
    if args.marks:
        # The leaderboard's accuracy is balanced across the domains, so a
        # question gained in the smaller one is worth more than one lost in
        # the larger.
        in_a, in_b = _by_domain(only_a, a), _by_domain(only_b, b)
        for domain in sorted(set(in_a) | set(in_b)):
            print(f"[compare] {domain}: only A {in_a.get(domain, 0)}, only B {in_b.get(domain, 0)}")
    for label, numbers, mine, other in (("A", only_a, a, b), ("B", only_b, b, a)):
        for number in numbers:
            row = mine[number]
            said = f"[compare] only {label}: {number} ({row['domain']}, {row['question_type']})"
            print(said if args.marks else f"{said}; the other: {_where(other[number])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
