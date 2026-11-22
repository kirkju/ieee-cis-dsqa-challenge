"""How well the entry tells the domain of a question (dsqa/route.py), on
question sets whose domain is known.

    python -m tools.eval_route --model-id <hf repo id>
    python -m tools.eval_route --model-id <id> --questions dev/questions/tuning_qa.json \
        --questions dev/questions/control_qa.json

Each set is a JSON list of questions with their "domain" ("DONES" or "AI
Papers"). The corpus is read and indexed as tools/eval_retrieval.py does it
for --system 03_dsqa, with the parsed text it keeps, and the ranking is the
entry's own, under the adjustments DSQA_VARIANT names. No model is loaded
but the tokenizer and the embedder.

Three rules are reported side by side: the one route.py decides by (the
question's words and the documents ranked first, each told by its own
words), the documents alone, and domain.of() with the opening of the best
document, an undecided question counted as AI as route.py would send it.
What counts is how often each is right, and on which side it errs: a
question on AI papers sent the IFMIF-DONES way loses the retrieval that
answers it; one on IFMIF-DONES kept on the other side is answered as before.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from tools import corpus, eval_retrieval, run_local
from tools.entry_system import Entry

DEFAULT_SETS = (
    Path("sample/sample_qa.json"),
    Path("dev/questions/tuning_qa.json"),
    Path("dev/questions/control_qa.json"),
)


def labelled(question: dict, dones: str, ai: str) -> str:
    """A question's known domain, in the names route.py decides with."""
    return dones if question.get("domain") == "DONES" else ai


def rules(route, domain, index):
    """The rules compared, by name: each tells a question's domain."""

    def opening(question):
        order = index.rank(question)
        first = int(index.documents[int(order[0])])
        number = int(np.flatnonzero(np.asarray(index.documents) == first)[0])
        return domain.of(question, index.chunks[number]) or domain.AI

    def documents_alone(question):
        saved = route.QUESTION_WEIGHT
        route.QUESTION_WEIGHT = 0.0
        try:
            return route.decided(question, index)
        finally:
            route.QUESTION_WEIGHT = saved

    return {
        "route": lambda question: route.decided(question, index),
        "documents alone": documents_alone,
        "domain.of with the opening": opening,
    }


def measure(questions: list[dict], decide, dones: str, ai: str) -> dict:
    """How often a rule is right on some questions, and its errors by side."""
    errors = {"ai_sent_dones": [], "dones_kept_ai": []}
    right = 0
    for question in questions:
        known = labelled(question, dones, ai)
        told = decide(question["question"])
        if told == known:
            right += 1
        elif told == dones:
            errors["ai_sent_dones"].append(question.get("id"))
        else:
            errors["dones_kept_ai"].append(question.get("id"))
    return {"questions": len(questions), "right": right, **errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure how the entry tells a question's domain.")
    parser.add_argument("--model-id", required=True, help="the base model; only its tokenizer")
    parser.add_argument("--questions", type=Path, action="append", help="a set; can be repeated")
    parser.add_argument("--corpus-dir", type=Path, default=corpus.CORPUS_DIR)
    parser.add_argument("--cache", type=Path, default=eval_retrieval.CACHE_DIR)
    parser.add_argument("--out", type=Path, help="a JSON file for the results")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sets = args.questions or [eval_retrieval.REPO_ROOT / path for path in DEFAULT_SETS]
    pdfs = sorted(args.corpus_dir.glob("*.pdf"))
    if not pdfs:
        parser.error(f"no PDFs in {args.corpus_dir}")

    with run_local._environment(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"):
        system = Entry.load(args.model_id)
        documents = eval_retrieval.read_corpus(system, pdfs, args.cache)
        chunks = []
        for document in documents:
            if document.text is not None:
                chunks.extend(system.chunk(document.text))
        system.index(chunks)
    # The entry's package is importable once tools/entry_system.py is.
    domain = importlib.import_module("dsqa.domain")
    route = importlib.import_module("dsqa.route")
    index = system.built
    results = {}
    for path in sets:
        questions = run_local.load_questions(path)
        for name, decide in rules(route, domain, index).items():
            got = measure(questions, decide, domain.DONES, domain.AI)
            results.setdefault(path.stem, {})[name] = got
            print(
                f"[route] {path.stem:12s} {name:28s} right {got['right']}/{got['questions']}"
                f" ({got['right'] / got['questions']:.1%}); AI sent the DONES way "
                f"{len(got['ai_sent_dones'])} {got['ai_sent_dones']}; DONES kept the AI way "
                f"{len(got['dones_kept_ai'])} {got['dones_kept_ai']}"
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
