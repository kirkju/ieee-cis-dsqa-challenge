"""What the model costs: the time to load it, and the time and the GPU memory
a prompt takes, by the length of the prompt.

    python -m tools.probe_engine --model-id <hf repo id> --corpus-dir data/corpus
    python -m tools.probe_engine --model-id <id> --corpus-dir data/corpus --record

The entry gives a question 10,000 tokens of context, its answer 1,024 tokens
and the two together 50 s. All three were set before the base model was ever
run; the first is the organizers' own figure, who found that prompts near the
model's window ran an 80 GB card out of memory. This measures what a prompt
costs, through the entry's own engine and over the entry's own contexts, so
that those numbers can be set from what the model does:

    load       the model loaded, as the first call of a run pays for it
    index      the corpus read and indexed on this machine
    by size    for contexts of growing size: the tokens of the prompt, the
               time the model takes to read it (it is asked for one token),
               the tokens a second it then writes at, and the peak of GPU
               memory. The sizes end at the first one the GPU has no memory
               for.

For the rate the model is asked to write on, a summary of the passages, so
that it writes the tokens it is allowed and the rate is taken over them. The
prompt is read twice, once for each of the two requests, and the rate takes
the second reading to have cost what the first did.

With DSQA_ENGINE naming a stand-in (tools/engines/ollama.py) the stand-in is
what gets measured. That checks this tool and says nothing of the base model.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tools import entry, ledger, run_local

# Tokens of context, as the entry counts them. The entry's own is among them.
SIZES = (0, 1_250, 2_500, 5_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000)
# Tokens the model is asked to write for the rate.
WRITTEN = 128
WRITE_ON = "Write a long and detailed summary of the passages above, document by document."
WRITE_ON_ALONE = (
    "Explain in detail, step by step, how a neutron source for testing materials works."
)


def _torch():
    """torch, when the engine has used CUDA through it; None otherwise."""
    torch = sys.modules.get("torch")
    try:
        return torch if torch is not None and torch.cuda.is_initialized() else None
    except Exception:
        return None


def prompt_tokens(engine, messages: list[dict]) -> int:
    """The tokens of a conversation as the model is given it, its template
    included; without the model's tokenizer, the tokens of what is said."""
    tokenizer = getattr(engine, "tokenizer", None)
    if hasattr(tokenizer, "apply_chat_template"):
        given = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return int(given["input_ids"].shape[-1])
    return sum(engine.count(message["content"]) for message in messages)


def measure(engine, messages: list[dict], written: int = WRITTEN, clock=time.perf_counter) -> dict:
    """One prompt put to the engine twice: for one token, and for `written`."""
    torch = _torch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()
    started = clock()
    engine.generate(messages, 1)
    first = clock() - started
    engine.release()
    started = clock()
    reply = engine.generate(messages, written)
    whole = clock() - started
    engine.release()
    count = engine.count(reply) if isinstance(reply, str) else 0
    row = {"first_token_s": round(first, 3), "written": count, "written_s": round(whole, 3)}
    if count > 1 and whole > first:
        row["tokens_per_s"] = round((count - 1) / (whole - first), 2)
    if torch is not None:
        row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return row


def probe(
    engine,
    index,
    question: str,
    sizes=SIZES,
    written: int = WRITTEN,
    say=print,
    clock=time.perf_counter,
    keep=None,
) -> list[dict]:
    """The cost of a prompt at each size of context, smallest first. A size
    the GPU has no memory for is the last one tried, and so is one the
    corpus cannot fill. Without an index only the bare request is measured.
    `keep` is given each row as soon as it is measured: a model that takes
    the process down with it leaves the sizes before that one."""
    rows = []
    for size in sizes if index is not None else (0,):
        placed, used = [], 0
        if size:
            placed, used, _ = index.context(question, engine.count, budget=size)
        if placed:
            context = entry.prompt.context_of(placed, index.documents)
            messages = entry.prompt.messages(WRITE_ON, context)
        else:
            messages = entry.prompt.messages(WRITE_ON_ALONE)
        row = {"context_budget": size, "chunks": len(placed), "context_tokens": int(used)}
        row["prompt_tokens"] = prompt_tokens(engine, messages)
        try:
            row.update(measure(engine, messages, written, clock))
        except Exception as error:
            engine.release()
            row["error"] = f"{type(error).__name__}: {error}"[:300]
            row["out_of_memory"] = entry.engine.out_of_memory(error)
        rows.append(row)
        if keep is not None:
            keep(row)
        say(f"[probe] {_line(row)}")
        if "error" in row:
            break
        if size and used < 0.5 * size:
            say("[probe] the corpus has no more to fill a larger context with")
            break
    return rows


def _keeping(rows: list, file, number):
    """What keeps a row of a question: among `rows`, and written at once."""

    def keep(row: dict) -> None:
        rows.append({"id": number, **row})
        file.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        file.flush()

    return keep


def _line(row: dict) -> str:
    said = f"context {row['context_budget']:>6,}  prompt {row['prompt_tokens']:>6,} tokens"
    if "error" in row:
        kind = "out of memory" if row["out_of_memory"] else row["error"]
        return f"{said}  {kind}"
    said += f"  read in {row['first_token_s']:.2f}s"
    if "tokens_per_s" in row:
        said += f"  {row['tokens_per_s']:.1f} tokens/s over {row['written']}"
    if "peak_gb" in row:
        said += f"  peak {row['peak_gb']:.1f} GB"
    return said


def summarize(rows: list[dict], entry_budget: int) -> dict:
    """What the ledger keeps of a probe: the largest prompt that went through,
    where memory ran out, and the cost at the entry's own context."""
    done = [row for row in rows if "error" not in row]
    failed = [row for row in rows if "error" in row]
    metrics = {"sizes": len(done)}
    if done:
        metrics["largest_prompt_tokens"] = max(row["prompt_tokens"] for row in done)
        rates = [row["tokens_per_s"] for row in done if "tokens_per_s" in row]
        if rates:
            metrics["tokens_per_s_median"] = round(statistics.median(rates), 2)
        peaks = [row["peak_gb"] for row in done if "peak_gb" in row]
        if peaks:
            metrics["peak_gb"] = max(peaks)
    if failed:
        metrics["failed_at_prompt_tokens"] = failed[0]["prompt_tokens"]
        metrics["out_of_memory"] = bool(failed[0]["out_of_memory"])
    for row in done:
        if row["context_budget"] == entry_budget:
            for name in ("prompt_tokens", "first_token_s", "tokens_per_s", "peak_gb"):
                if name in row:
                    metrics[f"entry_{name}"] = row[name]
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what a prompt costs the model.")
    parser.add_argument("--model-id", required=True, help="the model, as answer() is given it")
    parser.add_argument("--corpus-dir", type=Path, help="folder of PDFs the contexts are made of")
    parser.add_argument("--questions", type=Path, default=run_local.DEFAULT_QUESTIONS)
    parser.add_argument("--ids", help="the questions contexts are made for (default: the first)")
    parser.add_argument("--sizes", help="tokens of context to try, comma-separated")
    parser.add_argument("--written", type=int, default=WRITTEN, help="tokens written for the rate")
    parser.add_argument("--out", type=Path, default=run_local.DEFAULT_OUT)
    record = parser.add_argument_group("recording the probe in the ledger")
    record.add_argument("--record", action="store_true", help="append a row to the ledger")
    record.add_argument("--dataset", help="name of the question set (default: its file name)")
    record.add_argument("--corpus", help="name of the corpus (default: its folder name)")
    record.add_argument("--engine", default="transformers", help="what generates the text")
    record.add_argument("--machine", help="where this runs")
    record.add_argument("--note", default="", help="free text stored with the row")
    record.add_argument("--ledger", type=Path, default=ledger.LEDGER_PATH)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    questions = run_local.load_questions(args.questions)
    if args.ids:
        questions = run_local.select(questions, ids=args.ids.split(","))
    else:
        questions = questions[:1]
    if not questions:
        parser.error("no questions selected")
    sizes = tuple(int(size) for size in args.sizes.split(",")) if args.sizes else SIZES
    # As in the evaluation: nothing is fetched, and what is missing fails here.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    started = datetime.now(timezone.utc)
    git = ledger.git_state()
    run_id = ledger.new_run_id(started, git["commit"])
    run_dir = args.out.resolve() / run_id
    run_dir.mkdir(parents=True)
    clock = time.perf_counter
    began = clock()

    engine = entry.engine.load(args.model_id)
    load_s = clock() - began
    print(f"[probe] {args.model_id} loaded in {load_s:.1f}s")
    index, index_s = None, 0.0
    if args.corpus_dir is not None:
        before = clock()
        index = entry.retrieval.build(args.corpus_dir, entry.retrieval.load_embedder())
        index_s = clock() - before
        print(f"[probe] corpus indexed in {index_s:.1f}s")

    rows = []
    with open(run_dir / "probe.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for item in questions:
            keep = _keeping(rows, f, item.get("id"))
            probe(engine, index, item["question"], sizes, args.written, keep=keep)

    metrics = summarize(rows, entry.config.CONTEXT_TOKENS)
    timings = {"load_s": round(load_s, 3), "index_s": round(index_s, 3)}
    timings["total_s"] = round(clock() - began, 3)
    torch = _torch()
    machine = args.machine or (torch.cuda.get_device_name(0) if torch is not None else None)
    summary = {"run_id": run_id, "git": git, "model_id": args.model_id, "machine": machine}
    summary.update(metrics=metrics, timings=timings)
    with open(run_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[probe] {json.dumps(metrics)}")
    print(f"[probe] rows in {run_dir.as_posix()}")

    if args.record:
        corpus = args.corpus or (args.corpus_dir.resolve().name if args.corpus_dir else "none")
        ledger.record(
            "probe",
            metrics,
            dataset=args.dataset or args.questions.stem,
            corpus=corpus,
            engine=args.engine,
            model=args.model_id,
            config={
                "ids": args.ids,
                "sizes": list(sizes),
                "written": args.written,
                "entry_context_tokens": entry.config.CONTEXT_TOKENS,
                "chunks": len(index.chunks) if index is not None else 0,
            },
            timings=timings,
            notes=args.note,
            machine=machine,
            run_id=run_id,
            state=git,
            path=args.ledger,
        )
        print(f"[probe] recorded in {args.ledger.name} as {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
