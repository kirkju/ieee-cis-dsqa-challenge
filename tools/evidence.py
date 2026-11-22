"""Reference evidence: where in the corpus each question is answered.

dev/questions/sample_evidence.json says, for every sample question, which
document it was written from and which passages of it hold the answer.
Retrieval is measured against it: a question is covered when one of its
passages reaches the context handed to the model. The question sets of our
own are annotated the same way, each in a file next to its questions.

A passage is a page of a document and a quote copied from the text of that
page. The quote is checked against the PDF, so a passage cannot be made up or
drift from the document:

    python -m tools.evidence                      # check the sample's file and summarize it
    python -m tools.evidence --questions Q.json --evidence E.json   # another question set
    python -m tools.evidence --find DOC_ID TEXT   # pages of a document where TEXT occurs

Fields of an item:

    id        the question's id in its question set
    status    traced       the source document is in the corpus and the passages are located
              unavailable  the source is identified but its PDF could not be obtained
              untraced     the source document was not found
    doc_id    the document the question was most likely written from (absent when untraced)
    evidence  the passages: doc_id, page (counted from 1), kind and quote, and
              for passages that answer only together, the same group
    note      what a reader of the annotation should know

The status is about where the question came from; the evidence is about
where its answer can be found. A passage is usually in the source document.
It can also be in another document that states the same facts: a result
reported twice, a sentence carried over from one paper to the next. Either
one puts the answer in front of the model, so both count, and a question
whose source we do not have can still have passages.

The kind of a passage says what a system has to read to use it: "text", a
"table", or a "figure" (the quote is then the caption or the sentence that
points at it, since the values themselves are in the drawing).

A passage answers its question by itself unless it carries a group. Passages
with the same group are needed together: a table for the number and a
sentence for the rule to apply to it, the two halves of a paragraph that a
page break cuts. The sample's annotations have no groups, since nothing says
which of a sample's passages its answer needs.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

from tools import corpus, ledger

REPO_ROOT = ledger.REPO_ROOT
EVIDENCE = REPO_ROOT / "dev" / "questions" / "sample_evidence.json"
SAMPLE = REPO_ROOT / "sample" / "sample_qa.json"

STATUSES = {"traced", "unavailable", "untraced"}
KINDS = {"text", "table", "figure"}

# Invisible, and left behind by some PDF producers where a word may break.
SOFT_HYPHEN = "\N{SOFT HYPHEN}"


def clean(text: str) -> str:
    """Text with the marks of its layout undone: ligatures and compatibility
    characters unfolded, soft hyphens dropped, words rejoined across line
    breaks, whitespace collapsed. This is the form quotes are written in."""
    text = unicodedata.normalize("NFKC", text).replace(SOFT_HYPHEN, "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """The form two texts are compared in: clean(), with case ignored."""
    return clean(text).casefold()


def contains(text: str, quote: str) -> bool:
    return normalize(quote) in normalize(text)


def page_texts(path: Path) -> list[str]:
    """The text of each page of a PDF, in order."""
    with pymupdf.open(path) as document:
        return [page.get_text() for page in document]


def load(path: Path = EVIDENCE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def problems(
    items: list[dict], questions: list[dict], manifest: list[dict], listed: bool = True
) -> list[str]:
    """What is wrong with the annotations themselves, before any PDF is opened.

    listed says whether the manifest has to name each question under the
    documents that answer it. It does for the organizers' sample; a set of
    our own questions only has to name documents that are in the manifest.
    """
    found = []
    asked = [question["id"] for question in questions]
    annotated = [item.get("id") for item in items]
    for missing in sorted(set(asked) - set(annotated)):
        found.append(f"question {missing} has no annotation")
    for extra in sorted(set(annotated) - set(asked)):
        found.append(f"question {extra} is annotated but is not among the questions")
    if len(annotated) != len(set(annotated)):
        found.append("a question is annotated more than once")

    documents = {row["doc_id"]: row for row in manifest}

    def not_listed(doc_id, question) -> str | None:
        """Why the manifest does not back doc_id as holding question's answer."""
        row = documents.get(doc_id)
        if row is None:
            return f"{doc_id!r} is not in the manifest"
        if listed and (row["role"] != "source" or question not in corpus.question_ids(row)):
            return f"the manifest does not list it under {doc_id}"
        return None

    for item in items:
        where = f"question {item.get('id')}"
        status, passages = item.get("status"), item.get("evidence", [])
        source = item.get("doc_id")
        if status not in STATUSES:
            found.append(f"{where}: status must be one of {', '.join(sorted(STATUSES))}")
            continue
        if status == "untraced":
            if source:
                found.append(f"{where}: an untraced question cannot name its source document")
        else:
            complaint = not_listed(source, item["id"])
            if complaint:
                found.append(f"{where}: {complaint}")
        in_source = [passage for passage in passages if source and passage.get("doc_id") == source]
        if status == "traced" and not in_source:
            found.append(f"{where}: a traced question needs a passage in its source document")
        if status == "unavailable" and in_source:
            found.append(f"{where}: passages cannot be located in a document we do not have")
        for passage in passages:
            page, quote = passage.get("page"), passage.get("quote", "")
            if passage.get("doc_id") != source or status == "untraced":
                complaint = not_listed(passage.get("doc_id"), item["id"])
                if complaint:
                    found.append(f"{where}: {complaint}")
            if not isinstance(page, int) or page < 1:
                found.append(f"{where}: a page is a number counted from 1")
            if len(normalize(quote)) < 20:
                found.append(f"{where}: a quote of under 20 characters pins nothing down")
            if passage.get("kind") not in KINDS:
                found.append(f"{where}: kind must be one of {', '.join(sorted(KINDS))}")
        groups = [passage["group"] for passage in passages if "group" in passage]
        if not all(isinstance(group, str) and group for group in groups):
            found.append(f"{where}: a group is a name shared by passages needed together")
        elif any(groups.count(group) == 1 for group in groups):
            found.append(f"{where}: a group of one passage is not a group")
    return found


def misplaced(items: list[dict], manifest: list[dict], dest: Path = corpus.CORPUS_DIR):
    """Check every quote against the page it claims to come from.

    Returns the passages that are not where they say, and the documents that
    could not be checked because their PDF is not on this machine.
    """
    wrong, absent = [], set()
    texts: dict[str, list[str]] = {}
    documents = {row["doc_id"]: row for row in manifest}
    for item in items:
        for passage in item.get("evidence", []):
            doc_id, page, quote = passage.get("doc_id"), passage["page"], passage["quote"]
            if doc_id not in documents:
                continue
            path = corpus.pdf_path(documents[doc_id], dest)
            if not path.exists():
                absent.add(doc_id)
                continue
            pages = texts.setdefault(doc_id, page_texts(path))
            if not (1 <= page <= len(pages) and contains(pages[page - 1], quote)):
                elsewhere = [n for n, text in enumerate(pages, 1) if contains(text, quote)]
                hint = f"; it is on page {elsewhere[0]}" if elsewhere else ""
                wrong.append(
                    f"question {item['id']}: not on page {page} of {doc_id}{hint}: {quote[:60]!r}"
                )
    return wrong, sorted(absent)


def find(path: Path, text: str, width: int = 160) -> list[tuple[int, str]]:
    """Pages of a PDF where text occurs, each with the text around it, in the
    form a quote can be copied from."""
    wanted = re.compile(re.escape(clean(text)), re.IGNORECASE)
    hits = []
    for number, page in enumerate(page_texts(path), 1):
        flat = clean(page)
        for match in wanted.finditer(flat):
            hits.append((number, flat[max(0, match.start() - width) : match.end() + width]))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the reference evidence of a question set.")
    parser.add_argument("--find", nargs=2, metavar=("DOC_ID", "TEXT"), help="search a document")
    parser.add_argument("--questions", type=Path, default=SAMPLE, help="the questions annotated")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--manifest", type=Path, default=corpus.MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=corpus.CORPUS_DIR)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    manifest = corpus.load_manifest(args.manifest)
    if args.find:
        doc_id, text = args.find
        rows = [row for row in manifest if row["doc_id"] == doc_id]
        if not rows or not corpus.pdf_path(rows[0], args.corpus_dir).exists():
            parser.error(f"{doc_id} is not in the manifest or its PDF is not on this machine")
        hits = find(corpus.pdf_path(rows[0], args.corpus_dir), text)
        for page, context in hits:
            print(f"page {page}: ...{context}...")
        print(f"[evidence] {len(hits)} occurrence(s) in {doc_id}")
        return 0

    items = load(args.evidence)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    # Only the sample's questions are listed in the manifest, by their ids.
    listed = args.questions.resolve() == SAMPLE.resolve()
    found = problems(items, questions, manifest, listed)
    wrong, absent = misplaced(items, manifest, args.corpus_dir) if not found else ([], [])
    for line in found + wrong:
        print(f"[evidence] {line}")

    domains = {question["id"]: question["domain"] for question in questions}
    for domain in sorted(set(domains.values())):
        mine = [item for item in items if domains.get(item["id"]) == domain]
        counts = {status: sum(item["status"] == status for item in mine) for status in STATUSES}
        answerable = sum(bool(item.get("evidence")) for item in mine)
        print(
            f"[evidence] {domain}: {counts['traced']} traced, {counts['unavailable']} "
            f"unavailable, {counts['untraced']} untraced (of {len(mine)}); "
            f"{answerable} with a passage in the corpus"
        )
    passages = [(item, passage) for item in items for passage in item.get("evidence", [])]
    kinds = [passage["kind"] for _, passage in passages]
    summary = ", ".join(f"{kinds.count(kind)} {kind}" for kind in sorted(KINDS))
    elsewhere = sum(passage["doc_id"] != item.get("doc_id") for item, passage in passages)
    print(f"[evidence] {len(kinds)} passages: {summary}; {elsewhere} outside the source document")
    if absent:
        print(f"[evidence] not checked, PDF not on this machine: {', '.join(absent)}")
    return 1 if found or wrong else 0


if __name__ == "__main__":
    sys.exit(main())
