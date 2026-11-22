"""The development corpus: which documents it holds and where they came from.

dev/corpus/manifest.csv lists every document: the source papers of the sample
questions and the distractors around them. The PDFs are copyrighted and are
not in the repository. This tool downloads the ones that are openly available
into data/corpus/ and checks every file against the hash recorded in the
manifest, so the corpus is the same on every machine.

    python -m tools.corpus                 # download what is missing, verify the rest
    python -m tools.corpus --record        # also write the hash and page count of new files
    python -m tools.corpus --only doc-a,doc-b
    python -m tools.corpus --no-download   # verify what is there, fetch nothing

A document marked "manual" cannot be downloaded without a subscription. It is
placed in data/corpus/ by hand, as <doc_id>.pdf, and verified like the others.

Columns of the manifest:

    doc_id     short unique name; the file is data/corpus/<doc_id>.pdf
    domain     DONES or AI Papers
    role       source (it holds the answer to a sample question: the document
               the question was written from, or one that states the same
               facts) or distractor
    questions  ids of the sample questions it answers, separated by ";"
    title, year
    reference  DOI, arXiv id or report number, for citing it
    url        where the PDF is downloaded from
    access     open or manual
    sha256     of the PDF as downloaded
    pages      number of pages
    note       anything worth knowing, e.g. "preprint, differs from the published version"
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pymupdf

from tools import ledger

REPO_ROOT = ledger.REPO_ROOT
MANIFEST = REPO_ROOT / "dev" / "corpus" / "manifest.csv"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

FIELDS = [
    "doc_id",
    "domain",
    "role",
    "questions",
    "title",
    "year",
    "reference",
    "url",
    "access",
    "sha256",
    "pages",
    "note",
]
DOMAINS = {"DONES", "AI Papers"}
ROLES = {"source", "distractor"}
ACCESS = {"open", "manual"}

# Publisher sites turn away clients that do not say who they are.
USER_AGENT = "dsqa-dev-corpus/1.0 (research; one polite request at a time)"
# Seconds between two requests to the same site. arXiv asks for at least 3.
PAUSE = 5.0

# What can become of a document when the corpus folder is synchronized.
OUTCOMES = {
    "verified": "the file matches the hash in the manifest",
    "recorded": "a new file whose hash was written into the manifest",
    "unrecorded": "a file the manifest has no hash for yet",
    "mismatch": "the file is not the one the manifest describes",
    "missing": "not there: a manual document not placed yet, or nothing was downloaded",
    "failed": "the download did not produce a PDF",
}


class NotAPdf(Exception):
    """What came back is not a PDF: usually a landing page or a login form."""


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path}: expected the columns {', '.join(FIELDS)}")
        return list(reader)


def save_manifest(rows: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def question_ids(row: dict) -> list[int]:
    """Ids of the sample questions a document answers."""
    return [int(part) for part in row["questions"].split(";") if part.strip()]


def manifest_problems(rows: list[dict]) -> list[str]:
    """Everything wrong with a manifest, in words; empty when it is sound."""
    problems = []
    seen = set()
    for number, row in enumerate(rows, 2):  # line 1 is the header
        where = f"line {number} ({row['doc_id'] or 'no doc_id'})"
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", row["doc_id"]):
            problems.append(f"{where}: doc_id must be lowercase letters, digits and hyphens")
        if row["doc_id"] in seen:
            problems.append(f"{where}: doc_id is used more than once")
        seen.add(row["doc_id"])
        for field, allowed in (("domain", DOMAINS), ("role", ROLES), ("access", ACCESS)):
            if row[field] not in allowed:
                problems.append(f"{where}: {field} must be one of {', '.join(sorted(allowed))}")
        if not re.fullmatch(r"(\d+(;\d+)*)?", row["questions"]):
            problems.append(f"{where}: questions must be ids separated by ';'")
        elif row["questions"] and row["role"] != "source":
            problems.append(f"{where}: only a source document answers questions")
        if not row["title"].strip():
            problems.append(f"{where}: title is empty")
        if row["access"] == "open" and not row["url"].startswith(("http://", "https://")):
            problems.append(f"{where}: an open document needs the url it is downloaded from")
        if row["sha256"] and not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            problems.append(f"{where}: sha256 must be 64 lowercase hexadecimal digits")
        if row["pages"] and not row["pages"].isdigit():
            problems.append(f"{where}: pages must be a number")
    return problems


def pdf_path(row: dict, dest: Path = CORPUS_DIR) -> Path:
    return dest / f"{row['doc_id']}.pdf"


def describe_pdf(path: Path) -> tuple[str, int]:
    """The file's SHA-256 and its number of pages. Raises NotAPdf if it cannot
    be opened as a PDF."""
    data = path.read_bytes()
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            pages = document.page_count
    except Exception as e:
        raise NotAPdf(f"{path.name} does not open as a PDF ({e})") from e
    if pages == 0:
        raise NotAPdf(f"{path.name} has no pages")
    return hashlib.sha256(data).hexdigest(), pages


def _with_urllib(url: str, partial: Path, timeout: float) -> str:
    """Fetch url into partial with Python's own client; returns the content type."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as f:
        while chunk := response.read(1 << 16):
            f.write(chunk)
        return response.headers.get("Content-Type", "an unknown type")


def _with_curl(url: str, partial: Path, timeout: float) -> str:
    """Fetch url into partial with the curl command; returns the content type."""
    command = ["curl", "--silent", "--show-error", "--location", "--fail"]
    command += ["--max-time", str(int(timeout)), "--user-agent", USER_AGENT]
    command += ["--output", str(partial), "--write-out", "%{content_type}", url]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise OSError(f"curl: {done.stderr.strip() or 'exit ' + str(done.returncode)}")
    return done.stdout.strip() or "an unknown type"


def _clients() -> list:
    """The ways to fetch a file, in the order they are tried.

    Some publisher sites answer Python's HTTP client with a web page and a
    plain curl with the PDF, whatever the client says it is. curl is the
    second attempt where it is installed.
    """
    return [_with_urllib] + ([_with_curl] if shutil.which("curl") else [])


def download(url: str, target: Path, timeout: float = 120.0, clients: list | None = None) -> None:
    """Fetch url into target. Nothing is left at target unless a PDF arrived."""
    partial = target.with_suffix(".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    failure: Exception = NotAPdf(f"{url}: no way to fetch it")
    try:
        for client in _clients() if clients is None else clients:
            try:
                content_type = client(url, partial, timeout)
            except (urllib.error.URLError, OSError) as e:
                failure = e
                continue
            with open(partial, "rb") as f:
                head = f.read(1024)
            # The PDF header may be preceded by a few bytes of junk.
            if b"%PDF-" in head:
                os.replace(partial, target)
                return
            failure = NotAPdf(f"{url} returned {content_type}, not a PDF")
        raise failure
    finally:
        partial.unlink(missing_ok=True)


def sync(
    rows: list[dict],
    dest: Path = CORPUS_DIR,
    *,
    only: set[str] | None = None,
    record: bool = False,
    pause: float = PAUSE,
    fetch=download,
    sleep=time.sleep,
    say=print,
) -> dict[str, list[str]]:
    """Bring dest in line with the manifest and report, per outcome, which
    documents ended up there (see OUTCOMES). With fetch=None nothing is
    downloaded: the files that are there are verified and the rest is missing."""
    outcome: dict[str, list[str]] = {name: [] for name in OUTCOMES}
    last_request: dict[str, float] = {}
    for row in rows:
        doc_id, path = row["doc_id"], pdf_path(row, dest)
        if only is not None and doc_id not in only:
            continue
        if not path.exists():
            if row["access"] != "open" or fetch is None:
                outcome["missing"].append(doc_id)
                if row["access"] != "open":
                    say(f"[corpus] {doc_id}: to be placed by hand as {path.name}")
                continue
            # One request at a time, and not too often to the same site.
            host = urllib.parse.urlsplit(row["url"]).netloc
            if host in last_request:
                sleep(max(0.0, pause - (time.monotonic() - last_request[host])))
            try:
                fetch(row["url"], path)
            except (NotAPdf, urllib.error.URLError, OSError) as e:
                outcome["failed"].append(doc_id)
                say(f"[corpus] {doc_id}: download failed: {e}")
                continue
            finally:
                last_request[host] = time.monotonic()
            say(f"[corpus] {doc_id}: downloaded")

        try:
            sha256, pages = describe_pdf(path)
        except NotAPdf as e:
            outcome["failed"].append(doc_id)
            say(f"[corpus] {doc_id}: {e}")
            continue
        if row["sha256"]:
            matches = row["sha256"] == sha256
            outcome["verified" if matches else "mismatch"].append(doc_id)
            if not matches:
                say(f"[corpus] {doc_id}: {path.name} is not the file the manifest describes")
        elif record:
            row["sha256"], row["pages"] = sha256, str(pages)
            outcome["recorded"].append(doc_id)
        else:
            outcome["unrecorded"].append(doc_id)
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the development corpus.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dest", type=Path, default=CORPUS_DIR, help="folder for the PDFs")
    parser.add_argument("--only", help="only these doc_ids, comma-separated")
    parser.add_argument("--record", action="store_true", help="write hashes of new files")
    parser.add_argument("--pause", type=float, default=PAUSE, help="seconds between requests")
    parser.add_argument("--no-download", action="store_true", help="only verify what is there")
    args = parser.parse_args(argv)

    rows = load_manifest(args.manifest)
    problems = manifest_problems(rows)
    if problems:
        print("\n".join(f"[corpus] {problem}" for problem in problems))
        return 2
    only = set(args.only.split(",")) if args.only else None
    unknown = (only or set()) - {row["doc_id"] for row in rows}
    if unknown:
        parser.error(f"not in the manifest: {', '.join(sorted(unknown))}")

    fetch = None if args.no_download else download
    outcome = sync(rows, args.dest, only=only, record=args.record, pause=args.pause, fetch=fetch)
    if outcome["recorded"]:
        save_manifest(rows, args.manifest)
    print("[corpus] " + ", ".join(f"{len(ids)} {name}" for name, ids in outcome.items() if ids))
    if outcome["unrecorded"]:
        print("[corpus] run again with --record to write the hashes of the unrecorded files")
    return 1 if outcome["mismatch"] or outcome["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
