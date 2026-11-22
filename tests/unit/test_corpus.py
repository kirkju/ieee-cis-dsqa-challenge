"""tools/corpus.py: the corpus on disk is the one the manifest describes.

Downloads go to a web server started on this machine for the test, so nothing
here needs the internet.
"""

import functools
import hashlib
import http.server
import threading

import pymupdf
import pytest

from tools import corpus


def make_pdf(pages: int = 2) -> bytes:
    with pymupdf.open() as document:
        for number in range(pages):
            document.new_page().insert_text((72, 72), f"page {number + 1}")
        return document.tobytes()


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def site(tmp_path):
    """A web server on this machine with one PDF and one page that is not."""
    root = tmp_path / "site"
    root.mkdir()
    (root / "paper.pdf").write_bytes(make_pdf(pages=2))
    (root / "other.pdf").write_bytes(make_pdf(pages=3))
    (root / "login.pdf").write_text("<html>Please sign in</html>", encoding="utf-8")
    handler = functools.partial(QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def row(doc_id="paper", **fields):
    values = dict.fromkeys(corpus.FIELDS, "")
    values.update(doc_id=doc_id, domain="DONES", role="distractor", title="A paper", access="open")
    values.update(fields)
    return values


def quiet(line):
    pass


def no_sleep(seconds):
    pass


def test_a_missing_open_document_is_downloaded_and_recorded(site, tmp_path):
    rows = [row(url=f"{site}/paper.pdf")]
    outcome = corpus.sync(rows, tmp_path / "corpus", record=True, say=quiet, sleep=no_sleep)

    stored = tmp_path / "corpus" / "paper.pdf"
    assert outcome["recorded"] == ["paper"]
    assert rows[0]["sha256"] == hashlib.sha256(stored.read_bytes()).hexdigest()
    assert rows[0]["pages"] == "2"


def test_without_record_a_new_file_is_only_reported(site, tmp_path):
    rows = [row(url=f"{site}/paper.pdf")]
    outcome = corpus.sync(rows, tmp_path / "corpus", say=quiet, sleep=no_sleep)
    assert outcome["unrecorded"] == ["paper"]
    assert rows[0]["sha256"] == ""


def test_a_file_already_there_is_verified_without_a_request(tmp_path):
    dest = tmp_path / "corpus"
    dest.mkdir()
    data = make_pdf()
    (dest / "paper.pdf").write_bytes(data)
    rows = [row(url="http://203.0.113.5/never", sha256=hashlib.sha256(data).hexdigest())]

    def must_not_fetch(url, target):
        raise AssertionError("a file that is already there was requested again")

    outcome = corpus.sync(rows, dest, fetch=must_not_fetch, say=quiet, sleep=no_sleep)
    assert outcome["verified"] == ["paper"]


def test_a_file_that_differs_from_the_manifest_is_a_mismatch_and_is_kept(tmp_path):
    dest = tmp_path / "corpus"
    dest.mkdir()
    (dest / "paper.pdf").write_bytes(make_pdf())
    rows = [row(url="http://203.0.113.5/never", sha256="0" * 64)]

    outcome = corpus.sync(rows, dest, record=True, say=quiet, sleep=no_sleep)

    assert outcome["mismatch"] == ["paper"]
    assert rows[0]["sha256"] == "0" * 64  # the manifest is not bent to fit the file
    assert (dest / "paper.pdf").exists()


def test_a_page_that_is_not_a_pdf_is_refused_and_nothing_is_left(site, tmp_path):
    dest = tmp_path / "corpus"
    outcome = corpus.sync([row(url=f"{site}/login.pdf")], dest, say=quiet, sleep=no_sleep)
    assert outcome["failed"] == ["paper"]
    assert list(dest.iterdir()) == []


def test_a_second_client_is_tried_when_the_first_gets_a_web_page(tmp_path):
    asked = []

    def turned_away(url, partial, timeout):
        asked.append("first")
        partial.write_text("<html>Are you a robot?</html>", encoding="utf-8")
        return "text/html"

    def served(url, partial, timeout):
        asked.append("second")
        partial.write_bytes(make_pdf())
        return "application/pdf"

    target = tmp_path / "corpus" / "paper.pdf"
    corpus.download("https://publisher.example/paper.pdf", target, clients=[turned_away, served])
    assert asked == ["first", "second"]
    assert target.read_bytes().startswith(b"%PDF-")

    # The second is left alone when the first already brought the PDF.
    asked.clear()
    corpus.download("https://publisher.example/paper.pdf", target, clients=[served, turned_away])
    assert asked == ["second"]


def test_when_every_client_fails_the_last_reason_is_given_and_nothing_is_left(tmp_path):
    def unreachable(url, partial, timeout):
        raise OSError("connection refused")

    def turned_away(url, partial, timeout):
        partial.write_text("<html>Are you a robot?</html>", encoding="utf-8")
        return "text/html"

    target = tmp_path / "corpus" / "paper.pdf"
    with pytest.raises(corpus.NotAPdf, match="returned text/html"):
        corpus.download("https://x.example/paper.pdf", target, clients=[unreachable, turned_away])
    assert list(target.parent.iterdir()) == []


def test_a_broken_link_is_a_failure_and_the_rest_goes_on(site, tmp_path):
    rows = [row("gone", url=f"{site}/nothing-here.pdf"), row("paper", url=f"{site}/paper.pdf")]
    outcome = corpus.sync(rows, tmp_path / "corpus", say=quiet, sleep=no_sleep)
    assert outcome["failed"] == ["gone"]
    assert outcome["unrecorded"] == ["paper"]


def test_a_file_that_is_not_a_pdf_inside_is_a_failure(tmp_path):
    dest = tmp_path / "corpus"
    dest.mkdir()
    (dest / "paper.pdf").write_bytes(b"%PDF-1.7 but nothing else")
    outcome = corpus.sync([row(access="manual")], dest, say=quiet, sleep=no_sleep)
    assert outcome["failed"] == ["paper"]


def test_a_manual_document_is_never_requested(tmp_path):
    def must_not_fetch(url, target):
        raise AssertionError("a document behind a subscription was requested")

    rows = [row(access="manual", url="https://publisher.example/paywalled.pdf")]
    outcome = corpus.sync(rows, tmp_path / "corpus", fetch=must_not_fetch, say=quiet)
    assert outcome["missing"] == ["paper"]


def test_a_manual_document_placed_by_hand_is_verified_like_the_others(tmp_path):
    dest = tmp_path / "corpus"
    dest.mkdir()
    data = make_pdf()
    (dest / "paper.pdf").write_bytes(data)
    rows = [row(access="manual", sha256=hashlib.sha256(data).hexdigest())]
    assert corpus.sync(rows, dest, say=quiet)["verified"] == ["paper"]


def test_without_downloading_what_is_there_is_verified_and_the_rest_is_missing(tmp_path):
    dest = tmp_path / "corpus"
    dest.mkdir()
    data = make_pdf()
    (dest / "here.pdf").write_bytes(data)
    rows = [
        row("here", url="https://x.example/a.pdf", sha256=hashlib.sha256(data).hexdigest()),
        row("not-yet", url="https://x.example/b.pdf"),
    ]
    outcome = corpus.sync(rows, dest, fetch=None, say=quiet)
    assert outcome["verified"] == ["here"]
    assert outcome["missing"] == ["not-yet"]


def test_only_the_documents_asked_for_are_touched(site, tmp_path):
    rows = [row("first", url=f"{site}/paper.pdf"), row("second", url=f"{site}/other.pdf")]
    outcome = corpus.sync(rows, tmp_path / "corpus", only={"second"}, say=quiet, sleep=no_sleep)
    assert outcome["unrecorded"] == ["second"]
    assert not (tmp_path / "corpus" / "first.pdf").exists()


def test_requests_to_the_same_site_are_spaced_out(site, tmp_path):
    waits = []
    rows = [row("first", url=f"{site}/paper.pdf"), row("second", url=f"{site}/other.pdf")]
    corpus.sync(rows, tmp_path / "corpus", pause=5.0, say=quiet, sleep=waits.append)
    # Nothing to wait for before the first request; close to the full pause
    # before the second, which follows it immediately.
    assert len(waits) == 1
    assert 4.0 < waits[0] <= 5.0


def test_different_sites_are_not_made_to_wait_for_each_other(tmp_path):
    waits = []

    def fetch(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(make_pdf())

    rows = [row("first", url="https://one.example/a.pdf"), row("second", url="https://two.example/b.pdf")]
    corpus.sync(rows, tmp_path / "corpus", fetch=fetch, say=quiet, sleep=waits.append)
    assert waits == []


def test_the_manifest_survives_a_round_trip(tmp_path):
    rows = [
        row("first", title='A title, with a comma and "quotes"', questions="", year="2021"),
        row("second", role="source", questions="3;4", note="preprint"),
    ]
    path = tmp_path / "dev" / "manifest.csv"
    corpus.save_manifest(rows, path)

    assert corpus.load_manifest(path) == rows
    assert b"\r" not in path.read_bytes()
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(corpus.FIELDS)


def test_a_manifest_with_other_columns_is_refused(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("doc_id,title\npaper,A paper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected the columns"):
        corpus.load_manifest(path)


def test_a_sound_manifest_has_no_problems():
    rows = [
        row("ai-simclr", domain="AI Papers", role="source", questions="22", url="https://x.example/a.pdf"),
        row("dones-loop-2", url="https://x.example/b.pdf", sha256="a" * 64, pages="12"),
        row("paywalled", access="manual"),
    ]
    assert corpus.manifest_problems(rows) == []


@pytest.mark.parametrize(
    ("fields", "complaint"),
    [
        ({"doc_id": "Has Spaces"}, "lowercase letters, digits and hyphens"),
        ({"domain": "Physics"}, "domain must be one of"),
        ({"role": "noise"}, "role must be one of"),
        ({"access": "free"}, "access must be one of"),
        ({"questions": "3,4", "role": "source"}, "separated by ';'"),
        ({"questions": "3"}, "only a source document answers questions"),
        ({"title": "  "}, "title is empty"),
        ({"url": ""}, "needs the url"),
        ({"sha256": "ABC"}, "64 lowercase hexadecimal digits"),
        ({"pages": "twelve"}, "pages must be a number"),
    ],
)
def test_each_kind_of_mistake_in_a_manifest_is_named(fields, complaint):
    problems = corpus.manifest_problems([row(url="https://x.example/a.pdf") | fields])
    assert len(problems) == 1
    assert complaint in problems[0]
    assert problems[0].startswith("line 2")


def test_a_repeated_doc_id_is_a_problem():
    rows = [row(url="https://x.example/a.pdf"), row(url="https://x.example/b.pdf")]
    assert any("more than once" in problem for problem in corpus.manifest_problems(rows))


def test_question_ids_are_read_from_a_row():
    assert corpus.question_ids(row(questions="3;4")) == [3, 4]
    assert corpus.question_ids(row()) == []


def test_the_command_writes_recorded_hashes_into_the_manifest(site, tmp_path, capsys):
    manifest = tmp_path / "manifest.csv"
    corpus.save_manifest([row(url=f"{site}/paper.pdf")], manifest)
    arguments = ["--manifest", str(manifest), "--dest", str(tmp_path / "corpus"), "--pause", "0"]

    assert corpus.main(arguments) == 0
    assert "--record" in capsys.readouterr().out
    assert corpus.load_manifest(manifest)[0]["sha256"] == ""

    assert corpus.main([*arguments, "--record"]) == 0
    assert len(corpus.load_manifest(manifest)[0]["sha256"]) == 64

    # From now on the file is checked against what was recorded.
    (tmp_path / "corpus" / "paper.pdf").write_bytes(make_pdf(pages=5))
    assert corpus.main(arguments) == 1


def test_the_command_refuses_an_unsound_manifest(tmp_path, capsys):
    manifest = tmp_path / "manifest.csv"
    corpus.save_manifest([row(url="")], manifest)
    assert corpus.main(["--manifest", str(manifest), "--dest", str(tmp_path / "corpus")]) == 2
    assert "needs the url" in capsys.readouterr().out
