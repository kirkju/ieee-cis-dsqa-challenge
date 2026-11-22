"""The development data kept in the repository is sound.

dev/corpus/manifest.csv and dev/questions/sample_evidence.json are partly
written by hand. These tests keep them consistent with each other, with the
organizers' sample and, for the PDFs that are on this machine, with the
documents themselves.
"""

import json

import pytest

from tools import corpus, evidence

MANIFEST = corpus.load_manifest()
ITEMS = evidence.load()
SAMPLE = json.loads(evidence.SAMPLE.read_text(encoding="utf-8"))


def test_the_manifest_is_sound():
    assert corpus.manifest_problems(MANIFEST) == []


def test_every_open_document_has_been_downloaded_once_and_its_hash_recorded():
    # A row without a hash describes a file nobody has checked.
    unrecorded = [
        row["doc_id"]
        for row in MANIFEST
        if row["access"] == "open" and not (row["sha256"] and row["pages"])
    ]
    assert unrecorded == []


def test_no_two_rows_are_the_same_file():
    hashes = [row["sha256"] for row in MANIFEST if row["sha256"]]
    assert len(hashes) == len(set(hashes))


def test_the_manifest_and_the_annotations_agree_on_which_document_answers_what():
    # A document listed under a question is its source or holds one of its
    # passages; the other direction is checked by evidence.problems().
    used = set()
    for item in ITEMS:
        used.add((item.get("doc_id"), item["id"]))
        used.update((passage["doc_id"], item["id"]) for passage in item.get("evidence", []))
    listed = {
        (row["doc_id"], question) for row in MANIFEST for question in corpus.question_ids(row)
    }
    assert listed - used == set()
    assert {question for _, question in listed} <= {question["id"] for question in SAMPLE}


def test_the_annotations_are_sound():
    assert evidence.problems(ITEMS, SAMPLE, MANIFEST) == []


def test_every_quote_is_on_the_page_it_names():
    wrong, absent = evidence.misplaced(ITEMS, MANIFEST)
    assert wrong == []
    with_passages = {passage["doc_id"] for item in ITEMS for passage in item.get("evidence", [])}
    if with_passages and set(absent) == with_passages:
        pytest.skip("the corpus is not on this machine: python -m tools.corpus")


# The question sets of our own: written from documents of the corpus and
# annotated like the sample. What is tuned on the first is confirmed on the
# second, so they must share nothing.
OWN = {
    name: (
        json.loads((evidence.EVIDENCE.parent / f"{name}_qa.json").read_text(encoding="utf-8")),
        evidence.load(evidence.EVIDENCE.parent / f"{name}_evidence.json"),
    )
    for name in ("tuning", "control")
}
QUESTION_TYPES = {question["question_type"] for question in SAMPLE}


@pytest.mark.parametrize("name", sorted(OWN))
def test_a_question_set_of_our_own_has_the_shape_of_the_sample(name):
    questions, _ = OWN[name]
    assert all(set(question) == set(SAMPLE[0]) for question in questions)
    assert {question["domain"] for question in questions} <= corpus.DOMAINS
    assert {question["question_type"] for question in questions} <= QUESTION_TYPES
    assert all(question["question"].strip() for question in questions)
    assert all(question["answer"].strip() for question in questions)


@pytest.mark.parametrize("name", sorted(OWN))
def test_the_annotations_of_a_question_set_of_our_own_are_sound(name):
    questions, items = OWN[name]
    assert evidence.problems(items, questions, MANIFEST, listed=False) == []
    assert all(item["status"] == "traced" for item in items)
    # A question is of the domain of the document it was written from.
    domains = {row["doc_id"]: row["domain"] for row in MANIFEST}
    asked = {question["id"]: question["domain"] for question in questions}
    assert all(domains[item["doc_id"]] == asked[item["id"]] for item in items)


@pytest.mark.parametrize("name", sorted(OWN))
def test_every_quote_of_a_question_set_of_our_own_is_on_its_page(name):
    _, items = OWN[name]
    wrong, absent = evidence.misplaced(items, MANIFEST)
    assert wrong == []
    if absent:
        pytest.skip("the corpus is not on this machine: python -m tools.corpus")


def test_the_question_sets_share_no_question_and_no_document():
    ids = [question["id"] for questions, _ in OWN.values() for question in questions]
    ids += [question["id"] for question in SAMPLE]
    assert len(ids) == len(set(ids))
    texts = [question["question"] for questions, _ in OWN.values() for question in questions]
    assert len(texts) == len(set(texts))
    tuning = {item["doc_id"] for item in OWN["tuning"][1]}
    control = {item["doc_id"] for item in OWN["control"][1]}
    assert tuning & control == set()


def test_the_corpus_on_this_machine_is_the_one_in_the_manifest():
    if not corpus.CORPUS_DIR.is_dir():
        pytest.skip("the corpus is not on this machine: python -m tools.corpus")
    outcome = corpus.sync(MANIFEST, fetch=None, say=lambda line: None)
    assert outcome["mismatch"] == []
    assert outcome["failed"] == []
