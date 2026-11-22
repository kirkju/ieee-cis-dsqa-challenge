"""tools/evidence.py: a reference passage has to be where it says it is."""

import json

import pymupdf
import pytest

from tools import corpus, evidence

SAMPLE = [{"id": 1}, {"id": 2}, {"id": 3}]


def manifest_row(doc_id="loop-paper", questions="1;2", role="source"):
    values = dict.fromkeys(corpus.FIELDS, "")
    values.update(doc_id=doc_id, domain="DONES", role=role, questions=questions, title="A paper")
    values.update(access="manual")
    return values


QUOTE = "The nominal flow rate of the main loop is 97.5 l/s."


def passage(page=1, quote=QUOTE, kind="text", doc_id="loop-paper"):
    return {"doc_id": doc_id, "page": page, "quote": quote, "kind": kind}


def traced(question_id=1, doc_id="loop-paper", passages=None):
    passages = [passage(doc_id=doc_id)] if passages is None else passages
    return {"id": question_id, "status": "traced", "doc_id": doc_id, "evidence": passages}


def sound_items():
    return [
        traced(1),
        {"id": 2, "status": "unavailable", "doc_id": "loop-paper", "evidence": []},
        {"id": 3, "status": "untraced", "evidence": []},
    ]


def write_pdf(path, pages):
    with pymupdf.open() as document:
        for text in pages:
            document.new_page().insert_text((72, 72), text, fontsize=9)
        path.parent.mkdir(parents=True, exist_ok=True)
        document.save(path)
    return path


def test_normalizing_keeps_what_survives_a_change_of_layout():
    assert evidence.normalize("The  Heat\nExchanger") == "the heat exchanger"
    # A word split across two lines is one word again.
    assert evidence.normalize("activa-\ntion of lithium") == "activation of lithium"
    # Ligatures and superscripts are spelled out; soft hyphens vanish.
    spelled = "ef\N{LATIN SMALL LIGATURE FI}cient \N{SUPERSCRIPT SEVEN}Be co\N{SOFT HYPHEN}de"
    assert evidence.normalize(spelled) == "efficient 7be code"
    # A hyphen inside a line is part of the text.
    assert evidence.normalize("FISPACT-II") == "fispact-ii"
    # Quotes are written in the clean form: the same, with its case kept.
    assert evidence.clean("The  Heat\nExchan-\nger") == "The Heat Exchanger"


def test_a_quote_is_found_whatever_the_line_breaks():
    page = "the lithium mass flow rate on the\nshell side of HX1 is 49.7 kg/s at nomi-\nnal"
    page += " conditions"
    assert evidence.contains(page, "shell side of HX1 is 49.7 kg/s at nominal conditions")
    assert not evidence.contains(page, "shell side of HX1 is 94.7 kg/s")


def test_sound_annotations_have_no_problems():
    assert evidence.problems(sound_items(), SAMPLE, [manifest_row()]) == []


def test_every_sample_question_needs_one_annotation():
    found = evidence.problems(sound_items()[:2], SAMPLE, [manifest_row()])
    assert found == ["question 3 has no annotation"]

    extra = [*sound_items(), {"id": 9, "status": "untraced", "evidence": []}]
    assert "question 9 is annotated but is not among the questions" in evidence.problems(
        extra, SAMPLE, [manifest_row()]
    )

    twice = [*sound_items(), {"id": 3, "status": "untraced", "evidence": []}]
    assert "a question is annotated more than once" in evidence.problems(
        twice, SAMPLE, [manifest_row()]
    )


@pytest.mark.parametrize(
    ("item", "complaint"),
    [
        ({"id": 1, "status": "guessed", "evidence": []}, "status must be one of"),
        (traced(1, doc_id="no-such-paper"), "is not in the manifest"),
        (traced(1, passages=[]), "needs a passage in its source document"),
        (traced(1, passages=[passage(page=0)]), "counted from 1"),
        (traced(1, passages=[passage(quote="97.5 l/s")]), "under 20 characters"),
        (traced(1, passages=[passage(kind="plot")]), "kind must be one of"),
        (
            {"id": 1, "status": "unavailable", "doc_id": "loop-paper", "evidence": [passage()]},
            "a document we do not have",
        ),
        (
            {"id": 1, "status": "untraced", "doc_id": "loop-paper", "evidence": []},
            "cannot name its source document",
        ),
    ],
)
def test_each_kind_of_mistake_in_an_annotation_is_named(item, complaint):
    items = [item, *sound_items()[1:]]
    found = evidence.problems(items, SAMPLE, [manifest_row()])
    assert len(found) == 1
    assert complaint in found[0]
    assert found[0].startswith("question 1")


def test_a_passage_can_be_in_another_document_that_states_the_same_facts():
    # The overview repeats the sentence of the loop paper; the manifest says
    # so by listing question 1 under both.
    manifest = [manifest_row(), manifest_row("overview", questions="1")]
    items = sound_items()
    items[0] = traced(1, passages=[passage(), passage(doc_id="overview", page=27)])
    assert evidence.problems(items, SAMPLE, manifest) == []

    # Not if the manifest does not back it...
    unlisted = [manifest_row(), manifest_row("overview", questions="", role="distractor")]
    assert evidence.problems(items, SAMPLE, unlisted) == [
        "question 1: the manifest does not list it under overview"
    ]

    # ... and, for a traced question, never instead of a passage in its source.
    items[0] = traced(1, passages=[passage(doc_id="overview", page=27)])
    assert evidence.problems(items, SAMPLE, manifest) == [
        "question 1: a traced question needs a passage in its source document"
    ]


def test_a_question_whose_source_we_lack_can_still_have_passages_elsewhere():
    # The source of question 2 is paywalled and that of question 3 unknown,
    # but the overview states what both answers need.
    manifest = [manifest_row(), manifest_row("overview", questions="2;3")]
    items = sound_items()
    items[1]["evidence"] = [passage(doc_id="overview", page=27)]
    items[2]["evidence"] = [passage(doc_id="overview", page=28)]
    assert evidence.problems(items, SAMPLE, manifest) == []

    # Not in the document we do not have, though.
    items[1]["evidence"] = [passage(doc_id="loop-paper")]
    assert evidence.problems(items, SAMPLE, manifest) == [
        "question 2: passages cannot be located in a document we do not have"
    ]


def test_the_manifest_has_to_list_the_question_under_the_document():
    # The paper is in the manifest, but as the source of other questions.
    found = evidence.problems(sound_items(), SAMPLE, [manifest_row(questions="2")])
    assert found == ["question 1: the manifest does not list it under loop-paper"]

    as_distractor = manifest_row(questions="", role="distractor")
    assert len(evidence.problems(sound_items(), SAMPLE, [as_distractor])) == 2


def test_passages_needed_together_share_a_group():
    together = [dict(passage(), group="a"), dict(passage(page=2), group="a"), passage(page=3)]
    items = [traced(1, passages=together), *sound_items()[1:]]
    assert evidence.problems(items, SAMPLE, [manifest_row()]) == []

    items[0] = traced(1, passages=[dict(passage(), group="a"), passage(page=2)])
    assert evidence.problems(items, SAMPLE, [manifest_row()]) == [
        "question 1: a group of one passage is not a group"
    ]
    items[0] = traced(1, passages=[dict(passage(), group=1), dict(passage(page=2), group=1)])
    assert evidence.problems(items, SAMPLE, [manifest_row()]) == [
        "question 1: a group is a name shared by passages needed together"
    ]


def test_questions_of_our_own_only_have_to_name_documents_of_the_manifest():
    # The manifest lists the sample's questions by their ids. A set written
    # from other documents is not in it, and its documents are distractors there.
    as_distractor = [manifest_row(questions="", role="distractor")]
    assert evidence.problems(sound_items(), SAMPLE, as_distractor, listed=False) == []

    elsewhere = [traced(1, doc_id="no-such-paper"), *sound_items()[1:]]
    found = evidence.problems(elsewhere, SAMPLE, as_distractor, listed=False)
    assert found == ["question 1: 'no-such-paper' is not in the manifest"]


def test_a_quote_on_its_page_passes_and_one_elsewhere_is_pointed_out(tmp_path):
    write_pdf(
        tmp_path / "loop-paper.pdf",
        [
            "Introduction to the lithium loop.",
            "The nominal flow rate of the main\nloop is 97.5 l/s.",
        ],
    )
    right = traced(1, passages=[passage(page=2)])
    wrong_page = traced(1, passages=[passage(page=1)])
    nowhere = traced(1, passages=[passage(page=2, quote="The nominal flow rate is 12.0 l/s.")])

    assert evidence.misplaced([right], [manifest_row()], tmp_path) == ([], [])

    wrong, absent = evidence.misplaced([wrong_page], [manifest_row()], tmp_path)
    assert absent == []
    assert len(wrong) == 1
    assert "not on page 1 of loop-paper; it is on page 2" in wrong[0]

    wrong, _ = evidence.misplaced([nowhere], [manifest_row()], tmp_path)
    assert "not on page 2 of loop-paper:" in wrong[0]


def test_a_page_beyond_the_end_of_the_document_is_misplaced(tmp_path):
    write_pdf(tmp_path / "loop-paper.pdf", ["The nominal flow rate of the main loop is 97.5 l/s."])
    beyond = traced(1, passages=[passage(page=7)])
    wrong, _ = evidence.misplaced([beyond], [manifest_row()], tmp_path)
    assert "not on page 7 of loop-paper; it is on page 1" in wrong[0]


def test_a_document_that_is_not_on_this_machine_is_reported_not_failed(tmp_path):
    assert evidence.misplaced([traced(1)], [manifest_row()], tmp_path) == ([], ["loop-paper"])


def test_find_gives_the_pages_and_the_text_around(tmp_path):
    path = write_pdf(
        tmp_path / "loop-paper.pdf",
        [
            "Nothing about it here.",
            "The ICS extracts about 2.015 l/s from the loop.",
            "Again: 2.015 l/s.",
        ],
    )
    hits = evidence.find(path, "2.015 L/S")
    assert [page for page, _ in hits] == [2, 3]
    # The text around a hit keeps its case, so a quote can be copied from it.
    assert "The ICS extracts about 2.015 l/s from the loop" in hits[0][1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A manifest, a corpus and an annotation file for a three-question sample."""
    sample = tmp_path / "sample_qa.json"
    sample.write_text(
        json.dumps([{"id": n, "domain": "DONES"} for n in (1, 2, 3)]), encoding="utf-8"
    )
    monkeypatch.setattr(evidence, "SAMPLE", sample)
    corpus.save_manifest([manifest_row(questions="1;2;3")], tmp_path / "manifest.csv")
    write_pdf(
        tmp_path / "corpus" / "loop-paper.pdf",
        ["The nominal flow rate of the main loop is 97.5 l/s."],
    )

    def arguments(items):
        path = tmp_path / "sample_evidence.json"
        path.write_text(json.dumps({"items": items}), encoding="utf-8")
        return ["--evidence", str(path), "--manifest", str(tmp_path / "manifest.csv")] + [
            "--corpus-dir",
            str(tmp_path / "corpus"),
        ]

    return arguments


def test_the_command_passes_sound_annotations_and_summarizes_them(project, capsys):
    assert evidence.main(project(sound_items())) == 0
    out = capsys.readouterr().out
    assert "DONES: 1 traced, 1 unavailable, 1 untraced (of 3); 1 with a passage" in out
    assert "1 passages: 0 figure, 0 table, 1 text; 0 outside the source document" in out


def test_the_command_counts_a_passage_of_a_question_without_a_source(project, capsys):
    # Question 3 has no known source, but the loop paper states its answer.
    items = sound_items()
    items[2]["evidence"] = [passage()]
    assert evidence.main(project(items)) == 0
    out = capsys.readouterr().out
    assert "1 untraced (of 3); 2 with a passage in the corpus" in out
    assert "2 passages: 0 figure, 0 table, 2 text; 1 outside the source document" in out


def test_the_command_fails_on_a_quote_that_is_not_in_the_document(project, capsys):
    items = sound_items()
    items[0] = traced(1, passages=[passage(quote="The nominal flow rate is 12.0 l/s.")])
    assert evidence.main(project(items)) == 1
    assert "not on page 1 of loop-paper" in capsys.readouterr().out


def test_the_command_checks_a_question_set_other_than_the_sample(project, tmp_path, capsys):
    # The same annotations, for questions the manifest knows nothing of.
    arguments = project(sound_items())
    corpus.save_manifest(
        [manifest_row(questions="", role="distractor")], tmp_path / "manifest.csv"
    )
    assert evidence.main(arguments) == 1  # as the sample's, the manifest would have to list them
    assert "the manifest does not list it under loop-paper" in capsys.readouterr().out

    own = tmp_path / "own_qa.json"
    own.write_text(json.dumps([{"id": n, "domain": "DONES"} for n in (1, 2, 3)]), encoding="utf-8")
    assert evidence.main([*arguments, "--questions", str(own)]) == 0
    assert "DONES: 1 traced, 1 unavailable, 1 untraced (of 3)" in capsys.readouterr().out


def test_the_command_searches_a_document(project, capsys):
    assert evidence.main([*project(sound_items()), "--find", "loop-paper", "97.5"]) == 0
    assert "page 1:" in capsys.readouterr().out
