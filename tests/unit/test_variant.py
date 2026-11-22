"""dsqa/variant.py as this entry carries it: only the ways of route.py."""

import pytest
from dsqa import variant

# What this branch ships with: the adjustments written as DEFAULT. A
# candidate's branch writes its own here and in variant.py, and nothing else.
SHIPPED = "doc-focus,doc-name,doc-reserve,dones-plain,llm-one,llm-proof,llm-rank,llm-wide,symbol-glyphs"


def test_without_the_variable_the_entry_is_the_one_the_branch_ships(monkeypatch):
    monkeypatch.delenv("DSQA_VARIANT", raising=False)
    assert variant.DEFAULT == SHIPPED
    assert variant.chosen() == frozenset(name for name in SHIPPED.split(",") if name)
    assert variant.described() == (SHIPPED or "none")


def test_an_empty_variable_chooses_none(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "")
    assert variant.chosen() == frozenset()


def test_the_names_are_read_from_the_variable(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", " dones-zero , dones-plain ")
    assert variant.chosen() == {"dones-zero", "dones-plain"}
    assert variant.on("dones-zero") and not variant.on("dones-example")
    assert variant.described() == "dones-plain,dones-zero"


def test_a_name_that_is_not_an_adjustment_is_an_error(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "rules-near")
    with pytest.raises(variant.UnknownVariant, match="rules-near"):
        variant.chosen()
    with pytest.raises(variant.UnknownVariant):
        variant.on("dones-zeroo")


def test_every_default_name_is_known():
    assert set(filter(None, variant.DEFAULT.split(","))) <= set(variant.KNOWN)


def test_the_entry_knows_the_ways_of_route_py_plain_relevance_doc_focus_and_glyphs():
    from dsqa import glyphs, relevance, route

    # And plain, the base model's judgment of the dones-plain ranking, the
    # context of the document a question on AI papers is about, llm-acted and
    # llm-proof, and the glyphs of the Symbol font read as their characters.
    doc_focus = {"doc-focus", "doc-reserve", "doc-name"}
    others = {"plain", route.ACTED, route.PROOF, glyphs.NAME} | set(relevance.WAYS) | doc_focus
    assert set(variant.KNOWN) - set(route.WAYS) == others
    assert set(route.WAYS) <= set(variant.KNOWN)
