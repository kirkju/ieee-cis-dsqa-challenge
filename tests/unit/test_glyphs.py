"""dsqa/glyphs.py (symbol-glyphs): the codes the Symbol font draws, read as
the characters they stand for, on a page where every character of the
private use area comes from that font; and its hook where the entry reads
the text of a PDF (tables.py and parse.py)."""

import pymupdf
import pytest
from dsqa import glyphs, parse, tables


class Page:
    """A page that answers get_text("dict") with the spans given, and counts
    how often it was asked."""

    def __init__(self, spans):
        self.spans = spans
        self.asked = 0

    def get_text(self, how):
        assert how == "dict"
        self.asked += 1
        lines = [{"spans": [{"text": text, "font": font} for text, font in self.spans]}]
        return {"blocks": [{"lines": lines}]}


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", glyphs.NAME)


def test_the_symbol_fonts_codes_are_the_characters_they_stand_for():
    text = "10 Sv/h,  = 0.5  0.1, T  300 C, x  2  3"
    assert text.translate(glyphs.SYMBOL) == "10 μSv/h, α = 0.5 ± 0.1, T ≈ 300 °C, Δx ≤ 2 × 3"
    # Every code of the table a character, none of the private use area.
    assert all(not glyphs.PRIVATE.search(char) for char in glyphs.SYMBOL.values())
    assert all(0xF020 <= code <= 0xF0FF for code in glyphs.SYMBOL)


def test_the_pieces_of_large_signs_are_left_as_they_are():
    # The pieces the font draws large brackets, braces, integrals, radicals
    # and arrows with: an extender read as "|" would be a sign where there is none.
    pieces = [0xF060, 0xF0BD, 0xF0BE, *range(0xF0E6, 0xF0F0), *range(0xF0F3, 0xF0FF)]
    assert not set(pieces) & set(glyphs.SYMBOL)
    assert "".translate(glyphs.SYMBOL) == ""
    assert glyphs.SYMBOL[0xF0F2] == "∫" and glyphs.SYMBOL[0xF0E5] == "∑"


@pytest.mark.parametrize(
    ("font", "symbolic"),
    [
        ("Symbol", True),
        ("SymbolMT", True),
        ("Symbol,Bold", True),
        ("ABCDEF+Symbol", True),
        ("WineSymbol", True),
        ("Segoe UI Symbol", False),
        ("MT Extra", False),
        ("Academicons", False),
        ("ABCDEF+MTExtra", False),
        ("TimesNewRomanPSMT", False),
        ("Wingdings-Regular", False),
    ],
)
def test_the_symbol_font_by_its_name(font, symbolic):
    assert glyphs.symbolic(font) is symbolic


def test_the_codes_the_symbol_font_draws_on_a_page_are_read_as_their_characters(on):
    page = Page([("10 Sv/h  1", "Symbol"), ("plain text", "Times")])
    table = glyphs.table_of(page, ["10 Sv/h  1"])
    assert table == {0xF06D: "μ", 0xF0B1: "±"}
    assert "10 Sv/h  1".translate(table) == "10 μSv/h ± 1"


def test_a_code_another_font_draws_on_the_page_is_left_as_it_is(on):
    # Wingdings draws its own glyphs at the same codes: its bullet is not a
    # Symbol sign, and a code both fonts draw on the page is left alone.
    page = Page([("10 Sv/h", "Symbol"), (" ", "Wingdings"), ("", "Symbol")])
    table = glyphs.table_of(page, ["10 Sv/h  "])
    assert table == {0xF06D: "μ"}
    assert "  10 Sv/h".translate(table) == "  10 μSv/h"


def test_a_page_whose_codes_are_all_of_other_fonts_is_left_as_it_is(on):
    page = Page([("", "CMEX10"), ("", "Wingdings")])
    assert glyphs.table_of(page, [" item"]) is None


def test_a_page_without_such_characters_is_not_looked_at(on):
    page = Page([("10 mSv/h", "Times")])
    assert glyphs.table_of(page, ["10 mSv/h", "a ± b"]) is None
    assert page.asked == 0


def test_off_unless_the_variable_names_it(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "")
    page = Page([("10 Sv/h", "Symbol")])
    assert glyphs.table_of(page, ["10 Sv/h"]) is None
    assert page.asked == 0


def pdf_with(tmp_path, text):
    path = tmp_path / "page.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 100), text, fontsize=11)
        document.save(path)
    return path


def test_the_hook_where_the_entry_reads_a_pdf(tmp_path, monkeypatch, on):
    """The translation is applied to the blocks and the words of a page, in
    tables.py and in parse.py, when glyphs.table_of gives one: here a stand-in
    that reads "m" as "μ", on a page of an ordinary font."""
    path = pdf_with(tmp_path, "The limit is 10 mSv/h in the room.")
    stand_in = {ord("m"): "μ"}
    monkeypatch.setattr(glyphs, "table_of", lambda page, texts: stand_in)
    with pymupdf.open(path) as document:
        page = tables._Page(document[0])
        assert page.texts == ["The liμit is 10 μSv/h in the rooμ."]
    assert "10 μSv/h" in parse.read(path, title=False, tables=False)
    assert "10 μSv/h" in parse.read(path, title=False, tables=True)
    monkeypatch.setattr(glyphs, "table_of", lambda page, texts: None)
    assert "10 mSv/h" in parse.read(path, title=False, tables=True)


def test_without_the_adjustment_a_pdf_reads_as_before(tmp_path, monkeypatch):
    path = pdf_with(tmp_path, "The limit is 10 mSv/h in the room.")
    monkeypatch.setenv("DSQA_VARIANT", "")
    before = parse.read(path)
    monkeypatch.setenv("DSQA_VARIANT", glyphs.NAME)
    assert parse.read(path) == before  # no character of the range on it
