"""The glyphs a symbol font draws, read as the characters they are
(symbol-glyphs).

A PDF that sets a Greek letter, a unit's micro sign or a sign such as ± or ≈
in the Symbol font often gives no Unicode for it, and PyMuPDF then reads the
glyph's code in the font as a character of the private use area, U+F020 to
U+F0FF: "10 µSv/h" comes out as "10 \\uf06dSv/h". The model cannot tell such a
character from any other, and of the 75 PDFs on IFMIF-DONES of the
development corpus, 17 have them; they hold 72 of the 239 passages that
answer the questions written on that domain.

The Symbol font has one encoding, so its codes are read here as the
characters they stand for: on each page, a code that only the Symbol font
draws there (or WineSymbol, its copy with the same encoding). Another font
that draws in the private use area (Wingdings, an icon font, the extra font
of an equation editor, TeX's pieces of large brackets) gives its own meaning
to the same codes, so a code it draws on a page is left as it is there,
whatever else draws it.
"""

import re
from collections.abc import Iterable

from . import variant

NAME = "symbol-glyphs"
PRIVATE = re.compile("[\uf020-\uf0ff]")

# The Symbol font's encoding, code by code, as Unicode. The pieces it draws
# large brackets, braces, integrals, radicals and arrows with (0x60, 0xBD,
# 0xBE, 0xE6 to 0xEF and 0xF3 to 0xFE) are not here, and are left as they
# are: a piece is not a sign, and an extender read as "|" would be a sign
# where there is none.
# fmt: off
_SYMBOL = {
    0x20: " ", 0x21: "!", 0x22: "∀", 0x23: "#", 0x24: "∃", 0x25: "%", 0x26: "&", 0x27: "∋",
    0x28: "(", 0x29: ")", 0x2A: "∗", 0x2B: "+", 0x2C: ",", 0x2D: "−", 0x2E: ".", 0x2F: "/",
    0x30: "0", 0x31: "1", 0x32: "2", 0x33: "3", 0x34: "4", 0x35: "5", 0x36: "6", 0x37: "7",
    0x38: "8", 0x39: "9", 0x3A: ":", 0x3B: ";", 0x3C: "<", 0x3D: "=", 0x3E: ">", 0x3F: "?",
    0x40: "≅", 0x41: "Α", 0x42: "Β", 0x43: "Χ", 0x44: "Δ", 0x45: "Ε", 0x46: "Φ", 0x47: "Γ",
    0x48: "Η", 0x49: "Ι", 0x4A: "ϑ", 0x4B: "Κ", 0x4C: "Λ", 0x4D: "Μ", 0x4E: "Ν", 0x4F: "Ο",
    0x50: "Π", 0x51: "Θ", 0x52: "Ρ", 0x53: "Σ", 0x54: "Τ", 0x55: "Υ", 0x56: "ς", 0x57: "Ω",
    0x58: "Ξ", 0x59: "Ψ", 0x5A: "Ζ", 0x5B: "[", 0x5C: "∴", 0x5D: "]", 0x5E: "⊥", 0x5F: "_",
    0x61: "α", 0x62: "β", 0x63: "χ", 0x64: "δ", 0x65: "ε", 0x66: "φ", 0x67: "γ",
    0x68: "η", 0x69: "ι", 0x6A: "ϕ", 0x6B: "κ", 0x6C: "λ", 0x6D: "μ", 0x6E: "ν", 0x6F: "ο",
    0x70: "π", 0x71: "θ", 0x72: "ρ", 0x73: "σ", 0x74: "τ", 0x75: "υ", 0x76: "ϖ", 0x77: "ω",
    0x78: "ξ", 0x79: "ψ", 0x7A: "ζ", 0x7B: "{", 0x7C: "|", 0x7D: "}", 0x7E: "∼",
    0xA0: "€", 0xA1: "ϒ", 0xA2: "′", 0xA3: "≤", 0xA4: "⁄", 0xA5: "∞", 0xA6: "ƒ", 0xA7: "♣",
    0xA8: "♦", 0xA9: "♥", 0xAA: "♠", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓",
    0xB0: "°", 0xB1: "±", 0xB2: "″", 0xB3: "≥", 0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•",
    0xB8: "÷", 0xB9: "≠", 0xBA: "≡", 0xBB: "≈", 0xBC: "…", 0xBF: "↵",
    0xC0: "ℵ", 0xC1: "ℑ", 0xC2: "ℜ", 0xC3: "℘", 0xC4: "⊗", 0xC5: "⊕", 0xC6: "∅", 0xC7: "∩",
    0xC8: "∪", 0xC9: "⊃", 0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈", 0xCF: "∉",
    0xD0: "∠", 0xD1: "∇", 0xD2: "®", 0xD3: "©", 0xD4: "™", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅",
    0xD8: "¬", 0xD9: "∧", 0xDA: "∨", 0xDB: "⇔", 0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓",
    0xE0: "◊", 0xE1: "⟨", 0xE2: "®", 0xE3: "©", 0xE4: "™", 0xE5: "∑", 0xF1: "⟩", 0xF2: "∫",
}
# fmt: on
SYMBOL = {0xF000 + code: char for code, char in _SYMBOL.items()}


def chosen() -> bool:
    """Whether the glyphs of the Symbol font are read as their characters."""
    return variant.on(NAME)


def symbolic(font: str) -> bool:
    """Whether a font is the Symbol font: its name, past the six letters and
    the plus sign of a subset, opens with "Symbol", or is WineSymbol."""
    name = font.split("+", 1)[1] if "+" in font[:7] else font
    return name.startswith("Symbol") or name == "WineSymbol"


def table_of(page, texts: Iterable[str]) -> dict[int, str] | None:
    """The translation for the texts read from a page: each code of the
    Symbol range that only the Symbol font draws on that page, as its
    character. None when there is none, or when symbol-glyphs is off. The
    fonts are looked at only on a page that has such a code."""
    if not chosen() or not any(PRIVATE.search(text) for text in texts):
        return None
    drawn: dict[bool, set[int]] = {True: set(), False: set()}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                codes = {ord(char) for char in PRIVATE.findall(span["text"])}
                drawn[symbolic(span["font"])] |= codes
    codes = drawn[True] - drawn[False]
    return {code: SYMBOL[code] for code in sorted(codes) if code in SYMBOL} or None
