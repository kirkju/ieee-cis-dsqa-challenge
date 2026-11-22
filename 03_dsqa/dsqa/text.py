"""Text as it comes out of a PDF, put in a form that can be read and compared."""

import re
import unicodedata

# Invisible, and left behind by some PDF producers where a word may break.
SOFT_HYPHEN = "\N{SOFT HYPHEN}"


def clean(text: str) -> str:
    """Text with the marks of its layout undone: ligatures and compatibility
    characters unfolded, soft hyphens dropped, words rejoined across line
    breaks, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).replace(SOFT_HYPHEN, "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def squash(text: str) -> str:
    """The letters and digits of a text, case folded and with compatibility
    characters unfolded: what two spellings of the same words have in common
    whatever their spacing, hyphenation and punctuation."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())
