"""The entry, for the tools that measure it and borrow its parts.

The entry is the folder that becomes the submission zip, 03_dsqa, and its
code is the package dsqa inside it. The evaluation imports main.py from that
folder, so the package is importable there by its bare name; importing this
module makes it so here, and gives the package's modules by name:

    from tools.entry import chunk, parse
"""

import hashlib
import importlib.metadata
import sys
from pathlib import Path

from tools.ledger import REPO_ROOT

ENTRY = REPO_ROOT / "03_dsqa"
if str(ENTRY) not in sys.path:
    sys.path.insert(0, str(ENTRY))

from dsqa import (  # noqa: E402, F401
    chunk,
    config,
    context,
    engine,
    parse,
    prompt,
    rank,
    retrieval,
    tables,
    text,
)


def source_hash(*modules) -> str:
    """A short name for the code of some of the entry's modules as it is now.
    What a tool keeps of their output is kept under it, and is made again
    when the code changes."""
    digest = hashlib.sha256()
    for module in modules:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:8]


def parse_key(reading: str = "tables") -> str:
    """The name PyMuPDF's text of a document is kept under, as the entry
    reads it: with its tables, only "titled", or as bare "blocks". It names
    the version of PyMuPDF and the code that does the reading."""
    code = source_hash(parse, tables, text)
    return f"pymupdf-{importlib.metadata.version('pymupdf')}-{code}-{reading}"
