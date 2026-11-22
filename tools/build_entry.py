"""Build the submission zip of our entry, and check it.

    python -m tools.build_entry                    # -> 03_dsqa_submission.zip
    python -m tools.build_entry --check some.zip   # check a zip that is already built

What goes in is named, not left to whatever the folder holds: main.py, the
package dsqa next to it, the embedding model, and a file VERSION that says
which commit this is. Nothing else of the folder goes in.

A zip can be uploaded once a day and what comes back is a status and a
score, so the zip is held to what can be known here: it is built from a
clean tree, or it says that it was not; it is the same zip, byte for byte,
whenever it is built from the same commit; and it is checked for what the
rules ask and for what would fail unseen: main.py at its root, no more than
1 GB once unzipped, nothing in it but what is named above, every file of
code readable by the evaluation's Python.

The same zip on whichever machine, too: built on Windows and on Linux from
one commit it has one hash, so the zip that was run on a rented GPU can be
told to be the zip that is uploaded. Two things had to be taken out of the
machine's hands for that, the system a zip says it was made on and the line
endings of the embedding model's small text files.

The embedding model is not in the repository. It is the organizers'
example's, sentence-transformers/all-MiniLM-L6-v2, and --fetch-embedder
downloads it into the entry's folder, where the examples' build scripts put
theirs.

When the entry ships with dones-example (variant.DEFAULT names it), the
organizers' RAG example goes in as well: example_rag.py, its main.py as it
came, and Docling's models in docling_models/, both at the root of the zip
where the example finds them, as its own build.sh bundles them. The models
are not in the repository either: they are the RAG example's, taken from
03_dsqa/docling_models when they are there and else from the example's own
02_rag/docling_models, where its build.sh and the rented machine's setup
download them. What a download keeps of itself (.cache/) is left out, which
makes the zip the same wherever the models were fetched.
"""

import argparse
import ast
import hashlib
import sys
import zipfile
from pathlib import Path

from tools import ledger

REPO_ROOT = ledger.REPO_ROOT
ENTRY = REPO_ROOT / "03_dsqa"
ZIP = REPO_ROOT / "03_dsqa_submission.zip"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# The rules' limit on a submission once unzipped.
LIMIT_BYTES = 10**9
# The one date every file in the zip carries, so that the zip depends on what
# is in the files and on nothing else: the earliest a zip can state.
DATE = (1980, 1, 1, 0, 0, 0)
# The organizers' RAG example, which the entry answers with when it ships
# with dones-example (dsqa/example.py): its main.py as it came, at the root
# of the zip next to Docling's models, where its paths find them; and what
# tells that Docling's models are there.
EXAMPLE_WAY = "dones-example"
# The example's code over the entry's own text needs its file and no models.
LITE_WAY = "dones-lite"
EXAMPLE_FILE = "example_rag.py"
EXAMPLE_MODELS = "docling_models"
EXAMPLE_WEIGHTS = (".safetensors", ".onnx")
# What may be at the root of the zip, and what the embedding model needs to load.
ROOT_NAMES = ("main.py", "VERSION", EXAMPLE_FILE)
FOLDERS = ("dsqa/", "embed_model/", f"{EXAMPLE_MODELS}/")
EMBEDDER_FILES = ("embed_model/modules.json", "embed_model/config.json")
EMBEDDER_WEIGHTS = ("embed_model/model.safetensors", "embed_model/pytorch_model.bin")
# The embedding model's text files. sentence-transformers writes them with
# the line endings of the machine that downloads the model.
EMBEDDER_TEXT = (".json", ".md", ".txt")
# The system a zip says it was made on, for every file: Unix, whose file
# permissions are the ones given below. Python would say the machine's own.
UNIX = 3


class BuildError(Exception):
    """The zip cannot be built from what is there."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _default_of(source: str) -> set[str]:
    """The names variant.DEFAULT has in the text of a variant.py."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT" for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            return {name.strip() for name in names.split(",") if name.strip()}
    return set()


def shipped(entry: Path = ENTRY) -> set[str]:
    """The adjustments the entry runs with as it ships: variant.DEFAULT, in
    the entry's own dsqa/variant.py."""
    source = entry / "dsqa" / "variant.py"
    return _default_of(source.read_text(encoding="utf-8")) if source.is_file() else set()


def ships_example(entry: Path = ENTRY) -> bool:
    """Whether the entry answers with the organizers' RAG example as it
    ships: when variant.DEFAULT names dones-example. Then the example's file
    and Docling's models go in."""
    return EXAMPLE_WAY in shipped(entry)


def ships_lite(entry: Path = ENTRY) -> bool:
    """Whether the entry answers with the RAG example's code over its own
    text as it ships (dones-lite): then the example's file goes in, and
    Docling's models do not, since nothing reads with them."""
    return LITE_WAY in shipped(entry)


def docling_models(entry: Path = ENTRY) -> Path | None:
    """Where Docling's models are to be taken from for the zip: next to the
    entry's main.py, or else where the organizers' RAG example keeps them,
    02_rag/docling_models beside the entry's folder, where its build.sh and
    the rented machine's setup download them. The same files either way.
    None when they are in neither."""
    for folder in (entry / EXAMPLE_MODELS, entry.parent / "02_rag" / EXAMPLE_MODELS):
        if folder.is_dir():
            return folder
    return None


def contents(entry: Path = ENTRY, embedder: bool = True) -> list[tuple[str, Path]]:
    """What goes into the zip from the entry's folder: the name each file has
    there and where it is, in the order of the names."""
    if not (entry / "main.py").is_file():
        raise BuildError(f"no main.py in {entry}")
    files = [entry / "main.py", *(entry / "dsqa").rglob("*.py")]
    if embedder:
        model = entry / "embed_model"
        if not model.is_dir():
            raise BuildError(
                f"no embedding model in {model}: run with --fetch-embedder to download {EMBEDDER}"
            )
        files += [path for path in model.rglob("*") if path.is_file()]
    named_apart: dict[str, Path] = {}
    if ships_example(entry):
        example, models = entry / EXAMPLE_FILE, docling_models(entry)
        if not example.is_file():
            raise BuildError(f"the entry ships with {EXAMPLE_WAY} and {example} is not there")
        if models is None:
            raise BuildError(
                f"the entry ships with {EXAMPLE_WAY} and Docling's models are neither in "
                f"{entry / EXAMPLE_MODELS} nor in {entry.parent / '02_rag' / EXAMPLE_MODELS}: "
                "download them as the RAG example's build.sh does"
            )
        files.append(example)
        # In the zip they are next to main.py, wherever they were found.
        for path in models.rglob("*"):
            if path.is_file():
                named_apart[f"{EXAMPLE_MODELS}/{path.relative_to(models).as_posix()}"] = path
    elif ships_lite(entry):
        example = entry / EXAMPLE_FILE
        if not example.is_file():
            raise BuildError(f"the entry ships with {LITE_WAY} and {example} is not there")
        files.append(example)
    named = {path.relative_to(entry).as_posix(): path for path in files}
    named.update(named_apart)
    # What a download keeps of itself is no part of a model, and says when
    # and on which machine it was made.
    return sorted(
        (name, path)
        for name, path in named.items()
        if "__pycache__" not in name and "/.cache/" not in name
    )


def _bytes(name: str, path: Path) -> bytes:
    """A file as it goes into the zip: as it is, but for the embedding
    model's text files, which go in with LF line endings wherever the model
    was downloaded."""
    data = path.read_bytes()
    if name.startswith("embed_model/") and name.endswith(EMBEDDER_TEXT):
        data = data.replace(b"\r\n", b"\n")
    return data


def version(state: dict, files: list[tuple[str, Path]]) -> str:
    """The text of VERSION: the commit the zip was built from, whether the
    tree had changes that are not in it, and the embedding model by the hash
    of its weights, which no commit names."""
    lines = [f"commit {state.get('commit') or 'unknown'}"]
    if state.get("dirty") is not False:
        lines.append("built from a tree with uncommitted changes")
    for name, path in files:
        if name in EMBEDDER_WEIGHTS:
            lines.append(f"{name} sha256 {_sha256(path.read_bytes())} ({EMBEDDER})")
        elif name.startswith(EXAMPLE_MODELS + "/") and name.endswith(EXAMPLE_WEIGHTS):
            lines.append(f"{name} sha256 {_sha256(path.read_bytes())} (Docling's layout model)")
    return "\n".join(lines) + "\n"


def build(
    out: Path = ZIP,
    entry: Path = ENTRY,
    *,
    state: dict | None = None,
    allow_dirty: bool = False,
    embedder: bool = True,
) -> dict:
    """Write the zip and return what it is: its hash, its files and its size
    once unzipped. `state` is the tree's, as tools.ledger gives it."""
    state = ledger.git_state() if state is None else state
    if state.get("dirty") is not False and not allow_dirty:
        raise BuildError(
            "the tree has uncommitted changes: commit them, or pass --allow-dirty for a zip "
            "that is not to be uploaded"
        )
    files = contents(entry, embedder)
    written = {"VERSION": version(state, files).encode("utf-8")}
    written.update({name: _bytes(name, path) for name, path in files})
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(written):
            info = zipfile.ZipInfo(name, date_time=DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = UNIX
            info.external_attr = 0o644 << 16
            archive.writestr(info, written[name])
    return {
        "path": out,
        "sha256": _sha256(out.read_bytes()),
        "files": sorted(written),
        "unzipped_bytes": sum(len(data) for data in written.values()),
        "zipped_bytes": out.stat().st_size,
        "commit": state.get("commit"),
        "dirty": state.get("dirty") is not False,
    }


def check(path: Path, limit: int = LIMIT_BYTES, embedder: bool = True) -> list[str]:
    """What is wrong with a zip as a submission of ours; nothing, when it is
    fit to upload."""
    if not zipfile.is_zipfile(path):
        return [f"{path.name} is not a zip file"]
    problems = []
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if "main.py" not in names:
            nested = sorted(name for name in names if name.endswith("/main.py"))
            where = f"; there is one at {nested[0]}" if nested else ""
            problems.append(f"no main.py at the root of the zip{where}")
        for name in names:
            if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                problems.append(f"{name}: not a plain path inside the zip")
            elif name not in ROOT_NAMES and not name.startswith(FOLDERS):
                problems.append(f"{name}: not part of the entry")
            elif "__pycache__" in name or name.endswith(".pyc"):
                problems.append(f"{name}: compiled code, not source")
        size = sum(info.file_size for info in infos)
        if size > limit:
            problems.append(f"{size:,} bytes once unzipped, over the limit of {limit:,}")
        for name in names:
            if name.endswith(".py"):
                try:
                    ast.parse(archive.read(name).decode("utf-8"), name, feature_version=(3, 10))
                except (SyntaxError, UnicodeDecodeError) as error:
                    problems.append(f"{name}: Python 3.10 cannot read it: {error}")
        if "VERSION" not in names:
            problems.append("no VERSION: the zip does not say which commit it is")
        elif "uncommitted changes" in archive.read("VERSION").decode("utf-8", "replace"):
            problems.append("built from a tree with uncommitted changes: not to be uploaded")
        if embedder:
            missing = [name for name in EMBEDDER_FILES if name not in names]
            if not any(name in names for name in EMBEDDER_WEIGHTS):
                missing.append(" or ".join(EMBEDDER_WEIGHTS))
            problems += [f"{name}: missing, the embedding model cannot load" for name in missing]
        # The organizers' RAG example and Docling's models go in together.
        has_models = any(
            name.startswith(EXAMPLE_MODELS + "/") and name.endswith(EXAMPLE_WEIGHTS)
            for name in names
        )
        # The example's file alone is what dones-lite needs: its code, over
        # the entry's own text. Any other entry with it needs Docling's models.
        default = set()
        if "dsqa/variant.py" in names:
            try:
                default = _default_of(archive.read("dsqa/variant.py").decode("utf-8"))
            except (SyntaxError, ValueError, UnicodeDecodeError):
                default = set()
        lite_alone = LITE_WAY in default and EXAMPLE_WAY not in default
        if EXAMPLE_FILE in names and not has_models and not lite_alone:
            problems.append(f"{EXAMPLE_FILE} without Docling's models: the example cannot read")
        if has_models and EXAMPLE_FILE not in names:
            problems.append(f"Docling's models without {EXAMPLE_FILE}: nothing reads with them")
    return problems


def fetch_embedder(entry: Path = ENTRY) -> Path:
    """Download the embedding model into the entry's folder. Needs the internet."""
    from sentence_transformers import SentenceTransformer

    target = entry / "embed_model"
    SentenceTransformer(EMBEDDER).save(str(target))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the submission zip of our entry.")
    parser.add_argument("--out", type=Path, default=ZIP, help="where the zip goes")
    parser.add_argument("--entry", type=Path, default=ENTRY, help="the entry's folder")
    parser.add_argument("--check", type=Path, metavar="ZIP", help="only check this zip")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="build from a tree with uncommitted changes"
    )
    parser.add_argument(
        "--fetch-embedder", action="store_true", help=f"download {EMBEDDER} into the entry first"
    )
    args = parser.parse_args(argv)

    if args.check is None:
        try:
            if args.fetch_embedder:
                print(f"[build] embedding model in {fetch_embedder(args.entry).as_posix()}")
            built = build(args.out, args.entry, allow_dirty=args.allow_dirty)
        except BuildError as error:
            print(f"[build] {error}")
            return 1
        megabytes = built["unzipped_bytes"] / 1e6
        print(
            f"[build] {built['path'].as_posix()}: {len(built['files'])} files, "
            f"{megabytes:.1f} MB once unzipped ({megabytes / (LIMIT_BYTES / 1e6):.0%} of the limit)"
        )
        print(f"[build] commit {built['commit']}, sha256 {built['sha256']}")
    problems = check(args.check or args.out)
    for problem in problems:
        print(f"[build] PROBLEM: {problem}")
    if not problems:
        print("[build] the zip is fit to upload as far as can be told here")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
