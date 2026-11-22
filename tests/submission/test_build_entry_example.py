"""tools/build_entry.py with dones-example: the organizers' RAG example and
Docling's models go into the zip when the entry ships with that adjustment,
and only then."""

import zipfile

import pytest

from tools.build_entry import BuildError, build, check, docling_models, ships_example

CLEAN = {"commit": "c0ffee" * 6 + "c0ff", "branch": "main", "dirty": False}


def entry_shipping(tmp_path, default: str, example: bool = True, models: bool = True):
    folder = tmp_path / "entry"
    (folder / "dsqa").mkdir(parents=True)
    (folder / "embed_model").mkdir()
    (folder / "main.py").write_text("def answer(q, c, m):\n    return 'x'\n", encoding="utf-8")
    (folder / "dsqa" / "__init__.py").write_text('"""The package."""\n', encoding="utf-8")
    (folder / "dsqa" / "variant.py").write_text(
        f'"""Adjustments."""\n\nDEFAULT = "{default}"\nKNOWN = {{}}\n', encoding="utf-8"
    )
    model = (("modules.json", b"[]"), ("config.json", b"{}"), ("model.safetensors", b"w"))
    for name, data in model:
        (folder / "embed_model" / name).write_bytes(data)
    if example:
        (folder / "example_rag.py").write_text("_cache = {}\n", encoding="utf-8")
    if models:
        layout = folder / "docling_models" / "docling-project--docling-layout-heron"
        (layout / ".cache" / "huggingface").mkdir(parents=True)
        (layout / "model.safetensors").write_bytes(b"the layout model")
        (layout / "config.json").write_text("{}", encoding="utf-8")
        (layout / ".cache" / "huggingface" / "CACHEDIR.TAG").write_text("made here and now")
    return folder


def test_without_the_adjustment_neither_goes_in(tmp_path):
    folder = entry_shipping(tmp_path, "dones-zero")
    assert not ships_example(folder)
    built = build(tmp_path / "a.zip", folder, state=CLEAN)
    assert not any(name.startswith(("example_rag", "docling_models")) for name in built["files"])
    assert check(built["path"]) == []


def test_with_it_the_example_and_its_models_go_in_without_their_download_cache(tmp_path):
    folder = entry_shipping(tmp_path, "dones-example")
    assert ships_example(folder)
    built = build(tmp_path / "a.zip", folder, state=CLEAN)
    layout = "docling_models/docling-project--docling-layout-heron"
    assert "example_rag.py" in built["files"]
    assert f"{layout}/model.safetensors" in built["files"]
    assert f"{layout}/config.json" in built["files"]
    assert not any("/.cache/" in name for name in built["files"])
    with zipfile.ZipFile(built["path"]) as archive:
        stated = archive.read("VERSION").decode("utf-8")
    assert f"{layout}/model.safetensors sha256 " in stated
    assert check(built["path"]) == []


@pytest.mark.parametrize("missing", ["example", "models"])
def test_the_entry_shipping_with_it_needs_both(tmp_path, missing):
    folder = entry_shipping(
        tmp_path, "dones-example", example=missing != "example", models=missing != "models"
    )
    with pytest.raises(BuildError, match="dones-example"):
        build(tmp_path / "a.zip", folder, state=CLEAN)


@pytest.mark.parametrize(
    ("kept", "said"),
    [
        ("example_rag.py", "without Docling's models: the example cannot read"),
        ("docling_models/x/model.safetensors", "without example_rag.py: nothing reads"),
    ],
)
def test_a_zip_with_one_and_not_the_other_is_not_fit(tmp_path, kept, said):
    path = tmp_path / "one.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("main.py", "def answer(q, c, m):\n    return 'x'\n")
        archive.writestr("VERSION", "commit c0ffee\n")
        archive.writestr("embed_model/modules.json", "[]")
        archive.writestr("embed_model/config.json", "{}")
        archive.writestr("embed_model/model.safetensors", "w")
        archive.writestr(kept, "x")
    (problem,) = check(path)
    assert said in problem


def test_docling_s_models_are_taken_from_the_example_s_folder_when_not_next_to_main_py(tmp_path):
    folder = entry_shipping(tmp_path, "dones-example")
    beside = build(tmp_path / "beside.zip", folder, state=CLEAN)
    # As the RAG example's build.sh and the rented machine's setup leave them:
    # in 02_rag/, next to the entry's folder.
    (tmp_path / "02_rag").mkdir()
    (folder / "docling_models").rename(tmp_path / "02_rag" / "docling_models")
    assert docling_models(folder) == tmp_path / "02_rag" / "docling_models"
    apart = build(tmp_path / "apart.zip", folder, state=CLEAN)
    # The same names in the zip, and the same zip.
    assert apart["files"] == beside["files"] and apart["sha256"] == beside["sha256"]
    assert check(apart["path"]) == []
    # In neither place: not built, and both places named.
    (tmp_path / "02_rag" / "docling_models").rename(tmp_path / "elsewhere")
    with pytest.raises(BuildError, match="neither in .* nor in .*02_rag"):
        build(tmp_path / "none.zip", folder, state=CLEAN)


def test_with_dones_lite_the_example_goes_in_and_docling_s_models_do_not(tmp_path):
    folder = entry_shipping(tmp_path, "dones-lite")
    built = build(tmp_path / "a.zip", folder, state=CLEAN)
    assert "example_rag.py" in built["files"]
    assert not any(name.startswith("docling_models/") for name in built["files"])
    # The zip says it ships with dones-lite, so the example alone is fit.
    assert check(built["path"]) == []
    # And the example's file is needed.
    (folder / "example_rag.py").unlink()
    with pytest.raises(BuildError, match="dones-lite"):
        build(tmp_path / "b.zip", folder, state=CLEAN)
