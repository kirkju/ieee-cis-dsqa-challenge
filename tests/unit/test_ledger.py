"""tools/ledger.py: a recorded number has to be traceable to a commit."""

import json
import subprocess

import pytest

from tools import ledger

# Identity and signing for the throwaway repositories, so the tests do not
# depend on the git configuration of whoever runs them.
GIT_OPTIONS = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
GIT_OPTIONS += ["-c", "commit.gpgsign=false"]


def _git(repo, *args):
    done = subprocess.run(
        ["git", *GIT_OPTIONS, *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A throwaway repository holding one committed code file."""
    _git(tmp_path, "init", "--quiet")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    return tmp_path


def _ledger(repo):
    return repo / "results" / "experiments.jsonl"


def _record(repo, kind="retrieval", metrics=None, **overrides):
    arguments = {
        "dataset": "sample-25",
        "corpus": "dev-v1",
        "engine": "none",
        "path": _ledger(repo),
        "repo": repo,
        **overrides,
    }
    return ledger.record(kind, {"recall_at_5": 0.8} if metrics is None else metrics, **arguments)


def test_a_run_on_a_clean_tree_carries_its_commit(repo):
    head = _git(repo, "rev-parse", "HEAD")
    row = _record(repo)
    assert row["commit"] == head
    assert row["dirty"] is False
    assert row["run_id"].split("-")[1] == head[:7]


def test_modified_code_marks_the_run_dirty(repo):
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    assert _record(repo)["dirty"] is True


def test_untracked_code_marks_the_run_dirty(repo):
    (repo / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    assert _record(repo)["dirty"] is True


def test_a_staged_rename_marks_the_run_dirty(repo):
    _git(repo, "mv", "code.py", "renamed.py")
    assert _record(repo)["dirty"] is True


def test_recording_does_not_taint_the_next_run(repo):
    # The ledger and the journal change between commits by design.
    _record(repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "journal.md").write_text("entry\n", encoding="utf-8")
    assert _record(repo)["dirty"] is False


def test_rows_are_appended_one_per_line(repo):
    first = _record(repo)
    before = _ledger(repo).read_bytes()
    second = _record(repo, metrics={"recall_at_5": 0.9})
    after = _ledger(repo).read_bytes()

    assert after.startswith(before)
    assert b"\r" not in after
    assert [json.loads(line) for line in after.decode("utf-8").splitlines()] == [first, second]
    assert first["run_id"] != second["run_id"]


def test_a_missing_final_newline_does_not_glue_rows_together(repo):
    _record(repo)
    _ledger(repo).write_bytes(_ledger(repo).read_bytes().rstrip(b"\n"))
    _record(repo)
    assert len(ledger.read(_ledger(repo))) == 2


def test_config_hash_ignores_key_order():
    assert ledger.config_hash({"k": 5, "chunk": 800}) == ledger.config_hash({"chunk": 800, "k": 5})
    assert ledger.config_hash({"k": 5}) != ledger.config_hash({"k": 6})
    assert ledger.config_hash(None) == ledger.config_hash({})


def test_a_non_finite_metric_is_refused_and_nothing_is_written(repo):
    with pytest.raises(ValueError):
        _record(repo, metrics={"accuracy": float("nan")})
    assert not _ledger(repo).exists()


def test_scalars_from_numeric_libraries_are_stored_as_plain_numbers(repo):
    np = pytest.importorskip("numpy")
    row = _record(repo, metrics={"recall_at_5": np.float32(0.5), "n": np.int64(25)})
    assert row["metrics"] == {"recall_at_5": 0.5, "n": 25}
    assert ledger.read(_ledger(repo)) == [row]


def test_an_unserializable_value_is_refused(repo):
    with pytest.raises(TypeError):
        _record(repo, metrics={"answers": {1, 2}})
    assert not _ledger(repo).exists()


def test_outside_a_repository_the_git_fields_are_empty(tmp_path, monkeypatch):
    # Stop git from finding a repository in a parent of the temporary folder.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    row = _record(tmp_path)
    assert (row["commit"], row["branch"], row["dirty"]) == (None, None, None)
    assert row["run_id"].split("-")[1] == "nogit"


def test_read_names_the_line_of_a_corrupt_row(repo):
    _record(repo)
    with open(_ledger(repo), "a", encoding="utf-8") as f:
        f.write("{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        ledger.read(_ledger(repo))


def test_read_of_a_missing_ledger_is_empty(tmp_path):
    assert ledger.read(tmp_path / "absent.jsonl") == []


def test_listing_filters_by_kind_and_flags_uncommitted_code(repo, capsys):
    _record(repo, kind="retrieval")
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    dirty = _record(repo, kind="end_to_end")

    assert ledger.main(["--path", str(_ledger(repo)), "--kind", "end_to_end"]) == 0
    out = capsys.readouterr().out
    assert dirty["commit"][:7] + "*" in out
    assert "retrieval" not in out
    assert "not to be quoted" in out


def test_listing_keeps_the_most_recent_runs(repo, capsys):
    for value in (0.1, 0.2, 0.3):
        _record(repo, metrics={"recall_at_5": value})

    assert ledger.main(["--path", str(_ledger(repo)), "--last", "1"]) == 0
    out = capsys.readouterr().out
    # Rows written within the same second keep the order they were written in.
    assert out.count("\n") == 1
    assert "recall_at_5=0.3" in out


def test_a_tool_can_name_the_run_and_give_the_state_it_started_in(repo):
    # A long run records the tree as it was at the start, under the name its
    # output folder already has.
    started_in = ledger.git_state(repo)
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")

    row = _record(repo, run_id="20261008T120000Z-abc1234-beef", state=started_in)

    assert row["run_id"] == "20261008T120000Z-abc1234-beef"
    assert row["dirty"] is False
    assert row["commit"] == started_in["commit"]
