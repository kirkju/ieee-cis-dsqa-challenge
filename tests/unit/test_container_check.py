"""tools/container_check.py: what is asked of Docker, and which files of the
working tree go over the clone. Docker itself is not run here.
"""

import subprocess

import pytest

from tools import container_check

GIT_OPTIONS = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
GIT_OPTIONS += ["-c", "commit.gpgsign=false"]


def git(repo, *args):
    subprocess.run(["git", *GIT_OPTIONS, *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A repository with three committed files and one that is ignored."""
    git(tmp_path, "init", "--quiet")
    (tmp_path / ".gitignore").write_text("/data/\n", encoding="utf-8")
    for name in ("kept.py", "changed.py", "deleted.py"):
        (tmp_path / name).write_text("first\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "--quiet", "-m", "first")
    return tmp_path


def test_a_clean_tree_has_no_files_to_copy_over_the_clone(repo):
    assert container_check.changed_files(repo) == []


def test_the_files_copied_are_the_changed_and_the_new_ones(repo):
    (repo / "changed.py").write_text("second\n", encoding="utf-8")
    (repo / "deleted.py").unlink()
    (repo / "tools").mkdir()
    (repo / "tools" / "new.sh").write_text("new\n", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / "ignored.pdf").write_text("ignored\n", encoding="utf-8")
    (repo / "staged.py").write_text("staged\n", encoding="utf-8")
    git(repo, "add", "staged.py")
    # Not the deleted one, which cannot be copied, nor what git ignores.
    assert container_check.changed_files(repo) == ["changed.py", "staged.py", "tools/new.sh"]


def mounts(command):
    return [command[at + 1] for at, word in enumerate(command) if word == "--mount"]


def test_the_setup_has_the_network_the_host_s_repository_to_read_and_the_volume_to_write(repo):
    command = container_check.setup_command("abc123", ["tools/new.sh"], gpu=True, repo=repo)
    assert command[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert "--network" not in command and "--read-only" not in command
    assert mounts(command) == [
        f"type=bind,source={repo},target=/host/repo,readonly",
        "type=volume,source=dsqa-workspace,target=/workspace",
    ]
    assert command[-5:] == [
        "dsqa-bare",
        "bash",
        "/host/repo/tools/container/setup.sh",
        "abc123",
        "tools/new.sh",
    ]


def test_the_check_has_no_network_and_nothing_to_write_on_but_tmp_and_the_answers(tmp_path):
    command = container_check.check_command(dirty=False, gpu=True, out=tmp_path)
    at = command.index("--network")
    assert command[at : at + 2] == ["--network", "none"]
    assert "--read-only" in command
    assert command[command.index("--tmpfs") + 1].startswith("/tmp:rw,exec")
    # The zip the host built is read, the answers are written, nothing else.
    assert mounts(command) == [
        "type=volume,source=dsqa-workspace,target=/workspace,readonly",
        f"type=bind,source={tmp_path / 'zip'},target=/host/zip,readonly",
        f"type=bind,source={tmp_path / 'answers'},target=/out",
    ]
    assert command[-3:] == ["dsqa-bare", "bash", "tools/container/check.sh"]
    assert command[command.index("--workdir") + 1] == "/workspace/ieee-cis-dsqa"


def test_a_check_of_uncommitted_files_says_so_and_one_without_a_gpu_asks_for_none(tmp_path):
    command = container_check.check_command(dirty=True, gpu=False, out=tmp_path)
    assert command[-1] == "--allow-dirty"
    assert "--gpus" not in command
