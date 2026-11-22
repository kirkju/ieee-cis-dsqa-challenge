"""The entry in a Linux container: the nearest thing to the evaluation that
the laptop can run.

    python -m tools.container_check            # set the volume up, then check
    python -m tools.container_check --setup    # only set it up (needs the internet)
    python -m tools.container_check --check    # only the checks, on a volume set up before

The evaluation image is not public. What is known of it is the list of its
packages, that its Python is most likely 3.10, that it has no internet and
that it lacks the system library cv2 needs. So the container is a bare
Ubuntu, and two things are done in it:

    setup   With the network. The repository is cloned onto a Docker volume
            and tools/pod_bootstrap.sh is run there as on a rented pod,
            without the base model: Python 3.10, the pinned packages, the
            models the entry and the tests need. This is also the rehearsal
            of that script.
    check   With the network cut and every disk read-only, the volume
            included, but for the runner's /tmp and the folder the answers
            go to. The fast tests, then the entry from its zip with its
            embedder and a small real model on the GPU: over three papers,
            and over the whole development corpus from the zip that was
            built here on the host, the file that would be uploaded. Built
            again in the container, that zip has to be the same file.

The clone holds what is committed. Files that are changed or new in the
working tree are copied over it, and the check then says so: the one that
counts is of a clean tree.

Needs Docker, and the GPU passed through to its containers for the entry to
be run with a real model. The answers, the zip and what was printed are left
in runs/container.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from tools import build_entry, ledger

REPO_ROOT = ledger.REPO_ROOT
IMAGE = "dsqa-bare"
VOLUME = "dsqa-workspace"
CLONE = "/workspace/ieee-cis-dsqa"
OUT = REPO_ROOT / "runs" / "container"
# What the runner may fill of the container's memory with the zip it unpacks
# and the folder it runs in. Programs can be run from it, as from a disk.
TMP = "/tmp:rw,exec,nosuid,size=3g"


def _git(*args: str, repo: Path = REPO_ROOT) -> str:
    done = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True
    )
    return done.stdout


def changed_files(repo: Path = REPO_ROOT) -> list[str]:
    """The files of the working tree that are not as committed: changed, or
    new and not ignored. One that was deleted is not among them."""
    names = _git("diff", "--name-only", "-z", "HEAD", repo=repo).split("\0")
    names += _git("ls-files", "--others", "--exclude-standard", "-z", repo=repo).split("\0")
    return sorted({name for name in names if name and (repo / name).is_file()})


def _mount(kind: str, source, target: str, readonly: bool = False) -> list[str]:
    spec = f"type={kind},source={source},target={target}"
    return ["--mount", spec + (",readonly" if readonly else "")]


def setup_command(commit: str, files: list[str], gpu: bool, repo: Path = REPO_ROOT) -> list[str]:
    """The container that sets the volume up: the host's repository to clone
    from, read-only, and the volume to write."""
    command = ["docker", "run", "--rm", *(["--gpus", "all"] if gpu else [])]
    command += _mount("bind", repo, "/host/repo", readonly=True)
    command += _mount("volume", VOLUME, "/workspace")
    return [*command, IMAGE, "bash", "/host/repo/tools/container/setup.sh", commit, *files]


def check_command(dirty: bool, gpu: bool, out: Path = OUT) -> list[str]:
    """The container of the checks: no network, and nothing to write on but
    /tmp, which is memory, and the folder of the answers. The host's zip is
    in out/zip and the answers go to out/answers."""
    command = ["docker", "run", "--rm", *(["--gpus", "all"] if gpu else [])]
    command += ["--network", "none", "--read-only", "--tmpfs", TMP]
    command += _mount("volume", VOLUME, "/workspace", readonly=True)
    command += _mount("bind", out / "zip", "/host/zip", readonly=True)
    command += _mount("bind", out / "answers", "/out")
    command += ["--env", "HOME=/root", "--env", "HF_HOME=/workspace/hf"]
    command += ["--env", "PYTHONDONTWRITEBYTECODE=1", "--workdir", CLONE]
    return [
        *command,
        IMAGE,
        "bash",
        "tools/container/check.sh",
        *(["--allow-dirty"] if dirty else []),
    ]


def _run(command: list[str], log: Path) -> int:
    """Run a command, its output shown as it comes and kept in a file."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", newline="\n") as kept:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="")
            kept.write(line)
        return process.wait()


def _gpu_in_containers() -> bool:
    probe = ["docker", "run", "--rm", "--gpus", "all", IMAGE, "nvidia-smi", "-L"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the entry in a Linux container.")
    parser.add_argument("--setup", action="store_true", help="only set the volume up")
    parser.add_argument("--check", action="store_true", help="only run the checks")
    parser.add_argument("--out", type=Path, default=OUT, help="where the answers and logs go")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    commit = _git("rev-parse", "HEAD").strip()
    files = changed_files()
    state = f"commit {commit[:7]}" + (
        f" with {len(files)} files of the working tree" if files else ""
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    built = subprocess.run(
        ["docker", "build", "--quiet", "--tag", IMAGE, str(REPO_ROOT / "tools" / "container")]
    )
    if built.returncode != 0:
        print("[container] the image could not be built: is Docker running?")
        return 1
    gpu = _gpu_in_containers()
    if not gpu:
        print("[container] no GPU in containers: the entry will not be run with a real model")

    if not args.check:
        print(f"[container] setting the volume up at {state}")
        if _run(setup_command(commit, files, gpu), out / "setup.log") != 0:
            print(f"[container] the setup failed: {(out / 'setup.log').as_posix()}")
            return 1
    if not args.setup:
        (out / "answers").mkdir(exist_ok=True)
        try:
            zipped = build_entry.build(out / "zip" / "entry.zip", allow_dirty=bool(files))
        except build_entry.BuildError as error:
            print(f"[container] {error}")
            return 1
        print(f"[container] the zip built here: sha256 {zipped['sha256']}")
        print(f"[container] checking {state}, no network, read-only")
        if _run(check_command(bool(files), gpu, out), out / "check.log") != 0:
            print(f"[container] the check failed: {(out / 'check.log').as_posix()}")
            return 1
        if files:
            print("[container] passed, with uncommitted files: not the check that counts")
        else:
            print(f"[container] passed at commit {commit[:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
