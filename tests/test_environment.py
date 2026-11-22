"""The local virtualenv has to mirror the evaluation image.

Every check we run before uploading rests on that: the image is not public,
so requirements-participant.txt (its exact pins) is the only description of
it we have. These tests fail when the environment drifts from the pins, for
example after pulling an upstream change to that file without reinstalling.
"""

import sys
from importlib import metadata
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PINS_FILE = ROOT / "requirements-participant.txt"

# Pins that cannot be installed on a platform we develop on.
NO_WHEEL = {"win32": {"triton"}}


def _requirement_lines() -> list[str]:
    lines = PINS_FILE.read_text(encoding="utf-8").splitlines()
    return [stripped for line in lines if (stripped := line.split("#", 1)[0].strip())]


def _pins() -> list[tuple[str, str]]:
    pins = []
    for line in _requirement_lines():
        name, separator, version = line.partition("==")
        if separator:
            pins.append((name.split("[", 1)[0].strip(), version.strip()))
    return pins


PINS = _pins()


def test_every_requirement_is_an_exact_pin():
    # A line in any other form ("pkg>=1.0", an index option) would be skipped
    # by _pins() and go unchecked without anyone noticing.
    assert PINS
    assert len(PINS) == len(_requirement_lines())


def test_python_is_3_10():
    # The image's Python version is not published. 3.10 is inferred from the
    # pins: numpy, scipy, pandas, scikit-learn and networkx are each the newest
    # release that still installs on it.
    assert sys.version_info[:2] == (3, 10)


@pytest.mark.parametrize(("name", "pinned"), PINS, ids=[name for name, _ in PINS])
def test_installed_version_matches_the_pin(name, pinned):
    if name in NO_WHEEL.get(sys.platform, ()):
        pytest.skip(f"{name} has no wheel for {sys.platform}")
    installed = Version(metadata.version(name))
    # .public drops a local build label such as "+cu128" on the CUDA wheels.
    assert Version(installed.public) == Version(pinned)
