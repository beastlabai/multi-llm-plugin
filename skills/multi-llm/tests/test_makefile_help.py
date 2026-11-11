"""``make help`` is generated from the ``## description`` suffixes, so it cannot go stale.

The ``Makefile`` used to carry a hand-written ``echo`` block that omitted six
declared targets. Now every ``target: ... ## one line`` is listed under its
``##@`` heading by a short ``awk`` over ``$(MAKEFILE_LIST)``, and these tests keep
three lists in step: the ``.PHONY`` line, the targets defined, and the targets
described.

Rules: docs/development/release-gates.md.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[3] / "Makefile"

#: A rule line that names its targets and carries a description. No `=` may sit
#: after the colon on such a line: `test_makefile_lanes.py` reads rule lines with a
#: pattern that stops at one, and a description with `=` would hide the rule.
_DESCRIBED = re.compile(r"^(?P<target>[A-Za-z0-9_.\-]+):(?P<rest>[^\n]*?)## (?P<text>\S[^\n]*)$")
_RULE = re.compile(r"^(?P<target>[A-Za-z0-9_.\-]+)\s*:(?![:=])")


@pytest.fixture(scope="module")
def text() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _phony(text: str) -> set[str]:
    joined = text.replace("\\\n", " ")
    return {name for line in re.findall(r"^\.PHONY:(.*)$", joined, re.M) for name in line.split()}


def _defined(text: str) -> set[str]:
    """Targets that have a rule line (not variables, `$(error ...)` or recipe lines)."""
    names = set()
    for line in text.splitlines():
        if not line or line[0] in "\t #$":
            continue
        match = _RULE.match(line)
        if match and match.group("target") != ".PHONY":
            names.add(match.group("target"))
    return names


def _described(text: str) -> dict[str, str]:
    return {m.group("target"): m.group("text")
            for m in (_DESCRIBED.match(line) for line in text.splitlines()) if m}


#: Targets that are real files, not commands, and so need no description. None today.
FILE_TARGETS: frozenset[str] = frozenset()


def test_every_defined_target_is_described(text):
    """A target with a rule line must show in `make help`, phony or not."""
    missing = sorted(_defined(text) - set(_described(text)) - FILE_TARGETS)
    assert not missing, (
        f"targets without a `## description` on their rule line: {missing}. List a genuine "
        "file target in FILE_TARGETS instead.")


def test_every_phony_target_has_a_description(text):
    missing = sorted(_phony(text) - set(_described(text)))
    assert not missing, (
        f"targets without a `## description` on their rule line: {missing}. "
        "`make help` is generated from those suffixes (docs/development/release-gates.md).")


def test_every_described_target_is_defined_and_phony(text):
    described = set(_described(text))
    assert not described - _defined(text), sorted(described - _defined(text))
    assert not described - _phony(text), (
        f"described but not in .PHONY: {sorted(described - _phony(text))}")


def test_every_phony_target_is_defined(text):
    assert not _phony(text) - _defined(text), sorted(_phony(text) - _defined(text))


def test_a_description_has_no_equals_sign(text):
    """`test_makefile_lanes.py` parses rule lines with a pattern that stops at `=`."""
    bad = [line for line in text.splitlines()
           if _DESCRIBED.match(line) and "=" in _DESCRIBED.match(line).group("text")]
    assert not bad, bad


@pytest.mark.skipif(shutil.which("make") is None or os.name == "nt",
                    reason="runs make, which needs a POSIX shell")
def test_make_help_lists_every_described_target(text):
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
    done = subprocess.run(["make", "--no-print-directory", "help"], cwd=MAKEFILE.parent, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    listed = {m.group(1) for m in re.finditer(r"^  make (\S+)", done.stdout, re.M)}
    assert listed == set(_described(text))
    assert _phony(text) <= listed
    # each `##@` heading is printed once, in file order
    headings = [line[5:].strip() for line in text.splitlines() if line.startswith("##@ ")]
    printed = [h for h in headings if h in done.stdout]
    assert printed == headings
