"""Comment citations: a ratchet that stops new references to unpublished plans.

A comment states the rule or the reason and points at the doc section that holds
the rest; it does not cite a design plan a clone cannot open. Root ``plans/`` is
local-only, so ``§7.2``, ``fix round 14`` and ``plans/<name>.md`` mean nothing to
a reader of the repository. The convention is written once, in
``skills/multi-llm/AGENTS.md#comments-and-docstrings``.

This test counts those three citation shapes per file across

* ``skills/multi-llm/**/*.py``, tests excluded (a test may name its fixtures),
* the root ``Makefile``, and
* ``skills/multi-llm/instructions/**``,

and compares each count with ``comment_citation_baseline.json``, committed beside
this file. A count may only go down. An increase fails
``test_no_file_exceeds_its_baseline`` naming the file and the lines; a decrease
fails ``test_the_baseline_is_tight`` until the baseline is lowered, so progress
stays won. Lower it with::

    uv run --project skills/multi-llm -- python skills/multi-llm/tests/test_comment_citations.py --write

An example path in a usage string (``plans/my-plan.md``) is not a citation:
``EXAMPLE_PLAN_PATHS`` lists, in full, the paths that appear only as examples. A
path is exempt only when it matches one of them exactly, so a design document
under an example-named directory still counts.

``--write`` only lowers counts: it refuses any increase, a new citation in a clean
file included, and writes nothing. ``--init`` creates the very first baseline and
refuses to overwrite one.

What to do when a check fails: skills/multi-llm/AGENTS.md#comments-and-docstrings.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).with_name("comment_citation_baseline.json")

_SKIPPED_PARTS = {"tests", "__pycache__", ".venv", "node_modules", ".pytest_cache"}

#: ``§12``, ``§4.2.1``; ``fix round 14``, ``Fix rounds 3 and 4``; ``plans/x.md`` and
#: ``plans/z_done/x.md``. A bare ``round 14`` is not counted: it names nothing alone.
CITATION = re.compile(
    r"§\s*\d"
    r"|[Ff]ix\s+rounds?\s+\d"
    r"|\bplans/(?P<path>[\w.\-/]*\.md)"
)

#: Whole paths (after ``plans/``) that appear as examples in usage strings, docs and
#: instructions, never as a reference to a design plan. Matched exactly: a directory
#: or filename that merely starts or ends like one of these is still a citation.
EXAMPLE_PLAN_PATHS = frozenset({
    "my-plan.md",
    "my-feature.md",
    "x.md",
    "foo.md",
    "todo/my-feature.md",
    "api/index.md",
    "ui/index.md",
    "my-feature/my-feature_implementation_summary.md",
    "my-feature/my-feature_tasks.md",
})

CONVENTION = "skills/multi-llm/AGENTS.md#comments-and-docstrings"


def citations_in(text: str) -> List[Tuple[int, str]]:
    """``[(line number, matched text)]`` for every citation in *text*."""
    found: List[Tuple[int, str]] = []
    for match in CITATION.finditer(text):
        path = match.group("path")
        if path is not None and path in EXAMPLE_PLAN_PATHS:
            continue
        found.append((text.count("\n", 0, match.start()) + 1, match.group(0)))
    return found


def _scanned_files() -> Iterator[Path]:
    for path in sorted(SKILL_ROOT.rglob("*.py")):
        if _SKIPPED_PARTS & set(path.relative_to(SKILL_ROOT).parts):
            continue
        yield path
    instructions = SKILL_ROOT / "instructions"
    if instructions.is_dir():
        for path in sorted(instructions.rglob("*")):
            if path.is_file() and path.suffix in {".md", ".tmpl", ".txt", ".yaml"}:
                yield path
    makefile = REPO_ROOT / "Makefile"
    if makefile.is_file():
        yield makefile


def current_counts() -> Dict[str, List[Tuple[int, str]]]:
    counts: Dict[str, List[Tuple[int, str]]] = {}
    for path in _scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found = citations_in(text)
        if found:
            counts[path.relative_to(REPO_ROOT).as_posix()] = found
    return counts


def load_baseline() -> Dict[str, int]:
    return dict(json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["counts"])


def increases(baseline: Dict[str, int], counts: Dict[str, List[Tuple[int, str]]]) -> List[str]:
    """One line per file whose citations exceed its allowance (0 for a file not listed)."""
    return [f"{name}: {len(found)} citations, baseline {baseline.get(name, 0)}"
            for name, found in sorted(counts.items()) if len(found) > baseline.get(name, 0)]


def _write(path: Path, counts: Dict[str, List[Tuple[int, str]]]) -> None:
    document = {
        "_comment": (
            "Plan citations (section signs, fix-round labels, plans/*.md paths) per file. "
            f"A count may only go down; see {CONVENTION}. "
            "Lower it with: python skills/multi-llm/tests/test_comment_citations.py --write"
        ),
        "counts": {name: len(found) for name, found in sorted(counts.items())},
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def write_baseline(path: Path = BASELINE_PATH,
                   counts: Dict[str, List[Tuple[int, str]]] | None = None) -> None:
    """Lower the baseline to today's counts. Refuses, writing nothing, on any increase."""
    current = current_counts() if counts is None else counts
    grown = increases(dict(json.loads(path.read_text(encoding="utf-8"))["counts"]), current)
    if grown:
        raise SystemExit(
            "refusing to raise the baseline; remove these citations instead "
            f"({CONVENTION}):\n  " + "\n  ".join(grown))
    _write(path, current)


def init_baseline(path: Path = BASELINE_PATH,
                  counts: Dict[str, List[Tuple[int, str]]] | None = None) -> None:
    """Create the first baseline from today's counts. Never overwrites an existing one."""
    if path.exists():
        raise SystemExit(f"{path} exists; use --write to lower it")
    _write(path, current_counts() if counts is None else counts)


@pytest.fixture(scope="module")
def counts() -> Dict[str, List[Tuple[int, str]]]:
    return current_counts()


def test_no_file_exceeds_its_baseline(counts):
    baseline = load_baseline()
    problems = []
    for name, found in sorted(counts.items()):
        allowed = baseline.get(name, 0)
        if len(found) > allowed:
            where = ", ".join(f"{line}:{text.strip()[:24]!r}" for line, text in found[:6])
            problems.append(
                f"{name}: {len(found)} plan citations, baseline {allowed} (lines {where})")
    assert not problems, (
        "Comments must not cite unpublished plans (§N, fix round N, plans/*.md): state the "
        "rule or reason and point at the doc section instead, and move any history worth "
        f"keeping into docs/development/. Convention: {CONVENTION}.\n  "
        + "\n  ".join(problems))


def test_the_baseline_is_tight(counts):
    """A file that got cleaner must lower its baseline, or the next edit could re-add."""
    baseline = load_baseline()
    loose = [f"{name}: baseline {allowed}, now {len(counts.get(name, []))}"
             for name, allowed in sorted(baseline.items())
             if len(counts.get(name, [])) < allowed]
    assert not loose, (
        "the baseline may only go down, and it has room it no longer needs; lower it with "
        "`python skills/multi-llm/tests/test_comment_citations.py --write`:\n  "
        + "\n  ".join(loose))


def test_every_baseline_entry_names_a_scanned_file():
    scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _scanned_files()}
    stale = sorted(name for name in load_baseline() if name not in scanned)
    assert not stale, f"baseline entries for files that are no longer scanned: {stale}"


def test_the_baseline_holds_positive_counts_only():
    bad = {name: n for name, n in load_baseline().items() if not isinstance(n, int) or n <= 0}
    assert not bad, bad


def test_the_convention_section_exists():
    text = (SKILL_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "\n## Comments and docstrings\n" in text, (
        f"the failure message points at {CONVENTION}")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see §7.2 and §4", 2),
        ("# (fix round 14)", 1),
        ("Fix rounds 3 and 4", 1),
        ("plans/review-commits.md and plans/z_done/old-plan.md", 2),
        ("round 14 alone names nothing", 0),
        ("review-code plans/my-plan.md --models x", 0),
        ("plans/api/index.md", 0),
        ("a plans/ directory and a .md file", 0),
        ("§ alone, then plans/x.md", 0),
    ],
)
def test_the_counter(text, expected):
    assert len(citations_in(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "plans/foo/security-design.md",
        "plans/my-feature-sandbox-policy.md",
        "plans/my-plan/design.md",
        "plans/x/y.md",
        "plans/todo/my-feature-v2.md",
        "plans/api/index-design.md",
        "plans/my-feature_tasks.md",
    ],
)
def test_a_design_doc_near_an_example_name_still_counts(text):
    assert len(citations_in(text)) == 1


@pytest.mark.parametrize("path", sorted(EXAMPLE_PLAN_PATHS))
def test_each_example_path_is_exempt(path):
    assert citations_in(f"plans/{path}") == []


def _found(**per_file):
    return {name.replace("_py", ".py"): [(1, "§1")] * n for name, n in per_file.items()}


def _baseline_file(tmp_path, **per_file):
    path = tmp_path / "baseline.json"
    _write(path, _found(**per_file))
    return path


def test_write_baseline_refuses_an_increase(tmp_path):
    path = _baseline_file(tmp_path, a_py=1)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match=r"a\.py: 2 citations, baseline 1"):
        write_baseline(path, _found(a_py=2))
    assert path.read_text(encoding="utf-8") == before


def test_write_baseline_refuses_a_citation_in_a_clean_file(tmp_path):
    path = _baseline_file(tmp_path, a_py=1)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match=r"new\.py: 1 citations, baseline 0"):
        write_baseline(path, _found(a_py=1, new_py=1))
    assert path.read_text(encoding="utf-8") == before


def test_write_baseline_lowers_and_drops_clean_files(tmp_path):
    path = _baseline_file(tmp_path, a_py=3, b_py=1)
    write_baseline(path, _found(a_py=2))
    assert json.loads(path.read_text(encoding="utf-8"))["counts"] == {"a.py": 2}


def test_init_baseline_creates_one_and_never_overwrites(tmp_path):
    path = tmp_path / "baseline.json"
    init_baseline(path, _found(a_py=2))
    assert json.loads(path.read_text(encoding="utf-8"))["counts"] == {"a.py": 2}
    with pytest.raises(SystemExit, match="exists"):
        init_baseline(path, _found(a_py=9))
    assert json.loads(path.read_text(encoding="utf-8"))["counts"] == {"a.py": 2}


def test_write_baseline_needs_an_existing_baseline(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_baseline(tmp_path / "missing.json", _found(a_py=1))


def test_the_counter_reports_line_numbers():
    assert citations_in("a\nb §3\nc\nfix round 9") == [(2, "§3"), (4, "fix round 9")]


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        write_baseline()
        print(f"lowered {BASELINE_PATH}")
    elif sys.argv[1:] == ["--init"]:
        init_baseline()
        print(f"created {BASELINE_PATH}")
    else:
        print(__doc__)
