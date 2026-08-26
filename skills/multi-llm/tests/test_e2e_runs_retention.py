"""Retention for `tests/container/runs/`, checked without a Docker daemon.

`runs/` is gitignored, and for a long time that was treated as the whole answer:
ephemeral, prunable, absent from a fresh clone, so nobody capped it. **Gitignored
is not absent.** Nothing that copies a *working tree* consults `.gitignore` —
`codex plugin add` against a local-path marketplace and a local-path plugin
install both take the directory as it stands — so an uncapped `runs/` is a
distribution problem before it is ever a disk one. It reached 1.5 GB in this
checkout against 18 MB of tracked files, and the advertised install would have
copied every byte.

`harness.prune_runs` is the cap. This module is the guard on it, and it is
deliberately **not** named `test_container_*`: that glob is what the build context
ships into the image and what the marker-gating check enumerates, and this is a
host-side sweep with no container in it.

The two properties worth a test are the two a careless rewrite would lose: the
sweep touches only things shaped like bundles, and it orders by the UTC stamp in
the *name* rather than by mtime — which a copy, a restore or a `chmod -R`
rewrites.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR.parent))

from tests.container.harness import RUNS_KEEP_DEFAULT, prune_runs  # noqa: E402


def _bundle(root: Path, name: str, *, mtime: float | None = None) -> Path:
    path = root / name
    (path / "row-id").mkdir(parents=True)
    (path / "row-id" / "stub.jsonl").write_text("{}\n", encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


def test_keeps_the_newest_and_deletes_the_rest(runs: Path) -> None:
    names = [f"2026081{d}T120000Z" for d in range(5)]
    for name in names:
        _bundle(runs, name)

    removed = prune_runs(runs, keep=2)

    assert sorted(p.name for p in removed) == names[:3]
    assert sorted(p.name for p in runs.iterdir()) == names[3:]


def test_leaves_everything_that_is_not_a_bundle(runs: Path) -> None:
    """`README.md`, `lock-update/` and `surface/` live in `runs/` too.

    `make e2e-clean` is `rm -rf runs/*/` and takes those with it. This sweep is
    keyed off the bundle *shape*, which is the difference.
    """
    _bundle(runs, "20260814T120000Z")
    _bundle(runs, "20260815T120000Z")
    (runs / "README.md").write_text("# not a bundle\n", encoding="utf-8")
    (runs / "lock-update").mkdir()
    (runs / "surface").mkdir()

    prune_runs(runs, keep=1)

    survivors = sorted(p.name for p in runs.iterdir())
    assert survivors == ["20260815T120000Z", "README.md", "lock-update", "surface"]


def test_a_suffixed_bundle_is_still_a_bundle(runs: Path) -> None:
    """Real bundles carry probe suffixes: `20260825T165622Z-h2probe`."""
    _bundle(runs, "20260825T165622Z-h2probe")
    _bundle(runs, "20260825T165633Z-h2probe")

    removed = prune_runs(runs, keep=1)

    assert [p.name for p in removed] == ["20260825T165622Z-h2probe"]


def test_ordering_is_the_name_not_the_mtime(runs: Path) -> None:
    """A restored or copied tree has mtimes that say nothing about chronology.

    The oldest bundle is given the newest mtime here. Sorting by mtime would keep
    it and delete the run someone actually just made.
    """
    _bundle(runs, "20260814T120000Z", mtime=2_000_000_000)
    _bundle(runs, "20260825T120000Z", mtime=1_000_000_000)

    removed = prune_runs(runs, keep=1)

    assert [p.name for p in removed] == ["20260814T120000Z"]
    assert [p.name for p in runs.iterdir()] == ["20260825T120000Z"]


def test_protect_survives_however_old_it_sorts(runs: Path) -> None:
    """The sweep runs *during* a row, so the bundle being written is exempt.

    Without this, a row whose stamp sorts oldest — a clock skew, a replayed
    stamp, an empty `runs/` and `keep` of 0 — would delete its own evidence
    directory out from under the container about to write into it.
    """
    current = _bundle(runs, "20260101T000000Z")
    _bundle(runs, "20260825T120000Z")

    removed = prune_runs(runs, keep=1, protect=current)

    assert removed == []
    assert sorted(p.name for p in runs.iterdir()) == [
        "20260101T000000Z", "20260825T120000Z"]


def test_keep_zero_disables_the_sweep(runs: Path) -> None:
    _bundle(runs, "20260814T120000Z")

    assert prune_runs(runs, keep=0) == []
    assert [p.name for p in runs.iterdir()] == ["20260814T120000Z"]


def test_the_environment_overrides_the_default(runs: Path, monkeypatch) -> None:
    for day in range(4):
        _bundle(runs, f"2026081{day}T120000Z")
    monkeypatch.setenv("MLLM_E2E_RUNS_KEEP", "1")

    removed = prune_runs(runs)

    assert len(removed) == 3


@pytest.mark.parametrize("raw", ["", "   ", "eight", "-"])
def test_an_unreadable_override_falls_back_to_the_default(
        runs: Path, monkeypatch, raw: str) -> None:
    """A typo'd env var must not silently mean "keep nothing" or "keep all"."""
    for day in range(RUNS_KEEP_DEFAULT + 2):
        _bundle(runs, f"202608{day:02d}T120000Z")
    monkeypatch.setenv("MLLM_E2E_RUNS_KEEP", raw)

    removed = prune_runs(runs)

    assert len(removed) == 2
    assert len(list(runs.iterdir())) == RUNS_KEEP_DEFAULT


def test_a_missing_runs_directory_is_not_an_error(tmp_path: Path) -> None:
    assert prune_runs(tmp_path / "nope") == []
