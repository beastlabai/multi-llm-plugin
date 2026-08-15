"""Shared golden-bundle machinery: normalize, diff, and load an allowlist (C8).

Two guards compare normalized run bundles now — the instruction-refactor guard
(`tests/test_instruction_refactor_guard.py`) and the container provider lane's
chain gate (`tests/test_container_full_chain.py`, capturing through
`tests/golden/container_bundle.py`) — and they must not grow two subtly
different sets of rules for what "the same run" means. So the **code** moves here:
`normalize_run`/`diff_bundles`/`load_allowlist` *and* the capture primitives both
bundles are assembled from, re-exported below under public names.

The container lane is named here as a fact about the code, not as an intention:
if that module stops importing this one, `tests/test_golden_container_bundle.py`
fails in plain CI. This docstring used to name `test_container_provider_lane.py`,
which never imported it — a claimed guard that was not wired to anything, and the
kind of thing only an assertion can keep honest.

**What is deliberately NOT shared: the baseline and the allowlist.**
`tests/golden/expected_changes.yaml` describes one specific refactor against one
pinned tag. Pointing the container lane at it would break in both directions: an
entry like `/output_root_paths` would silently excuse a real container regression
that happened to land under it, and any entry whose diff exists only relative to
that tag would fail container runs for reasons that have nothing to do with the
container. The container lane therefore gets its own baseline per CLI under
`tests/golden/container/<cli>/` and its own
`tests/golden/container/expected_changes.yaml`, which ships **empty** — the same
strongest-possible outcome the instruction guard recorded — with its own
stale-entry enforcement over its own diffs.

The allowlist carries two rules worth having, and this module enforces both:

* a diff **outside** the list fails — that is the guard;
* an entry that matches **nothing** fails too — a stale exception is an
  unmaintained blind spot, and the only way to tell one from a live one is to
  require it to still be doing work.

The container lane adds one rule of its own on top of these, in
`container_bundle.validate_entries`: its entries must name a `cli`, because its
baselines are per CLI and an unscoped entry would excuse the same pointer on the
other ten rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

# The normalizer itself stays where it was captured, so the pre-refactor baseline
# and every fresh bundle are folded by the identical implementation. Re-exported
# here so both guards import one name.
#
# Imported both ways on purpose: the container lane reaches this as a package
# member (`tests.golden.bundle_compare`), while the instruction-refactor guard
# puts `tests/golden` on `sys.path` and imports it flat — a convention that
# predates this module and that the capture CLI inside a worktree depends on.
#
# The capture primitives travel with them, under public names. The container lane
# assembles a *different* bundle (from evidence on disk rather than from a run it
# drove itself) out of the *same* pieces: fold a value, fold a path set, read the
# `.status.json` step sequence, take a structure, count the derived scalars, lift
# the marker sequence. Re-exporting them is what keeps "the same run" one
# definition instead of two implementations that agree today.
try:  # pragma: no cover - exercised by whichever import style ran
    from .instruction_refactor_bundle import (  # noqa: F401
        diff_bundles,
        normalize_run,
        _COMPARED_JSON as COMPARED_JSON,
        _derived as derived_scalars,
        _load_json as load_json,
        _markers as markers,
        _relative_paths as relative_paths,
        _result_line as result_line,
        _status_steps as status_steps,
        _structure as structure,
    )
except ImportError:  # flat import, `tests/golden` on sys.path
    from instruction_refactor_bundle import (  # noqa: F401
        diff_bundles,
        normalize_run,
        _COMPARED_JSON as COMPARED_JSON,
        _derived as derived_scalars,
        _load_json as load_json,
        _markers as markers,
        _relative_paths as relative_paths,
        _result_line as result_line,
        _status_steps as status_steps,
        _structure as structure,
    )

__all__ = [
    "diff_bundles",
    "normalize_run",
    "COMPARED_JSON",
    "derived_scalars",
    "load_json",
    "markers",
    "relative_paths",
    "result_line",
    "status_steps",
    "structure",
    "load_allowlist",
    "partition",
    "format_unexplained",
    "AllowlistError",
]


class AllowlistError(AssertionError):
    """A malformed allowlist — raised at load, never tolerated at compare time."""


def load_allowlist(path: Path) -> list[dict]:
    """Parse and validate an ``expected_changes.yaml``.

    Every entry needs a JSON-pointer and a rationale. A blanket ``/`` is refused
    outright: an undocumented allowed diff is indistinguishable from an unnoticed
    one, and an allowlist that can match everything is not an allowlist.

    An absent file is an **empty** allowlist, not an error — that is the shipping
    state of the container lane's own list, and the strongest one.
    """
    if not path.exists():
        return []
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = document.get("changes") or []
    for entry in entries:
        if not isinstance(entry, dict):
            raise AllowlistError(f"{path.name}: entry is not a mapping: {entry!r}")
        pointer = entry.get("pointer")
        if not pointer or pointer == "/":
            raise AllowlistError(
                f"{path.name}: an entry needs a JSON-pointer path, and `/` "
                f"(a blanket ignore) is not one"
            )
        if not entry.get("rationale"):
            raise AllowlistError(
                f"{path.name}: entry {pointer} carries no rationale; an "
                f"undocumented allowed diff is indistinguishable from an "
                f"unnoticed one"
            )
    return entries


def partition(diffs: Sequence[tuple], entries: Iterable[dict]):
    """Split diffs into ``(allowed, unexplained, matched_counts)``.

    ``matched_counts`` is what makes the stale-entry rule checkable: an entry
    whose count is zero matched nothing and has to go.
    """
    entries = list(entries)
    matched = {entry["pointer"]: 0 for entry in entries}
    allowed, unexplained = [], []
    for pointer, before, after in diffs:
        hit = next(
            (e["pointer"] for e in entries if pointer.startswith(e["pointer"])), None
        )
        if hit is None:
            unexplained.append((pointer, before, after))
        else:
            matched[hit] += 1
            allowed.append((pointer, before, after))
    return allowed, unexplained, matched


def format_unexplained(unexplained: Sequence[tuple], allowlist_path: Path,
                       limit: int = 40) -> str:
    """A failure message someone can act on without re-deriving the diff."""
    lines = [
        f"{len(unexplained)} unexplained difference(s). Each is either a bug or an",
        f"entry in {allowlist_path} with a rationale:",
    ]
    for pointer, before, after in unexplained[:limit]:
        lines.append(f"  {pointer}\n    baseline: {before!r}\n    fresh:    {after!r}")
    if len(unexplained) > limit:
        lines.append(f"  ... and {len(unexplained) - limit} more")
    return "\n".join(lines)
