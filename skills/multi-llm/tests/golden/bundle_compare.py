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
from typing import Any, Iterable, List, Optional, Sequence, Tuple

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
        diff_bundles as _diff_positional,
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
        diff_bundles as _diff_positional,
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
    "align_record_lists",
    "align_batch_lists",
    "batch_identity",
    "record_identity",
    "ALIGNED_LIST_KEY",
    "RECORD_ID_FIELDS",
    "DISPATCH_ORDER_LEAF",
    "ORDER_LEAF",
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


# ---------------------------------------------------------------------------
# Record lists compare as multisets
#
# `_flatten` addresses a list by position, so a permuted `batches` list — the
# apply phases' dispatch order is sorted, and the sort key is code — reads as
# every leaf under it changing at once (67 leaves in this guard's own fixture
# when isolated-HIGH batches moved to the front). The only allowlist entry that
# could absorb that was a pointer on the whole list, and a prefix entry that
# wide excuses any future drift underneath it: a changed prompt template, a fix
# that moved batch, a `priority_score` that no longer sums. So the comparison,
# not the allowlist, understands what a record list is: the fresh list is
# aligned onto the baseline's slots by each record's stable identity, the
# leaves are then diffed in place (a changed leaf inside a moved record is
# still reported, at the baseline's index, where the leaf-exact entries already
# point), and the order itself is reported once, as its own leaf, so a reorder
# still needs one entry with a rationale — a narrow one.
#
# The same holds for every list whose records carry a stable id: grouped.json's
# `groups` (by `group_hash`) and their member `suggestions` (by
# `suggestion_hash`), and the apply phases' `to_apply` (by `group_id`). Group
# order is a sort whose key is code too (importance, then score), and a pointer
# on `groups/0` would excuse a changed description, membership or hash at that
# position along with the reorder.
# ---------------------------------------------------------------------------

#: The key whose list value is aligned by its members' titles. Only the apply
#: phases' `orchestrator_output.json` carries one, but the rule is by key, not
#: path, so the container lane's bundles get the same reading.
ALIGNED_LIST_KEY = "batches"
#: Keys whose list records are aligned by one id field, the same way.
RECORD_ID_FIELDS = {
    "groups": "group_hash",
    "to_apply": "group_id",
    "suggestions": "suggestion_hash",
}
#: The synthetic leaf a batch reorder is reported at, under the list's pointer.
DISPATCH_ORDER_LEAF = "@dispatch_order"
#: The synthetic leaf any other record list's reorder is reported at.
ORDER_LEAF = "@order"
#: Member lists a batch may carry, in the order they are looked for.
_BATCH_MEMBER_KEYS = ("fixes", "suggestions", "tasks")


def batch_identity(batch: Any) -> Optional[Tuple[str, ...]]:
    """The titles of a batch's members — what makes it *this* batch.

    ``None`` for a value that is not a batch (no member list), which keeps its
    position: alignment only ever moves what it can name.
    """
    if not isinstance(batch, dict):
        return None
    for key in _BATCH_MEMBER_KEYS:
        members = batch.get(key)
        if isinstance(members, list):
            return tuple(
                str(m.get("title", m.get("id", ""))) for m in members
                if isinstance(m, dict)
            )
    return None


def record_identity(key: str, record: Any) -> Any:
    """What names *record* in a list under *key*, or ``None`` (it keeps its
    position): a batch's member titles, or the record's id field
    (:data:`RECORD_ID_FIELDS`)."""
    if key == ALIGNED_LIST_KEY:
        return batch_identity(record)
    field = RECORD_ID_FIELDS.get(key)
    if field is None or not isinstance(record, dict):
        return None
    value = record.get(field)
    return value if isinstance(value, str) and value else None


def _align_list(baseline: list, fresh: list, key: str) -> Tuple[list, bool]:
    """Reorder *fresh* onto *baseline*'s slots by record identity.

    Each baseline slot takes the first unclaimed fresh record with the same
    identity; a slot nothing matches takes the next unclaimed fresh record in
    order (so a record whose *identity* changed still diffs leaf by leaf, in
    place, rather than as one absence plus one addition); anything still
    unclaimed is appended, where it reads as added. The second value says
    whether the sequence of identities differed at all (never for a list
    where nothing has one).
    """
    fresh_ids = [record_identity(key, r) for r in fresh]
    base_ids = [record_identity(key, r) for r in baseline]
    claimed = [False] * len(fresh)
    aligned: List[Any] = [None] * len(baseline)
    for slot, identity in enumerate(base_ids):
        if identity is None:
            continue
        for j, candidate in enumerate(fresh_ids):
            if not claimed[j] and candidate == identity:
                aligned[slot] = fresh[j]
                claimed[j] = True
                break
    leftovers = [j for j, taken in enumerate(claimed) if not taken]
    for slot in range(len(baseline)):
        if aligned[slot] is None and leftovers:
            aligned[slot] = fresh[leftovers.pop(0)]
    result = [r for r in aligned if r is not None]
    result.extend(fresh[j] for j in leftovers)
    named = any(i is not None for i in base_ids + fresh_ids)
    return result, named and fresh_ids != base_ids


def _pointer_token(key: Any) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def _identity_value(identity: Any) -> Any:
    return list(identity) if isinstance(identity, tuple) else identity


def align_record_lists(
    baseline: Any, fresh: Any, prefix: str = ""
) -> Tuple[Any, List[Tuple[str, Any, Any]]]:
    """Walk both bundles together; return *fresh* with every record list
    (``batches`` and the :data:`RECORD_ID_FIELDS` keys) aligned onto the
    baseline's, plus one order diff per list whose order differed
    (``(pointer, baseline_identities, fresh_identities)``, at
    ``<list>/@dispatch_order`` for batches, ``<list>/@order`` otherwise).
    """
    order_diffs: List[Tuple[str, Any, Any]] = []
    if isinstance(baseline, dict) and isinstance(fresh, dict):
        out: dict = {}
        for key, value in fresh.items():
            pointer = f"{prefix}/{_pointer_token(key)}"
            if key not in baseline:
                out[key] = value
                continue
            left = baseline[key]
            if ((key == ALIGNED_LIST_KEY or key in RECORD_ID_FIELDS)
                    and isinstance(left, list) and isinstance(value, list)):
                aligned, reordered = _align_list(left, value, key)
                if reordered:
                    leaf = DISPATCH_ORDER_LEAF if key == ALIGNED_LIST_KEY else ORDER_LEAF
                    order_diffs.append((
                        f"{pointer}/{leaf}",
                        [_identity_value(record_identity(key, r)) for r in left],
                        [_identity_value(record_identity(key, r)) for r in value],
                    ))
                value = aligned
            out[key], nested = align_record_lists(left, value, pointer)
            order_diffs.extend(nested)
        return out, order_diffs
    if isinstance(baseline, list) and isinstance(fresh, list):
        out_list = []
        for index, item in enumerate(fresh):
            if index < len(baseline):
                aligned_item, nested = align_record_lists(
                    baseline[index], item, f"{prefix}/{index}"
                )
                out_list.append(aligned_item)
                order_diffs.extend(nested)
            else:
                out_list.append(item)
        return out_list, order_diffs
    return fresh, order_diffs


#: The older name, from when only ``batches`` lists were aligned.
align_batch_lists = align_record_lists


def diff_bundles(baseline: Any, fresh: Any) -> List[Tuple[str, Any, Any]]:
    """Every difference as ``(json-pointer, baseline, fresh)``, pointer-sorted.

    Positional everywhere except a record list, which is compared as a
    multiset keyed by each record's identity (see the block comment above): a
    ``batches`` list by its members' titles, ``groups``, ``to_apply`` and
    ``suggestions`` by their id field. Its leaves are reported at the
    baseline's indices and its order once, at ``<list>/@dispatch_order``
    (batches) or ``<list>/@order``.
    """
    aligned, order_diffs = align_record_lists(baseline, fresh)
    return sorted(_diff_positional(baseline, aligned) + order_diffs)


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
