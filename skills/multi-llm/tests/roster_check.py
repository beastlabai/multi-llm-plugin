"""One failure message for every table that must hold a row per registered adapter.

The registry (``utils.provider_registry._PROVIDERS``, discovered from the modules
in ``utils/providers/``) is the adapter roster. A table that has to stay
hand-kept -- a measured config row, a pinned argv, a fixture, a docs table -- is
compared with it through :func:`assert_covers_registry`, a per-adapter lookup
goes through :func:`row_for`, and a pinned subset of the roster (the read-only
adapters) through :func:`assert_pinned_subset`. Either way a new adapter's first
failures name the table, the file it lives in, and the rows to add or remove,
and point at the recipe: docs/development/providers.md#adding-a-provider-adapter.

Imported by test modules; not a test module itself.
"""

from typing import Any, Iterable, List, Mapping, Optional

import pytest

#: Where the add-an-adapter steps are written down, relative to the repo root.
RECIPE = "docs/development/providers.md#adding-a-provider-adapter"


def registered_names() -> List[str]:
    """Every registered provider name, sorted (the order ``_PROVIDERS`` keeps)."""
    from utils.provider_registry import _PROVIDERS

    return sorted(_PROVIDERS)


def roster_mismatch(
    keys: Iterable[str],
    where: str,
    *,
    what: str = "a row",
    hint: str = "",
    registry: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """The failure message for ``keys`` against the registry, or ``None``.

    ``where`` names the table and its file (``"MODELS in
    tests/test_provider_housekeeping.py"``); ``hint`` says what a correct row
    holds. ``registry`` replaces the roster for a table that covers a known
    subset (an ordered docs list compared with ``providers.yaml``).
    """
    present = set(keys)
    expected = set(registered_names() if registry is None else registry)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if not missing and not extra:
        return None
    lines = [
        f"{where} must hold {what} for each registered adapter, and only for those "
        f"(the registry is utils/provider_registry._PROVIDERS, discovered from "
        f"utils/providers/)."
    ]
    if missing:
        lines.append(f"  add {what} for: {', '.join(missing)}")
    if extra:
        lines.append(f"  remove {what} for: {', '.join(extra)} (not a registered adapter)")
    if hint:
        lines.append(f"  {hint}")
    lines.append(f"  recipe: {RECIPE}")
    return "\n".join(lines)


def assert_covers_registry(
    keys: Iterable[str],
    where: str,
    *,
    what: str = "a row",
    hint: str = "",
    registry: Optional[Iterable[str]] = None,
) -> None:
    """Fail naming the rows to add or remove unless ``keys`` is the roster."""
    message = roster_mismatch(keys, where, what=what, hint=hint, registry=registry)
    if message is not None:
        pytest.fail(message)


def row_for(
    table: Mapping[str, Any],
    name: str,
    where: str,
    *,
    what: str = "a row",
    hint: str = "",
) -> Any:
    """``table[name]``, or a failure naming the table to add the row to.

    A per-adapter sweep that indexes a hand-kept table would otherwise fail on
    a bare ``KeyError`` for an adapter registered before its row was written.
    """
    try:
        return table[name]
    except KeyError:
        pass
    lines = [f"{where} is missing {what} for the registered adapter {name!r}: add it."]
    if hint:
        lines.append(f"  {hint}")
    lines.append(f"  recipe: {RECIPE}")
    pytest.fail("\n".join(lines))


def assert_pinned_subset(
    pinned: Iterable[str],
    derived: Iterable[str],
    where: str,
    *,
    members: str,
    hint: str = "",
) -> None:
    """Fail naming the rows to add or remove unless ``pinned`` is ``derived``.

    For a table that pins part of the roster: ``derived`` is the set the code
    produces, ``members`` says who belongs (``"the adapters that declare
    supports_file_output = False"``), and ``hint`` says what else a new member
    needs.
    """
    pinned, derived = set(pinned), set(derived)
    if pinned == derived:
        return
    lines = [f"{where} must list exactly {members}."]
    missing = sorted(derived - pinned)
    extra = sorted(pinned - derived)
    if missing:
        lines.append(f"  add: {', '.join(missing)}")
    if extra:
        lines.append(f"  remove: {', '.join(extra)}")
    if hint:
        lines.append(f"  {hint}")
    lines.append(f"  recipe: {RECIPE}")
    pytest.fail("\n".join(lines))
