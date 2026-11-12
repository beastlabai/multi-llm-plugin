"""Every configured model id is in its CLI's catalog, checked offline.

``is_model_valid`` compares a spec with our own ``providers.yaml``, so it cannot
see a retired id: ``cursor-agent:cursor-grok-4.7-high`` once passed the whole
suite while not existing. This module compares each id the base config names
(``providers.<name>.models``, ``defaults.*``, the ``driver`` runners) with a
committed snapshot of the catalog its CLI lists, split the way ``build_command``
splits it. It reads committed files only, so it runs in CI with no credentials;
``make refresh-model-catalogs`` (by a person, never CI) updates the snapshots.

A provider with no snapshot is not skipped silently: it is named in
``UNCHECKED`` with the reason, and a model a provider's snapshot cannot cover
is named in ``UNCHECKED_MODELS``. Both tables are checked for staleness too.
docs/development/providers.md#model-catalog-snapshots has the workflow.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pytest

from tests.model_catalogs import (
    CATALOG_DIR,
    SNAPSHOT_KEYS,
    SOURCES,
    STALE_AFTER_DAYS,
    Snapshot,
    committed_sources,
    load_snapshot,
    stale_message,
)
from tests.roster_check import assert_covers_registry
from utils.provider_registry import _PROVIDERS, load_base_config
from utils.providers.base import split_reasoning_effort


def _verbatim(model: str) -> Optional[str]:
    return model


def _backends(*names: str) -> Callable[[str], Optional[str]]:
    """The id unchanged when its first segment is one of ``names``, else None."""

    def route(model: str) -> Optional[str]:
        return model if model.split("/", 1)[0] in names else None

    return route


def _strip_backend(name: str) -> Callable[[str], Optional[str]]:
    """``<name>/<slug>`` -> ``<slug>``; any other backend is outside the source."""

    def route(model: str) -> Optional[str]:
        backend, _, rest = model.partition("/")
        return rest if backend == name and rest else None

    return route


@dataclass(frozen=True)
class Route:
    """Which catalog a provider's ids live in, and how an id maps into it."""

    source: str
    catalog_id: Callable[[str], Optional[str]]


#: provider -> where its ids are checked. ``catalog_id`` returns ``None`` for an
#: id the source cannot cover, which must then be named in UNCHECKED_MODELS.
ROUTES: Dict[str, Route] = {
    "codex": Route("codex", _verbatim),
    # The capture covers the OpenRouter and Kilo backends only; native ids are
    # in UNCHECKED_MODELS.
    "kilocode": Route("kilocode", _backends("openrouter", "kilo")),
    # `opencode models` lists only the providers logged in on the capture host
    # (Zen and OpenRouter); native ids are in UNCHECKED_MODELS.
    "opencode": Route("opencode", _backends("opencode", "openrouter")),
    "omp": Route("omp", _verbatim),
    "cline": Route("openrouter", _strip_backend("openrouter")),
    "goose": Route("openrouter", _strip_backend("openrouter")),
    "aider": Route("openrouter", _strip_backend("openrouter")),
    "pi": Route("openrouter", _strip_backend("openrouter")),
    "kimi": Route("openrouter", _verbatim),
    "qwen-code": Route("openrouter", _verbatim),
    "copilot": Route("openrouter", _verbatim),
    "fx": Route("openrouter", _verbatim),
}

_LOGGED_IN = "needs a logged-in capture"

#: provider -> why its ids are not checked. Every registered adapter is either
#: here or routed to a committed snapshot, never both and never neither.
UNCHECKED: Dict[str, str] = {
    "claude-code": "no listing command; aliases (`sonnet`, `opus`) resolve server-side",
    "gemini": "no listing command in gemini-cli",
    "muse": "no listing command, and `muse model-profile show <id>` accepts any id",
    "cursor-agent": f"{_LOGGED_IN}: `cursor-agent models` exits 1 without a login, "
                    "and its output format is not captured here, so it has no parser",
    "agy": f"{_LOGGED_IN}: `agy models` needs the Google sign-in, and its output "
           "format is not captured here, so it has no parser",
    "grok": f"{_LOGGED_IN}: `grok models`, and its output format is not captured "
            "here, so it has no parser",
    "vibe": "entries are aliases in $VIBE_HOME/config.toml, not catalog ids",
}

#: (provider, model id with its effort split off) -> why that one id is not
#: checked although its provider is.
UNCHECKED_MODELS: Dict[Tuple[str, str], str] = {
    ("kilocode", "openai/gpt-6-astra"):
        "native openai backend, not enabled on the capture host",
    ("kilocode", "google/gemini-4-carbon"):
        "native google backend, not enabled on the capture host",
    ("opencode", "google/gemini-4-carbon"):
        "native google provider, not logged in on the capture host",
    ("opencode", "openai/gpt-6.1-sol"):
        "native openai provider, not logged in on the capture host",
    ("opencode", "openai/gpt-6-astra"):
        "native openai provider, not logged in on the capture host",
}

#: Adapters whose ``build_command`` splits a spec on the ladder's
#: ``accepted_inbound`` (inbound-only aliases included) rather than
#: ``accepted``. ``_split`` uses ``accepted_inbound`` for every adapter, which is
#: the same set wherever a ladder declares no alias;
#: ``test_the_split_matches_build_command`` keeps that true.
INBOUND_SPLITTERS = frozenset({"codex"})


def _ladder(name: str):
    module = sys.modules[type(_PROVIDERS[name]).__module__]
    return getattr(module, "EFFORT_LADDER", None)


def _split(name: str, model: str) -> str:
    """``model`` with its ``:effort`` split off, as ``build_command`` does."""
    ladder = _ladder(name)
    efforts = ladder.accepted_inbound if ladder is not None else frozenset()
    return split_reasoning_effort(model, efforts)[0]


def configured_models() -> Dict[str, List[str]]:
    """Every model id the BASE config names, per provider, effort split off.

    The base layer, never ``load_config``: a developer's own
    ``.multi-llm/providers.yaml`` must not decide whether this passes.
    """
    base = load_base_config()
    out: Dict[str, List[str]] = {}

    def add(provider: str, model: str) -> None:
        if provider in _PROVIDERS:
            split = _split(provider, model)
            if split not in out.setdefault(provider, []):
                out[provider].append(split)

    for provider, block in (base.get("providers") or {}).items():
        for model in (block or {}).get("models") or []:
            add(provider, str(model))
    defaults = base.get("defaults") or {}
    specs = list(defaults.get("models") or []) + list(defaults.get("quick_models") or [])
    for mode in (defaults.get("modes") or {}).values():
        mode = mode or {}
        specs += list(mode.get("models") or []) + list(mode.get("quick") or [])
    driver = base.get("driver") or {}
    specs += [driver.get("subagent_runner"), driver.get("subagent_runner_cheap")]
    default_provider = str(base.get("default_provider") or "")
    for spec in specs:
        if not isinstance(spec, str) or not spec:
            continue
        provider, sep, model = spec.partition(":")
        if not sep:
            provider, model = default_provider, spec
        add(provider, model)
    return out


def _checked() -> List[str]:
    return sorted(name for name in ROUTES if name not in UNCHECKED)


_SNAPSHOTS: Dict[str, Optional[Snapshot]] = {}


def _snapshot(source: str) -> Optional[Snapshot]:
    if source not in _SNAPSHOTS:
        _SNAPSHOTS[source] = load_snapshot(source)
    return _SNAPSHOTS[source]


class TestEveryAdapterIsCheckedOrNamed:
    def test_the_roster_is_covered(self):
        assert_covers_registry(
            set(_checked()) | set(UNCHECKED),
            "ROUTES (with a committed snapshot) or UNCHECKED in "
            "tests/test_model_catalogs.py",
            what="a catalog route or an UNCHECKED reason",
            hint="Route a CLI that lists its models to a snapshot source "
                 "(tests/model_catalogs.py SOURCES); name any other in UNCHECKED "
                 "with the reason.",
        )

    def test_routes_name_known_sources(self):
        known = set(SOURCES) | {"kilocode"}
        unknown = {name: r.source for name, r in ROUTES.items() if r.source not in known}
        assert not unknown, f"ROUTES name sources with no definition: {unknown}"

    @pytest.mark.parametrize("name", _checked())
    def test_a_checked_provider_has_its_snapshot(self, name):
        source = ROUTES[name].source
        assert _snapshot(source) is not None, (
            f"{name} is checked against {source!r}, which has no committed "
            f"snapshot. Run `make refresh-model-catalogs SOURCES={source}` or put "
            f"{name} back in UNCHECKED with the reason."
        )

    @pytest.mark.parametrize("name", sorted(UNCHECKED))
    def test_an_unchecked_provider_has_no_snapshot_waiting(self, name):
        route = ROUTES.get(name)
        if route is None:
            pytest.skip(f"{name} has no catalog source")
        assert _snapshot(route.source) is None, (
            f"{route.source!r} now has a committed snapshot, so {name} can be "
            f"checked: delete UNCHECKED[{name!r}] and fix any id the test then names."
        )


class TestSnapshotFiles:
    @pytest.mark.parametrize("path", sorted(CATALOG_DIR.glob("*.json")), ids=lambda p: p.stem)
    def test_the_file_is_well_formed(self, path):
        assert path.stem in SOURCES, (
            f"{path.name} names no source in tests/model_catalogs.py SOURCES"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == SNAPSHOT_KEYS, f"{path.name}: keys {sorted(data)}"
        _dt.date.fromisoformat(data["captured"])
        assert isinstance(data["cli_version"], str) and data["cli_version"]
        assert isinstance(data["command"], str) and data["command"]
        ids = data["ids"]
        assert ids and all(isinstance(i, str) and i for i in ids), path.name
        assert ids == sorted(set(ids)), f"{path.name}: ids must be sorted and unique"

    def test_every_committed_snapshot_loads(self):
        for source in committed_sources():
            snapshot = load_snapshot(source)
            assert snapshot is not None and snapshot.ids, source


class TestConfiguredIdsAreInTheCatalog:
    @pytest.mark.parametrize("name", _checked())
    def test_every_configured_id_is_listed(self, name):
        route = ROUTES[name]
        snapshot = _snapshot(route.source)
        if snapshot is None:
            pytest.fail(f"no snapshot for {route.source!r}")
        missing, uncovered = [], []
        for model in configured_models().get(name, []):
            catalog_id = route.catalog_id(model)
            if catalog_id is None:
                if (name, model) not in UNCHECKED_MODELS:
                    uncovered.append(model)
            elif catalog_id not in snapshot.ids:
                missing.append(f"{model} (as {catalog_id!r})" if catalog_id != model else model)
        assert not uncovered, (
            f"{name}: {uncovered} fall outside what {route.source!r} covers. Name "
            f"each in UNCHECKED_MODELS with the reason."
        )
        assert not missing, (
            f"{name}: configured ids absent from the {route.source!r} catalog "
            f"(captured {snapshot.captured}, {snapshot.cli_version}): {missing}. "
            f"Either the id is retired (fix providers.yaml) or the snapshot is old "
            f"(refresh: {snapshot.command})."
        )

    @pytest.mark.parametrize(
        "key", sorted(UNCHECKED_MODELS), ids=lambda k: f"{k[0]}:{k[1]}"
    )
    def test_an_unchecked_model_is_still_configured_and_uncovered(self, key):
        name, model = key
        assert name in _checked(), f"{name} is not a checked provider"
        assert model in configured_models().get(name, []), (
            f"{name}:{model} is no longer configured; delete its UNCHECKED_MODELS row"
        )
        assert ROUTES[name].catalog_id(model) is None, (
            f"{name}:{model} is inside {ROUTES[name].source!r}'s coverage now; "
            f"delete its UNCHECKED_MODELS row"
        )

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_the_split_matches_build_command(self, name):
        """``_split`` uses ``accepted_inbound``; only INBOUND_SPLITTERS may differ."""
        ladder = _ladder(name)
        if ladder is None or name in INBOUND_SPLITTERS:
            return
        assert ladder.accepted_inbound == ladder.accepted, (
            f"{name} declares inbound-only effort aliases. Check which set its "
            f"build_command splits on and add it to INBOUND_SPLITTERS if it is "
            f"accepted_inbound."
        )

    def test_the_split_drops_a_suffix_build_command_drops(self):
        """codex takes `ultra` on a spec and sends `max`; both are split off."""
        assert _split("codex", "gpt-6.1-sol:xhigh") == "gpt-6.1-sol"
        assert _split("codex", "gpt-6.1-sol:ultra") == "gpt-6.1-sol"
        argv = _PROVIDERS["codex"].build_command("p", "gpt-6.1-sol:ultra")
        assert argv[argv.index("--model") + 1] == "gpt-6.1-sol"


class TestStaleness:
    def test_an_old_snapshot_warns_and_never_fails(self):
        """Over STALE_AFTER_DAYS is a warning in the run's summary, not a red test."""
        today = _dt.date.today()
        for source in committed_sources():
            message = stale_message(load_snapshot(source), today)
            if message:
                warnings.warn(message, UserWarning, stacklevel=1)

    def test_the_threshold(self, tmp_path):
        captured = _dt.date(2026, 1, 1)
        snapshot = Snapshot("codex", captured, "v", "c", frozenset({"x"}), tmp_path)
        edge = captured + _dt.timedelta(days=STALE_AFTER_DAYS)
        assert stale_message(snapshot, edge) is None
        message = stale_message(snapshot, edge + _dt.timedelta(days=1))
        assert message and "SOURCES=codex" in message
        assert f"{STALE_AFTER_DAYS + 1} days" in message
