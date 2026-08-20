#!/usr/bin/env python3
"""Every provider a default run dispatches must reach some behavioural lane.

The cursor `--mode ask` regression shipped in the **default_provider**, in
`defaults.models` and in `quick_models` — the three places a user reaches
without typing a provider name — and no lane in this repository would have
caught it, because `cursor-agent` had no container row, no golden bundle, no
`expected_argv.yaml` entry and no `cli-surface.yaml` entry. That absence was
recorded honestly in `redirect.yaml`, but nothing *asserted* on it, so "the
most-used provider is uncovered" was a fact you had to go looking for.

This file makes it a failing test instead. Coverage is deliberately defined
loosely — *some* lane that observes behaviour, not a particular one — because
the point is to notice a provider nobody can see, not to dictate how it is seen:

* `lane: provider` — the real binary runs in the image against the stub;
* `lane: fake` — the binary cannot be redirected at all, so `mock_llm.py` is its
  only possible coverage, and it must actually have an identity there;
* the credentialed live lane — the real binary against the real vendor endpoint.

A provider in `defaults.models` that satisfies none of those fails here, naming
itself, so the choice is explicit: cover it, or stop shipping it as a default.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.container import harness  # noqa: E402
from tests.harness.skill_runner import SkillRunner  # noqa: E402

SKILL_DIR = Path(__file__).parent.parent
PROVIDERS_YAML = SKILL_DIR / "providers.yaml"
REDIRECT = harness.HERE / "redirect.yaml"


def _config():
    return yaml.safe_load(PROVIDERS_YAML.read_text(encoding="utf-8"))


def _default_specs():
    """Every model spec a user reaches without naming a provider."""
    defaults = _config().get("defaults", {}) or {}
    specs = list(defaults.get("models") or [])
    specs += list(defaults.get("quick_models") or [])
    for mode in (defaults.get("modes") or {}).values():
        specs += list((mode or {}).get("models") or [])
        specs += list((mode or {}).get("quick") or [])
    return specs


def _provider_of(spec):
    """The provider half of a spec, or the configured default for a bare one."""
    head = str(spec).split(":", 1)[0]
    known = set(_config().get("providers", {}) or {})
    return head if head in known else _config().get("default_provider")


def _redirect_rows():
    doc = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    rows = doc.get("clis") if isinstance(doc, dict) else doc
    return {row["cli"]: row for row in rows if isinstance(row, dict) and "cli" in row}


def _live_lane_clis():
    from tests.test_container_live_provider import LANE_CLIS

    return set(LANE_CLIS)


DEFAULT_PROVIDERS = sorted({_provider_of(spec) for spec in _default_specs()})


def test_the_default_roster_is_not_empty():
    """A guard on the guard: an empty roster would pass everything below."""
    assert DEFAULT_PROVIDERS, "no default models parsed out of providers.yaml"


@pytest.mark.parametrize("provider", DEFAULT_PROVIDERS)
def test_every_default_provider_has_a_redirect_row(provider):
    """Even 'uncovered' is a row: the reason has to be written down somewhere."""
    rows = _redirect_rows()
    assert provider in rows, (
        f"{provider} is dispatched by a default run but has no row in "
        f"redirect.yaml — not even one recording why it cannot be covered."
    )


@pytest.mark.parametrize("provider", DEFAULT_PROVIDERS)
def test_every_default_provider_reaches_a_behavioural_lane(provider):
    row = _redirect_rows()[provider]
    lane = row.get("lane")

    if lane == "provider":
        return  # real binary, stub backend, offline and free

    if provider in _live_lane_clis():
        return  # real binary, real endpoint, credentialed

    assert lane == "fake" and provider in SkillRunner.PROVIDERS, (
        f"{provider} is dispatched by a default run, its redirect row is "
        f"lane={lane!r}, it is not on the live lane, and it has no fake-binary "
        f"identity — so nothing anywhere observes what it does. Give it a lane "
        f"or take it out of defaults.models/quick_models."
    )


def test_the_default_provider_itself_is_covered():
    """Named separately because it is the one every bare run uses."""
    default = _config().get("default_provider")
    row = _redirect_rows().get(default)

    assert row is not None, f"default_provider {default!r} has no redirect row"
    assert (
        row.get("lane") == "provider"
        or default in _live_lane_clis()
        or (row.get("lane") == "fake" and default in SkillRunner.PROVIDERS)
    ), f"the default_provider ({default}) reaches no behavioural lane"
