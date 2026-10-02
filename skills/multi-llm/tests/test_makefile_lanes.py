#!/usr/bin/env python3
"""The Makefile's lane wiring, asserted as text — in plain CI, with no Docker.

``test_dockerfile_pins.py`` proves what the Dockerfile *says*; this proves what
the Makefile *says*. Both are static reads for the same reason: a guard that only
runs where the container lane runs is a guard that does not run.

The property that matters here is a money one. ``e2e-live-provider`` drives real
CLIs against ``https://openrouter.ai`` on the developer's own key
(harness-expansion.md §7.4), and the whole design of that lane assumes nobody
reaches it by accident. "It is opt-in" is a claim about a graph — one edge added
to ``check`` or ``e2e-container`` in a hurry and every future suite run bills
somebody. So the edge count is asserted rather than remembered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[3] / "Makefile"

#: The target every assertion below is about.
LIVE = "e2e-live-provider"

#: `target: prereq prereq` — the only construct that can create an edge. Recipe
#: lines start with a tab and are excluded by the leading-character check.
RULE = re.compile(r"^(?P<targets>[A-Za-z0-9_.\-/ ]+):(?!=)(?P<prereqs>[^=]*)$")


@pytest.fixture(scope="module")
def makefile_text() -> str:
    assert MAKEFILE.is_file(), f"no Makefile at {MAKEFILE}"
    return MAKEFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rules(makefile_text: str) -> list[tuple[list[str], list[str]]]:
    """`[(targets, prereqs)]` for every rule line, comments and recipes excluded."""
    parsed = []
    for line in makefile_text.splitlines():
        if not line or line[0] in ("\t", "#", " "):
            continue
        match = RULE.match(line)
        if not match:
            continue
        targets = match.group("targets").split()
        if targets == [".PHONY"]:
            continue
        parsed.append((targets, match.group("prereqs").split()))
    return parsed


def test_the_live_lane_target_exists(rules):
    assert any(LIVE in targets for targets, _ in rules), (
        f"{LIVE} is not defined — §7.4's lane is unreachable without it")


def test_it_is_declared_phony(makefile_text):
    phony = " ".join(
        line for line in makefile_text.splitlines() if ".PHONY" in line
        or (line.startswith(" ") and "e2e-" in line))
    assert LIVE in phony, f"{LIVE} is missing from .PHONY"


def test_no_other_goal_reaches_it(rules):
    """The assertion the target's own comment block promises.

    A credit-spending lane reachable from ``check``, ``e2e-container`` or any
    other goal is a bill nobody typed. This is the graph edge, checked.
    """
    offenders = [targets for targets, prereqs in rules
                 if LIVE in prereqs and LIVE not in targets]
    assert not offenders, (
        f"{LIVE} is a prerequisite of {offenders} — it must be reachable only "
        f"when typed directly, because it spends real credits")


def test_no_recipe_invokes_it_recursively(makefile_text):
    """`$(MAKE) e2e-live-provider` inside another recipe is the same edge.

    Prerequisites are not the only way to reach a target: a recursive `$(MAKE)`
    in some other target's recipe would do it too, and would not show up in the
    dependency graph at all.
    """
    for number, line in enumerate(makefile_text.splitlines(), start=1):
        if line.startswith("\t") and "$(MAKE)" in line and LIVE in line:
            pytest.fail(f"Makefile:{number} invokes {LIVE} recursively: {line.strip()}")


def test_it_does_not_build_the_image_implicitly(rules, makefile_text):
    """A stale image is a visible failure, never a silent rebuild.

    Rebuilding under the target would mean the credits get spent on whatever
    binary the rebuild happened to install, which is the opposite of what a
    pinned smoke is for.
    """
    for targets, prereqs in rules:
        if LIVE in targets:
            assert "e2e-image" not in prereqs, (
                f"{LIVE} must not depend on e2e-image — it would spend credits "
                f"on a freshly built, possibly different binary")

    body = makefile_text.split(f"{LIVE}:", 1)[1].split("\n\n", 1)[0]
    assert "e2e-image" not in body, (
        f"{LIVE}'s recipe must not build the image implicitly")


def test_the_recipe_sources_the_key_from_the_documented_path(makefile_text):
    """One credential path, no other (§7.4's credential-injection rule)."""
    body = makefile_text.split(f"{LIVE}:", 1)[1].split("\n\n", 1)[0]
    assert "MULTI_LLM_E2E_ENV" in body and "e2e.env" in body, (
        "the recipe must read the key from ~/.config/multi-llm/e2e.env "
        "(overridable by MULTI_LLM_E2E_ENV), never from an ambient export")
    assert "live_provider" in body, (
        "the recipe must select the lane by its marker, so a bare pytest and CI "
        "both deselect it")


def test_the_marker_is_registered():
    """An unregistered marker is a warning, not a filter — and warnings get muted."""
    pyproject = MAKEFILE.parent / "skills" / "multi-llm" / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "live_provider:" in text, (
        "pyproject.toml must register the live_provider marker, or -m selection "
        "silently matches nothing")


def test_the_host_lane_target_does_not_demand_the_key(makefile_text):
    """codex and claude-code run on the subscription; only opencode needs the key.

    Operator decision 2026-10-02. The census (`_refuse_empty_lane`) is what stops
    a run with no credential at all from passing, so the Makefile need not.
    """
    body = makefile_text.split("\ne2e-container-host: e2e-image\n", 1)[1].split(
        "\n\n", 1)[0]
    assert "exit 1" not in body
    assert "subscription" in body


def test_help_says_the_host_lane_takes_the_subscription(makefile_text):
    line = next(l for l in makefile_text.splitlines()
                if "make e2e-container-host " in l and "@echo" in l)
    assert "subscription" in line and "OPENROUTER_API_KEY" in line
