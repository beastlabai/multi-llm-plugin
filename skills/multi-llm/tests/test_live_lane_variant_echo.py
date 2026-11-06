#!/usr/bin/env python3
"""The live lane's a2 grade drops kilocode's consumed `:variant`, and only that.

providers.yaml's kilocode block: a trailing `:variant` from the whitelist (none,
minimal, low, medium, high, xhigh, max, thinking, instant) becomes
`run --variant <name>` and is not part of the model name; anything else
(`:free`) is. The 2026-11-06 live row pinned `openrouter/moonshotai/kimi-k3:max`
and kilocode's banner said `> code · moonshotai/kimi-k3`, so a2 asked the CLI
to echo a string nothing on the wire carried. These pin the pure helper a2 now
calls (`harness.model_echo_without_consumed_variant`) — zero spend, no Docker.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.container import harness  # noqa: E402
from utils.providers.kilocode import KiloCodeProvider  # noqa: E402

PROVIDERS_YAML = Path(__file__).parent.parent / "providers.yaml"

#: The whitelist exactly as providers.yaml's kilocode comment states it.
DOCUMENTED_VARIANTS = ("none", "minimal", "low", "medium", "high", "xhigh",
                       "max", "thinking", "instant")

#: The banner the 2026-11-06 live row printed (kilocode 7.8.27).
MEASURED_BANNER = "\x1b[0m\n> code · moonshotai/kimi-k3\n\x1b[0m\n"


def _strip(cli, spec):
    return harness.model_echo_without_consumed_variant(cli, spec)


def test_kilocode_max_is_stripped():
    assert _strip("kilocode", "openrouter/moonshotai/kimi-k3:max") == \
        "openrouter/moonshotai/kimi-k3"


@pytest.mark.parametrize("variant", DOCUMENTED_VARIANTS)
def test_every_documented_kilocode_variant_is_stripped(variant):
    assert _strip("kilocode", f"openrouter/z-ai/glm-5.3:{variant}") == \
        "openrouter/z-ai/glm-5.3"


@pytest.mark.parametrize("spec", [
    "openrouter/moonshotai/kimi-k3:free",
    "openrouter/moonshotai/kimi-k3:turbo",
    "openrouter/moonshotai/kimi-k3",
])
def test_kilocode_keeps_a_suffix_outside_the_whitelist(spec):
    assert _strip("kilocode", spec) == spec


@pytest.mark.parametrize("cli,spec", [
    ("grok", "grok-4.6:high"),
    ("cline", "openrouter/z-ai/glm-5.3:max"),
    ("opencode", "openrouter/deepseek/deepseek-v4.1-flash:high"),
    ("codex", "gpt-6-luna:high"),
    ("omp", "openrouter/openai/gpt-6-luna:thinking"),
])
def test_other_clis_keep_their_pinned_slug(cli, spec):
    assert _strip(cli, spec) == spec


def test_only_kilocode_drops_a_variant():
    assert set(harness.MODEL_ECHO_DROPS_VARIANT) == {"kilocode"}


def test_the_grade_strips_exactly_what_the_argv_drops():
    """Every shipped kilocode spec: a2's expectation equals the `-m` value."""
    config = yaml.safe_load(PROVIDERS_YAML.read_text(encoding="utf-8"))
    specs = config["providers"]["kilocode"]["models"]
    assert any(":" in spec for spec in specs), "no variant spec left to grade"
    provider = KiloCodeProvider()
    for spec in specs:
        argv = provider.build_command("p", spec)
        assert _strip("kilocode", spec) == argv[argv.index("-m") + 1], spec


def test_the_measured_banner_passes_and_a_substitution_still_fails():
    """a2's expectation for the live row (route and variant dropped)."""
    expected = _strip("kilocode", "openrouter/moonshotai/kimi-k3:max").partition("/")[2]
    assert expected == "moonshotai/kimi-k3"
    assert expected in MEASURED_BANNER
    assert expected not in MEASURED_BANNER.replace("kimi-k3", "kimi-k2.7-code")
