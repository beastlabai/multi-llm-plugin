"""The PATH wrapper must not scrub away what the product's own adapter sets.

Plain CI, no Docker, no ``container`` marker: this is a cross-check between two
tables in this repository, and it needs no image to run.

**The bug this exists to prevent, which already happened once.**
``tests/container/wrapper.py`` builds each child's environment *from scratch* and
carries over an allowlist — that scrub is a requirement of the host lane, where a
provider subprocess would otherwise inherit a live ``OPENROUTER_API_KEY`` on an
open network. But an adapter's ``get_env()`` is the product deliberately shaping
its CLI's behaviour, and the scrub was silently dropping it:

``AiderProvider.get_env()`` sets ``COLUMNS=10000`` to stop aider hard-wrapping its
output. Scrubbed, aider wrapped at ~78 columns, and the JSON document it printed
came back with **raw newlines inside its string values** — invalid JSON. Three
layers up that surfaced as ``No JSON found in aider output``, which reads like a
parser bug and is nothing of the kind. The container was testing a configuration
the product never ships.

So the rule is: **anything an adapter declares through ``get_env()`` must survive
the wrapper.** This test is what makes that a property of the code rather than a
comment, in the same declaration-vs-implementation spirit as
``tests/test_providers.py``'s profile cross-checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.provider_registry import get_provider, load_base_config  # noqa: E402

from .container.wrapper import ENV_CARRY_EXACT, ENV_CARRY_PREFIX  # noqa: E402

REDIRECT_YAML = Path(__file__).parent / "container" / "redirect.yaml"


def _first_model(config: dict, name: str) -> str:
    """A representative model spec, since ``get_env`` takes one."""
    models = (config["providers"].get(name) or {}).get("models") or []
    return str(models[0]) if models else "model"


def _adapters_with_env():
    """Every configured provider whose adapter contributes environment."""
    config = load_base_config()
    out = []
    for name in config["providers"]:
        provider = get_provider(name)
        if provider is None or not hasattr(provider, "get_env"):
            continue
        try:
            env = provider.get_env(_first_model(config, name))
        except TypeError:
            env = provider.get_env()
        if env:
            out.append((name, dict(env)))
    return out


ADAPTERS_WITH_ENV = _adapters_with_env()


def _redirect_env_extra() -> dict:
    """Per-CLI variable names ``redirect.yaml`` sets on the child itself.

    The carry-over allowlist is not the only way a variable reaches the CLI:
    ``build_child_env`` applies the row's own keys *after* the scrub, with values
    the rig owns. For a **credential-capable** variable that is the only correct
    route — carrying it over would hand every provider subprocess the host's live
    key, which is precisely what the sibling test below forbids. So this check
    has to accept it too, or the remedy the failure message offers is one that
    cannot make the test pass.

    **All four row keys the wrapper applies count, not just ``env_extra``.**
    ``build_child_env`` writes ``config_roots`` (rule 3), ``credential_vars`` and
    ``base_url_vars`` (rule 2) as well, each with a rig-owned per-invocation
    value — and for those three that is the *only* admissible route, since a base
    URL carried over from the host would point a container row at the host's
    endpoint and a carried credential is the exact leak the scrub exists to stop.
    This helper read ``env_extra`` alone until ``kimi`` became the first adapter
    whose ``get_env`` sets a base URL, a credential **and** a config root; before
    that no adapter exercised the other three keys, so the gap was invisible
    rather than absent. Reading all four is what keeps the failure message's
    advice true.

    Rows are indexed by both keys ``wrapper.load_row`` matches on (``cli`` and
    the optional ``binary`` alias), so a lookup by provider id behaves the same
    way here as it does in the wrapper.
    """
    if not REDIRECT_YAML.exists():
        return {}
    data = yaml.safe_load(REDIRECT_YAML.read_text(encoding="utf-8")) or {}
    out: dict = {}
    for row in data.get("clis") or []:
        declared = set(row.get("env_extra") or {})
        declared |= set(row.get("config_roots") or {})
        declared |= set(row.get("base_url_vars") or {})
        declared |= set(row.get("credential_vars") or [])
        for field in ("cli", "binary"):
            if row.get(field):
                out.setdefault(str(row[field]), set()).update(declared)
    return out


REDIRECT_ENV_EXTRA = _redirect_env_extra()


def _is_carried(key: str, cli: str) -> bool:
    return (
        key in ENV_CARRY_EXACT
        or key.startswith(ENV_CARRY_PREFIX)
        or key in REDIRECT_ENV_EXTRA.get(cli, ())
    )


class TestAdapterEnvironmentSurvivesTheScrub:
    def test_at_least_one_adapter_contributes_environment(self):
        """A control: if this goes empty the test below is vacuously green."""
        assert ADAPTERS_WITH_ENV, (
            "no adapter declares get_env() any more — either the mechanism was "
            "removed (delete this test) or the discovery here broke (fix it), but "
            "do not leave a guard that asserts nothing"
        )

    @pytest.mark.parametrize(
        "name,env",
        ADAPTERS_WITH_ENV,
        ids=[n for n, _ in ADAPTERS_WITH_ENV],
    )
    def test_every_declared_variable_is_carried_over(self, name, env):
        dropped = sorted(key for key in env if not _is_carried(key, name))
        assert not dropped, (
            f"{name}'s adapter sets {dropped} through get_env(), and neither the "
            f"container wrapper's carry-over allowlist nor this row's "
            f"`env_extra` puts it back — so the child never sees it. A provider "
            f"row would then run under a configuration the product never ships — "
            f"the way aider's COLUMNS=10000 went missing and its JSON came back "
            f"line-wrapped and unparseable.\n"
            f"Fix: add the variable to ENV_CARRY_EXACT in "
            f"tests/container/wrapper.py if it cannot carry a credential, or "
            f"declare it under the `{name}` row's `env_extra` in "
            f"tests/container/redirect.yaml if it can — the wrapper applies "
            f"env_extra after the scrub, with a value the rig owns, so the CLI "
            f"gets the variable without inheriting the host's."
        )


class TestTheAllowlistStaysCredentialFree:
    """The carry-over list is the one place a scrub can be widened by accident."""

    def test_no_carried_variable_looks_like_a_credential(self):
        suspicious = sorted(
            key for key in ENV_CARRY_EXACT
            if any(token in key.upper()
                   for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH",
                                 "CREDENTIAL"))
        )
        assert not suspicious, (
            f"{suspicious} would be inherited by every provider subprocess, "
            f"including on the host lane where the parent holds a live key on an "
            f"open network. The scrub exists precisely to stop that."
        )

    def test_the_prefix_allowlist_is_ours_alone(self):
        assert ENV_CARRY_PREFIX == ("MLLM_",), (
            "widening the prefix allowlist beyond this rig's own namespace makes "
            "the scrub a denylist again — a guess about which variable next "
            "month's CLI release reads its key from"
        )
