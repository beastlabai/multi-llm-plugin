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

import json
import os
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


class TestFxWrapperOwnsItsSeededHome:
    @pytest.mark.parametrize("only_skill", [False, True])
    def test_login_path_cannot_restore_the_skill_venv(self, monkeypatch, tmp_path, only_skill):
        skill_env = tmp_path / "skill-venv"
        (skill_env / "bin").mkdir(parents=True)
        alias = tmp_path / "venv-alias"
        alias.symlink_to(skill_env, target_is_directory=True)
        monkeypatch.setattr(wrapper_module, "SKILL_ENVIRONMENT", skill_env)
        monkeypatch.setattr(wrapper_module, "REDIRECT_YAML", REDIRECT_YAML)
        entries = [str(skill_env / "bin"), str(alias / "bin")]
        expected = os.defpath
        if not only_skill:
            entries += ["/opt/mllm/bin", "/usr/bin", "/user/venv/bin"]
            expected = "/opt/mllm/bin:/usr/bin:/user/venv/bin"
        monkeypatch.setenv("PATH", os.pathsep.join(entries))
        monkeypatch.setenv("MULTI_LLM_FX_PATH", str(skill_env / "bin"))
        env = wrapper_module.build_child_env(
            "fx", "abc123", wrapper_module.load_row("fx"), tmp_path / "home")
        assert env["PATH"] == expected
        assert env["MULTI_LLM_FX_PATH"] == expected

    def test_bare_adapter_env_has_only_pinned_knobs_and_an_optional_key(self, monkeypatch):
        from utils.providers.fx import FxProvider

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env = FxProvider().get_env("stub-model")
        assert env == {
            "FX_PROVIDER": "openrouter", "FX_MAX_AGENT_STEPS": "60",
            "FX_AUTO_UPGRADE": "0", "FX_PERMISSION_MODE": "auto", "NO_COLOR": "1",
        }
        assert all(_is_carried(key, "fx") for key in env)

    def test_active_dispatch_cannot_replace_the_wrappers_home(self, monkeypatch, tmp_path):
        from utils.providers.fx import FxProvider

        monkeypatch.setattr(wrapper_module, "REDIRECT_YAML", REDIRECT_YAML)
        provider = FxProvider()
        row = wrapper_module.load_row("fx")
        root = tmp_path / "wrapper-home"
        monkeypatch.setenv("PATH", "/opt/mllm/bin:/usr/bin:/bin")
        monkeypatch.setenv("OPENROUTER_API_KEY", SENTINEL)
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/real-user/.gitconfig")
        with provider.hermetic_dispatch("generate", "stub-model"):
            delta = provider.get_env("stub-model")
            managed = Path(delta["HOME"])
            assert managed.is_dir()
            for name, value in delta.items():
                monkeypatch.setenv(name, value)
            env = wrapper_module.build_child_env("fx", "abc123", row, root)
            assert env["HOME"] == str(root)
            assert env["XDG_CONFIG_HOME"] == str(root / ".config")
            assert env["XDG_DATA_HOME"] == str(root / ".local/share")
            assert env["XDG_CACHE_HOME"] == str(root / ".cache")
            assert "GIT_CONFIG_GLOBAL" not in env
            assert env["OPENROUTER_API_KEY"] == "mllm-stub-abc123"
            assert env["MULTI_LLM_FX_PATH"] == "/opt/mllm/bin:/usr/bin:/bin"
            assert all(env[k] == delta[k] for k in row["env_extra"])
        assert not managed.exists()

    def test_seed_matches_production_and_becomes_private_after_substitution(self, monkeypatch, tmp_path):
        from utils.providers.fx import render_settings

        source = Path(__file__).parent / "container/home_seed/fx/.fx/settings.json"
        assert json.loads(source.read_text()) == render_settings(base_url="{{base_url}}")
        homes = tmp_path / "homes"
        settings = homes / "fx/.fx/settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(source.read_text())
        settings.parent.chmod(0o755)
        settings.chmod(0o644)
        monkeypatch.setattr(wrapper_module, "HOMES_ROOT", homes)
        url = "http://127.0.0.1:18881/s/abc123/v1"
        monkeypatch.setenv("MLLM_STUB_BASE_URL", url)
        root = tmp_path / "invocation"
        wrapper_module.seed_invocation_root("fx", root, wrapper_module.load_row("fx"))
        rendered = root / ".fx/settings.json"
        assert json.loads(rendered.read_text()) == render_settings(base_url=url)
        assert "{{" not in rendered.read_text()
        assert rendered.parent.stat().st_mode & 0o777 == 0o700
        assert rendered.stat().st_mode & 0o777 == 0o600
        assert (root / ".bash_profile").read_text() == 'export PATH="$MULTI_LLM_FX_PATH"\n'


# --------------------------------------------------------------------------- #
# T4.9 (plans/opencode-validator-false-rate-limit-and-denied-turn.md): the one
# structured value the carry-over admits, and the guard that keeps it keyless.
# --------------------------------------------------------------------------- #

from .container import wrapper as wrapper_module  # noqa: E402

#: What `OpenCodeProvider` emits on a hermetic `agent-readonly` slot with a
#: built-in provider: the permission rows under the agent layer, no key.
PERMISSION_CONTENT = json.dumps({
    "$schema": "https://opencode.ai/config.json",
    "agent": {"plan": {"permission": {
        "external_directory": {"*": "deny"},
        "read": {"*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"},
        "doom_loop": "deny"}}},
    "experimental": {"continue_loop_on_deny": True},
})
SENTINEL = "SENTINEL-not-a-real-key-7f3a"


def _child_env(monkeypatch, tmp_path, **env):
    for key in [k for k in os.environ if k.startswith("OPENCODE_")]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "inv"
    root.mkdir(exist_ok=True)
    return wrapper_module.build_child_env("opencode", "abc123", {}, root)


class TestTheOpencodeConfigContentIsCarried:
    def test_it_is_an_exact_name_carry(self):
        assert "OPENCODE_CONFIG_CONTENT" in ENV_CARRY_EXACT
        assert "OPENCODE_CONFIG_CONTENT" in wrapper_module.CARRY_GUARDED_JSON

    def test_the_permission_block_reaches_the_child_byte_identical(
            self, monkeypatch, tmp_path):
        env = _child_env(monkeypatch, tmp_path,
                         OPENCODE_CONFIG_CONTENT=PERMISSION_CONTENT)
        assert env["OPENCODE_CONFIG_CONTENT"] == PERMISSION_CONTENT

    def test_its_siblings_are_still_scrubbed(self, monkeypatch, tmp_path):
        # Only the exact name: `OPENCODE_CONFIG` / `_DIR` / `OPENCODE_PERMISSION`
        # would point the child at a config the rig does not own.
        env = _child_env(monkeypatch, tmp_path,
                         OPENCODE_CONFIG_CONTENT=PERMISSION_CONTENT,
                         OPENCODE_CONFIG="/host/opencode.json",
                         OPENCODE_CONFIG_DIR="/host/opencode",
                         OPENCODE_PERMISSION='{"*": "allow"}')
        assert not {"OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR",
                    "OPENCODE_PERMISSION"} & set(env)

    def test_absent_stays_absent(self, monkeypatch, tmp_path):
        assert "OPENCODE_CONFIG_CONTENT" not in _child_env(monkeypatch, tmp_path)


class TestTheCarriedConfigHoldsNoKey:
    @pytest.mark.parametrize("document,where", [
        ({"provider": {"stub": {"options": {"apiKey": SENTINEL}}}},
         "$.provider.stub.options.apiKey"),
        ({"apiKey": SENTINEL}, "$.apiKey"),
        ({"provider": {"x": {"options": {"api_key": SENTINEL}}}},
         "$.provider.x.options.api_key"),
        ({"mcp": [{"env": {"APIKEY": SENTINEL}}]}, "$.mcp[0].env.APIKEY"),
    ], ids=["nested", "top-level", "snake-case", "inside-a-list"])
    def test_a_key_anywhere_is_a_violation_named_by_path(self, document, where):
        reason = wrapper_module.carried_config_violation(
            "OPENCODE_CONFIG_CONTENT", json.dumps(document))
        assert reason and where in reason
        assert SENTINEL not in reason

    def test_the_shipped_shapes_pass(self):
        assert wrapper_module.carried_config_violation(
            "OPENCODE_CONFIG_CONTENT", PERMISSION_CONTENT) is None
        # A `config_providers` block (credentials excluded by design) passes.
        provider = json.dumps({"provider": {"local": {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "http://127.0.0.1:1234/v1"}}}})
        assert wrapper_module.carried_config_violation(
            "OPENCODE_CONFIG_CONTENT", provider) is None

    def test_a_key_value_that_is_merely_mentioned_is_not_a_key(self):
        # Only a key NAMED apiKey is a credential slot; the word inside a
        # permission pattern or a value (T4.7 re-lists a user's patterns as
        # keys) is not.
        doc = json.dumps({"agent": {"plan": {"permission": {
            "read": {"*apiKey*": "deny"}}}}, "note": "apiKey"})
        assert wrapper_module.carried_config_violation(
            "OPENCODE_CONFIG_CONTENT", doc) is None

    def test_unparseable_fails_closed_without_quoting_the_value(self):
        reason = wrapper_module.carried_config_violation(
            "OPENCODE_CONFIG_CONTENT", '{"apiKey": "' + SENTINEL)
        assert reason and "not valid JSON" in reason
        assert SENTINEL not in reason

    @pytest.mark.parametrize("value", [
        json.dumps({"provider": {"s": {"options": {"apiKey": SENTINEL}}}}),
        "{not json " + SENTINEL,
    ], ids=["nested-key", "unparseable"])
    def test_the_invocation_is_refused_loudly_and_leaves_nothing(
            self, monkeypatch, tmp_path, capsys, value):
        root = tmp_path / "inv"
        with pytest.raises(SystemExit) as refused:
            _child_env(monkeypatch, tmp_path, OPENCODE_CONFIG_CONTENT=value)
        assert refused.value.code == wrapper_module.CARRY_GUARD_EXIT
        err = capsys.readouterr().err
        assert "mllm-wrapper: opencode:" in err
        assert "OPENCODE_CONFIG_CONTENT" in err
        assert SENTINEL not in err
        assert not root.exists(), "the per-invocation root outlived the refusal"

    def test_main_refuses_before_the_call_record_is_written(
            self, monkeypatch, tmp_path):
        """The guard runs inside `build_child_env`, which `main` calls before
        `write_call_record`, so a key never reaches the call log's capture."""
        real = tmp_path / "real"
        real.mkdir()
        (real / "opencode").write_text("#!/bin/sh\nexit 0\n")
        log = tmp_path / "calls.jsonl"
        monkeypatch.setattr(wrapper_module, "REAL_DIR", real)
        monkeypatch.setattr(wrapper_module, "HOMES_ROOT", tmp_path / "homes")
        monkeypatch.setattr(wrapper_module, "REDIRECT_YAML",
                            tmp_path / "absent.yaml")
        monkeypatch.setattr(wrapper_module.sys, "argv", [str(real / "opencode")])
        monkeypatch.setenv("MLLM_CALL_LOG", str(log))
        monkeypatch.setenv("MLLM_INVOCATION_ROOT", str(tmp_path / "inv"))
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(
            {"provider": {"s": {"options": {"apiKey": SENTINEL}}}}))
        monkeypatch.delenv("MLLM_STUB_BASE_URL", raising=False)
        with pytest.raises(SystemExit) as refused:
            wrapper_module.main(["run", "hello"])
        assert refused.value.code == wrapper_module.CARRY_GUARD_EXIT
        assert not log.exists()


class TestTheCallLogCapturesTheBlock:
    def test_the_carried_content_is_captured(self, monkeypatch, tmp_path):
        env = _child_env(monkeypatch, tmp_path,
                         OPENCODE_CONFIG_CONTENT=PERMISSION_CONTENT)
        log = tmp_path / "calls.jsonl"
        monkeypatch.setenv("MLLM_CALL_LOG", str(log))
        wrapper_module.write_call_record(
            "opencode", ["opencode", "run"], env, "abc123", tmp_path / "inv")
        record = json.loads(log.read_text(encoding="utf-8"))
        assert record["captured_env"]["OPENCODE_CONFIG_CONTENT"] == \
            PERMISSION_CONTENT
        assert "OPENCODE_CONFIG_CONTENT" in record["child_env_names"]

    def test_the_capture_allowlist_names_it(self):
        assert "OPENCODE_CONFIG_CONTENT" in wrapper_module.ENV_CAPTURE_ALLOW
