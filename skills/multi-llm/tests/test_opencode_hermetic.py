"""M4 (donor §2.3) — hermetic opencode runs, and the directory lifecycle under them.

The goal: neutralize the user's own opencode config, which can otherwise silently
change the effective model, agent or permission set per machine and per
repository. The mechanism is an empty, **run-scoped** ``XDG_CONFIG_HOME`` plus
``opencode --pure``, with ``XDG_DATA_HOME`` retained so stored auth still
resolves — auth is never config-borne.

The five ship gates §2.3 names are the five classes below:

* ``TestGateAAuthRetention``        — stored auth survives the redirect
* ``TestGateBCustomProvider``       — a provider declared to *multi-llm* still
  resolves from ``OPENCODE_CONFIG_CONTENT`` alone; one declared **only** in the
  user's own opencode config is refused before spawn rather than mis-resolved
* ``TestGateCProjectConfigIsolation`` — a workspace ``opencode.json`` no longer
  changes the effective run
* ``TestGateDConcurrency``          — N concurrent slots create exactly one
  directory, share it, and none observes another's env
* ``TestGateECleanup``             — the directory is gone after normal exit, a
  slot timeout and a ``KeyboardInterrupt``, and a user-supplied
  ``XDG_CONFIG_HOME`` is never touched

Everything here is offline. The three gates whose *subject* is opencode's own
behaviour (a, b's resolution half, c) additionally carry a ``live``-marked
assertion against the real binary, deselected by ``-m "not live"``; the offline
half pins what this repo controls — the env delta and the argv — which is where
a regression would actually land.

Measured against **opencode 1.18.18**:

* ``XDG_CONFIG_HOME=<empty dir> opencode --pure debug config`` yields
  ``{"$schema": …, "agent": {}, "mode": {}, "plugin": [], "command": {},
  "username": …}`` against a 6 360-byte control;
* ``opencode auth list`` is byte-identical with and without the redirect;
* a **workspace** ``opencode.json`` survives both the redirect and ``--pure`` —
  ``OPENCODE_DISABLE_PROJECT_CONFIG=1`` is what closes that layer, so it is set
  alongside the redirect and gate (c) is about all three together;
* the directory does **not** stay empty: opencode seeds ``opencode/.gitignore``
  and an ``opencode/opencode.jsonc`` holding nothing but ``$schema``. That is
  fixed content with no per-slot state, so the one-shared-directory rule stands
  and cleanup is ``rmtree``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import llm_client  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.providers import opencode as oc  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
)

#: The version every measurement in this module was taken on.
OBSERVED_ON = "1.18.18"

#: XDG variables that must reach the child untouched. ``XDG_DATA_HOME`` is what
#: keeps ``auth.json`` reachable; ``XDG_CACHE_HOME`` holds ``models.json``, and
#: §6.3 measured a cold cache degrading *silently* (exit 0, empty stderr, a
#: truncated model list), which is why redirecting it alongside the config home
#: is the tempting symmetry and is forbidden.
PASSTHROUGH_XDG = (
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
)


@pytest.fixture(autouse=True)
def clean_hermetic_state():
    """No directory, no memoised help read and no warn-once state leaks a test."""
    oc.sweep_config_home()
    oc.reset_hermetic_state()
    yield
    oc.sweep_config_home()
    oc.reset_hermetic_state()


@pytest.fixture
def provider():
    return registry.get_provider("opencode")


@pytest.fixture
def user_xdg(tmp_path, monkeypatch):
    """A user-supplied ``XDG_CONFIG_HOME`` with a real opencode config in it.

    Every test runs against one, because "the user's own config home is
    overridden but never touched" is only meaningful when there is one.
    """
    home = tmp_path / "user-config"
    (home / "opencode").mkdir(parents=True)
    (home / "opencode" / "opencode.json").write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "model": "opencode/should-never-win",
                "provider": {
                    "only-in-user-config": {
                        "options": {"baseURL": "https://gateway.invalid/v1"},
                        "models": {"private-model": {}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    for name in PASSTHROUGH_XDG:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    return home


def _use_config(tmp_path, monkeypatch, body: str) -> Path:
    """Point the loader at one override file, the way `runner_env.use` does."""
    path = tmp_path / "hermetic-config.yaml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
    registry._config = None
    registry._config_key = None
    return path


def _child_env(provider, model: str, profile: str) -> dict:
    """The child environment a spawn site would compose, delta rules included."""
    delta = llm_client.expand_env_home(provider.resolve_env(model, profile))
    env = {**os.environ, **delta}
    for key in provider.get_remove_env() or ():
        env.pop(key, None)
    return env


def _pure_help(monkeypatch, *, supported: bool = True) -> None:
    """Pin the `--help` token sniff instead of spawning the real binary."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        oc,
        "_read_help_tokens",
        lambda binary: frozenset({oc.PURE_FLAG}) if supported else frozenset(),
    )


def _hermetic_dirs_on_disk() -> set:
    root = Path(os.environ.get("TMPDIR", "/tmp"))
    return {str(p) for p in root.glob(f"{oc.HERMETIC_DIR_PREFIX}*")}


@pytest.fixture
def new_hermetic_dirs():
    """Directories *this test* created, ignoring any other process's.

    The prefix sweep is deliberately over the shared ``TMPDIR``, so a concurrent
    orchestrator (or another pytest process) legitimately has its own directory
    there. Only the difference from this test's own baseline is this test's
    business — counting the raw glob would make the assertion depend on what
    else is running on the machine.
    """
    baseline = _hermetic_dirs_on_disk()

    def _new() -> list:
        return sorted(_hermetic_dirs_on_disk() - baseline)

    return _new


# --------------------------------------------------------------------------- #
# Gate (a) — stored auth retention
# --------------------------------------------------------------------------- #

class TestGateAAuthRetention:
    """Exactly one variable changes, so a logged-in opencode stays logged in."""

    def test_only_the_config_home_is_redirected(self, provider, user_xdg):
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet") as r:
            assert r is None
            delta = provider.resolve_env("opencode/sonnet", PROFILE_AGENT_READONLY)

        changed = sorted(key for key in delta if key.startswith("XDG_"))
        assert changed == ["XDG_CONFIG_HOME"], (
            "redirecting any XDG variable other than the config home is what "
            "breaks stored auth (XDG_DATA_HOME) or silently truncates the model "
            "catalog (XDG_CACHE_HOME)"
        )

    def test_the_retained_variables_reach_the_child_unchanged(
        self, provider, user_xdg
    ):
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)

        for name in PASSTHROUGH_XDG:
            assert env[name] == os.environ[name], f"{name} was not passed through"
        assert env["XDG_CONFIG_HOME"] != str(user_xdg)

    def test_the_users_own_config_home_is_overridden_never_deleted(
        self, provider, user_xdg
    ):
        marker = user_xdg / "opencode" / "opencode.json"
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            pass
        assert marker.is_file(), "a user-supplied XDG_CONFIG_HOME must never be removed"

    @pytest.mark.live
    def test_auth_list_is_byte_identical_under_the_run_env(self, provider):
        """The real gate: `opencode auth list` unchanged by the redirect."""
        import shutil as _shutil

        if not _shutil.which("opencode"):
            pytest.skip("opencode is not installed")
        control = subprocess.run(
            list(provider.auth_check), capture_output=True, text=True, timeout=60
        )
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)
            redirected = subprocess.run(
                list(provider.auth_check), capture_output=True, text=True,
                timeout=60, env=env,
            )
        assert redirected.stdout == control.stdout
        assert redirected.returncode == control.returncode


# --------------------------------------------------------------------------- #
# Gate (b) — custom providers
# --------------------------------------------------------------------------- #

class TestGateBCustomProvider:
    """Declared to multi-llm: it survives. Declared only to opencode: refused."""

    CONFIG = (
        "providers:\n"
        "  opencode:\n"
        "    config_providers:\n"
        "      my-gateway:\n"
        "        options:\n"
        "          baseURL: https://gateway.example.com/v1\n"
        "        models:\n"
        "          my-model: {}\n"
    )

    def test_a_declared_provider_is_carried_in_the_config_content(
        self, provider, user_xdg, tmp_path, monkeypatch
    ):
        _use_config(tmp_path, monkeypatch, self.CONFIG)
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "my-gateway/my-model") as r:
            assert r is None, "a provider declared to multi-llm must not be refused"
            env = _child_env(provider, "my-gateway/my-model", PROFILE_AGENT_READONLY)

        document = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert list(document["provider"]) == ["my-gateway"], (
            "the carrier must hold only the block the slot needs — it is a "
            "last-applied merge layer, so whatever it carries wins"
        )
        assert document["provider"]["my-gateway"]["options"]["baseURL"] == (
            "https://gateway.example.com/v1"
        )
        # The user's config home is unreachable while it resolves.
        assert env["XDG_CONFIG_HOME"] != str(user_xdg)

    def test_the_carrier_holds_no_credential(
        self, provider, user_xdg, tmp_path, monkeypatch
    ):
        """Credentials stay in auth.json under the retained XDG_DATA_HOME."""
        _use_config(tmp_path, monkeypatch, self.CONFIG)
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "my-gateway/my-model"):
            env = _child_env(provider, "my-gateway/my-model", PROFILE_AGENT_READONLY)

        blob = env["OPENCODE_CONFIG_CONTENT"].lower()
        for secret in ("apikey", "api_key", "token", "secret", "password"):
            assert secret not in blob

    def test_a_built_in_provider_carries_no_provider_block(
        self, provider, user_xdg, tmp_path, monkeypatch
    ):
        """A built-in provider needs no `provider` block.

        Before T4.1 the variable was not set at all for it. It is set now,
        because every hermetic slot carries the permission rows (§4.2 step 1),
        but it still holds no provider.
        """
        _use_config(tmp_path, monkeypatch, self.CONFIG)
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet") as r:
            assert r is None
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)

        document = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert "provider" not in document
        assert "permission" in document["agent"]["plan"]

    def test_an_inherited_config_carrier_is_stripped(
        self, provider, user_xdg, monkeypatch
    ):
        """`OPENCODE_CONFIG_CONTENT` is a merge layer applied LAST.

        An inherited one would survive — and win over — the redirect, which is
        exactly the containment this section exists to provide.
        """
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"model": "smuggled/in"}')
        monkeypatch.setenv("OPENCODE_CONFIG", "/home/someone/opencode.json")
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)

        # The value the child sees is ours (the T4.1 permission rows), never
        # the inherited one.
        assert "smuggled" not in env["OPENCODE_CONFIG_CONTENT"]
        assert "model" not in json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert "OPENCODE_CONFIG" not in env

    def test_a_provider_only_in_the_users_config_is_refused_before_spawn(
        self, provider, user_xdg, new_hermetic_dirs
    ):
        """The negative half: refused with a named error, never mis-resolved.

        A silent model-resolution failure is the one outcome the redirect may not
        produce — it does not return a worse answer, it burns the slot.
        """
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, "only-in-user-config/private-model"
        ) as refusal:
            assert refusal, "an unreconstructible provider must be refused"
            assert "only-in-user-config" in refusal
            assert "config_providers" in refusal
            assert "hermetic: off" in refusal
        # Refused BEFORE anything was claimed: no directory was created.
        assert oc._CONFIG_HOME is None
        assert new_hermetic_dirs() == []

    def test_declaring_it_to_multi_llm_is_the_whole_migration_ask(
        self, provider, user_xdg, tmp_path, monkeypatch
    ):
        """The same slug stops being refused once it is declared here."""
        _use_config(
            tmp_path, monkeypatch,
            "providers:\n"
            "  opencode:\n"
            "    config_providers:\n"
            "      only-in-user-config:\n"
            "        options:\n"
            "          baseURL: https://gateway.invalid/v1\n"
            "        models:\n"
            "          private-model: {}\n",
        )
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, "only-in-user-config/private-model"
        ) as refusal:
            assert refusal is None

    def test_the_refusal_reaches_the_lane_b_spawn_site_with_nothing_spawned(
        self, provider, user_xdg, monkeypatch
    ):
        def _must_not_run(**_kwargs):  # pragma: no cover - the point of the test
            raise AssertionError("a refused slot must never reach the spawn")

        monkeypatch.setattr(executor, "_spawn_in_context", _must_not_run)
        capture = executor._spawn_child(
            provider=provider, model="only-in-user-config/private-model",
            prompt="hi", profile=PROFILE_AGENT_READONLY, transport="argv",
            timeout=5, cwd=None,
        )
        assert capture.launch_error and "only-in-user-config" in capture.launch_error
        assert capture.argv == []

    def test_the_refusal_reaches_the_lane_a_spawn_site(self, provider, user_xdg):
        with patch("shutil.which", return_value="/usr/bin/opencode"), \
                patch("utils.llm_client.subprocess.run") as spawned:
            result = llm_client.invoke_with_provider(
                prompt="hi",
                model_spec="opencode:only-in-user-config/private-model",
                timeout=5,
            )
        assert result["success"] is False
        assert "only-in-user-config" in result["error"]
        assert spawned.call_count == 0

    def test_an_unparseable_user_config_never_becomes_a_refusal(
        self, provider, user_xdg
    ):
        """A config problem may not end a run — the same rule R3 follows."""
        (user_xdg / "opencode" / "opencode.json").write_text("{not json", encoding="utf-8")
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, "only-in-user-config/private-model"
        ) as refusal:
            assert refusal is None

    @pytest.mark.parametrize(
        "block",
        [
            {"options": {"timeout": 30}},
            {"models": {"some-new-model": {}}},
            {"options": {"timeout": 30}, "models": {"some-new-model": {}}},
        ],
        ids=["options-only", "extra-models", "both"],
    )
    def test_an_override_of_a_built_in_is_not_a_definition(
        self, provider, user_xdg, block
    ):
        """Only a provider *implementation* is unreconstructible.

        A block that adds a model to, or tweaks options on, a provider opencode
        already has still resolves after the redirect — it merely loses the
        override. Refusing it would be exactly the over-refusal this design
        forbids, and it is the single most likely shape in a real config.
        """
        (user_xdg / "opencode" / "opencode.json").write_text(
            json.dumps({"provider": {"openai": block}}), encoding="utf-8"
        )
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "openai/gpt-5.5") as r:
            assert r is None

    @pytest.mark.parametrize(
        "block",
        [
            {"npm": "@ai-sdk/some-provider", "models": {"private-model": {}}},
            {"options": {"baseURL": "https://gateway.invalid/v1"},
             "models": {"private-model": {}}},
        ],
        ids=["npm-package", "custom-base-url"],
    )
    def test_only_an_implementation_source_is_unreconstructible(
        self, provider, user_xdg, block
    ):
        """The two shapes that cannot be rebuilt once the config is out of reach."""
        (user_xdg / "opencode" / "opencode.json").write_text(
            json.dumps({"provider": {"only-in-user-config": block}}), encoding="utf-8"
        )
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, "only-in-user-config/private-model"
        ) as refusal:
            assert refusal and "only-in-user-config" in refusal


# --------------------------------------------------------------------------- #
# Gate (c) — project-config isolation
# --------------------------------------------------------------------------- #

class TestGateCProjectConfigIsolation:
    """A workspace `opencode.json` must stop deciding what the run does."""

    def test_the_run_env_closes_the_project_config_layer(self, provider, user_xdg):
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)

        # Measured on 1.18.18: `XDG_CONFIG_HOME` + `--pure` remove the GLOBAL
        # layer only — a workspace `opencode.json` pinning a model and an extra
        # agent still shows up in `opencode debug config` under both. This is
        # what closes it.
        assert env[oc.PROJECT_CONFIG_ENV] == "1"

    def test_the_config_home_the_child_sees_is_empty_and_ours(
        self, provider, user_xdg
    ):
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)
            home = Path(env["XDG_CONFIG_HOME"])
            assert home.is_dir()
            assert list(home.iterdir()) == []
            assert home.name.startswith(oc.HERMETIC_DIR_PREFIX)
            assert oct(home.stat().st_mode & 0o777) == "0o700"

    def test_pure_is_rendered_only_inside_the_context(
        self, provider, user_xdg, monkeypatch
    ):
        _pure_help(monkeypatch, supported=True)
        outside = provider.build_agent_command(
            "hi", "opencode/sonnet", PROFILE_AGENT_READONLY
        )
        assert oc.PURE_FLAG not in outside, (
            "`--pure` without the paired config-home redirect is meaningless — "
            "it must never be rendered outside a hermetic dispatch"
        )
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            inside = provider.build_agent_command(
                "hi", "opencode/sonnet", PROFILE_AGENT_READONLY
            )
        assert oc.PURE_FLAG in inside
        # The flag is the ONLY difference: same order, same everything else.
        assert [a for a in inside if a != oc.PURE_FLAG] == outside

    def test_an_unsupported_pure_omits_the_flag_and_keeps_the_run(
        self, provider, user_xdg, monkeypatch, capsys
    ):
        """§5.4.1's decline-toward-status-quo rule: reduced containment, not a failure."""
        _pure_help(monkeypatch, supported=False)
        capsys.readouterr()
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            argv = provider.build_agent_command(
                "hi", "opencode/sonnet", PROFILE_AGENT_READONLY
            )
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)

        assert oc.PURE_FLAG not in argv
        warning = capsys.readouterr().err
        assert oc.PURE_FLAG in warning and "Containment is reduced" in warning
        # The redirect still applies — it is the load-bearing half.
        assert env["XDG_CONFIG_HOME"].split(os.sep)[-1].startswith(
            oc.HERMETIC_DIR_PREFIX
        )

    @pytest.mark.live
    def test_debug_config_yields_the_empty_shape(self, provider, tmp_path):
        """The real gate, on opencode 1.18.18: a workspace config stops applying."""
        import shutil as _shutil

        if not _shutil.which("opencode"):
            pytest.skip("opencode is not installed")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "opencode.json").write_text(
            json.dumps({
                "$schema": "https://opencode.ai/config.json",
                "model": "opencode/should-never-win",
                "agent": {"zzz-marker": {"description": "x", "mode": "subagent"}},
            }),
            encoding="utf-8",
        )
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)
            argv = ["opencode", oc.PURE_FLAG, "debug", "config"]
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=120,
                env=env, cwd=str(workspace),
            )
        assert done.returncode == 0, done.stderr
        document = json.loads(done.stdout)
        # The workspace's agent never loads. The only agent entry left is the
        # permission rows this adapter carries itself (T4.1, §4.2 step 1).
        assert set(document.get("agent") or {}) <= {oc.READONLY_AGENT}
        assert "zzz-marker" not in (document.get("agent") or {})
        assert "model" not in document

    @pytest.mark.live
    @pytest.mark.debug_config
    def test_the_reader_on_the_real_binary_with_a_canary_config(self, provider, tmp_path):
        """T8.7's live twin: the real `debug config`, reduced, on a scratch HOME.

        No model call and no network (``OPENCODE_DISABLE_MODELS_FETCH=1``); the
        only key anywhere is a canary, and it must not survive the reduction.
        """
        import shutil as _shutil

        if not _shutil.which("opencode"):
            pytest.skip("opencode is not installed")
        home = tmp_path / "home"
        (home / ".config" / "opencode").mkdir(parents=True)
        (home / ".config" / "opencode" / "opencode.jsonc").write_text(
            '{\n  // a comment\n  "provider": {"stub": {"npm": "@ai-sdk/openai-compatible",'
            ' "options": {"baseURL": "http://127.0.0.1:9/v1",'
            ' "apiKey": "sk-canary-not-a-key-live"},},},\n}\n',
            encoding="utf-8",
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        env = {
            "HOME": str(home), "PATH": os.environ.get("PATH", ""),
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
        }
        declared = oc._read_declared_providers(env, str(workspace), False)
        assert declared is not None
        assert declared.providers["stub"] == oc.ProviderDecl(custom=True, has_key=True)
        assert "sk-canary" not in repr(declared)


# --------------------------------------------------------------------------- #
# T8.5 — the user's own config, read through `opencode debug config`
# --------------------------------------------------------------------------- #

class TestUserConfigReader:
    """`_user_defines_provider` onto the reader: jsonc fixed, memoised, and a
    reader failure never changes today's verdict."""

    #: `debug config` for a user whose only config is a commented
    #: `opencode.jsonc` declaring `only-in-jsonc` (the 1.18.32 capture's
    #: `jsonc_commented` shape: opencode parses what `json.load` cannot).
    JSONC_RESOLVED = json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"only-in-jsonc": {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "http://127.0.0.1:9/v1",
                        "apiKey": "sk-canary-not-a-key-5"},
            "models": {"m": {}},
        }},
    })

    @pytest.fixture
    def jsonc_user(self, tmp_path, monkeypatch):
        home = tmp_path / "jsonc-config"
        (home / "opencode").mkdir(parents=True)
        (home / "opencode" / "opencode.jsonc").write_text(
            '{\n  // a comment\n  "provider": {"only-in-jsonc": {\n'
            '    "npm": "@ai-sdk/openai-compatible",\n'
            '    "options": {"baseURL": "http://127.0.0.1:9/v1",},\n  },},\n}\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
        monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
        return home

    def _serve(self, monkeypatch, stdout, code=0):
        calls = []

        def spawn(argv, env, cwd, timeout):
            calls.append({"argv": list(argv), "env": env})
            if stdout is None:
                return None
            return subprocess.CompletedProcess(list(argv), code, stdout, "")

        monkeypatch.setattr(oc, "_spawn_debug_config", spawn)
        return calls

    def test_a_commented_jsonc_provider_is_now_refused_before_spawn(
        self, provider, jsonc_user, monkeypatch
    ):
        calls = self._serve(monkeypatch, self.JSONC_RESOLVED)
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, "only-in-jsonc/m"
        ) as refusal:
            assert refusal and "only-in-jsonc" in refusal
            assert "sk-canary" not in refusal
        # The read is the user's own: no hermetic delta, no `--pure`.
        assert calls == [{"argv": ["opencode", "debug", "config"], "env": None}]

    def test_a_plain_json_config_is_answered_without_a_spawn(
        self, provider, tmp_path, monkeypatch
    ):
        """Only a plain global `opencode.json`, no `OPENCODE_CONFIG*`, no
        project config: the file read is conclusive and nothing is spawned."""
        home = tmp_path / "plain-config"
        (home / "opencode").mkdir(parents=True)
        (home / "opencode" / "opencode.json").write_text(json.dumps({
            "provider": {"only-in-json": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": "http://127.0.0.1:9/v1"},
            }},
        }), encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
        for name in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
        calls = self._serve(monkeypatch, self.JSONC_RESOLVED)
        assert oc._user_defines_provider("only-in-json")
        assert not oc._user_defines_provider("openrouter")
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "openrouter/m") as refusal:
            assert refusal is None
        assert calls == []

    def test_the_file_read_alone_used_to_miss_it(self, jsonc_user):
        """The pre-T8.5 reader, kept as the fallback, still cannot parse it."""
        assert oc._user_defines_provider_from_files("only-in-jsonc") is False

    @pytest.mark.parametrize(
        "stdout, code",
        [(None, 0), ("{not json", 0), ("{}", 1)],
        ids=["spawn-failure", "non-json", "exit-1"],
    )
    def test_an_unreadable_reader_result_still_never_refuses(
        self, provider, jsonc_user, monkeypatch, stdout, code
    ):
        """Reader failure → today's file read → an unparseable file → dispatch."""
        self._serve(monkeypatch, stdout, code)
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "only-in-jsonc/m") as refusal:
            assert refusal is None

    def test_the_read_is_memoised_per_process(self, provider, jsonc_user, monkeypatch):
        calls = self._serve(monkeypatch, self.JSONC_RESOLVED)
        for _ in range(3):
            assert oc._user_defines_provider("only-in-jsonc")
            assert not oc._user_defines_provider("openrouter")
        assert len(calls) == 1
        # A failed read is cached too.
        oc.reset_hermetic_state()
        calls = self._serve(monkeypatch, None)
        oc._user_defines_provider("x")
        oc._user_defines_provider("y")
        assert len(calls) == 1

    def test_a_bounded_read_never_poisons_the_memo(self, jsonc_user, monkeypatch):
        calls = self._serve(monkeypatch, None)
        assert oc.read_user_declared_config(timeout=0.5) is None
        calls = self._serve(monkeypatch, self.JSONC_RESOLVED)
        assert oc.read_user_declared_config() is not None
        assert len(calls) == 1

    def test_an_override_read_through_the_reader_is_not_a_definition(
        self, provider, user_xdg, monkeypatch
    ):
        self._serve(monkeypatch, json.dumps(
            {"provider": {"openrouter": {"options": {"apiKey": "sk-canary-not-a-key-4"}}}}
        ))
        assert oc._user_defines_provider("openrouter") is False


# --------------------------------------------------------------------------- #
# Gate (d) — concurrency
# --------------------------------------------------------------------------- #

class TestGateDConcurrency:
    """One directory per process, shared by every concurrent slot."""

    SLOTS = 8

    def test_concurrent_slots_create_exactly_one_directory_and_share_it(
        self, provider, user_xdg, new_hermetic_dirs
    ):
        # +1: the main thread waits on it too, so the assertions below run while
        # every slot is genuinely inside its context.
        barrier = threading.Barrier(self.SLOTS + 1)
        seen: list = []
        errors: list = []
        release = threading.Event()

        def slot() -> None:
            try:
                with provider.hermetic_dispatch(
                    PROFILE_AGENT_READONLY, "opencode/sonnet"
                ):
                    delta = provider.resolve_env(
                        "opencode/sonnet", PROFILE_AGENT_READONLY
                    )
                    seen.append(delta["XDG_CONFIG_HOME"])
                    barrier.wait(timeout=30)
                    # Hold the claim until every slot has taken one, so the
                    # directory is genuinely shared rather than sequentially
                    # created and destroyed.
                    release.wait(timeout=30)
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=slot) for _ in range(self.SLOTS)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=30)
        try:
            assert len(set(seen)) == 1, f"slots did not share one directory: {seen}"
            assert len(new_hermetic_dirs()) == 1
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=30)

        assert errors == []
        # Last one out removes it.
        assert new_hermetic_dirs() == []

    def test_no_slot_observes_another_slots_env(self, provider, user_xdg):
        """A slot outside the context sees today's env, concurrently with one inside."""
        inside_ready = threading.Event()
        finish = threading.Event()
        observed: dict = {}

        def held() -> None:
            with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
                inside_ready.set()
                finish.wait(timeout=30)

        thread = threading.Thread(target=held)
        thread.start()
        try:
            assert inside_ready.wait(timeout=30)
            # This thread never entered the context: its env delta must carry no
            # hermetic keys at all, even while another slot holds one.
            observed = provider.resolve_env("opencode/sonnet", PROFILE_AGENT_READONLY)
        finally:
            finish.set()
            thread.join(timeout=30)

        assert "XDG_CONFIG_HOME" not in observed
        assert oc.PROJECT_CONFIG_ENV not in observed


# --------------------------------------------------------------------------- #
# Gate (e) — cleanup
# --------------------------------------------------------------------------- #

class TestGateECleanup:
    """`finally` plus an idempotent `atexit` sweep, and nothing borrowed is deleted."""

    def test_normal_exit_removes_the_directory(self, provider, user_xdg):
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            home = provider.resolve_env(
                "opencode/sonnet", PROFILE_AGENT_READONLY
            )["XDG_CONFIG_HOME"]
            assert Path(home).is_dir()
        assert not Path(home).exists()
        assert oc._CONFIG_HOME is None

    @pytest.mark.parametrize(
        "raised",
        [
            RuntimeError("a dispatch blew up"),
            subprocess.TimeoutExpired(cmd=["opencode"], timeout=1.0),
            KeyboardInterrupt(),
        ],
        ids=["exception", "slot-timeout", "keyboard-interrupt"],
    )
    def test_every_abnormal_exit_removes_the_directory(
        self, provider, user_xdg, raised
    ):
        home = None
        with pytest.raises(type(raised)):
            with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
                home = provider.resolve_env(
                    "opencode/sonnet", PROFILE_AGENT_READONLY
                )["XDG_CONFIG_HOME"]
                raise raised
        assert home and not Path(home).exists()
        assert oc._CONFIG_HOME is None

    def test_a_user_supplied_config_home_survives_untouched(self, provider, user_xdg):
        marker = user_xdg / "opencode" / "opencode.json"
        before = marker.read_text(encoding="utf-8")
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            pass
        oc.sweep_config_home()
        assert marker.read_text(encoding="utf-8") == before
        assert Path(os.environ["XDG_CONFIG_HOME"]) == user_xdg

    def test_the_sweep_is_idempotent_and_registered(self, provider, user_xdg):
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            pass
        assert oc._ATEXIT_REGISTERED, "the atexit backstop must be registered"
        # Tolerates the directory already being gone, however many times.
        oc.sweep_config_home()
        oc.sweep_config_home()
        assert oc._CONFIG_HOME is None

    def test_a_non_empty_directory_is_still_removed(self, provider, user_xdg):
        """opencode seeds a stub config there (measured, 1.18.18) — `rmtree`, not `rmdir`."""
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            home = Path(
                provider.resolve_env(
                    "opencode/sonnet", PROFILE_AGENT_READONLY
                )["XDG_CONFIG_HOME"]
            )
            (home / "opencode").mkdir()
            (home / "opencode" / "opencode.jsonc").write_text("{}", encoding="utf-8")
        assert not home.exists()

    def test_the_lane_b_spawn_site_releases_in_a_finally(
        self, provider, user_xdg, new_hermetic_dirs
    ):
        """The `finally` half, at the spawn site rather than in a helper."""
        seen: dict = {}

        def _boom(**_kwargs):
            seen["active"] = oc._ACTIVE.get()
            raise RuntimeError("the dispatch blew up")

        with patch.object(executor, "_spawn_in_context", _boom):
            with pytest.raises(RuntimeError):
                executor._spawn_child(
                    provider=provider, model="opencode/sonnet", prompt="hi",
                    profile=PROFILE_AGENT_READONLY, transport="argv",
                    timeout=5, cwd=None,
                )

        assert seen["active"] is not None, "the spawn ran outside the context"
        assert oc._ACTIVE.get() is None
        assert oc._CONFIG_HOME is None
        assert new_hermetic_dirs() == []

    def test_the_lane_a_spawn_site_dispatches_inside_the_context_and_releases(
        self, provider, user_xdg, new_hermetic_dirs
    ):
        captured: dict = {}

        def _run(cmd, **kwargs):
            captured["env"] = kwargs.get("env") or {}
            captured["cmd"] = list(cmd)
            captured["active"] = oc._ACTIVE.get()
            return subprocess.CompletedProcess(
                args=list(cmd), returncode=0,
                stdout=json.dumps(
                    {"type": "text", "part": {"type": "text", "text": '{"ok": true}'}}
                ),
                stderr="",
            )

        with patch("shutil.which", return_value="/usr/bin/opencode"), \
                patch("utils.llm_client.subprocess.run", side_effect=_run):
            llm_client.invoke_with_provider(
                prompt="hi", model_spec="opencode:opencode/sonnet", timeout=5
            )

        assert captured["active"] is not None, "the fan-out spawned outside the context"
        assert captured["env"]["XDG_CONFIG_HOME"].split(os.sep)[-1].startswith(
            oc.HERMETIC_DIR_PREFIX
        )
        assert captured["env"][oc.PROJECT_CONFIG_ENV] == "1"
        # Released the moment the call returned.
        assert oc._ACTIVE.get() is None
        assert oc._CONFIG_HOME is None
        assert new_hermetic_dirs() == []

    @pytest.mark.parametrize(
        "break_at",
        ["argv", "env"],
        ids=["argv-build", "env-composition"],
    )
    def test_a_lane_a_setup_failure_after_entry_still_releases_the_context(
        self, provider, user_xdg, new_hermetic_dirs, break_at
    ):
        """The Lane A sibling of the `finally` test above, for the *setup* half.

        The context is entered BEFORE argv and env are rendered (they are read
        off the same context state), so those two steps are the window in which
        a raise could leave the active context and its config-home reference
        behind. Both run inside the dispatch's own `ExitStack`, so neither can.
        """
        seen: dict = {}

        def _boom(*_args, **_kwargs):
            seen["active"] = oc._ACTIVE.get()
            raise RuntimeError(f"the {break_at} step blew up")

        broken = (
            patch.object(provider, "build_command_for_transport", _boom)
            if break_at == "argv"
            else patch.object(llm_client, "build_child_env", _boom)
        )
        with patch("shutil.which", return_value="/usr/bin/opencode"), broken:
            # `as raised` is load-bearing: the traceback keeps the failing frame
            # — and with it the dispatch's own `ExitStack` — alive, so release
            # has to be something the code did on the way out rather than
            # something refcounting did for it afterwards. That is the leak this
            # test is about; a caller that logs the exception holds it just as
            # long, and a traceback cycle can outlive the run entirely.
            with pytest.raises(RuntimeError) as raised:
                llm_client.invoke_with_provider(
                    prompt="hi", model_spec="opencode:opencode/sonnet", timeout=5
                )

        assert raised.traceback  # the frame is still reachable while we assert
        assert seen["active"] is not None, "the setup step ran outside the context"
        assert oc._ACTIVE.get() is None
        assert oc._CONFIG_HOME is None
        assert new_hermetic_dirs() == []


# --------------------------------------------------------------------------- #
# Scope and the escape hatch
# --------------------------------------------------------------------------- #

class TestHermeticScopeAndHatch:
    """Which slots are in, and the one config key that turns it all off."""

    def test_the_default_is_on(self, provider, user_xdg):
        assert oc.hermetic_policy() == oc.HERMETIC_DEFAULT == "on"

    @pytest.mark.parametrize("policy", ["on", "auto"])
    def test_an_enabled_policy_redirects_the_scoped_profiles(
        self, provider, user_xdg, tmp_path, monkeypatch, policy
    ):
        _use_config(
            tmp_path, monkeypatch,
            f"providers:\n  opencode:\n    hermetic: {policy!r}\n",
        )
        assert oc.hermetic_policy() == policy
        for profile in (PROFILE_GENERATE, PROFILE_AGENT_READONLY):
            assert oc.hermetic_enabled(profile)

    def test_a_write_slot_keeps_todays_inherited_config(
        self, provider, user_xdg, new_hermetic_dirs
    ):
        """`agent-write` joins only once the gates are green on a write slot.

        The built-in `build` agent is *expected* to survive `--pure`, and
        "expected" is not the standard this repo applies to permission-carrying
        dispatch.
        """
        assert not oc.hermetic_enabled(PROFILE_AGENT_WRITE)
        with provider.hermetic_dispatch(PROFILE_AGENT_WRITE, "opencode/sonnet") as r:
            assert r is None
            delta = provider.resolve_env("opencode/sonnet", PROFILE_AGENT_WRITE)
            argv = provider.build_agent_command(
                "hi", "opencode/sonnet", PROFILE_AGENT_WRITE
            )
        assert "XDG_CONFIG_HOME" not in delta
        assert oc.PURE_FLAG not in argv
        assert new_hermetic_dirs() == []

    def test_off_restores_todays_inheritance(
        self, provider, user_xdg, tmp_path, monkeypatch, new_hermetic_dirs
    ):
        _use_config(
            tmp_path, monkeypatch, "providers:\n  opencode:\n    hermetic: 'off'\n"
        )
        _pure_help(monkeypatch, supported=True)
        assert oc.hermetic_policy() == "off"
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet") as r:
            assert r is None
            delta = provider.resolve_env("opencode/sonnet", PROFILE_AGENT_READONLY)
            argv = provider.build_agent_command(
                "hi", "opencode/sonnet", PROFILE_AGENT_READONLY
            )
        assert "XDG_CONFIG_HOME" not in delta
        assert oc.PROJECT_CONFIG_ENV not in delta
        assert oc.PURE_FLAG not in argv
        assert new_hermetic_dirs() == []

    def test_off_also_stops_refusing_an_unreconstructible_provider(
        self, provider, user_xdg, tmp_path, monkeypatch
    ):
        """`hermetic: off` is the no-edit exit, so it cannot leave a refusal behind."""
        _use_config(
            tmp_path, monkeypatch, "providers:\n  opencode:\n    hermetic: 'off'\n"
        )
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, "only-in-user-config/private-model"
        ) as refusal:
            assert refusal is None

    @pytest.mark.parametrize("written,expected", [(True, "on"), (False, "off")])
    def test_a_yaml_boolean_means_the_documented_word(
        self, provider, user_xdg, tmp_path, monkeypatch, written, expected
    ):
        """YAML 1.1 resolves a bare `on`/`off` to a boolean; honour the intent."""
        _use_config(
            tmp_path, monkeypatch,
            f"providers:\n  opencode:\n    hermetic: {'on' if written else 'off'}\n",
        )
        assert oc.hermetic_policy() == expected

    def test_an_unrecognized_policy_warns_once_and_uses_the_default(
        self, provider, user_xdg, tmp_path, monkeypatch, capsys
    ):
        _use_config(
            tmp_path, monkeypatch, "providers:\n  opencode:\n    hermetic: sometimes\n"
        )
        capsys.readouterr()
        assert oc.hermetic_policy() == oc.HERMETIC_DEFAULT
        first = capsys.readouterr().err
        assert "providers.opencode.hermetic" in first
        assert oc.hermetic_policy() == oc.HERMETIC_DEFAULT
        assert capsys.readouterr().err == "", "the warning must be once per process"

    def test_an_unreadable_config_falls_back_to_the_default(
        self, provider, user_xdg, monkeypatch
    ):
        """A config read that raises may not break a dispatch."""
        def _explode(*_args, **_kwargs):
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(registry, "load_config", _explode)
        assert oc.hermetic_policy() == oc.HERMETIC_DEFAULT

    def test_no_other_adapter_declares_the_context(self):
        """Swept over the registry, so a new adapter cannot quietly inherit it."""
        from utils.provider_registry import _PROVIDERS

        for name in sorted(_PROVIDERS):
            has_hook = callable(
                getattr(registry.get_provider(name), "hermetic_dispatch", None)
            )
            assert has_hook == (name == "opencode"), (
                f"{name} declares a hermetic dispatch context; opencode's config "
                f"layering is what this contains, and it is not a general concern"
            )


# --------------------------------------------------------------------------- #
# D3 step 4 — `describe_unlisted_model` reads the listing in the dispatch's own
# configuration, and an unreadable listing is unknown, never a refusal
# --------------------------------------------------------------------------- #

_LISTING = (
    "opencode/big-pickle\n"
    "opencode/ling-3.0-flash-fin-free\n"
    "opencode/mimo-v2.5-free\n"
    "opencode/muse-spark-1.2-contributor-free\n"
    "opencode/muse-spark-1.3-contributor-free\n"
    "opencode/nemotron-3-ultra-free\n"
    "opencode/nemotron-3.5-lightning-free\n"
    "openai/gpt-5.5\n"
    "google/gemini-3.1-pro-preview\n"
)


def _listing(monkeypatch, stdout=_LISTING, *, returncode=0, seen=None):
    """Pin `_list_models` to one stdout, recording the env each call was given."""
    calls = seen if seen is not None else []

    def _fake(argv, env):
        calls.append((list(argv), dict(env)))
        if stdout is None:
            return None
        return subprocess.CompletedProcess(
            args=list(argv), returncode=returncode, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(oc, "_list_models", _fake)
    return calls


@pytest.mark.model_catalog
class TestDescribeUnlistedModel:
    def test_a_listed_token_is_known(self, provider, user_xdg, monkeypatch):
        _listing(monkeypatch)
        verdict = provider.describe_unlisted_model("opencode/big-pickle")
        assert verdict.known is True
        assert verdict.detail == "" and verdict.suggestions == ()

    def test_the_effort_suffix_is_split_off_before_comparing(
        self, provider, user_xdg, monkeypatch
    ):
        """The token compared is the token dispatched (`--variant` split)."""
        _listing(monkeypatch)
        assert provider.describe_unlisted_model("openai/gpt-5.5:high").known is True
        # A suffix that is NOT an accepted effort stays part of the token.
        assert provider.describe_unlisted_model("openai/gpt-5.5:nope").known is False

    def test_comparison_is_whole_line_exact_and_case_sensitive(
        self, provider, user_xdg, monkeypatch
    ):
        _listing(monkeypatch)
        assert provider.describe_unlisted_model("opencode/Big-Pickle").known is False
        assert provider.describe_unlisted_model("opencode/big").known is False
        assert provider.describe_unlisted_model("opencode/big-pickle-free").known is False

    def test_an_unlisted_token_composes_its_own_diagnostic(
        self, provider, user_xdg, monkeypatch
    ):
        _listing(monkeypatch)
        verdict = provider.describe_unlisted_model("opencode/sonnet")
        assert verdict.known is False
        assert len(verdict.suggestions) <= 3
        assert all(s.startswith("opencode/") for s in verdict.suggestions)
        assert "lists 7 id(s) under 'opencode/'" in verdict.detail
        assert "'opencode/sonnet' is not one of them" in verdict.detail
        for suggestion in verdict.suggestions:
            assert suggestion in verdict.detail
        # Composed here from the listing, never quoted from a run-time line.
        assert "Did you mean" not in verdict.detail

    def test_suggestions_are_nearest_first_and_capped_at_three(
        self, provider, user_xdg, monkeypatch
    ):
        _listing(monkeypatch)
        verdict = provider.describe_unlisted_model("opencode/nemotron-3-ultra")
        assert verdict.suggestions[0] == "opencode/nemotron-3-ultra-free"
        assert len(verdict.suggestions) == 3

    def test_an_unknown_prefix_says_so_with_the_total(
        self, provider, user_xdg, monkeypatch
    ):
        _listing(monkeypatch)
        verdict = provider.describe_unlisted_model("nosuch/model")
        assert verdict.known is False
        assert verdict.suggestions == ()
        assert "lists no ids under 'nosuch/'" in verdict.detail
        assert "9 ids listed in all" in verdict.detail

    @pytest.mark.parametrize(
        "stdout,returncode",
        [
            (None, 0),                                   # probe could not run
            ("", 0),                                     # empty listing
            (_LISTING, 1),                               # non-zero exit
            ("timestamp=1 level=ERROR boom\n" + _LISTING, 0),  # opens with a log line
            ("not a listing at all\nnor this\nopencode/big-pickle\n", 0),  # mostly prose
            ("Available models:\n\n  - big-pickle\n", 0),  # decorated, no slugs
        ],
    )
    def test_unreadable_listings_are_unknown_never_false(
        self, provider, user_xdg, monkeypatch, stdout, returncode
    ):
        _listing(monkeypatch, stdout, returncode=returncode)
        verdict = provider.describe_unlisted_model("opencode/sonnet")
        assert verdict.known is None
        assert verdict.detail == "" and verdict.suggestions == ()

    def test_a_bare_token_is_left_to_the_structural_refusal(
        self, provider, user_xdg, monkeypatch
    ):
        seen = _listing(monkeypatch)
        assert provider.describe_unlisted_model("sonnet").known is None
        assert seen == []

    def test_a_raising_probe_is_unknown(self, provider, user_xdg, monkeypatch):
        def _boom(argv, env):
            raise RuntimeError("no")

        monkeypatch.setattr(oc, "_list_models", _boom)
        assert provider.describe_unlisted_model("opencode/sonnet").known is None

    def test_the_listing_is_read_under_the_hermetic_redirect(
        self, provider, user_xdg, monkeypatch
    ):
        """Same context as the dispatch it vets: XDG_CONFIG_HOME redirected,
        the config carriers stripped, the user's own config home not consulted."""
        _pure_help(monkeypatch)
        monkeypatch.setenv("OPENCODE_CONFIG", "/etc/nope/opencode.json")
        seen = _listing(monkeypatch)
        provider.describe_unlisted_model(
            "opencode/big-pickle", profile=PROFILE_AGENT_READONLY
        )
        assert len(seen) == 1
        argv, env = seen[0]
        assert argv == ["opencode", "models"]
        assert env["XDG_CONFIG_HOME"] != str(user_xdg)
        assert env["XDG_CONFIG_HOME"].startswith(
            str(Path(oc.tempfile.gettempdir()))
        ) or oc.HERMETIC_DIR_PREFIX in env["XDG_CONFIG_HOME"]
        assert "OPENCODE_CONFIG" not in env
        for name in PASSTHROUGH_XDG:
            assert env[name] == os.environ[name], f"{name} must pass through"
        # The slot is released again: nothing stays claimed after the read.
        assert oc._ACTIVE.get() is None

    def test_the_listing_carries_the_rebuilt_config_content(
        self, provider, user_xdg, monkeypatch, tmp_path
    ):
        _pure_help(monkeypatch)
        _use_config(
            tmp_path, monkeypatch,
            "providers:\n  opencode:\n    config_providers:\n"
            "      my-gateway:\n        options: {baseURL: https://gw.invalid/v1}\n"
            "        models: {my-model: {}}\n",
        )
        seen = _listing(monkeypatch, _LISTING + "my-gateway/my-model\n")
        verdict = provider.describe_unlisted_model("my-gateway/my-model")
        assert verdict.known is True
        _argv, env = seen[0]
        assert "my-gateway" in env["OPENCODE_CONFIG_CONTENT"]

    def test_a_provider_only_in_the_users_config_is_unknown_not_listed(
        self, provider, user_xdg, monkeypatch
    ):
        """The slot is refused before spawn; a listing here would describe a
        configuration no dispatch will use, so nothing is read."""
        _pure_help(monkeypatch)
        seen = _listing(monkeypatch)
        verdict = provider.describe_unlisted_model("only-in-user-config/private-model")
        assert verdict.known is None
        assert seen == []

    def test_a_plain_read_when_hermetic_is_off(
        self, provider, user_xdg, monkeypatch, tmp_path
    ):
        _use_config(tmp_path, monkeypatch, "providers:\n  opencode:\n    hermetic: off\n")
        seen = _listing(monkeypatch)
        provider.describe_unlisted_model("opencode/big-pickle")
        _argv, env = seen[0]
        assert env["XDG_CONFIG_HOME"] == str(user_xdg)

    def test_a_write_profile_reads_plain_like_its_dispatch(
        self, provider, user_xdg, monkeypatch
    ):
        seen = _listing(monkeypatch)
        provider.describe_unlisted_model(
            "opencode/big-pickle", profile=PROFILE_AGENT_WRITE
        )
        _argv, env = seen[0]
        assert env["XDG_CONFIG_HOME"] == str(user_xdg)


class TestTheListingProbePins:
    """`_list_models` mirrors the executor's probe pins, so a hanging catalog
    read is killed at 30 s and read as unknown."""

    def test_the_probe_is_bounded_and_never_raises(self, monkeypatch):
        calls = []

        def _run(argv, **kwargs):
            calls.append(kwargs)
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

        monkeypatch.setattr(oc.subprocess, "run", _run)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert oc._list_models(["opencode", "models"], dict(os.environ)) is None
        assert calls and calls[0]["timeout"] == 30
        assert calls[0]["stdin"] is subprocess.DEVNULL

    def test_the_parser_shape_rules(self):
        assert oc._parse_model_listing("") is None
        assert oc._parse_model_listing("opencode/a\n") == frozenset({"opencode/a"})
        assert oc._parse_model_listing("level=ERROR x\nopencode/a\n") is None
        assert oc._parse_model_listing("a b\nc d\nopencode/a\n") is None
        # Whitespace is stripped; blank lines are ignored.
        assert oc._parse_model_listing("  opencode/a  \n\n") == frozenset({"opencode/a"})
