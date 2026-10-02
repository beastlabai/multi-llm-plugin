"""T4.7 — the permission rows over the user's own opencode config.

plans/opencode-validator-false-rate-limit-and-denied-turn.md §4.4 steps 1-3.

`agent-write` (always) and `agent-readonly` under `hermetic: off` keep the
user's config: no `--pure`, no `XDG_CONFIG_HOME` redirect, nothing stripped.
Their three deny rows go under `agent.build` / `agent.plan`, deep-merged over
the inherited `OPENCODE_CONFIG_CONTENT`, and the user's own
`external_directory` rules — read through `opencode debug config` (T8.2) — are
re-listed after our `"*": "deny"`, because a user allow before it is shadowed
(rig `t47_shared_toplevel_*`) and one after it survives (rig `t47_shared_*`).

Every test here patches the one spawn seam (`_spawn_debug_config`) or asserts
it is never reached; nothing spawns a real opencode, and nothing spends.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers import opencode as oc  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
)

from tests.opencode_fixtures import CAPTURE as CAPTURE_ROOT  # noqa: E402  (capture/<pin>/)
MODEL = "opencode/big-pickle"
FLOOR, CEILING = oc.OpenCodeProvider.CONTINUE_LOOP_ON_DENY_RANGE
SENTINEL = "sk-canary-not-a-key-t47"

READ_ROWS = {"*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"}


@pytest.fixture(autouse=True)
def clean_state():
    oc.sweep_config_home()
    oc.reset_hermetic_state()
    executor.reset_version_cache()
    yield
    oc.sweep_config_home()
    oc.reset_hermetic_state()
    executor.reset_version_cache()


@pytest.fixture
def provider():
    return registry.get_provider("opencode")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The names the built-in allows resolve from, and no inherited carriers."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    for name in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT", "OPENCODE_PERMISSION"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _use_config(tmp_path, monkeypatch, body: str) -> None:
    path = tmp_path / "multi-llm-config.yaml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
    registry._config = None
    registry._config_key = None


def _hermetic_off(root, monkeypatch) -> None:
    _use_config(root, monkeypatch, "providers:\n  opencode:\n    hermetic: 'off'\n")


class _Reader:
    """The ``debug config`` spawn seam: a canned resolved config, counted."""

    def __init__(self, document: Optional[Dict[str, Any]], *, delay: float = 0.0):
        self.document = document
        self.delay = delay
        self.calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, argv, env, cwd, timeout):
        with self._lock:
            self.calls.append({"argv": list(argv), "env": env, "cwd": cwd})
        if self.delay:
            time.sleep(self.delay)
        if self.document is None:
            return None
        return subprocess.CompletedProcess(
            args=list(argv), returncode=0, stdout=json.dumps(self.document), stderr=""
        )


def _reader(monkeypatch, document: Optional[Dict[str, Any]], **kwargs) -> _Reader:
    fake = _Reader(document, **kwargs)
    monkeypatch.setattr(oc, "_spawn_debug_config", fake)
    return fake


def _no_spawn(monkeypatch) -> None:
    def _refuse(*_args, **_kwargs):  # pragma: no cover - the point
        raise AssertionError("opencode debug config was spawned")

    monkeypatch.setattr(oc, "_spawn_debug_config", _refuse)


def _delta(provider, profile: str, *, cwd: Optional[str] = None,
           version: Optional[str] = CEILING) -> Dict[str, str]:
    kwargs = {"runner_version": version} if version is not None else {}
    with provider.hermetic_dispatch(profile, MODEL, **kwargs) as refusal:
        assert refusal is None
        return dict(provider.resolve_env(MODEL, profile, cwd=cwd) or {})


def _document(provider, profile: str = PROFILE_AGENT_WRITE, **kwargs) -> Dict[str, Any]:
    return json.loads(_delta(provider, profile, **kwargs)["OPENCODE_CONFIG_CONTENT"])


def _permission(provider, profile: str = PROFILE_AGENT_WRITE, **kwargs) -> Dict[str, Any]:
    agent = oc.WRITE_AGENT if profile == PROFILE_AGENT_WRITE else oc.READONLY_AGENT
    return _document(provider, profile, **kwargs)["agent"][agent]["permission"]


def _builtins(root: Path, agent: str) -> List[str]:
    allows = [str(root / "tmp" / "opencode" / "*")]
    if agent == oc.READONLY_AGENT:
        allows.append(str(root / "data" / "opencode" / "plans" / "*"))
    return allows


def _external(root: Path, agent: str, *user) -> Dict[str, str]:
    rules = {"*": "deny"}
    rules.update((pattern, "allow") for pattern in _builtins(root, agent))
    rules.update(user)
    return rules


def _top(rules) -> Dict[str, Any]:
    return {"permission": {"external_directory": dict(rules)}}


# --------------------------------------------------------------------------- #
# Step 1-2: the rows, where they go, and what the slot does NOT render
# --------------------------------------------------------------------------- #

class TestTheWriteProfileCarriesTheRows:
    def test_the_three_rows_go_under_agent_build_with_no_edit_or_bash_row(
        self, provider, root, monkeypatch
    ):
        _reader(monkeypatch, {})
        document = _document(provider)
        assert set(document["agent"]) == {"build"}
        permission = document["agent"]["build"]["permission"]
        assert permission == {
            "external_directory": _external(root, "build"),
            "read": READ_ROWS,
            "doom_loop": "deny",
            oc.TRAILING_EXTERNAL_KEY: _external(root, "build"),
        }
        assert "edit" not in permission and "bash" not in permission
        assert "provider" not in document
        assert document["experimental"] == {"continue_loop_on_deny": True}

    def test_the_version_gate_applies_to_the_write_profile_too(
        self, provider, root, monkeypatch, capsys
    ):
        _reader(monkeypatch, {})
        assert "experimental" not in _document(provider, version="1.17.0")
        assert "experimental" not in _document(provider, version="99.0.0")
        assert "experimental" not in _document(provider, version=None)
        err = capsys.readouterr().err
        assert err.count("above the measured continue_loop_on_deny range") == 1
        assert err.count("was not read") == 1

    def test_no_pure_no_redirect_nothing_stripped(self, provider, root, monkeypatch):
        _reader(monkeypatch, {})
        monkeypatch.setenv("OPENCODE_CONFIG_DIR", "/users/own/dir")
        with provider.hermetic_dispatch(PROFILE_AGENT_WRITE, MODEL, runner_version=CEILING):
            argv = provider.build_agent_command("P", MODEL, PROFILE_AGENT_WRITE)
            delta = provider.resolve_env(MODEL, PROFILE_AGENT_WRITE)
            removed = list(provider.get_remove_env() or ())
        assert oc.PURE_FLAG not in argv
        assert argv[:4] == ["opencode", "run", "--agent", "build"]
        assert "XDG_CONFIG_HOME" not in delta
        assert oc.PROJECT_CONFIG_ENV not in delta
        assert not set(oc.CONFIG_ENV_NAMES) & set(removed)
        assert oc._CONFIG_HOME is None

    def test_the_generate_profile_under_hermetic_off_gets_nothing(
        self, provider, root, monkeypatch
    ):
        """A recorded limitation, not a guess: `default_agent` may move it."""
        _hermetic_off(root, monkeypatch)
        _no_spawn(monkeypatch)
        delta = _delta(provider, PROFILE_GENERATE)
        assert "OPENCODE_CONFIG_CONTENT" not in delta


# --------------------------------------------------------------------------- #
# Step 3: the user's rules, re-listed after our deny
# --------------------------------------------------------------------------- #

class TestTheUsersRulesAreRelisted:
    def test_a_user_allow_follows_the_deny_and_the_builtins(
        self, provider, root, monkeypatch
    ):
        _reader(monkeypatch, _top({"/shared/*": "allow"}))
        external = _permission(provider)["external_directory"]
        assert list(external.items()) == list(
            _external(root, "build", ("/shared/*", "allow")).items()
        )

    def test_a_failed_read_is_the_deny_alone_with_one_warning(
        self, provider, root, monkeypatch, capsys
    ):
        _reader(monkeypatch, None)
        assert _permission(provider)["external_directory"] == _external(root, "build")
        assert _permission(provider)["external_directory"] == _external(root, "build")
        assert capsys.readouterr().err.count("could not be read") == 1

    def test_a_deny_exception_survives_after_the_allow_it_narrows(
        self, provider, root, monkeypatch
    ):
        _reader(monkeypatch, _top({"/shared/*": "allow", "/shared/private/*": "deny"}))
        external = _permission(provider)["external_directory"]
        assert list(external)[-2:] == ["/shared/*", "/shared/private/*"]
        assert external["/shared/private/*"] == "deny"

    def test_an_ask_exception_is_emitted_as_deny(self, provider, root, monkeypatch):
        _reader(monkeypatch, _top({"/shared/*": "allow", "/shared/private/*": "ask"}))
        external = _permission(provider)["external_directory"]
        assert external["/shared/private/*"] == "deny"
        assert "ask" not in external.values()

    def test_a_trailing_catch_all_deny_relists_nothing(self, provider, root, monkeypatch):
        _reader(monkeypatch, _top({"/a/*": "allow", "*": "deny"}))
        assert _permission(provider)["external_directory"] == _external(root, "build")

    @pytest.mark.parametrize("top_action,agent_action", [("allow", "deny"), ("deny", "allow")])
    def test_the_agent_layer_wins_a_pattern_named_in_both(
        self, provider, root, monkeypatch, top_action, agent_action
    ):
        """Once, at the agent-layer position, with the agent-layer action."""
        _reader(monkeypatch, {
            "permission": {"external_directory": {"/x/*": top_action, "/y/*": "allow"}},
            "agent": {"build": {"permission": {
                "external_directory": {"/z/*": "allow", "/x/*": agent_action},
            }}},
        })
        external = _permission(provider)["external_directory"]
        assert list(external)[-3:] == ["/y/*", "/z/*", "/x/*"]
        assert external["/x/*"] == agent_action

    def test_the_plan_layer_is_never_read_into_the_build_list(
        self, provider, root, monkeypatch
    ):
        _reader(monkeypatch, {"agent": {"plan": {"permission": {
            "external_directory": {"/plan-only/*": "allow"}}}}})
        assert "/plan-only/*" not in _permission(provider)["external_directory"]

    def test_the_serialized_order_is_the_rule_order(self, provider, root, monkeypatch):
        """Not alphabetical: `/zz/*` stays before `/aa/*`, and `!x/*` (which
        sorts before `*`) still follows the deny."""
        _reader(monkeypatch, _top({"/zz/*": "allow", "/aa/*": "deny", "!x/*": "allow"}))
        text = _delta(provider, PROFILE_AGENT_WRITE)["OPENCODE_CONFIG_CONTENT"]
        positions = [text.index(json.dumps(p)) for p in
                     ("*", _builtins(root, "build")[0], "/zz/*", "/aa/*", "!x/*")]
        assert positions == sorted(positions)

    def test_a_users_star_allow_top_level_does_not_beat_the_deny(
        self, provider, root, monkeypatch
    ):
        """`q2_ustar_agentlayer_build`'s user block: nothing is re-listed, and
        our rows are the agent layer the rig measured beating it."""
        _reader(monkeypatch, {"permission": {"external_directory": "ask", "*": "allow"}})
        permission = _permission(provider)
        assert permission["external_directory"] == _external(root, "build")
        assert "*" not in permission


class TestShapeNormalization:
    @pytest.mark.parametrize("user", [
        "allow",
        {"*": "allow"},
        {"*": "allow", "/d/*": "allow"},
        {"/**": "allow", "/d/*": "allow"},
        {"/d/*": "allow", "**": "allow"},
    ], ids=["string", "star-map", "star-plus-dir", "slash-starstar", "starstar-last"])
    def test_a_catch_all_allow_is_overridden_with_one_warning(
        self, provider, root, monkeypatch, capsys, user
    ):
        _reader(monkeypatch, {"permission": {"external_directory": user}})
        external = _permission(provider)["external_directory"]
        expected = _external(root, "build")
        if isinstance(user, dict) and "/d/*" in user:
            expected["/d/*"] = "allow"
        assert external == expected
        assert not set(external) & (oc.CATCH_ALL_PATTERNS - {"*"})
        assert external["*"] == "deny"
        _permission(provider)
        assert capsys.readouterr().err.count("allows every external directory") == 1

    @pytest.mark.parametrize("user", [
        "ask", "deny", {"*": "ask"}, {"/d/*": "allow", "*": "ask"}, {"/**": "deny"},
    ])
    def test_ask_and_deny_catch_alls_relist_nothing_and_warn_nothing(
        self, provider, root, monkeypatch, capsys, user
    ):
        _reader(monkeypatch, {"permission": {"external_directory": user}})
        assert _permission(provider)["external_directory"] == _external(root, "build")
        assert "multi-llm warning: your opencode" not in capsys.readouterr().err

    @pytest.mark.parametrize("user", [["/d/*"], 7, {"/d/*": 1}, "maybe"])
    def test_an_unexpected_shape_is_the_deny_alone_with_one_warning(
        self, provider, root, monkeypatch, capsys, user
    ):
        _reader(monkeypatch, {"permission": {"external_directory": user}})
        assert _permission(provider)["external_directory"] == _external(root, "build")
        _permission(provider)
        assert capsys.readouterr().err.count("cannot re-list") == 1


# --------------------------------------------------------------------------- #
# Step 2: the merge over the inherited value
# --------------------------------------------------------------------------- #

class TestTheInheritedValue:
    def test_the_users_keys_survive_and_it_is_not_stripped(
        self, provider, root, monkeypatch
    ):
        inherited = {
            "model": "anthropic/claude",
            "provider": {"mine": {"options": {"apiKey": SENTINEL}}},
            "permission": {"bash": "ask"},
            "agent": {"build": {"model": "x/y", "permission": {
                "*": "allow", "bash": "allow", "doom_loop": "ask",
                "read": {"/secret/*": "deny"}}}},
            "experimental": {"other": True},
        }
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(inherited))
        _reader(monkeypatch, {})
        with provider.hermetic_dispatch(PROFILE_AGENT_WRITE, MODEL, runner_version=CEILING):
            delta = provider.resolve_env(MODEL, PROFILE_AGENT_WRITE)
            removed = list(provider.get_remove_env() or ())
        assert "OPENCODE_CONFIG_CONTENT" not in removed
        document = json.loads(delta["OPENCODE_CONFIG_CONTENT"])
        assert document["model"] == "anthropic/claude"
        assert document["provider"] == inherited["provider"]
        assert document["permission"] == {"bash": "ask"}
        assert document["experimental"] == {"other": True, "continue_loop_on_deny": True}
        build = document["agent"]["build"]
        assert build["model"] == "x/y"
        permission = build["permission"]
        # The user's keys first, ours re-appended after them — so the user's
        # permission-level "*" cannot follow and beat our rows (rig
        # `t47_inherited_star_build` vs its `_ctrl` twin).
        assert list(permission) == [
            "*", "bash", "external_directory", "read", "doom_loop", oc.TRAILING_EXTERNAL_KEY]
        assert permission["doom_loop"] == "deny"
        # The user's own read rules are kept; our three read rows follow them.
        # The unrelated user rule does not drop our `*.env.example` allow, so
        # `.env.example` stays readable (findLast would otherwise hit `*.env.*`).
        assert permission["read"] == {
            "/secret/*": "deny", "*.env": "deny", "*.env.*": "deny",
            "*.env.example": "allow"}

    def test_a_string_agent_permission_becomes_a_catch_all_before_our_rows(
        self, provider, root, monkeypatch
    ):
        monkeypatch.setenv(
            "OPENCODE_CONFIG_CONTENT",
            json.dumps({"agent": {"build": {"permission": "allow"}}}),
        )
        _reader(monkeypatch, {})
        permission = _permission(provider)
        assert list(permission) == [
            "*", "external_directory", "read", "doom_loop", oc.TRAILING_EXTERNAL_KEY]
        assert permission["*"] == "allow"

    @pytest.mark.parametrize("value", [
        '{"agent": {"build": ' + json.dumps({"k": SENTINEL}) + ',}',
        json.dumps([SENTINEL]),
        json.dumps({"agent": SENTINEL}),
    ], ids=["invalid-json", "not-an-object", "agent-not-an-object"])
    def test_an_unparseable_value_passes_through_untouched_with_one_warning(
        self, provider, root, monkeypatch, capsys, value
    ):
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", value)
        _no_spawn(monkeypatch)
        with provider.hermetic_dispatch(PROFILE_AGENT_WRITE, MODEL, runner_version=CEILING):
            delta = provider.resolve_env(MODEL, PROFILE_AGENT_WRITE)
            removed = list(provider.get_remove_env() or ())
            provider.resolve_env(MODEL, PROFILE_AGENT_WRITE)
        assert "OPENCODE_CONFIG_CONTENT" not in delta
        assert "OPENCODE_CONFIG_CONTENT" not in removed
        err = capsys.readouterr().err
        assert err.count("OPENCODE_CONFIG_CONTENT is passed through untouched") == 1
        assert SENTINEL not in err
        assert value not in err


# --------------------------------------------------------------------------- #
# The same delivery for `agent-readonly` under `hermetic: off`
# --------------------------------------------------------------------------- #

class TestReadonlyUnderHermeticOff:
    def test_the_rows_go_under_agent_plan_over_the_inherited_value(
        self, provider, root, monkeypatch
    ):
        _hermetic_off(root, monkeypatch)
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps({"model": "a/b"}))
        _reader(monkeypatch, {
            "permission": {"external_directory": {
                "/shared/*": "allow", "/shared/private/*": "ask"}},
            "agent": {
                "plan": {"permission": {"external_directory": {"/plan/*": "allow"}}},
                "build": {"permission": {"external_directory": {"/build/*": "allow"}}},
            },
        })
        document = _document(provider, PROFILE_AGENT_READONLY)
        assert document["model"] == "a/b"
        assert set(document["agent"]) == {"plan"}
        external = document["agent"]["plan"]["permission"]["external_directory"]
        assert list(external.items()) == list(_external(
            root, "plan",
            ("/shared/*", "allow"), ("/shared/private/*", "deny"), ("/plan/*", "allow"),
        ).items())
        assert "/build/*" not in external

    def test_a_failed_read_is_the_deny_alone_with_one_warning(
        self, provider, root, monkeypatch, capsys
    ):
        _hermetic_off(root, monkeypatch)
        _reader(monkeypatch, None)
        permission = _permission(provider, PROFILE_AGENT_READONLY)
        assert permission["external_directory"] == _external(root, "plan")
        _permission(provider, PROFILE_AGENT_READONLY)
        assert capsys.readouterr().err.count("could not be read") == 1

    def test_no_pure_and_no_redirect(self, provider, root, monkeypatch):
        _hermetic_off(root, monkeypatch)
        _reader(monkeypatch, {})
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, MODEL, runner_version=CEILING):
            argv = provider.build_agent_command("P", MODEL, PROFILE_AGENT_READONLY)
            delta = provider.resolve_env(MODEL, PROFILE_AGENT_READONLY)
        assert oc.PURE_FLAG not in argv and "--agent" in argv and "plan" in argv
        assert "XDG_CONFIG_HOME" not in delta
        assert "OPENCODE_CONFIG_CONTENT" in delta
        assert oc._CONFIG_HOME is None


# --------------------------------------------------------------------------- #
# Caching, and where the reader is (never) invoked
# --------------------------------------------------------------------------- #

class TestTheReadIsMemoised:
    def test_concurrent_dispatches_in_one_cwd_read_once_and_a_second_cwd_once_more(
        self, provider, root, monkeypatch
    ):
        fake = _reader(monkeypatch, _top({"/shared/*": "allow"}), delay=0.05)
        results: List[str] = []
        errors: List[BaseException] = []

        def dispatch(cwd):
            try:
                delta = _delta(provider, PROFILE_AGENT_WRITE, cwd=cwd)
                results.append(delta["OPENCODE_CONFIG_CONTENT"])
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [
            threading.Thread(target=dispatch, args=(str(root),)) for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert len(results) == 8 and len(set(results)) == 1
        assert len(fake.calls) == 1
        assert fake.calls[0]["cwd"] == str(root)
        assert fake.calls[0]["env"] is None  # the user's own env: no hermetic delta
        assert oc.PURE_FLAG not in fake.calls[0]["argv"]
        other = root / "other"
        other.mkdir()
        dispatch(str(other))
        dispatch(str(other))
        assert len(fake.calls) == 2

    def test_a_failed_read_is_cached_and_warns_once(
        self, provider, root, monkeypatch, capsys
    ):
        fake = _reader(monkeypatch, None)
        for _ in range(3):
            _delta(provider, PROFILE_AGENT_WRITE, cwd=str(root))
        assert len(fake.calls) == 1
        assert capsys.readouterr().err.count("could not be read") == 1

    def test_a_plain_get_env_never_spawns(self, provider, root, monkeypatch):
        _no_spawn(monkeypatch)
        assert "OPENCODE_CONFIG_CONTENT" not in (provider.get_env(MODEL) or {})
        with provider.hermetic_dispatch(PROFILE_AGENT_WRITE, MODEL):
            assert "OPENCODE_CONFIG_CONTENT" not in (provider.get_env(MODEL) or {})

    @pytest.mark.parametrize("profile", [PROFILE_AGENT_WRITE, PROFILE_AGENT_READONLY])
    def test_the_dry_run_argv_renderer_never_spawns(
        self, provider, root, monkeypatch, profile
    ):
        _hermetic_off(root, monkeypatch)
        _no_spawn(monkeypatch)
        argv, refusal = executor._preview_argv(
            provider, prompt="P", model=MODEL, profile=profile, schema=None
        )
        assert refusal is None and argv and oc.PURE_FLAG not in argv

    def test_the_auth_legs_config_read_takes_no_merge_slot(
        self, provider, root, monkeypatch
    ):
        """`continue_gate=False` callers spawn no model turn: no rows, no read
        of the user's config on their behalf (only their own `debug config`)."""
        fake = _reader(monkeypatch, {})
        with provider.hermetic_dispatch(PROFILE_AGENT_WRITE, MODEL, continue_gate=False):
            assert oc._ACTIVE.get() is None
            assert "OPENCODE_CONFIG_CONTENT" not in provider.resolve_env(
                MODEL, PROFILE_AGENT_WRITE
            )
        assert fake.calls == []


# --------------------------------------------------------------------------- #
# The rig: what the adapter emits is what was measured (T4.6 captures)
# --------------------------------------------------------------------------- #

def _rig(case: str) -> List[Path]:
    found = sorted(CAPTURE_ROOT.glob(f"{case}.config.json"))
    assert found, f"rig case {case} is not committed under {CAPTURE_ROOT}"
    return found


def _outcome(config_path: Path) -> Dict[str, Any]:
    stream = config_path.with_name(config_path.name.replace(".config.json", ".jsonl"))
    stderr = config_path.with_name(config_path.name.replace(".config.json", ".stderr.txt"))
    events = []
    for line in stream.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    tool = [e for e in events if e.get("type") == "tool_use"][0]["part"]["state"]
    finishes = [e for e in events if e.get("type") == "step_finish"]
    return {
        "status": tool["status"],
        "deny_rule": "prevents you from using" in str(tool.get("error", "")),
        "finish": finishes[-1]["part"]["reason"],
        "auto_rejecting": "auto-rejecting" in stderr.read_text(encoding="utf-8"),
    }


class TestTheRigMeasuredTheMerge:
    @pytest.mark.parametrize("agent", ["plan", "build"])
    def test_the_relisted_rules_are_the_ones_the_rig_ran(self, agent):
        """`t47_shared_toplevel_<a>`'s user file, reduced and re-listed by the
        adapter, is the agent-layer list `t47_shared_<a>`/`t47_private_<a>`
        carried; with it a read under `shared/` completes and one under
        `shared/private/` returns the deny to the model."""
        for control in _rig(f"t47_shared_toplevel_{agent}"):
            user = json.loads(control.read_text(encoding="utf-8"))["global_opencode_json"]
            declared = oc._reduce_debug_config(json.dumps(user))
            relisted = oc.relisted_external_rules(declared, agent)
            for case, completes in ((f"t47_shared_{agent}", True),
                                    (f"t47_private_{agent}", False)):
                config_path = control.with_name(f"{case}.config.json")
                config = json.loads(config_path.read_text(encoding="utf-8"))
                carried = config["OPENCODE_CONFIG_CONTENT"]["agent"][agent][
                    "permission"]["external_directory"]
                assert list(carried.items()) == [("*", "deny"), *relisted]
                outcome = _outcome(config_path)
                assert outcome["finish"] == "stop" and not outcome["auto_rejecting"]
                assert (outcome["status"] == "completed") is completes
                assert outcome["deny_rule"] is (not completes)

    def test_the_pinned_write_case_returns_the_deny(self):
        """`q2_ustar_agentlayer_build` and T4.8's `write_outofcwd_build`."""
        for case in ("q2_ustar_agentlayer_build", "write_outofcwd_build"):
            for config_path in _rig(case):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                assert "--pure" not in config["argv"]
                carried = config["OPENCODE_CONFIG_CONTENT"]["agent"]["build"]["permission"]
                block = oc.permission_block()
                assert carried == block
                outcome = _outcome(config_path)
                assert outcome == {"status": "error", "deny_rule": True,
                                   "finish": "stop", "auto_rejecting": False}

    def test_our_rows_appended_after_a_users_star_win_and_before_it_lose(self):
        for config_path in _rig("t47_inherited_star_build"):
            permission = json.loads(config_path.read_text(encoding="utf-8"))[
                "OPENCODE_CONFIG_CONTENT"]["agent"]["build"]["permission"]
            assert list(permission)[0] == "*"
            merged = json.loads(oc.merge_permission_content(
                json.dumps({"agent": {"build": {"permission": {"*": "allow"}}}}),
                "build", {"*": "deny"},
            ))["agent"]["build"]["permission"]
            # The rig case predates the trailing key; the rows it measured are
            # the same, and the trailing key only follows them.
            assert list(merged)[:-1] == list(permission)
            assert list(merged)[-1] == oc.TRAILING_EXTERNAL_KEY
            assert _outcome(config_path)["deny_rule"]
            control = config_path.with_name("t47_inherited_star_ctrl_build.config.json")
            assert _outcome(control)["status"] == "completed"


class TestTheTrailingKeyOverAConfigFilesAgentLayer:
    """A config *file*'s own ``agent.<a>.permission`` keeps its key positions in
    opencode's merge, so re-ordering the content cannot move them. The adapter
    repeats its ``external_directory`` list under
    :data:`oc.TRAILING_EXTERNAL_KEY`, the last key (rig ``useragent_relist_*``
    and ``useragent_ustar_*``, each against its ``_ctrl`` twin)."""

    @staticmethod
    def _adapter_content(config: Dict[str, Any], agent: str) -> Dict[str, Any]:
        """What the adapter emits over the rig's user file, rig paths kept."""
        declared = oc._reduce_debug_config(json.dumps(config["global_opencode_json"]))
        relisted = oc.relisted_external_rules(declared, agent)
        external = {"*": "deny", **dict(relisted)}
        return json.loads(oc.merge_permission_content(None, agent, external))

    @pytest.mark.parametrize("agent", ["plan", "build"])
    def test_a_file_allow_relisted_under_the_trailing_key_is_honoured(self, agent):
        for config_path in _rig(f"useragent_relist_{agent}"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            carried = config["OPENCODE_CONFIG_CONTENT"]["agent"][agent]["permission"]
            emitted = self._adapter_content(config, agent)["agent"][agent]["permission"]
            assert list(carried) == list(emitted)
            assert list(emitted)[-1] == oc.TRAILING_EXTERNAL_KEY
            assert list(carried[oc.TRAILING_EXTERNAL_KEY].items()) == list(
                emitted[oc.TRAILING_EXTERNAL_KEY].items())
            assert _outcome(config_path)["status"] == "completed"
            private = _outcome(config_path.with_name(
                f"useragent_relist_private_{agent}.config.json"))
            assert private["deny_rule"] and private["finish"] == "stop"
            control = _outcome(config_path.with_name(
                f"useragent_relist_ctrl_{agent}.config.json"))
            assert control["deny_rule"], "the control no longer shows the shadowing"

    @pytest.mark.parametrize("agent", ["plan", "build"])
    def test_a_file_star_allow_after_the_row_loses_to_the_trailing_key(self, agent):
        for config_path in _rig(f"useragent_ustar_{agent}"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            carried = config["OPENCODE_CONFIG_CONTENT"]["agent"][agent]["permission"]
            emitted = self._adapter_content(config, agent)["agent"][agent]["permission"]
            assert list(carried) == list(emitted)
            assert carried[oc.TRAILING_EXTERNAL_KEY] == emitted[oc.TRAILING_EXTERNAL_KEY]
            assert _outcome(config_path) == {"status": "error", "deny_rule": True,
                                             "finish": "stop", "auto_rejecting": False}
            control = _outcome(config_path.with_name(
                f"useragent_ustar_ctrl_{agent}.config.json"))
            assert control["status"] == "completed", "the control no longer shows the silent allow"

    def test_an_inherited_trailing_key_is_replaced_and_moved_last(self):
        inherited = {"agent": {"build": {"permission": {
            oc.TRAILING_EXTERNAL_KEY: {"*": "allow"}, "*": "allow"}}}}
        permission = json.loads(oc.merge_permission_content(
            json.dumps(inherited), "build", {"*": "deny", "/x/*": "allow"},
        ))["agent"]["build"]["permission"]
        assert list(permission)[0] == "*"
        assert list(permission)[-1] == oc.TRAILING_EXTERNAL_KEY
        assert list(permission[oc.TRAILING_EXTERNAL_KEY].items()) == [
            ("*", "deny"), ("/x/*", "allow")]
