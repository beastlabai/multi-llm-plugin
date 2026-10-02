"""T4.1/T4.2 — the permission rows every hermetic opencode slot carries.

plans/opencode-validator-false-rate-limit-and-denied-turn.md §4.2 steps 1-2.

`opencode run` auto-rejects every permission `ask`, and a rejected permission
ends the agent loop with exit 0 and no answer (run b38de8bb). A `deny` goes
back to the model and the loop continues. So every hermetic slot
(`agent-readonly` under `agent.plan`, `generate` under opencode's default agent
`agent.build`) carries the three §4.2 step 1 rows as `deny`, merged into
`OPENCODE_CONFIG_CONTENT` beside any provider block, plus the version-gated
`experimental.continue_loop_on_deny` (T4.2).

The rig tests at the bottom bind this module to the offline captures under
`tests/fixtures/opencode/capture/` (T4.6): the rows the adapter emits are the
rows the rig measured, and the built-in allows the adapter re-lists are the
ones the rig saw our deny shadow. Everything here is offline and spends $0.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import llm_client  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers import opencode as oc  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
)

CAPTURE_ROOT = Path(__file__).parent / "fixtures" / "opencode" / "capture"
FLOOR, CEILING = oc.OpenCodeProvider.CONTINUE_LOOP_ON_DENY_RANGE

#: The three rows as the rig measured them (§4.2 step 1), before the built-in
#: allows are re-listed.
ROWS = {
    "external_directory": {"*": "deny"},
    "read": {"*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"},
    "doom_loop": "deny",
}


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
def dispatch_env(tmp_path, monkeypatch):
    """A user config home, plus the names the built-in allows resolve from."""
    home = tmp_path / "user-config"
    (home / "opencode").mkdir(parents=True)
    (home / "opencode" / "opencode.json").write_text(
        json.dumps({
            "provider": {
                "only-in-user-config": {
                    "options": {"baseURL": "https://gateway.invalid/v1"},
                    "models": {"private-model": {}},
                }
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
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


def _child_env(provider, model: str, profile: str) -> dict:
    delta = llm_client.expand_env_home(provider.resolve_env(model, profile))
    env = {**os.environ, **delta}
    for key in provider.get_remove_env() or ():
        env.pop(key, None)
    return env


def _content(provider, profile: str, model: str = "opencode/sonnet", **kwargs) -> dict:
    with provider.hermetic_dispatch(profile, model, **kwargs) as refusal:
        assert refusal is None
        env = _child_env(provider, model, profile)
    return json.loads(env["OPENCODE_CONFIG_CONTENT"])


def _expected_rows(root: Path, agent: str) -> dict:
    external = {"*": "deny", str(root / "tmp" / "opencode" / "*"): "allow"}
    if agent == oc.READONLY_AGENT:
        external[str(root / "data" / "opencode" / "plans" / "*")] = "allow"
    return {**ROWS, "external_directory": external}


# --------------------------------------------------------------------------- #
# T4.1 — the rows, where they go, and what else is in the content
# --------------------------------------------------------------------------- #

class TestTheRowsAreCarried:
    def test_agent_readonly_with_a_built_in_provider_carries_them_under_plan(
        self, provider, dispatch_env
    ):
        """The b38de8bb case: `openrouter/*` is built in, and before T4.1 its
        slot had no `OPENCODE_CONFIG_CONTENT` at all."""
        document = _content(provider, PROFILE_AGENT_READONLY, "openrouter/some/model")
        assert list(document["agent"]) == ["plan"]
        assert document["agent"]["plan"]["permission"] == _expected_rows(
            dispatch_env, "plan"
        )
        assert "permission" not in document, (
            "top-level `permission` loses to a user's `{\"*\": \"allow\"}` "
            "(pre-flight); the rows go under the agent layer only"
        )
        assert "provider" not in document

    def test_generate_carries_them_under_build_with_no_agent_flag(
        self, provider, dispatch_env
    ):
        """`generate` runs opencode's default agent, `build`, with no `--agent`."""
        with provider.hermetic_dispatch(PROFILE_GENERATE, "opencode/sonnet") as r:
            assert r is None
            env = _child_env(provider, "opencode/sonnet", PROFILE_GENERATE)
            argv = provider.build_command("hi", "opencode/sonnet")
        document = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert list(document["agent"]) == [oc.WRITE_AGENT]
        assert document["agent"]["build"]["permission"] == _expected_rows(
            dispatch_env, "build"
        )
        assert "--agent" not in argv

    def test_no_row_grants_edit_or_bash(self, provider, dispatch_env):
        """The rows only add denies: `build`'s `edit`/`bash` stay its own."""
        for profile in (PROFILE_AGENT_READONLY, PROFILE_GENERATE):
            document = _content(provider, profile)
            (permission,) = [a["permission"] for a in document["agent"].values()]
            assert set(permission) == {"external_directory", "read", "doom_loop"}

    def test_a_declared_provider_and_the_rows_travel_together(
        self, provider, dispatch_env, monkeypatch
    ):
        _use_config(
            dispatch_env, monkeypatch,
            "providers:\n"
            "  opencode:\n"
            "    config_providers:\n"
            "      my-gateway:\n"
            "        options:\n"
            "          baseURL: https://gateway.example.com/v1\n"
            "        models:\n"
            "          my-model: {}\n",
        )
        document = _content(provider, PROFILE_AGENT_READONLY, "my-gateway/my-model")
        assert list(document["provider"]) == ["my-gateway"]
        assert document["agent"]["plan"]["permission"]["doom_loop"] == "deny"

    def test_an_inherited_opencode_permission_is_stripped_while_hermetic(
        self, provider, dispatch_env, monkeypatch
    ):
        monkeypatch.setenv(
            "OPENCODE_PERMISSION", '{"external_directory": {"*": "allow"}}'
        )
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)
        assert "OPENCODE_PERMISSION" not in env
        assert "OPENCODE_PERMISSION" in oc.CONFIG_ENV_NAMES

    def test_a_non_hermetic_slot_keeps_opencode_permission(
        self, provider, dispatch_env, monkeypatch
    ):
        """`hermetic: off` is the user's own config in force: nothing is stripped.

        The rows still travel (T4.7 merges them over the inherited content,
        `test_opencode_write_profile.py`), with no hermetic rebuild."""
        _use_config(
            dispatch_env, monkeypatch, "providers:\n  opencode:\n    hermetic: 'off'\n"
        )
        monkeypatch.setenv("OPENCODE_PERMISSION", '{"read": "allow"}')
        with provider.hermetic_dispatch(PROFILE_AGENT_READONLY, "opencode/sonnet"):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)
        assert env["OPENCODE_PERMISSION"] == '{"read": "allow"}'
        document = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert "provider" not in document
        assert oc._CONFIG_HOME is None

    def test_the_write_profile_gets_no_hermetic_rebuild(
        self, provider, dispatch_env, monkeypatch
    ):
        """`agent-write` is T4.7's: no config home, no redirect, nothing stripped.

        Its content is the merge over the inherited value
        (`test_opencode_write_profile.py`), never this rebuild."""
        monkeypatch.setenv("OPENCODE_PERMISSION", '{"read": "allow"}')
        with provider.hermetic_dispatch(PROFILE_AGENT_WRITE, "opencode/sonnet") as r:
            assert r is None
            delta = provider.resolve_env("opencode/sonnet", PROFILE_AGENT_WRITE)
            removed = provider.get_remove_env() or ()
        assert "XDG_CONFIG_HOME" not in delta
        assert oc.PROJECT_CONFIG_ENV not in delta
        assert "provider" not in json.loads(delta["OPENCODE_CONFIG_CONTENT"])
        assert "OPENCODE_PERMISSION" not in removed
        assert "OPENCODE_CONFIG_CONTENT" not in removed
        assert oc._CONFIG_HOME is None


class TestTheRefusalFiresFirst:
    """§9 item 4: the unreconstructible-provider refusal precedes the merge."""

    MODEL = "only-in-user-config/private-model"

    def test_the_content_builder_raises_before_any_row_is_built(
        self, provider, dispatch_env, monkeypatch
    ):
        def _must_not_run(*_args, **_kwargs):  # pragma: no cover - the point
            raise AssertionError("the permission rows were built before the refusal")

        monkeypatch.setattr(oc, "permission_block", _must_not_run)
        monkeypatch.setattr(oc, "continue_loop_on_deny_allowed", _must_not_run)
        with pytest.raises(oc._UnreconstructibleProvider):
            provider._hermetic_config_content(
                self.MODEL, profile=PROFILE_AGENT_READONLY, runner_version=CEILING
            )

    def test_a_refused_slot_produces_no_content_and_claims_no_home(
        self, provider, dispatch_env
    ):
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, self.MODEL, runner_version=CEILING
        ) as refusal:
            assert refusal and "only-in-user-config" in refusal
            assert oc._ACTIVE.get() is None
            delta = provider.get_env(self.MODEL)
        assert "OPENCODE_CONFIG_CONTENT" not in delta
        assert oc._CONFIG_HOME is None


class TestSerialization:
    def test_the_dump_is_deterministic(self, provider, dispatch_env):
        first = provider._hermetic_config_content(
            "opencode/sonnet", profile=PROFILE_AGENT_READONLY, runner_version=CEILING
        )
        second = provider._hermetic_config_content(
            "opencode/sonnet", profile=PROFILE_AGENT_READONLY, runner_version=CEILING
        )
        assert first == second
        document = json.loads(first)
        assert list(document) == sorted(document)
        assert list(document["agent"]["plan"]["permission"]["external_directory"])[0] == "*"

    def test_rule_maps_keep_insertion_order_and_the_rest_is_sorted(self):
        """`findLast`: a rule map's order is its meaning (§4.4 step 3)."""
        dumped = oc._dump_config({
            "provider": {"zeta": {"models": {"b": {}, "a": {}}}, "alpha": {}},
            "agent": {"build": {"permission": {
                "external_directory": {
                    "*": "deny", "/zz/*": "allow", "/aa/*": "deny", "!x/*": "allow",
                },
                "doom_loop": "deny",
            }}},
        })
        document = json.loads(dumped)
        assert list(document) == ["agent", "provider"]
        assert list(document["provider"]) == ["alpha", "zeta"]
        assert list(document["provider"]["zeta"]["models"]) == ["a", "b"]
        external = document["agent"]["build"]["permission"]["external_directory"]
        assert list(external) == ["*", "/zz/*", "/aa/*", "!x/*"], (
            "a pattern that sorts before `*` (`!x/*`) would land ahead of the "
            "deny and be shadowed by it"
        )
        assert list(document["agent"]["build"]["permission"]) == [
            "external_directory", "doom_loop"
        ]

    def test_the_env_read_rows_keep_the_example_allow_last(self, provider, dispatch_env):
        document = _content(provider, PROFILE_AGENT_READONLY)
        read = document["agent"]["plan"]["permission"]["read"]
        assert list(read) == ["*.env", "*.env.*", "*.env.example"]


class TestTheBuiltinAllows:
    def test_they_resolve_the_way_node_and_opencode_do(self):
        env = {"TMP": "/x/tmp/", "TEMP": "/y", "HOME": "/h"}
        assert oc.builtin_external_allows("build", env) == [
            os.path.join("/x/tmp", "opencode", "*")
        ]
        assert oc.builtin_external_allows("plan", env) == [
            os.path.join("/x/tmp", "opencode", "*"),
            os.path.join("/h", ".local", "share", "opencode", "plans", "*"),
        ]
        bare = oc.builtin_external_allows("build", {"HOME": "/h"})
        assert bare == [os.path.join(os.sep, "tmp", "opencode", "*")]

    def test_permission_block_is_fresh_on_every_call(self):
        first = oc.permission_block("plan", {"HOME": "/h"})
        first["external_directory"]["/extra/*"] = "allow"
        assert "/extra/*" not in oc.permission_block("plan", {"HOME": "/h"})["external_directory"]
        assert oc.permission_block() == ROWS


# --------------------------------------------------------------------------- #
# T4.2 — the version-gated experimental key
# --------------------------------------------------------------------------- #

class TestContinueLoopOnDenyGate:
    def test_the_range_is_its_own_constant(self):
        assert (FLOOR, CEILING) == ("1.17.15", "1.18.34")
        assert oc.OpenCodeProvider.verified_version != CEILING

    def test_the_ceiling_is_backed_by_a_committed_rig_capture(self):
        """The bound moves only on a rig measurement (opencode.py's comment): the
        ceiling's own capture must exist and show the key keeping the loop going."""
        rig = Path(__file__).parent / "fixtures" / "opencode" / "capture" / CEILING
        rows = {
            line.split("\t")[0]: line.split("\t")
            for line in (rig / "summary.tsv").read_text("utf-8").splitlines()
        }
        for agent in ("plan", "build"):
            cont = rows[f"t40_cont_{agent}"]
            assert "tool_posts=2" in cont and "finish=stop" in cont
            assert "exit=0" in cont

    @pytest.mark.parametrize("version", [FLOOR, CEILING])
    def test_in_range_it_is_present(self, provider, dispatch_env, capsys, version):
        for profile in (PROFILE_AGENT_READONLY, PROFILE_GENERATE):
            document = _content(provider, profile, runner_version=version)
            assert document["experimental"] == {"continue_loop_on_deny": True}
        assert "continue_loop_on_deny" not in capsys.readouterr().err

    def test_above_the_ceiling_it_is_absent_and_warns_once(
        self, provider, dispatch_env, capsys
    ):
        for _ in range(3):
            document = _content(provider, PROFILE_AGENT_READONLY, runner_version="1.19.0")
            assert "experimental" not in document
            assert document["agent"]["plan"]["permission"]["doom_loop"] == "deny"
        err = capsys.readouterr().err
        assert err.count("continue_loop_on_deny") == 1
        assert "1.19.0" in err and FLOOR in err and CEILING in err

    def test_below_the_floor_it_is_absent_with_no_warning(
        self, provider, dispatch_env, capsys
    ):
        document = _content(provider, PROFILE_AGENT_READONLY, runner_version="1.17.0")
        assert "experimental" not in document
        assert capsys.readouterr().err == ""

    def test_below_a_single_build_range_is_silent_too(self, capsys):
        """The plan's own example: `verified_version` under a 1.18.32-only range."""
        assert not oc.continue_loop_on_deny_allowed(
            oc.OpenCodeProvider.verified_version, ("1.18.32", "1.18.32")
        )
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("version", [None, "", "unknown"])
    def test_an_unread_version_is_absent_and_warns_once(
        self, provider, dispatch_env, capsys, version
    ):
        for _ in range(2):
            document = _content(provider, PROFILE_AGENT_READONLY, runner_version=version)
            assert "experimental" not in document
        err = capsys.readouterr().err
        assert err.count("was not read") == 1

    def test_the_catalog_probe_neither_needs_nor_warns(self, provider, dispatch_env, capsys):
        with provider.hermetic_dispatch(
            PROFILE_AGENT_READONLY, "opencode/sonnet", continue_gate=False
        ):
            env = _child_env(provider, "opencode/sonnet", PROFILE_AGENT_READONLY)
        assert "experimental" not in json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert capsys.readouterr().err == ""


class TestTheVersionComesFromThePreflight:
    """The dispatch hands the preflight's cached read in; it never spawns."""

    @pytest.fixture
    def no_spawn(self, monkeypatch):
        def _refuse(*_args, **_kwargs):  # pragma: no cover - the point
            raise AssertionError("a dispatch spawned a version probe")

        monkeypatch.setattr(executor, "_run_probe", _refuse)
        monkeypatch.setattr(executor, "_run_sandbox_probe", _refuse)
        monkeypatch.setattr(oc.subprocess, "run", _refuse)

    def _dispatch(self, provider) -> dict:
        seen: dict = {}

        def _capture(**kwargs):
            seen["env"] = kwargs["provider"].resolve_env(
                kwargs["model"], kwargs["profile"]
            )
            return executor._Capture()

        with patch.object(executor, "_spawn_in_context", _capture):
            executor._spawn_child(
                provider=provider, model="opencode/sonnet", prompt="hi",
                profile=PROFILE_AGENT_READONLY, transport="argv", timeout=5, cwd=None,
            )
        return json.loads(seen["env"]["OPENCODE_CONFIG_CONTENT"])

    def test_a_cached_in_range_version_emits_the_key(
        self, provider, dispatch_env, no_spawn
    ):
        executor._VERSION_CACHE["opencode"] = CEILING
        assert self._dispatch(provider)["experimental"] == {
            "continue_loop_on_deny": True
        }

    def test_a_cached_unknown_omits_it_with_one_warning(
        self, provider, dispatch_env, no_spawn, capsys
    ):
        executor._VERSION_CACHE["opencode"] = "unknown"
        assert "experimental" not in self._dispatch(provider)
        assert "experimental" not in self._dispatch(provider)
        assert capsys.readouterr().err.count("was not read") == 1

    def test_no_cache_entry_is_unknown_and_nothing_is_spawned(
        self, provider, dispatch_env, no_spawn
    ):
        assert "opencode" not in executor._VERSION_CACHE
        assert "experimental" not in self._dispatch(provider)

    def test_the_lane_a_fan_out_carries_the_rows_under_build(
        self, provider, dispatch_env
    ):
        """Lane A runs no version preflight, so the key is omitted there."""
        captured: dict = {}

        def _run(cmd, **kwargs):
            captured["env"] = kwargs.get("env") or {}
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
        document = json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])
        assert document["agent"]["build"]["permission"] == _expected_rows(
            dispatch_env, "build"
        )
        assert "experimental" not in document


# --------------------------------------------------------------------------- #
# The T4.6 rig: what the adapter emits is what was measured
# --------------------------------------------------------------------------- #

def _rig_cases(case: str) -> list:
    found = sorted(CAPTURE_ROOT.glob(f"*/{case}.config.json"))
    assert found, f"rig case {case} is not committed under {CAPTURE_ROOT}"
    return found


def _events(config_path: Path) -> list:
    stream = config_path.with_name(config_path.name.replace(".config.json", ".jsonl"))
    events = []
    for line in stream.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def _without_builtin_allows(rows: dict) -> dict:
    return {**rows, "external_directory": {"*": rows["external_directory"]["*"]}}


class TestTheRigMeasuredTheseRows:
    @pytest.mark.parametrize("agent", ["plan", "build"])
    def test_the_pinned_case_a_users_star_allow_does_not_beat_the_agent_layer(
        self, agent
    ):
        """`q2_ustar_agentlayer_<agent>` (§4.2 step 1): a user top-level
        `{"external_directory": "ask", "*": "allow"}` plus our rows under
        `agent.<agent>.permission` → the deny returns and the turn ends `stop`."""
        for config_path in _rig_cases(f"q2_ustar_agentlayer_{agent}"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            user = config["global_opencode_json"]["permission"]
            assert user == {"external_directory": "ask", "*": "allow"}
            carried = config["OPENCODE_CONFIG_CONTENT"]["agent"][agent]["permission"]
            assert carried == _without_builtin_allows(oc.permission_block())
            events = _events(config_path)
            tool = [e for e in events if e.get("type") == "tool_use"]
            assert tool and tool[0]["part"]["state"]["status"] == "error"
            assert "prevents you from using" in tool[0]["part"]["state"]["error"]
            finishes = [e for e in events if e.get("type") == "step_finish"]
            assert finishes[-1]["part"]["reason"] == "stop", config_path

    @pytest.mark.parametrize(
        "case,agent",
        [("builtin_tmp_plan", "plan"), ("builtin_tmp_build", "build"),
         ("builtin_plansdir_plan", "plan")],
    )
    def test_the_relisted_allows_are_the_ones_the_deny_shadowed(self, case, agent):
        """The deny error lists the resolved rules. Every allow before our
        `"*": "deny"`, except `tool-output` (re-appended after the agent layer)
        and discovered skill dirs (not enumerable), is what the adapter
        re-lists, resolved from the rig's own env."""
        for config_path in _rig_cases(case):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tool = [e for e in _events(config_path) if e.get("type") == "tool_use"]
            error = tool[0]["part"]["state"]["error"]
            rules = json.loads(re.search(r"(\[.*\])", error, re.S).group(1))
            external = [r for r in rules if r["permission"] == "external_directory"]
            ours = next(
                i for i, r in enumerate(external)
                if r["pattern"] == "*" and r["action"] == "deny"
            )
            shadowed = [
                r["pattern"] for r in external[:ours]
                if r["action"] == "allow"
                and "/tool-output/" not in r["pattern"]
                and "/skills/" not in r["pattern"]
            ]
            assert oc.builtin_external_allows(agent, config["env"]) == shadowed, (
                config_path
            )
