"""Plan §8 (opencode-validator-false-rate-limit-and-denied-turn): providers
declared only in config, and the auth probe under the dispatch env.

* T8.2 — ``_read_declared_providers``: ``opencode [--pure] debug config``
  reduced in memory to ``{provider: (custom, key_declared)}`` plus the user's
  ``external_directory`` rules per layer; ``None`` on any failure.
* T8.3 — ``OpenCodeProvider.config_credential_source`` (inside the dispatch's
  hermetic context, env from the executor's shared helper) and the structural
  leg's hermetic reconstructibility refusal.
* T8.4 + T8.13 — ``_detect_auth``: the lazy config leg, the ``config`` state,
  the widened cache key, and the probe running under ``dispatch_child_env``.

Every ``debug config`` here is canned (the pinned release's capture,
``tests/fixtures/opencode/debug_config.<pin>.json``, or a literal); the spawn
seam ``opencode._spawn_debug_config`` is patched, so nothing runs a real
binary. ``auth list`` is the committed envelope of the same release.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers import base as pbase  # noqa: E402
from utils.providers import opencode as oc  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
)

from tests import opencode_fixtures as oc_pin  # noqa: E402

DEBUG_CONFIG = json.loads(oc_pin.DEBUG_CONFIG.read_text(encoding="utf-8"))
AUTH_LIST = json.loads(oc_pin.AUTH_LIST.read_text(encoding="utf-8"))

KEY = "driver.subagent_runner"


def _stdout(case: str) -> str:
    return DEBUG_CONFIG["cases"][case]["stdout"]


def _completed(stdout: str, code: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["opencode"], code, stdout, "")


def _auth_run(case: str) -> subprocess.CompletedProcess:
    run = AUTH_LIST["cases"][case]
    return subprocess.CompletedProcess(
        ["opencode", "auth", "list"], run["exit"], run["stdout"], run["stderr"]
    )


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    oc.sweep_config_home()
    oc.reset_hermetic_state()
    executor.reset_auth_cache()
    # Nothing here may reach the developer's `--pure` help sniff.
    monkeypatch.setattr(oc, "_read_help_tokens", lambda binary: frozenset({oc.PURE_FLAG}))
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    yield
    oc.sweep_config_home()
    oc.reset_hermetic_state()
    executor.reset_auth_cache()


@pytest.fixture
def provider():
    return registry.get_provider("opencode")


@pytest.fixture
def user_xdg(tmp_path, monkeypatch):
    """An empty user config home, so no real file is ever read.

    ``HOME`` (for ``~/.opencode/``) and the managed directory move too: the
    T8.3 key read walks every local layer opencode loads, and none of them may
    be the developer's own."""
    home = tmp_path / "user-config"
    (home / "opencode").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    # Windows' `expanduser` reads USERPROFILE, never HOME; some layer reads use it.
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user-home"))
    monkeypatch.setenv("OPENCODE_TEST_MANAGED_CONFIG_DIR", str(tmp_path / "no-managed-config"))
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    return home


def _use_config(tmp_path, monkeypatch, body: str) -> None:
    path = tmp_path / "multi-llm-config.yaml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
    registry._config = None
    registry._config_key = None


def _replay_files(case: str, user_xdg: Path, monkeypatch) -> None:
    """Lay down a capture's own global config files and set its env.

    The pinned build prints every ``apiKey`` as ``"***"`` (set, unset, empty
    and file references alike), so the auth leg reads the reference from the
    config sources themselves. A replay that fed only the capture's stdout
    would grade a config that does not exist; this writes the one it was
    captured from. Global-only cases: a project file would need a ``cwd``.
    """
    run = DEBUG_CONFIG["cases"][case]
    assert not run["project_files"], case
    for name, body in run["global_files"].items():
        text = body if isinstance(body, str) else json.dumps(body)
        (user_xdg / "opencode" / name).write_text(text, encoding="utf-8")
    for key, value in run["env"].items():
        if key != "OPENCODE_DISABLE_MODELS_FETCH":
            monkeypatch.setenv(key, value)


class _DebugConfig:
    """A recording stand-in for the ``debug config`` spawn seam."""

    def __init__(self, stdout: Optional[str], code: int = 0, *, dispatch: Optional[str] = None):
        self.stdout = stdout
        self.dispatch = dispatch
        self.code = code
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, argv, env, cwd, timeout):
        self.calls.append(
            {"argv": list(argv), "env": None if env is None else dict(env),
             "cwd": cwd, "timeout": timeout, "active": oc._ACTIVE.get()}
        )
        text = self.stdout
        if env is not None and self.dispatch is not None:
            text = self.dispatch
        if text is None:
            return None
        return _completed(text, self.code)


# --------------------------------------------------------------------------- #
# T8.2 — the reader and its reduction
# --------------------------------------------------------------------------- #

class TestTheReduction:
    """Canned ``debug config`` stdout → booleans and rule lists, never text."""

    def _read(self, monkeypatch, stdout, code=0):
        fake = _DebugConfig(stdout, code)
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        return oc._read_declared_providers(None, None, False), fake

    @pytest.mark.parametrize(
        "case, prefix, custom, key_declared",
        [
            ("custom_inline_key", "stub", True, True),
            ("custom_keyless", "keyless", True, False),
            # 1.18.34 masks every declared key as "***", an unset `{env:}`
            # reference included (1.18.32 printed ""): `key_declared` means
            # "declared". Custom rows pass on `custom` either way.
            ("custom_envkey_unset", "envkey", True, True),
            ("custom_envkey_set", "envkey", True, True),
            ("builtin_override_keyed", "openrouter", False, True),
            ("builtin_override_keyless", "openrouter", False, False),
            # The T8.3 shapes: declared, every one of them, whether or not the
            # reference resolves. Resolving it is `_config_key_resolves`'s job.
            ("builtin_override_envkey_unset", "openrouter", False, True),
            ("builtin_override_envkey_set", "openrouter", False, True),
            ("builtin_override_empty_key", "openrouter", False, True),
            ("builtin_override_filekey", "openrouter", False, True),
            ("jsonc_commented", "stub", True, True),
            ("project_layer", "stub", True, True),
            ("opencode_config_file", "stub", True, True),
            ("hermetic_config_content", "stub", True, False),
            ("hermetic_pure_config_content", "stub", True, False),
        ],
    )
    def test_each_captured_row_reduces_to_its_verdict(
        self, monkeypatch, case, prefix, custom, key_declared
    ):
        declared, _ = self._read(monkeypatch, _stdout(case))
        assert declared.providers[prefix] == oc.ProviderDecl(custom, key_declared)

    def test_the_pinned_build_masks_every_key(self):
        """MEASURED on the pin: `debug config` prints every declared `apiKey`
        as `"***"`, whatever its value — a canary, an unset `{env:}`
        reference, an empty string. Pinned so a change is noticed: an unmasked
        build would put the T8.3 env-key refusal back in reach."""
        for case, run in DEBUG_CONFIG["cases"].items():
            assert "sk-canary" not in run["stdout"], case
        keyed = ("custom_inline_key", "custom_envkey_unset", "custom_envkey_set",
                 "builtin_override_keyed", "builtin_override_envkey_unset",
                 "builtin_override_envkey_set", "builtin_override_empty_key",
                 "builtin_override_filekey", "jsonc_commented", "project_layer",
                 "opencode_config_file")
        for case in keyed:
            providers = json.loads(_stdout(case))["provider"]
            keys = {block["options"]["apiKey"] for block in providers.values()}
            assert keys == {"***"}, case

    def test_a_missing_file_reference_fails_the_whole_read(self, monkeypatch):
        """MEASURED on the pin: `{file:}` naming a missing file is a config
        error (exit 1), not an empty key — so it reads "unknown"."""
        run = DEBUG_CONFIG["cases"]["builtin_override_filekey_missing"]
        assert run["exit"] == 1 and "bad file reference" in run["stderr"]
        declared, _ = self._read(monkeypatch, run["stdout"], run["exit"])
        assert declared is None

    @pytest.mark.parametrize(
        "case", ["nothing_declared", "project_layer_disabled", "hermetic_pure_permission_only"]
    )
    def test_no_provider_object_declares_nothing(self, monkeypatch, case):
        declared, _ = self._read(monkeypatch, _stdout(case))
        assert declared.providers == {}

    def test_the_hermetic_rebuild_hides_the_users_own_provider(self, monkeypatch):
        """The capture carries a global `userstub` the redirect must hide."""
        declared, _ = self._read(monkeypatch, _stdout("hermetic_config_content"))
        assert set(declared.providers) == {"stub"}

    def test_the_permission_block_is_ignored_by_the_provider_verdict(self, monkeypatch):
        """§9 item 4: T4.1's block rides the content; only `provider` is read."""
        declared, _ = self._read(monkeypatch, _stdout("hermetic_pure_config_content"))
        assert "permission" in json.loads(_stdout("hermetic_pure_config_content"))["agent"]["plan"]
        assert declared.providers == {"stub": oc.ProviderDecl(True, False)}

    def test_the_reduction_carries_no_key(self, monkeypatch):
        for case in DEBUG_CONFIG["cases"]:
            declared, _ = self._read(monkeypatch, _stdout(case))
            assert "sk-canary" not in repr(declared), case

    @pytest.mark.parametrize(
        "stdout, code",
        [
            ("{not json sk-canary-not-a-key-9", 0),
            ("[1, 2]", 0),
            ("", 0),
            (_stdout("custom_inline_key"), 1),
        ],
        ids=["non-json", "not-an-object", "empty", "non-zero-exit"],
    )
    def test_a_bad_run_is_none(self, monkeypatch, stdout, code):
        declared, _ = self._read(monkeypatch, stdout, code)
        assert declared is None

    def test_a_spawn_failure_is_none(self, monkeypatch):
        declared, _ = self._read(monkeypatch, None)
        assert declared is None

    def test_a_raising_spawn_is_none(self, monkeypatch):
        def boom(*args):
            raise ValueError("sk-canary-not-a-key-1")

        monkeypatch.setattr(oc, "_spawn_debug_config", boom)
        assert oc._read_declared_providers(None, None, False) is None

    def test_a_tripwire_assertion_is_not_swallowed(self, monkeypatch):
        """The except is narrow: a test's `AssertionError` tripwire surfaces
        instead of reading as a failed read (which would pass the test)."""
        def tripwire(*args):
            raise AssertionError("opencode debug config was spawned")

        monkeypatch.setattr(oc, "_spawn_debug_config", tripwire)
        with pytest.raises(AssertionError):
            oc._read_declared_providers(None, None, False)

    def test_a_non_mapping_provider_value_is_unknown(self, monkeypatch):
        declared, _ = self._read(monkeypatch, json.dumps({"provider": ["x"]}))
        assert declared is not None and declared.providers is None

    def test_pure_and_the_argv(self, monkeypatch):
        fake = _DebugConfig("{}")
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        oc._read_declared_providers({"A": "1"}, "/repo", True)
        oc._read_declared_providers(None, None, False)
        assert fake.calls[0]["argv"] == ["opencode", "--pure", "debug", "config"]
        assert fake.calls[0]["env"] == {"A": "1"} and fake.calls[0]["cwd"] == "/repo"
        assert fake.calls[1]["argv"] == ["opencode", "debug", "config"]

    def test_the_timeout_is_the_ceiling_or_what_is_left(self, monkeypatch):
        fake = _DebugConfig("{}")
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        oc._read_declared_providers(None, None, False)
        oc._read_declared_providers(None, None, False, timeout=2.5)
        oc._read_declared_providers(None, None, False, timeout=99)
        assert [c["timeout"] for c in fake.calls] == [30.0, 2.5, 30.0]

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_an_exhausted_timeout_spawns_nothing(self, monkeypatch, timeout):
        fake = _DebugConfig("{}")
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        assert oc._read_declared_providers(None, None, False, timeout=timeout) is None
        assert fake.calls == []

    def test_a_real_expiry_is_none(self, monkeypatch):
        def expired(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

        monkeypatch.setattr(oc.subprocess, "run", expired)
        assert oc._read_declared_providers(None, None, False, timeout=0.1) is None


def _layers(permission=None, agents=None) -> Dict[str, oc.ExternalDirLayer]:
    document: Dict[str, Any] = {}
    if permission is not None:
        document["permission"] = permission
    if agents is not None:
        document["agent"] = agents
    return oc._reduce_debug_config(json.dumps(document)).external_directory


class TestExternalDirectoryRules:
    """T8.2's second half: the user's rules per layer, for T4.7 to merge."""

    LAYERS = {
        "top": lambda perm: _layers(permission=perm)["top"],
        "plan": lambda perm: _layers(agents={"plan": {"permission": perm}})["plan"],
        "build": lambda perm: _layers(agents={"build": {"permission": perm}})["build"],
    }

    @pytest.fixture(params=["top", "plan", "build"])
    def layer(self, request):
        return self.LAYERS[request.param]

    def test_absent(self):
        for name in oc.EXTERNAL_DIR_LAYERS:
            assert _layers()[name] == oc.ExternalDirLayer((), False)
        assert _layers(agents={"plan": {}})["plan"] == oc.ExternalDirLayer((), False)

    def test_no_external_directory_key(self, layer):
        assert layer({"read": {"*": "allow"}}) == oc.ExternalDirLayer((), False)

    def test_a_whole_permission_string(self, layer):
        assert layer("allow") == oc.ExternalDirLayer((("*", "allow"),), False)

    @pytest.mark.parametrize("action", ["ask", "allow", "deny"])
    def test_an_external_directory_string(self, layer, action):
        assert layer({"external_directory": action}) == oc.ExternalDirLayer(
            (("*", action),), False
        )

    def test_an_ordered_map_keeps_its_order(self, layer):
        rules = {"/zz/*": "allow", "*": "ask", "/aa/*": "deny", "!x/*": "allow"}
        assert layer({"external_directory": rules}).rules == tuple(rules.items())

    @pytest.mark.parametrize(
        "value",
        ["sometimes", 3, ["/a/*"], {"/a/*": 1}, {"/a/*": "maybe"}, None],
        ids=["unknown-string", "number", "list", "non-string-value", "unknown-value", "null"],
    )
    def test_any_other_shape_drops_the_whole_layer(self, layer, value):
        assert layer({"external_directory": value}) == oc.ExternalDirLayer((), True)

    def test_an_unknown_whole_permission_string_is_malformed(self, layer):
        assert layer("sometimes") == oc.ExternalDirLayer((), True)

    def test_a_permission_level_catch_all_is_not_returned(self, layer):
        got = layer({"external_directory": "ask", "*": "allow"})
        assert got == oc.ExternalDirLayer((("*", "ask"),), False)

    def test_the_captured_user_rules(self):
        declared = oc._reduce_debug_config(_stdout("user_external_directory_rules"))
        layers = declared.external_directory
        assert layers["top"].rules == (
            ("/zz/*", "allow"), ("/shared/*", "allow"),
            ("/shared/private/*", "deny"), ("*", "ask"),
        )
        assert layers["plan"].rules == (("/aa/*", "ask"),)
        assert layers["build"].rules == (("*", "deny"),)

    def test_a_reader_failure_is_none_not_empty_rules(self, monkeypatch):
        monkeypatch.setattr(oc, "_spawn_debug_config", _DebugConfig(None))
        assert oc._read_declared_providers(None, None, False) is None


# --------------------------------------------------------------------------- #
# T8.3 — config_credential_source and the structural leg
# --------------------------------------------------------------------------- #

class TestConfigCredentialSource:

    def test_the_base_default_is_unknown_for_every_other_adapter(self):
        for name, adapter in registry._PROVIDERS.items():
            if name == "opencode":
                continue
            assert adapter.config_credential_source("x/y") == pbase.CONFIG_AUTH_UNKNOWN, name

    @pytest.mark.parametrize(
        "case, model, expected",
        [
            ("custom_inline_key", "stub/stub-model", "declared"),
            ("custom_keyless", "keyless/m1", "declared"),
            ("custom_envkey_unset", "envkey/m2", "declared"),
            ("custom_envkey_set", "envkey/m2", "declared"),
            ("builtin_override_keyed", "openrouter/x/y", "declared"),
            ("builtin_override_keyless", "openrouter/x/y", "unknown"),
            # T8.3 on the pin: all four print `"apiKey": "***"`, so only the
            # reference the config file carries can tell them apart.
            ("builtin_override_envkey_unset", "openrouter/x/y", "unknown"),
            ("builtin_override_envkey_set", "openrouter/x/y", "declared"),
            ("builtin_override_empty_key", "openrouter/x/y", "unknown"),
            ("builtin_override_filekey", "openrouter/x/y", "declared"),
            ("builtin_override_filekey_missing", "openrouter/x/y", "unknown"),
            ("nothing_declared", "openrouter/x/y", "unknown"),
        ],
    )
    def test_the_verdict_table_on_a_write_slot(
        self, provider, user_xdg, tmp_path, monkeypatch, case, model, expected
    ):
        """`agent-write` keeps the user's config, so its resolved config decides:
        each row replays the capture's stdout over the files it came from."""
        _replay_files(case, user_xdg, monkeypatch)
        run = DEBUG_CONFIG["cases"][case]
        monkeypatch.setattr(oc, "_spawn_debug_config", _DebugConfig(run["stdout"], run["exit"]))
        assert provider.config_credential_source(
            model, profile=PROFILE_AGENT_WRITE, cwd=str(tmp_path)
        ) == expected

    def test_the_printed_mask_alone_never_clears_a_builtin(
        self, provider, user_xdg, tmp_path, monkeypatch
    ):
        """`"***"` with no config source behind it (a layer this read cannot
        see, such as a remote org config) is not evidence of a key."""
        monkeypatch.setattr(
            oc, "_spawn_debug_config", _DebugConfig(_stdout("builtin_override_keyed"))
        )
        assert provider.config_credential_source(
            "openrouter/x/y", profile=PROFILE_AGENT_WRITE, cwd=str(tmp_path)
        ) == "unknown"

    def test_the_read_happens_inside_the_hermetic_slot_with_pure(
        self, provider, user_xdg, tmp_path, monkeypatch
    ):
        _use_config(tmp_path, monkeypatch, (
            "providers:\n  opencode:\n    config_providers:\n      stub:\n"
            "        npm: '@ai-sdk/openai-compatible'\n"
            "        options: {baseURL: 'http://127.0.0.1:9/v1'}\n"
        ))
        fake = _DebugConfig(None, dispatch=_stdout("hermetic_pure_config_content"))
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        built: List[Any] = []

        def builder():
            built.append(oc._ACTIVE.get())
            return {"PATH": "/usr/bin", "XDG_CONFIG_HOME": "slot"}

        verdict = provider.config_credential_source(
            "stub/stub-model", profile=PROFILE_AGENT_READONLY, cwd="/repo",
            env_builder=builder,
        )
        assert verdict == "declared"
        call = fake.calls[-1]
        assert call["active"] is not None and built[0] is not None
        assert call["argv"] == ["opencode", "--pure", "debug", "config"]
        assert call["cwd"] == "/repo"
        assert call["env"] == {"PATH": "/usr/bin", "XDG_CONFIG_HOME": "slot"}
        # The slot is left behind: no context, no directory.
        assert oc._ACTIVE.get() is None

    def test_a_refused_hermetic_slot_reads_unknown(self, provider, user_xdg, monkeypatch):
        """A user-config-only custom prefix is refused by the slot → unknown."""
        # A commented jsonc the file read cannot answer, so the reader is asked.
        (user_xdg / "opencode" / "opencode.jsonc").write_text(
            "// served by debug config\n{}\n", encoding="utf-8"
        )
        fake = _DebugConfig(_stdout("custom_inline_key"))
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        assert provider.config_credential_source(
            "stub/stub-model", profile=PROFILE_AGENT_READONLY
        ) == "unknown"
        # The only read was the user-config one (no env): no dispatch read ran.
        assert all(call["env"] is None for call in fake.calls)

    def test_a_raising_builder_reads_unknown(self, provider, user_xdg, monkeypatch):
        monkeypatch.setattr(oc, "_spawn_debug_config", _DebugConfig(_stdout("custom_inline_key")))

        def boom():
            raise RuntimeError("no env")

        assert provider.config_credential_source(
            "stub/stub-model", profile=PROFILE_AGENT_WRITE, env_builder=boom
        ) == "unknown"

    def test_a_bare_token_is_unknown_and_reads_nothing(self, provider, monkeypatch):
        fake = _DebugConfig(_stdout("custom_inline_key"))
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        assert provider.config_credential_source("stub", profile=PROFILE_AGENT_WRITE) == "unknown"
        assert fake.calls == []


class TestTheKeyReference:
    """T8.3: where a masked `apiKey` comes from, and whether the child resolves it.

    The layer order is the one MEASURED on the pin (2026-10-02, the real binary
    from a scratch HOME, each layer setting its own `options.timeout` and the
    printed winner removed until none was left). Every value is fake."""

    #: Highest precedence first, as measured: a git worktree at `outer/repo`,
    #: `cwd` at `outer/repo/sub`.
    MEASURED = (
        "managed", "CONTENT", "CDIR", "D_home", "D_root", "D_sub_jsonc", "D_sub",
        "P_sub_jsonc", "P_sub_json", "P_root_jsonc", "P_root_json", "OC",
        "G_jsonc", "G_json", "G_config",
    )

    def _tree(self, root: Path, git: bool = True):
        home = root / "home"
        repo = root / "outer" / "repo"
        sub = repo / "sub"
        sub.mkdir(parents=True)
        (home / ".config" / "opencode").mkdir(parents=True)
        if git:
            (repo / ".git").mkdir()
        paths = {
            "G_config": home / ".config" / "opencode" / "config.json",
            "G_json": home / ".config" / "opencode" / "opencode.json",
            "G_jsonc": home / ".config" / "opencode" / "opencode.jsonc",
            "OC": root / "explicit.json",
            "outer_json": root / "outer" / "opencode.json",
            "P_root_json": repo / "opencode.json",
            "P_root_jsonc": repo / "opencode.jsonc",
            "P_sub_json": sub / "opencode.json",
            "P_sub_jsonc": sub / "opencode.jsonc",
            "D_root": repo / ".opencode" / "opencode.json",
            "D_sub": sub / ".opencode" / "opencode.json",
            "D_sub_jsonc": sub / ".opencode" / "opencode.jsonc",
            "D_home": home / ".opencode" / "opencode.json",
            "CDIR": root / "cdir" / "opencode.json",
            "managed": root / "managed" / "opencode.json",
        }
        env = {
            "HOME": str(home), "PATH": "/usr/bin:/bin",
            "OPENCODE_CONFIG": str(paths["OC"]),
            "OPENCODE_CONFIG_DIR": str(paths["CDIR"].parent),
            "OPENCODE_TEST_MANAGED_CONFIG_DIR": str(paths["managed"].parent),
        }
        return paths, env, str(sub)

    @staticmethod
    def _doc(key: str) -> str:
        return json.dumps({"provider": {"openrouter": {"options": {"apiKey": key}}}})

    def _ranking(self, paths, env, cwd, layers):
        """Remove the winner until no layer is left: the order, highest first."""
        for name in layers:
            if name == "CONTENT":
                env["OPENCODE_CONFIG_CONTENT"] = self._doc("key-CONTENT")
            else:
                paths[name].parent.mkdir(parents=True, exist_ok=True)
                paths[name].write_text(self._doc(f"key-{name}"), encoding="utf-8")
        ranking = []
        while True:
            found = oc._api_key_reference("openrouter", env, cwd)
            if found is None:
                return ranking
            winner = found[0][len("key-"):]
            ranking.append(winner)
            if winner == "CONTENT":
                del env["OPENCODE_CONFIG_CONTENT"]
            else:
                paths[winner].unlink()

    def test_the_measured_precedence_in_a_git_worktree(self, tmp_path):
        paths, env, cwd = self._tree(tmp_path)
        layers = [*self.MEASURED, "outer_json"]
        # Outside the worktree the project walk stops: `outer_json` never loads.
        assert self._ranking(paths, env, cwd, layers) == list(self.MEASURED)

    def test_without_git_the_project_walk_reaches_the_root(self, tmp_path):
        paths, env, cwd = self._tree(tmp_path, git=False)
        expected = list(self.MEASURED)
        expected.insert(expected.index("OC"), "outer_json")
        assert self._ranking(paths, env, cwd, [*self.MEASURED, "outer_json"]) == expected

    def test_the_home_layer_follows_opencode_test_home(self, tmp_path):
        """opencode's home is `OPENCODE_TEST_HOME` when set, else `HOME` (the
        bundle's `Global.Path.home`): a hermetic slot points it at its empty
        config home, so the user's `~/.opencode/` is not a layer there."""
        paths, env, cwd = self._tree(tmp_path)
        empty = tmp_path / "slot"
        empty.mkdir()
        env["OPENCODE_TEST_HOME"] = str(empty)
        assert self._ranking(paths, env, cwd, ["D_home", "G_json"]) == ["G_json"]

    @pytest.mark.parametrize("value", ["1", "true", "TRUE"])
    def test_disabling_the_project_layer_keeps_home_and_config_dir(self, tmp_path, value):
        paths, env, cwd = self._tree(tmp_path)
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = value
        assert self._ranking(paths, env, cwd, list(self.MEASURED)) == [
            "managed", "CONTENT", "CDIR", "D_home", "OC", "G_jsonc", "G_json", "G_config",
        ]

    @pytest.mark.parametrize("value", ["yes", "0"])
    def test_only_1_or_true_disables_it(self, tmp_path, value):
        """opencode's flag reader: `"true"` or `"1"`, case-folded, nothing else."""
        paths, env, cwd = self._tree(tmp_path)
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = value
        assert self._ranking(paths, env, cwd, ["P_sub_json"]) == ["P_sub_json"]

    def test_a_later_layer_without_a_key_keeps_the_earlier_one(self, tmp_path):
        """A deep merge, measured: the content's own `timeout` does not drop
        the global `baseURL`, so a block without `apiKey` hides nothing."""
        paths, env, cwd = self._tree(tmp_path)
        paths["G_json"].write_text(self._doc("{env:FAKE_KEY}"), encoding="utf-8")
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            {"provider": {"openrouter": {"options": {"timeout": 5}}}}
        )
        assert oc._api_key_reference("openrouter", env, cwd) == (
            "{env:FAKE_KEY}", str(paths["G_json"].parent)
        )

    def test_jsonc_comments_and_trailing_commas_are_read(self, tmp_path):
        paths, env, cwd = self._tree(tmp_path)
        paths["G_jsonc"].write_text(
            '{\n  // "provider": {"openrouter": {"options": {"apiKey": "{file:/nope}"}}},\n'
            '  "provider": {"openrouter": {"options": {"apiKey": "{env:FAKE_KEY}",},},},\n}\n',
            encoding="utf-8",
        )
        assert oc._api_key_reference("openrouter", env, cwd)[0] == "{env:FAKE_KEY}"

    def test_an_unreadable_layer_above_the_winner_is_inconclusive(self, tmp_path):
        paths, env, cwd = self._tree(tmp_path)
        paths["G_json"].write_text(self._doc("fake-literal"), encoding="utf-8")
        paths["P_sub_json"].write_text("{ not json", encoding="utf-8")
        assert oc._api_key_reference("openrouter", env, cwd) is None

    def test_nothing_declared_is_none(self, tmp_path):
        paths, env, cwd = self._tree(tmp_path)
        assert oc._api_key_reference("openrouter", env, cwd) is None

    @pytest.mark.parametrize(
        "raw, env, expected",
        [
            ("fake-literal", {}, True),
            ("", {}, False),
            ("   ", {}, False),
            ("{env:FAKE_KEY}", {"FAKE_KEY": "fake-value"}, True),
            ("{env:FAKE_KEY}", {}, False),
            ("{env:FAKE_KEY}", {"FAKE_KEY": ""}, False),
            ("{env:A}{env:B}", {"B": "fake-value"}, True),
        ],
        ids=["literal", "empty", "blank", "env-set", "env-unset", "env-empty", "env-pair"],
    )
    def test_the_reference_resolves_in_the_env_it_is_given(self, tmp_path, raw, env, expected):
        assert oc._api_key_resolves(raw, str(tmp_path), env) is expected

    def test_a_file_reference_is_stat_ed_relative_to_its_layer(self, tmp_path):
        (tmp_path / "home").mkdir()
        (tmp_path / "fake.key").write_text("fake-value\n", encoding="utf-8")
        (tmp_path / "empty.key").write_text("", encoding="utf-8")
        (tmp_path / "a-dir").mkdir()
        (tmp_path / "home" / "fake.key").write_text("fake-value", encoding="utf-8")
        env = {"HOME": str(tmp_path / "home")}
        base = str(tmp_path)
        assert oc._api_key_resolves("{file:fake.key}", base, env)
        assert oc._api_key_resolves(f"{{file:{tmp_path / 'fake.key'}}}", "/elsewhere", env)
        assert oc._api_key_resolves("{file:~/fake.key}", "/elsewhere", env)
        assert not oc._api_key_resolves("{file:empty.key}", base, env)
        assert not oc._api_key_resolves("{file:missing.key}", base, env)
        assert not oc._api_key_resolves("{file:a-dir}", base, env)

    def test_the_content_layer_resolves_a_relative_file_against_cwd(self, tmp_path):
        paths, env, cwd = self._tree(tmp_path)
        env["OPENCODE_CONFIG_CONTENT"] = self._doc("{file:fake.key}")
        assert oc._api_key_reference("openrouter", env, cwd) == ("{file:fake.key}", cwd)


class TestTheStructuralLeg:
    """A hermetic slot on a user-config-only prefix: `unresolvable_model`,
    before the auth leg, and never `not_authenticated`."""

    @pytest.fixture
    def user_only(self, user_xdg):
        (user_xdg / "opencode" / "opencode.json").write_text(json.dumps({
            "provider": {"only-in-user-config": {
                "options": {"baseURL": "https://gateway.invalid/v1",
                            "apiKey": "sk-canary-not-a-key-7"},
                "models": {"private-model": {}},
            }},
        }), encoding="utf-8")
        return user_xdg

    def test_a_provider_only_in_home_opencode_is_refused_not_lost(
        self, provider, user_xdg, monkeypatch
    ):
        """A hermetic slot no longer loads `~/.opencode/`, so a provider the
        user defines only there is unreconstructible like any other user-only
        provider: the file read cannot see that layer, so `debug config` is
        asked, and the slot is refused instead of resolving nothing."""
        home_layer = Path(os.environ["HOME"]) / ".opencode"
        home_layer.mkdir(parents=True)
        block = {"npm": "@ai-sdk/openai-compatible",
                 "options": {"baseURL": "http://127.0.0.1:9/v1"}, "models": {"m": {}}}
        (home_layer / "opencode.json").write_text(
            json.dumps({"provider": {"home-only": block}}), encoding="utf-8"
        )
        resolved = json.dumps({"provider": {"home-only": block}})
        monkeypatch.setattr(oc, "_spawn_debug_config", _DebugConfig(resolved))
        text = provider.describe_unresolvable_model(
            "home-only/m", profile=PROFILE_AGENT_READONLY
        )
        assert text and "config_providers" in text

    def test_describe_unresolvable_model_answers_with_profile(self, provider, user_only):
        text = provider.describe_unresolvable_model(
            "only-in-user-config/private-model", profile=PROFILE_AGENT_READONLY
        )
        assert text and "config_providers" in text
        assert "sk-canary" not in text
        # No profile: the pre-§8 structural answer, untouched.
        assert provider.describe_unresolvable_model("only-in-user-config/private-model") is None

    def test_a_write_slot_or_hermetic_off_is_not_refused(
        self, provider, user_only, tmp_path, monkeypatch
    ):
        assert provider.describe_unresolvable_model(
            "only-in-user-config/private-model", profile=PROFILE_AGENT_WRITE
        ) is None
        _use_config(tmp_path, monkeypatch, "providers:\n  opencode:\n    hermetic: off\n")
        assert provider.describe_unresolvable_model(
            "only-in-user-config/private-model", profile=PROFILE_AGENT_READONLY
        ) is None

    def test_preflight_refuses_before_any_auth_or_catalog_spawn(
        self, provider, user_only, monkeypatch
    ):
        probes: List[Any] = []
        monkeypatch.setattr(executor, "_run_probe", lambda argv, env=None: probes.append(argv))
        # Patched on the class: `monkeypatch.setattr(instance, ...)` undoes by
        # writing the bound method back onto the shared registry instance,
        # which then shadows every later class-level patch of it.
        monkeypatch.setattr(type(provider), "is_available", lambda self: True)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            executor.preflight_runner(
                "opencode:only-in-user-config/private-model",
                profile=PROFILE_AGENT_READONLY, key=KEY,
            )
        assert excinfo.value.reason_code == "unresolvable_model"
        assert "config_providers" in str(excinfo.value)
        assert "sk-canary" not in str(excinfo.value)
        # Nothing spawned: no version, no `auth list`, no `opencode models`.
        assert probes == []

    def test_a_narrow_double_without_the_keywords_still_works(self, monkeypatch):
        class Narrow:
            def describe_unresolvable_model(self, model):
                return "refused: " + model

        provider = registry.get_provider("opencode")
        # Patched on the class: `monkeypatch.setattr(instance, ...)` undoes by
        # writing the bound method back onto the shared registry instance,
        # which then shadows every later class-level patch of it.
        monkeypatch.setattr(
            type(provider), "describe_unresolvable_model",
            lambda self, model: Narrow().describe_unresolvable_model(model),
        )
        monkeypatch.setattr(type(provider), "is_available", lambda self: True)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            executor.preflight_runner("opencode:x/y", key=KEY)
        assert "refused: x/y" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# T8.4 + T8.13 — _detect_auth
# --------------------------------------------------------------------------- #

class _Probe:
    """``_run_probe`` stand-in: records env, answers by what the env carries."""

    def __init__(self, answer):
        self.answer = answer
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, argv, env=None):
        self.calls.append({"argv": list(argv), "env": env, "active": oc._ACTIVE.get()})
        return self.answer(env)


def _detect(model: str, profile: str = PROFILE_AGENT_WRITE, **kwargs) -> str:
    return executor._detect_auth(
        "opencode", model, key=KEY, spec=f"opencode:{model}", profile=profile,
        cwd=kwargs.pop("cwd", "/repo"), **kwargs,
    )


class TestDetectAuthConfigLeg:

    def test_a_declared_custom_provider_records_config(self, user_xdg, monkeypatch):
        probe = _Probe(lambda env: _auth_run("config_provider_only"))
        monkeypatch.setattr(executor, "_run_probe", probe)
        monkeypatch.setattr(oc, "_spawn_debug_config", _DebugConfig(_stdout("custom_inline_key")))
        assert _detect("stub/stub-model") == "config"
        assert any(key.endswith("|stub|plain") for key in executor._AUTH_CACHE)

    def test_a_keyless_builtin_override_is_still_refused(self, user_xdg, monkeypatch):
        monkeypatch.setattr(executor, "_run_probe", _Probe(lambda env: _auth_run("empty")))
        monkeypatch.setattr(
            oc, "_spawn_debug_config", _DebugConfig(_stdout("builtin_override_keyless"))
        )
        with pytest.raises(executor.RunnerRejected) as excinfo:
            _detect("openrouter/x/y")
        assert excinfo.value.reason_code == "not_authenticated"
        assert "opencode auth login" in str(excinfo.value)

    def test_the_lazy_path_never_reads_config_on_a_login(self, user_xdg, monkeypatch):
        probe = _Probe(lambda env: _auth_run("stored"))
        monkeypatch.setattr(executor, "_run_probe", probe)
        fake = _DebugConfig(_stdout("custom_inline_key"))
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        assert _detect("openrouter/x/y") == "ok"
        assert fake.calls == [] and len(probe.calls) == 1

    def test_the_cache_key_widens_by_profile_prefix_and_hermeticity(
        self, user_xdg, monkeypatch
    ):
        probe = _Probe(lambda env: _auth_run("stored"))
        monkeypatch.setattr(executor, "_run_probe", probe)
        _detect("openrouter/a", PROFILE_AGENT_WRITE)
        _detect("openrouter/b", PROFILE_AGENT_WRITE)
        _detect("openrouter/a", PROFILE_AGENT_READONLY)
        _detect("stub/a", PROFILE_AGENT_WRITE)
        assert set(executor._AUTH_CACHE) == {
            "opencode|1|agent-write|openrouter|plain",
            "opencode|1|agent-readonly|openrouter|hermetic",
            "opencode|1|agent-write|stub|plain",
        }
        assert len(probe.calls) == 3


class TestTheProbeRunsUnderTheDispatchEnv:
    """T8.13 — `resolve_env` + `build_child_env`, outside any hermetic slot."""

    def _answer_by_env(self, env):
        env = os.environ if env is None else env
        return _auth_run("env_only" if "OPENROUTER_API_KEY" in env else "empty")

    def test_the_allowlist_drops_the_key_and_the_refusal_names_it(
        self, user_xdg, tmp_path, monkeypatch
    ):
        _use_config(tmp_path, monkeypatch, "env_allowlist: on\n")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-canary-not-a-key-8")
        probe = _Probe(self._answer_by_env)
        monkeypatch.setattr(executor, "_run_probe", probe)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            _detect("openrouter/x/y")
        message = str(excinfo.value)
        assert "env_allowlist" in message and "OPENROUTER_API_KEY" in message
        assert "sk-canary" not in message
        assert "OPENROUTER_API_KEY" not in probe.calls[0]["env"]

    def test_with_the_allowlist_off_nothing_changes(self, user_xdg, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-canary-not-a-key-8")
        monkeypatch.setattr(executor, "_run_probe", _Probe(self._answer_by_env))
        assert _detect("openrouter/x/y") == "ok"

    def test_a_host_guard_strips_it_from_the_probe_too(self, user_xdg, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-canary-not-a-key-8")
        monkeypatch.setattr(
            executor, "_host_env_guards", lambda: (["OPENROUTER_API_KEY"], [])
        )
        probe = _Probe(self._answer_by_env)
        monkeypatch.setattr(executor, "_run_probe", probe)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            _detect("openrouter/x/y")
        assert "host env guard" in str(excinfo.value)
        assert "sk-canary" not in str(excinfo.value)

    def test_the_env_is_built_once_per_cache_key(self, user_xdg, monkeypatch):
        provider = registry.get_provider("opencode")
        real = type(provider).resolve_env
        calls: List[str] = []

        def counting(self, model, profile="generate", **kwargs):
            calls.append(profile)
            return real(self, model, profile, **kwargs)

        monkeypatch.setattr(type(provider), "resolve_env", counting)
        monkeypatch.setattr(executor, "_run_probe", _Probe(lambda env: _auth_run("stored")))
        _detect("openrouter/x/y")
        _detect("openrouter/x/y")
        _detect("openrouter/other")
        assert calls == [PROFILE_AGENT_WRITE]

    def test_no_hermetic_slot_is_entered_for_the_probe(self, user_xdg, monkeypatch):
        probe = _Probe(lambda env: _auth_run("stored"))
        monkeypatch.setattr(executor, "_run_probe", probe)
        _detect("openrouter/x/y", PROFILE_AGENT_READONLY)
        assert probe.calls[0]["active"] is None

    def test_a_user_config_only_prefix_still_reaches_the_structural_refusal(
        self, user_xdg, monkeypatch
    ):
        """The helper never raises on that prefix, and no slot is entered."""
        (user_xdg / "opencode" / "opencode.json").write_text(json.dumps({
            "provider": {"mine": {"options": {"baseURL": "https://x.invalid/v1"}}},
        }), encoding="utf-8")
        env = executor.dispatch_child_env(
            registry.get_provider("opencode"), "mine/m", PROFILE_AGENT_READONLY
        )
        assert env is None or "XDG_CONFIG_HOME" not in env or env["XDG_CONFIG_HOME"] == str(user_xdg)
        probe = _Probe(lambda env: _auth_run("empty"))
        monkeypatch.setattr(executor, "_run_probe", probe)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            _detect("mine/m", PROFILE_AGENT_READONLY)
        # The auth leg alone still refuses (the structural leg runs first in
        # `preflight_runner`); the probe ran outside any slot.
        assert excinfo.value.reason_code == "not_authenticated"
        assert probe.calls[0]["active"] is None

    def _env_keyed_reader(self, user_xdg, monkeypatch):
        """`debug config` on the pin for `{env:MY_TEST_KEY}`, answered per env.

        Both captures print `"apiKey": "***"` — the pin masks a set and an
        unset reference alike — so nothing in the stdout carries the verdict.
        The capture's own `opencode.json` is laid down, and the auth leg
        resolves its reference in the env the child gets."""
        _replay_files("builtin_override_envkey_unset", user_xdg, monkeypatch)
        unset = _stdout("builtin_override_envkey_unset")
        keyed = _stdout("builtin_override_envkey_set")
        assert unset == keyed
        seen: List[Optional[Dict[str, str]]] = []

        def spawn(argv, env, cwd, timeout):
            seen.append(env)
            source = os.environ if env is None else env
            return _completed(keyed if "MY_TEST_KEY" in source else unset)

        monkeypatch.setattr(oc, "_spawn_debug_config", spawn)
        return seen

    @pytest.mark.parametrize("dropped_by", ["env_allowlist", "host_guard"])
    def test_an_env_key_the_child_never_gets_is_refused(
        self, user_xdg, tmp_path, monkeypatch, dropped_by
    ):
        """T8.3: `{env:VAR}` with VAR set here but removed from the child's env
        does not resolve in the child, so the refusal stands — on the pinned
        build, whose `debug config` prints the key as `"***"` either way."""
        monkeypatch.setenv("MY_TEST_KEY", "sk-canary-not-a-key-3")
        if dropped_by == "env_allowlist":
            _use_config(tmp_path, monkeypatch, "env_allowlist: on\n")
        else:
            monkeypatch.setattr(executor, "_host_env_guards", lambda: (["MY_TEST_KEY"], []))
        seen = self._env_keyed_reader(user_xdg, monkeypatch)
        monkeypatch.setattr(executor, "_run_probe", _Probe(lambda env: _auth_run("empty")))
        with pytest.raises(executor.RunnerRejected) as excinfo:
            _detect("openrouter/x/y", cwd=str(tmp_path))
        assert excinfo.value.reason_code == "not_authenticated"
        assert "MY_TEST_KEY" in str(excinfo.value)
        assert "sk-canary" not in str(excinfo.value)
        assert seen and all(env is not None and "MY_TEST_KEY" not in env for env in seen)

    def test_the_same_env_key_counts_when_the_child_gets_it(
        self, user_xdg, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("MY_TEST_KEY", "sk-canary-not-a-key-3")
        self._env_keyed_reader(user_xdg, monkeypatch)
        monkeypatch.setattr(executor, "_run_probe", _Probe(lambda env: _auth_run("empty")))
        assert _detect("openrouter/x/y", cwd=str(tmp_path)) == "config"

    def test_an_unset_env_key_is_refused(self, user_xdg, tmp_path, monkeypatch):
        """The reviewer's repro: a built-in override whose `{env:}` key is set
        nowhere used to pass as `config` and fail at dispatch (offline, ~65 s of
        retries graded `connection_error`, retried by the ladder)."""
        monkeypatch.delenv("MY_TEST_KEY", raising=False)
        self._env_keyed_reader(user_xdg, monkeypatch)
        monkeypatch.setattr(executor, "_run_probe", _Probe(lambda env: _auth_run("empty")))
        with pytest.raises(executor.RunnerRejected) as excinfo:
            _detect("openrouter/x/y", cwd=str(tmp_path))
        assert excinfo.value.reason_code == "not_authenticated"

    def test_a_hermetic_slot_never_sees_the_users_home_opencode_key(
        self, user_xdg, tmp_path, monkeypatch
    ):
        """The user's `~/.opencode/opencode.json` declares a key: a hermetic
        slot does not load that layer, so the key is not evidence there — while
        on a non-hermetic write slot it is the user's config in force."""
        home_layer = Path(os.environ["HOME"]) / ".opencode"
        home_layer.mkdir(parents=True)
        (home_layer / "opencode.json").write_text(json.dumps(
            DEBUG_CONFIG["cases"]["builtin_override_keyed"]["global_files"]["opencode.json"]
        ), encoding="utf-8")
        fake = _DebugConfig(None, dispatch=_stdout("builtin_override_keyed"))
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        monkeypatch.setattr(executor, "_run_probe", _Probe(lambda env: _auth_run("empty")))
        with pytest.raises(executor.RunnerRejected) as excinfo:
            _detect("openrouter/x/y", PROFILE_AGENT_READONLY, cwd=str(tmp_path))
        assert excinfo.value.reason_code == "not_authenticated"
        child = fake.calls[-1]["env"]
        assert child["OPENCODE_TEST_HOME"] == child["XDG_CONFIG_HOME"]
        executor.reset_auth_cache()
        fake.stdout, fake.dispatch = _stdout("builtin_override_keyed"), None
        assert _detect("openrouter/x/y", PROFILE_AGENT_WRITE, cwd=str(tmp_path)) == "config"

    def test_a_hermetic_slot_resolves_the_key_its_own_content_carries(
        self, user_xdg, tmp_path, monkeypatch
    ):
        """`agent-readonly` is hermetic: the user's own literal key is
        redirected away with the rest of their config, and the `{env:}`
        reference `config_providers` puts in the slot's
        `OPENCODE_CONFIG_CONTENT` is what decides."""
        _replay_files("builtin_override_keyed", user_xdg, monkeypatch)
        _use_config(tmp_path, monkeypatch, (
            "providers:\n  opencode:\n    config_providers:\n      openrouter:\n"
            "        options: {apiKey: '{env:MY_TEST_KEY}'}\n"
        ))
        fake = _DebugConfig(None, dispatch=_stdout("builtin_override_envkey_unset"))
        monkeypatch.setattr(oc, "_spawn_debug_config", fake)
        monkeypatch.setattr(executor, "_run_probe", _Probe(lambda env: _auth_run("empty")))
        monkeypatch.delenv("MY_TEST_KEY", raising=False)
        with pytest.raises(executor.RunnerRejected):
            _detect("openrouter/x/y", PROFILE_AGENT_READONLY, cwd=str(tmp_path))
        assert "MY_TEST_KEY" in fake.calls[-1]["env"]["OPENCODE_CONFIG_CONTENT"]
        executor.reset_auth_cache()
        monkeypatch.setenv("MY_TEST_KEY", "sk-canary-not-a-key-10")
        assert _detect("openrouter/x/y", PROFILE_AGENT_READONLY, cwd=str(tmp_path)) == "config"

    def test_run_probe_without_env_inherits(self, monkeypatch):
        seen: Dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs)
            return _completed("")

        monkeypatch.setattr(executor.subprocess, "run", fake_run)
        executor._run_probe(["opencode", "--version"])
        assert "env" not in seen
        executor._run_probe(["opencode", "--version"], env={"A": "1"})
        assert seen["env"] == {"A": "1"}


class TestCursorProfileScope:
    """T8.13: preflight's env build plus one dispatch leave one profile dir."""

    def test_preflight_and_one_dispatch_share_the_per_run_profile(
        self, tmp_path, monkeypatch
    ):
        from utils.providers import cursor_agent as cm

        scratch = tmp_path / "tmpdir"
        scratch.mkdir()
        monkeypatch.setenv("TMPDIR", str(scratch))
        monkeypatch.setattr(cm.tempfile, "tempdir", str(scratch))
        monkeypatch.delenv("CURSOR_CONFIG_DIR", raising=False)
        monkeypatch.delenv("CURSOR_DATA_DIR", raising=False)
        monkeypatch.setattr(cm, "PROFILE_DIR_SCOPE", cm.SCOPE_PER_RUN)
        _use_config(
            tmp_path, monkeypatch,
            "providers:\n  cursor-agent:\n    isolate_profile: \"on\"\n",
        )
        cm.reset_profile_state()
        cm.reset_side_effect_guard()
        try:
            probe = _Probe(lambda env: subprocess.CompletedProcess(
                ["cursor-agent"], 0, '{"isAuthenticated": true}', ""
            ))
            monkeypatch.setattr(executor, "_run_probe", probe)
            provider = registry.get_provider("cursor-agent")
            executor._detect_auth(
                "cursor-agent", "auto", key=KEY, spec="cursor-agent:auto",
                profile=PROFILE_AGENT_READONLY, cwd=str(tmp_path),
            )
            executor.dispatch_child_env(
                provider, "auto", PROFILE_AGENT_READONLY, cwd=str(tmp_path)
            )
            dirs = cm.isolated_profile_dirs()
            assert "CURSOR_CONFIG_DIR" in probe.calls[0]["env"]
            assert len(dirs) == 1
            assert probe.calls[0]["env"]["CURSOR_CONFIG_DIR"] == dirs[0]
        finally:
            cm.cleanup_isolated_profiles()
            cm.reset_profile_state()
            cm.reset_side_effect_guard()
