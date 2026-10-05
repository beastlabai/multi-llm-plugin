"""A runner never inherits the skill's own Python environment (fix round 8).

Every host starts the dispatcher with ``uv run --project <skill dir>``, and uv
exports ``VIRTUAL_ENV`` naming the skill's environment to it; a forwarded or
image-wide ``UV_PROJECT_ENVIRONMENT`` can name the same environment. Every
child the dispatcher spawns copies its environment, so without a strip a
subagent working in the user's repository inherited both. Round 7 (3b9cf25)
exported ``UV_PROJECT_ENVIRONMENT`` from the codex MCP server, and a ``uv run``
in the user's repository then targeted the skill's environment: inside the
plugin's jail it failed with Permission denied (exit 2), outside it the
repository's dependencies were installed into the skill's environment. That
export is reverted; this file pins that the skill's environment cannot reach a
runner by any route: every adapter on both lanes, the codex agent dispatch on
and off the plugin's jail, the fan-out spawn site, and the repository's own
verification command. A value naming any other environment is the user's and
is kept.

Fix round 9: uv also puts the environment's ``bin`` first on ``PATH``, and with
neither name set and no ``.venv`` in the repository uv falls back to the first
``python`` on ``PATH``, so ``uv pip install`` in an empty repository still named
the skill's environment and a runner's ``python``/``pytest`` were the skill's.
The same builder now drops that ``PATH`` entry too, and only that entry.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.llm_client import invoke_with_provider  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.providers import base  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
)
from utils.providers.codex import PLUGIN_LANDLOCK_RUNG, CodexProvider  # noqa: E402
from tests.test_plugin_landlock_executor import (  # noqa: E402,F401
    _plan_request,
    jail_home,
    repo,
    wrapper,
)

NAMES = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")

#: The environment this process runs in: the skill's, since the suite runs
#: under ``uv run --project skills/multi-llm``.
OWN = sys.prefix

pytestmark = pytest.mark.skipif(
    sys.prefix == sys.base_prefix,
    reason="the suite is not running inside the skill's virtual environment",
)


#: Where uv puts the environment's executables, first on ``PATH``.
OWN_BIN = os.path.join(OWN, "Scripts" if os.name == "nt" else "bin")


def _names_own_bin(entry: str) -> bool:
    return bool(entry) and any(
        os.path.normcase(a) == os.path.normcase(b)
        for a, b in ((os.path.abspath(entry), os.path.abspath(OWN_BIN)),
                     (os.path.realpath(entry), os.path.realpath(OWN_BIN))))


def _user_path() -> str:
    """This process's ``PATH`` without the skill's ``bin``: the user's own."""
    return os.pathsep.join(
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if not _names_own_bin(entry))


def _leaky(own: str = OWN) -> dict:
    """What the dispatcher's environment carries under ``uv run``."""
    return {"VIRTUAL_ENV": own, "UV_PROJECT_ENVIRONMENT": own,
            "PATH": os.pathsep.join([OWN_BIN, _user_path()]),
            "MLLM_UNRELATED": "keep-me"}


def _assert_clean(env, *, expect_unrelated: bool = True) -> None:
    # `None` means "inherit this process's environment", which carries both.
    assert env is not None, "the child inherits the dispatcher's environment unchanged"
    for name in NAMES:
        assert name not in env, f"{name} reaches the runner: {env.get(name)!r}"
    leaked = [entry for entry in env.get("PATH", "").split(os.pathsep)
              if _names_own_bin(entry)]
    assert not leaked, f"the skill environment's {OWN_BIN} is on the runner's PATH"
    if expect_unrelated:
        assert env.get("MLLM_UNRELATED") == "keep-me"
        # Only that entry goes: the rest of the user's PATH, in order.
        assert env.get("PATH") == _user_path()


class _Bare:
    name = "bare"

    def get_remove_env(self):
        return []


class TestTheBuilder:
    def test_the_skill_environment_is_dropped_with_no_delta(self):
        _assert_clean(base.build_child_env(_Bare(), provider_env={}, base_env=_leaky()))

    def test_a_resolved_spelling_of_the_same_environment_is_dropped(self, tmp_path):
        link = tmp_path / "venv-link"
        link.symlink_to(OWN, target_is_directory=True)
        env = base.build_child_env(
            _Bare(), provider_env={},
            base_env={"VIRTUAL_ENV": str(link), "UV_PROJECT_ENVIRONMENT": OWN + os.sep},
        )
        _assert_clean(env, expect_unrelated=False)

    def test_an_adapter_delta_cannot_put_it_back(self):
        env = base.build_child_env(
            _Bare(), provider_env={"UV_PROJECT_ENVIRONMENT": OWN}, base_env={})
        _assert_clean(env, expect_unrelated=False)

    def test_a_user_environment_elsewhere_is_kept(self, tmp_path):
        theirs = str(tmp_path / "their-venv")
        source = {"VIRTUAL_ENV": theirs, "UV_PROJECT_ENVIRONMENT": theirs}
        assert base.build_child_env(_Bare(), provider_env={}, base_env=source) is None
        env = base.build_child_env(_Bare(), provider_env={"X": "1"}, base_env=source)
        assert env["VIRTUAL_ENV"] == theirs and env["UV_PROJECT_ENVIRONMENT"] == theirs

    def test_the_skill_environment_bin_alone_leaves_path(self):
        """``uv run`` with VIRTUAL_ENV already unset: the PATH entry is enough."""
        env = base.build_child_env(
            _Bare(), provider_env={},
            base_env={"PATH": os.pathsep.join([OWN_BIN, "/usr/bin", "", "rel/bin"])})
        assert env is not None
        assert env["PATH"] == os.pathsep.join(["/usr/bin", "", "rel/bin"])

    def test_every_spelling_of_the_bin_is_dropped(self, tmp_path):
        link = tmp_path / "venv-link"
        link.symlink_to(OWN, target_is_directory=True)
        spellings = [str(link / os.path.basename(OWN_BIN)), OWN_BIN + os.sep,
                     os.path.join(OWN_BIN, "..", os.path.basename(OWN_BIN))]
        env = base.build_child_env(
            _Bare(), provider_env={},
            base_env={"PATH": os.pathsep.join([*spellings, "/usr/bin", OWN_BIN])})
        assert env["PATH"] == "/usr/bin"

    def test_a_user_environment_bin_on_path_is_kept(self, tmp_path):
        theirs = tmp_path / "their-venv"
        (theirs / "bin").mkdir(parents=True)
        path = os.pathsep.join([str(theirs / "bin"), OWN_BIN, "/usr/bin"])
        env = base.build_child_env(
            _Bare(), provider_env={},
            base_env={"VIRTUAL_ENV": str(theirs), "PATH": path})
        assert env["VIRTUAL_ENV"] == str(theirs)
        assert env["PATH"] == os.pathsep.join([str(theirs / "bin"), "/usr/bin"])
        # With nothing of the skill's on it, the environment is inherited as is.
        clean = {"VIRTUAL_ENV": str(theirs),
                 "PATH": os.pathsep.join([str(theirs / "bin"), "/usr/bin"])}
        assert base.build_child_env(_Bare(), provider_env={}, base_env=clean) is None

    def test_an_adapter_delta_cannot_put_the_bin_back(self):
        env = base.build_child_env(
            _Bare(), provider_env={"PATH": os.pathsep.join([OWN_BIN, "/usr/bin"])},
            base_env={})
        assert env["PATH"] == "/usr/bin"

    def test_a_path_of_only_the_skill_bin_falls_back_to_the_default(self):
        env = base.build_child_env(_Bare(), provider_env={}, base_env={"PATH": OWN_BIN})
        assert "PATH" not in env

    def test_outside_a_virtual_environment_path_is_untouched(self, monkeypatch):
        """Then ``sys.prefix`` is the system's, and its ``bin`` is everyone's."""
        monkeypatch.setattr(sys, "base_prefix", sys.prefix)
        path = os.pathsep.join([OWN_BIN, "/usr/bin"])
        assert base.skill_environment_path(path) == path
        assert base.build_child_env(_Bare(), provider_env={},
                                    base_env={"PATH": path}) is None

    def test_python_and_pytest_are_not_the_skills(self):
        env = base.build_child_env(_Bare(), provider_env={}, base_env=_leaky())
        for tool in ("python", "python3", "pytest"):
            found = shutil.which(tool, path=env["PATH"])
            assert found is None or not _names_own_bin(os.path.dirname(found)), (
                f"a runner's {tool} is the skill's: {found}")

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY,
                                         PROFILE_AGENT_WRITE])
    def test_every_adapter_on_both_lanes(self, name, profile):
        provider = _PROVIDERS[name]
        with patch.dict(os.environ, _leaky(), clear=False):
            try:
                if profile == PROFILE_GENERATE:
                    env = base.build_child_env(
                        provider,
                        provider_env=provider.resolve_env("m", PROFILE_GENERATE))
                else:
                    env = executor.dispatch_child_env(provider, "m", profile)
            except base.ProfileUnsupported:
                pytest.skip(f"{name} declares no {profile}")
        _assert_clean(env)

    def test_the_repository_verification_command(self):
        """`write_batch_scheduler._verify_env` composes with no provider."""
        with patch.dict(os.environ, _leaky(), clear=False):
            env = base.build_child_env(None, provider_env={})
        _assert_clean(env)


class TestARealUvInAnEmptyRepository:
    """The measured symptom, end to end: ``uv pip`` in a repository with no
    ``.venv`` of its own, under the environment the builder hands a runner."""

    @staticmethod
    def _uv_pip(uv, cwd, env):
        # Offline and already satisfied: no index is reached, and the first
        # line names the environment uv picked (or says it found none).
        return subprocess.run(
            [uv, "pip", "install", "--dry-run", "--offline", "pyyaml"],
            cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
            stdin=subprocess.DEVNULL,
        )

    def test_uv_pip_does_not_target_the_skill_environment(self, tmp_path):
        uv = shutil.which("uv", path=_user_path())
        if uv is None:
            pytest.skip("uv is not on PATH")
        # A source that is the dispatcher's under `uv run`, minus anything else
        # that could pick an environment for uv.
        source = {key: value for key, value in os.environ.items()
                  if not key.startswith(("UV_", "CONDA_", "VIRTUAL_ENV", "PYTHON"))}
        source.update(_leaky())
        source["PATH"] = os.pathsep.join([OWN_BIN, _user_path()])
        repo = tmp_path / "userrepo"
        repo.mkdir()

        # Round 8's child: both names gone, the skill's bin still first.
        leaked = self._uv_pip(uv, repo, {k: v for k, v in source.items() if k not in NAMES})
        if OWN not in leaked.stdout + leaked.stderr:
            pytest.skip("this uv does not fall back to the python on PATH: "
                        + (leaked.stdout + leaked.stderr).strip()[:200])

        env = base.build_child_env(_Bare(), provider_env={}, base_env=source)
        done = self._uv_pip(uv, repo, env)
        assert OWN not in done.stdout + done.stderr, (
            "uv pip in the user's repository still targets the skill's environment:\n"
            + done.stdout + done.stderr)
        _assert_clean(env, expect_unrelated=False)


class TestTheSpawnSites:
    def test_the_fan_out_spawn(self):
        class _Fake(base.LLMProvider):
            name = "fake-venv"
            binary = "fake-venv"
            default_timeout = 60

            def build_command(self, prompt, model, *, timeout_s=None, schema=None):
                return [self.binary, "--print", prompt]

            def parse_output(self, stdout, stderr=""):
                return {"success": True, "data": stdout}

            def is_available(self):
                return True

        done = subprocess.CompletedProcess(args=["fake-venv"], returncode=0,
                                           stdout="ok", stderr="")
        with patch.dict(os.environ, _leaky(), clear=False), \
                patch("shutil.which", return_value="/usr/bin/fake-venv"), \
                patch("utils.llm_client.subprocess.run", return_value=done) as run, \
                patch("utils.llm_client.get_provider", return_value=_Fake()):
            invoke_with_provider(prompt="Test", model_spec="fake-venv:m", timeout=5)
        _assert_clean(run.call_args.kwargs.get("env"))

    @staticmethod
    def _stub_codex(monkeypatch, argv):
        provider = CodexProvider()
        monkeypatch.setattr(
            type(provider), "build_agent_command_for_transport",
            lambda self, *a, **k: (list(argv), None),
        )
        spawned = []

        class NoPopen:
            def __init__(self, *args, **kwargs):
                spawned.append(kwargs.get("env"))
                raise OSError("stubbed")

        monkeypatch.setattr(executor.subprocess, "Popen", NoPopen)
        return provider, spawned

    @pytest.mark.parametrize("profile", [PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE])
    def test_the_codex_agent_dispatch(self, monkeypatch, tmp_path, profile):
        for key, value in _leaky().items():
            monkeypatch.setenv(key, value)
        provider, spawned = self._stub_codex(
            monkeypatch, ["codex", "exec", "--sandbox", "read-only", "p"])
        capture = executor._spawn_in_context(
            provider=provider, model="gpt-5.5", prompt="p", profile=profile,
            transport="argv", timeout=5, cwd=str(tmp_path),
        )
        assert capture.launch_error == "stubbed" and len(spawned) == 1
        _assert_clean(spawned[0])

    def test_the_codex_agent_dispatch_in_the_plugin_jail(self, repo, monkeypatch, wrapper):
        from utils import landlock_dispatch

        for key, value in _leaky().items():
            monkeypatch.setenv(key, value)
        provider, spawned = self._stub_codex(
            monkeypatch,
            [sys.executable, "-I", "-S", landlock_dispatch.WRAPPER_PATHS["landlock_exec"],
             "--", "codex", "exec", "--sandbox", "danger-full-access", "p"],
        )
        (repo / "slot").mkdir()
        jail = executor._JailSpawn(
            rung=PLUGIN_LANDLOCK_RUNG,
            request=_plan_request(repo, scope="repo-wide", kind="implement"),
            tmpdir=str(repo / "slot"),
        )
        executor._spawn_in_context(
            provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_WRITE,
            transport="argv", timeout=5, cwd=str(repo), jail=jail,
        )
        assert len(spawned) == 1
        _assert_clean(spawned[0])
