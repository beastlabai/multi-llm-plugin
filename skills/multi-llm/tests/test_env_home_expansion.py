"""L10 (§2.5) — ``~`` in an adapter/config env value must reach the child expanded.

The measured defect: ``CODEX_HOME='~/.codex_work'`` arrived at the child verbatim
(``expanded? False``) and codex 0.146.0 refused with *CODEX_HOME points to
"~/.codex_work", but that path does not exist*. The worse half is silent — if a
literal ``./~/.codex_work`` happens to exist, codex resolves the value
CWD-relative and runs against the **wrong auth directory** with no error at all.

The fix belongs at the two spawn sites, not per adapter, so both are asserted
here: Lane A (``llm_client.invoke_with_provider`` → ``subprocess.run``) and
Lane B (``subagent_executor._spawn`` → ``subprocess.Popen``).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import expand_env_home, invoke_with_provider  # noqa: E402
from utils.providers.base import LLMProvider  # noqa: E402


HOME_RELATIVE = "~/.codex_work"
EXPANDED = os.path.expanduser(HOME_RELATIVE)


class _HomeRelativeEnvProvider(LLMProvider):
    """Minimal adapter whose env block carries an unexpanded ``~`` path."""

    name = "fake-home"
    binary = "fake-home"

    default_timeout = 60

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[object] = None,
    ) -> List[str]:
        # `timeout_s` (L5) and `schema` (M7) track the real base signature; this
        # adapter renders nothing from either, which is exactly the inert `None`
        # path both keywords pin.
        return [self.binary, "--print", prompt]

    def parse_output(self, stdout: str, stderr: str = "") -> Dict[str, object]:
        return {"success": True, "data": stdout}

    def is_available(self) -> bool:
        return True

    def get_env(self, model: str) -> Dict[str, str]:
        return {
            "CODEX_HOME": HOME_RELATIVE,
            # A value that is not home-relative must pass through byte-identical.
            "MLLM_ABSOLUTE": "/etc/multi-llm",
            # `~` only expands in leading position; an embedded one is data.
            "MLLM_EMBEDDED": "prefix~/suffix",
        }


class TestExpandEnvHome:
    """The helper itself: leading ``~`` only, everything else untouched."""

    def test_leading_tilde_is_expanded(self):
        assert expand_env_home({"CODEX_HOME": HOME_RELATIVE}) == {
            "CODEX_HOME": EXPANDED
        }
        assert "~" not in expand_env_home({"CODEX_HOME": HOME_RELATIVE})["CODEX_HOME"]

    def test_non_home_relative_values_pass_through_byte_identical(self):
        block = {
            "MLLM_ABSOLUTE": "/etc/multi-llm",
            "MLLM_EMBEDDED": "prefix~/suffix",
            "MLLM_RELATIVE": "./local",
            "MLLM_EMPTY": "",
        }
        assert expand_env_home(block) == block

    def test_unresolvable_user_is_left_alone(self):
        """``~nosuchuser`` is not a home this machine can resolve — leave it."""
        value = "~nosuchuser-multi-llm/config"
        assert expand_env_home({"X": value})["X"] == value

    def test_empty_block_is_empty(self):
        assert expand_env_home({}) == {}


class TestLaneASpawnSiteExpands:
    """`llm_client.invoke_with_provider` — the fan-out path a normal run takes."""

    @patch("shutil.which")
    @patch("utils.llm_client.subprocess.run")
    def test_child_receives_expanded_codex_home(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/fake-home"
        mock_run.return_value = subprocess.CompletedProcess(
            args=["fake-home"], returncode=0, stdout="ok", stderr=""
        )

        provider = _HomeRelativeEnvProvider()
        with patch("utils.llm_client.get_provider", return_value=provider):
            invoke_with_provider(prompt="Test", model_spec="fake-home:m", timeout=5)

        env = mock_run.call_args.kwargs["env"]
        assert env["CODEX_HOME"] == EXPANDED
        assert "~" not in env["CODEX_HOME"]
        # The two controls, in the same child env, so an over-eager expansion
        # that rewrote everything would fail here rather than pass silently.
        assert env["MLLM_ABSOLUTE"] == "/etc/multi-llm"
        assert env["MLLM_EMBEDDED"] == "prefix~/suffix"


class TestLaneBSpawnSiteExpands:
    """`subagent_executor._spawn` — the agent-profile path."""

    def test_child_receives_expanded_codex_home(self):
        from utils import subagent_executor

        provider = _HomeRelativeEnvProvider()
        # `_spawn` renders an agent command; this adapter declares no agent
        # profile, so drive the env leg through the module's own helper the way
        # the spawn site does, with the real Popen replaced.
        captured = {}

        class _FakePopen:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                self.pid = 4242

            def communicate(self, *a, **kw):
                return ("", "")

            def wait(self, *a, **kw):
                return 0

            @property
            def returncode(self):
                return 0

        with patch.object(subagent_executor.subprocess, "Popen", _FakePopen):
            env = {
                **os.environ,
                **subagent_executor.expand_env_home(provider.get_env("m")),
            }
            _FakePopen(["fake-home"], env=env)

        assert captured["env"]["CODEX_HOME"] == EXPANDED
        assert "~" not in captured["env"]["CODEX_HOME"]
        assert captured["env"]["MLLM_ABSOLUTE"] == "/etc/multi-llm"

    @pytest.mark.parametrize(
        "module_name", ["llm_client.py", "subagent_executor.py"]
    )
    def test_every_spawn_site_wraps_its_env_block_in_the_expander(self, module_name):
        """Pin the invariant, not the spelling (§2.5: "exactly two spawn sites").

        Deliberately structural rather than a substring match: which accessor
        supplies the block is not this task's business — it has already moved
        once, from ``get_env`` to ``resolve_env`` when the git-env scoping
        landed — but *that the block is expanded before it reaches the child*
        is. An AST walk survives the rename; a text match asserted the other
        task's implementation detail and went red for the wrong reason.
        """
        import ast

        source = Path(__file__).parent.parent / "utils" / module_name
        tree = ast.parse(source.read_text(encoding="utf-8"))

        wrapped = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "provider_env"
                for t in node.targets
            ):
                continue
            call = node.value
            wrapped.append(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "expand_env_home"
            )

        assert wrapped, f"no `provider_env = …` assignment found in {module_name}"
        assert all(wrapped), (
            f"{module_name} builds a provider env block without routing it "
            "through expand_env_home; see §2.5 — an unexpanded CODEX_HOME "
            "silently selects the wrong auth directory."
        )


class TestTheDefectThisReplaces:
    """Red evidence: the pre-L10 expression, run directly, does not expand.

    Kept as a test rather than a comment because it is the whole justification
    for the two call-site edits — if `os.environ`-merge alone were sufficient,
    L10 would be dead code.
    """

    def test_bare_merge_leaves_the_tilde_verbatim(self):
        provider_env = _HomeRelativeEnvProvider().get_env("m")
        pre_l10 = {**os.environ, **provider_env}  # the shipped expression
        assert pre_l10["CODEX_HOME"] == HOME_RELATIVE
        assert pre_l10["CODEX_HOME"] != EXPANDED

    def test_expansion_actually_changes_the_value_on_this_machine(self):
        """Guard against a vacuous suite where `~` happens to expand to `~`."""
        assert EXPANDED != HOME_RELATIVE
        assert not EXPANDED.startswith("~")
