"""An aider sibling's tag cache is not an uncontained slot's write (D4.3).

plans/ask-answer-recovery-and-false-success.md D4.3 / T4.3. On 2026-09-26 a
fan-out ran codex on the uncontained rung beside aider, and the porcelain write
check counted aider's ``.aider.tags.cache.v4/`` (its repo-map cache, written at
the repo root on every run) against codex. The cure mirrors the container
harness's per-CLI ``tool_artifacts``:

- the cache is declared per adapter (``LLMProvider.workspace_caches``), not
  hardcoded in the check;
- it is exempt only when an adapter declaring it is in the same fan-out, read
  off the roster ``spawn_fanout_slot`` hands each slot;
- the measured version is exempt, never a glob, so ``v5`` still counts.

The lane cases run real ``git`` in a temp repository, as the porcelain cases in
``test_subagent_executor.py`` and ``test_code_review_orchestrator.py`` do; the
slot's dispatch is a fake that writes into the tree the way the case names.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.providers.aider import AiderProvider  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_GENERATE,
    LLMProvider,
)

V4 = ".aider.tags.cache.v4"
V5 = ".aider.tags.cache.v5"
SLOT = "fakecodex:m1"
AIDER = "aider:openrouter/some-model"
PLAIN = "fakeplain:m1"
BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"


def _git_repo(root: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    (root / "tracked.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True, env=env)


def _write_cache(root: Path, cache_dir: str) -> str:
    """What aider's repo map leaves behind: a diskcache ``cache.db``."""
    (root / cache_dir).mkdir(exist_ok=True)
    (root / cache_dir / "cache.db").write_bytes(b"SQLite format 3\0")
    return f"{cache_dir}/cache.db"


def _in_fanout(roster, coro):
    """Run ``coro`` as one slot of a fan-out over ``roster``."""
    async def drive():
        return await rob.spawn_fanout_slot(roster, coro)

    return asyncio.run(drive())


# ---------------------------------------------------------------------------
# The declaration and the roster-aware prefixes
# ---------------------------------------------------------------------------


class TestTheCacheIsDeclaredPerAdapter:
    def test_aider_declares_the_measured_version_and_the_base_declares_none(self):
        assert AiderProvider.workspace_caches == (f"{V4}/",)
        assert LLMProvider.workspace_caches == ()
        # The adapter the incident blamed declares nothing of its own.
        assert registry.get_provider("codex").workspace_caches == ()


class TestFanoutIgnorePrefixes:
    @pytest.fixture
    def plan(self, tmp_path):
        path = tmp_path / "plan.md"
        path.write_text("plan", encoding="utf-8")
        return path

    def test_outside_a_fanout_it_is_the_plugins_own_trees_alone(self, plan, tmp_path):
        root = str(tmp_path)
        assert rob.fanout_workspace_caches(plan) == []
        assert rob.fanout_ignore_prefixes(plan, root) == rob.bypass_ignore_prefixes(
            plan, root)

    def test_an_aider_slot_in_the_roster_adds_its_cache(self, plan, tmp_path):
        root = str(tmp_path)

        async def slot():
            return rob.fanout_ignore_prefixes(plan, root, anchor=plan)

        prefixes = _in_fanout(["codex:gpt-5.5", AIDER], slot())
        assert prefixes == rob.bypass_ignore_prefixes(plan, root) + [f"{V4}/"]

    def test_a_roster_without_aider_adds_nothing(self, plan, tmp_path):
        root = str(tmp_path)

        async def slot():
            return rob.fanout_ignore_prefixes(plan, root)

        prefixes = _in_fanout(["codex:gpt-5.5", "claude-code:sonnet"], slot())
        assert prefixes == rob.bypass_ignore_prefixes(plan, root)

    def test_an_unresolvable_entry_is_skipped_and_the_roster_does_not_leak(
        self, plan, tmp_path
    ):
        async def slot():
            return rob.fanout_workspace_caches(plan)

        # A name with no adapter contributes nothing and ends nothing.
        assert _in_fanout(["nosuch:model", AIDER, AIDER], slot()) == [f"{V4}/"]
        # The roster lived in the slot task's own context copy.
        assert rob.fanout_workspace_caches(plan) == []

    def test_the_measured_version_is_ignored_and_v5_is_not(self, plan, tmp_path):
        """The prefixes through the real snapshot and comparison."""
        _git_repo(tmp_path)
        root = str(tmp_path)

        async def slot():
            return rob.fanout_ignore_prefixes(plan, root, anchor=plan)

        ignore = _in_fanout(["codex:gpt-5.5", AIDER], slot())
        before = rob.porcelain_snapshot(root, ignore_prefixes=ignore)
        _write_cache(tmp_path, V4)
        v5 = _write_cache(tmp_path, V5)
        after = rob.porcelain_snapshot(root, ignore_prefixes=ignore)
        assert rob.unexpected_writes(before, after, ignore_prefixes=ignore) == [v5]


# ---------------------------------------------------------------------------
# Both review lanes: a codex-like slot on the uncontained rung
# ---------------------------------------------------------------------------


class _CodexLike:
    """An adapter whose ``generate`` probe fails, as codex's does where bwrap
    cannot start, with codex's uncontained rung as its whole ladder and the
    shipped ``sandbox_fallback: auto`` — so the slot takes the uncontained rung
    and the porcelain check applies. The ``sh`` is real; the line is replayed."""

    internal_retries = None

    def sandbox_probe_command(self, profile):
        if profile != PROFILE_GENERATE:
            return None
        return ["sh", "-c", f"echo '{BWRAP_LINE}' >&2; exit 1"]

    def sandbox_fallback_rungs(self, profile):
        from utils.providers.codex import UNSANDBOXED_RUNG

        return (UNSANDBOXED_RUNG,) if profile == PROFILE_GENERATE else ()

    def sandbox_fallback_policy(self, profile, anchor=None):
        return "auto" if profile in (PROFILE_GENERATE, PROFILE_AGENT_READONLY) else None


class _Unprobed:
    """A sibling with no OS sandbox and no workspace cache."""

    internal_retries = None

    def sandbox_probe_command(self, profile):
        return None


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """A git repo, the fake adapters and the real aider adapter, by name."""
    _git_repo(tmp_path)
    adapters = {
        "fakecodex": _CodexLike(),
        "fakeplain": _Unprobed(),
        # The real adapter: the declaration under test is aider's own.
        "aider": registry.get_provider("aider"),
    }
    monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
    monkeypatch.setattr(rob, "get_provider", lambda name: adapters.get(name))
    monkeypatch.setattr(rob, "get_project_root", lambda _p: str(tmp_path))
    monkeypatch.setattr(cro, "get_project_root", lambda _p: str(tmp_path))
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    return plan


def _dispatch_writing(root: Path, *cache_dirs: str):
    """A slot dispatch that answers cleanly while the named caches appear."""
    def dispatch(**kwargs):
        for cache_dir in cache_dirs:
            _write_cache(root, cache_dir)
        return {
            "success": True, "data": [], "source": "stdout_fallback",
            "output_file": str(root / "out" / "o.json"),
            "details": {"sandbox": "bypassed"},
        }

    return dispatch


class TestTheReviewPlanLane:
    """``run_single_model``: one ``ignore_prefixes`` for both snapshots and the
    comparison, computed in the slot's task where the roster lives."""

    def _run(self, plan, roster):
        return _in_fanout(roster, rob.run_single_model(
            model_spec=SLOT, plan_path=str(plan),
            prompt_template="{prefix}{output_json_path}", prompt_context={},
            phase_name="review-plan", invoke_phase="plan_review",
            timeout=1800, out_dir=str(plan.parent), prefix="plan",
        ))

    def test_only_the_tag_cache_beside_an_aider_sibling_passes(
        self, lane, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(rob, "invoke_with_file_output",
                            _dispatch_writing(tmp_path, V4))
        result = self._run(lane, [SLOT, AIDER])
        assert result["success"] is True, result["error"]
        assert "[SANDBOX_BYPASSED] fakecodex: the working tree changed" not in (
            capsys.readouterr().out)

    def test_without_the_sibling_the_tag_cache_still_fails(
        self, lane, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(rob, "invoke_with_file_output",
                            _dispatch_writing(tmp_path, V4))
        result = self._run(lane, [SLOT, PLAIN])
        assert result["success"] is False and result["output"] is None
        assert f"({V4}/cache.db)" in result["error"]
        assert "fails closed" in result["error"]

    def test_an_unmeasured_version_still_fails_beside_the_sibling(
        self, lane, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(rob, "invoke_with_file_output",
                            _dispatch_writing(tmp_path, V4, V5))
        result = self._run(lane, [SLOT, AIDER])
        assert result["success"] is False and result["output"] is None
        # Named alone: the measured v4 cache beside it stays exempt.
        assert f"({V5}/cache.db)" in result["error"]
        assert "fails closed" in result["error"]


class TestTheReviewCodeLane:
    """``run_single_review``: the before snapshot and the after check each
    compute the prefixes, and both take the roster-aware form."""

    def _run(self, plan, roster):
        return _in_fanout(roster, cro.run_single_review(
            SLOT, plan, plan.parent / "out",
        ))

    def test_only_the_tag_cache_beside_an_aider_sibling_passes(
        self, lane, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cro, "invoke_with_file_output",
                            _dispatch_writing(tmp_path, V4))
        success, issues, error = self._run(lane, [SLOT, AIDER])
        assert (success, issues, error) == (True, [], None)

    def test_without_the_sibling_the_tag_cache_still_fails(
        self, lane, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cro, "invoke_with_file_output",
                            _dispatch_writing(tmp_path, V4))
        success, issues, error = self._run(lane, [SLOT, PLAIN])
        assert success is False and issues is None
        assert f"({V4}/cache.db)" in error and "fails closed" in error

    def test_an_unmeasured_version_still_fails_beside_the_sibling(
        self, lane, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cro, "invoke_with_file_output",
                            _dispatch_writing(tmp_path, V4, V5))
        success, issues, error = self._run(lane, [SLOT, AIDER])
        assert success is False and issues is None
        assert f"({V5}/cache.db)" in error and "fails closed" in error
