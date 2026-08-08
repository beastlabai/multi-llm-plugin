"""The write half of the subagent executor: D3f/D3g/D3h/D3l/D3n.

Every dispatch here goes through a real subprocess against
``tests/mocks/mock_llm.py`` — never a live API — because the things worth pinning
live in the seam between this process and an edit-capable agent CLI: the rendered
argv, the pinned ``cwd``, the pre-batch snapshot, the byte comparison that decides
the outcome, and the audit line written before the child starts.

Test groups map onto the design items:

* **D3f** — write containment: every escape shape refused with nothing written.
* **D3g** — the write-result contract: all four outcomes (applied, clean failure,
  partial application, schema rejection) distinguished on real mock edits, plus
  the ``prior_changes`` entry and the ``metrics.py record`` each one drives.
* **D3h** — workspace safety: concurrency defaults to 1, overlapping declared file
  sets are refused co-scheduling, snapshots live under the phase dir, the dirty
  tree is reported at start, the repo root is pinned.
* **D3l** — auditable and reversible before it is trusted: a denied path fails the
  batch and is restored, the audit log carries no prompt or credential bytes,
  ``--dry-run-subagents`` starts no child, ``util restore-batch`` refuses a file
  that changed after the run.
* **D3n** — *a write phase schedules, it does not fan out*: ``batch_index`` order
  with verification gating the next dispatch, late ``{prior_changes_context}``
  binding, the three human-task strategies, finalization, and stop/resume.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import provider_registry as registry  # noqa: E402
from utils import restore_batch  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_safety  # noqa: E402
from utils.providers.base import PROFILE_AGENT_WRITE  # noqa: E402
from utils.subagent_tasks import (  # noqa: E402
    PRIOR_CHANGES_TOKEN,
    WRITE_KINDS,
    build_inputs,
    make_write_task,
    write_output_name,
)
from utils.write_batch_scheduler import (  # noqa: E402
    STRATEGY_PAUSE,
    STRATEGY_SKIP_CONTINUE,
    STRATEGY_SKIP_DEPENDENTS,
    WriteBatchScheduler,
    resolve_human_task_plan,
)

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"

#: The three adapters that ship a real `agent-write` invocation. Parametrized over
#: so a profile that silently stops being edit-capable fails here rather than on a
#: user's repository.
WRITE_RUNNERS = [
    ("codex", "codex:gpt-5.5"),
    ("opencode", "opencode:opencode/big-pickle"),
    ("claude", "claude-code:sonnet"),
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _install_launcher(bin_dir: Path, name: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / name
    if os.name == "nt":  # pragma: no cover - POSIX CI
        target = bin_dir / f"{name}.py"
        shutil.copyfile(MOCK_LLM, target)
    else:
        try:
            target.symlink_to(MOCK_LLM)
        except (OSError, NotImplementedError):  # pragma: no cover
            shutil.copyfile(MOCK_LLM, target)
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return target


class WriteEnv:
    """A real git repo, a mock PATH, a plan and a phase dir.

    A **real** repo (not a bare tmp dir) because the pinned root is
    ``git rev-parse --show-toplevel`` and half of D3h is that every snapshotted or
    observed path resolves beneath it.
    """

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        import subprocess

        self.root = tmp_path / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "plans").mkdir()
        self.plan = self.root / "plans" / "p.md"
        self.plan.write_text("# Plan\n\n## Step 1\nbody\n", encoding="utf-8")
        (self.root / "src").mkdir()
        (self.root / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
        (self.root / "src" / "b.py").write_text("b = 1\n", encoding="utf-8")
        self.out_dir = self.root / ".multi-llm" / "p"
        self.phase_dir = self.out_dir / "apply-suggestions"
        self.phase_dir.mkdir(parents=True)
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path

        bin_dir = tmp_path / "bin"
        for name in ("codex", "opencode", "claude"):
            _install_launcher(bin_dir, name)
        # PATH is REPLACED, never prepended: this suite spawns real subprocesses
        # with EDIT access, so it must be structurally impossible to reach a
        # developer's real `codex`/`claude`.
        monkeypatch.setenv("PATH", os.pathsep.join(
            [str(bin_dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
        ))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
        for name in ("codex", "opencode", "claude"):
            resolved = shutil.which(name)
            assert resolved and resolved.startswith(str(bin_dir)), (
                f"{name} resolves to {resolved!r}, outside the mock bin dir"
            )
        executor.reset_version_cache()
        self.bin_dir = bin_dir
        self.use()

    def use(self, runner: str = "codex:gpt-5.5", **driver) -> Path:
        lines = ["driver:", f"  subagent_runner: {runner}"]
        for key, value in driver.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub, patterns in value.items():
                    lines.append(f"    {sub}:")
                    lines.extend(f"      - '{p}'" for p in patterns)
            else:
                lines.append(f"  {key}: {value}")
        path = self.tmp_path / "driver-config.yaml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
        registry._config = None
        registry._config_key = None
        return path

    def edits(self, entries, result=None, prose=False) -> None:
        """Program the mock's write path: real edits, then the batch result JSON."""
        self.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps(entries))
        if result is not None:
            self.monkeypatch.setenv("MOCK_LLM_WRITE_RESULT", json.dumps(result))
        else:
            self.monkeypatch.delenv("MOCK_LLM_WRITE_RESULT", raising=False)
        if prose:
            self.monkeypatch.setenv("MOCK_LLM_PROSE", "1")
        else:
            self.monkeypatch.delenv("MOCK_LLM_PROSE", raising=False)

    def enable_call_log(self) -> Path:
        path = self.tmp_path / "calls.jsonl"
        self.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(path))
        return path

    # -- task factories -------------------------------------------------

    def apply_task(self, index: int, **over):
        kwargs = dict(
            kind="apply_suggestions",
            task_id=f"apply-batch-{index}",
            prompt=(
                f"Apply batch {index} to {self.plan}\n\n"
                "## Changes Applied in Prior Batches\n\n"
                f"{PRIOR_CHANGES_TOKEN}\n"
            ),
            inputs=build_inputs(plan_file=str(self.plan)),
            batch_index=index,
            prior_changes_id=f"sec-{index}",
            phase_dir=str(self.phase_dir),
            declared_files=[str(self.plan)],
            expected_ids=[f"s{index}"],
        )
        kwargs.update(over)
        return make_write_task(**kwargs)

    def scheduler(self, tasks, **over):
        kwargs = dict(
            tasks=tasks,
            phase_dir=str(self.phase_dir),
            kind="apply_suggestions",
            output_dir=str(self.out_dir),
            plan_file=str(self.plan),
            anchor=str(self.plan),
            repo_root=str(self.root),
            log=lambda line: None,
        )
        kwargs.update(over)
        return WriteBatchScheduler(**kwargs)


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = WriteEnv(tmp_path, monkeypatch)
    yield e
    registry._config = None
    registry._config_key = None


class FakeState:
    """The slice of ``StateManager`` the scheduler touches, with a real state file."""

    def __init__(self, path: Path) -> None:
        self.state_file = str(path)
        self.state = {}
        path.write_text(json.dumps({"metrics": {}}), encoding="utf-8")

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value

    def save(self):
        pass

    def update_task_status(self, task_id, status, reason=None):
        self.state.setdefault("task_status", {})[task_id] = status
        if reason is not None:
            self.state.setdefault("task_status_reasons", {})[task_id] = reason

    def record_dependency_override(self, task_id, overridden_deps):
        self.state.setdefault("dependency_overrides", {})[task_id] = {
            "overridden_deps": list(overridden_deps)
        }

    def mark_phase_completed(self, phase):
        self.state.setdefault("phases_completed", {})[phase] = "now"

    def is_phase_completed(self, phase):
        return phase in self.state.get("phases_completed", {})


def _patch_edit(path: Path, text: str = "edit\n"):
    return {"path": str(path), "mode": "patch", "content": text}


# ===========================================================================
# D3e/D3g — the edit-capable invocation
# ===========================================================================

class TestWriteProfile:
    """`agent-write` is derived from the kind and rendered by the adapter."""

    @pytest.mark.parametrize("kind", sorted(WRITE_KINDS))
    def test_every_write_kind_runs_agent_write(self, kind):
        assert executor.profile_for_kind(kind) == PROFILE_AGENT_WRITE

    @pytest.mark.parametrize("provider_name,flags", [
        ("claude-code", ["--permission-mode", "acceptEdits"]),
        ("codex", ["--sandbox", "workspace-write"]),
        ("opencode", ["--agent", "build"]),
    ])
    def test_the_command_carries_the_runners_own_write_mechanism(
        self, provider_name, flags
    ):
        provider = registry.get_provider(provider_name)
        argv = provider.build_agent_command("P", "m", PROFILE_AGENT_WRITE)
        joined = " ".join(argv)
        for flag in flags:
            assert flag in argv, f"{provider_name}: {flag!r} missing from {argv}"
        # Never a blanket bypass: a runner with no honored write sandbox is
        # refused for `agent-write`, so fabricating one here would delete the
        # boundary that makes the profile eligible at all.
        for forbidden in ("--dangerously-skip-permissions", "--yolo",
                          "--dangerously-bypass-approvals-and-sandbox"):
            assert forbidden not in joined

    @pytest.mark.parametrize("provider_name", ["claude-code", "codex", "opencode"])
    def test_the_write_invocation_has_a_non_argv_transport(self, provider_name):
        """Task 6: each adapter's edit-capable invocation must not need argv.

        Apply and implement prompts are the largest the skill builds (they embed
        the batch AND the accumulated prior-batch context), so an argv-only write
        path would hit the Windows command-line cap on the last batch of a long run.
        """
        provider = registry.get_provider(provider_name)
        argv, stdin_text = provider.build_agent_command_for_transport(
            "PROMPT", "m", PROFILE_AGENT_WRITE, "stdin"
        )
        assert stdin_text == "PROMPT"
        assert "PROMPT" not in argv

    @pytest.mark.parametrize("name,runner", WRITE_RUNNERS)
    def test_a_real_dispatch_pins_cwd_to_the_repo_root(self, env, name, runner):
        """D3h: the sandbox root is the resolved repo root, never inherited."""
        env.use(runner)
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)])
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.exit_code == 0, result.messages
        records = [json.loads(l) for l in log.read_text().splitlines() if l]
        dispatches = [r for r in records if "--version" not in r["argv"]]
        assert dispatches, records
        assert Path(dispatches[-1]["cwd"]).resolve() == env.root.resolve()
        # The prompt left argv entirely (D2d), which is why the cap does not apply.
        assert dispatches[-1]["prompt_transport"] == "stdin"


# ===========================================================================
# D3g — all four outcomes, on real mock edits
# ===========================================================================

class TestFourWriteOutcomes:
    """The model's claim is never the verdict; the snapshot diff is."""

    def test_applied(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "did it"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.status == executor.STATUS_OK
        assert outcome.files_touched == ["plans/p.md"]
        assert result.exit_code == 0

    def test_clean_failure_is_success_claimed_with_no_bytes_moved(self, env):
        """The batch says `applied`; nothing changed. Recorded, loop continues."""
        env.edits([], result={"status": "applied", "ids": ["s0"],
                              "files_touched": ["plans/p.md"],
                              "summary": "claimed to apply"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
        assert outcome.status == executor.STATUS_OK
        assert outcome.files_touched == []
        assert "CLEAN FAILURE" in outcome.reason
        # The loop continued and the phase still completed.
        assert result.exit_code == 0
        # And the documented clean-failure form reached the prior-changes log, so a
        # later batch reads why this one contributed nothing.
        log = (env.out_dir / "prior_changes.jsonl").read_text()
        assert "verified no file changes" in log

    def test_partial_application_is_bytes_moved_plus_a_reported_failure(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"],
                          "summary": "could not finish"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert outcome.status == executor.STATUS_PARTIAL
        assert outcome.stops_the_loop
        assert "PARTIAL APPLICATION" in outcome.reason
        assert result.exit_code == 3
        # The half-applied bytes are deliberately still there.
        assert "edit" in env.plan.read_text()

    def test_partial_application_is_also_missing_expected_ids(self, env):
        """"Bytes moved" is not "this batch's changes landed" (D3g spot-check)."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "half"})
        task = env.apply_task(0, expected_ids=["s0", "s1", "s2"])
        result = env.scheduler([task]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert "s1" in outcome.reason and "s2" in outcome.reason

    def test_schema_rejection_is_prose_instead_of_result_json(self, env):
        """A runner that narrates fails the task, never counts as applied.

        Prose carries no JSON at all, so the ladder lands on rung 5
        (``unparseable_result``) rather than rung 6 — a *different rung* for the
        same D3g bucket, which is why the write outcome is its own axis: to a
        consumer, "narrated" and "returned the wrong object" are one failure.
        """
        env.edits([_patch_edit(env.plan)], prose=True)
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_UNPARSEABLE
        assert outcome.rung == 5
        assert outcome.write_outcome == executor.WRITE_SCHEMA_REJECTED
        assert result.exit_code == 3
        # Nothing was persisted as a batch result — the edit it DID make is left
        # in place and reported, never laundered into "probably applied".
        assert not (env.phase_dir / "apply_batch_0.json").exists()

    def test_schema_rejection_is_also_well_formed_json_of_the_wrong_shape(self, env):
        """Rung 6: it parsed, it is just not the batch-result contract."""
        env.edits([_patch_edit(env.plan)],
                  result={"verdict": "looks good", "groups": []})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_SCHEMA_INVALID
        assert outcome.rung == 6
        assert outcome.write_outcome == executor.WRITE_SCHEMA_REJECTED
        assert not (env.phase_dir / "apply_batch_0.json").exists()

    def test_a_result_missing_summary_is_a_schema_rejection(self, env):
        """`summary` is contract, not decoration: prior_changes reads it verbatim."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "files_touched": []})
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.outcomes[0].status == executor.STATUS_SCHEMA_INVALID


# ===========================================================================
# D3f — write containment
# ===========================================================================

class TestWriteContainment:
    """Both sides of every comparison are canonicalized (D1c); nothing is written."""

    @pytest.mark.parametrize("candidate", [
        "../outside.py",
        "../../etc/passwd",
        "/etc/passwd",
        "C:evil.py",
        "\\\\host\\share\\evil.py",
    ])
    def test_every_escape_shape_is_refused(self, env, candidate):
        with pytest.raises(write_safety.PathEscape):
            write_safety.contain_under_root(candidate, str(env.root))

    def test_a_symlinked_component_cannot_redirect_the_write(self, env):
        outside = env.tmp_path / "outside"
        outside.mkdir()
        link = env.root / "linked"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):  # pragma: no cover
            pytest.skip("symlinks unavailable")
        with pytest.raises(write_safety.PathEscape):
            write_safety.contain_under_root("linked/x.py", str(env.root))

    def test_a_task_declaring_a_file_outside_the_root_writes_nothing(self, env):
        outside = env.tmp_path / "outside.md"
        outside.write_text("untouched\n", encoding="utf-8")
        env.edits([_patch_edit(env.plan)])
        task = env.apply_task(0, declared_files=[str(outside)])
        result = env.scheduler([task]).run()
        assert result.exit_code == 3
        assert result.outcomes[0].status == executor.STATUS_REJECTED
        assert outside.read_text() == "untouched\n"
        # Nothing was dispatched, so the plan is untouched too.
        assert "edit" not in env.plan.read_text()

    def test_the_result_filename_must_match_the_kinds_pattern(self, env):
        task = env.apply_task(0)
        task["output_name"] = "implement_batch_0.json"
        task.pop("output_path", None)
        with pytest.raises(executor.ResultRejected) as exc:
            executor.resolve_output_target(task, str(env.phase_dir))
        assert "apply_batch" in exc.value.message

    @pytest.mark.parametrize("name", [
        "subagent_audit.jsonl", "subagent_dry_run.json", "prior_changes.jsonl",
        "write_batch_report.json", "orchestrator_output.json",
    ])
    def test_a_write_result_may_never_land_on_an_executor_owned_artifact(
        self, env, name
    ):
        task = env.apply_task(0)
        task["output_name"] = name
        task.pop("output_path", None)
        with pytest.raises(executor.ResultRejected):
            executor.resolve_output_target(task, str(env.phase_dir))

    def test_the_literal_prior_changes_token_never_reaches_a_runner(self, env):
        """A dispatch whose prompt still carries the token is a contract error."""
        env.edits([_patch_edit(env.plan)])
        outcome = executor.dispatch_task(
            env.apply_task(0),
            phase_dir=str(env.phase_dir),
            runner="codex:gpt-5.5",
            cwd=str(env.root),
        )
        assert outcome.status == executor.STATUS_REJECTED
        assert PRIOR_CHANGES_TOKEN in outcome.reason
        assert "edit" not in env.plan.read_text()


# ===========================================================================
# D3h — workspace safety
# ===========================================================================

class TestWorkspaceSafety:

    def test_write_concurrency_defaults_to_one(self, env):
        """A separate key from the read gate, and its default is 1."""
        assert registry.get_max_concurrent_write_subagents() == 1
        # Even with the READ gate at 4 (the shipped default), writes stay at 1.
        assert registry.get_max_concurrent_subagents() == 4

    def test_a_shared_artifact_phase_is_pinned_to_one_regardless(self, env):
        env.use("codex:gpt-5.5", max_concurrent_write_subagents=8)
        scheduler = env.scheduler([env.apply_task(0)])
        assert scheduler.write_ceiling() == 1

    def test_a_partitioned_phase_may_raise_the_ceiling(self, env):
        env.use("codex:gpt-5.5", max_concurrent_write_subagents=4)
        scheduler = env.scheduler([], kind="apply_code_fixes")
        assert scheduler.write_ceiling() == 4

    def test_overlapping_declared_file_sets_are_refused_co_scheduling(self, env):
        """Checked executor-side, so a hand-edited task file cannot beat it."""
        shared = [
            {"task_id": "a", "declared_files": ["src/a.py"], "can_parallelize": True},
            {"task_id": "b", "declared_files": ["src/a.py"], "can_parallelize": True},
        ]
        groups = write_safety.admit_disjoint_groups(shared, ceiling=4)
        assert [len(g) for g in groups] == [1, 1]
        assert write_safety.overlapping_pairs(shared) == [("a", "b", ["src/a.py"])]

    def test_disjoint_declared_file_sets_are_admitted_together(self, env):
        disjoint = [
            {"task_id": "a", "declared_files": ["src/a.py"], "can_parallelize": True},
            {"task_id": "b", "declared_files": ["src/b.py"], "can_parallelize": True},
        ]
        groups = write_safety.admit_disjoint_groups(disjoint, ceiling=4)
        assert [len(g) for g in groups] == [2]

    def test_a_task_declaring_no_file_set_runs_alone(self, env):
        """"Undeclared" is not "touches nothing"."""
        tasks = [
            {"task_id": "a", "declared_files": [], "can_parallelize": True},
            {"task_id": "b", "declared_files": ["src/b.py"], "can_parallelize": True},
        ]
        groups = write_safety.admit_disjoint_groups(tasks, ceiling=4)
        assert [len(g) for g in groups] == [1, 1]

    def test_a_ceiling_of_one_never_co_schedules(self, env):
        tasks = [
            {"task_id": "a", "declared_files": ["src/a.py"], "can_parallelize": True},
            {"task_id": "b", "declared_files": ["src/b.py"], "can_parallelize": True},
        ]
        assert [len(g) for g in write_safety.admit_disjoint_groups(tasks, ceiling=1)] \
            == [1, 1]

    def test_the_snapshot_lives_under_the_phase_dir(self, env):
        env.edits([_patch_edit(env.plan)])
        env.scheduler([env.apply_task(0)]).run()
        snapshot_dir = env.phase_dir / "pre_batch" / "apply-batch-0"
        assert (snapshot_dir / "manifest.json").is_file()
        payloads = list((snapshot_dir / "files").iterdir())
        assert payloads, "the snapshot payload is the rollback point"
        assert "edit" not in payloads[0].read_text(), (
            "the snapshot must be the PRE-batch bytes"
        )

    def test_a_task_with_nothing_to_snapshot_is_refused(self, env):
        """No snapshot means no verdict, so it is a refusal, not a warning."""
        task = env.apply_task(0, declared_files=[], snapshot_files=[])
        result = env.scheduler([task]).run()
        assert result.outcomes[0].status == executor.STATUS_REJECTED
        assert "declares no files to snapshot" in result.outcomes[0].reason

    def test_the_repo_root_is_the_git_toplevel_canonicalized(self, env):
        assert Path(write_safety.resolve_repo_root(str(env.plan))).resolve() == \
            env.root.resolve()

    def test_every_write_mode_records_a_dirty_tree_baseline(self, env, capsys):
        """D3h: `record_pre_existing_changes` for every executor-backed write mode."""
        from utils.apply_orchestrator_base import ApplyOrchestratorBase

        class Probe(ApplyOrchestratorBase):
            phase_name = "apply-suggestions"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False
            marks_phase_completed = False

        probe = Probe.__new__(Probe)
        probe.state = FakeState(env.tmp_path / "state.json")
        probe.args = None
        monkey = env.monkeypatch
        monkey.chdir(env.root)
        (env.root / "dirty.txt").write_text("x", encoding="utf-8")
        import subprocess
        subprocess.run(["git", "add", "dirty.txt"], cwd=env.root, check=True)
        probe._record_pre_existing_changes()
        out = capsys.readouterr().out
        assert "dirty tree at start" in out
        assert "dirty.txt" in probe.state.get("pre_existing_changes")


# ===========================================================================
# D3n — a write phase schedules, it does not fan out
# ===========================================================================

class TestScheduler:

    def _three(self, env):
        return [env.apply_task(i) for i in range(3)]

    def test_three_batches_dispatch_in_batch_index_order(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0", "s1", "s2"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        # Deliberately out of order in the list: the scheduler sorts by index.
        tasks = self._three(env)
        result = env.scheduler([tasks[2], tasks[0], tasks[1]]).run()
        assert result.exit_code == 0, result.messages
        assert result.batches_dispatched == [0, 1, 2]
        assert [o.task_id for o in result.outcomes] == [
            "apply-batch-0", "apply-batch-1", "apply-batch-2",
        ]
        # Each batch was verified before the next started: the audit log alternates
        # dispatched/completed rather than emitting three dispatches up front.
        events = [
            r["event"] for r in write_safety.AuditLog(str(env.phase_dir)).records()
        ]
        assert events == ["dispatched", "completed"] * 3
        assert len([l for l in log.read_text().splitlines() if l]) >= 3

    def test_batch_three_sees_batch_one_and_two_and_no_literal_token(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)])
        outcome_summaries = []

        # Each batch reports a distinguishable summary, so the injected context is
        # identifiable rather than merely non-empty.
        summaries = ["FIRST-CHANGE", "SECOND-CHANGE", "THIRD-CHANGE"]
        results = []
        for index, summary in enumerate(summaries):
            env.edits(
                [_patch_edit(env.plan, f"edit-{index}\n")],
                result={"status": "applied", "ids": [f"s{index}"],
                        "files_touched": ["plans/p.md"], "summary": summary},
            )
            scheduler = env.scheduler([env.apply_task(index)], finalize=False)
            results.append(scheduler.run())
            outcome_summaries.append(results[-1].outcomes[0].summary)

        assert outcome_summaries == summaries
        dispatched = [
            json.loads(l) for l in log.read_text().splitlines() if l
        ]
        third = [r for r in dispatched if "batch 2" in (r.get("prompt") or "")]
        assert third, "batch 2's prompt was never dispatched"
        prompt = third[-1]["prompt"]
        assert "FIRST-CHANGE" in prompt
        assert "SECOND-CHANGE" in prompt
        assert PRIOR_CHANGES_TOKEN not in prompt, (
            "the literal token must never reach a runner"
        )
        # And the immutable task JSON was never rewritten.
        assert PRIOR_CHANGES_TOKEN in env.apply_task(2)["prompt"]

    def test_the_first_batch_gets_the_documented_empty_context(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)])
        env.scheduler([env.apply_task(0)]).run()
        dispatched = [json.loads(l) for l in log.read_text().splitlines() if l]
        prompt = [r["prompt"] for r in dispatched if r.get("prompt")][-1]
        assert "(none — this is the first batch)" in prompt

    def test_an_apply_prompt_without_the_token_fails_the_dispatch(self, env):
        """A producer that dropped the token would silently lose all context."""
        task = env.apply_task(0, prompt="Apply it, no token here.\n")
        result = env.scheduler([task]).run()
        assert result.outcomes[0].status == executor.STATUS_REJECTED
        assert PRIOR_CHANGES_TOKEN in result.outcomes[0].reason

    def test_finalization_runs_and_writes_the_summary_and_plan_reference(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        state = FakeState(env.tmp_path / "state.json")
        summary = env.out_dir / "implement" / "implementation_summary.md"
        result = env.scheduler(
            [env.apply_task(0)], state=state, summary_file=str(summary),
        ).run()
        assert result.exit_code == 0
        assert result.finalized is True
        assert Path(result.summary_file).is_file()
        assert "## Task Results" in Path(result.summary_file).read_text()
        assert state.is_phase_completed("apply-suggestions")
        assert "## Implementation Summary" in env.plan.read_text()

    def test_a_run_that_dispatched_every_batch_but_skipped_finalization_fails(
        self, env
    ):
        """The write-side counterpart of "a prompt with no consumer is unclosed".

        A run with `finalize=False` dispatches identically and produces identical
        batch results — and is still a FAILED run, because no phase was completed,
        no summary was written and no plan reference was inserted. This test is the
        guard that the suite notices.
        """
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        state = FakeState(env.tmp_path / "state.json")
        summary = env.out_dir / "implement" / "implementation_summary.md"
        result = env.scheduler(
            [env.apply_task(0)], state=state, summary_file=str(summary),
            finalize=False,
        ).run()
        assert result.outcomes[0].write_outcome == executor.WRITE_APPLIED
        assert result.finalized is False
        assert result.summary_file is None
        assert not summary.exists()
        assert not state.is_phase_completed("apply-suggestions")
        assert "## Implementation Summary" not in env.plan.read_text()

    def test_a_mid_loop_partial_application_stops_at_the_batch_boundary(self, env):
        state = FakeState(env.tmp_path / "state.json")
        tasks = self._three(env)
        # Batch 0 applies cleanly; batch 1 half-applies. Batch 2 must never run.
        env.edits([_patch_edit(env.plan, "ok-0\n")],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "first"})
        first = env.scheduler([tasks[0]], state=state, finalize=False).run()
        assert first.outcomes[0].write_outcome == executor.WRITE_APPLIED

        env.edits([_patch_edit(env.plan, "half-1\n")],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"], "summary": "broke"})
        result = env.scheduler(tasks, state=state).run()

        assert result.exit_code == 3
        assert result.stopped_at_batch == 1
        assert 2 not in result.batches_dispatched
        assert not state.is_phase_completed("apply-suggestions")
        # Snapshots intact for both attempted batches.
        assert (env.phase_dir / "pre_batch" / "apply-batch-1" / "manifest.json").is_file()
        # The exit-3 message names the restore command rather than reverting.
        joined = "\n".join(result.messages)
        assert "util restore-batch" in joined
        assert "--task-id apply-batch-1" in joined
        # Batch 2 was never dispatched, so no result file exists for it.
        assert not (env.phase_dir / "apply_batch_2.json").exists()

    def test_resume_redispatches_only_the_unfinished_batches(self, env):
        state = FakeState(env.tmp_path / "state.json")
        tasks = self._three(env)
        env.edits([_patch_edit(env.plan, "ok-0\n")],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "first"})
        env.scheduler([tasks[0]], state=state, finalize=False).run()

        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan, "ok-rest\n")],
                  result={"status": "applied", "ids": ["s0", "s1", "s2"],
                          "files_touched": ["plans/p.md"], "summary": "rest"})
        result = env.scheduler(tasks, state=state).run()
        assert result.exit_code == 0, result.messages

        skipped = [o for o in result.outcomes if o.skipped]
        assert [o.task_id for o in skipped] == ["apply-batch-0"]

        dispatched_prompts = [
            json.loads(l)["prompt"] for l in log.read_text().splitlines() if l
        ]
        assert not any("batch 0" in p for p in dispatched_prompts), (
            "a completed batch must not be re-dispatched"
        )
        # The prior-changes log was REPLAYED from disk, not reconstructed: batch 1
        # saw batch 0's summary even though this process never ran batch 0.
        assert any("first" in p for p in dispatched_prompts)


# ===========================================================================
# D3n — the three human-task strategies
# ===========================================================================

class TestHumanTaskStrategies:
    """One resolution, then a graph walk — identical on both invokers."""

    def _graph(self):
        # T001 (human) -> T003 -> T005 ; T002 independent
        return [
            {"task_id": "T001", "executor_role": "human", "depends_on": [],
             "task_title": "Create the API key"},
            {"task_id": "T002", "executor_role": "agent", "depends_on": []},
            {"task_id": "T003", "executor_role": "agent", "depends_on": ["T001"]},
            {"task_id": "T005", "executor_role": "agent", "depends_on": ["T003"]},
        ]

    def test_skip_dependents_propagates_transitively(self):
        plan = resolve_human_task_plan(self._graph(),
                                       strategy=STRATEGY_SKIP_DEPENDENTS)
        assert set(plan.skipped) == {"T001", "T003", "T005"}
        assert "T002" not in plan.skipped
        assert plan.chain == {"T003": "T001", "T005": "T003"}
        assert "depends on skipped task T001" in plan.skipped["T003"]
        assert plan.overrides == {}

    def test_skip_continue_records_a_dependency_override_per_unblocked_task(self):
        plan = resolve_human_task_plan(self._graph(),
                                       strategy=STRATEGY_SKIP_CONTINUE)
        assert set(plan.skipped) == {"T001"}
        assert plan.overrides == {"T003": ["T001"]}
        assert "T005" not in plan.skipped, "dependents run under skip-continue"

    def test_pause_and_ask_needs_an_answer_and_honors_it(self):
        pending = resolve_human_task_plan(self._graph(), strategy=STRATEGY_PAUSE)
        assert pending.pending == ["T001"]
        assert pending.skipped == {}

        done = resolve_human_task_plan(
            self._graph(), strategy=STRATEGY_PAUSE, decisions={"T001": "done"}
        )
        assert done.completed == ["T001"] and done.skipped == {}

        skip = resolve_human_task_plan(
            self._graph(), strategy=STRATEGY_PAUSE, decisions={"T001": "skip"}
        )
        assert set(skip.skipped) == {"T001", "T003", "T005"}
        assert "pause-and-ask" in skip.skipped["T001"]

    def test_an_unanswered_human_task_is_exit_three_with_both_re_invocations(
        self, env
    ):
        state = FakeState(env.tmp_path / "state.json")
        tasks = [
            env.apply_task(0, executor_role="human"),
            env.apply_task(1),
        ]
        result = env.scheduler(
            tasks, state=state, kind="implement",
            human_task_strategy=STRATEGY_PAUSE,
        ).run()
        assert result.exit_code == 3
        joined = "\n".join(result.messages)
        assert "--human-task apply-batch-0=done" in joined
        assert "--human-task apply-batch-0=skip" in joined
        assert not state.is_phase_completed("implement")
        # Fail closed: nothing was dispatched at all.
        assert result.outcomes == []

    @pytest.mark.parametrize("strategy", list(
        [STRATEGY_SKIP_CONTINUE, STRATEGY_SKIP_DEPENDENTS]
    ))
    def test_the_state_entries_are_the_same_on_both_invokers(self, env, strategy):
        """The skip set, the reasons and the override records must be identical."""
        graph = [
            env.apply_task(0, executor_role="human", kind="implement",
                           task_id="T001", output_name="implement_batch_0.json"),
            env.apply_task(1, kind="implement", task_id="T003",
                           depends_on=["T001"],
                           output_name="implement_batch_1.json"),
        ]
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s1"],
                          "files_touched": ["plans/p.md"], "summary": "S"})

        def run(respond):
            state = FakeState(env.tmp_path / f"state-{respond is not None}.json")
            scheduler = env.scheduler(
                graph, state=state, kind="implement",
                human_task_strategy=strategy, respond=respond, finalize=False,
            )
            scheduler.run()
            return state.state

        native = run(lambda task, prompt: {
            "status": "applied", "ids": ["s1"],
            "files_touched": ["plans/p.md"], "summary": "S",
        })
        # Re-create the plan for the second run so the byte comparison is fair.
        env.plan.write_text("# Plan\n\n## Step 1\nbody\n", encoding="utf-8")
        shutil.rmtree(env.phase_dir / "pre_batch", ignore_errors=True)
        for stale in env.phase_dir.glob("implement_batch_*.json"):
            stale.unlink()
        executed = run(None)

        assert native.get("task_status", {}).get("T001") == "skipped"
        assert executed.get("task_status", {}).get("T001") == "skipped"
        assert native["task_status_reasons"]["T001"] == \
            executed["task_status_reasons"]["T001"]
        assert native.get("dependency_overrides") == \
            executed.get("dependency_overrides")


# ===========================================================================
# D3l — auditable and reversible before it is trusted
# ===========================================================================

class TestSecuritySet:

    def test_a_denied_path_fails_the_batch_and_is_restored(self, env):
        """Post-run enforcement is detective, and it restores from the snapshot."""
        secret = env.root / ".env"
        secret.write_text("TOKEN=original\n", encoding="utf-8")
        env.edits([{"path": ".env", "mode": "patch", "content": "LEAKED=1\n"}],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": [".env"], "summary": "edited .env"})
        task = env.apply_task(0, declared_files=[str(env.plan)],
                             snapshot_files=[str(env.plan), str(secret)])
        result = env.scheduler([task]).run()

        assert result.exit_code == 3
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION
        assert outcome.status == executor.STATUS_POLICY_VIOLATION
        assert ".env" in outcome.reason
        assert "driver.write_paths.deny" in outcome.reason
        # Restored from the pre-batch snapshot.
        assert secret.read_text() == "TOKEN=original\n"
        events = [r["event"] for r in
                  write_safety.AuditLog(str(env.phase_dir)).records()]
        assert "restored" in events

    def test_a_denied_declared_path_is_refused_before_the_runner_starts(self, env):
        """Pre-dispatch enforcement: no child, so nothing to restore."""
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)])
        task = env.apply_task(0, declared_files=[".git/config"],
                             snapshot_files=[str(env.plan)])
        result = env.scheduler([task]).run()
        assert result.outcomes[0].status == executor.STATUS_REJECTED
        assert result.outcomes[0].write_outcome == executor.WRITE_POLICY_VIOLATION
        dispatches = [
            json.loads(l) for l in
            (log.read_text().splitlines() if log.exists() else [])
        ]
        assert not any(r.get("prompt") for r in dispatches), (
            "a policy-refused batch must not spawn a runner"
        )

    def test_the_audit_log_records_every_dispatch_without_prompt_bytes(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        task = env.apply_task(0)
        env.scheduler([task]).run()
        audit = write_safety.AuditLog(str(env.phase_dir))
        records = audit.records()
        assert [r["event"] for r in records] == ["dispatched", "completed"]

        dispatched = records[0]
        assert dispatched["task_id"] == "apply-batch-0"
        assert dispatched["profile"] == PROFILE_AGENT_WRITE
        assert Path(dispatched["sandbox_root"]).resolve() == env.root.resolve()
        assert Path(dispatched["cwd"]).resolve() == env.root.resolve()
        assert len(dispatched["prompt_sha256"]) == 64
        assert dispatched["prompt_bytes"] > 0

        # No prompt bytes anywhere in the file, and no credential-shaped value.
        raw = Path(audit.path).read_text(encoding="utf-8")
        assert "Apply batch 0" not in raw
        assert PRIOR_CHANGES_TOKEN not in raw
        for token in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "sk-"):
            assert token not in raw
        # The argv IS recorded, with the prompt elided to its digest.
        assert any("<prompt sha256:" in arg for arg in dispatched["argv"])

        completed = records[1]
        assert completed["write_outcome"] == executor.WRITE_APPLIED
        assert completed["files_touched"] == ["plans/p.md"]
        assert completed["observed"]["plans/p.md"]["sha256_before"] != \
            completed["observed"]["plans/p.md"]["sha256_after"]

    def test_the_audit_log_redacts_a_secret_shaped_argv_value(self):
        argv = ["claude", "--api-key", "sk-secret", "--token=abc", "PROMPT"]
        redacted = write_safety.redact_argv(argv, "PROMPT")
        assert "sk-secret" not in redacted
        assert "<redacted>" in redacted
        assert "--token=<redacted>" in redacted
        assert redacted[-1].startswith("<prompt sha256:")

    def test_the_audit_log_is_append_only(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        env.scheduler([env.apply_task(0)]).run()
        first = len(write_safety.AuditLog(str(env.phase_dir)).records())
        env.plan.write_text("# Plan\n", encoding="utf-8")
        shutil.rmtree(env.phase_dir / "pre_batch")
        (env.phase_dir / "apply_batch_0.json").unlink()
        env.scheduler([env.apply_task(0)]).run()
        assert len(write_safety.AuditLog(str(env.phase_dir)).records()) > first

    def test_dry_run_renders_every_command_and_starts_no_child(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)])
        tasks = [env.apply_task(i) for i in range(3)]
        result = env.scheduler(tasks, dry_run=True).run()

        assert result.exit_code == 0
        report = json.loads(Path(result.dry_run_path).read_text())
        assert len(report["tasks"]) == 3
        assert all(entry["would_dispatch"] for entry in report["tasks"])
        for entry in report["tasks"]:
            assert entry["argv"], "every command must be rendered"
            assert any("<prompt sha256:" in arg for arg in entry["argv"])
            assert entry["profile"] == PROFILE_AGENT_WRITE
            assert Path(entry["cwd"]).resolve() == env.root.resolve()
            assert entry["prompt_transport"] == "stdin"
            assert entry["prompt_utf16_units"] > 0
        assert report["ceilings"]["effective_write_parallelism"] == 1
        assert report["write_paths"]["deny"]

        # Zero dispatch, and nothing else changed.
        prompts = [
            r for r in (json.loads(l) for l in
                        (log.read_text().splitlines() if log.exists() else []))
            if r.get("prompt")
        ]
        assert prompts == []
        assert "edit" not in env.plan.read_text()
        assert not (env.phase_dir / "pre_batch").exists()
        assert not (env.out_dir / "prior_changes.jsonl").exists()

    def test_dry_run_reports_a_batch_that_would_be_refused(self, env):
        result = env.scheduler(
            [env.apply_task(0, declared_files=[".git/config"])], dry_run=True,
        ).run()
        report = json.loads(Path(result.dry_run_path).read_text())
        entry = report["tasks"][0]
        assert entry["would_dispatch"] is False
        assert "driver.write_paths.deny" in entry["reason"]

    def test_restore_batch_restores_a_failed_batch(self, env, capsys):
        env.edits([_patch_edit(env.plan, "half\n")],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"], "summary": "broke"})
        original = env.plan.read_text()
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.exit_code == 3
        assert env.plan.read_text() != original

        code = restore_batch.main([
            "--phase-dir", str(env.phase_dir), "--task-id", "apply-batch-0",
        ])
        assert code == 0
        assert env.plan.read_text() == original
        # And the rollback is itself an audit event.
        events = [r["event"] for r in
                  write_safety.AuditLog(str(env.phase_dir)).records()]
        assert events[-1] == "restored"

    def test_restore_batch_refuses_a_file_that_changed_after_the_run(self, env):
        env.edits([_patch_edit(env.plan, "half\n")],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"], "summary": "broke"})
        env.scheduler([env.apply_task(0)]).run()
        # A human inspected the half-applied file and edited it.
        env.plan.write_text("hand-repaired\n", encoding="utf-8")

        code = restore_batch.main([
            "--phase-dir", str(env.phase_dir), "--task-id", "apply-batch-0",
            "--json",
        ])
        assert code == 3
        assert env.plan.read_text() == "hand-repaired\n", (
            "reverting an edit made after the run would be a second data-loss bug"
        )
        # --force is the explicit override.
        assert restore_batch.main([
            "--phase-dir", str(env.phase_dir), "--task-id", "apply-batch-0",
            "--force",
        ]) == 0
        assert env.plan.read_text() != "hand-repaired\n"

    def test_restore_batch_all_failed_finds_the_batch_from_the_audit_log(self, env):
        env.edits([_patch_edit(env.plan, "half\n")],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"], "summary": "broke"})
        original = env.plan.read_text()
        env.scheduler([env.apply_task(0)]).run()
        assert restore_batch.main([
            "--phase-dir", str(env.phase_dir), "--all-failed",
        ]) == 0
        assert env.plan.read_text() == original

    def test_restore_batch_without_a_selector_refuses(self, env):
        """Restoring everything by default IS the automatic revert to avoid."""
        assert restore_batch.main(["--phase-dir", str(env.phase_dir)]) == 3

    def test_restore_batch_is_nothing_to_do_when_no_batch_failed(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        env.scheduler([env.apply_task(0)]).run()
        assert restore_batch.main([
            "--phase-dir", str(env.phase_dir), "--all-failed",
        ]) == 2


# ===========================================================================
# The result contract, under both invokers, for every write kind
# ===========================================================================

class TestEveryWriteKindClosesTheLoop:
    """A kind that dispatches but whose result no consumer reads is a failed row."""

    KIND_ARTIFACT = {
        "apply_suggestions": "plans/p.md",
        "apply_task_suggestions": "plans/p.md",
        "apply_code_fixes": "src/a.py",
        "implement": "src/a.py",
    }

    def _task(self, env, kind):
        artifact = env.root / self.KIND_ARTIFACT[kind]
        prompt = f"Do the {kind} work on {artifact}\n"
        if kind != "implement":
            prompt += f"\n{PRIOR_CHANGES_TOKEN}\n"
        return make_write_task(
            kind=kind,
            task_id=f"{kind}-0",
            prompt=prompt,
            inputs=build_inputs(plan_file=str(env.plan)),
            batch_index=0,
            prior_changes_id=f"{kind}-key",
            phase_dir=str(env.phase_dir),
            declared_files=[str(artifact)],
            expected_ids=["x1"],
            output_name=write_output_name(kind, 0),
        )

    @pytest.mark.parametrize("kind", sorted(WRITE_KINDS))
    @pytest.mark.parametrize("invoker", ["executor", "native"])
    def test_a_batch_result_a_prior_change_and_a_metric(self, env, kind, invoker):
        artifact = env.root / self.KIND_ARTIFACT[kind]
        state = FakeState(env.tmp_path / f"state-{kind}-{invoker}.json")
        task = self._task(env, kind)

        if invoker == "executor":
            env.edits(
                [{"path": self.KIND_ARTIFACT[kind], "mode": "patch",
                  "content": "# edited\n"}],
                result={"status": "applied", "ids": ["x1"],
                        "files_touched": [self.KIND_ARTIFACT[kind]],
                        "summary": f"{kind} applied"},
            )
            respond = None
        else:
            def respond(t, prompt, _artifact=artifact, _kind=kind):
                # The native invoker's subagent really edits the workspace too,
                # otherwise the byte comparison would classify it a clean failure —
                # which is exactly the equivalence being asserted.
                _artifact.write_text(
                    _artifact.read_text() + "# edited\n", encoding="utf-8"
                )
                return {
                    "status": "applied", "ids": ["x1"],
                    "files_touched": [self.KIND_ARTIFACT[_kind]],
                    "summary": f"{_kind} applied",
                }

        result = env.scheduler(
            [task], kind=kind, state=state, respond=respond, finalize=False,
        ).run()

        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_APPLIED, outcome.reason

        # 1. the batch result file
        written = env.phase_dir / write_output_name(kind, 0)
        assert written.is_file()
        payload = json.loads(written.read_text())
        assert payload["status"] == "applied"
        assert payload["summary"] == f"{kind} applied"

        # 2. the prior_changes entry, keyed by the STABLE key
        entries = [
            json.loads(l) for l in
            (env.out_dir / "prior_changes.jsonl").read_text().splitlines() if l
        ]
        mine = [e for e in entries if e["id"] == f"{kind}-key"]
        assert len(mine) == 1
        assert mine[0]["summary"] == f"{kind} applied"

        # 3. the metrics record
        metrics = json.loads(Path(state.state_file).read_text())["metrics"]
        phase_metrics = metrics[result.phase]
        assert phase_metrics, metrics
        entry = phase_metrics[-1]
        assert f"{kind}-0" in entry["label"]
        assert entry["duration_ms"] >= 0
        # Usage is reported or absent, never fabricated as zero.
        assert "token_count" not in entry or entry["token_count"] > 0

    def test_both_invokers_write_byte_identical_result_files(self, env):
        """D3a extended to the write path: `persist_result` is shared."""
        payload = {
            "status": "applied", "ids": ["x1"],
            "files_touched": ["plans/p.md"], "summary": "identical",
        }
        task = self._task(env, "apply_suggestions")

        env.edits([_patch_edit(env.plan)], result=payload)
        env.scheduler([task], finalize=False).run()
        via_executor = (env.phase_dir / "apply_batch_0.json").read_bytes()

        (env.phase_dir / "apply_batch_0.json").unlink()
        shutil.rmtree(env.phase_dir / "pre_batch")
        env.plan.write_text("# Plan\n\n## Step 1\nbody\n", encoding="utf-8")

        def respond(t, prompt):
            env.plan.write_text(env.plan.read_text() + "edit\n", encoding="utf-8")
            return payload

        env.scheduler([task], respond=respond, finalize=False).run()
        via_native = (env.phase_dir / "apply_batch_0.json").read_bytes()
        assert via_executor == via_native


# ===========================================================================
# Additivity and detach
# ===========================================================================

class TestAdditivityAndDetach:

    def test_with_the_flag_off_the_write_executor_never_runs(self, env):
        """Claude Code's native path is byte-identical to before (task 6 rule)."""
        from utils.write_batch_scheduler import run_write_phase

        class Args:
            exec_subagents = False
            dry_run_subagents = False

        assert run_write_phase(Args(), tasks=[], phase_dir=str(env.phase_dir),
                               kind="apply_suggestions",
                               output_dir=str(env.out_dir)) is None
        # No executor-owned artifact was created.
        assert not (env.phase_dir / "subagent_audit.jsonl").exists()
        assert not (env.phase_dir / "pre_batch").exists()
        assert not (env.phase_dir / "write_batch_report.json").exists()

    def test_the_apply_base_declines_without_a_flag(self, env):
        from utils.apply_orchestrator_base import ApplyOrchestratorBase

        class Probe(ApplyOrchestratorBase):
            phase_name = "apply-suggestions"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False
            marks_phase_completed = False
            write_kind = "apply_suggestions"

        probe = Probe.__new__(Probe)

        class Args:
            exec_subagents = False
            dry_run_subagents = False

        probe.args = Args()
        probe.batches = [{"prompt": "x"}]
        assert probe.run_write_executor() is None

    @pytest.mark.parametrize("mode", [
        "generate-tasks", "implement", "apply-suggestions",
        "apply-task-suggestions", "apply-code-fixes", "full",
    ])
    def test_every_executor_backed_mode_accepts_detach(self, mode):
        import modes as mode_table
        from utils import detach

        assert mode_table.resolve(mode).supports_detach
        assert detach.mode_supports_detach(mode)

    def test_status_json_carries_per_task_write_progress(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0", "s1"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        tasks = [env.apply_task(0), env.apply_task(1)]
        env.scheduler(tasks).run()
        status = json.loads((env.phase_dir / ".status.json").read_text())
        progress = status["progress"]["write_subagents"]
        assert progress["total"] == 2
        assert progress["batches_dispatched"] == [0, 1]
        assert progress["finalized"] is True
        assert [t["task_id"] for t in progress["tasks"]] == [
            "apply-batch-0", "apply-batch-1",
        ]
        assert all(t["write_outcome"] == executor.WRITE_APPLIED
                   for t in progress["tasks"])
        assert progress["failed_task_ids"] == []

    def test_status_json_records_the_stopping_batch(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"], "summary": "broke"})
        env.scheduler([env.apply_task(0), env.apply_task(1)]).run()
        progress = json.loads(
            (env.phase_dir / ".status.json").read_text()
        )["progress"]["write_subagents"]
        assert progress["stopped_at_batch"] == 0
        assert progress["finalized"] is False
        assert progress["failed_task_ids"] == ["apply-batch-0"]

    def test_a_run_ends_in_exactly_one_terminal_state(self, env):
        """0 or 3, never both, and never a report that says neither."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        ok = env.scheduler([env.apply_task(0)]).run()
        assert ok.exit_code == 0 and ok.finalized and ok.stopped_at_batch is None

        env.plan.write_text("# Plan\n", encoding="utf-8")
        shutil.rmtree(env.phase_dir / "pre_batch")
        (env.phase_dir / "apply_batch_0.json").unlink()
        env.edits([_patch_edit(env.plan)],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"], "summary": "broke"})
        bad = env.scheduler([env.apply_task(0)]).run()
        assert bad.exit_code == 3 and not bad.finalized
        assert bad.stopped_at_batch == 0

        report = json.loads(
            (env.phase_dir / "write_batch_report.json").read_text()
        )
        assert report["exit"] == 3
        assert report["finalized"] is False


# ===========================================================================
# The mock's own write path — it must be able to escape, and be stopped
# ===========================================================================

class TestMockWritePath:

    def test_an_edit_outside_cwd_is_a_loud_test_authoring_error(self, env, tmp_path):
        """The sandbox assertions must never pass vacuously."""
        import subprocess

        outside = tmp_path / "escape.txt"
        completed = subprocess.run(
            [sys.executable, str(MOCK_LLM), "exec", "--sandbox", "workspace-write",
             "--json", "--model", "m", "prompt"],
            cwd=str(env.root), capture_output=True, text=True,
            env={**os.environ, "MULTI_LLM_TEST_MODE": "1",
                 "MOCK_LLM_PROVIDER": "codex",
                 "MOCK_LLM_EDITS": json.dumps([
                     {"path": str(outside), "mode": "create", "content": "x"}
                 ])},
        )
        assert completed.returncode == 9, completed.stderr
        assert "test-authoring error" in completed.stderr
        assert not outside.exists()

    def test_the_mock_really_edits_files_relative_to_cwd(self, env):
        import subprocess

        completed = subprocess.run(
            [sys.executable, str(MOCK_LLM), "exec", "--sandbox", "workspace-write",
             "--json", "--model", "m", "prompt"],
            cwd=str(env.root), capture_output=True, text=True,
            env={**os.environ, "MULTI_LLM_TEST_MODE": "1",
                 "MOCK_LLM_PROVIDER": "codex",
                 "MOCK_LLM_EDITS": json.dumps([
                     {"path": "src/new.py", "mode": "create", "content": "n = 1\n"}
                 ])},
        )
        assert completed.returncode == 0, completed.stderr
        assert (env.root / "src" / "new.py").read_text() == "n = 1\n"


# ===========================================================================
# D3l — --dry-run-subagents applies to the READ half too
# ===========================================================================

class TestReadDryRun:
    """Without this the flag would dispatch for real on a read phase."""

    def _validation_task_file(self, env):
        from utils.subagent_tasks import make_task, make_task_file, write_task_file

        prompt = (
            'You are validating review suggestions.\n\n'
            '[{"index": 0, "group_hash": "aaaa000000000000"}]\n\n'
            "Print ONE JSON object and nothing else."
        )
        task = make_task(
            kind="validation",
            task_id="validation-batch-0",
            prompt=prompt,
            output_name="validation_batch_0.json",
            inputs=build_inputs(grouped_file=str(env.phase_dir / "grouped.json")),
            phase_dir=str(env.phase_dir),
            extra={"batch_index": 0, "group_ids": ["aaaa000000000000"],
                   "group_indices": [0], "groups_count": 1},
        )
        document = make_task_file(
            kind="validation", tasks=[task],
            inputs=task["inputs"], task_list_key="batches",
            extra={"total_batches": 1, "phase_dir": str(env.phase_dir)},
        )
        path = env.phase_dir / "validation_tasks.json"
        write_task_file(str(path), document)
        return path

    def test_a_read_phase_renders_and_does_not_dispatch(self, env):
        log = env.enable_call_log()
        task_file = self._validation_task_file(env)

        class Args:
            exec_subagents = True
            dry_run_subagents = True

        continued = executor.continue_past_marker(
            Args(), task_file=str(task_file), phase_dir=str(env.phase_dir),
            phase="review-plan", cwd=str(env.root),
        )
        # False = "host-driven": the phase prints its marker and its follow-up
        # command, exactly as it would with the executor off.
        assert continued is False
        report = json.loads((env.phase_dir / "subagent_dry_run.json").read_text())
        assert report["kind"] == "validation"
        entry = report["tasks"][0]
        assert entry["would_dispatch"] is True
        assert entry["profile"] == "agent-readonly"
        assert any("<prompt sha256:" in arg for arg in entry["argv"])
        assert "--sandbox" in entry["argv"] and "read-only" in entry["argv"]
        assert entry["prompt_transport"] == "stdin"
        # No result file, and nothing but `--version` was ever spawned.
        assert not (env.phase_dir / "validation_batch_0.json").exists()
        prompts = [
            r for r in (json.loads(line) for line in
                        (log.read_text().splitlines() if log.exists() else []))
            if r.get("prompt")
        ]
        assert prompts == []

    def test_a_human_task_is_reported_as_never_dispatched(self, env):
        from utils.subagent_tasks import make_task, make_task_file, write_task_file

        task = make_task(
            kind="human_decision_judge",
            task_id="judge-g1",
            prompt="judge this",
            output_name="human_decision_batch_0.json",
            inputs=build_inputs(),
            phase_dir=str(env.phase_dir),
            executor_role="human",
            extra={"group_id": "g1"},
        )
        document = make_task_file(
            kind="human_decision_judge", tasks=[task], inputs=task["inputs"],
        )
        path = env.phase_dir / "human_decision_tasks.json"
        write_task_file(str(path), document)
        report_path = executor.dry_run_report(
            str(path), phase_dir=str(env.phase_dir), phase="apply-suggestions",
        )
        entry = json.loads(Path(report_path).read_text())["tasks"][0]
        assert entry["would_dispatch"] is False
        assert "executor_role: human" in entry["reason"]
