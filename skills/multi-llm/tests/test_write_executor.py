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

from utils import metrics  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import restore_batch  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_safety  # noqa: E402
from utils.providers import opencode as oc  # noqa: E402
from utils.providers.base import PROFILE_AGENT_WRITE  # noqa: E402
from utils.providers.claude_code import (  # noqa: E402
    JSON_SCHEMA_FLAG,
    reset_flag_gate_cache,
)
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
    SchedulerResult,
    WriteBatchScheduler,
    resolve_human_task_plan,
)

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"
SCENARIO_MATCHER = Path(__file__).parent / "scenario_matcher.py"

#: The three adapters that ship a real `agent-write` invocation. Parametrized over
#: so a profile that silently stops being edit-capable fails here rather than on a
#: user's repository.
WRITE_RUNNERS = [
    ("codex", "codex:gpt-5.5"),
    ("opencode", "opencode:opencode/big-pickle"),
    ("claude", "claude-code:sonnet"),
]

#: W2 — the rows ``--dry-run-subagents``' renderer and ``dispatch_task``'s
#: renderer must agree on, each with the flag whose absence from BOTH would make
#: the comparison vacuous.
#:
#: **Both** ``SCHEMA_PHASE_BY_KIND`` kinds on claude-code, never one: a preview
#: that hard-codes a schema instead of resolving the phase from the kind passes
#: one row and fails the other, and hard-coding is exactly the shape this must
#: not permit. The opencode row is M4's half of the same defect — ``--pure`` and
#: the ``XDG_CONFIG_HOME`` redirect are emitted only inside the dispatch context,
#: deliberately and together, so a preview rendered outside one shows an argv no
#: opencode run ever uses.
RENDERER_ROWS = [
    ("task_generation", "claude-code:sonnet", JSON_SCHEMA_FLAG),
    ("consolidation", "claude-code:sonnet", JSON_SCHEMA_FLAG),
    ("validation", "opencode:opencode/big-pickle", oc.PURE_FLAG),
]

#: ``(task_id, output_name)`` per kind in :data:`RENDERER_ROWS`.
_RENDERER_TASK_NAMES = {
    "task_generation": ("task-generation", "task_generation_result.json"),
    "consolidation": ("consolidation-batch-0", "consolidation_batch_0.json"),
    "validation": ("validation-batch-0", "validation_batch_0.json"),
}


def _option_flags(argv):
    """The option tokens in ``argv``, in order.

    The bare ``--`` is excluded on purpose: it is the prompt separator (§9.8),
    so it travels with the trailing positional and the preview renders the argv
    transport while a claude-code dispatch renders the stdin one. W2 is about
    the *flag set*, and the transport is not what it changes.
    """
    return [
        token for token in argv
        if isinstance(token, str) and token.startswith("--") and token != "--"
    ]


def _hermetic_dirs():
    """opencode's run-scoped config homes currently on disk, by name."""
    import tempfile

    root = Path(tempfile.gettempdir())
    return {str(p) for p in root.glob(f"{oc.HERMETIC_DIR_PREFIX}*")}


class _RecordingPopen:
    """A child that records its argv and answers nothing.

    The dispatch half of the W2 comparison needs the argv ``_spawn_in_context``
    really built — inside the adapter's dispatch context and with the schema the
    kind resolved — and needs no output at all, because the flags are the
    subject and the outcome is not. ``pid = -1`` so the timeout path's
    process-group signal can never reach a real process.
    """

    calls = []

    def __init__(self, args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        self.pid = -1
        self.returncode = None
        self.stdin = self.stdout = self.stderr = None
        type(self).calls.append(list(args))

    def communicate(self, input=None, timeout=None):
        self.returncode = 0
        return ("", "")

    def kill(self):  # pragma: no cover - only the teardown fallback calls this
        self.returncode = -9


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _install_launcher(bin_dir: Path, name: str) -> Path:
    """Put a mock provider CLI called ``name`` on a temp PATH.

    A *copied* launcher cannot find ``tests/scenario_matcher.py`` through its own
    ``__file__`` the way the real file and a symlink to it can, so the one
    definition of what a scenario means is copied beside it.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / name
    if os.name == "nt":  # pragma: no cover - POSIX CI
        target = bin_dir / f"{name}.py"
        shutil.copyfile(MOCK_LLM, target)
        shutil.copyfile(SCENARIO_MATCHER, bin_dir / SCENARIO_MATCHER.name)
    else:
        try:
            target.symlink_to(MOCK_LLM)
        except (OSError, NotImplementedError):  # pragma: no cover
            shutil.copyfile(MOCK_LLM, target)
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
            shutil.copyfile(SCENARIO_MATCHER, bin_dir / SCENARIO_MATCHER.name)
    return target


def _install_crashing_launcher(bin_dir: Path, name: str, code: int = 17) -> Path:
    """A launcher that answers normally and *then* dies with a non-zero exit.

    The shape D3k's write rule exists for: the runner printed its optimistic
    result JSON, kept editing, and was killed part way. Version and auth probes
    pass through untouched, or the runner would be refused before dispatch and
    the classification under test would never run.
    """
    target = bin_dir / (f"{name}.py" if os.name == "nt" else name)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys\n"
        "argv = sys.argv[1:]\n"
        "env = dict(os.environ, MOCK_LLM_PROVIDER=%r)\n"
        "rc = subprocess.run([sys.executable, %r] + argv, env=env).returncode\n"
        "probe = any(a in ('--version', '-v', 'login', 'status', 'sandbox') for a in argv)\n"
        "sys.exit(rc if probe or rc else %d)\n"
        % ("claude-code" if name == "claude" else name, str(MOCK_LLM), code),
        encoding="utf-8",
    )
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
        self.state = {"metrics": {}}
        self._path = path
        self.save()

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value

    def save(self):
        """A **real** write, because ``pass`` is what hid F3/F4 from CI.

        `_finalize` interleaves this object's saves with `finalize_tracking`'s
        own file-level write of the same file; with a no-op `save()` the clobber
        was unobservable, and the metrics assertion below passed against a
        second writer's file that production's next `save()` was erasing."""
        self._path.write_text(
            json.dumps(self.state, indent=2), encoding="utf-8"
        )

    def reload(self):
        self.state = json.loads(self._path.read_text(encoding="utf-8"))

    def start_metrics_phase(self, phase, total_batches):
        metrics.start_phase_in_state(self.state, phase, total_batches)

    def finish_metrics_phase(self, phase):
        return metrics.finish_phase_in_state(self.state, phase)

    def record_metric(self, phase, label, **kwargs):
        return metrics.record_metric_in_state(self.state, phase, label, **kwargs)

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

    def refresh_plan_hash(self):
        """F5: `_finalize` calls this beside `mark_phase_completed`. Recorded
        rather than computed for real, since this fake carries no plan hash of
        its own — see ``TestScheduler`` for the real-``StateManager`` version."""
        self.state["plan_hash_refreshed"] = True


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
        env.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["s0"]))
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

    def test_partial_application_is_also_a_non_zero_exit(self, env):
        """D3k inverts the read rule for writes: a crash is never an application.

        The runner prints a schema-valid ``applied`` naming every expected id and
        *then* dies mid-edit. Claim and diff alone read that as a full
        application, so the loop would dispatch batch N+1 on top of half-applied
        bytes; the exit code is what makes it a partial application.
        """
        _install_crashing_launcher(env.bin_dir, "codex")
        executor.reset_version_cache()
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "did it"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert outcome.status == executor.STATUS_PARTIAL
        assert outcome.stops_the_loop
        assert "exited 17" in outcome.reason
        assert result.exit_code == 3

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


class TestVerificationEvidenceOnEveryRow:
    """F9: `expected_ids`/`returned_ids`/`sha256_before`/`sha256_after` land on
    the outcome itself, so both reports built from it — `write_batch_report.json`
    (`SchedulerResult.to_dict`) and this run's own `subagent_batch_report.json`
    (`BatchReport`, same `TaskOutcome` list) — can answer "did every item come
    back?" and "is the hash chain contiguous?" without re-deriving it from
    `subagent_audit.jsonl` or the per-batch result file.
    """

    def test_an_applied_batch_carries_ids_and_hashes_on_the_outcome(self, env):
        before = write_safety.file_sha256(str(env.plan))
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "did it"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        after = write_safety.file_sha256(str(env.plan))

        assert outcome.expected_ids == ["s0"]
        assert outcome.returned_ids == ["s0"]
        assert outcome.sha256_before["plans/p.md"] == before
        assert outcome.sha256_after["plans/p.md"] == after
        assert before != after

    def test_a_partial_application_still_names_the_ids_it_did_not_get_back(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "half"})
        task = env.apply_task(0, expected_ids=["s0", "s1", "s2"])
        result = env.scheduler([task]).run()
        outcome = result.outcomes[0]
        assert outcome.expected_ids == ["s0", "s1", "s2"]
        assert outcome.returned_ids == ["s0"]

    def test_a_no_op_batch_still_carries_the_unchanged_hash(self, env):
        before = write_safety.file_sha256(str(env.plan))
        env.edits([], result={"status": "no-op", "ids": ["s0"],
                              "files_touched": [], "summary": "already there"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_NO_OP
        assert outcome.sha256_before["plans/p.md"] == before
        assert outcome.sha256_after["plans/p.md"] == before

    def test_both_reports_built_from_the_same_outcome_carry_the_same_evidence(
        self, env
    ):
        """`write_batch_report.json` and `subagent_batch_report.json` disagreed
        about this before F9 — not in which outcome they recorded, but in what
        each row could answer about it.
        """
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "did it"})
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.report_path is not None

        write_report = json.loads(Path(result.report_path).read_text())
        executor_report = json.loads(
            (env.phase_dir / executor.BATCH_REPORT_FILENAME).read_text()
        )
        for document in (write_report, executor_report):
            row = document["tasks"][0]
            assert row["expected_ids"] == ["s0"]
            assert row["returned_ids"] == ["s0"]
            assert row["sha256_before"]["plans/p.md"]
            assert row["sha256_after"]["plans/p.md"]
            assert row["sha256_before"] != row["sha256_after"]

    def test_a_read_kind_row_carries_none_of_this(self, tmp_path):
        """The fields are write-only: a read/analyze row has no `write_outcome`
        and stays exactly as it was — no `expected_ids`/`sha256_*` key at all.
        """
        outcome = executor.TaskOutcome(
            task_id="validation-batch-0", kind="validation", runner="codex:gpt-5.5",
            status=executor.STATUS_OK,
        )
        row = outcome.to_dict()
        assert "expected_ids" not in row
        assert "returned_ids" not in row
        assert "sha256_before" not in row
        assert "sha256_after" not in row


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

    def test_an_implement_prompt_without_the_token_fails_the_dispatch(self, env):
        """F6: implement prompts used to be exempted from the token contract
        (their inter-task context was the dependency section alone) and a
        missing token silently passed through as a no-op. Since
        ``prompts/implementation_task.txt`` now carries the token too, a
        producer that drops it fails the same way an apply producer's would."""
        task = env.apply_task(
            0, kind="implement", task_id="T001", expected_ids=["T001"],
            output_name="implement_batch_0.json",
            prompt="Implement T001, no token here.\n",
        )
        result = env.scheduler([task], kind="implement", finalize=False).run()
        assert result.outcomes[0].status == executor.STATUS_REJECTED
        assert PRIOR_CHANGES_TOKEN in result.outcomes[0].reason

    def test_an_implement_batch_sees_the_prior_task_summary(self, env):
        """F6: implement's inter-task context used to be the dependency
        section's bare titles alone. With the token bound, a later task's
        dispatched prompt now also carries the verified one-line summary
        `prior_changes.jsonl` already records for each finished task."""
        log = env.enable_call_log()

        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["T001"],
                          "files_touched": ["plans/p.md"],
                          "summary": "T001-DONE-SUMMARY"})
        first = env.apply_task(
            0, kind="implement", task_id="T001", expected_ids=["T001"],
            output_name="implement_batch_0.json",
        )
        env.scheduler([first], kind="implement", finalize=False).run()

        env.edits([_patch_edit(env.plan, "edit-1\n")],
                  result={"status": "applied", "ids": ["T002"],
                          "files_touched": ["plans/p.md"],
                          "summary": "T002-DONE-SUMMARY"})
        second = env.apply_task(
            1, kind="implement", task_id="T002", expected_ids=["T002"],
            output_name="implement_batch_1.json", depends_on=["T001"],
        )
        result = env.scheduler([second], kind="implement", finalize=False).run()
        assert result.exit_code == 0, result.messages

        dispatched = [json.loads(l) for l in log.read_text().splitlines() if l]
        second_prompt = [
            r["prompt"] for r in dispatched
            if "batch 1" in (r.get("prompt") or "")
        ][-1]
        assert "T001-DONE-SUMMARY" in second_prompt
        assert PRIOR_CHANGES_TOKEN not in second_prompt
        # The materialized task JSON is never rewritten.
        assert PRIOR_CHANGES_TOKEN in second["prompt"]

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

    def test_finalize_re_stamps_plan_hash_so_the_next_session_sees_no_change(
        self, env
    ):
        """F5: apply-suggestions' whole job is rewriting the plan, so without a
        re-stamp the next `StateManager` built on it reads this run's own edit
        as a human editing the plan mid-workflow and sets `plan_changed`."""
        from utils.state_manager import StateManager

        state = StateManager(env.plan)
        state.save()
        assert state.has_plan_changed() is False

        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        result = env.scheduler([env.apply_task(0)], state=state).run()
        assert result.exit_code == 0
        assert result.finalized is True
        # The plan really changed on disk — the point of the phase.
        assert "edit\n" in env.plan.read_text()

        # The stamp `_finalize` wrote matches what is actually on disk now...
        assert state.get("plan_hash") == state._compute_plan_hash()
        assert "plan_changed" not in state.state

        # ...and a FRESH session on this plan does not mistake the edit for a
        # human's.
        reloaded = StateManager(env.plan)
        assert reloaded.has_plan_changed() is False

    def test_the_batch_report_records_the_runners_version(self, env):
        """F12: the write path preflights inline, per dispatch, and its
        version was being discarded — the read path's `run_tasks` records the
        same information via `_preflight_batch_runners`. `apply-suggestions`'s
        `subagent_batch_report.json` had `"runner_versions": {}` while the log
        beside it warned about that exact runner's version."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.exit_code == 0
        batch_report = json.loads(
            (env.phase_dir / "subagent_batch_report.json").read_text()
        )
        assert batch_report["runner_versions"] == {"codex:gpt-5.5": "0.146.0"}

    def test_the_batch_report_names_write_tasks_json(self, env):
        """N9: the write path never takes a `task_file` argument the way
        `run_task_file` does on the read path — `WriteBatchScheduler` is
        constructed from an already-materialized task list — but every write
        orchestrator persists that same list via `write_task_file_for_phase`
        at `{phase_dir}/write_tasks.json` before building the scheduler, so
        `subagent_batch_report.json` should name it rather than `null`, the
        one field the report otherwise got right (ceilings, runner_versions,
        counts, usage all resolved)."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.exit_code == 0
        batch_report = json.loads(
            (env.phase_dir / "subagent_batch_report.json").read_text()
        )
        assert batch_report["task_file"] == str(env.phase_dir / "write_tasks.json")

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

    def test_resume_refuses_a_result_file_that_attests_to_another_task(self, env):
        """The file exists, parses, validates and says `applied` — for someone else.

        A resumed implement phase that re-materializes a SHORTER task list used to
        re-point each survivor at a predecessor's result file; the survivor had no
        recorded fingerprint (it never finished), so every earlier clause passed
        and five never-dispatched tasks were declared complete (2026-09-16). The
        file's own `ids` were the evidence, and nothing read them.
        """
        state = FakeState(env.tmp_path / "state.json")
        done = env.apply_task(0, kind="implement", task_id="T001",
                              expected_ids=["T001"])
        env.edits([_patch_edit(env.plan, "ok-0\n")],
                  result={"status": "applied", "ids": ["T001"],
                          "files_touched": ["plans/p.md"], "summary": "first"})
        env.scheduler([done], state=state, kind="implement",
                      finalize=False).run()
        assert (env.phase_dir / "implement_batch_T001.json").is_file()

        # The corruption, written out by hand: a never-run task pointed at the
        # finished task's result file, exactly as the old counter named it.
        stolen = env.apply_task(1, kind="implement", task_id="T006",
                                expected_ids=["T006"],
                                output_name="implement_batch_T001.json")
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan, "ok-6\n")],
                  result={"status": "applied", "ids": ["T006"],
                          "files_touched": ["plans/p.md"], "summary": "sixth"})
        messages = []
        result = env.scheduler(
            [stolen], state=state, kind="implement", resume=True,
            finalize=False, log=messages.append,
        ).run()

        assert result.exit_code == 0, result.messages
        assert [o.skipped for o in result.outcomes] == [False], (
            "a result attesting to T001 is not T006's completed work"
        )
        assert any("another task's result" in line for line in messages)
        assert [
            json.loads(line)["prompt"]
            for line in log.read_text().splitlines() if line
        ], "the re-dispatch really started a runner"

    def test_a_resume_skip_leaves_a_line_in_the_audit_log(self, env):
        """The audit log is the record of what the phase did — skips included."""
        state = FakeState(env.tmp_path / "state.json")
        tasks = self._three(env)
        env.edits([_patch_edit(env.plan, "ok-0\n")],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "first"})
        env.scheduler([tasks[0]], state=state, finalize=False).run()

        env.edits([_patch_edit(env.plan, "ok-rest\n")],
                  result={"status": "applied", "ids": ["s1"],
                          "files_touched": ["plans/p.md"], "summary": "rest"})
        env.scheduler(tasks[:2], state=state, resume=True, finalize=False).run()

        records = write_safety.AuditLog(str(env.phase_dir)).records()
        skips = [r for r in records if r["event"] == "skipped"]
        assert [r["task_id"] for r in skips] == ["apply-batch-0"]
        assert skips[0]["result_ids"] == ["s0"], "name what the skip trusted"
        assert skips[0]["output_path"].endswith("apply_batch_0.json")
        assert skips[0]["files_touched"] == ["plans/p.md"]

    def test_a_carried_human_task_keeps_the_deferred_section_on_a_resume(self, env):
        """What the resumed materialization's human task buys downstream.

        `format_batches_for_output(skip_completed=True)` now keeps a deferred
        human task; this is the consequence — `skipped_tasks` in the report and
        the `## Manual Tasks (Deferred)` section in the summary, which a resumed
        run lost entirely when the task was filtered out (2026-09-16).
        """
        state = FakeState(env.tmp_path / "state.json")
        summary = env.out_dir / "implement" / "implementation_summary.md"
        env.edits([_patch_edit(env.plan, "ok\n")],
                  result={"status": "applied", "ids": ["T002"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        tasks = [
            env.apply_task(0, kind="implement", task_id="T002",
                           expected_ids=["T002"]),
            env.apply_task(1, kind="implement", task_id="T014",
                           expected_ids=["T014"], executor_role="human"),
        ]
        result = env.scheduler(
            tasks, state=state, kind="implement", resume=True,
            summary_file=str(summary),
            human_task_strategy=STRATEGY_SKIP_DEPENDENTS,
        ).run()

        assert result.exit_code == 0, result.messages
        assert list(result.to_dict()["skipped_tasks"]) == ["T014"]
        text = Path(result.summary_file).read_text()
        assert "## Manual Tasks (Deferred)" in text
        assert "| T014 |" in text

    def test_a_stale_result_file_from_an_earlier_naming_is_named_not_deleted(
        self, env
    ):
        """N6: the F1 rename left old positional `implement_batch_N.json`
        files beside the new task-id-keyed ones. Nothing reads them any more,
        but a resumed run should say so once rather than leave a human to
        notice thirteen result files for five batches on their own."""
        (env.phase_dir / "implement_batch_0.json").write_text("{}", encoding="utf-8")
        (env.phase_dir / "implement_batch_5.json").write_text("{}", encoding="utf-8")

        env.edits([_patch_edit(env.plan, "ok\n")],
                  result={"status": "applied", "ids": ["T006"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        messages = []
        task = env.apply_task(0, kind="implement", task_id="T006",
                              expected_ids=["T006"])
        result = env.scheduler(
            [task], kind="implement", resume=True, finalize=False,
            log=messages.append,
        ).run()
        assert result.exit_code == 0, result.messages

        stale_lines = [
            m for m in messages
            if "stale result files from an earlier naming" in m
        ]
        assert len(stale_lines) == 1, "one warning, not one per file"
        assert "implement_batch_0.json" in stale_lines[0]
        assert "implement_batch_5.json" in stale_lines[0]
        # The task's OWN result file must never be named as stale.
        assert "implement_batch_T006.json" not in stale_lines[0]
        # Advisory only: both old files are still on disk afterwards.
        assert (env.phase_dir / "implement_batch_0.json").is_file()
        assert (env.phase_dir / "implement_batch_5.json").is_file()

    def test_no_stale_warning_for_a_positional_write_kind(self, env):
        """Scoped to `implement`: apply-suggestions' filenames have never been
        renamed, and with no id to key on, flagging an earlier run's already-
        completed batch would be noise rather than a finding."""
        (env.phase_dir / "apply_batch_0.json").write_text("{}", encoding="utf-8")
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s1"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        messages = []
        result = env.scheduler(
            [env.apply_task(1)], resume=True, finalize=False,
            log=messages.append,
        ).run()
        assert result.exit_code == 0, result.messages
        assert not any(
            "stale result files" in m for m in messages
        )


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
        # Every write kind's prompt carries the token, implement included since
        # F6 (2026-09-16) — it no longer gets the no-token exemption.
        artifact = env.root / self.KIND_ARTIFACT[kind]
        prompt = f"Do the {kind} work on {artifact}\n\n{PRIOR_CHANGES_TOKEN}\n"
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

    def test_status_json_carries_item_counts_distinct_from_batch_counts(
        self, env
    ):
        """F9: `items_total`/`items_applied` count the suggestions inside a
        batch, not the batches themselves — a batch of three applied
        suggestions is one *batch* but three *items*, and the two counts must
        not collapse into one number the way `total`/`dispatched` do."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0", "s1", "s2"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        # One batch, three expected ids — batches and items diverge on purpose.
        task = env.apply_task(0, expected_ids=["s0", "s1", "s2"])
        env.scheduler([task]).run()
        status = json.loads((env.phase_dir / ".status.json").read_text())
        progress = status["progress"]["write_subagents"]
        assert progress["total"] == 1  # one batch
        assert progress["items_total"] == 3  # three suggestions in it
        assert progress["items_applied"] == 3

    def test_a_no_op_batch_counts_zero_items_applied_though_it_is_ok(self, env):
        """A `no-op` outcome is `status: ok` (the loop continues past it) but
        it applied nothing — `items_applied` must not credit its
        `expected_ids` the way a naive "every ok batch" count would."""
        env.edits([], result={"status": "no-op", "ids": ["s0"],
                              "files_touched": [], "summary": "already present"})
        env.scheduler([env.apply_task(0)]).run()
        status = json.loads((env.phase_dir / ".status.json").read_text())
        progress = status["progress"]["write_subagents"]
        assert progress["items_total"] == 1
        assert progress["items_applied"] == 0

    def test_the_step_moves_with_the_write_phase_instead_of_freezing(self, env):
        """F4: `progress.step` is set before a dispatch and never replaced, so a
        write phase preceded by judging printed `dispatching human-decision
        judging` for its whole run and in the terminal payload, next to
        `run.state: done`."""
        from utils import status_file

        status_file.update_status(
            str(env.phase_dir),
            {"progress": {"step": "dispatching human-decision judging"}},
        )
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        env.scheduler([env.apply_task(0)]).run()
        progress = json.loads(
            (env.phase_dir / ".status.json").read_text()
        )["progress"]
        assert progress["step"] == "complete"
        # Every sibling counter survives the key-by-key merge.
        assert progress["write_subagents"]["total"] == 1

    def test_the_step_reads_applying_before_the_first_batch_lands(
        self, env, monkeypatch
    ):
        """The mirror runs after a dispatch returns, so a long first batch would
        otherwise keep the stale step on screen for its whole duration."""
        from utils import status_file

        steps = []
        real_update = status_file.update_status

        def record(phase_dir, updates):
            step = (updates.get("progress") or {}).get("step")
            if step:
                steps.append(step)
            real_update(phase_dir, updates)

        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        monkeypatch.setattr(status_file, "update_status", record)
        env.scheduler([env.apply_task(0)]).run()
        assert steps[0] == "applying"
        assert steps[-1] == "complete"

    def test_a_stopped_loop_says_stopped_rather_than_the_previous_step(self, env):
        env.edits([], result={"status": "failed", "ids": [],
                              "files_touched": [], "summary": "no"})
        env.scheduler([env.apply_task(0)]).run()
        progress = json.loads(
            (env.phase_dir / ".status.json").read_text()
        )["progress"]
        assert progress["step"] == "stopped"

    def test_the_write_mirror_moves_once_per_task_not_once_per_batch(
        self, env, monkeypatch
    ):
        """D4a: the mirror sat below the dispatch loop, so a batch carrying more
        than one task reported nothing until its last task landed."""
        from utils import status_file

        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0", "s1"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        # Both in batch 0: one dispatch loop, two tasks.
        tasks = [env.apply_task(0), env.apply_task(1, batch_index=0)]
        rows = []
        real_update = status_file.update_status

        def record(phase_dir, updates):
            progress = updates.get("progress") or {}
            mirror = progress.get("write_subagents")
            # F10: `_begin_sweep` also patches `write_subagents` (its
            # `in_flight` entry), one write per dispatch START rather than
            # per task completion — real, but a different advance than the
            # one this test pins. Only a full `_mirror_status` write carries
            # `step`, so that is what distinguishes the two here.
            if "step" in progress and isinstance(mirror, dict):
                rows.append(len(mirror["tasks"]))
            real_update(phase_dir, updates)

        monkeypatch.setattr(status_file, "update_status", record)
        env.scheduler(tasks).run()
        # Row 0 is the pre-dispatch mirror that plants `progress.step:
        # applying` (F4); the per-task advance is what follows it.
        assert rows[:3] == [0, 1, 2]
        progress = json.loads(
            (env.phase_dir / ".status.json").read_text()
        )["progress"]["write_subagents"]
        assert progress["total"] == 2
        # The end state is what it was: one row per task, counted once.
        assert [t["task_id"] for t in progress["tasks"]] == [
            "apply-batch-0", "apply-batch-1",
        ]

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

    # -- W2: the preview and the dispatch are one renderer, not two --------

    def _renderer_task(self, env, kind: str):
        """One task of ``kind``, shaped the way its own phase materializes it."""
        from utils.subagent_tasks import make_task

        task_id, output_name = _RENDERER_TASK_NAMES[kind]
        extra, context_mode = {}, "inline"
        if kind in ("validation", "consolidation"):
            extra = {"batch_index": 0, "group_indices": [0],
                     "group_ids": ["aaaa000000000000"], "groups_count": 1}
        if kind == "consolidation":
            extra["section_key"] = "s"
        if kind == "task_generation":
            # The one read kind whose context is the repository (D3e).
            context_mode, extra = "workspace-read", {"repo_root": str(env.root)}
        return make_task(
            kind=kind,
            task_id=task_id,
            prompt="Do the thing and print ONE JSON object and nothing else.",
            output_name=output_name,
            inputs=build_inputs(grouped_file=str(env.phase_dir / "grouped.json")),
            phase_dir=str(env.phase_dir),
            context_mode=context_mode,
            extra=extra,
        )

    @pytest.mark.parametrize("kind,runner,required_flag", RENDERER_ROWS,
                             ids=[f"{k}-{r.split(':')[0]}" for k, r, _f in RENDERER_ROWS])
    def test_the_preview_renders_the_flags_the_dispatch_renders(
        self, env, monkeypatch, kind, runner, required_flag
    ):
        """``--dry-run-subagents``' argv is the argv the real dispatch builds.

        Compared, not pinned twice: the preview's flags come out of
        ``dry_run_report``'s own report file and the dispatch's out of the argv a
        really-dispatched task handed ``Popen``, so this cannot pass on a preview
        that hard-codes what a dispatch happens to render today. ``required_flag``
        is what stops it passing *vacuously* — both renderers dropping the same
        flag is agreement, and it is not the agreement this asserts.

        The rows are the whole reason this is parametrized. Both
        ``SCHEMA_PHASE_BY_KIND`` kinds on claude-code, because one alone would
        pass on a preview that hard-codes a schema instead of resolving the phase
        from the kind; and opencode, whose hermetic ``--pure`` renders only
        inside the M4 dispatch context — so a preview that renders outside it
        shows argv no run of that adapter ever uses.
        """
        from utils.subagent_tasks import make_task_file, write_task_file

        env.use(runner)
        reset_flag_gate_cache()
        # `--pure` renders only when `opencode run --help` documents it; the mock
        # CLI's help does not, so pin the sniff rather than the adapter's rule.
        monkeypatch.setattr(oc, "_read_help_tokens",
                            lambda binary: frozenset({oc.PURE_FLAG}))
        # Deterministic: `_user_defines_provider` reads the *inherited* config
        # home, and a developer's own opencode config would decide the refusal.
        monkeypatch.setenv("XDG_CONFIG_HOME", str(env.tmp_path / "xdg-config"))
        oc.sweep_config_home()
        oc.reset_hermetic_state()
        hermetic_before = _hermetic_dirs()
        task = self._renderer_task(env, kind)

        # (1) the preview renderer, through its own report file.
        document = make_task_file(kind=kind, tasks=[task], inputs=task["inputs"])
        task_path = env.phase_dir / f"{kind}_tasks.json"
        write_task_file(str(task_path), document)
        report_path = executor.dry_run_report(
            str(task_path), phase_dir=str(env.phase_dir), phase="preview",
            cwd=str(env.root),
        )
        entry = json.loads(Path(report_path).read_text())["tasks"][0]
        assert entry["would_dispatch"] is True, entry.get("reason")
        preview = _option_flags(entry["argv"])

        # A dry run renders commands and starts no child, so the shared hermetic
        # config home it claimed must be released the way a dispatch releases it.
        assert _hermetic_dirs() == hermetic_before

        # (2) the dispatch renderer, through the argv a real dispatch spawned.
        _RecordingPopen.calls = []
        monkeypatch.setattr(executor.subprocess, "Popen", _RecordingPopen)
        # 1800s is not a token budget: §7.5 rule 4 starts no attempt that cannot
        # hold one worst-case run of the adapter's DECLARED envelope, and
        # claude-code declares 1200s, so a small budget here would record
        # `budget_exhausted` with nothing spawned and no argv to compare.
        # Nothing sleeps — the canned child returns immediately.
        outcome = executor.dispatch_task(
            task, phase_dir=str(env.phase_dir), runner=runner, retries=0,
            timeout=1800, cwd=str(env.root), sleep=lambda _seconds: None,
        )
        assert _RecordingPopen.calls, (
            f"nothing was spawned: {outcome.status} — {outcome.reason}"
        )
        dispatched = _option_flags(_RecordingPopen.calls[-1])

        assert required_flag in dispatched, (
            f"the {runner} dispatch stopped rendering {required_flag}, so this "
            f"comparison would pass on two renderers that both drop it"
        )
        assert preview == dispatched, (
            f"--dry-run-subagents previews {preview} for a {kind} task on "
            f"{runner}, but the dispatch renders {dispatched}"
        )
        oc.sweep_config_home()
        oc.reset_hermetic_state()


# ===========================================================================
# The result contract the runner is shown, and the rollback point a resume keeps
# ===========================================================================

class TestResultContract:
    """Every dispatched write prompt ends with the JSON shape the verdict reads.

    The batchers' prompts end in a prose "Return Format" the native markdown
    flow compresses into a summary line; the scheduler validates the reply
    against ``write_batch_result.schema.json`` and reads ``ids`` against
    ``expected_ids``. Measured 2026-09-15 on ``apply-task-suggestions`` against
    a real runner: it applied the batch, reported it exactly as the prompt
    asked, and was recorded ``unparseable_result`` four attempts in a row.
    """

    def _dispatched(self, log: Path) -> str:
        lines = [l for l in log.read_text().splitlines() if l]
        return json.loads(lines[-1])["prompt"]

    def test_the_contract_names_every_expected_id_in_prompt_order(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0", "s1"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        task = env.apply_task(0, expected_ids=["s0", "s1"])
        result = env.scheduler([task]).run()
        assert result.exit_code == 0, result.messages
        prompt = self._dispatched(log)
        assert "## Result (required" in prompt
        assert '"status": "applied"' in prompt and "`no-op`" in prompt
        assert "suggestion 1 = `s0`, suggestion 2 = `s1`" in prompt
        assert "or found already applied by a prior batch" in prompt
        assert PRIOR_CHANGES_TOKEN not in prompt
        # The contract is bound at dispatch, like the prior-changes context: the
        # hash-bound task file never learns it.
        assert "## Result" not in task["prompt"]
        assert prompt.index("Apply batch 0") < prompt.index("## Result")

    def test_the_implement_kind_names_its_task_id(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["T001"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        task = env.apply_task(0, kind="implement", task_id="T001",
                              expected_ids=["T001"],
                              output_name="implement_batch_0.json")
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler([task], state=state, kind="implement",
                               finalize=False).run()
        assert result.exit_code == 0, result.messages
        prompt = self._dispatched(log)
        assert "this task's id, `T001`" in prompt
        assert "## Result (required" in prompt

    def test_the_native_invoker_is_shown_the_same_contract(self, env):
        seen = {}

        def respond(task, prompt):
            seen["prompt"] = prompt
            (env.plan).write_text(env.plan.read_text() + "native\n")
            return {"status": "applied", "ids": ["s0"],
                    "files_touched": ["plans/p.md"], "summary": "S"}

        result = env.scheduler([env.apply_task(0)], respond=respond).run()
        assert result.exit_code == 0, result.messages
        assert "## Result (required" in seen["prompt"]
        assert "suggestion 1 = `s0`" in seen["prompt"]


class TestSnapshotReuseOnResume:
    """A resumed batch keeps the rollback point from before its FIRST dispatch.

    A failed batch leaves its half-applied edits in place by design. Measured
    2026-09-15 across four resumed runs of one batch: each run re-snapshotted
    the previous run's partial edits as the "pre-batch" bytes, so ``util
    restore-batch`` could no longer reach the file as it was before the batch
    and the "pre-batch snapshots are intact" line was untrue.
    """

    def _fail_once(self, env) -> bytes:
        original = env.plan.read_bytes()
        # The runner edits, then narrates: rung 5, edit left on disk.
        env.edits([_patch_edit(env.plan, "half\n")], prose=True)
        first = env.scheduler([env.apply_task(0)]).run()
        assert first.exit_code == 3
        assert env.plan.read_bytes() != original
        return original

    def _snapshot(self, env):
        directory = env.phase_dir / "pre_batch" / "apply-batch-0"
        manifest = json.loads((directory / "manifest.json").read_text())
        payload = next((directory / "files").iterdir()).read_bytes()
        return manifest, payload

    def test_a_resumed_batch_keeps_the_original_rollback_point(self, env):
        original = self._fail_once(env)
        manifest_before, payload_before = self._snapshot(env)
        assert payload_before == original

        env.edits([_patch_edit(env.plan, "rest\n")],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "done"})
        second = env.scheduler([env.apply_task(0)]).run()  # resume is the default
        assert second.exit_code == 0, second.messages
        assert second.outcomes[0].write_outcome == executor.WRITE_APPLIED

        manifest_after, payload_after = self._snapshot(env)
        assert manifest_after == manifest_before
        assert payload_after == original
        dispatched = [
            r for r in write_safety.AuditLog(str(env.phase_dir)).records()
            if r["event"] == "dispatched"
        ]
        # The prose failure is retried inside the first run, so it records more
        # than one dispatch; every one of them, and the resumed run's, names
        # the same pre-batch hash.
        assert len(dispatched) >= 2
        assert {json.dumps(r["snapshot_before"], sort_keys=True) for r in dispatched} \
            == {json.dumps(dispatched[0]["snapshot_before"], sort_keys=True)}
        # And the rollback command really lands on the pre-batch bytes.
        from utils import restore_batch
        restore_batch.restore(
            phase_dir=str(env.phase_dir), task_ids=["apply-batch-0"],
            force=True, dry_run=False,
        )
        assert env.plan.read_bytes() == original

    def test_a_fresh_run_retakes_the_snapshot(self, env):
        self._fail_once(env)
        env.edits([_patch_edit(env.plan, "rest\n")],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "done"})
        result = env.scheduler([env.apply_task(0)], resume=False).run()
        assert result.exit_code == 0, result.messages
        _manifest, payload = self._snapshot(env)
        assert b"half" in payload, "without resume the current bytes are the baseline"

    def test_a_changed_declared_set_takes_a_fresh_snapshot(self, env):
        self._fail_once(env)
        other = env.root / "plans" / "other.md"
        other.write_text("other\n", encoding="utf-8")
        env.edits([_patch_edit(env.plan, "rest\n")],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "done"})
        task = env.apply_task(0, declared_files=[str(env.plan), str(other)])
        result = env.scheduler([task]).run()
        assert result.exit_code == 0, result.messages
        manifest, _payload = self._snapshot(env)
        assert len(manifest["files"]) == 2


# ===========================================================================
# The empty-diff branch has three verdicts, not one (measured 2026-09-16)
# ===========================================================================

class TestTheEmptyDiffBranch:
    """`applied`, `no-op` and `failed` with nothing moved are three different
    things: a contradiction, an agreement, and a failure."""

    def test_a_no_op_claim_with_no_bytes_moved_is_a_no_op(self, env):
        env.edits([], result={"status": "no-op", "ids": ["s0"],
                              "files_touched": [],
                              "summary": "already present"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_NO_OP
        assert outcome.status == executor.STATUS_OK
        assert outcome.files_touched == []
        assert "NO-OP" in outcome.reason
        assert result.exit_code == 0
        # The prior-changes entry says "no-op", never "failed": the next batch
        # reads that the work was already there.
        entries = [json.loads(l) for l in
                   (env.out_dir / "prior_changes.jsonl").read_text().splitlines() if l]
        assert [e["summary"] for e in entries] == [
            "(no-op — already present, verified no file changes)"
        ]

    def test_an_applied_claim_with_no_bytes_moved_is_still_a_clean_failure(self, env):
        env.edits([], result={"status": "applied", "ids": ["s0"],
                              "files_touched": ["plans/p.md"],
                              "summary": "claimed to apply"})
        result = env.scheduler([env.apply_task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
        assert outcome.status == executor.STATUS_OK
        assert result.exit_code == 0
        entries = [json.loads(l) for l in
                   (env.out_dir / "prior_changes.jsonl").read_text().splitlines() if l]
        assert [e["summary"] for e in entries] == [
            "(batch failed — claimed to apply, verified no file changes)"
        ]

    def test_a_failed_claim_with_no_bytes_moved_stops_the_loop(self, env):
        """The runner said it could not do the work, and the bytes agree.
        Not `ok`, no later batch, exit 3 naming --resume."""
        env.edits([], result={"status": "failed", "ids": [],
                              "files_touched": [],
                              "summary": "could not find the section"})
        result = env.scheduler([env.apply_task(0), env.apply_task(1)]).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert not outcome.ok
        assert "FAILED" in outcome.reason
        assert "could not find the section" in outcome.reason
        assert result.exit_code == 3
        assert result.stopped_at_batch == 0
        assert 1 not in result.batches_dispatched
        assert not result.finalized
        joined = "\n".join(result.messages)
        assert "--resume" in joined
        progress = json.loads(
            (env.phase_dir / ".status.json").read_text()
        )["progress"]["write_subagents"]
        assert progress["failed_task_ids"] == ["apply-batch-0"]

    def test_resume_does_not_count_a_recorded_failure_as_done(self, env):
        """A schema-valid `{"status": "failed"}` result file on disk is the
        batch --resume exists for, not one it may skip."""
        env.edits([], result={"status": "failed", "ids": [],
                              "files_touched": [], "summary": "broke"})
        first = env.scheduler([env.apply_task(0)]).run()
        assert first.exit_code == 3
        assert (env.phase_dir / "apply_batch_0.json").is_file()

        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "now"})
        second = env.scheduler([env.apply_task(0)]).run()
        assert second.exit_code == 0, second.messages
        assert not second.outcomes[0].skipped
        assert second.outcomes[0].write_outcome == executor.WRITE_APPLIED
        assert log.read_text().strip(), "the batch was re-dispatched"

    def test_a_no_op_result_file_counts_as_done_on_resume(self, env):
        env.edits([], result={"status": "no-op", "ids": ["s0"],
                              "files_touched": [], "summary": "already there"})
        env.scheduler([env.apply_task(0)]).run()
        log = env.enable_call_log()
        second = env.scheduler([env.apply_task(0)]).run()
        assert second.exit_code == 0
        assert second.outcomes[0].skipped
        assert not log.exists() or not log.read_text().strip()


# ===========================================================================
# N2: a `no-op` claim with a NON-empty diff is still applied, but why the
# claim and the bytes disagree is not one thing (measured 2026-09-16)
# ===========================================================================

class TestANoOpClaimWithBytesMoved:
    """T006 (the audit's own case) claimed `no-op` while its pre-batch
    snapshot had been reused from an earlier, killed dispatch of the same
    batch — the diff was that attempt's work, not this one's, and `applied`
    is the right verdict. The DIFFERENT case — this dispatch's own fresh
    snapshot disagreeing with a `no-op` claim — is a foreign-write signature
    and should not read identically."""

    def test_a_fresh_snapshot_disagreeing_with_no_op_is_applied_and_warns(
        self, env
    ):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "no-op", "ids": ["s0"],
                          "files_touched": [], "summary": "already present"})
        lines = []
        result = env.scheduler([env.apply_task(0)], log=lines.append).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.status == executor.STATUS_OK
        assert result.exit_code == 0
        assert "no-op claimed, bytes moved" in outcome.reason
        assert "reused" not in outcome.reason
        warnings = [l for l in lines if "claimed 'no-op'" in l]
        assert len(warnings) == 1, lines
        assert "plans/p.md" in warnings[0]

    def test_a_reused_snapshot_disagreeing_with_no_op_is_applied_and_attributed(
        self, env
    ):
        """The audit's exact T006 shape: round 1 fails (rung 5) leaving its
        edit on disk and its pre-batch snapshot intact; round 2 makes no
        further edit and reports `no-op` — the diff round 2 sees is entirely
        round 1's, because round 2 reused round 1's snapshot rather than
        retaking one over round 1's half-applied bytes."""
        original = env.plan.read_bytes()
        env.edits([_patch_edit(env.plan, "half\n")], prose=True)
        first = env.scheduler([env.apply_task(0)]).run()
        assert first.exit_code == 3
        assert env.plan.read_bytes() != original

        env.edits([], result={"status": "no-op", "ids": ["s0"],
                              "files_touched": [], "summary": "already there"})
        lines = []
        second = env.scheduler(
            [env.apply_task(0)], log=lines.append
        ).run()  # resume is the default
        outcome = second.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.status == executor.STATUS_OK
        assert second.exit_code == 0
        assert "no-op claimed, bytes moved" in outcome.reason
        assert "reused from an earlier dispatch" in outcome.reason
        # This is the documented, expected shape of a resume — no foreign-
        # write warning belongs on the log for it.
        assert not any("claimed 'no-op'" in l for l in lines)


# ===========================================================================
# Finalization hands the verified touched set to finalize_tracking
# ===========================================================================

class TestFinalizeTrackingSeesTheVerifiedFiles:

    def test_every_applied_file_travels_as_touched(self, env, monkeypatch):
        import finalize_tracking

        seen = []
        monkeypatch.setattr(finalize_tracking, "main",
                            lambda argv: seen.append(list(argv)) or 0)
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler([env.apply_task(0)], state=state).run()
        assert result.exit_code == 0, result.messages
        assert len(seen) == 1
        argv = seen[0]
        assert argv[:2] == ["--state-file", state.state_file]
        touched = [argv[i + 1] for i, a in enumerate(argv) if a == "--touched"]
        assert touched == ["plans/p.md"]

    def test_a_batch_skipped_on_resume_still_contributes_its_files(self, env, monkeypatch):
        """The earlier run's result file records what it changed; a resumed
        run's finalization must not lose those files."""
        import finalize_tracking

        state = FakeState(env.tmp_path / "state.json")
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        env.scheduler([env.apply_task(0)], state=state, finalize=False).run()

        seen = []
        monkeypatch.setattr(finalize_tracking, "main",
                            lambda argv: seen.append(list(argv)) or 0)
        result = env.scheduler([env.apply_task(0)], state=state).run()
        assert result.outcomes[0].skipped
        assert result.outcomes[0].files_touched == ["plans/p.md"]
        touched = [seen[0][i + 1] for i, a in enumerate(seen[0]) if a == "--touched"]
        assert touched == ["plans/p.md"]

    def test_a_no_op_batch_contributes_nothing(self, env, monkeypatch):
        import finalize_tracking

        seen = []
        monkeypatch.setattr(finalize_tracking, "main",
                            lambda argv: seen.append(list(argv)) or 0)
        env.edits([], result={"status": "no-op", "ids": ["s0"],
                              "files_touched": [], "summary": "there"})
        state = FakeState(env.tmp_path / "state.json")
        env.scheduler([env.apply_task(0)], state=state).run()
        assert seen and "--touched" not in seen[0]

    def test_the_finalized_state_keeps_the_tracked_files_and_the_metrics(
        self, env, monkeypatch
    ):
        """F3/F4, against the REAL ``StateManager``: `_finalize` used to save,
        let `finalize_tracking` write `tracked_files` into the same file, then
        dump its pre-finalize dict over the top. An implement run whose log said
        "Tracked 1 implementation files: index.html" finalized with the two stale
        entries earlier phases had left, and the metric every batch recorded went
        with it (three audited runs ended `metrics: null`)."""
        import finalize_tracking
        from utils.state_manager import StateManager

        # The census is `git status` on the *test process's* cwd; the `--touched`
        # set is what this assertion is about, so pin the other half to empty.
        monkeypatch.setattr(finalize_tracking, "get_all_modified_files", set)
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        state = StateManager(env.plan)
        state.set("tracked_files", [
            {"path": "stale.md", "action": "modified", "task_id": "implementation"}
        ])
        state.save()

        summary = env.out_dir / "apply-suggestions" / "implementation_summary.md"
        result = env.scheduler(
            [env.apply_task(0)], state=state, summary_file=str(summary),
        ).run()
        assert result.exit_code == 0, result.messages
        assert result.finalized is True

        on_disk = json.loads(Path(state.state_file).read_text(encoding="utf-8"))
        assert [f["path"] for f in on_disk["tracked_files"]] == ["plans/p.md"]
        assert len(on_disk["metrics"]["apply-suggestions"]) == 1
        # ...and the dict this object holds agrees with the file it just wrote.
        assert state.get("tracked_files") == on_disk["tracked_files"]

    def test_the_scheduler_stamps_an_eta_baseline_before_the_first_dispatch(
        self, env
    ):
        """F4: only the `util metrics start` CLI leg ever called `start_phase`,
        so on the executor path `metrics_progress` — and the `[ETA]` line
        `instructions/implement.md` step 5 describes — never existed."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        state = FakeState(env.tmp_path / "state.json")
        lines = []
        result = env.scheduler(
            [env.apply_task(0)], state=state, finalize=False, log=lines.append,
        ).run()
        assert result.exit_code == 0, result.messages
        on_disk = json.loads(Path(state.state_file).read_text(encoding="utf-8"))
        assert on_disk["metrics_progress"]["apply-suggestions"]["total_batches"] == 1
        eta = [l for l in lines if l.startswith("[ETA] apply-suggestions:")]
        assert len(eta) == 1, lines
        assert "batch 1/1" in eta[0]

    def test_every_batch_leaves_one_line_in_the_phase_log(self, env):
        """F7: every `self.log` inside the loop was exceptional, so a 25-minute
        run that applied five batches left an 11-line log — and `status` points
        the user at that log."""
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        lines = []
        result = env.scheduler([env.apply_task(0)], log=lines.append).run()
        assert result.exit_code == 0, result.messages
        batch_lines = [l for l in lines if "batch 0: apply-batch-0" in l]
        assert len(batch_lines) == 1, lines
        assert "applied" in batch_lines[0]
        assert "plans/p.md" in batch_lines[0]

    def test_the_mirror_records_a_phase_state(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        env.scheduler([env.apply_task(0)]).run()
        status = json.loads((env.phase_dir / ".status.json").read_text())
        assert status["state"] == "complete"
        assert status["phase"] == "apply-suggestions"

        env.plan.write_text("# Plan\n", encoding="utf-8")
        shutil.rmtree(env.phase_dir / "pre_batch")
        (env.phase_dir / "apply_batch_0.json").unlink()
        env.edits([_patch_edit(env.plan)],
                  result={"status": "failed", "ids": [],
                          "files_touched": ["plans/p.md"], "summary": "broke"})
        env.scheduler([env.apply_task(0)]).run()
        status = json.loads((env.phase_dir / ".status.json").read_text())
        assert status["state"] == "subagents_failed"


# ===========================================================================
# N1: the `[ETA]` line is this run's own dispatch order and dispatchable
# count, never the plan's global decomposition index (measured 2026-09-16)
# ===========================================================================

class TestTheEtaLineUsesThisRunsOwnBatchCount:
    """A resumed run's materialized tasks carry `batch_index` wherever the
    plan's decomposition put them (5..9 here) and still include a deferred
    human task never dispatched — `_record_metrics`/`_metrics_start` used to
    pass both straight through, so every line read `batch 6/6` ...
    `batch 10/6 ... WARNING: batch_index > total_batches`."""

    def test_batch_and_total_are_this_runs_own_not_the_plans(self, env):
        state = FakeState(env.tmp_path / "state.json")
        tasks = [
            env.apply_task(
                i, kind="implement", task_id=f"T{6 + i:03d}",
                expected_ids=[f"T{6 + i:03d}"], batch_index=5 + i,
            )
            for i in range(5)
        ]
        tasks.append(env.apply_task(
            5, kind="implement", task_id="T014", executor_role="human",
            batch_index=13,
        ))
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied",
                          "ids": [f"T{6 + i:03d}" for i in range(5)],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        lines = []
        result = env.scheduler(
            tasks, state=state, kind="implement", resume=True,
            human_task_strategy=STRATEGY_SKIP_DEPENDENTS, finalize=False,
            log=lines.append,
        ).run()
        assert result.exit_code == 0, result.messages

        eta_lines = [l for l in lines if l.startswith("[ETA] implement:")]
        assert len(eta_lines) == 5, lines
        for position, line in enumerate(eta_lines, start=1):
            assert f"batch {position}/5" in line, line
            if position < 5:
                assert "ALL BATCHES COMPLETE" not in line, line
            else:
                assert "ALL BATCHES COMPLETE" in line, line
            assert "WARNING" not in line, line

        on_disk = json.loads(Path(state.state_file).read_text(encoding="utf-8"))
        # `_metrics_finish` (via `_finalize`, skipped here by `finalize=False`)
        # is what pops `metrics_progress`; with it off the baseline this run
        # started with is still on disk to assert against directly.
        assert on_disk["metrics_progress"]["implement"]["total_batches"] == 5


class TestTheRollbackStoreWarningNamesItsFiles:
    """A count alone left the reader guessing which files, and whether the gap
    was a problem (measured 2026-09-16: three under `.multi-llm/`, denied by
    design)."""

    def test_a_denied_file_is_named_with_its_reason(self, env):
        (env.root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        lines = []
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        result = env.scheduler([env.apply_task(0)], log=lines.append).run()
        assert result.exit_code == 0, result.messages
        warning = [l for l in lines if "no rollback point" in l]
        assert len(warning) == 1, lines
        assert ".env (denied by driver.write_paths)" in warning[0]
        assert "1 file(s)" in warning[0]

    def test_the_rollback_store_utilization_is_reported_once_per_run(self, env):
        """F15: the only visible signal toward ``SWEEP_RETAIN_MAX_BYTES`` was
        this same warning, for the files *already* denied a rollback point —
        nothing said how close a run came before that happened. One line,
        after the sweep is done, names the fraction of the budget retained."""
        import re

        lines = []
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        result = env.scheduler([env.apply_task(0)], log=lines.append).run()
        assert result.exit_code == 0, result.messages
        reported = [
            l for l in lines if l.startswith("[write-subagents] rollback store:")
        ]
        assert len(reported) == 1, lines
        match = re.search(
            r"rollback store: \d+\.\d MB of 64 MB budget retained "
            r"\((\d+) files, (\d+) shards\)",
            reported[0],
        )
        assert match, reported[0]
        # At least the plan itself was censused and retained.
        assert int(match.group(1)) >= 1
        assert int(match.group(2)) >= 1

    def test_no_rollback_utilization_line_when_nothing_was_ever_retained(
        self, env
    ):
        """``--dry-run-subagents`` censuses nothing, so there is no budget to
        report — the line would otherwise read "0.0 MB … (0 files, 0 shards)"
        on every dry run, which says nothing a reader needs."""
        lines = []
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        env.scheduler([env.apply_task(0)], dry_run=True, log=lines.append).run()
        assert not any(
            l.startswith("[write-subagents] rollback store:") for l in lines
        )


class TestTheRollbackStoreBudgetCountsBlobsAlreadyOnDisk:
    """F12: `_store_blob` returned early for an already-present
    content-addressed blob WITHOUT adding its size to `_retained_bytes`, so a
    run that inherited another run's store (e.g. one killed before its own
    `finally` tore it down) reported a fraction of the store's real size and
    the 64 MB cap went unenforced for the rest of that run."""

    def test_a_blob_already_on_disk_is_counted_once(self, env):
        relative = "src/a.py"
        absolute = env.root / relative
        info = absolute.stat()
        stamp = (info.st_size, info.st_mtime_ns)
        digest = write_safety.file_sha256(str(absolute))

        scheduler = env.scheduler([env.apply_task(0)])
        blob = scheduler._blob_path(digest)
        os.makedirs(os.path.dirname(blob), exist_ok=True)
        shutil.copy2(absolute, blob)
        assert scheduler._retained_bytes == 0

        returned = scheduler._store_blob(relative, stamp)
        assert returned == digest
        assert scheduler._retained_bytes == info.st_size

        # A second call for the same content (a re-hash after an edit then a
        # revert, or a second relative path with identical bytes) must not
        # double-count it — the store is content-addressed once.
        again = scheduler._store_blob(relative, stamp)
        assert again == digest
        assert scheduler._retained_bytes == info.st_size

    def test_a_fresh_copy_is_still_counted_exactly_once(self, env):
        """The companion case: nothing pre-exists, so `_store_blob` copies
        the bytes itself — this must still count, and only once."""
        relative = "src/b.py"
        absolute = env.root / relative
        info = absolute.stat()
        stamp = (info.st_size, info.st_mtime_ns)

        scheduler = env.scheduler([env.apply_task(0)])
        digest = scheduler._store_blob(relative, stamp)
        assert digest is not None
        assert scheduler._retained_bytes == info.st_size
        assert scheduler._store_blob(relative, stamp) == digest
        assert scheduler._retained_bytes == info.st_size

    def test_a_run_that_inherits_a_killed_runs_store_counts_its_bytes_too(
        self, env
    ):
        """End to end: a second scheduler pointed at a phase dir a killed
        run's store survived in retains its own file AND the inherited
        bytes, rather than reporting only the smaller, most-recent slice.

        ``run()``'s own ``finally`` tears the store down (and zeroes
        ``_retained_bytes`` with it) right after reporting the utilisation
        line, so ``_discard_rollback_store`` is stubbed out here to inspect
        the counter ``run()`` itself computed — the fixture's files are only
        tens of bytes, too small for the printed ``X.Y MB`` text to tell a
        few bytes counted from a few bytes silently dropped.
        """
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        first = env.scheduler([env.apply_task(0)])
        # A killed run never reaches `run()`'s `finally` (which tears the
        # store down) — simulate that by calling the retain step directly.
        first._retain(first._census())
        assert (env.phase_dir / "rollback_store").is_dir()
        inherited_bytes = first._retained_bytes
        assert inherited_bytes > 0

        lines = []
        second = env.scheduler(
            [env.apply_task(1, expected_ids=["s0"])], log=lines.append,
        )
        second._discard_rollback_store = lambda: None
        result = second.run()
        assert result.exit_code == 0, result.messages
        assert second._retained_bytes >= inherited_bytes
        reported = [
            l for l in lines if l.startswith("[write-subagents] rollback store:")
        ]
        assert len(reported) == 1, lines


class TestMirrorDenominatorsExcludeDeferredHumanTasks:
    """F9 (mirror half): `total`/`items_total` used to include the human
    task(s) `human_task_strategy` skips — they never dispatch, so a clean run
    that finished everything it could mirrored one short of its own task
    count (measured 2026-09-16: a 14-task implement run with one deferred
    manual task read `13/14 batches ok`)."""

    def test_a_skipped_human_task_does_not_count_toward_the_denominators(
        self, env
    ):
        from utils import status_file

        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s1"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        tasks = [
            env.apply_task(0, task_id="T-human", executor_role="human",
                           expected_ids=["T-human"]),
            env.apply_task(1, task_id="T-real", expected_ids=["s1"]),
        ]
        result = env.scheduler(
            tasks, human_task_strategy=STRATEGY_SKIP_DEPENDENTS,
        ).run()
        assert result.exit_code == 0, result.messages
        mirror = status_file.read_status(
            str(env.phase_dir)
        )["progress"]["write_subagents"]
        assert mirror["skipped_task_ids"] == ["T-human"]
        # One real task, and the mirror's own denominators say so — not two.
        assert mirror["total"] == 1
        assert mirror["items_total"] == 1
        assert mirror["items_applied"] == 1


class TestTheInFlightMirrorNamesADispatchInProgress:
    """F10: `.status.json`'s write-progress counter used to count FINISHED
    tasks, so a batch that runs for minutes read `dispatched: 0` for its
    whole duration — indistinguishable from a stall. `_begin_sweep` (the one
    place a dispatch's start time is known) now patches `in_flight` directly,
    and the classification step that follows it clears the entry again."""

    def test_a_dispatch_in_progress_is_named_then_cleared_on_completion(
        self, env
    ):
        from utils import status_file

        task = env.apply_task(0)
        scheduler = env.scheduler([task])
        result = SchedulerResult(kind=scheduler.kind, phase=scheduler.phase)
        # The pre-loop mirror `_run` writes before dispatching anything —
        # `_begin_sweep`'s patch preserves whatever this last wrote.
        scheduler._mirror_status(result)

        snapshot = scheduler._snapshot(task)
        scheduler._begin_sweep("apply-batch-0", ["plans/p.md"])

        mirror = status_file.read_status(
            str(env.phase_dir)
        )["progress"]["write_subagents"]
        assert [row["task_id"] for row in mirror["in_flight"]] == ["apply-batch-0"]
        assert mirror["in_flight"][0]["dispatched_at"]
        assert mirror["completed"] == 0
        # The patch touched only `in_flight` — the fuller fields survive.
        assert mirror["total"] == 1

        scheduler._undeclared_writes("apply-batch-0", snapshot)
        scheduler._mirror_status(result)
        mirror = status_file.read_status(
            str(env.phase_dir)
        )["progress"]["write_subagents"]
        assert mirror["in_flight"] == []


class TestTheContainmentSweepIgnoresOnlyInFlightDeclarations:
    """F13: `_begin_sweep` used to union every dispatch's declared set into a
    PHASE-LIFETIME `_declared_seen`, so once ANY task declared a path, a
    LATER task's undeclared write to that same path was invisible to
    `_undeclared_writes` for the rest of the run — even long after the
    declaring task had completed. The stated justification (a co-scheduled
    sibling writing its own declared files while a neighbour is being
    classified) only ever needed the declared sets of tasks still IN FLIGHT."""

    def test_an_undeclared_write_after_the_declaring_task_completed_is_caught(
        self, env
    ):
        secret = env.root / ".env"
        secret.write_text("TOKEN=original\n", encoding="utf-8")
        task_a = env.apply_task(0, task_id="T-A", declared_files=[str(secret)],
                                 snapshot_files=[str(secret)])
        task_b = env.apply_task(1, task_id="T-B", declared_files=[str(env.plan)],
                                 snapshot_files=[str(env.plan)])
        scheduler = env.scheduler([task_a, task_b])

        # T-A dispatches, declares .env, edits it (its OWN declared write —
        # correctly never flagged), and is classified: no longer in flight.
        snap_a = scheduler._snapshot(task_a)
        scheduler._begin_sweep("T-A", [".env"])
        secret.write_text("TOKEN=updated-by-A\n", encoding="utf-8")
        undeclared_a = scheduler._undeclared_writes("T-A", snap_a)
        assert undeclared_a.paths == []

        # T-B dispatches later, declares only the plan — but its runner
        # touches `.env` too: a genuine containment failure T-B never
        # declared, and T-A is long done by now.
        snap_b = scheduler._snapshot(task_b)
        scheduler._begin_sweep("T-B", ["plans/p.md"])
        secret.write_text("TOKEN=leaked-by-B\n", encoding="utf-8")
        undeclared_b = scheduler._undeclared_writes("T-B", snap_b)

        assert ".env" in undeclared_b.paths
        with pytest.raises(write_safety.PolicyViolation):
            scheduler.policy.check(undeclared_b.paths, when="post-run")

    def test_a_co_scheduled_siblings_own_declared_write_is_still_ignored(
        self, env
    ):
        """The one case the phase-lifetime union WAS protecting: two tasks in
        flight AT THE SAME TIME, each legitimately editing its own declared
        file while the other is still being classified. Pruning on
        completion rather than never must not reintroduce this as a false
        positive."""
        file_a = env.root / "src" / "a.py"
        file_b = env.root / "src" / "b.py"
        task_a = env.apply_task(0, task_id="T-A", declared_files=[str(file_a)],
                                 snapshot_files=[str(file_a)])
        task_b = env.apply_task(1, task_id="T-B", declared_files=[str(file_b)],
                                 snapshot_files=[str(file_b)])
        scheduler = env.scheduler([task_a, task_b])

        snap_a = scheduler._snapshot(task_a)
        snap_b = scheduler._snapshot(task_b)
        # Both dispatched into the same disjoint group: both in flight.
        scheduler._begin_sweep("T-A", ["src/a.py"])
        scheduler._begin_sweep("T-B", ["src/b.py"])
        file_a.write_text("a = 2\n", encoding="utf-8")
        file_b.write_text("b = 2\n", encoding="utf-8")

        # T-A is classified first, while T-B is still in flight — T-B's own
        # (legitimate, declared) edit must not read as T-A's undeclared write.
        undeclared_a = scheduler._undeclared_writes("T-A", snap_a)
        assert "src/b.py" not in undeclared_a.paths


class TestTheContractCarriesTheProcessSafetyRule:
    """Every write kind's dispatched prompt forbids process-wide signals.

    N3 (2026-09-16): the implement template used to carry the rule a second
    time in its own ``## Constraints``, which duplicated this seam and
    stranded the template's terminal "Proceed with the implementation." ahead
    of two more required sections. It is gone from the template now — this
    shared result-contract seam is the only place any of the four kinds picks
    the rule up.
    """

    def test_every_write_kind_is_told_never_to_kill(self, tmp_path):
        from utils.write_batch_scheduler import WriteBatchScheduler

        for kind in ("apply_suggestions", "apply_task_suggestions",
                     "apply_code_fixes", "implement"):
            scheduler = WriteBatchScheduler.__new__(WriteBatchScheduler)
            scheduler.kind = kind
            text = scheduler._with_result_contract(
                {"expected_ids": ["X1"]}, "do the thing"
            )
            assert "## Process safety (required)" in text, kind
            assert "`killall`" in text and "background servers" in text, kind
            assert text.index("## Process safety") < text.index("## Result"), kind
