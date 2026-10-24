"""A failed write attempt is never sent again on top of its own edits.

Fix round 8 (review 7, r5): an ``implement`` batch that edited ``src/main.py``
and then replied in prose was recorded ``unparseable_result``, which is on the
transport retry rung, so the executor re-sent the byte-identical batch prompt
and the edit landed twice. The pre-retry changed-files check (T4.8) ran only for
a turn a denial ended. It now runs before every re-dispatch of a write batch's
own prompt — a timeout, a runner error, a prose reply, a rate limit — and an
attempt that moved bytes is never re-sent: it is a partial application (loop
stops, snapshot kept, exit 3), or, for an apply kind whose reply could not be
read, the one report re-ask, which restates the report and edits nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import implement_orchestrator  # noqa: E402
from tests.mock_launchers import install_mock_llm, mock_path  # noqa: E402
from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.output_handler import get_phase_dir  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.subagent_tasks import PRIOR_CHANGES_TOKEN  # noqa: E402

REASK_MARKER = "You are re-sending the report of a batch of edits."
VERDICT_WARNING = "could not record the verdict"

GOOD = {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
        "summary": "added the step"}
PROSE = "I went ahead and made the changes. Everything looks good now!"


def _prompts(log: Path):
    if not log.exists():
        return []
    return [json.loads(line)["prompt"] for line in log.read_text().splitlines()]


# ===========================================================================
# The apply kinds: a prose reply after edits gets the report re-ask
# ===========================================================================

class TestAnApplyBatchThatRepliedInProse:

    def test_edits_then_prose_get_the_report_re_ask_not_the_batch(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)], prose=True)
        env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps(GOOD))

        result = env.scheduler([env.apply_task(0)]).run()

        first, second = _prompts(log)
        assert "Apply batch 0" in first
        assert second.startswith(REASK_MARKER)
        assert "Apply batch 0" not in second, "the batch is never sent twice"
        assert PROSE.split(".")[0] in second or "made the changes" in second, (
            "the re-ask quotes the prose reply it asks to restate"
        )
        assert env.plan.read_text(encoding="utf-8").count("edit\n") == 1
        outcome = result.outcomes[0]
        assert [(a.status, a.repair) for a in outcome.attempts] == [
            (executor.STATUS_UNPARSEABLE, None),
            (executor.STATUS_OK, executor.REPAIR_SCHEMA_INVALID),
        ]
        # What was applied is still the snapshot comparison's verdict.
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert result.exit_code == 0

    def test_a_prose_re_ask_is_retried_as_the_re_ask_never_as_the_batch(self, env):
        """Fix round 9 (review of round 8): when the granted re-ask fails too,
        the bytes the batch moved make the outcome a partial application, not
        a schema rejection: the stop says what landed and names the restore."""
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)], prose=True)
        env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", PROSE)
        lines = []

        result = env.scheduler([env.apply_task(0)], log=lines.append).run()

        prompts = _prompts(log)
        assert sum("Apply batch 0" in p for p in prompts) == 1
        assert all(p.startswith(REASK_MARKER) for p in prompts[1:])
        assert env.plan.read_text(encoding="utf-8").count("edit\n") == 1
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_PARTIAL
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert outcome.reason.startswith("PARTIAL APPLICATION: 1 file(s) changed (plans/p.md)")
        assert "(after the one schema-repair re-ask)" in outcome.reason
        assert outcome.files_touched == ["plans/p.md"]
        assert result.exit_code == 3
        assert not (env.phase_dir / "apply_batch_0.json").exists()
        assert not [line for line in lines if VERDICT_WARNING in line], lines
        _assert_the_stop_reads_partial(env, result)

    def test_a_re_ask_that_times_out_stops_as_a_partial_application(
        self, env, monkeypatch
    ):
        """The re-ask's timeout is retried as the re-ask (never the batch); when
        every try times out the batch's moved bytes still decide the outcome,
        which used to be an unlabelled ``timeout``."""
        env.edits([_patch_edit(env.plan)], prose=True)
        real = executor._spawn
        prompts = []

        def spawn(**kwargs):
            prompts.append(kwargs.get("prompt") or "")
            if len(prompts) == 1:
                return real(**kwargs)
            return executor._Capture(timed_out=True, argv=["codex"], duration=0.01)

        monkeypatch.setattr(executor, "_spawn", spawn)
        result = env.scheduler([env.apply_task(0)]).run()

        assert sum("Apply batch 0" in p for p in prompts) == 1
        assert len(prompts) >= 2 and all(p.startswith(REASK_MARKER) for p in prompts[1:])
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_PARTIAL, outcome.reason
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert outcome.files_touched == ["plans/p.md"]
        assert result.exit_code == 3
        _assert_the_stop_reads_partial(env, result)

    def test_the_partial_resumes_like_any_partial(self, env):
        """``--resume`` takes the existing partial-application path: the
        batch is dispatched again against the snapshot taken before its FIRST
        dispatch (the rollback point stays the original bytes), and what the
        resumed runner reports is judged against that snapshot."""
        env.edits([_patch_edit(env.plan)], prose=True)
        env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", PROSE)
        first = env.scheduler([env.apply_task(0)]).run()
        assert first.outcomes[0].write_outcome == executor.WRITE_PARTIAL
        # A runner that finds the change already there and reports it.
        env.edits([], result=GOOD)
        lines = []
        second = env.scheduler(
            [env.apply_task(0)], resume=True, log=lines.append,
        ).run()
        assert any("reusing the pre-batch snapshot" in line for line in lines), lines
        outcome = second.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_APPLIED, outcome.reason
        assert second.exit_code == 0
        assert env.plan.read_text(encoding="utf-8").count("edit\n") == 1

    def test_prose_with_nothing_moved_is_retried_as_before(self, env):
        """No edit on disk: re-sending the batch applies nothing twice."""
        log = env.enable_call_log()
        env.edits([], prose=True)
        lines = []

        result = env.scheduler([env.apply_task(0)], log=lines.append).run()

        prompts = _prompts(log)
        assert len(prompts) == 2 and all("Apply batch 0" in p for p in prompts)
        assert result.outcomes[0].status == executor.STATUS_UNPARSEABLE
        assert result.exit_code == 3
        # No result file was ever persisted, so there is no verdict to stamp
        # and nothing to warn about.
        assert not [line for line in lines if VERDICT_WARNING in line], lines


def _assert_the_stop_reads_partial(env, result) -> None:
    """The stop, the audit record and the report all name the partial."""
    text = "\n".join(result.messages)
    assert "partial application (bytes moved, the batch did not finish)" in text, text
    assert "util restore-batch" in text and "--task-id apply-batch-0" in text
    from utils.write_safety import AuditLog

    completed = [
        r for r in AuditLog(str(env.phase_dir)).records()
        if r.get("event") == "completed"
    ]
    assert completed[-1]["write_outcome"] == executor.WRITE_PARTIAL
    report = json.loads((env.phase_dir / "write_batch_report.json").read_text())
    row = next(r for r in report["tasks"] if r["task_id"] == "apply-batch-0")
    assert row["write_outcome"] == executor.WRITE_PARTIAL


# ===========================================================================
# Every retryable rung: the check runs before the batch is re-sent
# ===========================================================================

def _capture(name: str) -> "executor._Capture":
    return {
        executor.STATUS_TIMEOUT: executor._Capture(
            timed_out=True, argv=["opencode"], duration=0.01,
        ),
        executor.STATUS_RUNNER_ERROR: executor._Capture(
            stderr="boom\n", exit_code=1, argv=["opencode"], duration=0.01,
        ),
        executor.STATUS_UNPARSEABLE: executor._Capture(
            stdout=PROSE + "\n", exit_code=0, argv=["opencode"], duration=0.01,
        ),
        executor.STATUS_RATE_LIMITED: executor._Capture(
            stderr="Error: 429 Too Many Requests: rate limit exceeded\n",
            exit_code=1, argv=["opencode"], duration=0.01,
        ),
    }[name]


RETRYABLE = [
    executor.STATUS_TIMEOUT, executor.STATUS_RUNNER_ERROR,
    executor.STATUS_UNPARSEABLE, executor.STATUS_RATE_LIMITED,
]


class TestEveryRetryableRungAsksFirst:

    def _dispatch(self, env, monkeypatch, status, moved, **extra):
        spawns = []

        def fake_spawn(**kwargs):
            spawns.append(kwargs.get("prompt"))
            return _capture(status)

        monkeypatch.setattr(executor, "_spawn", fake_spawn)
        asked = []
        task = env.apply_task(0)
        outcome = executor.dispatch_task(
            task,
            phase_dir=str(env.phase_dir),
            runner="opencode:opencode/big-pickle",
            cwd=str(env.root),
            retries=1,
            prompt=task["prompt"].replace(PRIOR_CHANGES_TOKEN, "(none)"),
            changed_files=lambda context: asked.append(context) or list(moved),
            **extra,
        )
        return outcome, spawns, asked

    @pytest.mark.parametrize("status", RETRYABLE)
    def test_bytes_moved_is_a_partial_application_with_one_spawn(
        self, env, monkeypatch, status
    ):
        outcome, spawns, asked = self._dispatch(
            env, monkeypatch, status, ["plans/p.md"]
        )
        assert len(spawns) == 1, f"{status}: the batch was sent again"
        assert len(asked) == 1 and asked[0]["attempt"] == 1
        assert outcome.status == executor.STATUS_PARTIAL
        assert [a.status for a in outcome.attempts] == [executor.STATUS_PARTIAL]
        assert outcome.reason.startswith("PARTIAL APPLICATION: 1 file(s) changed")
        assert "plans/p.md" in outcome.reason
        assert "No further attempt is started" in outcome.reason

    @pytest.mark.parametrize("status", RETRYABLE)
    def test_nothing_moved_keeps_the_rung(self, env, monkeypatch, status):
        outcome, spawns, _asked = self._dispatch(env, monkeypatch, status, [])
        assert len(spawns) >= 2, f"{status}: a clean attempt is retried as before"
        assert outcome.status == status
        assert len(set(spawns)) == 1, "a transport retry re-sends the same batch"

    def test_a_hook_that_cannot_tell_fails_closed(self, env, monkeypatch):
        def broken(context):
            raise OSError("gone")

        monkeypatch.setattr(
            executor, "_spawn",
            lambda **kwargs: _capture(executor.STATUS_TIMEOUT),
        )
        task = env.apply_task(0)
        outcome = executor.dispatch_task(
            task, phase_dir=str(env.phase_dir),
            runner="opencode:opencode/big-pickle", cwd=str(env.root), retries=1,
            prompt=task["prompt"].replace(PRIOR_CHANGES_TOKEN, "(none)"),
            changed_files=broken,
        )
        assert outcome.status == executor.STATUS_PARTIAL
        assert "could not be compared" in outcome.reason
        assert len(outcome.attempts) == 1

    def test_prose_without_the_file_guard_stops_as_partial(self, env, monkeypatch):
        """No ``tree_fingerprint`` hook, no proof a re-ask would move nothing:
        the apply kind's prose reply ends as the partial application."""
        outcome, spawns, _asked = self._dispatch(
            env, monkeypatch, executor.STATUS_UNPARSEABLE, ["plans/p.md"]
        )
        assert len(spawns) == 1
        assert outcome.status == executor.STATUS_PARTIAL

    def test_prose_with_the_file_guard_is_re_asked(self, env, monkeypatch):
        outcome, spawns, _asked = self._dispatch(
            env, monkeypatch, executor.STATUS_UNPARSEABLE, ["plans/p.md"],
            tree_fingerprint=lambda context: {"plans/p.md": "same"},
        )
        assert len(spawns) == 3, "the batch once, then the re-ask and its retry"
        assert spawns[1].startswith(REASK_MARKER)
        assert spawns[2] == spawns[1], "a re-ask's retry re-sends the re-ask"
        assert outcome.status == executor.STATUS_UNPARSEABLE
        assert [a.repair for a in outcome.attempts] == [
            None, executor.REPAIR_SCHEMA_INVALID, executor.REPAIR_SCHEMA_INVALID,
        ]


# ===========================================================================
# implement: the measured r5 shape, end to end (the report re-ask since fix
# round 14)
# ===========================================================================

PLAN = """# Two edits

## Overview
Two edits in a row.

## Tasks

### T001: Edit: create the module
Create the module file.
- Depends on: none
- Files to modify: src/main.py

### T002: Edit: extend the module
Extend it.
- Depends on: T001
- Files to modify: src/main.py
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "plans").mkdir()
    plan = root / "plans" / "two.md"
    plan.write_text(PLAN, encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, env=git_env)

    bin_dir = tmp_path / "bin"
    install_mock_llm(bin_dir, "codex")
    monkeypatch.setenv("PATH", mock_path(bin_dir, ("codex",)))
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    config = tmp_path / "driver.yaml"
    config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n", encoding="utf-8")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
    monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
    ]))
    monkeypatch.setenv("MOCK_LLM_PROSE", "1")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
    registry._RUN_ANCHOR = None
    registry._config = None
    registry._config_key = None
    executor.reset_version_cache()
    monkeypatch.chdir(root)

    state = get_or_create_state(plan)
    state.mark_phase_completed("generate-tasks")
    state.mark_phase_completed("apply-suggestions")
    state.save()
    yield {"root": root, "plan": plan, "log": log}
    registry._RUN_ANCHOR = None
    registry._config = None
    registry._config_key = None


def test_an_implement_batch_that_replied_in_prose_is_not_sent_again(repo, capsys):
    """The prose reply after the edit gets the report re-ask (fix round 14),
    never the batch again; a re-ask that attests nothing leaves it a partial
    application."""
    code = implement_orchestrator.main(
        ["--plan-file", str(repo["plan"]), "--yes", "--exec-subagents"]
    )
    captured = capsys.readouterr()
    text = captured.out + captured.err

    assert code == 3
    prompts = _prompts(repo["log"])
    assert len(prompts) == 2, "T001 was dispatched again on top of its own edit"
    assert "T001" in prompts[0] and prompts[1].startswith(REASK_MARKER)
    main = (repo["root"] / "src" / "main.py").read_text(encoding="utf-8")
    assert main.count("# edited\n") == 1

    phase_dir = Path(get_phase_dir(repo["plan"], "implement"))
    report = json.loads((phase_dir / "write_batch_report.json").read_text())
    rows = {row["task_id"]: row for row in report["tasks"]}
    assert rows["T001"]["status"] == executor.STATUS_PARTIAL
    assert rows["T001"]["write_outcome"] == executor.WRITE_PARTIAL
    assert "src/main.py" in rows["T001"]["reason"]
    assert "pre-batch snapshot is kept at" in rows["T001"]["reason"]
    assert "T002" in report["not_run_tasks"]
    # The stop names the remedy for a half-applied batch.
    assert "util restore-batch" in text and "--task-id T001" in text
    assert VERDICT_WARNING not in text
    assert get_or_create_state(repo["plan"]).get_task_status("T001") == "failed"


def test_an_implement_batch_whose_report_is_restated_is_applied(repo, capsys, monkeypatch):
    """Fix round 14 (OpenCode racing run 5, T013 and the stop fixture's T003):
    an implement task that finished its edit and replied in prose stopped the
    phase as a partial application. The report re-ask restates the report,
    the byte comparison finds the edit, and the next task runs."""
    monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps({
        "status": "applied", "ids": ["T001", "T002"],
        "files_touched": ["src/main.py"], "summary": "restated the report",
    }))
    code = implement_orchestrator.main(
        ["--plan-file", str(repo["plan"]), "--yes", "--exec-subagents"]
    )
    captured = capsys.readouterr()

    assert code == 0, captured.out + captured.err
    prompts = _prompts(repo["log"])
    batches = [p for p in prompts if not p.startswith(REASK_MARKER)]
    reasks = [p for p in prompts if p.startswith(REASK_MARKER)]
    assert len(batches) == 2 and len(reasks) == 2, "one re-ask per task, no batch twice"
    main = (repo["root"] / "src" / "main.py").read_text(encoding="utf-8")
    assert main.count("# edited\n") == 2, "each task's edit landed once"
    state = get_or_create_state(repo["plan"])
    assert state.get_task_status("T001") == "completed"
    assert state.get_task_status("T002") == "completed"


# ===========================================================================
# A dispatch that persisted no result file has no verdict to stamp
# ===========================================================================

def test_a_failed_dispatch_with_no_result_file_draws_no_verdict_warning(env):
    """``output_path`` names where the result would go; a prose reply never
    wrote it, and the stamp used to warn ``could not record the verdict on
    apply_batch_0.json … No such file`` about a file nobody meant to write."""
    env.edits([], prose=True)
    lines = []

    result = env.scheduler([env.apply_task(0)], log=lines.append).run()

    assert result.outcomes[0].status == executor.STATUS_UNPARSEABLE
    assert not (env.phase_dir / "apply_batch_0.json").exists()
    assert not [line for line in lines if VERDICT_WARNING in line], lines


@pytest.mark.parametrize("mode", ["full", "improve-plan", "improve-code"])
def test_parent_runtime_files_are_not_runner_edits(env, mode):
    from types import SimpleNamespace
    from utils import detach

    parent = env.out_dir / mode
    parent.mkdir()
    env.monkeypatch.setattr(detach, "_ACTIVE", SimpleNamespace(phase_dir=str(parent)))
    scheduler = env.scheduler([env.apply_task(0)])
    before = scheduler._census()
    for name in (".status.json", ".status.lock", "orchestrator-run.log"):
        (parent / name).write_text("runtime update\n")
    assert scheduler._census() == before
    # A source edit, a sibling artifact and a composite result are still visible.
    (env.root / "src/a.py").write_text("changed source bytes\n")
    (parent / "result.json").write_text("changed result\n")
    sibling = env.out_dir / "review-plan"
    sibling.mkdir()
    (sibling / ".status.json").write_text("unrelated phase\n")
    after = scheduler._census()
    changed = {p for p in set(before) | set(after) if before.get(p) != after.get(p)}
    assert changed == {"src/a.py", f".multi-llm/p/{mode}/result.json", ".multi-llm/p/review-plan/.status.json"}


def test_report_reask_survives_parent_heartbeat_and_log_updates(env):
    from types import SimpleNamespace
    from utils import detach, write_batch_scheduler as scheduler_module

    parent = env.out_dir / "full"
    parent.mkdir()
    env.monkeypatch.setattr(detach, "_ACTIVE", SimpleNamespace(phase_dir=str(parent)))
    env.edits([_patch_edit(env.plan)], prose=True)
    env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps(GOOD))
    original = scheduler_module.dispatch_task
    fingerprints = []

    def dispatch(*args, tree_fingerprint=None, **kwargs):
        def fingerprint(context):
            fingerprints.append(context)
            for name in (".status.json", "orchestrator-run.log"):
                (parent / name).write_text(f"parent update {len(fingerprints)}\n")
            return tree_fingerprint(context)
        return original(*args, tree_fingerprint=fingerprint, **kwargs)

    env.monkeypatch.setattr(scheduler_module, "dispatch_task", dispatch)
    result = env.scheduler([env.apply_task(0)]).run()
    assert len(fingerprints) >= 2
    assert result.exit_code == 0
    assert result.outcomes[0].write_outcome == executor.WRITE_APPLIED
