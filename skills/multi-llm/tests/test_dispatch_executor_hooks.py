"""The write scheduler's default dispatch, with only ``_spawn`` stood in for.

``WriteBatchScheduler._dispatch_executor`` is where the scheduler's write rules
connect to ``dispatch_task``: the audit line and the kept prompt
(``before_attempt``), the classification from the bytes (``after_result``) and
rule 9's re-send (``nothing_changed``). Most scheduler tests inject the
simulated native invoker through the ``dispatcher`` seam
(``tests/harness/native_invoker.py``) and never reach those hooks, so these go
through the default path, with ``_spawn`` wrapped to see what each attempt
was handed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.harness.native_invoker import native_dispatcher  # noqa: E402
from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401
from utils import subagent_executor as executor  # noqa: E402
from utils import dispatch_spawn  # noqa: E402
from utils import subagent_prompts  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402
from utils import write_grant  # noqa: E402

APPLIED_S0 = {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
              "summary": "added the step"}


def _spawns(monkeypatch):
    """Record every ``_spawn`` call's keywords, then run the real spawn."""
    real = executor._spawn
    calls = []

    def spawn(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(dispatch_spawn, "_spawn", spawn)
    return calls


def _audit(env, event):
    rows = [
        json.loads(line) for line in
        (env.phase_dir / "subagent_audit.jsonl").read_text().splitlines() if line
    ]
    return [row for row in rows if row.get("event") == event]


def _report(env):
    return json.loads(
        (env.phase_dir / executor.BATCH_REPORT_FILENAME).read_text(encoding="utf-8")
    )


class TestTheDefaultPathWiresTheHooks:

    def test_before_attempt_audits_and_keeps_the_prompt_the_spawn_sent(self, env):
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
        calls = _spawns(env.monkeypatch)

        result = env.scheduler([env.apply_task(0)]).run()

        assert result.exit_code == 0, result.messages
        assert len(calls) == 1
        sent = calls[0]["prompt"]
        dispatched = _audit(env, "dispatched")
        assert [(row["task_id"], row["attempt"], row["runner"]) for row in dispatched] == [
            ("apply-batch-0", 1, "codex:gpt-5.5"),
        ]
        assert dispatched[0]["prompt_sha256"] == hashlib.sha256(
            sent.encode("utf-8")
        ).hexdigest()
        assert [row["task_id"] for row in _audit(env, "completed")] == ["apply-batch-0"]
        kept = env.phase_dir / "prompts" / "apply-batch-0.prompt.txt"
        assert kept.read_text(encoding="utf-8") == sent

    def test_after_result_classifies_from_the_bytes(self, env):
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
        _spawns(env.monkeypatch)

        result = env.scheduler([env.apply_task(0)]).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_OK, outcome.reason
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.files_touched == ["plans/p.md"]
        assert outcome.runner == "codex:gpt-5.5"
        assert _report(env)["invoker"] == "executor"

    def test_nothing_changed_sends_the_one_re_send(self, env):
        """An `applied` claim over an unchanged tree goes back once (rule 9),
        and a second one fails the batch."""
        env.edits([], result=APPLIED_S0)
        calls = _spawns(env.monkeypatch)

        result = env.scheduler([env.apply_task(0)]).run()

        assert len(calls) == 2
        heading = subagent_prompts.NOTHING_CHANGED_RESEND_HEADING
        assert heading not in calls[0]["prompt"]
        assert heading in calls[1]["prompt"]
        outcome = result.outcomes[0]
        assert outcome.applied_without_changes
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert [row["attempt"] for row in _audit(env, "dispatched")] == [1, 2]

    def test_every_spawn_is_handed_the_full_keyword_set(self, env):
        """Keywords are passed unconditionally: a hook that is off is `None`,
        never a missing keyword a narrow stand-in could rely on."""
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
        calls = _spawns(env.monkeypatch)

        env.scheduler([env.apply_task(0)]).run()

        assert {"read_dirs", "jail_request", "write_consent"} <= set(calls[0])


class TestTheDispatchSeam:

    def test_an_injected_dispatcher_replaces_the_executor(self, env, monkeypatch):
        def no_runner(*args, **kwargs):
            raise AssertionError("an injected dispatcher starts no runner")

        monkeypatch.setattr(write_grant, "preflight_runner", no_runner)
        monkeypatch.setattr(dispatch_spawn, "_spawn", no_runner)

        def respond(task, prompt):
            env.plan.write_text(env.plan.read_text() + "edit\n", encoding="utf-8")
            return APPLIED_S0

        result = env.scheduler(
            [env.apply_task(0)], dispatcher=native_dispatcher(respond),
        ).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_OK, outcome.reason
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.runner == "native"
        assert _report(env)["invoker"] == "native-simulated"
        assert [row["runner"] for row in _audit(env, "dispatched")] == ["native"]

    def test_an_injected_dispatcher_must_name_its_invoker(self, env):
        def unnamed(scheduler, task, prompt, snapshot, profile, declared, **_):
            raise AssertionError("never dispatched")

        with pytest.raises(AttributeError):
            env.scheduler([env.apply_task(0)], dispatcher=unnamed)
