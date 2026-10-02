"""Write batches on the plugin-landlock rung: the scheduler's half (plan step 3).

Plan ``codex-sandbox-unavailable-findings-and-fixes`` step 3 option (b), decided
by the user on 2026-09-30: codex editing jobs run on the plugin-landlock rung for
the ``plan-file`` (apply-suggestions) and ``tasks`` (apply-task-suggestions)
write scopes only. What this file pins is the **scheduler's** share of that
(F1 "How a dispatch receives its grant", "Supported edits under a file-only
grant", "A denied edit is never recorded as applied"; §8.2 "Step 3, if
approved"; §8.3 items 8-11, 13, 14; §5 I11):

* the write target is checked at construction and every task's declared set is
  checked against it before its snapshot, on every rung;
* ONE ``JailRequest`` per phase reaches the preflight and the dispatch, with
  the scope from the kind map and ``target_dir`` only for the default tasks
  directory;
* the stray-writes-only line is printed once before each batch on the rung,
  and never before a batch on codex's own sandbox;
* a denied edit is never recorded as applied: every verdict leg of ``_classify``
  fed by ``after_result``'s ``jail_denials`` and the target's inode;
* I11: a batch-0 refusal leaves no batch in flight in ``.status.json``.

The executor's side (the rung walk, the probe, the wrap, the stream reading)
belongs to ``subagent_executor`` and is stubbed here: ``dispatch_task`` and
``preflight_runner`` are replaced in the scheduler's namespace and
``sandbox_rung``/``write_rung_notice`` on the executor module, so these tests
hold whichever executor is installed. The denials themselves are computed by
the real ``landlock_dispatch.jail_denied_edits`` over a codex-shaped JSONL
stream, so the classifier and the verdict are exercised together. No real
codex, no network, tmp_path scratch repos only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import landlock_dispatch  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_WRITE,
    SANDBOX_RUNG_PLUGIN_LANDLOCK,
    JailContext,
    JailRequest,
)
from utils.subagent_tasks import (  # noqa: E402
    PRIOR_CHANGES_TOKEN,
    build_inputs,
    make_write_task,
)
from utils.write_batch_scheduler import WriteBatchScheduler  # noqa: E402

RUNG = SANDBOX_RUNG_PLUGIN_LANDLOCK
NOTICE_STEM = "[SANDBOX_INCIDENTAL]"
F6_LINE = (
    "To run this now, set `driver.subagent_runner` to a runner that is not "
    "Codex (for example `claude-code:<model>`) in `.multi-llm/providers.yaml`."
)


# ---------------------------------------------------------------------------
# Codex-shaped streams
# ---------------------------------------------------------------------------

def failed_file_change(*paths: str) -> Dict[str, Any]:
    return {"type": "item.completed", "item": {
        "type": "file_change", "status": "failed",
        "changes": [{"path": p, "kind": "update"} for p in paths],
    }}


def command_denied(output: str) -> Dict[str, Any]:
    return {"type": "item.completed", "item": {
        "type": "command_execution", "status": "failed", "exit_code": 1,
        "aggregated_output": output,
    }}


def stream(*events: Dict[str, Any]) -> str:
    rows = list(events) + [{"type": "turn.completed", "usage": {}}]
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def claim(status: str, ids=None, summary: str = "S") -> Dict[str, Any]:
    """A batch result; ``ids`` ``None`` means "every expected id of the task
    it answers", filled in at dispatch."""
    return {"status": status, "ids": None if ids is None else list(ids),
            "files_touched": [], "summary": summary}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Repo:
    """A real git repo with a plan, its default tasks dir, and a phase dir."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.root = (tmp_path / "repo").resolve()
        self.root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "plans").mkdir()
        self.plan = self.root / "plans" / "p.md"
        self.plan.write_text("# Plan\n\n## Step 1\nbody\n", encoding="utf-8")
        self.tasks_dir = self.root / "plans" / "p" / "tasks"
        self.tasks_dir.mkdir(parents=True)
        self.tasks = self.tasks_dir / "tasks.md"
        self.tasks.write_text("# Tasks\n\n### T001: one\n", encoding="utf-8")
        self.out_dir = self.root / "plans" / "p"
        self.phase_dir = self.out_dir / "apply-suggestions"
        self.phase_dir.mkdir(parents=True)
        self.tmp_path = tmp_path
        config = tmp_path / "driver-config.yaml"
        config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n",
                          encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        registry._config = None
        registry._config_key = None

    def task(self, index: int, *, kind: str = "apply_suggestions",
             target: Optional[Path] = None, declared=None, **over):
        target = target or self.plan
        prefix = "apply-batch" if kind == "apply_suggestions" else "apply-task-batch"
        kwargs = dict(
            kind=kind,
            task_id=f"{prefix}-{index}",
            prompt=(
                f"Apply batch {index} to {target}\n\n"
                "## Changes Applied in Prior Batches\n\n"
                f"{PRIOR_CHANGES_TOKEN}\n"
            ),
            inputs=build_inputs(plan_file=str(self.plan)),
            batch_index=index,
            prior_changes_id=f"sec-{index}",
            phase_dir=str(self.phase_dir),
            declared_files=[str(p) for p in (declared or [target])],
            expected_ids=[f"s{index}"],
        )
        kwargs.update(over)
        return make_write_task(**kwargs)

    def scheduler(self, tasks, *, kind: str = "apply_suggestions",
                  write_target: Any = "default", lines: Optional[list] = None,
                  **over) -> WriteBatchScheduler:
        if write_target == "default":
            write_target = self.plan if kind == "apply_suggestions" else self.tasks
        kwargs = dict(
            tasks=tasks,
            phase_dir=str(self.phase_dir),
            kind=kind,
            output_dir=str(self.out_dir),
            plan_file=str(self.plan),
            anchor=str(self.plan),
            repo_root=str(self.root),
            log=(lines.append if lines is not None else (lambda line: None)),
            write_target=str(write_target) if write_target else None,
        )
        kwargs.update(over)
        return WriteBatchScheduler(**kwargs)

    def report(self) -> Dict[str, Any]:
        return json.loads(
            (self.phase_dir / "write_batch_report.json").read_text()
        )

    def status(self) -> Dict[str, Any]:
        return json.loads((self.phase_dir / ".status.json").read_text())

    def prior_changes(self) -> List[str]:
        path = self.out_dir / "prior_changes.jsonl"
        if not path.exists():
            return []
        return [json.loads(line)["summary"] for line in path.read_text().splitlines() if line]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    yield Repo(tmp_path, monkeypatch)
    registry._config = None
    registry._config_key = None


Step = Callable[[Repo], "tuple"]


class FakeExecutor:
    """The executor's plan-step-3 API, stubbed at the scheduler's seams.

    Records every ``preflight_runner`` / ``dispatch_task`` call and what it was
    handed (the jail request above all), runs one scripted *step* per dispatch
    — the "child": it edits files and returns ``(claimed result, stream)`` —
    then computes ``jail_denials`` with the real ``jail_denied_edits`` and calls
    ``after_result`` with the context the executor contract promises.
    """

    def __init__(self, repo: Repo, monkeypatch, *, rung: Optional[str] = RUNG,
                 denials_as_dict: bool = False) -> None:
        self.repo = repo
        self.rung = rung
        self.denials_as_dict = denials_as_dict
        self.steps: List[Step] = []
        self.preflights: List[Dict[str, Any]] = []
        self.dispatches: List[Dict[str, Any]] = []
        self.notices: List[Any] = []
        self.rung_queries: List[Optional[str]] = []
        monkeypatch.setattr(wbs, "dispatch_task", self.dispatch_task)
        monkeypatch.setattr(wbs, "preflight_runner", self.preflight_runner)
        monkeypatch.setattr(executor, "sandbox_rung", self.sandbox_rung)
        monkeypatch.setattr(executor, "write_rung_notice",
                            self.write_rung_notice, raising=False)

    def script(self, *steps: Step) -> "FakeExecutor":
        self.steps.extend(steps)
        return self

    # -- the executor API -----------------------------------------------

    def preflight_runner(self, spec, *, profile, anchor=None, key="",
                         jail_request=None, **kwargs):
        self.preflights.append({"spec": spec, "profile": profile,
                                "jail_request": jail_request})
        return {"version": "0.158.0"}

    def sandbox_rung(self, spec, profile, write_scope=None):
        self.rung_queries.append(write_scope)
        return self.rung

    def write_rung_notice(self, spec, jail_request):
        self.notices.append(jail_request)
        if self.rung != RUNG:
            return None
        return (
            f"{NOTICE_STEM} codex: running write scope "
            f"{jail_request.write_scope} ({jail_request.target}) on fallback "
            f"rung 'plugin-landlock', which stops stray writes, not the model; "
            f"the network is not confined"
        )

    def dispatch_task(self, task, *, phase_dir, runner, anchor=None, cwd=None,
                      prompt=None, before_attempt=None, after_result=None,
                      changed_files=None, jail_request=None, **kwargs):
        self.dispatches.append({"task_id": task["task_id"],
                                "jail_request": jail_request})
        if before_attempt is not None:
            before_attempt({"task_id": task["task_id"], "attempt": 1,
                            "cwd": cwd})
        step = self.steps.pop(0)
        data, events = step(self.repo)
        if data.get("ids") is None:
            data = dict(data, ids=list(task.get("expected_ids") or []))
        outcome = executor.TaskOutcome(
            task_id=task["task_id"], kind=task["kind"], runner=runner,
            status=executor.STATUS_OK, profile=PROFILE_AGENT_WRITE,
        )
        outcome.output_path = executor.persist_result(task, phase_dir, data)
        outcome.attempts.append(executor.Attempt(
            index=1, status=executor.STATUS_OK, exit_code=0,
        ))
        denials = None
        if self.rung == RUNG and jail_request is not None:
            context = JailContext(request=jail_request,
                                  tmpdir=str(self.repo.tmp_path / "slot"),
                                  codex_home=str(self.repo.tmp_path / "home"))
            denials = landlock_dispatch.jail_denied_edits(
                events, context, cwd=cwd,
            )
            if self.denials_as_dict:
                denials = denials.to_dict()
        after_result({
            "task": task, "task_id": task["task_id"], "kind": task["kind"],
            "profile": PROFILE_AGENT_WRITE, "runner": runner, "cwd": cwd,
            "attempt": 1, "prompt": prompt, "result": data, "exit_code": 0,
            "stderr_tail": "", "usage": None, "duration_seconds": 0.1,
            "outcome": outcome,
            "sandbox_rung": self.rung,
            "jail_denials": denials,
            "jail_evidence": None,
        })
        return outcome


def edit_in_place(path_attr: str = "plan", text: str = "\nedited in place\n"):
    """A step that rewrites the target in place (the measured apply_patch
    shape: same inode, new bytes)."""
    def step(repo: Repo):
        path = getattr(repo, path_attr)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
        return claim("applied"), stream()
    return step


def untouched(result: Dict[str, Any], *events: Dict[str, Any]):
    def step(repo: Repo):
        return result, stream(*events)
    return step


# ===========================================================================
# The write target and the declared set (F1 "How a dispatch receives its grant")
# ===========================================================================

class TestTheWriteTargetIsCheckedAtConstruction:

    def test_the_target_is_realpath_resolved(self, repo):
        dotted = repo.root / "plans" / ".." / "plans" / "p.md"
        scheduler = repo.scheduler([], write_target=dotted)
        assert scheduler.write_target == os.path.realpath(repo.plan)

    def test_a_missing_target_refuses_the_phase(self, repo):
        with pytest.raises(executor.ExecutorError, match="does not exist"):
            repo.scheduler([], write_target=repo.root / "plans" / "gone.md")

    def test_a_directory_is_not_a_target(self, repo):
        with pytest.raises(executor.ExecutorError, match="not a regular file"):
            repo.scheduler([], write_target=repo.tasks_dir)

    def test_a_symlink_is_refused_on_the_path_as_given(self, repo):
        link = repo.root / "plans" / "link.md"
        link.symlink_to(repo.plan)
        with pytest.raises(executor.ExecutorError, match="symlink"):
            repo.scheduler([], write_target=link)

    def test_a_target_outside_the_repo_is_refused(self, repo, tmp_path):
        outside = tmp_path / "elsewhere.md"
        outside.write_text("x\n", encoding="utf-8")
        with pytest.raises(executor.ExecutorError, match="outside the repository"):
            repo.scheduler([], write_target=outside)

    def test_no_target_builds_a_request_with_no_grant(self, repo):
        scheduler = repo.scheduler([], write_target=None)
        assert scheduler.write_target is None
        assert scheduler.jail_request.write_scope == "plan-file"
        assert scheduler.jail_request.target is None

    @pytest.mark.parametrize("kind", ["apply_code_fixes", "implement"])
    def test_a_repo_wide_kind_carries_the_output_folder_and_the_deny_list(self, repo, kind):
        """Step 3b: no target — the grant is the enumeration — but this run's
        output folder (protected) and the policy's deny patterns ride the
        request; the declared set joins per dispatch, never here."""
        scheduler = repo.scheduler([], kind=kind, write_target=None)
        request = scheduler.jail_request
        assert request == JailRequest(
            repo_root=str(repo.root), kind=kind, write_scope="repo-wide",
            output_dir=os.path.realpath(scheduler.output_dir),
            deny_patterns=tuple(scheduler.policy.deny),
        )
        assert request.target is None and request.declared == ()
        assert ".git/**" in request.deny_patterns


class TestTheWriteScopeMap:

    def test_the_fallback_map_names_every_write_kind(self, monkeypatch):
        monkeypatch.delattr(executor, "WRITE_SCOPE_BY_KIND", raising=False)
        assert wbs.write_scope_for_kind("apply_suggestions") == "plan-file"
        assert wbs.write_scope_for_kind("apply_task_suggestions") == "tasks"
        assert wbs.write_scope_for_kind("apply_code_fixes") == "repo-wide"
        assert wbs.write_scope_for_kind("implement") == "repo-wide"
        assert wbs.write_scope_for_kind("validation") is None

    def test_the_executors_map_wins_when_it_exists(self, monkeypatch):
        monkeypatch.setattr(executor, "WRITE_SCOPE_BY_KIND",
                            {"apply_suggestions": "plan-file"}, raising=False)
        assert wbs.write_scope_for_kind("apply_suggestions") == "plan-file"
        assert wbs.write_scope_for_kind("implement") is None


class TestTheDeclaredSetMustBeTheTarget:
    """§8.2 "Carrying the grant": a task whose declared set is not the target
    is refused before its snapshot is taken — on every rung."""

    @pytest.mark.parametrize("rung", [RUNG, None])
    def test_a_task_declaring_something_else_is_refused_before_its_snapshot(
        self, repo, monkeypatch, rung
    ):
        fake = FakeExecutor(repo, monkeypatch, rung=rung)
        other = repo.root / "plans" / "other.md"
        other.write_text("# Other\n", encoding="utf-8")
        result = repo.scheduler([repo.task(0, declared=[other])]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_REJECTED
        assert "other.md" in outcome.reason and "grant" in outcome.reason
        assert not (repo.phase_dir / "pre_batch" / "apply-batch-0").exists()
        assert fake.dispatches == [] and fake.preflights == []
        assert result.exit_code == 3

    def test_the_target_plus_a_second_file_is_refused_too(self, repo, monkeypatch):
        fake = FakeExecutor(repo, monkeypatch)
        extra = repo.root / "plans" / "other.md"
        extra.write_text("# Other\n", encoding="utf-8")
        result = repo.scheduler(
            [repo.task(0, declared=[repo.plan, extra])]
        ).run()
        assert result.outcomes[0].status == executor.STATUS_REJECTED
        assert fake.dispatches == []

    def test_a_repo_relative_spelling_of_the_target_is_the_target(
        self, repo, monkeypatch
    ):
        fake = FakeExecutor(repo, monkeypatch).script(edit_in_place())
        result = repo.scheduler(
            [repo.task(0, declared=["plans/p.md"])]
        ).run()
        assert result.exit_code == 0, result.messages
        assert [d["task_id"] for d in fake.dispatches] == ["apply-batch-0"]


# ===========================================================================
# The request reaches the executor; the stray-writes-only line
# ===========================================================================

class TestTheGrantReachesTheExecutor:

    def test_apply_suggestions_announces_each_batch_on_the_rung(
        self, repo, monkeypatch
    ):
        fake = FakeExecutor(repo, monkeypatch).script(
            edit_in_place(text="\none\n"), edit_in_place(text="\ntwo\n"),
        )
        lines: List[str] = []
        result = repo.scheduler(
            [repo.task(0), repo.task(1)], lines=lines,
        ).run()
        assert result.exit_code == 0, result.messages

        request = fake.dispatches[0]["jail_request"]
        assert request == JailRequest(
            repo_root=str(repo.root), kind="apply_suggestions",
            write_scope="plan-file", target=str(repo.plan), target_dir=None,
        )
        # ONE request for the phase, handed to every preflight and dispatch.
        assert all(d["jail_request"] is request for d in fake.dispatches)
        assert all(p["jail_request"] is request for p in fake.preflights)
        assert set(fake.rung_queries) == {"plan-file"}

        notices = [i for i, line in enumerate(lines) if NOTICE_STEM in line]
        batch_lines = [
            i for i, line in enumerate(lines)
            if line.startswith("[write-subagents] batch ")
        ]
        assert len(notices) == 2, lines
        assert len(batch_lines) == 2, lines
        # Once before each batch: notice, batch 0, notice, batch 1.
        assert notices[0] < batch_lines[0] < notices[1] < batch_lines[1]
        assert "the network is not confined" in lines[notices[0]]
        assert all(f"[on {RUNG}]" in lines[i] for i in batch_lines)

        rows = repo.report()["tasks"]
        assert [row["sandbox_rung"] for row in rows] == [RUNG, RUNG]
        assert all(row["write_outcome"] == "applied" for row in rows)
        mirror = repo.status()["progress"]["write_subagents"]
        assert [t.get("sandbox_rung") for t in mirror["tasks"]] == [RUNG, RUNG]
        stamped = json.loads((repo.phase_dir / "apply_batch_0.json").read_text())
        assert stamped["verdict"]["sandbox_rung"] == RUNG

    def test_the_audited_argv_is_the_wrapped_preview_and_creates_nothing(
        self, repo, monkeypatch, tmp_path
    ):
        import tempfile

        before = set(Path(tempfile.gettempdir()).glob("multi-llm-slot-*"))
        FakeExecutor(repo, monkeypatch).script(edit_in_place())
        repo.scheduler([repo.task(0)]).run()
        dispatched = [
            json.loads(line)
            for line in (repo.phase_dir / "subagent_audit.jsonl").read_text().splitlines()
            if '"dispatched"' in line
        ]
        argv = dispatched[0]["argv"]
        assert any(str(a).endswith("landlock_exec.py") for a in argv), argv
        assert f"file:{repo.plan}" in argv
        assert f"dir:{landlock_dispatch.PLACEHOLDER_TMPDIR}" in argv
        assert f"dir:{landlock_dispatch.PLACEHOLDER_CODEX_HOME}" in argv
        assert "danger-full-access" in argv
        assert set(Path(tempfile.gettempdir()).glob("multi-llm-slot-*")) == before

    def test_codex_own_sandbox_gets_no_line_and_no_rung_record(
        self, repo, monkeypatch
    ):
        FakeExecutor(repo, monkeypatch, rung=None).script(edit_in_place())
        lines: List[str] = []
        result = repo.scheduler([repo.task(0)], lines=lines).run()
        assert result.exit_code == 0
        assert not any(NOTICE_STEM in line for line in lines)
        row = repo.report()["tasks"][0]
        assert "sandbox_rung" not in row
        audit = (repo.phase_dir / "subagent_audit.jsonl").read_text()
        assert "landlock_exec.py" not in audit

    def test_apply_task_suggestions_default_tasks_dir_gets_the_dir_rule(
        self, repo, monkeypatch
    ):
        fake = FakeExecutor(repo, monkeypatch).script(edit_in_place("tasks"))
        lines: List[str] = []
        result = repo.scheduler(
            [repo.task(0, kind="apply_task_suggestions", target=repo.tasks)],
            kind="apply_task_suggestions", lines=lines,
        ).run()
        assert result.exit_code == 0, result.messages
        request = fake.dispatches[0]["jail_request"]
        assert request.write_scope == "tasks"
        assert request.target == str(repo.tasks)
        assert request.target_dir == str(repo.tasks_dir)
        assert sum(NOTICE_STEM in line for line in lines) == 1

    def test_a_tasks_file_override_into_plans_gets_a_file_rule(
        self, repo, monkeypatch
    ):
        override = repo.root / "plans" / "tasks.md"
        override.write_text("# Tasks\n\n### T001: one\n", encoding="utf-8")

        def edit_override(r: Repo):
            with open(override, "a", encoding="utf-8") as handle:
                handle.write("\nedit\n")
            return claim("applied"), stream()

        fake = FakeExecutor(repo, monkeypatch).script(edit_override)
        result = repo.scheduler(
            [repo.task(0, kind="apply_task_suggestions", target=override)],
            kind="apply_task_suggestions", write_target=override,
        ).run()
        assert result.exit_code == 0, result.messages
        request = fake.dispatches[0]["jail_request"]
        assert request.write_scope == "tasks"
        assert request.target == str(override)
        assert request.target_dir is None

    def test_a_rename_inside_the_tasks_dir_is_not_a_replaced_target(
        self, repo, monkeypatch
    ):
        """Under the dir rule the watched identity is the DIRECTORY: a
        replacement inside it is a granted write shape, and claim plus bytes
        decide as today."""
        def rename_inside(r: Repo):
            temp = r.tasks_dir / "tasks.md.tmp"
            temp.write_text("# Tasks\n\n### T001: renamed in\n", encoding="utf-8")
            os.replace(temp, r.tasks)
            return claim("applied"), stream()

        FakeExecutor(repo, monkeypatch).script(rename_inside)
        result = repo.scheduler(
            [repo.task(0, kind="apply_task_suggestions", target=repo.tasks)],
            kind="apply_task_suggestions",
        ).run()
        assert result.exit_code == 0, result.messages
        assert result.outcomes[0].write_outcome == executor.WRITE_APPLIED

    def test_an_executor_without_the_new_keywords_still_dispatches(
        self, repo, monkeypatch
    ):
        """The executor package lands separately (2026-09-30): against the old
        signatures the scheduler passes no ``jail_request`` and still runs."""
        seen: Dict[str, Any] = {}

        def old_preflight(spec, *, profile, anchor=None, key=""):
            seen["preflight"] = True
            return {"version": "0.158.0"}

        def old_dispatch(task, *, phase_dir, runner, anchor=None, cwd=None,
                         prompt=None, before_attempt=None, after_result=None,
                         changed_files=None):
            repo.plan.write_text(repo.plan.read_text() + "\nx\n", encoding="utf-8")
            outcome = executor.TaskOutcome(
                task_id=task["task_id"], kind=task["kind"], runner=runner,
                status=executor.STATUS_OK, profile=PROFILE_AGENT_WRITE,
            )
            data = claim("applied", ids=task["expected_ids"])
            outcome.output_path = executor.persist_result(task, phase_dir, data)
            after_result({"outcome": outcome, "result": data, "attempt": 1,
                          "exit_code": 0, "duration_seconds": 0.1})
            return outcome

        monkeypatch.setattr(wbs, "preflight_runner", old_preflight)
        monkeypatch.setattr(wbs, "dispatch_task", old_dispatch)
        monkeypatch.setattr(executor, "sandbox_rung",
                            lambda spec, profile: None)
        monkeypatch.delattr(executor, "write_rung_notice", raising=False)
        result = repo.scheduler([repo.task(0)]).run()
        assert seen["preflight"]
        assert result.exit_code == 0, result.messages
        assert result.outcomes[0].write_outcome == executor.WRITE_APPLIED


# ===========================================================================
# A denied edit is never reported applied (§8.2)
# ===========================================================================

class TestADeniedEditIsNeverReportedApplied:

    @pytest.mark.parametrize("claimed", ["applied", "no-op"])
    @pytest.mark.parametrize("as_dict", [False, True])
    def test_a_failed_file_change_on_the_plan_fails_the_batch(
        self, repo, monkeypatch, claimed, as_dict
    ):
        fake = FakeExecutor(repo, monkeypatch, denials_as_dict=as_dict).script(
            untouched(claim(claimed), failed_file_change(str(repo.plan))),
            edit_in_place(),
        )
        before = repo.plan.read_bytes()
        result = repo.scheduler([repo.task(0), repo.task(1)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert not outcome.ok
        assert outcome.write_outcome != executor.WRITE_APPLIED
        assert "plans/p.md" in outcome.reason and "jail" in outcome.reason
        assert "apply_patch" in outcome.reason and "(F6)" in outcome.reason
        # The loop stopped at the boundary, the phase is not completed, exit 3.
        assert [d["task_id"] for d in fake.dispatches] == ["apply-batch-0"]
        assert result.exit_code == 3 and not result.finalized
        assert result.stopped_at_batch == 0
        assert repo.plan.read_bytes() == before
        row = repo.report()["tasks"][0]
        assert row["status"] == executor.STATUS_BATCH_FAILED
        assert row["write_outcome"] != executor.WRITE_APPLIED
        assert row["jail_denials"]["failed_file_changes"] == [[str(repo.plan)]]
        # Resume re-dispatches it: the verdict on disk is not `ok`.
        stamped = json.loads((repo.phase_dir / "apply_batch_0.json").read_text())
        assert stamped["verdict"]["ok"] is False
        assert repo.prior_changes() == []

    def test_a_command_denial_on_the_plan_fails_the_batch(self, repo, monkeypatch):
        FakeExecutor(repo, monkeypatch).script(untouched(
            claim("applied"),
            command_denied("sh: 1: cannot create plans/p.md: Permission denied\n"),
        ))
        result = repo.scheduler([repo.task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert "plans/p.md" in outcome.reason
        assert result.exit_code == 3

    def test_the_partial_patch_shape_is_a_partial_application(
        self, repo, monkeypatch
    ):
        def partial(r: Repo):
            with open(r.plan, "a", encoding="utf-8") as handle:
                handle.write("\nfirst file of the patch\n")
            return claim("applied"), stream(
                failed_file_change(str(r.plan), str(r.root / "src" / "b.md"))
            )

        fake = FakeExecutor(repo, monkeypatch).script(partial, edit_in_place())
        result = repo.scheduler([repo.task(0), repo.task(1)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_PARTIAL
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert outcome.stops_the_loop
        assert "src/b.md" in outcome.reason and "jail" in outcome.reason
        assert result.exit_code == 3 and not result.finalized
        assert len(fake.dispatches) == 1
        # The snapshot is kept, and the rollback set names the plan.
        assert (repo.phase_dir / "pre_batch" / "apply-batch-0").is_dir()
        assert "plans/p.md" in outcome.files_touched

    def test_an_out_of_scope_denial_beside_a_real_edit_is_applied_with_its_count(
        self, repo, monkeypatch
    ):
        def edit_and_stray(r: Repo):
            with open(r.plan, "a", encoding="utf-8") as handle:
                handle.write("\nreal edit\n")
            return claim("applied"), stream(
                command_denied("sh: 1: cannot create /tmp/x: Permission denied\n")
            )

        FakeExecutor(repo, monkeypatch).script(edit_and_stray)
        lines: List[str] = []
        result = repo.scheduler([repo.task(0)], lines=lines).run()
        outcome = result.outcomes[0]
        assert result.exit_code == 0, result.messages
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.ok
        assert "1 out-of-scope denial" in (outcome.reason or "")
        batch_line = next(line for line in lines if line.startswith("[write-subagents] batch 0"))
        assert "1 out-of-scope denial" in batch_line
        assert repo.report()["tasks"][0]["jail_denials"]["out_of_scope_count"] == 1

    def test_a_sed_temp_file_denial_with_the_plan_unchanged_fails_the_batch(
        self, repo, monkeypatch
    ):
        FakeExecutor(repo, monkeypatch).script(untouched(
            claim("applied"),
            command_denied(
                "sed: couldn't open temporary file plans/sedAbc123: "
                "Permission denied\n"
            ),
        ))
        result = repo.scheduler([repo.task(0)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert "plans/sedAbc123" in outcome.reason
        assert "in-place edits" in outcome.reason
        assert "apply_patch" in outcome.reason
        assert result.exit_code == 3

    def test_the_same_denial_then_an_in_place_edit_is_applied(
        self, repo, monkeypatch
    ):
        def recovered(r: Repo):
            with open(r.plan, "a", encoding="utf-8") as handle:
                handle.write("\nrecovered through apply_patch\n")
            return claim("applied"), stream(command_denied(
                "sed: couldn't open temporary file plans/sedAbc123: "
                "Permission denied\n"
            ))

        FakeExecutor(repo, monkeypatch).script(recovered)
        lines: List[str] = []
        result = repo.scheduler([repo.task(0)], lines=lines).run()
        outcome = result.outcomes[0]
        assert result.exit_code == 0, result.messages
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert outcome.status == executor.STATUS_OK
        assert not outcome.stops_the_loop
        batch_line = next(line for line in lines if line.startswith("[write-subagents] batch 0"))
        assert "1 replacement denial" in batch_line
        assert "1 replacement denial" in outcome.reason

    def test_the_plan_renamed_over_mid_turn_is_a_replaced_target(
        self, repo, monkeypatch
    ):
        new_bytes = b"# Someone else's plan\n"

        def swapped(r: Repo):
            temp = r.root / "plans" / "p.md.swap"
            temp.write_bytes(new_bytes)
            os.replace(temp, r.plan)
            return claim("applied"), stream()

        fake = FakeExecutor(repo, monkeypatch).script(swapped, edit_in_place())
        result = repo.scheduler([repo.task(0), repo.task(1)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert "replaced while the runner was editing it" in outcome.reason
        assert "grant stayed on the old file" in outcome.reason
        assert result.exit_code == 3 and not result.finalized
        assert len(fake.dispatches) == 1
        # Untouched by any rollback, and out of reach of `util restore-batch`.
        assert repo.plan.read_bytes() == new_bytes
        assert "plans/p.md" not in outcome.files_touched
        completed = [
            json.loads(line)
            for line in (repo.phase_dir / "subagent_audit.jsonl").read_text().splitlines()
            if '"completed"' in line
        ]
        assert all("plans/p.md" not in row["files_touched"] for row in completed)
        # The pre-batch snapshot is kept for comparison.
        assert (repo.phase_dir / "pre_batch" / "apply-batch-0").is_dir()
        assert repo.prior_changes() == []

    @pytest.mark.parametrize("ended", [
        executor.STATUS_TIMEOUT, executor.STATUS_REJECTED,
    ])
    def test_a_replaced_target_on_an_unclassified_dispatch_is_never_restored(
        self, repo, monkeypatch, ended
    ):
        """A timeout or a `[JAIL_DENIED]` rejection never reaches `_classify`:
        it ends in `_audit_terminal`, whose `files_touched` is the byte
        comparison against the snapshot. The file saved over the plan differs
        from it, and must still stay out of the restore set."""
        from utils import restore_batch

        new_bytes = b"# The user's own save\n"
        fake = FakeExecutor(repo, monkeypatch)

        def unclassified(task, *, phase_dir, runner, cwd=None,
                         before_attempt=None, jail_request=None, **kwargs):
            fake.dispatches.append({"task_id": task["task_id"],
                                    "jail_request": jail_request})
            before_attempt({"task_id": task["task_id"], "attempt": 1,
                            "cwd": cwd})
            temp = repo.root / "plans" / "p.md.swap"
            temp.write_bytes(new_bytes)
            os.replace(temp, repo.plan)
            outcome = executor.TaskOutcome(
                task_id=task["task_id"], kind=task["kind"], runner=runner,
                status=ended, profile=PROFILE_AGENT_WRITE,
                reason=f"the dispatch ended {ended}",
            )
            outcome.attempts.append(executor.Attempt(index=1, status=ended))
            return outcome

        monkeypatch.setattr(wbs, "dispatch_task", unclassified)
        result = repo.scheduler([repo.task(0), repo.task(1)]).run()
        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
        assert "FAILED (target replaced)" in outcome.reason
        assert f"whatever the dispatch itself ended as ({ended!r})" in outcome.reason
        assert f"the dispatch ended {ended}" in outcome.reason
        assert result.exit_code == 3 and len(fake.dispatches) == 1
        assert "plans/p.md" not in outcome.files_touched
        completed = [
            json.loads(line)
            for line in (repo.phase_dir / "subagent_audit.jsonl").read_text().splitlines()
            if '"completed"' in line
        ]
        assert completed and all(
            "plans/p.md" not in row["files_touched"] for row in completed
        )
        assert repo.prior_changes() == []
        # The stop message's restore command, and --all-failed, leave the
        # user's file alone.
        restore_batch.main([
            "--phase-dir", str(repo.phase_dir), "--task-id", "apply-batch-0",
        ])
        restore_batch.main(["--phase-dir", str(repo.phase_dir), "--all-failed"])
        assert repo.plan.read_bytes() == new_bytes

    def test_a_replaced_target_is_ahead_of_the_jail_denial_verdict(
        self, repo, monkeypatch
    ):
        def swapped_and_denied(r: Repo):
            temp = r.root / "plans" / "p.md.swap"
            temp.write_text("# new\n", encoding="utf-8")
            os.replace(temp, r.plan)
            return claim("applied"), stream(failed_file_change(str(r.plan)))

        FakeExecutor(repo, monkeypatch).script(swapped_and_denied)
        outcome = repo.scheduler([repo.task(0)]).run().outcomes[0]
        assert "FAILED (target replaced)" in outcome.reason

    def test_a_swap_on_codex_own_sandbox_is_not_judged_by_inode(
        self, repo, monkeypatch
    ):
        """Only the rung binds a grant to an inode: elsewhere an editor's
        rename-save is an ordinary write and the bytes decide."""
        def swapped(r: Repo):
            temp = r.root / "plans" / "p.md.swap"
            temp.write_text("# rewritten by rename\n", encoding="utf-8")
            os.replace(temp, r.plan)
            return claim("applied"), stream()

        FakeExecutor(repo, monkeypatch, rung=None).script(swapped)
        outcome = repo.scheduler([repo.task(0)]).run().outcomes[0]
        assert outcome.write_outcome == executor.WRITE_APPLIED

    @pytest.mark.parametrize("first", ["sed", "swap"])
    def test_resume_redispatches_under_the_same_grant(
        self, repo, monkeypatch, first
    ):
        """No silent widening: after the sed-replacement and the replaced-target
        variants, --resume re-dispatches with the same request — one file rule
        on the plan path, never `plans/` or another plan."""
        def swapped(r: Repo):
            temp = r.root / "plans" / "p.md.swap"
            temp.write_text("# swapped\n", encoding="utf-8")
            os.replace(temp, r.plan)
            return claim("applied"), stream()

        opening = swapped if first == "swap" else untouched(
            claim("applied"),
            command_denied("sed: couldn't open temporary file plans/sedAbc123: "
                           "Permission denied\n"),
        )
        fake = FakeExecutor(repo, monkeypatch).script(opening, edit_in_place())
        first_run = repo.scheduler([repo.task(0)]).run()
        assert first_run.exit_code == 3

        second_run = repo.scheduler([repo.task(0)]).run()
        assert second_run.exit_code == 0, second_run.messages
        assert not second_run.outcomes[0].skipped
        assert second_run.outcomes[0].write_outcome == executor.WRITE_APPLIED

        requests = [d["jail_request"] for d in fake.dispatches]
        assert len(requests) == 2 and requests[0] == requests[1]
        assert requests[0].target == str(repo.plan)
        assert requests[0].target_dir is None
        assert requests[0].write_scope == "plan-file"
        # The rendered allow list names the plan file and nothing else in the
        # repo, on both dispatches.
        argvs = [
            json.loads(line)["argv"]
            for line in (repo.phase_dir / "subagent_audit.jsonl").read_text().splitlines()
            if '"dispatched"' in line
        ]
        assert len(argvs) == 2
        for argv in argvs:
            allows = [argv[i + 1] for i, a in enumerate(argv) if a == "--allow"]
            in_repo = [a for a in allows if str(repo.root) in a]
            assert in_repo == [f"file:{repo.plan}"], allows


    def test_restore_then_resume_is_not_a_replaced_target(self, repo, monkeypatch):
        """§8.3 item 14: the partial application is restored with
        `util restore-batch` (whose `os.replace` leaves a new inode), then
        --resume re-dispatches the batch once. The identity compared after the
        child is the one recorded before THIS dispatch, so the restore does not
        read as a replaced target, and the edit appears once."""
        from utils import restore_batch

        def partial(r: Repo):
            with open(r.plan, "a", encoding="utf-8") as handle:
                handle.write("\nbatch edit\n")
            return claim("applied"), stream(
                failed_file_change(str(r.plan), str(r.root / "src" / "b.md"))
            )

        original = repo.plan.read_bytes()
        fake = FakeExecutor(repo, monkeypatch).script(
            partial, edit_in_place(text="\nbatch edit\n"),
        )
        first = repo.scheduler([repo.task(0)]).run()
        assert first.outcomes[0].write_outcome == executor.WRITE_PARTIAL
        inode_before_restore = os.lstat(repo.plan).st_ino
        assert restore_batch.main([
            "--phase-dir", str(repo.phase_dir), "--task-id", "apply-batch-0",
        ]) == 0
        assert repo.plan.read_bytes() == original
        assert os.lstat(repo.plan).st_ino != inode_before_restore

        second = repo.scheduler([repo.task(0)]).run()
        assert second.exit_code == 0, second.messages
        assert second.outcomes[0].write_outcome == executor.WRITE_APPLIED
        assert repo.plan.read_text().count("batch edit") == 1
        assert len(fake.dispatches) == 2


# ===========================================================================
# The dry run renders the rung, creating nothing
# ===========================================================================

class TestTheDryRunRendersTheRung:

    def test_the_dry_run_names_the_rung_and_the_wrapped_shape(
        self, repo, monkeypatch
    ):
        fake = FakeExecutor(repo, monkeypatch)
        result = repo.scheduler([repo.task(0)], dry_run=True).run()
        assert result.dry_run_path
        entry = json.loads(Path(result.dry_run_path).read_text())["tasks"][0]
        assert entry["sandbox_rung"] == RUNG
        assert any(str(a).endswith("landlock_exec.py") for a in entry["argv"])
        assert fake.dispatches == []
        assert fake.preflights[0]["jail_request"].write_scope == "plan-file"


# ===========================================================================
# I11: a batch-0 refusal leaves nothing in flight (§5, step 5)
# ===========================================================================

class TestABatchZeroRefusalLeavesNothingInFlight:

    def _refuse(self, repo, monkeypatch):
        FakeExecutor(repo, monkeypatch)

        def refusing_preflight(spec, **kwargs):
            raise executor.RunnerRejected(
                F6_LINE + " Codex's own sandbox cannot start here; see doctor.",
                key="driver.subagent_runner", spec=spec,
                reason_code="sandbox_unavailable",
            )

        monkeypatch.setattr(wbs, "preflight_runner", refusing_preflight)
        return repo.scheduler([repo.task(0), repo.task(1)]).run()

    def test_status_json_lists_no_batch_in_flight_and_names_the_refusal(
        self, repo, monkeypatch
    ):
        result = self._refuse(repo, monkeypatch)
        assert result.exit_code == 3
        assert result.stopped_at_batch == 0
        status = repo.status()
        mirror = status["progress"]["write_subagents"]
        assert mirror["in_flight"] == []
        assert status["state"] == "runner_rejected"
        assert status["reason_code"] == "sandbox_unavailable"
        assert status["runner_key"] == "driver.subagent_runner"
        assert status["refused_batch"] == 0

        from utils import run_status

        assert "runner_rejected" in run_status.CHECKPOINT_TERMINAL_STATES

    def test_the_batch_zero_report_carries_the_executor_f6_line(
        self, repo, monkeypatch
    ):
        result = self._refuse(repo, monkeypatch)
        joined = "\n".join(result.messages)
        assert "batch 0 was refused before any child started" in joined
        assert f"  - {F6_LINE}" in joined
        assert "Nothing was dispatched and no file was modified." in joined


# ===========================================================================
# The apply base hands the scheduler its artifact (§7.1 "Carrying the grant")
# ===========================================================================

class TestTheApplyBaseHandsOverItsArtifact:
    """``apply_orchestrator_base.run_write_executor`` passes
    ``write_target=write_phase_artifact()`` for a single-target phase, none for
    apply-code-fixes, and turns a target the scheduler refuses into exit 3."""

    def _orchestrator(self, tmp_path, monkeypatch, *, single=True):
        from tests.test_apply_orchestrator_base import (
            MockApplyOrchestrator,
            _make_args,
            _setup_plan_with_output,
        )
        from utils import write_phase_bridge

        plan = _setup_plan_with_output(tmp_path)
        orch = MockApplyOrchestrator(
            _make_args(plan_file=str(plan), exec_subagents=True)
        )
        orch._setup()
        orch.write_kind = "apply_suggestions" if single else "apply_code_fixes"
        orch.has_single_write_target = single
        orch.batches = [{"prompt": "x {prior_changes_context}"}]
        monkeypatch.setattr(orch, "_record_pre_existing_changes", lambda: None)
        monkeypatch.setattr(write_phase_bridge, "materialize_write_tasks",
                            lambda **kwargs: [])
        monkeypatch.setattr(write_phase_bridge, "write_task_file_for_phase",
                            lambda **kwargs: str(tmp_path / "write_tasks.json"))
        return orch, plan

    def _record_construction(self, monkeypatch):
        seen: Dict[str, Any] = {}

        class _Result:
            messages: List[str] = []
            exit_code = 0
            report_path = None

        class _Recorder:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def run(self):
                return _Result()

        monkeypatch.setattr(wbs, "WriteBatchScheduler", _Recorder)
        return seen

    def test_a_single_target_phase_passes_its_artifact(self, tmp_path, monkeypatch):
        orch, plan = self._orchestrator(tmp_path, monkeypatch)
        seen = self._record_construction(monkeypatch)
        assert orch.run_write_executor() == 0
        assert seen["write_target"] == orch.write_phase_artifact() == str(plan)

    def test_apply_code_fixes_passes_none(self, tmp_path, monkeypatch):
        orch, _ = self._orchestrator(tmp_path, monkeypatch, single=False)
        seen = self._record_construction(monkeypatch)
        assert orch.run_write_executor() == 0
        assert seen["write_target"] is None

    def test_a_target_the_scheduler_refuses_is_exit_3(
        self, tmp_path, monkeypatch, capsys
    ):
        orch, plan = self._orchestrator(tmp_path, monkeypatch)
        real = plan.with_name("real-plan.md")
        plan.rename(real)
        plan.symlink_to(real)
        assert orch.run_write_executor() == 3
        err = capsys.readouterr().err
        assert "ERROR:" in err and "symlink" in err


class TestTheJailLinesAreRepeatedAtThePhasesClose:
    """Plan F1 item 6 / Limits: a write batch dispatches through
    ``dispatch_task`` directly, so ``run_tasks``' closing summary never runs
    for it — the scheduler repeats this phase's ``[JAIL_DENIED]`` and
    ``[CODEX_HOME_CHECK]`` lines at its own close (2026-09-30)."""

    def test_a_home_check_line_printed_mid_phase_is_repeated_at_the_close(
        self, repo, monkeypatch
    ):
        from utils import subagent_executor as executor

        monkeypatch.setattr(executor, "_JAIL_SUMMARY_LINES", ["[CODEX_HOME_CHECK] an earlier phase"])
        line = "[CODEX_HOME_CHECK] codex:m: config.toml (mcp_servers.x) changed during this dispatch"

        def home_changed(repo_: Repo):
            executor._record_jail_line(line, echo=False)
            return edit_in_place()(repo_)

        FakeExecutor(repo, monkeypatch).script(home_changed)
        lines: List[str] = []
        result = repo.scheduler([repo.task(0)], lines=lines).run()
        assert result.exit_code == 0, result.messages
        closing = [ln for ln in lines if "closing summary:" in ln]
        # Only this phase's line, never an earlier phase's.
        assert closing == [f"[write-subagents] closing summary: {line}"]
