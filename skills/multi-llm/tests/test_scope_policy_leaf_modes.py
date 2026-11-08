"""``--scope-policy`` on generate-tasks and implement: accepted, carried, kept.

plans/scope-policy-and-report-tags.md §4: generation and implementation have
no review round, but they follow the accepted plan and tasks and must not
invent outcomes under ``preserve``. Both take the flag (``full`` hands its
resolved one over), add a short preserve-only block to their model prompts
(nothing changes under ``expand``), keep a typed choice for the re-run or
``--resume`` that never repeats it (``state.json`` → ``scope_policy_runs``),
and record a policy nobody typed as D6 row 16's resolution under ``--yes``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import generate_tasks_orchestrator as gto  # noqa: E402
import implement_orchestrator  # noqa: E402
from utils import scope_policy_runs as runs  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

from tests.support.write_orchestrators import (  # noqa: E402
    _seed_generate_tasks,
    orchestrator_repo,  # noqa: F401 - the repo fixture
)

PRESERVE_HEADING = "## Scope Policy: Preserve"


def _state_doc(plan: Path) -> dict:
    return json.loads((output_dir_path(plan) / "state.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# generate-tasks
# ---------------------------------------------------------------------------

class TestGenerateTasks:
    def _materialize(self, tmp_path, policy):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\n## Reply\nA plan heading that is not the reply.\n",
                        encoding="utf-8")
        phase_dir = tmp_path / "plan" / f"generate-tasks-{policy}"
        _task_file, document = gto.materialize_task_file(
            plan, str(phase_dir), repo_root=str(tmp_path), scope_policy=policy)
        return document["tasks"][0]["prompt"]

    def test_force_is_a_new_run_without_the_old_choice(self, repo, capsys):
        phase_dir = str(output_dir_path(repo.plan) / "generate-tasks")
        typed = gto.parse_args(["--plan-file", str(repo.plan), "--scope-policy", "preserve"])
        assert gto._resolve_scope_policy(typed, repo.plan, phase_dir) == "preserve"
        later = gto.parse_args(["--plan-file", str(repo.plan), "--yes"])
        assert gto._resolve_scope_policy(later, repo.plan, phase_dir) == "preserve"
        forced = gto.parse_args(["--plan-file", str(repo.plan), "--force", "--yes"])
        assert gto._resolve_scope_policy(forced, repo.plan, phase_dir) == "expand"
        assert gto._resolve_scope_policy(later, repo.plan, phase_dir) == "expand"

    def test_force_dry_run_keeps_the_live_choice(self, repo):
        phase_dir = str(output_dir_path(repo.plan) / "generate-tasks")
        typed = gto.parse_args(["--plan-file", str(repo.plan), "--scope-policy", "preserve"])
        gto._resolve_scope_policy(typed, repo.plan, phase_dir)
        before = _state_doc(repo.plan)[runs.RUNS_KEY]
        dry = gto.parse_args(["--plan-file", str(repo.plan), "--force", "--dry-run-subagents"])
        assert gto._resolve_scope_policy(dry, repo.plan, phase_dir) == "expand"
        assert _state_doc(repo.plan)[runs.RUNS_KEY] == before

    def test_expand_and_omission_leave_the_prompt_unchanged(self, tmp_path):
        default = self._materialize(tmp_path, None)
        assert self._materialize(tmp_path, "expand") == default
        assert PRESERVE_HEADING not in default

    def test_preserve_adds_the_block_ahead_of_the_reply_section(self, tmp_path):
        default = self._materialize(tmp_path, None)
        prompt = self._materialize(tmp_path, "preserve")
        block = runs.generation_guidance("preserve")
        assert block in prompt
        # Before the template's own reply section, after the embedded plan's
        # same-named heading.
        assert prompt.index(block) > prompt.index("A plan heading that is not the reply")
        assert prompt.index(block) < prompt.rindex("## Reply\n")
        assert "Do not add tasks for features" in prompt
        assert prompt.replace(block + "\n\n", "") == default

    def test_alternating_policies_never_reuse_a_rendering(self, tmp_path):
        first = self._materialize(tmp_path, "preserve")
        second = self._materialize(tmp_path, "expand")
        third = self._materialize(tmp_path, "preserve")
        assert PRESERVE_HEADING in first and PRESERVE_HEADING in third
        assert PRESERVE_HEADING not in second

    def test_a_typed_choice_is_kept_for_the_re_run(self, repo, capsys):
        args = gto.parse_args(["--plan-file", str(repo.plan), "--scope-policy", "preserve"])
        phase_dir = str(output_dir_path(repo.plan) / "generate-tasks")
        assert gto._resolve_scope_policy(args, repo.plan, phase_dir) == "preserve"
        assert runs.live_run_choice(_state_doc(repo.plan), "generate-tasks") == "preserve"
        again = gto.parse_args(["--plan-file", str(repo.plan), "--yes"])
        assert gto._resolve_scope_policy(again, repo.plan, phase_dir) == "preserve"
        assert ("[AUTO_RESOLVED] scope-policy: preserve (kept from this generate-tasks "
                "run's recorded --scope-policy)") in capsys.readouterr().out

    def test_a_completed_generation_answers_nothing(self, repo, capsys):
        args = gto.parse_args(["--plan-file", str(repo.plan), "--scope-policy", "preserve"])
        phase_dir = str(output_dir_path(repo.plan) / "generate-tasks")
        gto._resolve_scope_policy(args, repo.plan, phase_dir)
        manager = StateManager(repo.plan)
        manager.mark_phase_completed("generate-tasks")
        manager.save()
        fresh = gto.parse_args(["--plan-file", str(repo.plan), "--yes"])
        assert gto._resolve_scope_policy(fresh, repo.plan, phase_dir) == "expand"
        assert ("[AUTO_RESOLVED] scope-policy: expand (no --scope-policy given: the "
                "default for a new run)") in capsys.readouterr().out


# ---------------------------------------------------------------------------
# implement
# ---------------------------------------------------------------------------

@pytest.fixture
def handed_out(repo, monkeypatch):
    """Run implement past its detach boundary and capture the batches it would
    schedule (both legs build their write tasks from these prompts)."""
    seen = []

    def capture(args, *, output_data, **kwargs):
        seen.append(json.loads(json.dumps(output_data)))
        return 0

    monkeypatch.setattr(implement_orchestrator, "maybe_detach_implement_phase",
                        lambda *a, **kw: None)
    monkeypatch.setattr(implement_orchestrator, "run_implement_write_phase", capture)
    _seed_generate_tasks(repo)
    return seen


def _prompts(output_data):
    return [task["prompt"] for batch in output_data["batches"] for task in batch["tasks"]
            if not task.get("is_human")]


def _implement(repo, *argv):
    return implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--exec-subagents", *argv])


class TestImplement:
    def test_expand_and_omission_render_the_same_prompts(self, repo, handed_out, capsys):
        assert _implement(repo) == 0
        assert _implement(repo, "--fresh", "--scope-policy", "expand") == 0
        default, explicit = handed_out
        assert _prompts(default) == _prompts(explicit)
        assert all(PRESERVE_HEADING not in prompt for prompt in _prompts(default))

    def test_preserve_adds_the_block_to_every_code_task(self, repo, handed_out, capsys):
        assert _implement(repo, "--scope-policy", "preserve") == 0
        [data] = handed_out
        prompts = _prompts(data)
        assert prompts
        block = runs.implementation_guidance("preserve")
        assert all(prompt.rstrip("\n").endswith(block) for prompt in prompts)
        assert all("{prior_changes_context}" in prompt for prompt in prompts), (
            "the scheduler's late-bound token survives")

    def test_the_override_survives_a_resume_without_the_flag(self, repo, handed_out, capsys):
        assert _implement(repo, "--scope-policy", "preserve") == 0
        record = runs.run_record(_state_doc(repo.plan), "implement")
        assert record["policy"] == "preserve" and record["source"] == "explicit"
        assert _implement(repo, "--resume", "--yes") == 0
        resumed = handed_out[-1]
        assert all(PRESERVE_HEADING in prompt for prompt in _prompts(resumed))
        assert ("[AUTO_RESOLVED] scope-policy: preserve (kept from this implement run's "
                "recorded --scope-policy)") in capsys.readouterr().out

    def test_fresh_is_a_new_run_without_the_old_choice(self, repo, handed_out, capsys):
        assert _implement(repo, "--scope-policy", "preserve") == 0
        assert _implement(repo, "--fresh", "--yes") == 0
        fresh = handed_out[-1]
        assert all(PRESERVE_HEADING not in prompt for prompt in _prompts(fresh))
        assert _implement(repo, "--resume", "--yes") == 0
        assert all(PRESERVE_HEADING not in prompt for prompt in _prompts(handed_out[-1]))

    def test_a_human_task_checklist_is_left_alone(self):
        data = {"batches": [{"tasks": [
            {"id": "T1", "is_human": True, "prompt": "checklist"},
            {"id": "T2", "is_human": False, "prompt": "do it"},
        ]}]}
        implement_orchestrator._apply_scope_guidance(data, "preserve")
        assert data["batches"][0]["tasks"][0]["prompt"] == "checklist"
        assert PRESERVE_HEADING in data["batches"][0]["tasks"][1]["prompt"]
        untouched = {"batches": [{"tasks": [{"id": "T2", "prompt": "do it"}]}]}
        implement_orchestrator._apply_scope_guidance(untouched, "expand")
        assert untouched["batches"][0]["tasks"][0]["prompt"] == "do it"

    def test_an_unattended_default_is_recorded(self, repo, handed_out, capsys):
        assert _implement(repo, "--yes") == 0
        assert ("[AUTO_RESOLVED] scope-policy: expand (no --scope-policy given: the "
                "default for a new run)") in capsys.readouterr().out
        assert runs.run_record(_state_doc(repo.plan), "implement") is None
