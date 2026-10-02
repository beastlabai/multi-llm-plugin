"""``util check-tasks`` and the apply-task-suggestions consistency hook (issue 10).

The 2026-10-01 Claude Code run left tasks.md with T001 depending on a T012 that
an edit batch had appended last. Every id was declared and nothing was cyclic,
so only the ordering rule catches it.
"""

import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.tasks_graph_check import main, tasks_graph_warnings  # noqa: E402


def _task(task_id, depends_on=()):
    deps = ", ".join(depends_on) if depends_on else "none"
    return (
        f"### Task {task_id}: Do {task_id}\n\n"
        f"**Depends on**: {deps}\n\n- [ ] done\n\n"
    )


def _doc(*blocks):
    return "# Implementation Tasks\n\n" + "".join(blocks)


class TestTasksGraphWarnings:
    def test_a_consistent_graph_has_no_warnings(self):
        text = _doc(_task("T001"), _task("T002", ["T001"]), _task("T003", ["T001", "T002"]))
        assert tasks_graph_warnings(text) == []

    def test_a_dependency_on_a_later_listed_task_is_reported(self):
        """Exactly the run's shape: T012 appended last, T001 depends on it."""
        text = _doc(_task("T001", ["T012"]), _task("T002", ["T001"]), _task("T012"))
        warnings = tasks_graph_warnings(text)
        assert any("T001 depends on T012, listed after it" in w for w in warnings)

    def test_an_undeclared_dependency_is_reported(self):
        text = _doc(_task("T001"), _task("T002", ["T009"]))
        assert any("T009" in w for w in tasks_graph_warnings(text))

    def test_a_cycle_is_reported(self):
        text = _doc(_task("T001", ["T002"]), _task("T002", ["T001"]), _task("T003"))
        assert tasks_graph_warnings(text)

    def test_text_with_no_tasks_is_not_judged(self):
        assert tasks_graph_warnings("# nothing here\n") == []


class TestMain:
    def test_prints_warnings_and_still_exits_zero(self, tmp_path, capsys):
        path = tmp_path / "tasks.md"
        path.write_text(_doc(_task("T001", ["T002"]), _task("T002")), encoding="utf-8")
        assert main([str(path)]) == 0
        assert "[TASKS_GRAPH_WARNING] T001 depends on T002" in capsys.readouterr().out

    def test_ok_line_when_consistent(self, tmp_path, capsys):
        path = tmp_path / "tasks.md"
        path.write_text(_doc(_task("T001"), _task("T002", ["T001"])), encoding="utf-8")
        assert main([str(path)]) == 0
        assert "[TASKS_GRAPH_OK]" in capsys.readouterr().out

    def test_an_unreadable_file_is_exit_2(self, tmp_path):
        assert main([str(tmp_path / "missing.md")]) == 2

    def test_it_is_a_registered_util_command(self):
        from multi_llm import UTIL_COMMANDS
        assert UTIL_COMMANDS["check-tasks"][0] == "utils.tasks_graph_check"


class TestApplyTaskSuggestionsHook:
    def test_the_orchestrator_reads_tasks_md_and_reports_a_forward_dependency(self, tmp_path):
        from apply_task_suggestions_orchestrator import ApplyTaskSuggestionsOrchestrator

        tasks = tmp_path / "tasks.md"
        tasks.write_text(_doc(_task("T001", ["T002"]), _task("T002")), encoding="utf-8")
        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        orch.tasks_file = str(tasks)
        assert any("T001 depends on T002" in w for w in orch.consistency_warnings())

    def test_an_unreadable_tasks_file_is_no_warning(self, tmp_path):
        from apply_task_suggestions_orchestrator import ApplyTaskSuggestionsOrchestrator

        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        orch.tasks_file = str(tmp_path / "gone.md")
        assert orch.consistency_warnings() == []

    def test_the_batch_prompt_tells_each_batch_to_keep_the_graph_consistent(self):
        from apply_task_suggestions_orchestrator import ApplyTaskSuggestionsOrchestrator
        from utils.suggestion_batcher import SuggestionBatch

        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        orch.tasks_file, orch.plan_path = "/p/tasks.md", "/p/plan.md"
        batch = SuggestionBatch(
            batch_type="single", section_key="T005",
            suggestions=[{"title": "t", "type": "addition", "description": "d"}],
        )
        prompt = orch._generate_single_batch_prompt(batch)
        assert "Consistency across tasks" in prompt
        assert "never leave two tasks that both define it" in prompt
