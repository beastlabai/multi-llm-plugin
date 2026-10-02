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


class TestDependenciesComeFromTheDependenciesLine:
    """Adversarial review, 2026-10-02: prose read as dependencies.

    The decomposer's dependency reader also matches "after", "requires" and
    "depends on" in the narrative, so a sentence named ids nobody declared.
    """

    TASKS = [
        {"id": "T001", "title": "Setup",
         "description": "Create the scaffold. The loop ticks after 3 frames.",
         "depends_on": [], "acceptance_criteria": ["requires 2 passes to settle"]},
        {"id": "T002", "title": "Physics", "description": "Add physics.",
         "depends_on": ["T001"]},
    ]

    def test_a_rendered_task_list_with_such_prose_has_no_warning(self):
        from update_plan_tasks import generate_tasks_markdown

        markdown = generate_tasks_markdown(self.TASKS, "plan.md")
        assert "ticks after 3 frames" in markdown
        assert tasks_graph_warnings(markdown) == []

    def test_the_rendered_line_is_still_read(self):
        """A forward edge on the `**Dependencies**:` line is still caught."""
        from update_plan_tasks import generate_tasks_markdown

        tasks = [dict(self.TASKS[0], depends_on=["T002"]),
                 dict(self.TASKS[1], depends_on=[])]
        warnings = tasks_graph_warnings(generate_tasks_markdown(tasks, "plan.md"))
        assert any("T001 depends on T002, listed after it" in w for w in warnings)

    def test_a_sentence_naming_a_dependency_is_not_one(self):
        text = _doc(
            "### Task T001: Do T001\n\n**Dependencies**: None\n\n"
            "Run it after T002 has warmed the cache; depends on T009 at runtime.\n\n",
            _task("T002", ["T001"]),
        )
        assert tasks_graph_warnings(text) == []


class TestImplementAndCheckTasksReadOneGraph:
    """Racing run 2026-10-01 (still open after round 1): check-tasks read the
    bold `**Dependencies**:` line only, implement read prose too, so one
    tasks.md was two graphs. Both now call `declared_dependencies`."""

    FIXTURE = _doc(
        # The hand-written format: a forward edge implement orders by.
        "### Task T001: Start\nKick it off.\n- Depends on: T003\n\n",
        # Prose naming a later task: no edge for either reader.
        "### Task T002: Middle\n**Dependencies**: None\n\n"
        "Run it after T003 has warmed the cache; requires 2 passes.\n\n",
        "### Task T003: End\n**Depends on**: none\n\n",
    )

    def test_check_tasks_warns_about_exactly_the_edges_implement_parses(self):
        from utils.task_decomposer import TaskDecomposer

        tasks = TaskDecomposer().parse_plan(self.FIXTURE)
        assert {t.id: t.depends_on for t in tasks} == {
            "T001": ["T003"], "T002": [], "T003": []}
        position = {t.id: i for i, t in enumerate(tasks)}
        forward = [
            f"{t.id} depends on {', '.join(d for d in t.depends_on if position[d] > position[t.id])}"
            for t in tasks
            if any(position.get(d, -1) > position[t.id] for d in t.depends_on)
        ]
        warnings = tasks_graph_warnings(self.FIXTURE)
        assert forward == ["T001 depends on T003"]
        assert [w.split(",")[0] for w in warnings] == forward

    def test_there_is_one_dependency_reader(self):
        import utils.tasks_graph_check as check
        from utils import task_decomposer

        assert check.TaskDecomposer is task_decomposer.TaskDecomposer
        assert not [
            name for name, value in vars(check).items()
            if isinstance(value, type) and issubclass(value, task_decomposer.TaskDecomposer)
            and value is not task_decomposer.TaskDecomposer
        ]


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


class TestTheInstructionNamesAPathTheHostHas:
    """Adversarial review, 2026-10-02: `util check-tasks "{tasks_file}"` was an
    authoring placeholder `render_instruction_text` never substitutes (it knows
    SKILL_DIR, UV_RUN_FLAGS, DEFAULT_MODE and NEEDS_HUMAN_RULE), so the host saw
    the brace token literally."""

    def test_the_check_tasks_line_points_at_orchestrator_output(self):
        from utils.cli_common import render_instruction_text

        text = (SKILL_ROOT / "instructions" / "apply-task-suggestions.md").read_text(
            encoding="utf-8")
        rendered = render_instruction_text(text, "/skill dir")
        (line,) = [l for l in rendered.splitlines() if "util check-tasks" in l]
        assert "{tasks_file}" not in line
        assert line.rstrip().endswith(
            'util check-tasks "<tasks_file from orchestrator_output.json>"')
