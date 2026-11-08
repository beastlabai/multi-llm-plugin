"""The plan's task-list marker has one reader: ``utils/tasks_file.py``.

``<!-- TASKS_FILE: tasks/tasks.md -->`` tells every phase where the task list
is, and a plan with no marker is its own task list (the legacy embedded
format). A reader that misses a marker therefore parses the plan's prose into
tasks, so every phase reads it with one pattern and follows it with one
resolver (docs/development/workflows.md, "Task generation").
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import plan_updater  # noqa: E402
from utils.task_decomposer import decompose_plan_file  # noqa: E402
from utils.tasks_file import (  # noqa: E402
    TASKS_FILE_MARKER_PATTERN,
    TasksFileAmbiguous,
    TasksFileMissing,
    TasksFileRefused,
    find_tasks_file,
    read_task_list,
    resolve_tasks_marker,
    tasks_marker,
    tasks_marker_spans,
)

SKILL_DIR = Path(__file__).resolve().parent.parent

PROSE = (
    "# Rendering plan\n\n"
    "## 1. Overview\nWhat the renderer does.\n\n"
    "### 2.1 Rendering\nCall camera.update once per frame.\n\n"
    "{marker}\n"
)
TASKS = "# Tasks\n\n### T001: The real task\nRun the checks.\n- Depends on: none\n"

#: Markers that name nothing, however the comment is spaced: each is a marker.
EMPTY = [
    "<!-- TASKS_FILE:  -->",
    "<!--TASKS_FILE:-->",
    "<!-- TASKS_FILE:-->",
    "<!--TASKS_FILE:\n-->",
]

REFORMATTED = [
    "<!-- TASKS_FILE: tasks/tasks.md -->",
    "<!--TASKS_FILE: tasks/tasks.md-->",
    "<!--   TASKS_FILE:   tasks/tasks.md   -->",
    "<!--\nTASKS_FILE: tasks/tasks.md\n-->",
]


@pytest.fixture
def plan_dir(tmp_path):
    directory = tmp_path / "plans"
    (directory / "tasks").mkdir(parents=True)
    (directory / "tasks" / "tasks.md").write_text(TASKS, encoding="utf-8")
    (tmp_path / "outside.md").write_text(TASKS, encoding="utf-8")
    return directory


def _plan(plan_dir, marker):
    plan = plan_dir / "render.md"
    plan.write_text(PROSE.format(marker=marker), encoding="utf-8")
    return plan


class TestTasksMarker:
    @pytest.mark.parametrize("marker", REFORMATTED)
    def test_every_spelling_names_the_same_path(self, marker):
        assert tasks_marker(PROSE.format(marker=marker)) == "tasks/tasks.md"

    def test_no_marker_is_none(self):
        assert tasks_marker(PROSE.format(marker="")) is None

    @pytest.mark.parametrize("marker", EMPTY)
    def test_a_marker_naming_nothing_is_still_a_marker(self, marker):
        assert tasks_marker(PROSE.format(marker=marker)) == ""

    def test_the_plan_updater_reads_with_the_same_pattern(self):
        assert plan_updater.TASKS_FILE_REFERENCE_PATTERN == TASKS_FILE_MARKER_PATTERN
        for marker in REFORMATTED:
            assert plan_updater.get_tasks_file_path(marker) == "tasks/tasks.md"
            assert plan_updater.has_implementation_tasks(marker)

    def test_the_canonical_marker_the_plan_updater_writes_round_trips(self):
        content = plan_updater.insert_tasks_file_reference("# Plan\n", "p/tasks/tasks.md")
        assert tasks_marker(content) == "p/tasks/tasks.md"

    def test_no_other_module_spells_the_marker_pattern(self):
        """One pattern: a second copy is how implement drifted from
        review-tasks (a literal single space where the other took `\\s*`)."""
        spelled = re.compile(r"TASKS_FILE:\\s")
        owners = []
        for path in SKILL_DIR.rglob("*.py"):
            rel = path.relative_to(SKILL_DIR)
            if rel.parts[0] in ("tests", ".venv") or "site-packages" in rel.parts:
                continue
            if spelled.search(path.read_text(encoding="utf-8")):
                owners.append(rel.as_posix())
        assert owners == ["utils/tasks_file.py"]


class TestResolveTasksMarker:
    @pytest.mark.parametrize("marker", REFORMATTED)
    def test_follows_the_marker_inside_the_plan_directory(self, plan_dir, marker):
        plan = _plan(plan_dir, marker)
        expected = (plan_dir / "tasks" / "tasks.md").resolve()
        assert resolve_tasks_marker(plan) == expected
        assert find_tasks_file(str(plan)) == str(expected)

    def test_no_marker_resolves_to_none(self, plan_dir):
        assert resolve_tasks_marker(_plan(plan_dir, "")) is None

    @pytest.mark.parametrize("raw, reason", [
        ("../outside.md", "climbs out with .."),
        ("tasks/../../outside.md", "climbs out with .."),
    ])
    def test_refuses_a_path_that_climbs_out(self, plan_dir, raw, reason):
        plan = _plan(plan_dir, f"<!-- TASKS_FILE: {raw} -->")
        with pytest.raises(TasksFileRefused) as refused:
            resolve_tasks_marker(plan)
        assert refused.value.raw_path == raw
        assert reason in refused.value.reason

    def test_refuses_an_absolute_path(self, plan_dir):
        outside = (plan_dir.parent / "outside.md").resolve()
        plan = _plan(plan_dir, f"<!-- TASKS_FILE: {outside} -->")
        with pytest.raises(TasksFileRefused) as refused:
            resolve_tasks_marker(plan)
        assert "absolute" in refused.value.reason

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_refuses_a_symlink_that_leads_out(self, plan_dir):
        (plan_dir / "link.md").symlink_to(plan_dir.parent / "outside.md")
        plan = _plan(plan_dir, "<!-- TASKS_FILE: link.md -->")
        with pytest.raises(TasksFileRefused) as refused:
            resolve_tasks_marker(plan)
        assert "resolves to" in refused.value.reason

    def test_find_tasks_file_keeps_its_exit_one_for_a_refusal(self, plan_dir, capsys):
        plan = _plan(plan_dir, "<!-- TASKS_FILE: ../outside.md -->")
        with pytest.raises(SystemExit) as excinfo:
            find_tasks_file(str(plan))
        assert excinfo.value.code == 1
        assert "ERROR: TASKS_FILE comment contains path traversal (..)" in capsys.readouterr().err


class TestReadTaskList:
    @pytest.mark.parametrize("marker", REFORMATTED)
    def test_a_marked_plan_reads_its_task_list_not_its_prose(self, plan_dir, marker):
        plan = _plan(plan_dir, marker)
        task_list = read_task_list(plan)
        assert task_list.text == TASKS
        assert task_list.path == (plan_dir / "tasks" / "tasks.md").resolve()

    def test_a_plan_with_no_marker_is_its_own_task_list(self, plan_dir):
        plan = _plan(plan_dir, "")
        task_list = read_task_list(plan)
        assert task_list.path is None
        assert task_list.text == plan.read_text(encoding="utf-8")

    def test_a_missing_task_list_raises_rather_than_reading_the_plan(self, plan_dir):
        plan = _plan(plan_dir, "<!-- TASKS_FILE: gone/tasks.md -->")
        with pytest.raises(TasksFileMissing) as missing:
            read_task_list(plan)
        assert isinstance(missing.value, FileNotFoundError)
        assert missing.value.raw_path == "gone/tasks.md"

    @pytest.mark.parametrize("marker", EMPTY)
    def test_a_marker_naming_nothing_raises_rather_than_reading_the_plan(
        self, plan_dir, marker
    ):
        with pytest.raises(TasksFileMissing):
            read_task_list(_plan(plan_dir, marker))

    def test_a_refused_marker_raises(self, plan_dir):
        with pytest.raises(TasksFileRefused):
            read_task_list(_plan(plan_dir, "<!-- TASKS_FILE: ../outside.md -->"))


class TestEveryTaskListReaderFollowsTheMarker:
    """The readers besides implement that parse the task list."""

    def test_decompose_plan_file(self, plan_dir):
        plan = _plan(plan_dir, "<!--TASKS_FILE: tasks/tasks.md-->")
        assert list(decompose_plan_file(plan).tasks) == ["T001"]

    def test_the_scheduler_check_task_ids(self, plan_dir):
        from utils.write_batch_scheduler import _check_task_ids_of_plan

        plan = _plan(plan_dir, "<!--TASKS_FILE: tasks/tasks.md-->")
        assert _check_task_ids_of_plan(plan) == {"T001"}

    def test_the_scheduler_names_no_check_for_a_refused_marker(self, plan_dir):
        from utils.write_batch_scheduler import _check_task_ids_of_plan

        plan = _plan(plan_dir, "<!-- TASKS_FILE: ../outside.md -->")
        assert _check_task_ids_of_plan(plan) == set()


#: Ways a plan quotes the marker syntax as code: none of them is its marker.
QUOTED = [
    "```\n<!-- TASKS_FILE: example.md -->\n```",
    "```markdown\n<!--TASKS_FILE: ../../etc/passwd-->\n```",
    "~~~\n<!-- TASKS_FILE: example.md -->\n~~~",
    "  ````\n  ```\n  <!-- TASKS_FILE: example.md -->\n  ```\n  ````",
    "Implement reads `<!-- TASKS_FILE: example.md -->` first.",
    "Spelled ``<!--TASKS_FILE: `example`.md-->`` by hand.",
    "```\n<!-- TASKS_FILE: example.md\n```",
]


class TestAMarkerQuotedAsCodeDoesNotCount:
    """A plan that documents the marker (a fenced block, inline code) carries
    examples, not its marker: the first match used to win, so an example
    above the real marker sent every reader to the example's file."""

    def _plan(self, plan_dir, text):
        (plan_dir / "example.md").write_text(
            "# Example\n\n### T999: Not this plan's task\nx\n", encoding="utf-8"
        )
        plan = plan_dir / "render.md"
        plan.write_text(text, encoding="utf-8")
        return plan

    @pytest.mark.parametrize("quoted", QUOTED)
    @pytest.mark.parametrize("where", ["before", "after"])
    def test_the_real_marker_wins_over_an_example(self, plan_dir, quoted, where):
        real = "## Implementation Tasks\n\n<!-- TASKS_FILE: tasks/tasks.md -->\n"
        parts = [PROSE.format(marker=""), quoted + "\n\n", real]
        if where == "after":
            parts[1], parts[2] = real + "\n", quoted + "\n"
        plan = self._plan(plan_dir, "".join(parts))
        assert tasks_marker(plan.read_text(encoding="utf-8")) == "tasks/tasks.md"
        task_list = read_task_list(plan)
        assert task_list.path == (plan_dir / "tasks" / "tasks.md").resolve()
        assert task_list.text == TASKS
        assert find_tasks_file(str(plan)) == str(task_list.path)

    @pytest.mark.parametrize("quoted", QUOTED)
    def test_a_plan_whose_only_marker_is_quoted_has_none(self, plan_dir, quoted):
        plan = self._plan(plan_dir, PROSE.format(marker=quoted))
        assert tasks_marker(plan.read_text(encoding="utf-8")) is None
        assert read_task_list(plan).path is None

    @pytest.mark.parametrize("before", [
        "```\nnever closed\n",
        "A stray ` backtick in the paragraph above.\n",
        "Text `with a span\n",
    ])
    def test_an_unclosed_fence_or_backtick_never_hides_the_marker(self, plan_dir, before):
        """Erring on the marker's side: missing it parses the plan's prose."""
        plan = self._plan(
            plan_dir, PROSE.format(marker="") + before
            + "<!-- TASKS_FILE: tasks/tasks.md -->\n"
        )
        assert tasks_marker(plan.read_text(encoding="utf-8")) == "tasks/tasks.md"


class TestMarkersThatDisagree:
    """Two markers that count and name different files: which list is current
    is the author's call, so every reader refuses rather than pick one."""

    def test_two_markers_naming_different_files_are_refused(self, plan_dir):
        plan = _plan(
            plan_dir,
            "<!-- TASKS_FILE: tasks/tasks.md -->\n\n<!--TASKS_FILE: other.md-->",
        )
        with pytest.raises(TasksFileAmbiguous) as refused:
            read_task_list(plan)
        assert isinstance(refused.value, TasksFileRefused)
        assert refused.value.raw_paths == ["tasks/tasks.md", "other.md"]
        assert "different task lists" in refused.value.reason

    def test_find_tasks_file_refuses_them_with_exit_one(self, plan_dir, capsys):
        plan = _plan(
            plan_dir, "<!-- TASKS_FILE: tasks/tasks.md -->\n<!-- TASKS_FILE: other.md -->"
        )
        with pytest.raises(SystemExit) as excinfo:
            find_tasks_file(str(plan))
        assert excinfo.value.code == 1
        assert "naming different task lists" in capsys.readouterr().err

    def test_two_spellings_of_one_file_are_one_marker(self, plan_dir):
        plan = _plan(
            plan_dir,
            "<!-- TASKS_FILE: tasks/tasks.md -->\n<!--TASKS_FILE: ./tasks/tasks.md-->",
        )
        assert read_task_list(plan).path == (plan_dir / "tasks" / "tasks.md").resolve()

    def test_a_refused_second_marker_is_refused(self, plan_dir):
        plan = _plan(
            plan_dir,
            "<!-- TASKS_FILE: tasks/tasks.md -->\n<!-- TASKS_FILE: ../outside.md -->",
        )
        with pytest.raises(TasksFileRefused) as refused:
            read_task_list(plan)
        assert "climbs out" in refused.value.reason


class TestTheWriterRewritesTheMarkersReadersFollow:
    """`plan_updater.insert_tasks_file_reference` uses the readers' rule:
    every marker that counts names the new list afterwards, and an example
    quoted as code is left as it was."""

    def test_an_example_is_left_alone_and_the_real_marker_rewritten(self):
        example = "```\n<!-- TASKS_FILE: example.md -->\n```\n"
        inline = "Write `<!--TASKS_FILE: x.md-->` by hand.\n"
        plan = plan_updater.insert_tasks_file_reference(
            "# Plan\n\n" + example + "\n" + inline, "p/tasks/old.md"
        )
        rewritten = plan_updater.insert_tasks_file_reference(plan, "p/tasks/tasks.md")
        assert example in rewritten and inline in rewritten
        assert tasks_marker(rewritten) == "p/tasks/tasks.md"
        assert "[p/tasks/tasks.md](p/tasks/tasks.md)" in rewritten
        assert "old.md" not in rewritten
        assert rewritten.count("## Implementation Tasks") == 1

    def test_a_plan_whose_only_marker_is_quoted_gets_its_section(self):
        content = "# Plan\n\n```\n<!-- TASKS_FILE: example.md -->\n```\n"
        written = plan_updater.insert_tasks_file_reference(content, "p/tasks/tasks.md")
        assert written.startswith(content)
        assert tasks_marker(written) == "p/tasks/tasks.md"

    @pytest.mark.parametrize("stray", [
        "<!-- TASKS_FILE: p/../p/tasks/tasks.md -->",
        "<!--TASKS_FILE: stale/tasks.md-->",
        "<!--TASKS_FILE:-->",
    ])
    def test_a_marker_outside_the_section_is_rewritten_in_place(self, stray):
        """A hand-placed marker under the title used to survive regeneration
        beside the rewritten section, for implement to refuse."""
        with_section = plan_updater.insert_tasks_file_reference(
            "# Plan\n\n## Design\n\nBody.\n", "p/tasks/old.md"
        )
        content = with_section.replace("# Plan\n", f"# Plan\n{stray}\n", 1)
        rewritten = plan_updater.insert_tasks_file_reference(content, "p/tasks/tasks.md")
        assert rewritten.startswith(
            "# Plan\n<!-- TASKS_FILE: p/tasks/tasks.md -->\n\n## Design\n\nBody.\n"
        )
        assert [raw for _, _, raw in tasks_marker_spans(rewritten)] == [
            "p/tasks/tasks.md", "p/tasks/tasks.md",
        ]
        assert "[p/tasks/tasks.md](p/tasks/tasks.md)" in rewritten
        assert "old.md" not in rewritten

    def test_a_lone_marker_outside_the_section_is_rewritten_in_place(self):
        content = "# Plan\n<!-- TASKS_FILE: p/../p/tasks/tasks.md -->\n\nBody.\n"
        rewritten = plan_updater.insert_tasks_file_reference(content, "p/tasks/tasks.md")
        assert rewritten == "# Plan\n<!-- TASKS_FILE: p/tasks/tasks.md -->\n\nBody.\n"
