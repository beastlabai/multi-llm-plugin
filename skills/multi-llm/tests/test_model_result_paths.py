"""Which files in a review phase dir are a model's result: one rule, every reader.

``utils/result_files.model_result_paths`` (its rule is ``is_model_result_name``)
decides it for both reaggregates, review-code's partial-completion guard
(``leftover_model_results``, which ``improve_workflow`` reads too) and
``retire_unselected_results``; the slot's ``snapshot_candidates`` uses the same
rule. Each reader once carried its own list, and review-code's guard list lacked
the executor's and write scheduler's reports, so a dir holding only
``subagent_batch_report.json`` read as "Found 1 per-model result file(s)".
"""

import ast
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
from utils import result_files  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402

#: Every non-model ``*.json`` a review phase dir can hold.
BOOKKEEPING = (
    "grouped.json", "validation.json", "validation_tasks.json", "validation_batch_0.json",
    "revalidation.json", "revalidation_tasks.json", "salvage_codex_x.json",
    "consolidation_tasks.json", "consolidation_batch_0.json", "consolidated.json",
    "consolidated_report_data.json", "issues.json", "fix_tasks.json", "report_data.json",
    "subagent_batch_report.json", "subagent_dry_run.json", "write_batch_report.json",
    "write_tasks.json", "state.json", ".status.json",
)

#: The executor's and the write scheduler's reports: objects, not result lists.
EXECUTOR_REPORTS = (
    "subagent_batch_report.json", "subagent_dry_run.json",
    "write_batch_report.json", "write_tasks.json",
)


def _phase_dir(tmp_path, names, models=("codex_gpt-6-luna_high", "gemini_gemini-3-pro")):
    phase_dir = tmp_path / "code-review"
    phase_dir.mkdir()
    for stem in models:
        (phase_dir / f"{stem}.json").write_text("[]", encoding="utf-8")
    for name in names:
        (phase_dir / name).write_text("{}", encoding="utf-8")
    (phase_dir / "failed").mkdir()
    (phase_dir / "subagent_audit.jsonl").write_text("{}\n", encoding="utf-8")
    return phase_dir


class TestOneRule:
    def test_only_the_model_files_are_results(self, tmp_path):
        phase_dir = _phase_dir(tmp_path, BOOKKEEPING)
        assert [Path(p).name for p in result_files.model_result_paths(phase_dir)] == [
            "codex_gpt-6-luna_high.json", "gemini_gemini-3-pro.json",
        ]

    def test_paths_are_spelled_like_a_glob_of_the_dir(self, tmp_path):
        phase_dir = _phase_dir(tmp_path, ())
        assert result_files.model_result_paths(phase_dir) == sorted(
            str(p) for p in phase_dir.glob("*.json")
        )
        assert result_files.model_result_paths(str(phase_dir) + "/") == [
            str(phase_dir) + "/" + name
            for name in ("codex_gpt-6-luna_high.json", "gemini_gemini-3-pro.json")
        ]

    def test_a_missing_dir_has_none(self, tmp_path):
        assert result_files.model_result_paths(tmp_path / "absent") == []

    def test_a_model_id_containing_state_is_still_a_model(self, tmp_path):
        """``state.json`` is excluded by its whole name, not as a fragment."""
        phase_dir = _phase_dir(tmp_path, ("state.json",), models=("openrouter_stateful-7b",))
        assert [Path(p).name for p in result_files.model_result_paths(phase_dir)] == [
            "openrouter_stateful-7b.json",
        ]

    def test_the_slot_reads_the_same_names(self, tmp_path):
        phase_dir = _phase_dir(tmp_path, BOOKKEEPING)
        assert sorted(result_files.snapshot_candidates(phase_dir)) == [
            Path(p).name for p in result_files.model_result_paths(phase_dir)
        ]


class TestTheGuardIgnoresExecutorReports:
    """review-code's partial-completion guard and ``improve_workflow`` read
    ``leftover_model_results``; an executor report alone is no fan-out result."""

    @pytest.mark.parametrize("name", EXECUTOR_REPORTS)
    def test_a_dir_with_only_a_report_has_no_leftover_results(self, tmp_path, name):
        phase_dir = _phase_dir(tmp_path, (name,), models=())
        assert cro.leftover_model_results(phase_dir) == []

    def test_the_guard_matches_the_shared_rule(self, tmp_path):
        phase_dir = _phase_dir(tmp_path, BOOKKEEPING)
        assert cro.leftover_model_results(phase_dir) == result_files.model_result_paths(phase_dir)


class TestRetireReadsTheSameNames:
    def test_a_bookkeeping_file_named_like_a_respelling_is_never_retired(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        (phase_dir / "codex_gpt-6-luna.json").write_text("[]", encoding="utf-8")
        (phase_dir / "codex_gpt-6-luna_validation.json").write_text("[]", encoding="utf-8")
        retired = rob.retire_unselected_results(str(phase_dir), ["codex:gpt-6-luna:high"])
        assert [Path(p).name for p in retired] == ["codex_gpt-6-luna.json.stale-unselected"]
        assert (phase_dir / "codex_gpt-6-luna_validation.json").exists()


class TestNoPrivateList:
    """A reader that keeps its own list drifts: the lists differed in five ways."""

    def test_no_module_spells_its_own_exclusion_list(self):
        found = []
        for path in sorted(SKILL_ROOT.rglob("*.py")):
            rel = path.relative_to(SKILL_ROOT)
            if rel.parts[0] in ("tests", ".venv") or rel == Path("utils/result_files.py"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                    values = {
                        e.value for e in node.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    }
                    if {"grouped", "salvage"} <= values:
                        found.append(f"{rel}:{node.lineno}")
        assert found == [], (
            "use utils/result_files.model_result_paths instead of a private list: "
            + ", ".join(found)
        )

    def test_the_status_file_is_never_a_result(self, tmp_path):
        phase_dir = tmp_path / "p"
        phase_dir.mkdir()
        (phase_dir / ".status.json").write_text(json.dumps({"models_requested": []}))
        assert result_files.model_result_paths(phase_dir) == []
