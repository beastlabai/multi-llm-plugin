"""The apply phase's closing report and scope gate, built without an orchestrator.

`ApplyPhaseReport` (utils/apply_report.py) reads only the data it is built
with, and `ApplyScopeGate` (utils/scope_apply.py) holds one run's scope
state; both were methods of `ApplyOrchestratorBase`. The orchestrator-level
behaviour stays covered by tests/test_apply_orchestrator_base.py and the
scope-policy tests; these check that each stands on its own.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import scope_policy as sp  # noqa: E402
from utils.apply_report import ApplyPhaseReport, applied_to_label  # noqa: E402
from utils.scope_apply import ApplyPolicy, ApplyScopeGate  # noqa: E402


class _Result:
    """A scheduler result as the report reads it: its rows."""

    def __init__(self, rows):
        self._rows = rows
        self.report_path = None
        self.summary_file = None

    def report_rows(self):
        return self._rows


def _report(tmp_path, **fields):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    base = dict(
        phase_name="apply-suggestions", write_kind="apply_suggestions",
        item_noun="suggestion", item_noun_plural="suggestions",
        plan_path=str(plan), out_dir=str(tmp_path / "out"), target_file=str(plan),
        write_result=_Result([{
            "task_id": "apply-batch-1", "status": "ok", "write_outcome": "applied",
            "expected_ids": ["a", "b"], "runner": "codex", "duration_seconds": 2.0,
            "files_touched": ["plan.md"],
        }]),
    )
    base.update(fields)
    return ApplyPhaseReport(**base)


def test_the_outcome_block_comes_from_the_data_alone(tmp_path, capsys):
    _report(tmp_path).print_write_outcome_summary()
    out = capsys.readouterr().out
    assert "APPLY-SUGGESTIONS: RESULT" in out
    assert "Batches: 1/1 ok" in out
    assert "Suggestions applied: 2" in out
    assert "  apply-batch-1: ok (applied), 2 suggestion(s), 2.0s, runner codex" in out


def test_the_summary_report_reads_the_phase_hooks_it_was_given(tmp_path, capsys):
    report = _report(
        tmp_path,
        consistency_warnings=lambda: ["T2 depends on T3, listed after it"],
        document_title=lambda: "Commit brief",
        resolve_base_ref=lambda: "abc123",
    )
    path = report.write_apply_summary_report()
    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("# Applied Suggestions Report\n")
    assert f"**Commit brief**: {report.plan_path}" in text
    assert "**Base ref**: abc123" in text
    assert "## Consistency warnings\n\n- T2 depends on T3, listed after it" in text
    assert report.run_notes()[0] == "consistency warning: T2 depends on T3, listed after it"


def test_the_touched_files_and_the_applied_to_label(tmp_path):
    report = _report(tmp_path)
    assert report.touched_files() == ["plan.md"]
    assert applied_to_label(None, report.plan_path) is None
    assert applied_to_label(report.target_file, report.plan_path).endswith("plan.md")


class TestTheScopeGate:

    GROUP = {"group_hash": "g" * 16, "theme": "Widen the API",
             "suggestions": [{"title": "Widen the API", "desc": "d"}]}

    def test_a_new_gate_reads_expand_and_passes_everything(self):
        gate = ApplyScopeGate()
        valid, human = [dict(self.GROUP)], []
        gate.gate_work_set(valid, human)
        assert len(valid) == 1 and not gate.deferred and not gate.preserve
        assert gate.output_fields() == {}
        assert gate.materialize_kwargs() == {} and gate.scheduler_kwargs() == {}
        assert gate.summary_lines() == [] and gate.notice("suggestions") is None

    def test_preserve_defers_a_group_without_a_bound_verdict(self, capsys):
        gate = ApplyScopeGate()
        gate.policy = ApplyPolicy(policy=sp.POLICY_PRESERVE, source=sp.SOURCE_EXPLICIT,
                                  review_policy=sp.POLICY_PRESERVE, baseline_id=None)
        valid, human = [dict(self.GROUP)], []
        gate.gate_work_set(valid, human)
        gate.gate_work_set(valid, human)  # idempotent: one entry per group
        assert valid == [] and [e["group_id"] for e in gate.deferred] == ["g" * 16]
        assert capsys.readouterr().err.count("[SCOPE_DEFERRED]") == 1
        assert gate.output_fields()["scope_policy"]["deferred_count"] == 1
        assert gate.summary_lines(would=True)[1].startswith(
            "Would defer by --scope-policy preserve (1, not applied):")
        assert gate.materialize_kwargs() == {"scope": gate.policy}
