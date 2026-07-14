"""Tests for the post-apply human/Claude decision overlay in HTML reports.

Covers the per-item "Let Claude decide" / --claude-decide decisions surfacing
in the regenerated HTML review report:
  - normalization of recorded state.json decision records
  - overlaying decisions onto report_data groups (joined by group_hash)
  - end-to-end report.html regeneration from the persisted report_data.json
  - graceful skip when the report_data.json sidecar is absent
"""

import json
import tempfile
from pathlib import Path

import pytest

from utils.html_report_generator import (
    _normalize_human_decision,
    _overlay_human_decisions,
    _read_human_decisions_from_state,
    regenerate_report_with_human_decisions,
    main as html_report_main,
)


# Recorded-decision fixtures matching references/human-decision-batch.md "State Recording".
SALVAGE_REC = {
    "decision": "approved",
    "reason": "Claude auto-decide (salvaged): kept null-check; dropped retry wrapper",
    "batch_context": {
        "batch_action": "claude_auto_decide_salvage",
        "decision_source": "claude_auto_decide_salvage",
        "salvaged_description": "Add null-check before user.profile",
        "dropped": "speculative retry wrapper",
        "importance_at_decision": "MEDIUM",
    },
}
APPROVE_REC = {
    "decision": "approved",
    "reason": "Claude auto-decide: sound bug fix",
    "batch_context": {
        "batch_action": "claude_auto_decide",
        "decision_source": "claude_auto_decide",
        "importance_at_decision": "HIGH",
    },
}
SKIP_REC = {
    "decision": "skipped",
    "reason": "Claude auto-decide: purely subjective",
    "batch_context": {
        "batch_action": "claude_auto_decide",
        "decision_source": "claude_auto_decide",
        "importance_at_decision": "LOW",
    },
}
USER_REC = {
    "decision": "approved",
    "reason": "user approved",
    "batch_context": {"decision_source": "user_individual"},
}


class TestNormalizeHumanDecision:
    def test_salvage_outcome_derived_from_marker_not_decision(self):
        # decision == "approved" but the salvage marker must win.
        info = _normalize_human_decision(SALVAGE_REC)
        assert info["outcome"] == "salvaged"
        assert info["claudeDecided"] is True
        assert info["salvagedDescription"] == "Add null-check before user.profile"
        assert info["dropped"] == "speculative retry wrapper"

    def test_approve_outcome(self):
        info = _normalize_human_decision(APPROVE_REC)
        assert info["outcome"] == "approved"
        assert info["claudeDecided"] is True

    def test_skip_outcome(self):
        info = _normalize_human_decision(SKIP_REC)
        assert info["outcome"] == "skipped"
        assert info["claudeDecided"] is True

    def test_user_decision_not_marked_claude(self):
        info = _normalize_human_decision(USER_REC)
        assert info["claudeDecided"] is False
        assert info["outcome"] == "approved"

    def test_missing_batch_context_is_safe(self):
        info = _normalize_human_decision({"decision": "skipped"})
        assert info["outcome"] == "skipped"
        assert info["claudeDecided"] is False
        assert info["salvagedDescription"] == ""


class TestOverlayHumanDecisions:
    def test_overlay_joins_by_group_hash(self):
        report_data = {
            "groups": [
                {"groupHash": "aaa", "validationStatus": "needs-human-decision"},
                {"groupHash": "bbb", "validationStatus": "needs-human-decision"},
                {"groupHash": "ccc", "validationStatus": "valid"},  # no decision
                {"validationStatus": "valid"},  # no hash at all
            ]
        }
        decisions = {"aaa": SALVAGE_REC, "bbb": SKIP_REC}
        matched = _overlay_human_decisions(report_data, decisions)

        assert matched == 2
        assert report_data["groups"][0]["humanDecision"]["outcome"] == "salvaged"
        assert report_data["groups"][1]["humanDecision"]["outcome"] == "skipped"
        assert "humanDecision" not in report_data["groups"][2]
        assert "humanDecision" not in report_data["groups"][3]

    def test_summary_counts(self):
        report_data = {
            "groups": [
                {"groupHash": "a"},
                {"groupHash": "b"},
                {"groupHash": "c"},
            ]
        }
        _overlay_human_decisions(
            report_data, {"a": APPROVE_REC, "b": SALVAGE_REC, "c": SKIP_REC}
        )
        summary = report_data["humanDecisionsSummary"]
        assert summary == {
            "approved": 1,
            "salvaged": 1,
            "skipped": 1,
            "claudeDecided": 3,
            "total": 3,
        }

    def test_empty_decisions_stamps_zero_summary(self):
        report_data = {"groups": [{"groupHash": "a"}]}
        matched = _overlay_human_decisions(report_data, {})
        assert matched == 0
        assert report_data["humanDecisionsSummary"]["total"] == 0


class TestReadHumanDecisionsFromState:
    def test_reads_phase_scoped_key(self):
        with tempfile.TemporaryDirectory() as d:
            sf = Path(d) / "state.json"
            sf.write_text(json.dumps({
                "human_decisions_apply-suggestions": {"aaa": APPROVE_REC},
                "human_decisions_apply-code-fixes": {"zzz": SKIP_REC},
            }), encoding="utf-8")
            got = _read_human_decisions_from_state(sf, "apply-suggestions")
            assert list(got.keys()) == ["aaa"]

    def test_missing_key_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            sf = Path(d) / "state.json"
            sf.write_text(json.dumps({}), encoding="utf-8")
            assert _read_human_decisions_from_state(sf, "apply-suggestions") == {}


class TestRegenerateReport:
    def _sidecar(self, phase_dir: Path):
        report_data = {
            "templateStyle": "pr",
            "phase": "review-plan",
            "groups": [
                {
                    "groupHash": "aaa",
                    "theme": "Null safety",
                    "validationStatus": "needs-human-decision",
                    "suggestions": [],
                }
            ],
        }
        (phase_dir / "report_data.json").write_text(json.dumps(report_data), encoding="utf-8")

    def test_regenerates_html_with_decision_embedded(self):
        with tempfile.TemporaryDirectory() as d:
            phase_dir = Path(d)
            self._sidecar(phase_dir)
            out = regenerate_report_with_human_decisions(
                phase_dir, {"aaa": SALVAGE_REC}
            )
            assert out is not None and out.exists()
            html = out.read_text(encoding="utf-8")
            # The overlaid decision data is embedded in reportData JSON.
            assert "salvaged" in html
            assert "Add null-check before user.profile" in html
            # The sidecar is updated in place so re-runs overlay cleanly.
            rd = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
            assert rd["groups"][0]["humanDecision"]["outcome"] == "salvaged"
            assert rd["humanDecisionsSummary"]["salvaged"] == 1

    def test_missing_sidecar_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            out = regenerate_report_with_human_decisions(
                Path(d), {"aaa": SALVAGE_REC}
            )
            assert out is None

    def test_cli_graceful_skip_when_no_sidecar(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            sf = Path(d) / "state.json"
            sf.write_text(json.dumps({"human_decisions_apply-suggestions": {}}), encoding="utf-8")
            rc = html_report_main([
                "regenerate-decisions",
                "--phase-dir", str(Path(d) / "noreport"),
                "--state-file", str(sf),
                "--apply-phase", "apply-suggestions",
            ])
            assert rc == 0
            assert "No report_data.json" in capsys.readouterr().out

    def test_cli_regenerates_from_state(self):
        with tempfile.TemporaryDirectory() as d:
            phase_dir = Path(d) / "review-plan"
            phase_dir.mkdir()
            self._sidecar(phase_dir)
            sf = Path(d) / "state.json"
            sf.write_text(json.dumps({
                "human_decisions_apply-suggestions": {"aaa": APPROVE_REC},
            }), encoding="utf-8")
            rc = html_report_main([
                "regenerate-decisions",
                "--phase-dir", str(phase_dir),
                "--state-file", str(sf),
                "--apply-phase", "apply-suggestions",
            ])
            assert rc == 0
            html = (phase_dir / "report.html").read_text(encoding="utf-8")
            assert "approved" in html
