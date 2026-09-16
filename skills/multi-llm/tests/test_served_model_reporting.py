"""§3.6 (task L7) — `served_model` travels from the envelope to every report.

The adapter half (``utils/providers/claude_code.py``) is covered by
``tests/test_providers.py::TestClaudeCodeServedModel``. This file covers the
*carriage*: adapter → ``invoke_with_provider`` → ``invoke_with_file_output`` →
the fan-out result records → the markdown/HTML surfaces that name a model.

Two properties are asserted throughout:

1. **A fallback that fired is never attributed to the primary.** One WARN per
   slot, the served model recorded on the result, and every report row that
   names a model labelled with it.
2. **Green stays green.** ``fallback_models`` ships empty, so with nothing
   configured the details dict, the fan-out result dict, ``report.md`` and
   ``report_data.json`` are exactly what they were.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro
from utils.html_report_generator import generate_html_report
from utils.llm_client import invoke_with_provider, invoke_with_file_output
from utils.providers.claude_code import ClaudeCodeProvider
from utils.review_orchestrator_base import aggregate_results, run_single_model

SERVED = "claude-haiku-4-5-20251001"
ISSUES = [{
    "title": "Boundary check missing",
    "desc": "d",
    "importance": "high",
    "file": "a.py",
    "line_range": [1, 2],
    "type": "bug",
    "anchor_text": "x",
}]


def _envelope(model_usage=None, result=None):
    envelope = {
        "type": "result",
        "result": json.dumps(result if result is not None else ISSUES),
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "total_cost_usd": 0.01,
    }
    if model_usage is not None:
        envelope["modelUsage"] = model_usage
    return json.dumps(envelope)


def _completed(stdout):
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = stdout
    proc.stderr = ""
    return proc


def _configure_fallbacks(monkeypatch, models):
    """Stand in for `providers.claude-code.fallback_models` in the config."""
    monkeypatch.setattr(
        ClaudeCodeProvider, "fallback_models", lambda self, model: list(models)
    )
    # The per-flag capability gate spawns `claude --help`; it governs argv, not
    # attribution, so it is pinned open rather than probed here.
    monkeypatch.setattr(
        ClaudeCodeProvider, "_fallback_flag_supported", lambda self: True
    )


# ---------------------------------------------------------------------------
# 1. adapter → invoke_with_provider → invoke_with_file_output
# ---------------------------------------------------------------------------

class TestInvocationResultCarriesServedModel:

    def test_a_fired_fallback_is_recorded_and_warned_once(
        self, monkeypatch, capsys
    ):
        _configure_fallbacks(monkeypatch, ["haiku"])

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_completed(_envelope({SERVED: {"outputTokens": 5}}))):
            result = invoke_with_provider("p", "claude-code:sonnet")

        assert result["success"] is True
        assert result["details"]["served_model"] == SERVED
        assert result["details"]["served_model_is_fallback"] is True
        err = capsys.readouterr().err
        assert err.count("[WARN]") == 1, err
        assert SERVED in err and "sonnet" in err

    def test_the_primary_serving_its_own_turn_is_not_a_fallback(
        self, monkeypatch, capsys
    ):
        _configure_fallbacks(monkeypatch, ["haiku"])
        served = "claude-sonnet-4-5-20250929"

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_completed(_envelope({served: {"outputTokens": 5}}))):
            result = invoke_with_provider("p", "claude-code:sonnet")

        assert result["details"]["served_model"] == served
        assert result["details"]["served_model_is_fallback"] is False
        assert "[WARN]" not in capsys.readouterr().err

    def test_absent_model_usage_records_the_request_and_never_fails_the_run(
        self, monkeypatch, capsys
    ):
        _configure_fallbacks(monkeypatch, ["haiku"])

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_completed(_envelope())):
            result = invoke_with_provider("p", "claude-code:sonnet")

        assert result["success"] is True
        assert result["details"]["served_model"] == "sonnet"
        assert result["details"]["served_model_is_fallback"] is False
        assert capsys.readouterr().err.count("[WARN]") == 1

    # ── green stays green ────────────────────────────────────────────────
    def test_the_shipped_empty_config_leaves_the_details_dict_untouched(
        self, capsys
    ):
        """`fallback_models` ships empty ⇒ today's result dict, byte for byte."""
        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_completed(_envelope({SERVED: {"outputTokens": 5}}))):
            result = invoke_with_provider("p", "claude-code:sonnet")

        assert set(result["details"]) == {"provider", "model", "duration_seconds"}
        assert "[WARN]" not in capsys.readouterr().err

    def test_an_adapter_without_the_claim_reports_nothing(self, capsys):
        """Every adapter but claude-code (§4.6 is not re-opened)."""
        wrapper = {"type": "result", "result": json.dumps(ISSUES)}

        with patch("shutil.which", return_value="/usr/bin/cursor-agent"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_completed(json.dumps(wrapper))):
            result = invoke_with_provider("p", "cursor-agent:auto")

        assert set(result["details"]) == {"provider", "model", "duration_seconds"}
        assert "[WARN]" not in capsys.readouterr().err

    def test_a_failed_attempt_is_not_attributed(self, monkeypatch, capsys):
        """Attribution belongs to an answer; a non-zero exit produced none."""
        _configure_fallbacks(monkeypatch, ["haiku"])
        proc = _completed(_envelope({SERVED: {"outputTokens": 5}}))
        proc.returncode = 1

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("utils.llm_client.subprocess.run", return_value=proc):
            result = invoke_with_provider("p", "claude-code:sonnet")

        assert result["success"] is False
        assert "served_model" not in result["details"]
        assert "[WARN]" not in capsys.readouterr().err


class TestFileOutputCarriesServedModel:

    def _stub(self, details):
        return {
            "success": True,
            "data": ISSUES,
            "details": details,
        }

    def test_the_stdout_fallback_path_forwards_the_attribution(self, tmp_path):
        details = {
            "provider": "claude-code",
            "model": "sonnet",
            "duration_seconds": 1.0,
            "served_model": SERVED,
            "served_model_is_fallback": True,
        }
        with patch("utils.llm_client.invoke_with_provider",
                   return_value=self._stub(details)) as spy:
            out = invoke_with_file_output(
                prompt_template="write to {output_json_path}",
                model_spec="claude-code:sonnet",
                prompt_context={"prefix": "p"},
                output_dir=tmp_path,
                phase="code_review",
            )

        assert spy.called
        assert out["source"] == "stdout_fallback"
        assert out["details"]["served_model"] == SERVED
        assert out["details"]["served_model_is_fallback"] is True

    def test_the_file_path_forwards_the_attribution(self, tmp_path):
        details = {
            "provider": "claude-code",
            "model": "sonnet",
            "duration_seconds": 1.0,
            "served_model": SERVED,
            "served_model_is_fallback": True,
        }

        def _write_then_return(*args, **kwargs):
            for path in tmp_path.glob("*.json"):
                path.unlink()
            return self._stub(details)

        with patch("utils.llm_client.invoke_with_provider",
                   side_effect=_write_then_return):
            out = invoke_with_file_output(
                prompt_template="write to {output_json_path}",
                model_spec="claude-code:sonnet",
                prompt_context={"prefix": "p"},
                output_dir=tmp_path,
                phase="code_review",
            )

        assert out["details"]["served_model"] == SERVED

    def test_no_attribution_leaves_the_details_dict_untouched(self, tmp_path):
        details = {"provider": "codex", "model": "gpt-5", "duration_seconds": 1.0}
        with patch("utils.llm_client.invoke_with_provider",
                   return_value=self._stub(details)):
            out = invoke_with_file_output(
                prompt_template="write to {output_json_path}",
                model_spec="codex:gpt-5",
                prompt_context={"prefix": "p"},
                output_dir=tmp_path,
                phase="code_review",
            )

        assert set(out["details"]) == {"provider", "model", "duration_seconds"}


# ---------------------------------------------------------------------------
# 2. the fan-out result records
# ---------------------------------------------------------------------------

class TestFanOutResultRecords:

    def _plan(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        return plan

    def _run(self, tmp_path, details):
        plan = self._plan(tmp_path)
        stub = {
            "success": True,
            "data": ISSUES,
            "source": "file",
            "output_file": str(tmp_path / "out.json"),
            "details": details,
        }
        with patch("utils.review_orchestrator_base.invoke_with_file_output",
                   return_value=stub):
            return asyncio.run(run_single_model(
                "claude-code:sonnet",
                str(plan),
                "review {output_json_path}",
                {},
                "review-plan",
                "plan_review",
                # claude-code's real configured slot, not an arbitrary 60 s.
                # M9 put Lane A on the shared ladder with §7.5 rule 4 on: a slot
                # whose budget cannot hold one worst-case attempt of the
                # adapter's DECLARED envelope (claude-code: 4 × API_TIMEOUT_MS =
                # 1200 s) is refused with nothing dispatched. Attribution is what
                # this class tests, and it needs a slot that really dispatches.
                timeout=1800,
                out_dir=str(tmp_path),
                prefix="plan",
            ))

    def test_run_single_model_carries_the_served_model(self, tmp_path):
        result = self._run(tmp_path, {
            "provider": "claude-code",
            "model": "sonnet",
            "duration_seconds": 1.0,
            "served_model": SERVED,
            "served_model_is_fallback": True,
        })

        assert result["success"] is True
        assert result["served_model"] == SERVED
        assert result["served_model_is_fallback"] is True

    def test_without_attribution_the_result_dict_is_unchanged(self, tmp_path):
        result = self._run(tmp_path, {
            "provider": "codex", "model": "gpt-5", "duration_seconds": 1.0,
        })

        assert set(result) == {
            "success", "output", "error", "prompt", "stdout", "stderr",
            "duration_seconds", "model_spec", "source",
        }


# ---------------------------------------------------------------------------
# 3. the surfaces a user reads
# ---------------------------------------------------------------------------

class TestMarkdownAggregate:

    def _phase(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        return phase_dir, plan

    def test_the_models_line_names_the_served_model(self, tmp_path):
        phase_dir, plan = self._phase(tmp_path)

        aggregate_results(
            "plan", str(tmp_path), str(phase_dir), "review-plan",
            ["claude-code:sonnet"], {}, [],
            plan_path=str(plan),
            served_models={
                "claude-code:sonnet": {"served_model": SERVED, "fallback": True},
            },
        )

        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert SERVED in report
        assert "fallback" in report

    def test_without_attribution_the_models_line_is_unchanged(self, tmp_path):
        phase_dir, plan = self._phase(tmp_path)

        aggregate_results(
            "plan", str(tmp_path), str(phase_dir), "review-plan",
            ["claude-code:sonnet"], {}, [],
            plan_path=str(plan),
        )

        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Models:** claude-code:sonnet" in report
        assert "served" not in report and "fallback" not in report


class TestCodeReviewReport:

    def test_the_report_labels_every_row_that_names_a_model(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        results = {"claude-code:sonnet": (True, list(ISSUES), None)}

        report = cro.generate_review_report(
            plan, results, ["a.py"],
            served_models={
                "claude-code:sonnet": {"served_model": SERVED, "fallback": True},
            },
        )

        # The aggregate line AND the per-issue row.
        assert report.count(SERVED) >= 2
        assert "**Model:** claude-code:sonnet (served by" in report

    def test_without_attribution_the_report_is_unchanged(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        results = {"claude-code:sonnet": (True, list(ISSUES), None)}

        report = cro.generate_review_report(plan, results, ["a.py"])

        assert "**Models:** claude-code:sonnet" in report
        # The per-issue row keeps today's exact trailing field.
        assert "**Model:** claude-code:sonnet\n" in report
        assert "served by" not in report and "fallback" not in report

    def test_the_report_stamps_the_issue_ordering_it_numbered(self, tmp_path):
        """F9: the `### N.` numbers are positions; say which list they index.

        `apply-code-fixes` resolves a checkbox by looking up N in issues.json,
        so the report has to name the ordering it numbered or a stale report.md
        marks the wrong findings in silence.
        """
        from utils.report_parser import (
            issue_numbering_digest, parse_issue_numbering_digest,
        )

        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        results = {"claude-code:sonnet": (True, list(ISSUES), None)}

        report = cro.generate_review_report(plan, results, ["a.py"])
        report_path = tmp_path / "report.md"
        report_path.write_text(report, encoding="utf-8")

        assert parse_issue_numbering_digest(str(report_path)) == (
            issue_numbering_digest(list(ISSUES))
        )
        # A comment, so the issue blocks the parsers read are untouched.
        assert report.count("### 1.") == 1

    def test_run_single_review_fills_the_collector(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        collected = {}
        stub = {
            "success": True,
            "data": list(ISSUES),
            "source": "file",
            "output_file": str(tmp_path / "out.json"),
            "details": {
                "provider": "claude-code",
                "model": "sonnet",
                "duration_seconds": 1.0,
                "served_model": SERVED,
                "served_model_is_fallback": True,
            },
        }

        with patch.object(cro, "invoke_with_file_output", return_value=stub):
            success, issues, error = asyncio.run(cro.run_single_review(
                "claude-code:sonnet", plan, tmp_path,
                served_models=collected,
            ))

        assert success is True and error is None
        assert collected == {
            "claude-code:sonnet": {"served_model": SERVED, "fallback": True}
        }

    def test_the_collector_stays_empty_without_attribution(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        collected = {}
        stub = {
            "success": True,
            "data": list(ISSUES),
            "source": "file",
            "output_file": str(tmp_path / "out.json"),
            "details": {"provider": "codex", "model": "gpt-5", "duration_seconds": 1.0},
        }

        with patch.object(cro, "invoke_with_file_output", return_value=stub):
            asyncio.run(cro.run_single_review(
                "codex:gpt-5", plan, tmp_path, served_models=collected,
            ))

        assert collected == {}


class TestHtmlReportData:

    def _plan(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\n### Step 1\nDo it.\n", encoding="utf-8")
        return plan

    def test_report_data_records_and_labels_the_served_model(self, tmp_path):
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        generate_html_report(
            groups=[],
            plan_path=self._plan(tmp_path),
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude-code:sonnet"],
            served_models={
                "claude-code:sonnet": {"served_model": SERVED, "fallback": True},
            },
        )

        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["servedModels"]["claude-code:sonnet"]["served_model"] == SERVED
        meta = data["modelMetadata"]["claude-code:sonnet"]
        assert meta["served_model"] == SERVED
        assert meta["fallback"] is True
        # The badge the templates render must not still read "sonnet" alone.
        assert SERVED in meta["model"]

    def test_without_attribution_report_data_is_unchanged(self, tmp_path):
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        generate_html_report(
            groups=[],
            plan_path=self._plan(tmp_path),
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude-code:sonnet"],
        )

        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert "servedModels" not in data
        meta = data["modelMetadata"]["claude-code:sonnet"]
        assert set(meta) == {
            "full", "provider", "model", "provider_color", "model_color",
            "icon", "provider_icon",
        }
        assert meta["model"] == "sonnet"


class TestServedModelLabel:

    @staticmethod
    def _label(*args):
        from utils.html_report_generator import served_model_label

        return served_model_label(*args)

    def test_no_entry_is_the_bare_model(self):
        assert self._label("claude-code:sonnet", None) == "claude-code:sonnet"
        assert self._label("claude-code:sonnet", {}) == "claude-code:sonnet"

    def test_a_fallback_is_named_and_marked(self):
        label = self._label("claude-code:sonnet", {
            "claude-code:sonnet": {"served_model": SERVED, "fallback": True},
        })
        assert label == f"claude-code:sonnet (served by {SERVED}, fallback)"

    def test_a_non_fallback_resolution_is_named_but_not_marked(self):
        label = self._label("claude-code:sonnet", {
            "claude-code:sonnet": {
                "served_model": "claude-sonnet-4-5-20250929", "fallback": False,
            },
        })
        assert label == "claude-code:sonnet (served by claude-sonnet-4-5-20250929)"
