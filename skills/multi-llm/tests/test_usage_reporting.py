"""Stage U — what a run spent travels from the envelope to every report.

The adapter half (``parse_usage`` itself) is each adapter's own, covered in
``tests/test_providers.py``. This file covers the *carriage*: adapter →
``invoke_with_provider`` → ``invoke_with_file_output`` → the fan-out result
records → ``.status.json`` → ``report.md`` / ``report.html`` /
``report_data.json`` → and back out again through ``--reaggregate``.

It is the sibling of ``tests/test_served_model_reporting.py`` and asserts the
same two properties one field over:

1. **A number that was reported reaches the reader.** A slot whose adapter can
   say what it spent has that on its result, in the checkpoint, on the model
   badge and on the ``**Models:**`` line — for the run that failed as much as
   for the run that answered, because the failure is the one whose cost gets
   asked about.
2. **A number that was never reported is never invented.** An adapter at
   ``usage_support = "none"`` contributes no ``usage`` key anywhere and renders
   no clause — never ``$0.00``, which would claim a measured free run.

The two adapters are **fakes injected into the registry**, not shipped ones, and
deliberately so: this is harness plumbing that must hold for every adapter that
declares ``USAGE_FULL``, including the ones not written yet, so the fixture is a
minimal multi-turn stream rather than any one CLI's envelope.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import provider_registry
from utils import review_orchestrator_base as rob
from utils.consolidation import generate_consolidated_html
from utils.html_report_generator import describe_usage, model_label, sum_usage
from utils.llm_client import ERROR_TIMEOUT, invoke_with_provider
from utils.providers.base import LLMProvider, USAGE_FULL, USAGE_NONE
from utils.review_orchestrator_base import (
    accumulate_usage,
    aggregate_results,
    collect_usage,
    run_single_model,
    usage_fields,
    usage_total_fields,
)
from utils.json_extractor import sanitize_model_name
from utils.output_handler import get_phase_dir
from utils.status_file import update_status


def _async_none_result():
    """``run_single_review``'s signature, answering with no usage at all."""
    async def _run(*_args, **_kwargs):
        return (True, [], None)
    return _run

FULL_SPEC = "usagefull:m"
RETRY_SPEC = "usageretry:m"
NONE_SPEC = "usagenone:m"

ISSUES = [{
    "title": "Boundary check missing",
    "desc": "d",
    "importance": "HIGH",
    "reference": "Step 1",
    "type": "bug",
}]

#: Three assistant turns and two tool calls — the shape a tool-using CLI emits
#: for one review, and the reason a summing extractor is not a "read the last
#: block" one. The numbers are chosen so the sums are recognisable in an
#: assertion and so the rendered clause is the one a reader would see:
#: ``$0.04 est., 1,234 output tokens``.
TURNS = [
    {"type": "message_end", "role": "assistant",
     "usage": {"input": 1000, "output": 200, "cacheRead": 100,
               "totalTokens": 1200, "cost": {"total": 0.01}}},
    {"type": "tool_execution_end", "name": "read"},
    {"type": "message_end", "role": "assistant",
     "usage": {"input": 1500, "output": 300, "cacheRead": 200,
               "totalTokens": 1800, "cost": {"total": 0.02}}},
    {"type": "tool_execution_end", "name": "grep"},
    {"type": "message_end", "role": "assistant", "text": json.dumps(ISSUES),
     "usage": {"input": 2000, "output": 734, "cacheRead": 300,
               "totalTokens": 2734, "cost": {"total": 0.0112}}},
]
MULTI_TURN_STREAM = "\n".join(json.dumps(event) for event in TURNS) + "\n"

#: What ``parse_usage`` must make of the stream above, summed over all three
#: turns rather than read off the last one.
EXPECTED_USAGE = {
    "input_tokens": 4500,
    "output_tokens": 1234,
    "cached_input_tokens": 600,
    "total_tokens": 5734,
    "cost_usd": 0.0412,
    "tool_uses": 2,
}
EXPECTED_CLAUSE = "$0.04 est., 1,234 output tokens"


# ---------------------------------------------------------------------------
# Two throwaway adapters: one that reports usage, one that cannot
# ---------------------------------------------------------------------------

class _ReportingProvider(LLMProvider):
    """The smallest ``USAGE_FULL`` adapter: a JSONL stream, summed."""

    usage_support = USAGE_FULL

    @property
    def name(self) -> str:
        return "usagefull"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return True

    def build_command(self, prompt, model, **_kwargs):
        return ["usagefull-cli", "--model", model, prompt]

    @staticmethod
    def _events(stdout):
        for line in (stdout or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event

    def parse_output(self, stdout, stderr):
        text = ""
        for event in self._events(stdout):
            if event.get("type") == "message_end" and event.get("text"):
                text = event["text"]
        if not text:
            return {"success": False, "error": "Empty output", "data": None}
        try:
            return {"success": True, "data": json.loads(text)}
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"Bad JSON: {exc}", "data": None}

    def parse_usage(self, stdout, stderr):
        totals = {}
        tool_uses = 0
        for event in self._events(stdout):
            if event.get("type") == "tool_execution_end":
                tool_uses += 1
            block = event.get("usage")
            if event.get("type") != "message_end" or not isinstance(block, dict):
                continue
            totals["input_tokens"] = totals.get("input_tokens", 0) + block["input"]
            totals["output_tokens"] = totals.get("output_tokens", 0) + block["output"]
            totals["cached_input_tokens"] = (
                totals.get("cached_input_tokens", 0) + block["cacheRead"]
            )
            totals["total_tokens"] = (
                totals.get("total_tokens", 0) + block["totalTokens"]
            )
            totals["cost_usd"] = round(
                totals.get("cost_usd", 0.0) + block["cost"]["total"], 6
            )
        if tool_uses:
            totals["tool_uses"] = tool_uses
        return totals or None


class _RetryingProvider(_ReportingProvider):
    """``_ReportingProvider`` whose losing exit says why, on the one channel
    Lane A classifies.

    Lane A reads ``result["error"]`` and nothing else (§7.2), and that string is
    ``f"{provider} exited with code {n}"`` plus whatever ``describe_failure``
    quotes onto it — so an adapter that declares no such hook can never produce
    a *retryable* non-zero exit here, and the retry loop this fake exists to
    drive would never run a second attempt. One line off stderr, which is the
    shape the base hook documents.
    """

    @property
    def name(self) -> str:
        return "usageretry"

    def build_command(self, prompt, model, **_kwargs):
        return ["usageretry-cli", "--model", model, prompt]

    def describe_failure(self, stdout, stderr):
        first = (stderr or "").strip().splitlines()
        return first[0] if first else None


class _SilentProvider(LLMProvider):
    """The duration-only adapter every negative here is measured against."""

    usage_support = USAGE_NONE

    @property
    def name(self) -> str:
        return "usagenone"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return True

    def build_command(self, prompt, model, **_kwargs):
        return ["usagenone-cli", "--model", model, prompt]

    def parse_output(self, stdout, stderr):
        try:
            return {"success": True, "data": json.loads(stdout)}
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"Bad JSON: {exc}", "data": None}


_REAL_RUN = subprocess.run


def _completed(stdout, returncode=0, stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _routed_run(streams):
    """Answer only the fake CLIs; let everything else really run.

    ``patch("utils.llm_client.subprocess.run")`` patches the attribute on the
    **shared** ``subprocess`` module, so a blanket ``return_value`` also answers
    the ``git rev-parse`` the orchestrator runs to find the project root — which
    then reads a JSONL stream as a directory path. Routing on ``argv[0]`` keeps
    the fake confined to the dispatch it was written for.
    """
    def run(cmd, **kwargs):
        argv0 = str(cmd[0]) if isinstance(cmd, (list, tuple)) and cmd else ""
        for binary, (stdout, returncode) in streams.items():
            if argv0.endswith(binary):
                return _completed(stdout, returncode)
        return _REAL_RUN(cmd, **kwargs)
    return run


def _routed_which(name, *args, **kwargs):
    return f"/usr/bin/{name}" if str(name).endswith("-cli") else None


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    """Both fakes in the registry for the duration of one test.

    Autouse, and scoped to one test: the registry is a module-level dict every
    registry-parametrized sweep in the suite reads, so a fake that outlived its
    test would be graded as a shipped adapter by a dozen other files.
    """
    monkeypatch.setitem(provider_registry._PROVIDERS, "usagefull", _ReportingProvider())
    monkeypatch.setitem(provider_registry._PROVIDERS, "usageretry", _RetryingProvider())
    monkeypatch.setitem(provider_registry._PROVIDERS, "usagenone", _SilentProvider())


def _dispatch(streams=None):
    """The patch pair every dispatch in this module runs under."""
    streams = streams or {
        "usagefull-cli": (MULTI_TURN_STREAM, 0),
        "usagenone-cli": (json.dumps(ISSUES), 0),
    }
    return (
        patch("shutil.which", side_effect=_routed_which),
        patch("utils.llm_client.subprocess.run", side_effect=_routed_run(streams)),
    )


def _plan(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")
    return plan


def _fan_out(tmp_path, specs, streams=None):
    """One slot per spec through the real ``run_single_model``, serially."""
    plan = _plan(tmp_path)
    which, run = _dispatch(streams)
    results = {}
    with which, run:
        for spec in specs:
            results[spec] = asyncio.run(run_single_model(
                spec, str(plan), "review {output_json_path}", {},
                "review-plan", "plan_review",
                timeout=600, out_dir=str(tmp_path), prefix="plan",
            ))
    return results


# ---------------------------------------------------------------------------
# 1. adapter → invoke_with_provider
# ---------------------------------------------------------------------------

class TestInvocationResultCarriesUsage:

    def test_a_successful_exit_zero_run_carries_what_it_spent(self):
        """The gap Stage U closes: the winning slot used to report nothing."""
        which, run = _dispatch()
        with which, run:
            result = invoke_with_provider("p", FULL_SPEC)

        assert result["success"] is True
        assert result["details"]["usage"] == EXPECTED_USAGE

    def test_the_numbers_are_summed_over_the_turns_not_read_off_the_last(self):
        """A one-turn read would report 734 output tokens for a 1,234-token run."""
        last_turn = TURNS[-1]["usage"]
        assert EXPECTED_USAGE["output_tokens"] > last_turn["output"]
        assert EXPECTED_USAGE["cost_usd"] > last_turn["cost"]["total"]

    def test_an_exit_zero_parse_failure_still_carries_it(self):
        """The run that produced no answer is not the run that spent nothing."""
        truncated = "\n".join(json.dumps(e) for e in TURNS[:-1]) + "\n" + json.dumps(
            {**TURNS[-1], "text": ""}
        )
        which, run = _dispatch({"usagefull-cli": (truncated, 0)})
        with which, run:
            result = invoke_with_provider("p", FULL_SPEC)

        assert result["success"] is False
        assert result["details"]["usage"] == EXPECTED_USAGE

    def test_a_losing_exit_carries_it_too(self):
        """The path that always had it, asserted here so the two cannot diverge."""
        which, run = _dispatch({"usagefull-cli": (MULTI_TURN_STREAM, 2)})
        with which, run:
            result = invoke_with_provider("p", FULL_SPEC)

        assert result["success"] is False
        assert result["details"]["usage"] == EXPECTED_USAGE

    def test_the_timeout_kill_carries_it_as_well(self):
        """The third documented outcome, and the one that never asked.

        The run killed at the deadline is the expensive failure Stage U exists
        to report, and `subprocess.run` re-raises a `TimeoutExpired` carrying
        the bytes read off the pipe before the kill — the same partial stream
        `_attach_stdout_excerpt` is bounded from one line above. An adapter
        whose usage rides on the stream (a terminal event, or a per-step sum)
        can answer from it, so the envelope says what the run spent instead of
        reporting a $25 loss as no number at all.
        """
        def _killed_at_the_deadline(cmd, **kwargs):
            argv0 = str(cmd[0]) if isinstance(cmd, (list, tuple)) and cmd else ""
            if argv0.endswith("usagefull-cli"):
                raise subprocess.TimeoutExpired(
                    cmd, 600, output=MULTI_TURN_STREAM, stderr=""
                )
            return _REAL_RUN(cmd, **kwargs)

        with patch("shutil.which", side_effect=_routed_which), patch(
            "utils.llm_client.subprocess.run", side_effect=_killed_at_the_deadline
        ):
            result = invoke_with_provider("p", FULL_SPEC)

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert result["details"]["usage"] == EXPECTED_USAGE
        # The sibling key on the same dict: both halves are read off the same
        # recovered stdout, so one arriving without the other is the bug.
        assert result["details"]["stdout_excerpt"]

    def test_a_kill_before_the_first_write_reports_no_usage(self):
        """Absent, never `{}` — "not reported" stays distinguishable."""
        def _killed_silent(cmd, **kwargs):
            argv0 = str(cmd[0]) if isinstance(cmd, (list, tuple)) and cmd else ""
            if argv0.endswith("usagefull-cli"):
                raise subprocess.TimeoutExpired(cmd, 600, output="", stderr="")
            return _REAL_RUN(cmd, **kwargs)

        with patch("shutil.which", side_effect=_routed_which), patch(
            "utils.llm_client.subprocess.run", side_effect=_killed_silent
        ):
            result = invoke_with_provider("p", FULL_SPEC)

        assert result["success"] is False
        assert "usage" not in result["details"]
        assert "stdout_excerpt" not in result["details"]

    # ── without usage ────────────────────────────────────────────────────
    def test_an_adapter_that_reports_nothing_adds_no_key(self):
        which, run = _dispatch()
        with which, run:
            result = invoke_with_provider("p", NONE_SPEC)

        assert result["success"] is True
        assert set(result["details"]) == {"provider", "model", "duration_seconds"}


# ---------------------------------------------------------------------------
# 2. the fan-out result records
# ---------------------------------------------------------------------------

class TestFanOutResultRecords:

    def test_the_record_of_a_slot_that_answered_carries_the_usage(self, tmp_path):
        results = _fan_out(tmp_path, [FULL_SPEC])
        assert results[FULL_SPEC]["success"] is True
        assert results[FULL_SPEC]["usage"] == EXPECTED_USAGE

    def test_the_record_of_a_slot_that_failed_carries_it_as_well(self, tmp_path):
        results = _fan_out(
            tmp_path, [FULL_SPEC],
            streams={"usagefull-cli": (MULTI_TURN_STREAM, 3)},
        )
        assert results[FULL_SPEC]["success"] is False
        assert results[FULL_SPEC]["usage"] == EXPECTED_USAGE

    def test_usage_fields_is_the_one_shape_both_records_are_built_from(self):
        assert usage_fields({"usage": EXPECTED_USAGE}) == {"usage": EXPECTED_USAGE}
        # Absent, empty and wrong-typed all mean "nothing was reported".
        assert usage_fields({}) == {}
        assert usage_fields({"usage": {}}) == {}
        assert usage_fields({"usage": None}) == {}
        assert usage_fields(None) == {}

    def test_collect_usage_keys_by_model_spec_and_skips_the_silent(self, tmp_path):
        results = _fan_out(tmp_path, [FULL_SPEC, NONE_SPEC])
        assert collect_usage(results) == {FULL_SPEC: EXPECTED_USAGE}

    # ── without usage ────────────────────────────────────────────────────
    def test_without_usage_the_result_dict_is_unchanged(self, tmp_path):
        results = _fan_out(tmp_path, [NONE_SPEC])
        assert set(results[NONE_SPEC]) == {
            "success", "output", "error", "prompt", "stdout", "stderr",
            "duration_seconds", "model_spec", "source",
        }


# ---------------------------------------------------------------------------
# 2b. a slot that retried spent what EVERY attempt spent
# ---------------------------------------------------------------------------

#: One turn, no tool calls, priced so the two attempts are told apart in a
#: failed assertion: ``$1.00`` against the winning attempt's ``$0.0412``.
FIRST_ATTEMPT_STREAM = json.dumps({
    "type": "message_end", "role": "assistant",
    "usage": {"input": 10, "output": 20, "cacheRead": 5,
              "totalTokens": 30, "cost": {"total": 1.0}},
}) + "\n"

FIRST_ATTEMPT_USAGE = {
    "input_tokens": 10,
    "output_tokens": 20,
    "cached_input_tokens": 5,
    "total_tokens": 30,
    "cost_usd": 1.0,
}

#: ``FIRST_ATTEMPT_USAGE + EXPECTED_USAGE``, spelled out rather than summed in
#: the assertion: a test that recomputes the thing under test passes for a
#: reason of its own.
BOTH_ATTEMPTS_USAGE = {
    "input_tokens": 4510,
    "output_tokens": 1254,
    "cached_input_tokens": 605,
    "total_tokens": 5764,
    "cost_usd": 1.0412,
    "tool_uses": 2,
}

#: A non-zero exit whose reason the classifier calls a server error, which is
#: rung 4 — retried once against ``driver.subagent_retries``.
RETRYABLE_STDERR = "Error: 500 internal server error from upstream\n"


def _sequenced_run(binary, responses):
    """``_routed_run``, but answering one dispatch at a time.

    The retry loop dispatches the *same* argv twice, so a table keyed by binary
    cannot express "failed, then answered". Later calls repeat the last row.
    """
    calls = {"n": 0}

    def run(cmd, **kwargs):
        argv0 = str(cmd[0]) if isinstance(cmd, (list, tuple)) and cmd else ""
        if argv0.endswith(binary):
            row = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            return _completed(*row)
        return _REAL_RUN(cmd, **kwargs)

    return run, calls


def _retried_slot(tmp_path, monkeypatch, responses):
    """One ``run_single_model`` slot over a scripted sequence of attempts."""
    monkeypatch.setattr(
        "utils.review_orchestrator_base.get_subagent_retries", lambda **_kw: 1
    )
    plan = _plan(tmp_path)
    run, calls = _sequenced_run("usageretry-cli", responses)
    with patch("shutil.which", side_effect=_routed_which), \
            patch("utils.llm_client.subprocess.run", side_effect=run):
        result = asyncio.run(run_single_model(
            RETRY_SPEC, str(plan), "review {output_json_path}", {},
            "review-plan", "plan_review",
            timeout=600, out_dir=str(tmp_path), prefix="plan",
        ))
    return result, calls["n"]


class TestARetriedSlotReportsEveryAttempt:
    """The retry loop rebinds ``details``; the record must not read the last one.

    A reproduced failure that cost $1 followed by a $2 success reported $2 —
    the checkpoint, the badge and the report header all understated the slot by
    exactly the attempts an operator is most likely to be asking about.
    """

    def test_the_winning_record_carries_the_failed_attempt_too(
        self, tmp_path, monkeypatch
    ):
        result, attempts = _retried_slot(tmp_path, monkeypatch, [
            (FIRST_ATTEMPT_STREAM, 3, RETRYABLE_STDERR),
            (MULTI_TURN_STREAM, 0, ""),
        ])
        assert attempts == 2, "the fixture must really drive the retry loop"
        assert result["success"] is True
        assert result["usage"] == BOTH_ATTEMPTS_USAGE

    def test_a_slot_that_never_answered_reports_both_losses(
        self, tmp_path, monkeypatch
    ):
        result, attempts = _retried_slot(tmp_path, monkeypatch, [
            (FIRST_ATTEMPT_STREAM, 3, RETRYABLE_STDERR),
            (FIRST_ATTEMPT_STREAM, 3, RETRYABLE_STDERR),
        ])
        assert attempts == 2
        assert result["success"] is False
        assert result["usage"] == {
            "input_tokens": 20, "output_tokens": 40, "cached_input_tokens": 10,
            "total_tokens": 60, "cost_usd": 2.0,
        }

    def test_one_attempt_is_what_it_always_was(self, tmp_path, monkeypatch):
        result, attempts = _retried_slot(
            tmp_path, monkeypatch, [(MULTI_TURN_STREAM, 0, "")]
        )
        assert attempts == 1
        assert result["usage"] == EXPECTED_USAGE

    # ── the accumulator itself ───────────────────────────────────────────
    def test_numbers_add_and_labels_come_from_the_last_reporter(self):
        total = accumulate_usage(None, {"usage": {"cost_usd": 0.01, "served_model": "a"}})
        total = accumulate_usage(total, {"usage": {"cost_usd": 0.02, "served_model": "b"}})
        # Rounded, because 0.01 + 0.02 is 0.030000000000000002 in binary.
        assert total == {"cost_usd": 0.03, "served_model": "b"}

    def test_a_key_only_one_attempt_reported_survives(self):
        total = accumulate_usage({"cost_usd": 1.0}, {"usage": {"tool_uses": 2}})
        assert total == {"cost_usd": 1.0, "tool_uses": 2}

    def test_nothing_reported_stays_absent(self):
        assert accumulate_usage(None, {}) is None
        assert accumulate_usage(None, {"usage": {}}) is None
        assert accumulate_usage(None, {"usage": None}) is None
        # An attempt that reported nothing never erases what earlier ones did.
        assert accumulate_usage({"cost_usd": 1.0}, {}) == {"cost_usd": 1.0}
        assert usage_total_fields(None) == {}
        assert usage_total_fields({"cost_usd": 1.0}) == {"usage": {"cost_usd": 1.0}}


# ---------------------------------------------------------------------------
# 3. the surfaces a user reads
# ---------------------------------------------------------------------------

class TestCheckpointAndReports:

    def _phase(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir(exist_ok=True)
        return phase_dir

    def _run(self, tmp_path):
        """The fan-out, its checkpoint, and the report built from both.

        ``update_status`` is called with what ``collect_usage`` returned, which
        is the pair the fan-out itself writes — the surrounding
        ``run_review_orchestration`` needs an argparse namespace, a grouping
        pass and a validation pass to reach that line, so this module drives the
        two halves directly, as ``test_served_model_reporting`` does for
        ``models_served``.
        """
        results = _fan_out(tmp_path, [FULL_SPEC, NONE_SPEC])
        phase_dir = self._phase(tmp_path)
        usage = collect_usage(results)
        update_status(str(phase_dir), {
            "state": "models_complete",
            "models_completed": [FULL_SPEC, NONE_SPEC],
            "models_failed": {},
            **({"usage": usage} if usage else {}),
        })
        aggregate_results(
            "plan", str(tmp_path), str(phase_dir), "review-plan",
            [FULL_SPEC, NONE_SPEC], {}, [],
            plan_path=str(tmp_path / "plan.md"),
            usage=usage or None,
        )
        return phase_dir, usage

    def test_the_checkpoint_records_the_sums_under_usage(self, tmp_path):
        phase_dir, _ = self._run(tmp_path)
        recorded = json.loads(
            (phase_dir / ".status.json").read_text(encoding="utf-8")
        )["usage"]

        assert recorded[FULL_SPEC] == EXPECTED_USAGE
        assert NONE_SPEC not in recorded

    def test_the_models_line_names_the_cost(self, tmp_path):
        phase_dir, _ = self._run(tmp_path)
        report = (phase_dir / "report.md").read_text(encoding="utf-8")

        # The whole line, so the silent model is pinned as named *bare* —
        # no empty parenthesis after it and no "$0.00" standing in for the
        # measurement its adapter cannot make.
        line = next(l for l in report.splitlines() if l.startswith("**Models:**"))
        assert line == f"**Models:** {FULL_SPEC} ({EXPECTED_CLAUSE}), {NONE_SPEC}"
        assert "$0.00" not in report

    def test_report_data_carries_the_block_and_the_total(self, tmp_path):
        phase_dir, usage = self._run(tmp_path)
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))

        assert data["modelMetadata"][FULL_SPEC]["usage"] == EXPECTED_USAGE
        assert data["modelMetadata"][FULL_SPEC]["usage_label"] == EXPECTED_CLAUSE
        assert data["usageTotal"] == sum_usage(usage) == EXPECTED_USAGE
        assert data["usageTotalLabel"] == EXPECTED_CLAUSE

    def test_the_html_report_renders_the_clause(self, tmp_path):
        phase_dir, _ = self._run(tmp_path)
        html = (phase_dir / "report.html").read_text(encoding="utf-8")

        assert EXPECTED_CLAUSE in html
        # The badge is rendered from `usage_label`, so the template has to read
        # the key the generator writes.
        assert "metadata.usage_label" in html
        assert "reportData.usageTotalLabel" in html

    def test_the_total_sums_every_reporting_model(self, tmp_path):
        """Two reporting slots, one number — the figure an operator looks for."""
        results = _fan_out(tmp_path, [FULL_SPEC])
        doubled = {FULL_SPEC: results[FULL_SPEC], "usagefull:other": results[FULL_SPEC]}
        total = sum_usage(collect_usage(doubled))

        assert total["cost_usd"] == pytest.approx(2 * EXPECTED_USAGE["cost_usd"])
        assert total["output_tokens"] == 2 * EXPECTED_USAGE["output_tokens"]
        assert total["tool_uses"] == 2 * EXPECTED_USAGE["tool_uses"]

    # ── without usage ────────────────────────────────────────────────────
    def test_without_usage_report_data_is_unchanged(self, tmp_path):
        results = _fan_out(tmp_path, [NONE_SPEC])
        phase_dir = self._phase(tmp_path)
        aggregate_results(
            "plan", str(tmp_path), str(phase_dir), "review-plan",
            [NONE_SPEC], {}, [], plan_path=str(tmp_path / "plan.md"),
            usage=collect_usage(results) or None,
        )
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))

        assert "usageTotal" not in data
        assert "usageTotalLabel" not in data
        assert set(data["modelMetadata"][NONE_SPEC]) == {
            "full", "provider", "model", "provider_color", "model_color",
            "icon", "provider_icon",
        }
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert f"**Models:** {NONE_SPEC}" in report
        assert "$" not in report.split("**Groups:**")[0]

    def test_without_usage_the_checkpoint_gains_no_key(self, tmp_path):
        results = _fan_out(tmp_path, [NONE_SPEC])
        phase_dir = self._phase(tmp_path)
        usage = collect_usage(results)
        update_status(str(phase_dir), {
            "state": "models_complete",
            **({"usage": usage} if usage else {}),
        })
        recorded = json.loads(
            (phase_dir / ".status.json").read_text(encoding="utf-8")
        )
        assert "usage" not in recorded


# ---------------------------------------------------------------------------
# 4. --reaggregate
# ---------------------------------------------------------------------------

class TestReaggregate:

    def test_the_numbers_survive_a_reaggregation(self, tmp_path):
        """`{model}.json` is a bare suggestion list, so the checkpoint is the
        only place the numbers exist by the time a reaggregate runs."""
        import argparse

        from utils.status_file import write_status

        plan = _plan(tmp_path)
        out_dir = tmp_path / "out"
        phase_dir = out_dir / "review-plan"
        phase_dir.mkdir(parents=True)
        (phase_dir / "usagefull_m.json").write_text(
            json.dumps(ISSUES), encoding="utf-8"
        )
        write_status(str(phase_dir), {
            "phase": "review-plan", "state": "models_complete",
            "started_at": "2026-09-17T00:00:00",
            "models_requested": [FULL_SPEC],
            "models_completed": [FULL_SPEC],
            "models_failed": {}, "error": None, "plan_hash": "deadbeef",
            "usage": {FULL_SPEC: EXPECTED_USAGE},
        })

        assert rob.recorded_usage(str(phase_dir)) == {FULL_SPEC: EXPECTED_USAGE}

        asyncio.run(rob.reaggregate_from_existing_files(
            prefix="plan", out_dir=str(out_dir), phase_dir=str(phase_dir),
            phase_name="review-plan", plan_path=str(plan),
            args=argparse.Namespace(
                skip_validation=True, validation_model="auto", timeout=None,
                internal_validation=False, force=False,
            ),
            orchestrator_script="review_plan_orchestrator.py",
        ))

        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert f"{FULL_SPEC} ({EXPECTED_CLAUSE})" in report
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["modelMetadata"][FULL_SPEC]["usage"] == EXPECTED_USAGE
        assert data["usageTotal"] == EXPECTED_USAGE

    def test_a_checkpoint_without_the_key_reaggregates_as_before(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        assert rob.recorded_usage(str(phase_dir)) is None


# ---------------------------------------------------------------------------
# 4b. resume
# ---------------------------------------------------------------------------

class TestResume:
    """A slot answered from its existing ``{model}.json`` keeps what it cost.

    The other half of section 4's argument. ``--reaggregate`` reads the
    checkpoint and re-reports it; a plain *resume* re-runs the fan-out, and the
    fan-out's opening ``write_status`` replaces every key but the two reserved
    objects — so the ``usage`` the previous attempt wrote is gone before the
    first slot starts, and a skipped slot invokes no adapter to refill it. The
    numbers were therefore lost by the run most likely to be asked about them:
    the second attempt at a partial run.
    """

    def test_the_opening_write_status_would_drop_the_key(self, tmp_path):
        """The mechanism, pinned: this is why the read has to happen first."""
        from utils.status_file import read_status, write_status

        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        write_status(str(phase_dir), {
            "phase": "review-plan", "state": "models_complete",
            "usage": {FULL_SPEC: EXPECTED_USAGE},
        })
        preserved = rob.recorded_usage(str(phase_dir))
        write_status(str(phase_dir), {
            "phase": "review-plan", "state": "models_running",
        })

        assert preserved == {FULL_SPEC: EXPECTED_USAGE}
        assert "usage" not in read_status(str(phase_dir))

    def _resumable(self, tmp_path, phase_name):
        """A phase dir holding one finished model's result list."""
        plan = _plan(tmp_path)
        phase_dir = get_phase_dir(plan, phase_name)
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / f"{sanitize_model_name(FULL_SPEC)}.json").write_text(
            json.dumps(ISSUES), encoding="utf-8"
        )
        return plan, phase_dir

    def test_the_base_fan_out_puts_it_back_on_the_skipped_slot(self, tmp_path):
        plan, _ = self._resumable(tmp_path, "review-plan")
        which, run = _dispatch()
        with which, run:
            results = asyncio.run(rob.run_all_models(
                [FULL_SPEC, NONE_SPEC], str(plan), "review {output_json_path}",
                lambda spec: {}, "review-plan", "plan_review",
                timeout=600, max_parallel=2, out_dir=str(tmp_path),
                prefix="plan", skip_existing=True,
                prior_usage={FULL_SPEC: EXPECTED_USAGE},
            ))

        assert results[FULL_SPEC]["source"] == "existing_file"
        # Through `collect_usage`, so the checkpoint this run writes cannot
        # tell a reused slot's numbers from a freshly reported slot's.
        assert collect_usage(results) == {FULL_SPEC: EXPECTED_USAGE}

    def test_a_slot_that_really_re_ran_is_never_seeded_from_it(self, tmp_path):
        """The silent adapter re-runs, so its stale entry must not come back."""
        plan, _ = self._resumable(tmp_path, "review-plan")
        which, run = _dispatch()
        with which, run:
            results = asyncio.run(rob.run_all_models(
                [FULL_SPEC, NONE_SPEC], str(plan), "review {output_json_path}",
                lambda spec: {}, "review-plan", "plan_review",
                timeout=600, max_parallel=2, out_dir=str(tmp_path),
                prefix="plan", skip_existing=True,
                prior_usage={FULL_SPEC: EXPECTED_USAGE,
                             NONE_SPEC: {"cost_usd": 99.0}},
            ))

        assert results[NONE_SPEC]["source"] != "existing_file"
        assert NONE_SPEC not in collect_usage(results)

    def test_review_code_restores_it_into_its_own_collector(self, tmp_path):
        """The second lane, whose collector is an out-parameter rather than a
        key on each record."""
        import code_review_orchestrator as cro

        plan, _ = self._resumable(tmp_path, "code-review")
        usage = {}
        with patch.object(
            cro, "run_single_review",
            side_effect=_async_none_result(),
        ):
            asyncio.run(cro.run_all_reviews(
                [FULL_SPEC, NONE_SPEC], plan, tmp_path / "out", timeout=600,
                max_parallel=2, tracked_files=None, base_ref="HEAD~1",
                skip_existing=True, served_models={}, usage=usage,
                prior_usage={FULL_SPEC: EXPECTED_USAGE,
                             NONE_SPEC: {"cost_usd": 99.0}},
            ))

        assert usage == {FULL_SPEC: EXPECTED_USAGE}

    def test_nothing_recorded_leaves_the_resume_as_it_was(self, tmp_path):
        """A duration-only fan-out's resume gains no key from any of this."""
        plan, _ = self._resumable(tmp_path, "review-plan")
        which, run = _dispatch()
        with which, run:
            results = asyncio.run(rob.run_all_models(
                [FULL_SPEC], str(plan), "review {output_json_path}",
                lambda spec: {}, "review-plan", "plan_review",
                timeout=600, max_parallel=1, out_dir=str(tmp_path),
                prefix="plan", skip_existing=True, prior_usage=None,
            ))

        assert "usage" not in results[FULL_SPEC]
        assert collect_usage(results) == {}


# ---------------------------------------------------------------------------
# 5. the renderers
# ---------------------------------------------------------------------------

class TestUsageRendering:

    def test_the_clause_names_the_cost_as_an_estimate(self):
        assert describe_usage(EXPECTED_USAGE) == EXPECTED_CLAUSE
        assert "est." in describe_usage({"cost_usd": 3.0})

    def test_nothing_reported_renders_nothing(self):
        assert describe_usage(None) == ""
        assert describe_usage({}) == ""
        # A free or sub-cent run says nothing about money rather than "$0.00".
        assert describe_usage({"cost_usd": 0.0, "output_tokens": 12}) == "12 output tokens"

    def test_model_label_joins_the_attribution_it_finds(self):
        assert model_label(FULL_SPEC) == FULL_SPEC
        assert model_label(FULL_SPEC, None, {FULL_SPEC: EXPECTED_USAGE}) == (
            f"{FULL_SPEC} ({EXPECTED_CLAUSE})"
        )
        assert model_label(
            FULL_SPEC,
            {FULL_SPEC: {"served_model": "other", "fallback": True}},
            {FULL_SPEC: EXPECTED_USAGE},
        ) == f"{FULL_SPEC} (served by other, fallback; {EXPECTED_CLAUSE})"

    def test_sum_usage_adds_numbers_and_skips_everything_else(self):
        totals = sum_usage({
            "a": {"cost_usd": 0.01, "output_tokens": 5, "served_model": "x"},
            "b": {"cost_usd": 0.02, "output_tokens": 7, "tool_uses": True},
            "c": "not a block",
        })
        # Floats are rounded, the string is dropped, and `True` is not one use.
        assert totals == {"cost_usd": 0.03, "output_tokens": 12}
        assert sum_usage({}) == {}
        assert sum_usage(None) == {}


# ---------------------------------------------------------------------------
# 6. the consolidated report
# ---------------------------------------------------------------------------

class TestTheConsolidatedReport:
    """The second front page of the same run names the same numbers.

    ``generate_consolidated_html`` builds its payload independently of
    ``generate_html_report`` — a separate invocation, hours later, reading only
    ``grouped.json`` and the batch outputs. Nothing in those files remembers
    what a slot spent, so the numbers reach this report only because the
    consolidation pass reads them back off ``.status.json``. Left unwired, the
    template's two Stage U conditions are dead code and the consolidated report
    is the one an operator reads after a partial run.
    """

    # Two groups merged into one: the 10%-reduction floor below which
    # `generate_consolidated_json` deletes the report instead of writing it.
    CGS = [{
        "consolidated_id": "0123456789ab",
        "display_index": 1,
        "title": "Boundary check missing",
        "description": "d",
        "importance": "HIGH",
        "reference": "Step 1",
        "type": "modification",
        "is_singleton": False,
        "model_count": 1,
        "original_suggestion_count": 2,
        "reasoning": "One finding, reported twice.",
        "underlying_group_indices": [0, 1],
        "underlying_group_ids": ["h1", "h2"],
    }]
    GROUPS = [
        {
            "theme": "Boundary check missing",
            "models": [FULL_SPEC],
            "display_label": f"G{idx}",
            "group_hash": f"h{idx}",
            "suggestions": [dict(ISSUES[0], source_model=FULL_SPEC)],
        }
        for idx in (1, 2)
    ]

    def _plan_file(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# My Feature Plan\n", encoding="utf-8")
        return plan

    def _payload(self, html):
        """The injected ``reportData`` object, parsed back out of the page."""
        body = html.split("var reportData = ", 1)[1]
        return json.JSONDecoder().raw_decode(body)[0]

    def test_the_badge_and_the_header_name_what_was_spent(self, tmp_path):
        plan = self._plan_file(tmp_path)
        # A second slot that reported and then failed: it has no badge in this
        # report at all, and its spend still belongs in the run's total.
        usage = {FULL_SPEC: EXPECTED_USAGE, "usagefull:failed": EXPECTED_USAGE}

        html = Path(generate_consolidated_html(
            self.CGS, self.GROUPS, str(tmp_path), str(plan), [FULL_SPEC],
            usage=usage,
        )).read_text(encoding="utf-8")
        data = self._payload(html)

        assert data["modelMetadata"][FULL_SPEC]["usage"] == EXPECTED_USAGE
        assert data["modelMetadata"][FULL_SPEC]["usage_label"] == EXPECTED_CLAUSE
        assert data["usageTotal"] == sum_usage(usage)
        assert data["usageTotalLabel"] == describe_usage(sum_usage(usage))
        # The header is the run's total, not the badges' — twice one badge.
        assert data["usageTotal"]["output_tokens"] == 2 * EXPECTED_USAGE["output_tokens"]

    def test_a_silent_fan_out_leaves_the_payload_as_it_was(self, tmp_path):
        plan = self._plan_file(tmp_path)

        for usage in (None, {}, {NONE_SPEC: {}}):
            data = self._payload(Path(generate_consolidated_html(
                self.CGS, self.GROUPS, str(tmp_path), str(plan), [FULL_SPEC],
                usage=usage,
            )).read_text(encoding="utf-8"))

            assert "usageTotal" not in data
            assert "usageTotalLabel" not in data
            assert "usage" not in data["modelMetadata"][FULL_SPEC]
            assert "usage_label" not in data["modelMetadata"][FULL_SPEC]

    def test_the_consolidation_pass_reads_them_off_the_checkpoint(
        self, tmp_path, monkeypatch
    ):
        """End of the carriage: ``--reaggregate-consolidation`` → the page.

        Consolidation runs in its own invocation, so the checkpoint is the only
        place the numbers still are; this pins the read at the call site rather
        than trusting the generator's new argument to be passed.
        """
        import review_plan_orchestrator as rpo

        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        plan = self._plan_file(tmp_path)
        (phase_dir / "grouped.json").write_text(
            json.dumps(self.GROUPS), encoding="utf-8"
        )
        (phase_dir / "consolidation_tasks.json").write_text(
            json.dumps({"total_batches": 1, "batches": [], "singleton_sections": {}}),
            encoding="utf-8",
        )
        update_status(str(phase_dir), {
            "state": "models_complete", "usage": {FULL_SPEC: EXPECTED_USAGE},
        })
        monkeypatch.setattr(
            rpo, "merge_consolidation_results",
            lambda *_a, **_k: ([dict(cg) for cg in self.CGS], {"count": 0}),
        )

        rpo.reaggregate_consolidation_mode(
            str(phase_dir), str(plan), "my-feature", [FULL_SPEC]
        )
        html = (phase_dir / "consolidated-report.html").read_text(encoding="utf-8")

        assert EXPECTED_CLAUSE in html
        assert self._payload(html)["usageTotalLabel"] == EXPECTED_CLAUSE
