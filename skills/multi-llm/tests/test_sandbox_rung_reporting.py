"""Landlock rung, T5 — reporting truthfully separates rung 1 from rung 2.

A slot on the **contained** ``landlock`` rung ran under a real kernel boundary,
so no artifact of it may say it ran without one: no ``[SANDBOX_BYPASSED]``
line, no ``sandbox: bypassed`` in ``models_served``, no ``(ran unsandboxed: …)``
report label, no *unsandboxed* on ``status``. It gets its own word instead —
``[SANDBOX_FALLBACK]`` and ``sandbox: fallback-landlock``. A slot on the
**uncontained** rung keeps every one of those markers exactly as before.

The companion guards live beside ``tests/test_served_model_reporting.py`` (the
carriage) and ``tests/test_sandbox_fallback_ladder.py`` (T1–T4). This file
drives the real ``review_plan_orchestrator.main()`` and ``ask_orchestrator``
with one seam each — the provider subprocess, or the invoker — and the
sandbox probe verdict pinned, because the probe is not what is under test.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ask_orchestrator  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import run_status, status_file  # noqa: E402
from utils.html_report_generator import (  # noqa: E402
    SANDBOX_BYPASSED_LABEL,
    served_model_label,
)
from utils.llm_client import (  # noqa: E402
    invoke_with_provider,
    sandbox_fallback_rung_of,
    sandbox_record_word,
)
from utils.output_handler import get_phase_dir, slugify_question  # noqa: E402
from utils.providers.base import (  # noqa: E402
    SANDBOX_BYPASSED_MARKER,
    SANDBOX_FALLBACK_MARKER,
    SANDBOX_RUNG_RETIRED_MARKER,
)
from utils.providers.codex import (  # noqa: E402
    LANDLOCK_GLOBAL_PANIC,
    LANDLOCK_GLOBAL_PANIC_REASON,
    LANDLOCK_RUNG,
    CodexProvider,
)
from utils.subagent_executor import SandboxProbeResult  # noqa: E402

BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
SPEC = "codex:gpt-5.5"
VERSION, DEPRECATION = LANDLOCK_RUNG.deprecated_at
REMOVED_VERSION = LANDLOCK_RUNG.removed_at[0]
#: codex 0.156.1's rung-1 panic line, as the walk's probe tail keeps it.
PANIC_TAIL = (
    "thread 'main' panicked at linux-sandbox/src/linux_run_main.rs:410:5: "
    "filesystem-restricted execution requires bubblewrap to isolate app-server sockets"
)

FINDINGS = [{
    "title": "Add input validation",
    "desc": "The plan does not validate the new email field.",
    "importance": "HIGH",
    "reference": "Overview",
    "type": "addition",
}]

#: The deprecation item, verbatim in shape from the 2026-09-22 capture: every
#: rung-1 stream carries it, healthy runs included.
DEPRECATION_ITEM = json.dumps({"type": "item.completed", "item": {
    "id": "item_0", "type": "error",
    "message": f"{DEPRECATION} Use the default sandbox instead.",
}})


def _stream(*, deprecation: bool) -> str:
    lines = [json.dumps({"type": "thread.started", "thread_id": "t1"})]
    if deprecation:
        lines.append(DEPRECATION_ITEM)
    lines += [
        json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(FINDINGS),
        }}),
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 10, "output_tokens": 5,
        }}),
    ]
    return "\n".join(lines)


def _completed(stdout, returncode=0, stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _pin_probe(monkeypatch, rung):
    """The fan-out's probe answers ``unavailable`` and the walk selected
    ``rung``; everything downstream of it — the key, the gate, the notice, the
    argv, the records — is the real code."""
    monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
    if rung == "unsandboxed":
        # What codex 0.156.1 measures (2026-09-24): rung 1 probed, its panic
        # classified by text, then the uncontained rung.
        verdict = SandboxProbeResult(
            "unavailable", BWRAP_LINE, rung,
            (("landlock", "retired"), (rung, "taken")),
            (("landlock", PANIC_TAIL),), REMOVED_VERSION,
        )
    else:
        verdict = SandboxProbeResult("unavailable", BWRAP_LINE, rung, ((rung, "taken"),))
    monkeypatch.setattr(base, "_fanout_sandbox_verdict", lambda _provider, _cwd: verdict)


# ---------------------------------------------------------------------------
# The record word, one mapping
# ---------------------------------------------------------------------------


class TestTheRecordWord:
    def test_only_the_uncontained_rung_is_bypassed(self):
        assert sandbox_record_word("bypassed") == "bypassed"
        assert sandbox_record_word("unsandboxed") == "bypassed"
        assert sandbox_record_word("landlock") == "fallback-landlock"
        # Idempotent: a word already recorded maps to itself, so every hop —
        # result record, `models_served`, reaggregate — may run it.
        assert sandbox_record_word("fallback-landlock") == "fallback-landlock"
        assert sandbox_fallback_rung_of("fallback-landlock") == "landlock"
        assert sandbox_fallback_rung_of("bypassed") is None

    @pytest.mark.parametrize("value", [None, "", "anything-else", "fallback-x", True, 1])
    def test_anything_undeclared_is_dropped(self, value):
        assert sandbox_record_word(value) is None


# ---------------------------------------------------------------------------
# The reporting guard: a stubbed review-plan fan-out, rung 1 vs rung 2
# ---------------------------------------------------------------------------


def _run_review_plan(tmp_path, monkeypatch, rung, *, prior_served=None):
    """``review_plan_orchestrator.main()`` for real over one codex slot, with
    the probe pinned and the provider subprocess stubbed. Returns
    ``(phase_dir, stdout, argv_seen, details_seen)``."""
    plan = tmp_path / "auth-feature.md"
    if not plan.exists():
        plan.write_text("# Plan\n\nAdd an email field with validation.\n", encoding="utf-8")
    _pin_probe(monkeypatch, rung)
    phase_dir = base.get_phase_dir(plan, "review-plan")
    if prior_served is not None:
        os.makedirs(phase_dir, exist_ok=True)
        status_file.write_status(str(phase_dir), {
            "phase": "review-plan", "state": "complete",
            "models_served": prior_served,
        })
    seen = {"argv": [], "details": []}
    real_invoke = base.invoke_with_file_output

    def recording_invoke(**kwargs):
        result = real_invoke(**kwargs)
        seen["details"].append(dict(result.get("details") or {}))
        return result

    def fake_run(cmd, **kwargs):
        # Only the dispatch itself — `codex exec …`; any other probe the lane
        # runs through the same module (a version check) is not the slot.
        if "exec" in list(cmd)[:3]:
            seen["argv"].append(list(cmd))
        return _completed(_stream(deprecation=rung == "landlock"))

    argv = [
        "--plan-file", str(plan), "--models", SPEC, "--skip-validation",
        "--timeout", "3600", "--max-parallel", "1", "--rerun-all", "--force",
    ]
    with patch("shutil.which", return_value="/usr/bin/codex"), \
         patch("utils.llm_client.subprocess.run", side_effect=fake_run), \
         patch.object(base, "invoke_with_file_output", side_effect=recording_invoke), \
         patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
        exit_code = asyncio.run(rpo.main(argv))
    assert exit_code == 0
    return Path(phase_dir), seen


class TestTheReportingPathSeparatesTheRungs:
    def test_rung_one_is_reported_as_a_contained_fallback_everywhere(
        self, tmp_path, monkeypatch, capsys
    ):
        phase_dir, seen = _run_review_plan(tmp_path, monkeypatch, "landlock")
        out = capsys.readouterr().out

        # The argv really took rung 1, and the invocation says which rung.
        (argv,) = seen["argv"]
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert "features.use_legacy_landlock=true" in argv
        assert seen["details"][0]["sandbox"] == "landlock"

        # The per-slot notice: the rung, its deprecation, and the warning that
        # the capture's `"type":"error"` item is not the failure.
        notice = next(line for line in out.splitlines() if SANDBOX_FALLBACK_MARKER in line)
        assert "'landlock' fallback rung" in notice
        assert VERSION in notice and DEPRECATION in notice
        assert '"type":"error"' in notice and "not the failure" in notice
        assert "[SANDBOX_BYPASSED]" not in out

        status = status_file.read_status(str(phase_dir))
        assert status["models_served"] == {SPEC: {"sandbox": "fallback-landlock"}}
        progress = status["progress"]
        assert "sandbox_bypassed" not in progress
        assert progress["sandbox_fallback"] == {SPEC: "landlock"}

        label = served_model_label(SPEC, status["models_served"])
        assert "unsandboxed" not in label
        assert label == f"{SPEC} (sandboxed by fallback rung: landlock)"

        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        html = (phase_dir / "report.html").read_text(encoding="utf-8")
        assert SANDBOX_BYPASSED_LABEL not in report
        assert SANDBOX_BYPASSED_LABEL not in html
        assert "(unsandboxed)" not in html
        assert "sandboxed by fallback rung: landlock" in report
        report_data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert "(unsandboxed)" not in json.dumps(report_data)
        meta = report_data["modelMetadata"][SPEC]
        assert meta["sandbox"] == "fallback-landlock"
        assert meta["model"].endswith("(sandboxed by fallback rung: landlock)")

        rendered = run_status._progress_text(progress)
        assert "unsandboxed" not in rendered
        assert f"sandbox-fallback landlock {SPEC}" in rendered

    def test_rung_two_keeps_every_marker_it_had(self, tmp_path, monkeypatch, capsys):
        phase_dir, seen = _run_review_plan(tmp_path, monkeypatch, "unsandboxed")
        out = capsys.readouterr().out

        (argv,) = seen["argv"]
        assert argv[argv.index("--sandbox") + 1] == "danger-full-access"
        assert seen["details"][0]["sandbox"] == "bypassed"
        # The fixed stem, with the rung clause appended (plan §3.1): which
        # contained rung was tried, its outcome and its probe tail.
        assert (
            f"[{SPEC}] [SANDBOX_BYPASSED] codex: bubblewrap cannot start; running "
            f"--sandbox danger-full-access under providers.codex.sandbox_fallback=auto "
            f"(read-only phase; no OS boundary between the model's shell and this "
            f"machine); fallback rung 'landlock' unavailable: requires bubblewrap "
            f"in this codex build (codex 0.156.1: \"{PANIC_TAIL}\")"
        ) in out
        assert SANDBOX_FALLBACK_MARKER not in out

        status = status_file.read_status(str(phase_dir))
        assert status["models_served"] == {SPEC: {"sandbox": "bypassed"}}
        assert status["progress"]["sandbox_bypassed"] == [SPEC]
        assert "sandbox_fallback" not in status["progress"]
        assert served_model_label(SPEC, status["models_served"]) == (
            f"{SPEC} ({SANDBOX_BYPASSED_LABEL})"
        )
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        # The usage clause joins the same parenthesis, after the label.
        assert f"**Models:** {SPEC} ({SANDBOX_BYPASSED_LABEL}" in report
        report_data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert report_data["modelMetadata"][SPEC]["sandbox"] == "bypassed"
        assert report_data["modelMetadata"][SPEC]["model"].endswith(" (unsandboxed)")
        assert f"unsandboxed {SPEC}" in run_status._progress_text(status["progress"])


# ---------------------------------------------------------------------------
# Open question 2: a contained rung that stopped working is named once
# ---------------------------------------------------------------------------


class TestTheRetiredRungIsNamedOnce:
    def test_contained_then_uncontained_names_the_rung_with_its_expiry(
        self, tmp_path, monkeypatch, capsys
    ):
        _run_review_plan(
            tmp_path, monkeypatch, "unsandboxed",
            prior_served={SPEC: {"sandbox": "fallback-landlock"}},
        )
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if "no longer usable" in line]
        assert len(lines) == 1
        (line,) = lines
        assert "'landlock'" in line and SPEC in line
        assert VERSION in line and DEPRECATION in line
        assert "sandbox_fallback=contained" in line
        # Its own marker: [SANDBOX_BYPASSED] stays one line per bypassed slot.
        assert line.startswith(SANDBOX_RUNG_RETIRED_MARKER)
        assert SANDBOX_BYPASSED_MARKER not in line
        assert out.count(SANDBOX_BYPASSED_MARKER) == 1

        # The next run compares `bypassed` with `bypassed`: silence.
        _run_review_plan(tmp_path, monkeypatch, "unsandboxed")
        assert "no longer usable" not in capsys.readouterr().out

    @pytest.mark.parametrize("prior", [
        None,
        {SPEC: {"sandbox": "bypassed"}},
        {"claude-code:sonnet": {"sandbox": "fallback-landlock"}},
    ])
    def test_no_transition_says_nothing(self, prior):
        assert base.announce_retired_rungs(prior, {SPEC: {"sandbox": "bypassed"}}) == []

    def test_still_contained_says_nothing(self):
        prior = {SPEC: {"sandbox": "fallback-landlock"}}
        assert base.announce_retired_rungs(prior, prior) == []


# ---------------------------------------------------------------------------
# ask: no models_served, so its only rung surface is the per-slot line
# ---------------------------------------------------------------------------


class _AskInvoker:
    def __init__(self):
        self.seen = []

    def __call__(self, prompt, model_spec, timeout=None, log_file=None, cwd=None,
                 anchor=None, want_answer_text=False, **kwargs):
        # Every ask dispatch asks for the reduced answer text and step 2 reads
        # only that key (plans/ask-answer-recovery-and-false-success.md D2.3),
        # so the answer rides `details["answer_text"]` as it does for real;
        # `seen` keeps only the sandbox kwargs this test is about.
        assert want_answer_text is True
        self.seen.append(kwargs)
        return {"success": True, "data": "An answer.",
                "details": {"answer_text": "An answer."}}


class TestTheAskLaneSplitsTheRungs:
    @pytest.mark.parametrize("rung, marker, absent", [
        ("landlock", "[SANDBOX_FALLBACK]", "[SANDBOX_BYPASSED]"),
        ("unsandboxed", "[SANDBOX_BYPASSED]", "[SANDBOX_FALLBACK]"),
    ])
    def test_each_rung_prints_its_own_line_and_invents_no_models_served(
        self, tmp_path, monkeypatch, capsys, rung, marker, absent
    ):
        plan = tmp_path / "my-feature.md"
        plan.write_text("# My Feature\n\n## Overview\nBuild a widget.\n", encoding="utf-8")
        _pin_probe(monkeypatch, rung)
        invoker = _AskInvoker()
        with patch.object(ask_orchestrator, "invoke_with_provider", invoker):
            rc = asyncio.run(ask_orchestrator.main([
                "--plan-file", str(plan), "--question", "Q", "--models", SPEC,
            ]))
        assert rc == 0
        assert invoker.seen == [{"sandbox_fallback": rung}]
        out = capsys.readouterr().out
        assert out.count(marker) == 1
        assert absent not in out
        if rung == "landlock":
            assert DEPRECATION in out and "not the failure" in out
        qdir = Path(get_phase_dir(Path(os.path.abspath(plan)), "ask")) / slugify_question("Q")
        status = json.loads((qdir / ".status.json").read_text(encoding="utf-8"))
        assert "models_served" not in status
        assert status["models_completed"] == [SPEC]


# ---------------------------------------------------------------------------
# Constraint 2: the global-landlock panic is named by its cause
# ---------------------------------------------------------------------------

PANIC_STDERR = (
    "thread 'main' panicked at linux-sandbox/src/linux_run_main.rs:392:9:\n"
    "permission profiles requiring direct runtime enforcement are incompatible\n"
    "with --use-legacy-landlock\n"
    "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
)


class TestTheGlobalLandlockPanicIsNamed:
    def test_the_adapter_maps_the_panic_to_its_cause_even_when_wrapped(self):
        reason = CodexProvider().describe_failure("", PANIC_STDERR)
        assert reason == LANDLOCK_GLOBAL_PANIC_REASON
        assert "use_legacy_landlock" in reason and "config.toml" in reason
        assert "never sets" in reason and "cannot unset" in reason
        assert "workspace-write" in reason
        assert CodexProvider().describe_failure("", "some other crash") is None
        assert LANDLOCK_GLOBAL_PANIC.startswith("permission profiles")

    def test_lane_a_reports_the_cause_rather_than_an_opaque_exit(self):
        with patch("shutil.which", return_value="/usr/bin/codex"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_completed("", returncode=101, stderr=PANIC_STDERR)):
            result = invoke_with_provider("p", SPEC)
        assert result["success"] is False
        assert result["error"].startswith("codex exited with code 101: ")
        assert "use_legacy_landlock" in result["error"]
        assert "config.toml" in result["error"]

    def test_lane_b_reports_the_cause_on_the_runner_error_rung(self):
        from utils.subagent_executor import ResultRejected, extract_result

        with pytest.raises(ResultRejected) as caught:
            extract_result(CodexProvider(), "", PANIC_STDERR, prefer_arrays=False)
        assert caught.value.status == "runner_error"
        assert LANDLOCK_GLOBAL_PANIC_REASON in caught.value.message

    def test_the_mapping_moves_no_classification(self):
        provider = CodexProvider()
        assert provider.classify_failure("", PANIC_STDERR, 101) == (
            provider.classify_failure("", "some other crash", 101)
        )


# ---------------------------------------------------------------------------
# The deprecation error item is not a failure signal to the classifier
# ---------------------------------------------------------------------------


class TestTheDeprecationItemIsNotAVerdict:
    """codex owns that sentence, so its non-effect on the classifier is pinned
    on the stream rather than trusted to stay unmatched."""

    @pytest.mark.parametrize("exit_code", [0, 1])
    def test_the_verdict_is_what_the_stream_gets_without_it(self, exit_code):
        provider = CodexProvider()
        with_item = _stream(deprecation=True)
        without = _stream(deprecation=False)
        assert provider.classify_failure(with_item, "", exit_code) == (
            provider.classify_failure(without, "", exit_code)
        )
        # And on a failing stream that carries nothing but the item.
        bare = "\n".join([DEPRECATION_ITEM])
        assert provider.classify_failure(bare, "", 1) == provider.classify_failure("", "", 1)

    def test_a_healthy_rung_one_stream_still_parses_to_the_answer(self):
        parsed = CodexProvider().parse_output(_stream(deprecation=True), "")
        assert parsed["success"] is True
        assert parsed["data"] == FINDINGS
