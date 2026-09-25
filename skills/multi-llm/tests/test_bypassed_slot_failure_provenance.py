"""A bypassed slot that FAILS still says it ran without a sandbox (2026-09-25).

Dogfood run: a codex slot on the uncontained rung was failed by the porcelain
write check, and its result record was built without ``served_model_fields``.
So ``models_served`` had no ``sandbox: bypassed`` for it,
``progress.sandbox_bypassed`` was absent, and ``announce_retired_rungs`` never
printed the fan-out ``[SANDBOX_RUNG_RETIRED]`` line — the slot that most needed
its provenance was the one that lost it. The same omission sat on every other
failure return of ``run_single_model`` for a slot whose child actually ran, and
``invoke_with_provider`` wrote ``details["sandbox"]`` only on exit 0.

The fan-out tests drive the real ``review_plan_orchestrator.main()`` with the
provider subprocess stubbed and the sandbox probe pinned, as
``test_sandbox_rung_reporting.py`` does; the write check's delta is pinned at
``unexpected_writes`` because a git tree is not what is under test.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import review_plan_orchestrator as rpo  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import status_file  # noqa: E402
from utils.llm_client import invoke_with_provider  # noqa: E402
from utils.providers.base import (  # noqa: E402
    SANDBOX_BYPASSED_MARKER,
    SANDBOX_RUNG_RETIRED_MARKER,
)
from utils.providers.codex import LANDLOCK_RUNG  # noqa: E402
from utils.subagent_executor import SandboxProbeResult  # noqa: E402

SPEC = "codex:gpt-5.5"
BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
REMOVED_VERSION = LANDLOCK_RUNG.removed_at[0]
STRAY = "src/stray.py"

FINDINGS = [{
    "title": "Add input validation",
    "desc": "The plan does not validate the new email field.",
    "importance": "HIGH",
    "reference": "Overview",
    "type": "addition",
}]


def _stream() -> str:
    return "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "t1"}),
        json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(FINDINGS),
        }}),
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 10, "output_tokens": 5,
        }}),
    ])


def _completed(stdout, returncode=0, stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _pin_uncontained(monkeypatch):
    monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
    verdict = SandboxProbeResult(
        "unavailable", BWRAP_LINE, "unsandboxed",
        (("landlock", "retired"), ("unsandboxed", "taken")),
        (("landlock", "panicked"),), REMOVED_VERSION,
    )
    monkeypatch.setattr(base, "_fanout_sandbox_verdict", lambda _p, _cwd: verdict)


def _pin_a_stray_write(monkeypatch):
    """The porcelain delta the dogfood slot tripped: one path it did not own."""
    monkeypatch.setattr(base, "porcelain_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        base, "unexpected_writes", lambda *_a, **_k: [STRAY]
    )
    monkeypatch.setattr(base, "boundaryless_sibling_count", lambda *_a, **_k: 0)


def _run_review_plan(tmp_path, monkeypatch, *, prior_served=None):
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nAdd an email field with validation.\n", encoding="utf-8")
    _pin_uncontained(monkeypatch)
    _pin_a_stray_write(monkeypatch)
    phase_dir = base.get_phase_dir(plan, "review-plan")
    if prior_served is not None:
        Path(phase_dir).mkdir(parents=True, exist_ok=True)
        status_file.write_status(str(phase_dir), {
            "phase": "review-plan", "state": "complete",
            "models_served": prior_served,
        })
    argv = [
        "--plan-file", str(plan), "--models", SPEC, "--skip-validation",
        "--timeout", "3600", "--max-parallel", "1", "--rerun-all", "--force",
    ]
    with patch("shutil.which", return_value="/usr/bin/codex"), \
         patch("utils.llm_client.subprocess.run",
               side_effect=lambda cmd, **kw: _completed(_stream())), \
         patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), \
         pytest.raises(SystemExit) as exited:
        # The one slot failed, so the run is D6 row 3's exit 3 — which is also
        # the path that used to skip the `models_served` write entirely.
        asyncio.run(rpo.main(argv))
    assert exited.value.code == 3
    return Path(phase_dir)


class TestTheWriteCheckFailureKeepsItsProvenance:
    def test_models_served_and_progress_still_name_the_bypass(
        self, tmp_path, monkeypatch, capsys
    ):
        phase_dir = _run_review_plan(tmp_path, monkeypatch)
        out = capsys.readouterr().out

        # The slot really was failed by the write check, not something else.
        assert STRAY in out and "fails closed" in out
        assert SANDBOX_BYPASSED_MARKER in out
        status = status_file.read_status(str(phase_dir))
        assert status["models_served"] == {SPEC: {"sandbox": "bypassed"}}
        assert status["progress"]["sandbox_bypassed"] == [SPEC]

    def test_the_fanout_still_prints_the_retired_rung_line(
        self, tmp_path, monkeypatch, capsys
    ):
        _run_review_plan(
            tmp_path, monkeypatch,
            prior_served={SPEC: {"sandbox": "fallback-landlock"}},
        )
        out = capsys.readouterr().out
        assert STRAY in out
        retired = [ln for ln in out.splitlines()
                   if ln.startswith(SANDBOX_RUNG_RETIRED_MARKER)]
        assert len(retired) == 1, out
        assert "'landlock'" in retired[0] and SPEC in retired[0]


# ---------------------------------------------------------------------------
# run_single_model: both lanes, every failure that ran a child
# ---------------------------------------------------------------------------


def _slot(tmp_path, monkeypatch, *, out_dir, invoke_name, reply):
    """One ``run_single_model`` call on the uncontained rung, dispatch stubbed."""
    _pin_uncontained(monkeypatch)
    monkeypatch.setattr(base, "porcelain_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(base, "unexpected_writes", lambda *_a, **_k: [])
    monkeypatch.setattr(base, invoke_name, lambda *a, **kw: reply)
    plan = tmp_path / "plan.md"
    plan.write_text("plan", encoding="utf-8")
    return asyncio.run(base.run_single_model(
        model_spec=SPEC, plan_path=str(plan),
        prompt_template="{output_json_path}{output_location}", prompt_context={},
        phase_name="review-plan", invoke_phase="plan_review", timeout=1800,
        out_dir=out_dir, prefix="plan",
    ))


def _reply(*, success, error=None, error_code=None, served=None):
    details = {"sandbox": "bypassed", "stderr": "", "stdout_excerpt": ""}
    if served:
        details["served_model"] = served
    result = {"success": success, "details": details, "source": "stdout"}
    if success:
        result["data"] = json.dumps(FINDINGS)
    else:
        result["error"] = error or "codex exited 1"
        result["error_code"] = error_code or "SUBPROCESS_FAILED"
    return result


class TestEveryFailureThatRanRecordsTheSandbox:
    def test_the_stdout_lane_write_check_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, "unexpected_writes", lambda *_a, **_k: [STRAY])
        _pin_uncontained(monkeypatch)
        monkeypatch.setattr(base, "porcelain_snapshot", lambda *_a, **_k: None)
        monkeypatch.setattr(base, "boundaryless_sibling_count", lambda *_a, **_k: 0)
        monkeypatch.setattr(
            base, "invoke_with_provider",
            lambda *a, **kw: _reply(success=True, served="gpt-5.5-mini"),
        )
        plan = tmp_path / "plan.md"
        plan.write_text("plan", encoding="utf-8")
        result = asyncio.run(base.run_single_model(
            model_spec=SPEC, plan_path=str(plan),
            prompt_template="{output_json_path}{output_location}",
            prompt_context={}, phase_name="review-plan",
            invoke_phase="plan_review", timeout=1800, out_dir=None,
        ))
        assert result["success"] is False and STRAY in result["error"]
        assert result["sandbox"] == "bypassed"
        assert result["served_model"] == "gpt-5.5-mini"

    def test_the_stdout_lane_ordinary_failure(self, tmp_path, monkeypatch):
        result = _slot(
            tmp_path, monkeypatch, out_dir=None,
            invoke_name="invoke_with_provider", reply=_reply(success=False),
        )
        assert result["success"] is False
        assert result["sandbox"] == "bypassed"

    def test_the_file_lane_ordinary_failure(self, tmp_path, monkeypatch):
        result = _slot(
            tmp_path, monkeypatch, out_dir=str(tmp_path),
            invoke_name="invoke_with_file_output", reply=_reply(success=False),
        )
        assert result["success"] is False
        assert result["sandbox"] == "bypassed"

    def test_the_file_lane_context_overflow(self, tmp_path, monkeypatch):
        result = _slot(
            tmp_path, monkeypatch, out_dir=str(tmp_path),
            invoke_name="invoke_with_file_output",
            reply=_reply(success=False, error="context_length_exceeded: prompt too long"),
        )
        assert result["success"] is False
        assert result["sandbox"] == "bypassed"


# ---------------------------------------------------------------------------
# invoke_with_provider: the word rides a failed child too
# ---------------------------------------------------------------------------


class TestTheInvocationRecordsTheRungOnFailure:
    def _invoke(self, run, tmp_path):
        with patch("shutil.which", return_value="/usr/bin/codex"), \
             patch("utils.llm_client.subprocess.run", side_effect=run):
            return invoke_with_provider(
                prompt="review", model_spec=SPEC, timeout=60,
                cwd=str(tmp_path), sandbox_fallback="unsandboxed",
            )

    def test_a_non_zero_exit(self, tmp_path):
        result = self._invoke(
            lambda cmd, **kw: _completed("", returncode=1, stderr="boom"), tmp_path
        )
        assert result["success"] is False
        assert result["details"]["sandbox"] == "bypassed"

    def test_a_timeout_kill(self, tmp_path):
        def run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 60, output="", stderr="")
        result = self._invoke(run, tmp_path)
        assert result["success"] is False
        assert result["details"]["sandbox"] == "bypassed"

    def test_no_override_adds_no_key(self, tmp_path):
        with patch("shutil.which", return_value="/usr/bin/codex"), \
             patch("utils.llm_client.subprocess.run",
                   side_effect=lambda cmd, **kw: _completed("", returncode=1)):
            result = invoke_with_provider(
                prompt="review", model_spec=SPEC, timeout=60, cwd=str(tmp_path),
            )
        assert "sandbox" not in result["details"]
