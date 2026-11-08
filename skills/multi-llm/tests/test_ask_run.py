"""``AskRun``: one ask slot, driven directly.

``ask_orchestrator.main`` is the end-to-end path (``test_ask.py``). These rows
call :meth:`AskRun.ask_one` and :meth:`AskRun.skip_existing_answer` on a run
built by hand, with the sandbox gate and ``invoke_with_provider`` faked, so the
answer-recovery ladder can be tested without the parse, detach and lock stages.

See docs/development/workflows.md#ask-mode.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ask_orchestrator  # noqa: E402
from ask_orchestrator import ASK_PROMPT_FILE, AskRun  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.json_extractor import sanitize_model_name  # noqa: E402
from utils.progress_tracker import FanoutProgress  # noqa: E402
from utils.prompt_loader import load_prompt  # noqa: E402

SPEC = "claude-code:sonnet"
STEM = sanitize_model_name(SPEC)
SANDBOXED = rob.SandboxVerdict(rob.SANDBOX_DISPATCH, None)


@pytest.fixture
def slot(tmp_path, monkeypatch):
    """A run over an empty question dir; ``slot["invoke"]`` answers the call."""
    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n\nBuild a widget.\n", encoding="utf-8")
    question_dir = tmp_path / "ask" / "q-12345678"
    question_dir.mkdir(parents=True)
    state = {"calls": [], "invoke": lambda **_: {"success": True, "error": None}}

    def invoke(**kwargs):
        state["calls"].append(kwargs)
        return state["invoke"](**kwargs)

    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_: SANDBOXED)
    monkeypatch.setattr(ask_orchestrator, "invoke_with_provider", invoke)

    def make(*, rerun_all=False):
        args = argparse.Namespace(timeout=60, max_parallel=1, rerun_all=rerun_all)
        return AskRun(
            args=args, question="Is it sound?", plan_path=str(plan),
            question_dir=str(question_dir), project_root=str(tmp_path),
            template=load_prompt(ASK_PROMPT_FILE), inline_plan=plan.read_text(),
            model_specs=[SPEC], unattended=True, skip_existing=not rerun_all,
            tracker=FanoutProgress(str(question_dir), [SPEC], anchor=str(plan),
                                   timeout=60, interval_s=60),
        )

    state.update(make=make, dir=question_dir,
                 answer=question_dir / f"answer_{STEM}.md",
                 partial=question_dir / f"partial_{STEM}.md")
    return state


def _ask(run):
    return asyncio.run(run.ask_one(SPEC, 0))


def test_an_answer_file_the_model_wrote_counts(slot):
    def invoke(**_):
        slot["answer"].write_text("The answer.\n", encoding="utf-8")
        return {"success": True, "error": None}

    slot["invoke"] = invoke
    run = slot["make"]()
    result = _ask(run)
    assert result == {"model": SPEC, "success": True, "error": None, "source": "file"}
    assert run.dispatched_specs == {SPEC}
    assert SPEC in run.provider_calls
    assert slot["calls"][0]["timeout"] == 60
    assert slot["calls"][0]["cwd"] == run.project_root


def test_a_stdout_answer_is_persisted_as_the_fallback(slot):
    slot["invoke"] = lambda **_: {"success": True, "error": None,
                                   "details": {"answer_text": "From stdout."}}
    result = _ask(slot["make"]())
    assert result["success"] and result["source"] == "fallback"
    assert "From stdout." in slot["answer"].read_text(encoding="utf-8")


def test_a_failed_run_keeps_its_text_as_a_partial(slot):
    slot["invoke"] = lambda **_: {"success": False, "error": "timed out",
                                   "error_code": "TIMEOUT",
                                   "details": {"answer_text": "Half an answer."}}
    run = slot["make"]()
    result = _ask(run)
    assert result == {"model": SPEC, "success": False, "error": "timed out",
                      "source": "failed"}
    assert not slot["answer"].exists()
    assert "Half an answer." in slot["partial"].read_text(encoding="utf-8")
    assert not run.answers_left_in_place


def test_a_call_that_raises_ends_the_slot_as_that_exception(slot):
    def invoke(**_):
        raise ValueError("boom")

    slot["invoke"] = invoke
    result = _ask(slot["make"]())
    assert result["success"] is False
    assert result["source"] == "exception"
    assert result["error"] == "Exception: ValueError: boom"


def test_a_refused_slot_never_dispatches_or_touches_the_last_answer(slot, monkeypatch):
    refused = rob.SandboxVerdict(rob.SANDBOX_REFUSE, "[SANDBOX_UNAVAILABLE] simulated")
    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_: refused)
    slot["answer"].write_text("An earlier answer.\n", encoding="utf-8")
    run = slot["make"](rerun_all=True)
    result = _ask(run)
    assert result["source"] == "not-dispatched"
    assert slot["calls"] == [] and run.dispatched_specs == set()
    assert slot["answer"].read_text(encoding="utf-8") == "An earlier answer.\n"


def test_skip_existing_answer_reads_the_answer_on_disk(slot):
    run = slot["make"]()
    assert run.skip_existing_answer(SPEC, 0) is None
    slot["answer"].write_text("Kept.\n", encoding="utf-8")
    assert run.skip_existing_answer(SPEC, 0) == {
        "model": SPEC, "success": True, "error": None, "source": "resume",
    }
    assert slot["make"](rerun_all=True).skip_existing_answer(SPEC, 0) is None
