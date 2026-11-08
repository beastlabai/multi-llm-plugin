"""Two fan-out slot endings the review lanes' ladder does not cover.

``run_single_model`` with no output directory makes one ``invoke_with_provider``
call, outside ``dispatch_with_ladder``. When that call raises, the slot still
gets the checks a returned call gets (the porcelain check on the uncontained
rung, the mode and home checks on the plugin's jail) before it is recorded as
the exception, as a raised ladder attempt is: its audit is missing, and its
``$CODEX_HOME`` comparison is the lane's own read-back of the jailed children
it armed. Those tests run the real ``invoke_with_provider`` with only the codex
spawn faked and codex's parser raising.

``run_models_concurrent`` harvests ``gather(return_exceptions=True)``: a slot
that raised ``CancelledError`` itself comes back as a cancellation object, not
an ``Exception``. It is named, left out of the results like a slot that
raised, and its siblings' results are kept; a cancel of the fan-out itself
still propagates.

See docs/development/providers.md and docs/development/workflows.md#ask-mode.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ask_orchestrator  # noqa: E402
from utils import llm_client, progress_tracker  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.output_handler import get_phase_dir, slugify_question  # noqa: E402
from utils.providers.base import (  # noqa: E402
    CODEX_HOME_CHECK_MARKER,
    MODE_CHECK_MARKER,
    SANDBOX_BYPASSED_MARKER,
)
from utils.providers.codex import CodexProvider  # noqa: E402

CODEX = "codex:gpt-5.5"
OTHER_CODEX = "codex:gpt-5.4"
EXCEPTION = "Exception: ValueError: parser exploded"

BYPASSED = rob.SandboxVerdict(
    rob.SANDBOX_DISPATCH_UNSANDBOXED, f"{SANDBOX_BYPASSED_MARKER} simulated",
    "unsandboxed",
)
JAILED = rob.SandboxVerdict(
    rob.SANDBOX_DISPATCH_UNSANDBOXED, "[SANDBOX_INCIDENTAL] simulated",
    "plugin-landlock", incidental=True,
)


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), check=True, capture_output=True, text=True,
    )


def _codex_stream(text):
    """A finished codex turn whose answer is *text*."""
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.completed", "item": {
            "id": "item_1", "type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    )) + "\n"


@pytest.fixture
def codex_slot(tmp_path, tmp_path_factory, monkeypatch):
    """A committed repo holding the plan, a source file and a script; outside
    it a ``$CODEX_HOME`` and the slot's scratch base. The codex spawn is
    faked by ``child(cmd)`` (set per test, for its side effects), and codex's
    parser raises unless a test restores one."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    plan = repo / "plan.md"
    plan.write_text("# plan\n\nBuild a widget.\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    outside = tmp_path_factory.mktemp("outside")
    home = outside / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    slot_base = outside / "slot-tmp"
    slot_base.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(slot_base))
    monkeypatch.setenv("HOME", str(outside))
    monkeypatch.setenv("CODEX_HOME", str(home))
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, *a, **k: "/usr/bin/codex" if name == "codex" else real_which(name, *a, **k),
    )
    state = {
        "repo": repo, "plan": plan, "home": home,
        "child": lambda _cmd: None, "spawned": 0, "answer": "[]",
    }

    def fake_cli(cmd, **_kwargs):
        state["spawned"] += 1
        state["child"](cmd)
        return subprocess.CompletedProcess(cmd, 0, _codex_stream(state["answer"]), "")

    def exploding_parse(self, stdout, stderr):
        raise ValueError("parser exploded")

    monkeypatch.setattr(llm_client, "_run_provider_cli", fake_cli)
    monkeypatch.setattr(CodexProvider, "parse_output", exploding_parse)
    return state


# ---------------------------------------------------------------------------
# run_single_model's single dispatch (no output directory)
# ---------------------------------------------------------------------------


def _single(codex_slot, monkeypatch, verdict):
    """One ``run_single_model`` slot with no output directory, the sandbox
    gate pinned to *verdict*."""
    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: verdict)
    return asyncio.run(rob.run_single_model(
        model_spec=CODEX, plan_path=str(codex_slot["plan"]),
        prompt_template="{output_json_path}{output_location}{working_limits}",
        prompt_context={}, phase_name="review-plan", invoke_phase="plan_review",
        timeout=3600, out_dir=None,
    ))


class TestARaisingSingleDispatchKeepsItsChecks:

    def test_a_home_write_then_a_raise_fails_with_the_home_line(
        self, codex_slot, monkeypatch, capsys
    ):
        def child(_cmd):
            (codex_slot["home"] / "AGENTS.md").write_text("injected\n", encoding="utf-8")

        codex_slot["child"] = child
        result = _single(codex_slot, monkeypatch, JAILED)
        assert codex_slot["spawned"] == 1
        assert result["success"] is False and result["output"] is None
        assert result["error"].startswith(CODEX_HOME_CHECK_MARKER), result["error"]
        assert "AGENTS.md" in result["error"] and "fails closed" in result["error"]
        assert (codex_slot["home"] / "AGENTS.md").exists(), "never reverted"

    def test_a_home_that_cannot_be_compared_fails_closed(self, codex_slot, monkeypatch):
        codex_slot["child"] = lambda _cmd: shutil.rmtree(codex_slot["home"])
        result = _single(codex_slot, monkeypatch, JAILED)
        assert result["error"].startswith(CODEX_HOME_CHECK_MARKER), result["error"]
        assert "was not checked" in result["error"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_a_mode_flip_then_a_raise_fails_the_jailed_slot(
        self, codex_slot, monkeypatch
    ):
        script = codex_slot["repo"] / "tests" / "run.sh"
        codex_slot["child"] = lambda _cmd: script.chmod(script.stat().st_mode | 0o111)
        result = _single(codex_slot, monkeypatch, JAILED)
        assert result["success"] is False
        assert MODE_CHECK_MARKER in result["error"], result["error"]
        assert "tests/run.sh" in result["error"]

    def test_a_tree_write_then_a_raise_fails_the_bypassed_slot_by_path(
        self, codex_slot, monkeypatch
    ):
        app = codex_slot["repo"] / "src" / "app.py"
        codex_slot["child"] = lambda _cmd: app.write_text("x = 2\n", encoding="utf-8")
        result = _single(codex_slot, monkeypatch, BYPASSED)
        assert result["success"] is False
        assert result["error"].startswith(SANDBOX_BYPASSED_MARKER), result["error"]
        assert "src/app.py" in result["error"]

    @pytest.mark.parametrize("verdict", [JAILED, BYPASSED], ids=["jailed", "bypassed"])
    def test_a_raise_that_left_everything_alone_is_only_the_exception(
        self, codex_slot, monkeypatch, capsys, verdict
    ):
        result = _single(codex_slot, monkeypatch, verdict)
        assert codex_slot["spawned"] == 1
        assert result["success"] is False
        assert result["error"] == EXCEPTION
        out = capsys.readouterr().out
        assert CODEX_HOME_CHECK_MARKER not in out and MODE_CHECK_MARKER not in out

    def test_a_returned_call_reads_nothing_back_in_the_lane(
        self, codex_slot, monkeypatch
    ):
        """The normal path is the invoker's own check: the lane's read-back
        runs only for a call that raised."""
        monkeypatch.setattr(
            rob, "raised_attempt_home_checks",
            lambda _armed: pytest.fail("read back after a returned call"),
        )
        monkeypatch.setattr(CodexProvider, "parse_output", lambda self, out, err: {
            "success": True, "data": [],
        })
        result = _single(codex_slot, monkeypatch, JAILED)
        assert codex_slot["spawned"] == 1
        assert result["success"] is True, result["error"]


# ---------------------------------------------------------------------------
# run_models_concurrent: a slot that cancelled itself
# ---------------------------------------------------------------------------


SPECS = ["gemini:gemini-3-flash", "gemini:gemini-3-pro"]


class TestASelfCancelledSlotIsNamedAndItsSiblingsKept:

    def test_the_sibling_is_harvested(self, capsys):
        async def slot(spec, _index):
            if spec == SPECS[0]:
                raise asyncio.CancelledError()
            return {"success": True, "source": "file", "error": None}

        results = asyncio.run(rob.run_models_concurrent(SPECS, slot, None, 5))
        assert results == {SPECS[1]: {"success": True, "source": "file", "error": None}}
        assert f"[{SPECS[0]}] Task cancelled" in capsys.readouterr().out

    def test_the_map_retires_it_as_cancelled(self, tmp_path):
        tracker = progress_tracker.FanoutProgress(
            tmp_path, SPECS, interval_s=0, timeout=60
        )

        async def slot(spec, _index):
            if spec == SPECS[1]:
                raise asyncio.CancelledError()
            return {"success": True, "source": "file", "error": None}

        results = asyncio.run(rob.run_models_concurrent(
            SPECS, slot, None, 5, tracker=tracker,
        ))
        assert list(results) == [SPECS[0]]
        models = tracker.snapshot()
        assert models[SPECS[0]]["state"] == "done"
        assert (models[SPECS[1]]["state"], models[SPECS[1]]["error"]) == (
            "failed", progress_tracker.CANCELLED_ERROR,
        )

    def test_a_cancel_of_the_fan_out_still_propagates(self):
        async def drive():
            started = asyncio.Event()

            async def slot(_spec, _index):
                started.set()
                await asyncio.Event().wait()

            task = asyncio.ensure_future(rob.run_models_concurrent(SPECS, slot, None, 5))
            await started.wait()
            task.cancel()
            await task

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(drive())

    def test_an_ask_run_answers_with_the_sibling(self, codex_slot, monkeypatch):
        """Through ask's real slot and invoker: one codex call surfaces a
        ``CancelledError`` from its worker thread, the other answers."""
        monkeypatch.setattr(CodexProvider, "parse_output", lambda self, out, err: {
            "success": True, "data": "An answer.",
        })
        codex_slot["answer"] = "An answer."

        def child(cmd):
            if any("gpt-5.4" in str(part) for part in cmd):
                raise asyncio.CancelledError()

        codex_slot["child"] = child
        monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: JAILED)
        rc = asyncio.run(ask_orchestrator.main([
            "--plan-file", str(codex_slot["plan"]), "--question", "Q",
            "--models", CODEX, OTHER_CODEX,
        ]))
        question_dir = Path(
            get_phase_dir(Path(os.path.abspath(codex_slot["plan"])), "ask")
        ) / slugify_question("Q")
        status = json.loads((question_dir / ".status.json").read_text(encoding="utf-8"))
        assert codex_slot["spawned"] == 2
        assert rc == 0 and status["state"] == "partial", "one of two answered"
        assert status["models_completed"] == [CODEX]
        assert OTHER_CODEX in status["models_failed"]
        models = status["progress"]["models"]
        assert models[CODEX]["state"] == "done"
        assert (models[OTHER_CODEX]["state"], models[OTHER_CODEX]["error"]) == (
            "failed", progress_tracker.CANCELLED_ERROR,
        )
