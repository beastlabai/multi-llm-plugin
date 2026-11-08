"""ask's single-shot slot checks a provider call that raised, as the review
lanes check an attempt that raised.

``ask_one`` makes one ``invoke_with_provider`` call and has no retry ladder.
When that call raises, the slot still gets the checks a returned call gets
(the porcelain check on the uncontained rung, the mode and home checks on the
plugin's jail) before it is recorded as ``Exception: <type>: <msg>`` with
``source: exception``. The raising call returned no ``details``, so its
``$CODEX_HOME`` comparison is the lane's own read-back of the jailed children
it armed, and a home it cannot compare fails the slot closed.

Every test runs the real ``invoke_with_provider`` with only the codex spawn
faked and codex's parser raising, as in the Codex review's probe.

See docs/development/workflows.md#ask-mode and docs/development/providers.md.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ask_orchestrator  # noqa: E402
from utils import llm_client, progress_tracker  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.json_extractor import sanitize_model_name  # noqa: E402
from utils.output_handler import get_phase_dir, slugify_question  # noqa: E402
from utils.providers.base import (  # noqa: E402
    CODEX_HOME_CHECK_MARKER,
    MODE_CHECK_MARKER,
    SANDBOX_BYPASSED_MARKER,
)
from utils.providers.codex import CodexProvider  # noqa: E402

CODEX = "codex:gpt-5.5"
QUESTION = "Q"
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


def _codex_stream(*items):
    """A finished codex turn with a prose answer, after *items* (completed
    stream items, such as a ``file_change`` the read-only audit cannot clear)."""
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "t1"},
        *({"type": "item.completed", "item": item} for item in items),
        {"type": "item.completed", "item": {
            "id": "item_1", "type": "agent_message", "text": "An answer."}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    )) + "\n"


@pytest.fixture
def codex_slot(tmp_path, tmp_path_factory, monkeypatch):
    """A committed repo holding the plan, a source file and a script; outside
    it a ``$CODEX_HOME`` and the slot's scratch base. The codex spawn is
    faked by ``child(cmd)`` (set per test, for its side effects) and codex's
    parser raises, so the real ``invoke_with_provider`` raises after the
    child exits."""
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

    question_dir = Path(get_phase_dir(Path(os.path.abspath(plan)), "ask")) / (
        slugify_question(QUESTION)
    )
    state = {
        "repo": repo, "plan": plan, "home": home, "dir": question_dir,
        "answer": question_dir / f"answer_{sanitize_model_name(CODEX)}.md",
        "partial": question_dir / f"partial_{sanitize_model_name(CODEX)}.md",
        "child": lambda _cmd: None, "spawned": 0, "stream": _codex_stream(),
    }

    def fake_cli(cmd, **_kwargs):
        state["spawned"] += 1
        state["child"](cmd)
        return subprocess.CompletedProcess(cmd, 0, state["stream"], "")

    def exploding_parse(self, stdout, stderr):
        raise ValueError("parser exploded")

    monkeypatch.setattr(llm_client, "_run_provider_cli", fake_cli)
    monkeypatch.setattr(CodexProvider, "parse_output", exploding_parse)
    return state


def _ask(codex_slot, monkeypatch, verdict, *, extra=()):
    """``(rc, models_failed reason, progress entry)`` of one codex slot with
    the sandbox gate pinned to ``verdict``."""
    monkeypatch.setattr(ask_orchestrator, "sandbox_refusal", lambda *_a, **_k: verdict)
    rc = asyncio.run(ask_orchestrator.main([
        "--plan-file", str(codex_slot["plan"]), "--question", QUESTION,
        "--models", CODEX, *extra,
    ]))
    status = json.loads((codex_slot["dir"] / ".status.json").read_text(encoding="utf-8"))
    return (
        rc,
        status.get("models_failed", {}).get(CODEX),
        status["progress"]["models"][CODEX],
    )


class TestARaisingJailedCallKeepsItsHomeCheck:
    """``$CODEX_HOME`` outlives the jail: a call that raised is compared here."""

    def test_a_home_write_then_a_raise_fails_with_the_home_line(
        self, codex_slot, monkeypatch, capsys
    ):
        def child(_cmd):
            (codex_slot["home"] / "AGENTS.md").write_text("injected\n", encoding="utf-8")

        codex_slot["child"] = child
        rc, reason, entry = _ask(codex_slot, monkeypatch, JAILED)
        assert codex_slot["spawned"] == 1
        assert rc == 3, "no answer counted"
        assert reason.startswith(CODEX_HOME_CHECK_MARKER), reason
        assert "AGENTS.md" in reason and "fails closed" in reason
        assert (entry["state"], entry["source"]) == ("failed", "exception")
        assert (codex_slot["home"] / "AGENTS.md").exists(), "never reverted"
        out = capsys.readouterr().out
        assert EXCEPTION in out, "the exception is still named"
        # Repeated in the closing summary: the change outlives the run.
        assert out.count(CODEX_HOME_CHECK_MARKER) >= 2

    def test_a_raise_that_left_everything_alone_is_only_the_exception(
        self, codex_slot, monkeypatch, capsys
    ):
        rc, reason, entry = _ask(codex_slot, monkeypatch, JAILED)
        assert codex_slot["spawned"] == 1
        assert rc == 3
        assert reason == EXCEPTION
        assert reason != progress_tracker.DEADLINE_ERROR
        assert (entry["state"], entry["source"], entry["error"]) == (
            "failed", "exception", EXCEPTION,
        )
        out = capsys.readouterr().out
        assert CODEX_HOME_CHECK_MARKER not in out
        assert MODE_CHECK_MARKER not in out

    def test_a_home_that_cannot_be_compared_fails_closed(
        self, codex_slot, monkeypatch
    ):
        """The home gone after the child: nothing to compare against, so it
        is unaudited and takes a change's verdict."""
        codex_slot["child"] = lambda _cmd: shutil.rmtree(codex_slot["home"])
        rc, reason, entry = _ask(codex_slot, monkeypatch, JAILED)
        assert rc == 3
        assert reason.startswith(CODEX_HOME_CHECK_MARKER), reason
        assert "was not checked" in reason and "fails closed" in reason
        assert entry["source"] == "exception"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_a_mode_flip_then_a_raise_fails_the_jailed_slot(
        self, codex_slot, monkeypatch
    ):
        script = codex_slot["repo"] / "tests" / "run.sh"
        codex_slot["child"] = lambda _cmd: script.chmod(script.stat().st_mode | 0o111)
        rc, reason, entry = _ask(codex_slot, monkeypatch, JAILED)
        assert rc == 3
        assert MODE_CHECK_MARKER in reason and "tests/run.sh" in reason, reason
        assert entry["source"] == "exception"

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
            "success": True, "data": "An answer.",
        })
        rc, reason, entry = _ask(codex_slot, monkeypatch, JAILED)
        assert codex_slot["spawned"] == 1
        assert rc == 0 and reason is None
        assert entry["state"] == "done"


class TestARaisingUncontainedCallKeepsItsTreeCheck:
    """On the uncontained rung the porcelain check runs after a raise, and
    the raising call's missing audit can never clear a change."""

    def test_a_tree_write_then_a_raise_fails_the_slot_by_path(
        self, codex_slot, monkeypatch
    ):
        app = codex_slot["repo"] / "src" / "app.py"
        codex_slot["child"] = lambda _cmd: app.write_text("x = 2\n", encoding="utf-8")
        rc, reason, entry = _ask(codex_slot, monkeypatch, BYPASSED)
        assert rc == 3
        assert reason.startswith(SANDBOX_BYPASSED_MARKER), reason
        assert "src/app.py" in reason
        assert (entry["state"], entry["source"]) == ("failed", "exception")

    def test_a_raise_that_left_the_tree_alone_is_only_the_exception(
        self, codex_slot, monkeypatch
    ):
        rc, reason, entry = _ask(codex_slot, monkeypatch, BYPASSED)
        assert rc == 3
        assert reason == EXCEPTION
        assert entry["source"] == "exception"


class TestAnAnswerFileFromARaisingCallIsNeverCounted:
    """The call never finished, so an answer file it wrote is set aside as a
    partial (D3.3) and the slot stays failed with its exception."""

    def test_the_file_becomes_the_partial(self, codex_slot, monkeypatch):
        def child(_cmd):
            codex_slot["answer"].write_text("A half answer.\n", encoding="utf-8")

        codex_slot["child"] = child
        rc, reason, _entry = _ask(codex_slot, monkeypatch, JAILED)
        assert rc == 3
        assert reason.startswith(EXCEPTION), reason
        assert not codex_slot["answer"].exists()
        assert codex_slot["partial"].read_text(encoding="utf-8") == "A half answer.\n"


class TestCancellationIsNotASlotException:
    """``CancelledError`` and ``KeyboardInterrupt`` are not ``Exception``s:
    ask's slot does not record them, so the fan-out's teardown does."""

    def test_a_call_the_deadline_cut_is_a_deadline_loss(self, codex_slot, monkeypatch):
        """The fan-out's deadline (2 s at ``--timeout 1``) cancels the slot
        while its call is still running; the call raises only afterwards."""
        codex_slot["child"] = lambda _cmd: time.sleep(2.5)
        monkeypatch.setattr(rob, "PROVIDER_STAGGER_DELAY", 0.0)
        rc, reason, entry = _ask(codex_slot, monkeypatch, JAILED, extra=("--timeout", "1"))
        assert codex_slot["spawned"] == 1
        assert rc == 3
        assert reason == progress_tracker.DEADLINE_ERROR
        assert entry["error"] == progress_tracker.DEADLINE_ERROR

    def test_a_keyboard_interrupt_propagates(self, codex_slot, monkeypatch):
        def interrupted(self, func, /, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(ask_orchestrator.ProviderCall, "run", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _ask(codex_slot, monkeypatch, JAILED)


class TestAViolatedAnswerThatCannotBeSetAsideIsWithheld:
    """A slot failed by a check whose answer file cannot be moved to
    ``answer_<model>.md.violated`` keeps the file where it is, as evidence,
    and the closing scan withholds it (``answers_left_in_place``): the slot
    stays failed and the run exits 3. Both violation branches, for a call
    that raised and for one that returned an answer."""

    @pytest.fixture(params=["raised", "returned"])
    def call(self, request, codex_slot, monkeypatch):
        """``returned``: the call answers, and its stream holds a write, so
        its read-only audit cannot clear the change."""
        if request.param == "returned":
            monkeypatch.setattr(CodexProvider, "parse_output", lambda self, out, err: {
                "success": True, "data": "An answer.",
            })
            codex_slot["stream"] = _codex_stream({
                "id": "item_0", "type": "file_change", "status": "completed",
                "changes": [{"path": "src/app.py", "kind": "update"}],
            })
        return request.param

    def _blocked_answer(self, codex_slot):
        """The child writes its answer file; the quarantine name is taken by
        a directory, so ``os.replace`` onto it fails."""
        codex_slot["dir"].mkdir(parents=True, exist_ok=True)
        violated = Path(str(codex_slot["answer"]) + ".violated")
        violated.mkdir()
        (violated / "keep").write_text("", encoding="utf-8")
        return violated

    def _assert_withheld(self, codex_slot, rc, reason, entry, marker):
        assert codex_slot["spawned"] == 1
        assert rc == 3, "a violated answer is never counted"
        assert reason.startswith(marker), reason
        assert "could not be set aside" in reason and "(not counted)" in reason
        assert entry["state"] == "failed"
        assert codex_slot["answer"].read_text(encoding="utf-8") == "Left behind.\n", (
            "the evidence stays in place"
        )
        answers_md = (codex_slot["dir"] / "answers.md").read_text(encoding="utf-8")
        assert "Left behind." not in answers_md

    def test_the_porcelain_branch(self, codex_slot, monkeypatch, call):
        self._blocked_answer(codex_slot)
        app = codex_slot["repo"] / "src" / "app.py"

        def child(_cmd):
            app.write_text("x = 2\n", encoding="utf-8")
            codex_slot["answer"].write_text("Left behind.\n", encoding="utf-8")

        codex_slot["child"] = child
        rc, reason, entry = _ask(codex_slot, monkeypatch, BYPASSED)
        self._assert_withheld(codex_slot, rc, reason, entry, SANDBOX_BYPASSED_MARKER)
        assert "src/app.py" in reason

    def test_the_incidental_branch(self, codex_slot, monkeypatch, call):
        self._blocked_answer(codex_slot)

        def child(_cmd):
            (codex_slot["home"] / "AGENTS.md").write_text("injected\n", encoding="utf-8")
            codex_slot["answer"].write_text("Left behind.\n", encoding="utf-8")

        codex_slot["child"] = child
        rc, reason, entry = _ask(codex_slot, monkeypatch, JAILED)
        self._assert_withheld(codex_slot, rc, reason, entry, CODEX_HOME_CHECK_MARKER)
        assert "AGENTS.md" in reason
