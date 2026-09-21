#!/usr/bin/env python3
"""Tests for the ``ask`` mode (ask_orchestrator + slugify_question + shared
concurrency helper + SKILL/instructions routing).

No live CLI calls — provider invocation is mocked via a FakeInvoker that writes
the answer file and/or returns a canned provider-result dict. End-to-end live
behavior is out of scope (would be marked ``live``).
"""

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import ask_orchestrator
from ask_orchestrator import (
    MAX_INLINE_PLAN_BYTES,
    MAX_PROMPT_BYTES,
    build_inline_plan,
    demote_headings,
    looks_like_ndjson,
    parse_args,
    recover_answer_text,
    render_prompt,
    render_prompt_guarded,
    resolve_question,
    resolve_question_dir,
    write_answers_md,
)
from utils.output_handler import get_phase_dir, slugify_question
from utils.json_extractor import sanitize_model_name
from utils.review_orchestrator_base import run_models_concurrent


SKILL_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    """Make staggered launches near-instant for flow tests."""
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")


@pytest.fixture
def plan(tmp_path):
    """A small plan file in an isolated temp dir."""
    p = tmp_path / "my-feature.md"
    p.write_text("# My Feature\n\n## Overview\nBuild a widget with rollback.\n", encoding="utf-8")
    return p


def ask_root(plan_path):
    return Path(get_phase_dir(Path(os.path.abspath(plan_path)), "ask"))


def qdir(plan_path, question):
    return ask_root(plan_path) / slugify_question(question)


def read_answers(question_dir):
    return (Path(question_dir) / "answers.md").read_text(encoding="utf-8")


def read_status(question_dir):
    return json.loads((Path(question_dir) / ".status.json").read_text(encoding="utf-8"))


def extract_output_path(prompt):
    """Extract the exact output_md_path the model is instructed to write — the
    first non-empty line after the 'exact absolute file path:' marker. Robust to
    temp dir names that happen to contain 'answer'.

    Returns None when the prompt carries no such instruction at all, which is
    the rendering an adapter declaring ``supports_file_output = False`` gets: it
    is asked for stdout and is never handed a path."""
    parts = prompt.split("exact absolute file path:", 1)
    if len(parts) < 2:
        return None
    for line in parts[1].splitlines():
        if line.strip():
            return line.strip()
    return None


class FakeInvoker:
    """Stand-in for ``invoke_with_provider``.

    Per-model behavior is configured via :meth:`configure`:
      - ``write_file`` (bool): write the answer file (the exact ``output_md_path``
        from the prompt) before returning — simulates an agentic CLI that wrote
        the file.
      - ``content`` (str): the file content to write.
      - ``result`` (dict): the provider-result dict to return.
    Default behavior writes a file with a generic answer.

    ``write_file`` on a model whose prompt carries no path (an adapter declaring
    ``supports_file_output = False``, which is asked for stdout) returns the same
    content on stdout instead — the channel that adapter really has. ``ask_one``
    step 2 then persists it to the same ``answer_<model>.md``, which is why
    ``source: "fallback"`` and not the absence of an answer is the outcome.
    """

    def __init__(self):
        self.behaviors = {}
        self.calls = []
        self.prompts = {}

    def configure(self, model_spec, **behavior):
        self.behaviors[model_spec] = behavior

    def __call__(self, prompt, model_spec, timeout=None, log_file=None, cwd=None,
                 anchor=None):
        self.calls.append(model_spec)
        self.prompts[model_spec] = prompt
        b = self.behaviors.get(model_spec, {"write_file": True})
        content = b.get("content", f"Answer from {model_spec}.")
        if b.get("write_file", False):
            answer_path = extract_output_path(prompt)
            if answer_path:
                os.makedirs(os.path.dirname(answer_path), exist_ok=True)
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(content)
            elif "result" not in b:
                # Asked for stdout, so it answers on stdout.
                return {"success": True, "data": content, "details": {}}
        return b.get("result", {"success": True, "data": "stdout text", "details": {}})


def run_ask(argv, invoker):
    """Run ``ask_orchestrator.main(argv)`` with ``invoke_with_provider`` mocked."""
    with patch.object(ask_orchestrator, "invoke_with_provider", invoker):
        return asyncio.run(ask_orchestrator.main(argv))


# ---------------------------------------------------------------------------
# 1. Slugification
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_lowercase_and_collapse(self):
        slug = slugify_question("Hello,   World!!!")
        assert slug.startswith("hello-world-")

    def test_hash_suffix_is_sha1_of_full_question(self):
        q = "Is the rollback strategy sufficient?"
        slug = slugify_question(q)
        expected = hashlib.sha1(q.encode("utf-8")).hexdigest()[:8]
        assert slug.rsplit("-", 1)[1] == expected

    def test_truncation_to_50(self):
        q = "a" * 80
        slug = slugify_question(q)
        prefix, h = slug.rsplit("-", 1)
        assert len(prefix) == 50
        assert len(h) == 8

    def test_identical_question_same_slug(self):
        q = "Compare the two designs"
        assert slugify_question(q) == slugify_question(q)

    def test_different_questions_same_prefix_distinct_slugs(self):
        a = "a" * 60 + " one"
        b = "a" * 60 + " two"
        slug_a = slugify_question(a)
        slug_b = slugify_question(b)
        # Same normalized 50-char prefix...
        assert slug_a.rsplit("-", 1)[0] == slug_b.rsplit("-", 1)[0]
        # ...but distinct slugs (distinct hash suffixes) -> no collision.
        assert slug_a != slug_b

    def test_punctuation_only_question_is_still_unique(self):
        a = slugify_question("???")
        b = slugify_question("!!!")
        assert a != b
        assert a.startswith("question-")


# ---------------------------------------------------------------------------
# 2 & 3. Collision / resume guard
# ---------------------------------------------------------------------------

class TestResolveQuestionDir:
    def test_fresh_dir(self, tmp_path):
        d, is_resume = resolve_question_dir(str(tmp_path), "slug-abc", "Q")
        assert d == str(tmp_path / "slug-abc")
        assert is_resume is False

    def test_same_question_resumes(self, tmp_path):
        base = tmp_path / "slug-abc"
        base.mkdir()
        (base / ".status.json").write_text(json.dumps({"question": "Q"}), encoding="utf-8")
        d, is_resume = resolve_question_dir(str(tmp_path), "slug-abc", "Q")
        assert d == str(base)
        assert is_resume is True

    def test_different_question_gets_sibling(self, tmp_path):
        base = tmp_path / "slug-abc"
        base.mkdir()
        (base / ".status.json").write_text(json.dumps({"question": "Q-original"}), encoding="utf-8")
        d, is_resume = resolve_question_dir(str(tmp_path), "slug-abc", "Q-different")
        assert d == str(tmp_path / "slug-abc-2")
        assert is_resume is False

    def test_missing_status_treated_as_resume(self, tmp_path):
        base = tmp_path / "slug-abc"
        base.mkdir()
        d, is_resume = resolve_question_dir(str(tmp_path), "slug-abc", "Q")
        assert d == str(base)
        assert is_resume is True


class TestCollisionSemanticsFlow:
    def test_forced_hash_collision_creates_sibling_not_overwrite(self, plan, monkeypatch):
        # Force two DIFFERENT questions to slug identically.
        monkeypatch.setattr(ask_orchestrator, "slugify_question", lambda q: "fixed-deadbeef")
        inv = FakeInvoker()

        rc = run_ask(["--plan-file", str(plan), "--question", "Question A",
                      "--models", "claude-code:sonnet"], inv)
        assert rc == 0
        first_dir = ask_root(plan) / "fixed-deadbeef"
        assert (first_dir / "answers.md").exists()
        first_answers = read_answers(first_dir)
        assert "Question A" in first_answers

        rc = run_ask(["--plan-file", str(plan), "--question", "Question B",
                      "--models", "claude-code:sonnet"], inv)
        assert rc == 0
        sibling = ask_root(plan) / "fixed-deadbeef-2"
        assert (sibling / "answers.md").exists()
        # The first question's answers.md was NOT regenerated for question B.
        assert read_answers(first_dir) == first_answers
        assert "Question B" in read_answers(sibling)


# ---------------------------------------------------------------------------
# 4. Warn-and-proceed model validation
# ---------------------------------------------------------------------------

class TestWarnAndProceed:
    def test_invalid_model_warns_but_is_asked(self, plan, capsys):
        inv = FakeInvoker()
        rc = run_ask(["--plan-file", str(plan), "--question", "Summarize",
                      "--models", "faketest:modelx", "claude-code:sonnet"], inv)
        out = capsys.readouterr().out
        assert "Unknown models (proceeding anyway)" in out
        assert "faketest:modelx" in out
        # Still asked (not dropped, not aborted).
        assert "faketest:modelx" in inv.calls
        assert rc == 0


# ---------------------------------------------------------------------------
# 5-7. Answer capture precedence / hard-failure-with-file / fallback persistence
# ---------------------------------------------------------------------------

class TestAnswerCapture:
    def test_file_takes_precedence(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="FILE ANSWER",
                      result={"success": True, "data": "ignored stdout"})
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        d = qdir(plan, "Q")
        sanitized = sanitize_model_name("claude-code:sonnet")
        assert (d / f"answer_{sanitized}.md").read_text(encoding="utf-8") == "FILE ANSWER"
        assert "FILE ANSWER" in read_answers(d)

    def test_string_data_fallback(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=False,
                      result={"success": True, "data": "STRING DATA ANSWER"})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet"], inv)
        assert rc == 0
        d = qdir(plan, "Q")
        assert "STRING DATA ANSWER" in read_answers(d)

    def test_dict_data_is_capture_failure(self, plan):
        # data is an erroneously-parsed JSON fragment, and no raw -> failure.
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=False,
                      result={"success": True, "data": [{"frag": 1}]})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet"], inv)
        # D6 row 3: zero answers has nothing to aggregate, so it is exit 3
        # ("needs a human") rather than a bare exit 1.
        assert rc == 3
        d = qdir(plan, "Q")
        status = read_status(d)
        assert status["state"] == "failed"
        assert "claude-code:sonnet" in status["models_failed"]

    def test_raw_fallback_used_when_not_ndjson(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=False,
                      result={"success": False, "data": None,
                              "raw": "RAW MARKDOWN ANSWER\nsecond line"})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet"], inv)
        assert rc == 0
        d = qdir(plan, "Q")
        assert "RAW MARKDOWN ANSWER" in read_answers(d)

    def test_ndjson_raw_is_capture_failure(self, plan):
        ndjson = '{"type":"reasoning","text":"x"}\n{"type":"tool","text":"y"}'
        inv = FakeInvoker()
        inv.configure("codex:gpt-5.2-codex", write_file=False,
                      result={"success": False, "error": "No text events found",
                              "data": None, "raw": ndjson})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "codex:gpt-5.2-codex"], inv)
        # D6 row 3: zero answers has nothing to aggregate, so it is exit 3
        # ("needs a human") rather than a bare exit 1.
        assert rc == 3
        d = qdir(plan, "Q")
        # The NDJSON event dump is never pasted into answers.md.
        assert "reasoning" not in read_answers(d)
        assert read_status(d)["state"] == "failed"

    def test_hard_failure_with_answer_file_is_success(self, plan):
        # Agentic CLI wrote the file mid-run, then timed out: still success.
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="PARTIAL BUT COMPLETE",
                      result={"success": False, "error": "timed out",
                              "error_code": "TIMEOUT"})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet"], inv)
        assert rc == 0
        d = qdir(plan, "Q")
        assert "PARTIAL BUT COMPLETE" in read_answers(d)
        assert read_status(d)["state"] == "completed"

    def test_hard_failure_without_file_goes_to_failed_map(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=False,
                      result={"success": False, "error": "timed out",
                              "error_code": "TIMEOUT"})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet"], inv)
        # D6 row 3: zero answers has nothing to aggregate, so it is exit 3
        # ("needs a human") rather than a bare exit 1.
        assert rc == 3
        d = qdir(plan, "Q")
        sanitized = sanitize_model_name("claude-code:sonnet")
        assert (d / f"error_{sanitized}.log").exists()
        assert "timed out" in read_status(d)["models_failed"]["claude-code:sonnet"]

    def test_fallback_persisted_so_resume_skips(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=False,
                      result={"success": True, "data": "RECOVERED"})
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        d = qdir(plan, "Q")
        sanitized = sanitize_model_name("claude-code:sonnet")
        assert (d / f"answer_{sanitized}.md").read_text(encoding="utf-8") == "RECOVERED"

        # Second run: model should be SKIPPED (file persisted from fallback).
        inv2 = FakeInvoker()
        inv2.configure("claude-code:sonnet", write_file=True, content="SHOULD NOT RUN")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv2)
        assert inv2.calls == []  # skipped on resume
        assert (d / f"answer_{sanitized}.md").read_text(encoding="utf-8") == "RECOVERED"


# ---------------------------------------------------------------------------
# helper-level unit tests for recover/ndjson
# ---------------------------------------------------------------------------

class TestRecoverHelpers:
    def test_recover_prefers_string_data(self):
        assert recover_answer_text({"data": "hello"}) == "hello"

    def test_recover_rejects_dict_data(self):
        assert recover_answer_text({"data": {"a": 1}}) is None

    def test_recover_falls_back_to_raw(self):
        assert recover_answer_text({"data": [1], "raw": "real answer\nmore"}) == "real answer\nmore"

    def test_recover_rejects_ndjson_raw(self):
        ndjson = '{"type":"a"}\n{"type":"b"}'
        assert recover_answer_text({"data": None, "raw": ndjson}) is None

    def test_recover_empty(self):
        assert recover_answer_text({"data": "", "raw": "   "}) is None

    def test_ndjson_detection(self):
        assert looks_like_ndjson('{"type":"x"}\n{"type":"y"}') is True
        assert looks_like_ndjson("# Heading\nsome prose\nmore prose") is False
        assert looks_like_ndjson('{"type":"x"}') is False  # single line


# ---------------------------------------------------------------------------
# 8-11, 22-23. Aggregation / full regeneration / resume / force
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_answers_md_structure(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="Answer one")
        inv.configure("cursor-agent:auto", write_file=True, content="Answer two")
        run_ask(["--plan-file", str(plan), "--question", "Is rollback ok?",
                 "--models", "claude-code:sonnet", "cursor-agent:auto"], inv)
        d = qdir(plan, "Is rollback ok?")
        text = read_answers(d)
        assert "# Multi-LLM Ask" in text
        assert "Is rollback ok?" in text
        assert "# claude-code:sonnet" in text
        assert "# cursor-agent:auto" in text
        assert "Answer one" in text and "Answer two" in text
        # No failed section when nothing failed.
        assert "## Failed Models" not in text

    def test_failed_section_only_when_failures(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="Good")
        inv.configure("cursor-agent:auto", write_file=False,
                      result={"success": False, "error": "boom", "error_code": "SUBPROCESS_FAILED"})
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet", "cursor-agent:auto"], inv)
        d = qdir(plan, "Q")
        text = read_answers(d)
        assert "## Failed Models" in text
        assert "cursor-agent:auto" in text
        assert "Good" in text

    def test_deterministic_ordering(self, plan):
        d = qdir(plan, "Q")
        d.mkdir(parents=True)
        for spec, content in [("claude-code:sonnet", "A"), ("cursor-agent:auto", "B"),
                              ("gemini:gemini-3-pro", "C")]:
            (d / f"answer_{sanitize_model_name(spec)}.md").write_text(content, encoding="utf-8")
        specs = ["gemini:gemini-3-pro", "claude-code:sonnet", "cursor-agent:auto"]
        path1, completed1 = write_answers_md(str(d), "Q", str(plan), specs, {}, "TS")
        text1 = Path(path1).read_text(encoding="utf-8")
        path2, completed2 = write_answers_md(str(d), "Q", str(plan), specs, {}, "TS")
        text2 = Path(path2).read_text(encoding="utf-8")
        assert text1 == text2
        assert completed1 == completed2 == specs  # in models_requested order
        # Section order follows models_requested, not filesystem/dict order.
        assert text1.index("# gemini:gemini-3-pro") < text1.index("# claude-code:sonnet") < text1.index("# cursor-agent:auto")

    def test_full_regeneration_includes_skipped_models(self, plan):
        # First run: both succeed.
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="first-sonnet")
        inv.configure("cursor-agent:auto", write_file=True, content="first-cursor")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet", "cursor-agent:auto"], inv)
        d = qdir(plan, "Q")

        # Delete only one model's answer to simulate a missing model, then resume.
        cursor_file = d / f"answer_{sanitize_model_name('cursor-agent:auto')}.md"
        cursor_file.unlink()
        inv2 = FakeInvoker()
        inv2.configure("claude-code:sonnet", write_file=True, content="SHOULD-NOT-RERUN")
        inv2.configure("cursor-agent:auto", write_file=True, content="second-cursor")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet", "cursor-agent:auto"], inv2)
        # Only the missing model was re-asked.
        assert inv2.calls == ["cursor-agent:auto"]
        text = read_answers(d)
        # Previously-completed (skipped) model still appears with its prior answer.
        assert "first-sonnet" in text
        assert "second-cursor" in text
        status = read_status(d)
        assert set(status["models_completed"]) == {"claude-code:sonnet", "cursor-agent:auto"}

    def test_rerun_all_reasks_all_models(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="round-1")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        d = qdir(plan, "Q")
        assert "round-1" in read_answers(d)

        inv2 = FakeInvoker()
        inv2.configure("claude-code:sonnet", write_file=True, content="round-2")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet", "--rerun-all"], inv2)
        # Re-asked despite existing answer file (--rerun-all discards results).
        assert inv2.calls == ["claude-code:sonnet"]
        assert "round-2" in read_answers(d)
        assert "round-1" not in read_answers(d)

    def test_force_resumes_keeps_existing_answers(self, plan):
        # --force now resumes (bypasses guards) rather than re-asking every
        # model: a model that already has a non-empty answer is skipped.
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="round-1")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        d = qdir(plan, "Q")
        assert "round-1" in read_answers(d)

        inv2 = FakeInvoker()
        inv2.configure("claude-code:sonnet", write_file=True, content="round-2")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet", "--force"], inv2)
        # Skipped on resume: not re-invoked, original answer preserved.
        assert inv2.calls == []
        assert "round-1" in read_answers(d)
        assert "round-2" not in read_answers(d)


# ---------------------------------------------------------------------------
# 10, 13. Resume / corrupt-empty resume
# ---------------------------------------------------------------------------

class TestResumeBehavior:
    def test_existing_nonempty_answer_skipped(self, plan):
        d = qdir(plan, "Q")
        d.mkdir(parents=True)
        sanitized = sanitize_model_name("claude-code:sonnet")
        (d / f"answer_{sanitized}.md").write_text("prior answer", encoding="utf-8")
        # Seed a status so resolve_question_dir resumes into the dir.
        (d / ".status.json").write_text(json.dumps({"question": "Q"}), encoding="utf-8")
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="SHOULD NOT RUN")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        assert inv.calls == []
        assert "prior answer" in read_answers(d)

    def test_empty_answer_file_triggers_rerun(self, plan, capsys):
        d = qdir(plan, "Q")
        d.mkdir(parents=True)
        sanitized = sanitize_model_name("claude-code:sonnet")
        (d / f"answer_{sanitized}.md").write_text("   \n  ", encoding="utf-8")  # whitespace-only
        (d / ".status.json").write_text(json.dumps({"question": "Q"}), encoding="utf-8")
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="fresh answer")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        assert inv.calls == ["claude-code:sonnet"]
        out = capsys.readouterr().out
        assert "empty/unreadable" in out
        assert "fresh answer" in read_answers(d)


# ---------------------------------------------------------------------------
# 12. Plan-freshness check
# ---------------------------------------------------------------------------

class TestPlanFreshness:
    def test_changed_plan_warns_on_resume(self, plan, capsys):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="ans")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        # Change the plan, then resume.
        plan.write_text("# My Feature\n\nCompletely different content now.\n", encoding="utf-8")
        inv2 = FakeInvoker()
        inv2.configure("claude-code:sonnet", write_file=True, content="ans")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv2)
        out = capsys.readouterr().out
        assert "plan file has changed" in out

    def test_unchanged_plan_no_warning(self, plan, capsys):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="ans")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        capsys.readouterr()
        inv2 = FakeInvoker()
        inv2.configure("claude-code:sonnet", write_file=True, content="ans")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv2)
        out = capsys.readouterr().out
        assert "plan file has changed" not in out


# ---------------------------------------------------------------------------
# 14, 15, 24. Status schema / exit codes / partial-failure status
# ---------------------------------------------------------------------------

class TestStatusAndExit:
    def test_status_schema_completed(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="a")
        run_ask(["--plan-file", str(plan), "--question", "Q?",
                 "--models", "claude-code:sonnet"], inv)
        d = qdir(plan, "Q?")
        status = read_status(d)
        for key in ("phase", "state", "started_at", "models_requested",
                    "models_completed", "models_failed", "question",
                    "question_slug", "plan_hash", "answers_md"):
            assert key in status, f"missing {key}"
        assert status["phase"] == "ask"
        assert status["state"] == "completed"
        assert status["question"] == "Q?"
        assert status["question_slug"] == slugify_question("Q?")
        assert status["models_completed"] == ["claude-code:sonnet"]
        assert status["models_failed"] == {}

    def test_partial_failure_status(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="ok")
        inv.configure("cursor-agent:auto", write_file=False,
                      result={"success": False, "error": "exited 1", "error_code": "SUBPROCESS_FAILED"})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet", "cursor-agent:auto"], inv)
        assert rc == 0  # partial success exits 0
        d = qdir(plan, "Q")
        status = read_status(d)
        assert status["state"] == "partial"
        assert status["models_completed"] == ["claude-code:sonnet"]
        assert set(status["models_failed"]) == {"cursor-agent:auto"}
        assert "exited 1" in status["models_failed"]["cursor-agent:auto"]

    def test_all_fail_exit_3_with_files(self, plan):
        inv = FakeInvoker()
        for spec in ("claude-code:sonnet", "cursor-agent:auto"):
            inv.configure(spec, write_file=False,
                          result={"success": False, "error": "nope", "error_code": "SUBPROCESS_FAILED"})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet", "cursor-agent:auto"], inv)
        # D6 row 3: zero answers has nothing to aggregate, so it is exit 3
        # ("needs a human") rather than a bare exit 1.
        assert rc == 3
        d = qdir(plan, "Q")
        assert (d / "answers.md").exists()
        assert (d / ".status.json").exists()
        text = read_answers(d)
        assert "## Failed Models" in text
        # No answer sections (no model header lines).
        assert "# claude-code:sonnet" not in text
        status = read_status(d)
        assert status["state"] == "failed"
        assert status["models_completed"] == []


# ---------------------------------------------------------------------------
# 16. Argparse / CLI dispatch
# ---------------------------------------------------------------------------

class TestArgparse:
    def test_plan_file_required(self):
        with pytest.raises(SystemExit):
            parse_args(["--question", "Q"])

    def test_question_group_required(self):
        with pytest.raises(SystemExit):
            parse_args(["--plan-file", "p.md"])

    def test_question_group_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            parse_args(["--plan-file", "p.md", "--question", "Q", "--question-env", "X"])

    def test_defaults(self):
        args = parse_args(["--plan-file", "p.md", "--question", "Q"])
        assert args.max_parallel == 5
        assert args.force is False
        assert args.quick is False
        assert args.interactive is False

    def test_three_sources_resolve_same_verbatim(self, tmp_path, monkeypatch):
        q = 'should we use `cmd` or $(other) with "quotes"?'
        # --question
        a = resolve_question(parse_args(["--plan-file", "p.md", "--question", q]))
        # --question-file (exact bytes, no trailing newline)
        qf = tmp_path / "q.txt"
        qf.write_text(q, encoding="utf-8")
        b = resolve_question(parse_args(["--plan-file", "p.md", "--question-file", str(qf)]))
        # --question-env
        monkeypatch.setenv("ASK_Q", q)
        c = resolve_question(parse_args(["--plan-file", "p.md", "--question-env", "ASK_Q"]))
        assert a == b == c == q

    def test_leading_hyphen_question_preserved_via_file_and_env(self, tmp_path, monkeypatch):
        q = "--this looks like a flag but is a question"
        qf = tmp_path / "q.txt"
        qf.write_text(q, encoding="utf-8")
        b = resolve_question(parse_args(["--plan-file", "p.md", "--question-file", str(qf)]))
        monkeypatch.setenv("ASK_Q", q)
        c = resolve_question(parse_args(["--plan-file", "p.md", "--question-env", "ASK_Q"]))
        assert b == c == q

    def test_quick_interactive_mutually_exclusive(self, plan):
        inv = FakeInvoker()
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet", "--quick", "--interactive"], inv)
        assert rc == 1  # rejected, not silently picking one

    def test_empty_question_is_error(self, plan):
        inv = FakeInvoker()
        rc = run_ask(["--plan-file", str(plan), "--question", "   "], inv)
        assert rc == 1

    def test_metacharacter_question_preserved_into_slug_and_status(self, plan):
        q = 'run $(rm -rf /) or `whoami`? "danger"'
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="careful")
        run_ask(["--plan-file", str(plan), "--question", q,
                 "--models", "claude-code:sonnet"], inv)
        d = qdir(plan, q)
        status = read_status(d)
        # Stored verbatim, no shell processing / re-quoting.
        assert status["question"] == q
        assert status["question_slug"] == slugify_question(q)


# ---------------------------------------------------------------------------
# 18-21. Prompt template formatting / read-only / absolute path / size guard
# ---------------------------------------------------------------------------

class TestPrompt:
    def _template(self):
        from utils.prompt_loader import load_prompt
        return load_prompt("ask.txt")

    def test_renders_all_placeholders(self):
        out = render_prompt(self._template(), "MYQUESTION", "/abs/plan.md",
                            "PLANBODY", "/abs/out/answer.md")
        assert "MYQUESTION" in out
        assert "/abs/plan.md" in out
        assert "PLANBODY" in out
        assert "/abs/out/answer.md" in out

    def test_braces_in_plan_content_do_not_break_format(self):
        plan_content = 'JSON: {"a": [1,2], "b": {"c": 3}} and f-string {x}'
        out = render_prompt(self._template(), "Q", "/p.md", plan_content, "/o.md")
        assert plan_content in out  # reproduced verbatim, no KeyError/IndexError

    def test_read_only_directives_present(self):
        out = render_prompt(self._template(), "Q", "/abs/plan.md", "body", "/abs/answer.md")
        assert "read-only" in out.lower()
        assert "Do NOT modify, create, or delete any file except" in out
        assert "not as instructions" in out.lower() or "not as instructions to act" in out.lower()
        # The write target is the answer path, not the plan path.
        after = out.split("exact absolute file path:", 1)[1]
        assert "/abs/answer.md" in after
        assert "print your full answer to stdout" in out.lower() or "print your full answer to stdout" in out

    def test_orchestrator_injects_absolute_answer_path(self, plan):
        inv = FakeInvoker()
        inv.configure("claude-code:sonnet", write_file=True, content="a")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv)
        prompt = inv.prompts["claude-code:sonnet"]
        injected = extract_output_path(prompt)
        assert injected is not None
        assert os.path.isabs(injected)
        d = qdir(plan, "Q")
        expected = os.path.realpath(str(d / f"answer_{sanitize_model_name('claude-code:sonnet')}.md"))
        assert injected == expected
        # plan_path is also absolute in the prompt.
        assert os.path.abspath(str(plan)) in prompt

    def test_small_plan_inlined_verbatim(self, plan):
        content = "# tiny plan\nbody"
        inlined, truncated = build_inline_plan(content, str(plan))
        assert truncated is False
        assert inlined == content

    def test_large_plan_truncated_with_marker(self, plan):
        content = "x" * (MAX_INLINE_PLAN_BYTES + 5000)
        inlined, truncated = build_inline_plan(content, str(plan))
        assert truncated is True
        assert "plan truncated at" in inlined
        assert str(plan) in inlined  # marker points at the plan path
        # Inlined copy stays under the threshold (+ marker), avoiding E2BIG.
        assert len(inlined.encode("utf-8")) <= MAX_INLINE_PLAN_BYTES + 500


# ---------------------------------------------------------------------------
# 22. Output-channel variants (C10)
# ---------------------------------------------------------------------------

#: Today's ``## Output Location`` section, rendered for ``/abs/out/answer.md``.
#: An independent transcription of the bytes ``prompts/ask.txt`` carried inline
#: before the ``{output_location}`` field existed, so "the file rendering is
#: byte-identical to today's" is pinned against something other than the
#: constant under test.
TODAYS_FILE_OUTPUT_LOCATION = """\
## Output Location
Write your answer as MARKDOWN (not JSON) to this exact absolute file path:

/abs/out/answer.md

- Write to the path exactly as given above. Do NOT reinterpret, relativize, or
  change it. It is an absolute path; write the file there.
- Write ONLY your markdown answer to that file (no JSON wrapper, no code-fence
  around the whole answer).
- ALSO print your full answer to stdout. Stdout is the fallback that is used if
  the file write fails, so the two should contain the same answer.
"""

#: Today's read-only write bullet, same transcription rule.
TODAYS_FILE_READONLY_BULLET = """\
- Do NOT modify, create, or delete any file except the single answer output
  file specified below (/abs/out/answer.md). In particular, NEVER edit the plan
  file or any other file in the repository.
"""


class TestOutputChannelVariants:
    """C10: the ask prompt asks each adapter for the channel it declares.

    ``cursor-agent`` declares ``supports_file_output = False`` (it runs
    ``--mode ask`` and can only answer a "write this file" instruction with a
    refusal sentence); ``claude-code`` inherits the default ``True``.
    """

    def _template(self):
        from utils.prompt_loader import load_prompt
        return load_prompt("ask.txt")

    def _render(self, provider_name):
        return render_prompt(self._template(), "MYQUESTION", "/abs/plan.md",
                             "PLANBODY", "/abs/out/answer.md", provider_name)

    def test_file_rendering_is_byte_identical_to_todays(self):
        out = self._render("claude-code")
        assert TODAYS_FILE_READONLY_BULLET in out
        assert TODAYS_FILE_OUTPUT_LOCATION in out
        # Splitting the template into two fields left no placeholder unfilled.
        assert "{" not in out and "}" not in out

    def test_stdout_rendering_never_names_a_file_to_write(self):
        out = self._render("cursor-agent")
        # (a) neither the answer path ...
        assert "/abs/out/answer.md" not in out
        assert "answer.md" not in out
        # ... nor any instruction to write a file.
        assert "exact absolute file path" not in out
        assert "Write your answer" not in out
        assert "Print your full answer to stdout as your final message" in out
        # The read-only bullet no longer carves out an exception for a file the
        # section below has just said cannot be written.
        assert "Do NOT modify, create, or delete any file." in out
        assert "except the single answer output" not in out
        # Every remaining mention of writing is the prohibition itself.
        for line in out.splitlines():
            low = line.lower()
            if "write" in low:
                assert "do not write" in low or "write will be refused" in low, line

    def test_stdout_rendering_is_never_longer_than_the_file_one(self):
        file_out = self._render("claude-code")
        stdout_out = self._render("cursor-agent")
        # (c) The truncation-budget promise as an assertion: the rendered
        # template is render_prompt_guarded's fixed overhead, so the stdout
        # variant must not cost the question a single byte.
        assert len(stdout_out) <= len(file_out)
        assert len(stdout_out.encode("utf-8")) <= len(file_out.encode("utf-8"))

    def test_unknown_or_absent_provider_renders_the_file_variant(self):
        # The posture llm_client.render_output_location already takes: a config
        # naming an adapter this build no longer ships gets what every caller
        # got before the field existed.
        assert self._render(None) == self._render("claude-code")
        assert self._render("no-such-provider") == self._render("claude-code")

    def test_guarded_oversize_truncation_holds_for_both_variants(self):
        template = self._template()
        # "Z" appears nowhere in the template, so counting it measures exactly
        # how many question bytes each variant's budget kept.
        assert "Z" not in template
        question = "Z" * (MAX_PROMPT_BYTES + 10000)
        kept = {}
        for provider_name in ("claude-code", "cursor-agent"):
            rendered, truncated = render_prompt_guarded(
                template, question, "/abs/plan.md", "PLANBODY",
                "/abs/out/answer.md", provider_name,
            )
            assert truncated is True
            assert len(rendered.encode("utf-8")) <= MAX_PROMPT_BYTES
            assert "question truncated to fit the argv size limit" in rendered
            assert "/abs/plan.md" in rendered  # never truncated
            kept[provider_name] = rendered.count("Z")
        # The file variant still names the path it must write, intact.
        file_rendered, _ = render_prompt_guarded(
            template, question, "/abs/plan.md", "PLANBODY",
            "/abs/out/answer.md", "claude-code",
        )
        assert "/abs/out/answer.md" in file_rendered
        # The stdout variant's smaller fixed overhead can only grow the budget.
        assert kept["cursor-agent"] >= kept["claude-code"]

    def test_read_only_adapter_still_lands_a_file_via_fallback(self, plan):
        """Nothing downstream moves. A ``supports_file_output = False`` adapter
        is asked for stdout, ``ask_one`` step 2 (``recover_answer_text``)
        persists that answer to ``answer_<model>.md``, so ``source: "fallback"``
        stays the outcome and aggregation still reads a file."""
        inv = FakeInvoker()
        inv.configure("cursor-agent:auto", write_file=True, content="stdout-only answer")
        captured = {}
        real_run = ask_orchestrator.run_models_concurrent

        async def spy(*args, **kwargs):
            results = await real_run(*args, **kwargs)
            captured.update(results)
            return results

        with patch.object(ask_orchestrator, "run_models_concurrent", spy):
            run_ask(["--plan-file", str(plan), "--question", "Q",
                     "--models", "cursor-agent:auto"], inv)

        # It was never handed a path, so the file can only have come from step 2.
        assert extract_output_path(inv.prompts["cursor-agent:auto"]) is None
        assert captured["cursor-agent:auto"]["source"] == "fallback"
        d = qdir(plan, "Q")
        answer = d / f"answer_{sanitize_model_name('cursor-agent:auto')}.md"
        assert answer.read_text(encoding="utf-8") == "stdout-only answer"
        assert "stdout-only answer" in read_answers(d)


# ---------------------------------------------------------------------------
# heading demotion
# ---------------------------------------------------------------------------

class TestDemoteHeadings:
    def test_demotes_atx_headings(self):
        out = demote_headings("# Title\ntext\n## Sub\n")
        assert "## Title" in out
        assert "### Sub" in out

    def test_caps_at_six(self):
        assert "######" in demote_headings("###### Deep")
        # Not seven hashes.
        assert "#######" not in demote_headings("###### Deep")

    def test_leaves_fenced_code_untouched(self):
        md = "```\n# not a heading\n```\n# real heading"
        out = demote_headings(md)
        assert "# not a heading" in out  # unchanged inside fence
        assert "## real heading" in out  # demoted outside fence


# ---------------------------------------------------------------------------
# 25. Shared concurrency helper
# ---------------------------------------------------------------------------

class TestRunModelsConcurrent:
    def test_returns_callback_results_unchanged(self):
        async def cb(spec, idx):
            return f"result::{spec}::{idx}"
        specs = ["gemini:gemini-3-flash", "gemini:gemini-3-pro"]
        res = asyncio.run(run_models_concurrent(specs, cb, None, 5))
        assert res == {
            "gemini:gemini-3-flash": "result::gemini:gemini-3-flash::0",
            "gemini:gemini-3-pro": "result::gemini:gemini-3-pro::1",
        }

    def test_respects_global_max_parallel(self, monkeypatch):
        monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
        state = {"current": 0, "max": 0}

        async def cb(spec, idx):
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
            await asyncio.sleep(0.05)
            state["current"] -= 1
            return spec

        # gemini has no max_concurrent -> only the global cap applies.
        specs = [f"gemini:gemini-3-flash-{i}" for i in range(4)]
        asyncio.run(run_models_concurrent(specs, cb, None, 2))
        assert state["max"] <= 2

    def test_respects_per_provider_limit(self, monkeypatch):
        monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
        state = {"current": 0, "max": 0}

        async def cb(spec, idx):
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
            await asyncio.sleep(0.05)
            state["current"] -= 1
            return spec

        # claude-code has max_concurrent=2 in providers.yaml; global cap is high.
        specs = [f"claude-code:m{i}" for i in range(4)]
        asyncio.run(run_models_concurrent(specs, cb, None, 10))
        assert state["max"] <= 2

    def test_applies_staggered_starts(self, monkeypatch):
        import utils.review_orchestrator_base as base
        recorded = []

        async def fake_sleep(delay):
            recorded.append(delay)

        monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(base, "get_backoff_delay", lambda x: x)

        async def cb(spec, idx):
            return spec

        specs = ["gemini:gemini-3-flash", "gemini:gemini-3-pro", "gemini:gemini-2.5-flash"]
        asyncio.run(run_models_concurrent(specs, cb, None, 5))
        # index 0 has no stagger; indices 1,2 -> index * PROVIDER_STAGGER_DELAY (2.0).
        assert recorded == [base.PROVIDER_STAGGER_DELAY, 2 * base.PROVIDER_STAGGER_DELAY]


# ---------------------------------------------------------------------------
# 17, 26. SKILL.md / instructions routing & shell-safe parsing rule coverage
# ---------------------------------------------------------------------------

class TestRoutingAndDocs:
    def test_instruction_file_exists_with_usage_and_invocation(self):
        ask_md = (SKILL_DIR / "instructions" / "ask.md").read_text(encoding="utf-8")
        assert "/multi-llm:multi-llm --ask" in ask_md
        # Host-neutral core: the skill dir is the `{SKILL_DIR}` template token
        # (substituted by `util instructions ask` before an agent reads it), the
        # command names the single entry point, and the expansion is still
        # double-quoted (Windows install paths may contain spaces).
        assert '--project "{SKILL_DIR}"' in ask_md
        assert '"{SKILL_DIR}/multi_llm.py" ask' in ask_md
        assert "ask_orchestrator.py" not in ask_md, (
            "the core names multi_llm.py, never a second entry point"
        )
        # Shell-safe question passing: the dispatcher takes the question as a
        # positional and owns the --question-file fallback itself, so the core no
        # longer instructs an agent to write that file. The plan path is passed as
        # given and the orchestrator resolves it natively — no $(realpath ...),
        # whose POSIX /c/... output breaks native Windows processes.
        assert "--question-file" in ask_md
        assert "--question-env" in ask_md
        assert '"$PLAN_PATH" "$RAW_QUESTION"' in ask_md
        assert '$(realpath "$PLAN_PATH")' not in ask_md
        # Single-quoted-question rule + error cases.
        assert "single argument" in ask_md
        assert "empty/whitespace-only question is an error" in ask_md or "empty question" in ask_md.lower()

    def test_skill_md_routes_ask_mode(self):
        """The shim's ask obligations, after the D5c slimming.

        The mode *list* and the mode-detection pseudocode left the shim: `modes.py`
        owns the vocabulary, `argument-hint` is rendered from it, and the mode word
        is mechanically the flag without its `--`. What a shim must still carry is
        the ask-specific argument rule (the one mode with two positionals) and a
        worked example whose quoted path is what naive re-typing breaks.
        """
        skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "--ask" in skill
        assert "argument-hint:" in skill
        # The mode word is derivable without a second list, and the shim says so.
        assert "the mode word without its" in skill
        # The ask-specific positional rule, and the quoted-path worked example.
        assert "the next single quoted" in skill
        assert '--ask "plans/my plan.md" "Is the rollback strategy sufficient?"' in skill

    def test_skill_and_instructions_agree_on_error_rule(self):
        """Both layers refuse to repair an unquoted or empty question."""
        skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        ask_md = (SKILL_DIR / "instructions" / "ask.md").read_text(encoding="utf-8")
        for doc in (skill, ask_md):
            assert "unquoted" in doc.lower()
        assert "usage hint" in ask_md.lower()
        assert "usage hint" in skill.lower()

    def test_documented_bash_timeouts_within_cap_and_fanout_runs_detached(self):
        """The Claude Code Bash tool hard-caps `timeout` at 600000 ms (10 min);
        any larger value is silently clamped. So no instruction may document a
        Bash timeout above the cap, and the long fan-out review/ask modes (whose
        runtime routinely exceeds 10 min) MUST run detached with stdout
        redirected to a log file rather than relying on an oversized foreground
        timeout.

        This replaces the old contract, which required the ask-mode Bash timeout
        to exceed the slowest provider budget (>=1.2M ms). That was the bug: the
        cap made it unachievable, so a foreground run was SIGTERM'd mid-run and
        lost the answers.md path printed on stdout. The fix is to run the fan-out
        modes detached (not subject to the cap) with output captured to a log."""
        BASH_TIMEOUT_CAP_MS = 600_000

        def documented_timeouts_ms(text):
            # Match both `timeout: 600000` (Bash-tool form) and prose like
            # "600000 ms" — i.e. any 6+ digit count presented as a Bash timeout
            # value or explicitly in milliseconds.
            matches = re.findall(r"timeout`?\s*[:=]\s*[`~]?\s*([\d,_]{6,})", text)
            matches += re.findall(r"[`~]?([\d,_]{6,})\s*ms\b", text)
            return [int(m.replace(",", "").replace("_", "")) for m in matches]

        # No instruction (SKILL.md or any mode file) may document a Bash timeout
        # above the hard cap — the regression guard for the original bug.
        docs = [SKILL_DIR / "SKILL.md"]
        docs += sorted((SKILL_DIR / "instructions").glob("*.md"))
        for doc in docs:
            over_cap = [
                t for t in documented_timeouts_ms(doc.read_text(encoding="utf-8"))
                if t > BASH_TIMEOUT_CAP_MS
            ]
            assert not over_cap, (
                f"{doc.name} documents Bash timeout(s) above the "
                f"{BASH_TIMEOUT_CAP_MS} ms cap: {over_cap}"
            )

        # The fan-out review/ask modes must run detached with stdout captured to
        # a log file, so markers and the final answers.md path survive past the
        # cap (backgrounding without redirection would lose them).
        skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert "run_in_background" in skill
        assert "PYTHONUNBUFFERED" in skill
        assert "orchestrator-run.log" in skill

        # The shared core states the same requirement in the host-neutral
        # vocabulary: it names the *run detached* capability verb, never a host's
        # background tool. The shim above resolves that verb to
        # `run_in_background`; the core must not.
        ask_md = (SKILL_DIR / "instructions" / "ask.md").read_text(encoding="utf-8")
        assert "run detached" in ask_md.lower()
        assert "run_in_background" not in ask_md, (
            "instructions/ask.md is host-neutral core: the mechanism belongs in "
            "the shim's verb resolution, not here"
        )
        assert "--detach" in ask_md
        assert "PYTHONUNBUFFERED" in ask_md
        assert ".log" in ask_md


# ---------------------------------------------------------------------------
# D6 step 3 — ask's lane asks the same pre-dispatch sandbox question
# ---------------------------------------------------------------------------

class TestTheAskFanoutGatesOnTheSandboxProbe:
    """`ask_one` is the third lane that reaches `generate`, so it asks too.

    An ask slot answers in prose, which is what makes the ungated failure
    invisible here: a runner whose sandbox cannot start writes a paragraph
    about the plan file it could not open, and a non-empty answer file *is*
    success. The refusal comes off the same once-per-run cache and carries the
    same `[SANDBOX_UNAVAILABLE]` wording as the review lanes — under
    `sandbox_fallback: off`; the shipped `auto` dispatches the slot
    unsandboxed instead (D6 override, 2026-09-17), below.
    """

    #: Real bubblewrap on a stock Ubuntu 24.04, replayed verbatim.
    BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"

    def _broken(self, monkeypatch, policy=None):
        """The real codex adapter with only its probe argv replaced: `sh`
        really runs, and what it prints is the line bwrap prints where it
        cannot start. ``policy`` pins the resolved key through the accessor,
        not the config file: `run_ask` reads the shipped base config, and what
        these tests pin is the gate, not the YAML layering
        (`test_subagent_executor` reads the key through the real file). `None`
        leaves the shipped default in place."""
        from utils import provider_registry as registry
        from utils import review_orchestrator_base as rob
        from utils.providers import codex as codex_module
        from utils.providers.base import PROFILE_GENERATE

        monkeypatch.setattr(rob, "_FANOUT_SANDBOX_CACHE", {})
        broken = ["sh", "-c", f"echo '{self.BWRAP_LINE}' >&2; exit 1"]
        provider = registry.get_provider("codex")
        monkeypatch.setattr(
            type(provider), "sandbox_probe_command",
            lambda _self, profile: broken if profile == PROFILE_GENERATE else None,
        )
        if policy is not None:
            monkeypatch.setattr(
                codex_module, "sandbox_fallback_policy", lambda anchor=None: policy
            )

    def test_the_ask_fanout_refuses_a_slot_whose_sandbox_cannot_start_under_off(
        self, plan, monkeypatch
    ):
        self._broken(monkeypatch, policy="off")
        inv = FakeInvoker()
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "codex:gpt-5.2-codex"], inv)

        # D6 row 3: zero answers has nothing to aggregate, so it is exit 3.
        assert rc == 3
        assert inv.calls == []  # nothing was dispatched
        d = qdir(plan, "Q")
        reason = read_status(d)["models_failed"]["codex:gpt-5.2-codex"]
        assert reason.startswith("[SANDBOX_UNAVAILABLE]")
        assert "codex's sandbox cannot start here" in reason
        assert self.BWRAP_LINE in reason
        assert "multi_llm.py doctor" in reason
        # The failed-model section of answers.md carries the same sentence.
        assert reason in read_answers(d)

    # -- the D6 override (2026-09-17): the third answer of the gate --------

    BYPASS_LINE = (
        "[SANDBOX_BYPASSED] codex: bubblewrap cannot start; running --sandbox "
        "danger-full-access under providers.codex.sandbox_fallback=auto "
        "(read-only phase; no OS boundary between the model's shell and this "
        "machine)"
    )

    def test_the_ask_fanout_dispatches_unsandboxed_by_default_with_the_flag_and_the_marker(
        self, plan, monkeypatch, capsys
    ):
        """No policy pinned: the shipped base config resolves `auto`, and that
        is what a fresh install does on a machine whose bwrap cannot start."""
        from utils.providers import codex as codex_module

        self._broken(monkeypatch)
        assert codex_module.sandbox_fallback_policy() == "auto"

        class Invoker(FakeInvoker):
            seen = []

            def __call__(self, prompt, model_spec, timeout=None, log_file=None,
                         cwd=None, anchor=None, **kwargs):
                self.seen.append(kwargs)
                return super().__call__(prompt, model_spec, timeout, log_file, cwd, anchor)

        inv = Invoker()
        inv.configure("codex:gpt-5.2-codex", write_file=True, content="An answer.")
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "codex:gpt-5.2-codex"], inv)
        assert rc == 0
        assert inv.calls == ["codex:gpt-5.2-codex"]
        assert inv.seen == [{"sandbox_fallback": True}]
        out = capsys.readouterr().out
        assert out.count("[SANDBOX_BYPASSED]") == 1
        assert f"[codex:gpt-5.2-codex] {self.BYPASS_LINE}" in out
        d = qdir(plan, "Q")
        assert read_status(d)["models_completed"] == ["codex:gpt-5.2-codex"]

    def test_a_bypassed_ask_slot_that_writes_is_failed_by_path_and_its_answer_set_aside(
        self, plan, monkeypatch, capsys
    ):
        """The answer-file policy ("a non-empty answer file is success") was
        written for a slot that could not write anything else, so the check
        runs ahead of it and the answer is set aside, not counted."""
        import os
        import subprocess

        root = os.path.dirname(str(plan))
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
        subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=env)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True, env=env)
        self._broken(monkeypatch)
        d = qdir(plan, "Q")
        answer_file = os.path.join(d, "answer_codex_gpt-5_2-codex.md")

        class Invoker(FakeInvoker):
            def __call__(self, prompt, model_spec, timeout=None, log_file=None,
                         cwd=None, anchor=None, **kwargs):
                # A model with no sandbox can write wherever it likes: outside
                # the repo's contract, and — though asked for stdout — its own
                # answer file too. Both are what the check has to handle.
                with open(os.path.join(root, "leak.md"), "w", encoding="utf-8") as f:
                    f.write("notes\n")
                os.makedirs(d, exist_ok=True)
                with open(answer_file, "w", encoding="utf-8") as f:
                    f.write("An answer written straight to disk.\n")
                return super().__call__(prompt, model_spec, timeout, log_file, cwd, anchor)

        inv = Invoker()
        inv.configure("codex:gpt-5.2-codex", write_file=True, content="An answer.")
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "codex:gpt-5.2-codex"], inv)
        assert rc == 3  # zero answers counted → nothing to aggregate
        reason = read_status(d)["models_failed"]["codex:gpt-5.2-codex"]
        assert reason == (
            "[SANDBOX_BYPASSED] codex wrote outside its read-only contract: leak.md"
        )
        assert reason in read_answers(d)
        assert read_status(d)["models_completed"] == []
        # The answer it wrote is set aside, never counted and never deleted.
        assert not os.path.exists(answer_file)
        assert os.path.exists(answer_file + ".violated")
        assert os.path.exists(os.path.join(root, "leak.md"))  # never restored


# ---------------------------------------------------------------------------
# 27. T4/T5 — per-model fan-out progress on the ask lane
# ---------------------------------------------------------------------------
# ask was the coarsest lane in the tree: no `fanout_slot_counter` at all, two
# `report_progress` calls for a whole run (`0` at the fan-out and the final
# count at grouping), so a poller mid-run read `models 0/4` beside models that
# had already answered. These tests pin what replaced it — a per-model map
# written into the *question* dir, both counters, and two tickers.


@pytest.fixture
def progress_lines(monkeypatch):
    """Every ``[PROGRESS]`` line, with the ticker's wait made instant.

    ``_tick_sleep`` is the only wait either ticker has, so replacing it is what
    keeps a test of a 30 s cadence from taking 30 s. The 5 ms is deliberate
    rather than ``sleep(0)``: the shape under test is a task that wakes, prints
    and sleeps again, and a zero sleep would spin the loop for every turn the
    mocked fan-out spends in ``asyncio.to_thread`` and bound the line count by
    the machine rather than by the script. Both tickers' teardown ``_tick``
    fires regardless, so at least one line per ticker is guaranteed.
    """
    from utils import progress_tracker

    lines = []
    monkeypatch.setattr(progress_tracker, "_emit", lines.append)

    async def instant(_delay):
        await asyncio.sleep(0.005)

    monkeypatch.setattr(progress_tracker, "_tick_sleep", instant)
    return lines


def progress_of(question_dir):
    """The checkpoint's reserved ``progress`` object, as a poller reads it."""
    return read_status(question_dir).get("progress", {})


def models_of(question_dir):
    return progress_of(question_dir).get("models", {})


def spy_on_the_fanout(monkeypatch):
    """Record the tracker ``ask`` hands the shared helper, and forbid the wrapper.

    ask reaches ``run_models_concurrent`` **directly** with its own per-model
    coroutine; it never goes through ``run_all_models``, which serves review-plan
    and review-tasks only. Anything hooked into the wrapper would therefore
    reach this lane not at all, so the wrapper is replaced with a raiser: these
    tests fail loudly rather than passing on a code path ask does not run.
    """
    import utils.review_orchestrator_base as base

    seen = {}
    real = ask_orchestrator.run_models_concurrent

    async def spy(*args, **kwargs):
        seen["tracker"] = kwargs.get("tracker")
        return await real(*args, **kwargs)

    def never(*_args, **_kwargs):
        raise AssertionError("ask must not reach run_all_models")

    monkeypatch.setattr(ask_orchestrator, "run_models_concurrent", spy)
    monkeypatch.setattr(base, "run_all_models", never)
    return seen


def spy_on_the_phase_ticker(monkeypatch):
    """Snapshot the map at the instant the phase-scoped ticker is started.

    That instant is the boundary the plan draws: the fan-out has swept and
    returned, so whatever the map says here is what a table rendered an hour
    later must still say. Anything the second ticker did to it would show up as a
    difference against the file at the end of the run.
    """
    from utils import progress_tracker

    seen = {}

    class Spy(progress_tracker.PhaseProgress):
        def start_ticker(self):
            seen["models"] = read_status(self.phase_dir).get("progress", {}).get("models")
            return super().start_ticker()

    monkeypatch.setattr(ask_orchestrator, "PhaseProgress", Spy)
    return seen


def break_the_codex_sandbox(monkeypatch, policy="off"):
    """The real codex adapter with only its probe argv replaced (see D6 above)."""
    from utils import provider_registry as registry
    from utils import review_orchestrator_base as rob
    from utils.providers import codex as codex_module
    from utils.providers.base import PROFILE_GENERATE

    monkeypatch.setattr(rob, "_FANOUT_SANDBOX_CACHE", {})
    line = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
    broken = ["sh", "-c", f"echo '{line}' >&2; exit 1"]
    provider = registry.get_provider("codex")
    monkeypatch.setattr(
        type(provider), "sandbox_probe_command",
        lambda _self, profile: broken if profile == PROFILE_GENERATE else None,
    )
    monkeypatch.setattr(
        codex_module, "sandbox_fallback_policy", lambda anchor=None: policy
    )


class TestTheAskFanoutWritesAPerModelMap:
    """T4: the map is complete from the first poll and terminal at the end."""

    def test_the_map_is_complete_at_the_first_poll_and_wholly_terminal_after(
        self, plan, monkeypatch, progress_lines
    ):
        """Seeded before the tasks exist, not filled in as models start.

        The launch write drops a predecessor's ``progress`` wholesale, so what
        this run writes is all any reader ever sees: a model missing from the map
        is a bug, never a model carried over. The first dispatched slot reads the
        checkpoint here — the earliest moment a poller could — and must already
        see every requested spec.

        That used to be credited to ``claim_run``, which only the *detached*
        worker calls; on the foreground lane nothing dropped it and
        ``write_status`` carries the reserved objects over from disk by design
        (F1, 2026-09-20, pinned by the test below).
        """
        from utils.progress_tracker import LIVE_STATES, TERMINAL_STATES

        d = qdir(plan, "Q")
        specs = ["claude-code:sonnet", "claude-code:haiku", "claude-code:opus"]
        seen = spy_on_the_fanout(monkeypatch)

        class Snapshotting(FakeInvoker):
            first_poll = None
            first_spec = None

            def __call__(self, prompt, model_spec, timeout=None, log_file=None,
                         cwd=None, anchor=None, **kwargs):
                if Snapshotting.first_poll is None:
                    Snapshotting.first_spec = model_spec
                    Snapshotting.first_poll = progress_of(d)
                return super().__call__(prompt, model_spec, timeout, log_file,
                                        cwd, anchor)

        inv = Snapshotting()
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", *specs], inv)
        assert rc == 0

        # (a) complete at the first poll, and every entry already carries the
        #     budget a queued row renders its denominator from.
        first = Snapshotting.first_poll
        assert sorted(first["models"]) == sorted(specs)
        assert first["models_total"] == 3
        for spec in specs:
            entry = first["models"][spec]
            assert entry["state"] in set(LIVE_STATES) | set(TERMINAL_STATES)
            assert entry["started_at"] and entry["timeout_s"]

        # (a2) F1, 2026-09-20. The poll above is taken from *inside* the
        #      invoker, so the slot it belongs to is being dispatched at that
        #      instant and must read `running` — with the attempt and the whole
        #      slot budget the row renders as `0m00s/30m`, exactly as
        #      `run_single_model` and `run_single_review` write it on the other
        #      two lanes. Until F1 this lane took no `running` transition at
        #      all: the entry read `queued` here and for the model's whole
        #      30-minute run, and every `[PROGRESS]` line of the fan-out said
        #      `0/N done | staggering 1 | queued N-1` without ever naming a
        #      model. The loop above deliberately keeps its own weaker
        #      assertion, which is the whole state enum and passes for any
        #      state — it says "every row exists and is seeded", not "this row
        #      is live".
        dispatching = first["models"][Snapshotting.first_spec]
        assert dispatching["state"] == "running", dispatching
        assert dispatching["attempt"] == 1
        assert dispatching["timeout_s"] == first["models"][specs[0]]["timeout_s"]
        # `error` is terminal-and-failed-only, and `source` is meaningful only
        # after a completed attempt.
        assert "error" not in dispatching
        assert dispatching["source"] is None

        # (b) the tracker is the fan-out's, not the wrapper's (see the spy).
        assert seen["tracker"] is not None
        assert seen["tracker"].phase_dir == str(d)

        # (c) a ticker really ran on this path.
        assert [ln for ln in progress_lines if " done (" in ln]

        # (d) wholly terminal once the fan-out returned.
        final = progress_of(d)
        assert {e["state"] for e in final["models"].values()} == {"done"}
        assert (final["models_done"], final["models_succeeded"],
                final["models_total"]) == (3, 3, 3)

    def test_the_launch_write_drops_a_predecessors_map(self, plan, monkeypatch,
                                                       progress_lines):
        """A re-asked question must not open on the last run's answers.

        A question dir is keyed by the question, so asking the same question
        twice lands in the same dir — and until F1 (2026-09-20) the previous
        run's per-model map was still in it when the new run reported its first
        progress. The snapshot is taken as the launch `report_progress` is
        entered, which is the first thing after the write that claims the dir
        and before the tracker seeds, because the seed replaces `models`
        wholesale and a final-state assertion would pass either way.
        """
        from utils import status_file

        d = qdir(plan, "Q")
        os.makedirs(d, exist_ok=True)
        status_file.write_status(d, {
            "phase": "ask", "state": "complete",
            "models_requested": ["gemini:old-a", "pi:old-b"],
            "progress": {
                "step": "complete", "models_total": 2, "models_done": 2,
                "models_succeeded": 2,
                "models": {
                    "gemini:old-a": {"state": "done", "elapsed_s": 300},
                    "pi:old-b": {"state": "done", "elapsed_s": 240},
                },
            },
        })

        snapshots = []
        real = ask_orchestrator.report_progress

        def record(phase_dir, **updates):
            if not snapshots:
                snapshots.append(read_status(str(phase_dir)).get("progress"))
            real(phase_dir, **updates)

        monkeypatch.setattr(ask_orchestrator, "report_progress", record)
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet"], FakeInvoker())
        assert rc == 0
        assert snapshots, "the launch never reported progress"
        assert not snapshots[0], (
            f"the predecessor's progress survived the launch write: "
            f"{snapshots[0]}"
        )
        assert set(models_of(d)) == {"claude-code:sonnet"}

    def test_a_resumed_run_does_not_report_a_skipped_model_as_pending(
        self, plan, monkeypatch, progress_lines
    ):
        """Acceptance criterion 2, on the lane where nothing dispatches at all.

        A model answered from disk is terminal the moment it is seeded, so it
        counts on **both** numbers: it has finished its slot and it has a usable
        result. Left ``queued`` — the default if the skip predicate's return were
        not fed to the tracker — `models_done` and the rendered table would lie
        about a run with nothing left to do.
        """
        specs = ["claude-code:sonnet", "claude-code:haiku", "claude-code:opus"]
        argv = ["--plan-file", str(plan), "--question", "Q", "--models", *specs]
        assert run_ask(argv, FakeInvoker()) == 0

        second = FakeInvoker()
        assert run_ask(argv, second) == 0
        assert second.calls == []  # every model came off disk

        d = qdir(plan, "Q")
        progress = progress_of(d)
        for spec in specs:
            entry = progress["models"][spec]
            assert entry["state"] == "done"
            # There is no `skipped` state; the distinction lives in `source`.
            assert entry["source"] == "resume"
            assert entry["attempt"] == 0
            assert entry["elapsed_s"] == 0
            assert "error" not in entry
        assert not [e for e in progress["models"].values() if e["state"] == "queued"]
        assert (progress["models_done"], progress["models_succeeded"],
                progress["models_total"]) == (3, 3, 3)


class TestEveryAskSourceLandsOnTheMap:
    """T4: ``ask_one``'s returns already carry the ``source`` the entry needs.

    One case per value of the enum this lane produces — ``file``, ``fallback``,
    ``failed``, ``resume`` and the pre-dispatch ``not-dispatched`` — because
    ``record_result`` copies the value through verbatim and a renderer labels a
    row "skipped" by reading it.
    """

    def test_file_fallback_and_failed_land_as_their_own_wording(
        self, plan, progress_lines
    ):
        inv = FakeInvoker()
        # claude-code inherits `supports_file_output = True`, so a model that
        # writes the file is the `file` source; one that answers on stdout is
        # recovered and persisted, which is `fallback`; one with nothing usable
        # anywhere is `failed`.
        inv.configure("claude-code:sonnet", write_file=True, content="FROM THE FILE")
        inv.configure("claude-code:haiku", write_file=False,
                      result={"success": True, "data": "RECOVERED ON STDOUT"})
        inv.configure("claude-code:opus", write_file=False,
                      result={"success": False, "error": "provider exploded",
                              "data": None})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q", "--models",
                      "claude-code:sonnet", "claude-code:haiku",
                      "claude-code:opus"], inv)
        assert rc == 0  # a partial answer set is usable output

        models = models_of(qdir(plan, "Q"))
        assert (models["claude-code:sonnet"]["state"],
                models["claude-code:sonnet"]["source"]) == ("done", "file")
        assert (models["claude-code:haiku"]["state"],
                models["claude-code:haiku"]["source"]) == ("done", "fallback")
        assert (models["claude-code:opus"]["state"],
                models["claude-code:opus"]["source"]) == ("failed", "failed")
        # `error` is terminal-only and is absent on a `done` entry.
        assert models["claude-code:opus"]["error"] == "provider exploded"
        assert "error" not in models["claude-code:sonnet"]

    def test_the_resume_source_marks_only_the_slot_that_came_off_disk(
        self, plan, progress_lines
    ):
        inv = FakeInvoker()
        assert run_ask(["--plan-file", str(plan), "--question", "Q",
                        "--models", "claude-code:sonnet"], inv) == 0

        # Same question, same dir: sonnet resumes, haiku is asked for the first
        # time, and the two sources have to be distinguishable afterwards.
        second = FakeInvoker()
        assert run_ask(["--plan-file", str(plan), "--question", "Q", "--models",
                        "claude-code:sonnet", "claude-code:haiku"], second) == 0
        assert second.calls == ["claude-code:haiku"]

        models = models_of(qdir(plan, "Q"))
        assert models["claude-code:sonnet"]["source"] == "resume"
        assert models["claude-code:haiku"]["source"] == "file"
        assert {e["state"] for e in models.values()} == {"done"}

    def test_the_pre_dispatch_refusal_lands_as_not_dispatched(
        self, plan, monkeypatch, progress_lines
    ):
        """The one terminal transition that happens before any CLI is launched.

        It is covered by ``record_result`` like every other return, so the
        refusal site adds no transition of its own — a second one would be the
        double-write the plan forbids.
        """
        break_the_codex_sandbox(monkeypatch, policy="off")
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "codex:gpt-5.2-codex"], FakeInvoker())
        assert rc == 3  # zero answers: nothing to aggregate

        entry = models_of(qdir(plan, "Q"))["codex:gpt-5.2-codex"]
        assert entry["state"] == "failed"
        assert entry["source"] == "not-dispatched"
        assert entry["error"].startswith("[SANDBOX_UNAVAILABLE]")


class TestEveryEndingRetiresItsSlot:
    """T4: after the fan-out returns, nothing is still described as live."""

    def test_no_slot_is_left_live_and_each_carries_its_own_ending(
        self, plan, progress_lines
    ):
        """Four different endings in one run: an answer, a refusal the lane
        produced, a slot that *raised* (retired by the helper as ``exception``,
        the only place the spec behind a dropped task is still known), and a
        model that came off disk."""
        from utils import progress_tracker
        from utils.progress_tracker import LIVE_STATES

        d = qdir(plan, "Q")
        seed = FakeInvoker()
        seed.configure("claude-code:opus", write_file=True, content="EARLIER ANSWER")
        assert run_ask(["--plan-file", str(plan), "--question", "Q",
                        "--models", "claude-code:opus"], seed) == 0

        class Exploding(FakeInvoker):
            def __call__(self, prompt, model_spec, timeout=None, log_file=None,
                         cwd=None, anchor=None, **kwargs):
                if model_spec == "claude-code:haiku":
                    raise RuntimeError("boom")
                return super().__call__(prompt, model_spec, timeout, log_file,
                                        cwd, anchor)

        inv = Exploding()
        inv.configure("claude-code:sonnet", write_file=True, content="AN ANSWER")
        inv.configure("claude-code:fable", write_file=False,
                      result={"success": False, "error": "no usable answer", "data": None})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q", "--models",
                      "claude-code:sonnet", "claude-code:haiku",
                      "claude-code:fable", "claude-code:opus"], inv)
        assert rc == 0

        progress = progress_of(d)
        models = progress["models"]
        assert not [s for s, e in models.items() if e["state"] in LIVE_STATES]
        assert (models["claude-code:sonnet"]["state"],
                models["claude-code:sonnet"]["source"]) == ("done", "file")
        assert (models["claude-code:opus"]["state"],
                models["claude-code:opus"]["source"]) == ("done", "resume")
        assert (models["claude-code:fable"]["state"],
                models["claude-code:fable"]["source"]) == ("failed", "failed")
        assert models["claude-code:fable"]["error"] == "no usable answer"
        # The slot that raised keeps its own exception text, while the report's
        # failed map — built from `results`, which a raised task never reaches —
        # falls to F11's backfill. The map is the finer-grained of the two.
        #
        # Updated for F4 (2026-09-20): this used to assert the sentence "No
        # answer produced (model did not complete)", which was ask's own third
        # wording for the one cause the other two lanes name once. A slot in
        # neither `results` nor the failure map is `DEADLINE_ERROR` here exactly
        # as it is in `run_models_concurrent`'s sibling test
        # (`test_a_slot_that_raises_retires_itself_and_leaves_its_sibling_alone`,
        # tests/test_review_plan_orchestrator.py), which pins the same
        # asymmetry: the map keeps the exception, the report's list — which has
        # no exception in hand — names the deadline wording. That the sentence
        # was *right* about this particular slot is a coincidence of a generic
        # phrasing; it was equally applied to a genuine deadline loss, whose
        # swept map entry said `DEADLINE_ERROR` in the same file.
        assert (models["claude-code:haiku"]["state"],
                models["claude-code:haiku"]["source"]) == ("failed", "exception")
        assert models["claude-code:haiku"]["error"] == "boom"
        assert read_status(d)["models_failed"]["claude-code:haiku"] == (
            progress_tracker.DEADLINE_ERROR
        )

        # Both counters, and the right way round: four finished slots, two of
        # which produced a usable answer.
        assert (progress["models_done"], progress["models_succeeded"],
                progress["models_total"]) == (4, 2, 4)
        assert progress["step"] == "complete"


class TestTheDeadlineIsNamedOnceOnThisLane:
    """F4, 2026-09-20: ask used to invent a **third** wording for one cause.

    The other two fan-out lanes hand every model the deadline lost to
    ``backfill_deadline_losses``, which writes
    ``progress_tracker.DEADLINE_ERROR`` into ``failed_models`` and into the map
    in one pass — the whole point being that the table and the report's failure
    list name one cause.  ask used neither: its failed map was built by a lone
    loop ending in ``or "No answer produced (model did not complete)"``, so a
    slot the deadline cancelled was described one way in ``answers.md`` and in
    ``models_failed`` and another way, in the same ``.status.json``, by the
    entry the fan-out's own sweep had just frozen.
    """

    def test_a_deadline_loss_reads_the_same_in_the_map_and_in_the_report(
        self, plan, progress_lines
    ):
        from utils import progress_tracker
        from utils import review_orchestrator_base as base

        specs = ["claude-code:sonnet", "claude-code:haiku"]
        # `--timeout 0` with the stagger patched out makes the fan-out's own
        # `total_timeout` 0, so `asyncio.wait_for` fires its deadline before any
        # slot can run: the real deadline path, not a simulated one.
        with patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                          "--timeout", "0", "--models", *specs], FakeInvoker())
        assert rc == 3  # zero answers: nothing to aggregate

        d = qdir(plan, "Q")
        status = read_status(d)
        assert status["models_failed"] == {
            spec: progress_tracker.DEADLINE_ERROR for spec in specs
        }
        # The map's swept entries, which always said this, and which the report
        # used to contradict.
        models = status["progress"]["models"]
        for spec in specs:
            assert models[spec]["state"] == "failed"
            assert models[spec]["error"] == progress_tracker.DEADLINE_ERROR

        answers = Path(d, "answers.md").read_text(encoding="utf-8")
        assert progress_tracker.DEADLINE_ERROR in answers
        assert "No answer produced (model did not complete)" not in answers


class TestThePhaseScopedTicker:
    """T5: the window between the last model and the report is not silent."""

    def test_it_freezes_the_fanouts_map_and_its_last_line_names_the_terminal_step(
        self, plan, monkeypatch, progress_lines
    ):
        seen = spy_on_the_phase_ticker(monkeypatch)
        inv = FakeInvoker()
        inv.configure("claude-code:haiku", write_file=False,
                      result={"success": False, "error": "nothing usable", "data": None})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet", "claude-code:haiku"], inv)
        assert rc == 0

        d = qdir(plan, "Q")
        # The phase ticker writes nothing at all, so the map the sweep froze is
        # byte-identical either side of it — which is what keeps a table
        # rendered during grouping showing each model's real duration rather
        # than a clock that kept running after the model finished.
        assert seen["models"] is not None
        assert json.dumps(seen["models"], sort_keys=True) == json.dumps(
            models_of(d), sort_keys=True
        )

        fanout_lines = [ln for ln in progress_lines if " done (" in ln]
        phase_lines = [ln for ln in progress_lines if ln.startswith("[PROGRESS] step: ")]
        assert fanout_lines and phase_lines
        # Stopped after the terminal step was written, so the log's last
        # [PROGRESS] reads `complete` instead of trailing off at `fanout`.
        assert progress_lines[-1].startswith("[PROGRESS] step: complete")
        # Lifetime rule 1: the fan-out's own ticker is dead before the phase's
        # first line, so no [PROGRESS] can interleave with the per-model summary.
        assert progress_lines.index(fanout_lines[-1]) < progress_lines.index(phase_lines[0])


class TestTheQuestionDirIsTheSelector:
    """T4: ``ask`` writes to the question dir, not the ``ask`` phase dir.

    Two questions asked about one plan are two independent runs with two answer
    sets; a tracker built against ``{plan}/ask/`` would merge both into one map
    and hand a poller a run it could not attribute to either question. (The
    ``status --question`` half of this belongs to the run-view lane; what is
    pinned here is the writing half.)
    """

    def test_two_questions_under_one_plan_keep_their_own_maps(self, plan, progress_lines):
        first = FakeInvoker()
        assert run_ask(["--plan-file", str(plan), "--question", "First question?",
                        "--models", "claude-code:sonnet"], first) == 0
        second = FakeInvoker()
        assert run_ask(["--plan-file", str(plan), "--question", "Second question?",
                        "--models", "claude-code:haiku", "claude-code:opus"], second) == 0

        a = qdir(plan, "First question?")
        b = qdir(plan, "Second question?")
        assert a != b
        assert sorted(models_of(a)) == ["claude-code:sonnet"]
        assert sorted(models_of(b)) == ["claude-code:haiku", "claude-code:opus"]
        assert progress_of(a)["models_total"] == 1
        assert progress_of(b)["models_total"] == 2
        # The phase dir above them is not a run and never gains a checkpoint.
        assert not (ask_root(plan) / ".status.json").exists()
