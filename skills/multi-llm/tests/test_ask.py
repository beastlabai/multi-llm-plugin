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
    build_inline_plan,
    demote_headings,
    looks_like_ndjson,
    parse_args,
    recover_answer_text,
    render_prompt,
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
    p.write_text("# My Feature\n\n## Overview\nBuild a widget with rollback.\n")
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
    temp dir names that happen to contain 'answer'."""
    after = prompt.split("exact absolute file path:", 1)[1]
    for line in after.splitlines():
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
    """

    def __init__(self):
        self.behaviors = {}
        self.calls = []
        self.prompts = {}

    def configure(self, model_spec, **behavior):
        self.behaviors[model_spec] = behavior

    def __call__(self, prompt, model_spec, timeout=None, log_file=None, cwd=None):
        self.calls.append(model_spec)
        self.prompts[model_spec] = prompt
        b = self.behaviors.get(model_spec, {"write_file": True})
        if b.get("write_file", False):
            answer_path = extract_output_path(prompt)
            if answer_path:
                os.makedirs(os.path.dirname(answer_path), exist_ok=True)
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(b.get("content", f"Answer from {model_spec}."))
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
        (base / ".status.json").write_text(json.dumps({"question": "Q"}))
        d, is_resume = resolve_question_dir(str(tmp_path), "slug-abc", "Q")
        assert d == str(base)
        assert is_resume is True

    def test_different_question_gets_sibling(self, tmp_path):
        base = tmp_path / "slug-abc"
        base.mkdir()
        (base / ".status.json").write_text(json.dumps({"question": "Q-original"}))
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
        assert (d / f"answer_{sanitized}.md").read_text() == "FILE ANSWER"
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
        assert rc == 1
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
        assert rc == 1
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
        assert rc == 1
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
        assert (d / f"answer_{sanitized}.md").read_text() == "RECOVERED"

        # Second run: model should be SKIPPED (file persisted from fallback).
        inv2 = FakeInvoker()
        inv2.configure("claude-code:sonnet", write_file=True, content="SHOULD NOT RUN")
        run_ask(["--plan-file", str(plan), "--question", "Q",
                 "--models", "claude-code:sonnet"], inv2)
        assert inv2.calls == []  # skipped on resume
        assert (d / f"answer_{sanitized}.md").read_text() == "RECOVERED"


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
            (d / f"answer_{sanitize_model_name(spec)}.md").write_text(content)
        specs = ["gemini:gemini-3-pro", "claude-code:sonnet", "cursor-agent:auto"]
        path1, completed1 = write_answers_md(str(d), "Q", str(plan), specs, {}, "TS")
        text1 = Path(path1).read_text()
        path2, completed2 = write_answers_md(str(d), "Q", str(plan), specs, {}, "TS")
        text2 = Path(path2).read_text()
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

    def test_force_reasks_all_models(self, plan):
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
        # Re-asked despite existing answer file.
        assert inv2.calls == ["claude-code:sonnet"]
        assert "round-2" in read_answers(d)
        assert "round-1" not in read_answers(d)


# ---------------------------------------------------------------------------
# 10, 13. Resume / corrupt-empty resume
# ---------------------------------------------------------------------------

class TestResumeBehavior:
    def test_existing_nonempty_answer_skipped(self, plan):
        d = qdir(plan, "Q")
        d.mkdir(parents=True)
        sanitized = sanitize_model_name("claude-code:sonnet")
        (d / f"answer_{sanitized}.md").write_text("prior answer")
        # Seed a status so resolve_question_dir resumes into the dir.
        (d / ".status.json").write_text(json.dumps({"question": "Q"}))
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
        (d / f"answer_{sanitized}.md").write_text("   \n  ")  # whitespace-only
        (d / ".status.json").write_text(json.dumps({"question": "Q"}))
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
        plan.write_text("# My Feature\n\nCompletely different content now.\n")
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

    def test_all_fail_exit_1_with_files(self, plan):
        inv = FakeInvoker()
        for spec in ("claude-code:sonnet", "cursor-agent:auto"):
            inv.configure(spec, write_file=False,
                          result={"success": False, "error": "nope", "error_code": "SUBPROCESS_FAILED"})
        rc = run_ask(["--plan-file", str(plan), "--question", "Q",
                      "--models", "claude-code:sonnet", "cursor-agent:auto"], inv)
        assert rc == 1
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
        ask_md = (SKILL_DIR / "instructions" / "ask.md").read_text()
        assert "/multi-llm:multi-llm --ask" in ask_md
        # Foreground --project orchestrator invocation.
        assert "--project ${CLAUDE_SKILL_DIR}" in ask_md
        assert "ask_orchestrator.py" in ask_md
        # Shell-safe mechanisms + realpath quoting.
        assert "--question-file" in ask_md
        assert "--question-env" in ask_md
        assert '"$(realpath "$PLAN_PATH")"' in ask_md
        # Single-quoted-question rule + error cases.
        assert "single argument" in ask_md
        assert "empty/whitespace-only question is an error" in ask_md or "empty question" in ask_md.lower()

    def test_skill_md_lists_ask_mode(self):
        skill = (SKILL_DIR / "SKILL.md").read_text()
        # argument-hint, mode list, quick start, mode detection.
        assert "--ask" in skill
        assert "11. **Ask**" in skill or "**Ask** (`--ask`)" in skill
        assert '/multi-llm:multi-llm --ask plans/my-feature.md "Is the rollback strategy sufficient?"' in skill
        assert '"--ask" -> mode = ask' in skill
        # Intro count bumped (no longer "nine").
        assert "Supports eleven workflow modes" in skill
        # Mode-detection parsing rule states the same single-quoted-question rule.
        assert "position-independent" in skill
        assert "single (quoted) argument" in skill

    def test_skill_and_instructions_agree_on_error_rule(self):
        skill = (SKILL_DIR / "SKILL.md").read_text()
        ask_md = (SKILL_DIR / "instructions" / "ask.md").read_text()
        for doc in (skill, ask_md):
            assert "unquoted" in doc.lower()
            assert "usage hint" in doc.lower()

    def test_documented_bash_timeout_covers_max_provider_timeout(self):
        """The Bash timeout documented for ask mode (in SKILL.md Critical Rule 1
        and instructions/ask.md) must be >= the max provider default_timeout, so
        a 20-min Bash cap can never silently sit over a 30-min provider budget
        and kill a slow model before its own timeout fires."""
        import yaml

        providers = yaml.safe_load((SKILL_DIR / "providers.yaml").read_text())
        max_provider_timeout_s = max(
            cfg.get("default_timeout", 1200)
            for cfg in providers.get("providers", {}).values()
        )

        skill = (SKILL_DIR / "SKILL.md").read_text()
        ask_md = (SKILL_DIR / "instructions" / "ask.md").read_text()
        max_provider_timeout_ms = max_provider_timeout_s * 1000

        def documented_timeouts_ms(text):
            # Match both `timeout: 2000000` (Bash-tool form) and prose like
            # "a Bash `timeout` of ~2000000 ms" — i.e. any 6+ digit count
            # presented as a Bash timeout value or explicitly in milliseconds.
            matches = re.findall(r"timeout`?\s*[:=]\s*[`~]?\s*([\d,_]{6,})", text)
            matches += re.findall(r"[`~]?([\d,_]{6,})\s*ms\b", text)
            return [int(m.replace(",", "").replace("_", "")) for m in matches]

        # SKILL.md Critical Rule 1 must explicitly carve out ask mode, and the
        # timeout it documents for that carve-out (the text after "Exception")
        # must exceed the slowest provider budget. The 20-min global default
        # that precedes the exception is intentionally NOT the ask budget.
        assert "ask mode" in skill and "`--ask`" in skill
        exception_clause = skill.split("**Exception", 1)
        assert len(exception_clause) == 2, "SKILL.md ask-mode timeout exception missing"
        skill_ask_clause = exception_clause[1].split("\n", 1)[0]
        skill_ask_timeouts = [
            v for v in documented_timeouts_ms(skill_ask_clause) if v >= 1_200_000
        ]
        assert skill_ask_timeouts, "no ask-mode Bash timeout found in SKILL.md exception"
        assert min(skill_ask_timeouts) >= max_provider_timeout_ms, (
            f"SKILL.md ask-mode Bash timeout ({min(skill_ask_timeouts)} ms) is below "
            f"the max provider default_timeout ({max_provider_timeout_ms} ms)"
        )

        # instructions/ask.md is entirely about ask mode: its documented Bash
        # timeout(s) must likewise cover the slowest provider budget.
        ask_md_timeouts = [
            v for v in documented_timeouts_ms(ask_md) if v >= 1_200_000
        ]
        assert ask_md_timeouts, "no documented Bash timeout found in ask.md"
        assert min(ask_md_timeouts) >= max_provider_timeout_ms, (
            f"ask.md documents a Bash timeout ({min(ask_md_timeouts)} ms) below "
            f"the max provider default_timeout ({max_provider_timeout_ms} ms)"
        )
