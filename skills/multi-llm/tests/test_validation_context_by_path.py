#!/usr/bin/env python3
"""Validation context by path (plans/validation-context-by-path-and-codex-0156-fallout.md §2).

Above ``VALIDATION_CONTEXT_LIMIT`` a plan (or review-tasks' tasks file) keeps its
head excerpt inline and is named by the round's hash-bound snapshot; the task is
``context_mode: workspace-read``. Below it nothing changes. A document with no
readable file keeps the bare cut and is surfaced as ``mode: truncated``.

Task ids in the class names are the plan's (2.1, 2.1b, 2.2, 2.4, 2.5, 2.6, 2.9,
2.10).
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import review_orchestrator_base as rob  # noqa: E402
from utils import status_file  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.subagent_tasks import (  # noqa: E402
    PromptBudget,
    TaskFileError,
    build_inputs,
    content_hash,
    file_bytes_sha256,
    read_task_file,
    refuse_reason,
    validate_subagent_task,
)
from utils.validation import (  # noqa: E402
    CONTEXT_CONTRACT,
    INSUFFICIENT_CONTEXT_PREFIX,
    LEGACY_TRUNCATION_MARKER,
    PLAN_SNAPSHOT_NAME,
    REVALIDATION_SNAPSHOT_NAME,
    SNAPSHOT_ERROR,
    TASKS_BACKUP_NAME,
    TASKS_SNAPSHOT_NAME,
    VALIDATION_CONTEXT_LIMIT,
    ValidationContextError,
    bind_document_file,
    bind_document_text,
    merge_batched_validation_results,
    prepare_batched_revalidation_tasks,
    prepare_batched_validation_tasks,
    verify_context_sources,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def big_doc(title="Plan", sections=6, size=15000, tail=""):
    """A markdown document over the limit: ``sections`` × ``size`` chars."""
    body = "".join(
        f"## {i}. Section {i}\n\n" + ("lorem ipsum " * (size // 12)) + "\n\n"
        for i in range(sections)
    )
    return f"# {title}\n\n{body}{tail}"


def groups(n=3, title="t", desc="d"):
    return [
        {"theme": f"Theme {i}", "category": "x", "models": ["m"],
         "suggestions": [{"title": f"{title} {i}", "desc": desc,
                          "importance": "MEDIUM", "type": "addition"}]}
        for i in range(n)
    ]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def prepare(phase: Path, plan: Path, document, n=3, **kwargs):
    return prepare_batched_validation_tasks(
        groups=groups(n), context=document.text, output_dir=str(phase),
        plan_file=str(plan), context_document=document, **kwargs,
    )


def snapshot_path_in(prompt: str):
    match = re.search(r"## (?:Plan|Tasks) file\n\n[^\n]*\n\n(\S+)\n", prompt)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Below the limit nothing changes
# ---------------------------------------------------------------------------

class TestUnderTheLimit:
    def test_inline_no_path_no_sources_and_nothing_published(self, tmp_path, capsys):
        plan = write(tmp_path / "plan.md", "# Plan\n\n## A\nshort\n")
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        result = prepare(phase, plan, document)
        for task in result["batches"]:
            assert task["context_mode"] == "inline"
            assert "context_sources" not in task
            assert "## Plan file" not in task["prompt"]
            assert str(phase) not in task["prompt"]
        assert result["context_contract"] == CONTEXT_CONTRACT
        # 2.5 (g): an under-limit source prints no [VALIDATION_CONTEXT_*] line.
        assert "[VALIDATION_CONTEXT_" not in capsys.readouterr().err
        # And publishes nothing: no checkpoint is created just to say so.
        assert not (phase / ".status.json").exists()

    def test_the_prompt_is_byte_identical_to_the_documentless_call(self, tmp_path):
        plan = write(tmp_path / "plan.md", "# Plan\n\n## A\nshort\n")
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        with_doc = prepare(phase, plan, document)
        without = prepare_batched_validation_tasks(
            groups=groups(), context=document.text, output_dir=str(phase),
            plan_file=str(plan),
        )
        assert [t["prompt"] for t in with_doc["batches"]] == [
            t["prompt"] for t in without["batches"]
        ]
        assert with_doc["inputs"] == without["inputs"]


# ---------------------------------------------------------------------------
# 2.1: one read, one snapshot, one hash domain; refusals
# ---------------------------------------------------------------------------

class TestOneReadOneSnapshot:
    def test_a_edit_between_the_read_and_the_snapshot(self, tmp_path):
        """(a) the snapshot is written from the bytes read, never re-copied."""
        v1, v2 = big_doc("v1"), big_doc("v2")
        plan = write(tmp_path / "plan.md", v2)  # the live file already moved on
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase), data=v1.encode("utf-8"))
        snapshot = phase / PLAN_SNAPSHOT_NAME
        assert snapshot.read_text(encoding="utf-8") == v1
        result = prepare(phase, plan, document)
        assert result["inputs"]["plan_sha256"] == content_hash(v1)
        task = result["batches"][0]
        assert "# v1" in task["prompt"] and "# v2" not in task["prompt"]
        assert task["context_sources"][0]["sha256"] == file_bytes_sha256(str(snapshot))

    def test_b_edit_between_the_snapshot_and_materialization(self, tmp_path):
        v1 = big_doc("v1")
        plan = write(tmp_path / "plan.md", v1)
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        write(plan, big_doc("v2"))
        result = prepare(phase, plan, document)
        task = result["batches"][0]
        assert "# v1" in task["prompt"] and "# v2" not in task["prompt"]
        assert result["inputs"]["plan_sha256"] == content_hash(v1)
        assert snapshot_path_in(task["prompt"]) == str(phase / PLAN_SNAPSHOT_NAME)
        assert (phase / PLAN_SNAPSHOT_NAME).read_text(encoding="utf-8") == v1

    def test_c_edit_between_materialization_and_dispatch(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc("v1"))
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        result = prepare(phase, plan, document)
        # An edit to the LIVE plan does not touch what the round names.
        write(plan, big_doc("v2"))
        assert verify_context_sources(result, str(phase)) is False
        # An edit to the SNAPSHOT refuses the round with the named error, and
        # the dispatch-side wrapper exits 3 before anything is dispatched.
        write(phase / PLAN_SNAPSHOT_NAME, big_doc("tampered"))
        with pytest.raises(ValidationContextError, match=re.escape(SNAPSHOT_ERROR)):
            verify_context_sources(result, str(phase))
        with pytest.raises(SystemExit) as exit_info:
            rob.verify_round_context(result, str(phase))
        assert exit_info.value.code == 3
        (phase / PLAN_SNAPSHOT_NAME).unlink()
        with pytest.raises(ValidationContextError, match="missing"):
            verify_context_sources(result, str(phase))

    def test_c2_a_dotdot_path_never_rebinds_outside_the_phase_dir(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc("v1"))
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        result = prepare(phase, plan, document)
        record = result["batches"][0]["context_sources"][0]
        # A forged record: a hash-matching copy outside the phase dir.
        outside = write(tmp_path / "elsewhere.md", (phase / PLAN_SNAPSHOT_NAME)
                        .read_text(encoding="utf-8"))
        assert file_bytes_sha256(str(outside)) == record["sha256"]
        record["path"] = os.path.join("..", "elsewhere.md")
        with pytest.raises(ValidationContextError, match="outside the phase dir"):
            verify_context_sources(result, str(phase))
        assert str(outside) not in result["batches"][0]["prompt"]

    def test_d_edit_between_a_run_and_its_reaggregate(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc("v1"))
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        result = prepare(phase, plan, document)
        tasks_path = phase / "validation_tasks.json"
        tasks_path.write_text(json.dumps(result), encoding="utf-8")
        grouped = phase / "grouped.json"
        # Unedited: the reaggregate's live inputs bind, and the snapshot verifies.
        read_task_file(str(tasks_path), phase_dir=str(phase), current_inputs=build_inputs(
            plan_file=str(plan), grouped_file=str(grouped)))
        write(plan, big_doc("v2"))
        with pytest.raises(TaskFileError, match="plan_sha256"):
            read_task_file(str(tasks_path), phase_dir=str(phase), current_inputs=build_inputs(
                plan_file=str(plan), grouped_file=str(grouped)))

    def test_a_non_utf8_plan_refuses_the_round(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_bytes(b"# Plan\n\xff\xfe broken\n")
        with pytest.raises(ValidationContextError, match="not valid UTF-8"):
            bind_document_file(str(plan), str(tmp_path / "review-plan"))

    def test_crlf_bytes_are_hashed_raw_and_excerpted_translated(self, tmp_path):
        text = big_doc("crlf").replace("\n", "\r\n")
        plan = tmp_path / "plan.md"
        plan.write_bytes(text.encode("utf-8"))
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        assert (phase / PLAN_SNAPSHOT_NAME).read_bytes() == plan.read_bytes()
        assert document.sha256 == file_bytes_sha256(str(plan))
        assert "\r" not in document.text

    def test_an_unwritable_snapshot_names_the_live_file_only_on_a_hash_match(
        self, tmp_path, monkeypatch
    ):
        import utils.validation as validation

        plan = write(tmp_path / "plan.md", big_doc())
        phase = tmp_path / "review-plan"

        def refuse(target, data):
            raise OSError("read-only file system")

        monkeypatch.setattr(validation, "_atomic_write_bytes", refuse)
        matched = bind_document_file(str(plan), str(phase))
        assert matched.path == str(plan) and not matched.in_phase_dir
        # The live file moved on after the read: nothing readable is bound.
        stale = bind_document_file(
            str(plan), str(phase), data=big_doc("older").encode("utf-8")
        )
        assert stale.path is None and "no longer matches" in stale.reason
        result = prepare(phase, plan, stale)
        source = result["batches"][0]["context_sources"][0]
        assert source["mode"] == "truncated" and "could not be written" in source["reason"]
        assert result["batches"][0]["context_mode"] == "inline"


class TestRefusals:
    def test_e_an_incapable_runner_materializes_truncated_and_says_so(
        self, tmp_path, capsys
    ):
        plan = write(tmp_path / "plan.md", big_doc())
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        result = prepare(
            phase, plan, document,
            runner_read_refusal="runner fake:blind cannot read files",
        )
        for task in result["batches"]:
            assert task["context_mode"] == "inline"
            assert "## Plan file" not in task["prompt"]
            assert LEGACY_TRUNCATION_MARKER in task["prompt"]
        err = capsys.readouterr().err
        assert "[VALIDATION_CONTEXT_TRUNCATED] context" in err
        assert "fake:blind cannot read files" in err
        status = status_file.read_status(str(phase))["validation_context"]
        assert status["mode"] == "truncated"
        assert status["sources"][0]["reason"] == "runner fake:blind cannot read files"

    def test_e_a_budget_without_workspace_reads_is_the_same_refusal(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc())
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        result = prepare(
            phase, plan, document,
            prompt_budget=PromptBudget(None, supports_workspace_reads=False),
        )
        assert {t["context_mode"] for t in result["batches"]} == {"inline"}

    def test_e_materialization_resolves_the_executor_runner(self, monkeypatch):
        args = argparse.Namespace(exec_subagents=True, validation_model="auto")
        monkeypatch.setattr(executor, "resolve_task_runner", lambda task, anchor=None: "blind:m")
        import utils.provider_registry as registry

        monkeypatch.setattr(registry, "parse_model_spec", lambda spec, anchor=None: ("blind", "m"))
        monkeypatch.setattr(
            registry, "get_provider",
            lambda name: SimpleNamespace(supports_workspace_reads=False),
        )
        assert rob.materialization_read_refusal(args, "/x/plan.md") == (
            "runner blind:m cannot read files"
        )
        # A native host (no --exec-subagents) is never refused here.
        native = rob.materialization_read_refusal(
            argparse.Namespace(exec_subagents=False), "/x/plan.md"
        )
        assert native("/x/plan.md") is None

    def test_f_a_workspace_read_batch_under_an_incapable_runner_fails_named(
        self, tmp_path
    ):
        plan = write(tmp_path / "plan.md", big_doc())
        phase = tmp_path / "review-plan"
        result = prepare(phase, plan, bind_document_file(str(plan), str(phase)))
        task = result["batches"][0]
        before = json.dumps(task, sort_keys=True)
        reason = refuse_reason(task, supports_workspace_reads=False, runner="blind:m")
        assert "validation batch 0 needs workspace reads; runner blind:m cannot " \
            "read files; rematerialize with a capable runner" in reason
        # Nothing was re-materialized inline behind the operator's back.
        assert json.dumps(task, sort_keys=True) == before
        assert task["context_mode"] == "workspace-read"

    def test_g_the_unsplittable_message_no_longer_blames_the_schema(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc())
        phase = tmp_path / "review-plan"
        with pytest.raises(TaskFileError) as raised:
            prepare(
                phase, plan, bind_document_file(str(plan), str(phase)), n=1,
                prompt_budget=PromptBudget(10, supports_workspace_reads=True),
            )
        message = str(raised.value)
        assert "the schema rejects" not in message
        assert "inline part" in message and "never truncated" in message


# ---------------------------------------------------------------------------
# 2.1b: revalidation
# ---------------------------------------------------------------------------

def _failed(n):
    return [
        {"group_index": i, "status": "validation_failed", "reason": "x",
         "error_type": "parsing_error"}
        for i in range(n)
    ]


class TestRevalidation:
    def test_a_an_oversized_plan_revalidation_names_its_fresh_snapshot(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc("current"))
        review = tmp_path / "review-plan"
        document = bind_document_file(
            str(plan), str(review), snapshot_name=REVALIDATION_SNAPSHOT_NAME
        )
        result = prepare_batched_revalidation_tasks(
            groups=groups(), validation_results=_failed(3), context="",
            output_dir=str(review), plan_file=str(plan),
            context_document=document,
        )
        snapshot = review / REVALIDATION_SNAPSHOT_NAME
        assert result["context_kind"] == "plan"
        for task in result["batches"]:
            assert task["context_mode"] == "workspace-read"
            assert snapshot_path_in(task["prompt"]) == str(snapshot)
            assert task["context_sources"][0]["sha256"] == file_bytes_sha256(str(snapshot))
        # The original round's snapshot is never overwritten by revalidation.
        assert not (review / PLAN_SNAPSHOT_NAME).exists()

    def test_b_an_oversized_diff_never_names_a_plan_or_asks_for_a_read(self, tmp_path):
        diff = "diff --git a/x b/x\n" + "+line\n" * 20000
        review = tmp_path / "code-review"
        result = prepare_batched_revalidation_tasks(
            groups=groups(), validation_results=_failed(3), context=diff,
            output_dir=str(review), plan_file=str(tmp_path / "plan.md"),
            context_document=bind_document_text(diff, label="diff"),
        )
        for task in result["batches"]:
            prompt = task["prompt"]
            assert task["context_mode"] == "inline"
            assert "## Plan file" not in prompt
            assert "Read it in full" not in prompt and "read the full" not in prompt
            assert "## Context (Diff context)" in prompt
            source = task["context_sources"][0]
            assert source["label"] == "diff" and source["mode"] == "truncated"
        assert result["context_kind"] == "diff"

    def test_c_the_missing_diff_fallback_is_labelled_as_plan(self, tmp_path):
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator

        plan = write(tmp_path / "plan.md", big_doc())
        orchestrator = object.__new__(ApplyCodeFixesOrchestrator)
        orchestrator.out_dir = str(tmp_path / "out")
        orchestrator.review_subdir = "code-review"
        orchestrator.plan_path = str(plan)
        review = tmp_path / "out" / "code-review"
        document = orchestrator.get_revalidation_document(str(review))
        assert document.label == "plan"
        assert document.path == str(review / REVALIDATION_SNAPSHOT_NAME)
        # And with the diff present, the source is a diff with no path.
        write(review / "diff_context.txt", "diff --git a/x b/x\n+y\n")
        document = orchestrator.get_revalidation_document(str(review))
        assert document.label == "diff" and document.path is None


# ---------------------------------------------------------------------------
# 2.2: the prompt's path variant
# ---------------------------------------------------------------------------

RETENTION_PLAN = big_doc(tail="## 9. Data retention\n\nRetention is 30 days, then purged.\n")


def _stub_validator(task, *, can_read=True):
    """A runner that reads the snapshot the prompt names, if it can."""
    path = snapshot_path_in(task["prompt"])
    verdicts = []
    for gid, index in zip(task["group_ids"], task["group_indices"]):
        if path is None:
            status, reason = "valid", "the excerpt never mentions retention"
        elif not can_read:
            status = "needs-human-decision"
            reason = f"{INSUFFICIENT_CONTEXT_PREFIX} could not read {path} (§9 unread)"
        else:
            text = Path(path).read_text(encoding="utf-8")
            if "Retention is 30 days" in text:
                status, reason = "invalid", "§9 defines retention (30 days)"
            else:
                status, reason = "valid", "the plan never defines retention"
        verdicts.append({"group_index": index, "group_hash": gid,
                         "status": status, "reason": reason, "confidence": 0.9})
    return {"groups": verdicts}


class TestThePathVariant:
    def _round(self, tmp_path):
        plan = write(tmp_path / "plan.md", RETENTION_PLAN)
        phase = tmp_path / "review-plan"
        document = bind_document_file(str(plan), str(phase))
        result = prepare_batched_validation_tasks(
            groups=groups(1, title="The plan never defines retention"),
            context=document.text, output_dir=str(phase), plan_file=str(plan),
            context_document=document,
        )
        return phase, result

    def test_a_no_sentence_judges_omissions_against_the_text_above(self, tmp_path):
        _phase, result = self._round(tmp_path)
        prompt = result["batches"][0]["prompt"]
        assert "judged against the plan or tasks\ntext above" not in prompt
        assert "text above" not in prompt
        assert "cannot be determined from the context above" not in prompt
        assert "judged\nagainst the full file named above" in prompt
        assert "`insufficient-context:`" in prompt
        assert "sections not shown: " in prompt and "9. Data retention" in prompt

    def test_b_an_uncited_omission_claim_past_the_cut_comes_back_invalid(self, tmp_path):
        phase, result = self._round(tmp_path)
        task = result["batches"][0]
        # The excerpt alone does not hold the answer — an excerpt-only judge
        # would call the claim valid.
        excerpt = task["prompt"].split("## Suggestion groups")[0]
        assert "Retention is 30 days" not in excerpt
        (phase / task["output_name"]).write_text(
            json.dumps(_stub_validator(task)), encoding="utf-8"
        )
        merged = merge_batched_validation_results(
            output_dir=str(phase), batch_metadata=result, total_groups=1
        )
        assert merged[0]["status"] == "invalid"

    def test_c_an_unreadable_snapshot_is_counted_as_insufficient(self, tmp_path):
        phase, result = self._round(tmp_path)
        task = result["batches"][0]
        (phase / task["output_name"]).write_text(
            json.dumps(_stub_validator(task, can_read=False)), encoding="utf-8"
        )
        grouped = groups(1, title="The plan never defines retention")
        validated = rob.merge_validation_batches(
            str(phase), result, grouped, str(phase / "validation.json")
        )
        assert validated[0]["validation_status"] == "needs-human-decision"
        context = status_file.read_status(str(phase))["validation_context"]
        assert context["insufficient"] == 1
        assert context["sources"][0]["mode"] == "path"


# ---------------------------------------------------------------------------
# 2.3 (schema) and 2.4 (the adapter invariant)
# ---------------------------------------------------------------------------

class TestSchemaAndInvariant:
    def test_validation_accepts_workspace_read_and_consolidation_does_not(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc())
        phase = tmp_path / "review-plan"
        task = prepare(phase, plan, bind_document_file(str(plan), str(phase)))["batches"][0]
        assert validate_subagent_task(task) == []
        consolidation = dict(task, kind="consolidation", task_id="consolidation-batch-0")
        consolidation.pop("prompt_path", None)
        consolidation.pop("prompt_lines", None)
        assert validate_subagent_task(consolidation)

    def test_every_agent_readonly_adapter_declares_workspace_reads(self):
        """2.4: no executor runner that can run a validation task lacks a read tool."""
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import PROFILE_AGENT_READONLY

        readonly = {
            name: provider for name, provider in _PROVIDERS.items()
            if PROFILE_AGENT_READONLY in provider.agent_profiles
        }
        assert readonly, "no adapter declares agent-readonly"
        for name, provider in readonly.items():
            assert provider.supports_workspace_reads, (
                f"{name} declares agent-readonly without supports_workspace_reads"
            )


# ---------------------------------------------------------------------------
# 2.2 / §2.2: reading outside cwd (--add-dir)
# ---------------------------------------------------------------------------

class TestReadScopeOutsideCwd:
    def _task_outside(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        plan = write(tmp_path / "elsewhere" / "plan.md", big_doc())
        phase = tmp_path / "elsewhere" / "plan" / "review-plan"
        result = prepare(phase, plan, bind_document_file(str(plan), str(phase)))
        return repo, phase, result["batches"][0]

    def test_a_snapshot_outside_cwd_needs_its_directory(self, tmp_path):
        repo, phase, task = self._task_outside(tmp_path)
        assert executor._context_read_dirs(task, str(repo)) == [os.path.realpath(phase)]
        assert executor._context_read_dirs(task, str(phase.parent)) == []

    def test_claude_code_gets_add_dir_for_that_directory(self, tmp_path, monkeypatch):
        repo, phase, task = self._task_outside(tmp_path)
        seen = {}

        class Stop(Exception):
            pass

        def fake_spawn(**kwargs):
            seen.update(kwargs)
            raise Stop()

        monkeypatch.setattr(executor, "_spawn", fake_spawn)
        with pytest.raises(Stop):
            executor.dispatch_task(
                task, phase_dir=str(phase), runner="claude-code:sonnet",
                cwd=str(repo), retries=0, timeout=3600,
            )
        assert seen["read_dirs"] == [os.path.realpath(phase)]
        from utils.provider_registry import get_provider

        claude = get_provider("claude-code")
        argv = claude.widen_read_scope(
            claude.build_agent_command("p", "sonnet"), seen["read_dirs"]
        )
        assert argv[1:3] == ["--add-dir", os.path.realpath(phase)]
        assert argv[-1] == "p"

    def test_codex_reads_anywhere_and_needs_no_flag(self, tmp_path):
        from utils.provider_registry import get_provider

        assert get_provider("codex").reads_outside_cwd is True

    def test_an_adapter_without_the_capability_is_refused_by_name(
        self, tmp_path, monkeypatch
    ):
        repo, phase, task = self._task_outside(tmp_path)
        blind = SimpleNamespace(
            supports_workspace_reads=True, reads_outside_cwd=False,
            read_scope_flag=None, supports_profile=lambda profile: True,
        )
        monkeypatch.setattr(executor, "parse_model_spec", lambda spec, anchor=None: ("blind", "m"))
        monkeypatch.setattr(executor, "get_provider", lambda name: blind)
        outcome = executor.dispatch_task(
            task, phase_dir=str(phase), runner="blind:m", cwd=str(repo),
        )
        assert "outside the runner's working directory" in outcome.reason
        assert "rematerialize with a capable runner" in outcome.reason


# ---------------------------------------------------------------------------
# 2.5 (g): the stderr lines
# ---------------------------------------------------------------------------

class TestStderrLines:
    def test_a_path_source_prints_exactly_one_excerpted_line(self, tmp_path, capsys):
        plan = write(tmp_path / "plan.md", big_doc())
        phase = tmp_path / "review-plan"
        prepare(phase, plan, bind_document_file(str(plan), str(phase)))
        err = capsys.readouterr().err
        lines = [ln for ln in err.splitlines() if "[VALIDATION_CONTEXT_" in ln]
        assert len(lines) == 1
        assert lines[0].startswith("[VALIDATION_CONTEXT_EXCERPTED] plan ")
        assert lines[0].endswith(f"full plan at {phase / PLAN_SNAPSHOT_NAME}")
        assert "[VALIDATION_CONTEXT_TRUNCATED]" not in err

    def test_a_truncated_source_prints_the_old_line_and_its_reason(self, tmp_path, capsys):
        prepare_batched_validation_tasks(
            groups=groups(), context=big_doc(), output_dir=str(tmp_path / "p"),
            plan_file="",
        )
        err = capsys.readouterr().err
        assert "[VALIDATION_CONTEXT_TRUNCATED] context" in err
        assert "(reason: no readable file is bound to this block)" in err
        assert "[VALIDATION_CONTEXT_EXCERPTED]" not in err


# ---------------------------------------------------------------------------
# 2.6: the secondary block
# ---------------------------------------------------------------------------

def _review_tasks_round(tmp_path, plan_text, tasks_text, *, tasks_readable=True):
    plan = write(tmp_path / "plan.md", plan_text)
    tasks = write(tmp_path / "tasks" / "tasks.md", tasks_text)
    phase = tmp_path / "plan" / "review-tasks"
    plan_doc = bind_document_file(str(plan), str(phase))
    kwargs = rob.validation_context_kwargs(
        "review-tasks", plan_doc.text, {"tasks_file": str(tasks)},
        plan_document=plan_doc, phase_dir=str(phase),
    )
    if not tasks_readable:
        doc = kwargs["secondary_document"]
        doc.path, doc.in_phase_dir = None, False
        doc.reason = "snapshot could not be written"
    result = prepare_batched_validation_tasks(
        groups=groups(), output_dir=str(phase), plan_file=str(plan),
        tasks_file=str(tasks), phase="review-tasks", **kwargs,
    )
    return phase, plan, tasks, result


class TestSecondaryBlock:
    def test_a_only_the_plan_over(self, tmp_path):
        phase, _plan, _tasks, result = _review_tasks_round(
            tmp_path, big_doc(), "# Tasks\n\n## T001\nsmall\n"
        )
        task = result["batches"][0]
        assert [s["mode"] for s in task["context_sources"]] == ["path", "inline"]
        assert "## Plan file" in task["prompt"] and "## Tasks file" not in task["prompt"]
        context = status_file.read_status(str(phase))["validation_context"]
        assert [(s["label"], s["mode"]) for s in context["sources"]] == [
            ("plan", "path"), ("tasks", "inline")]

    def test_b_only_the_tasks_file_over(self, tmp_path):
        phase, _plan, _tasks, result = _review_tasks_round(
            tmp_path, "# Plan\n\nshort\n", big_doc("Tasks")
        )
        task = result["batches"][0]
        assert [s["mode"] for s in task["context_sources"]] == ["inline", "path"]
        assert "## Tasks file" in task["prompt"] and "## Plan file" not in task["prompt"]
        # The tasks file has no tasks-backup.md here, so it got its own snapshot.
        assert snapshot_path_in(task["prompt"]) == str(phase / TASKS_SNAPSHOT_NAME)
        assert "tasks file truncated at" in task["prompt"]

    def test_b_a_matching_tasks_backup_is_reused(self, tmp_path):
        tasks_text = big_doc("Tasks")
        write(tmp_path / "plan" / "review-tasks" / TASKS_BACKUP_NAME, tasks_text)
        phase, _plan, _tasks, result = _review_tasks_round(
            tmp_path, "# Plan\n\nshort\n", tasks_text
        )
        assert snapshot_path_in(result["batches"][0]["prompt"]) == str(
            phase / TASKS_BACKUP_NAME
        )
        assert not (phase / TASKS_SNAPSHOT_NAME).exists()

    def test_c_both_over_two_paths(self, tmp_path):
        phase, _plan, _tasks, result = _review_tasks_round(
            tmp_path, big_doc(), big_doc("Tasks")
        )
        task = result["batches"][0]
        assert [s["mode"] for s in task["context_sources"]] == ["path", "path"]
        assert str(phase / PLAN_SNAPSHOT_NAME) in task["prompt"]
        assert str(phase / TASKS_SNAPSHOT_NAME) in task["prompt"]
        assert task["context_mode"] == "workspace-read"

    def test_d_both_over_one_readable(self, tmp_path):
        phase, _plan, _tasks, result = _review_tasks_round(
            tmp_path, big_doc(), big_doc("Tasks"), tasks_readable=False
        )
        task = result["batches"][0]
        assert [s["mode"] for s in task["context_sources"]] == ["path", "truncated"]
        context = status_file.read_status(str(phase))["validation_context"]
        assert context["mode"] == "truncated"
        from utils.validation_context import validation_context_notice

        notice = validation_context_notice(context)
        assert notice.startswith("Validation saw ") and "of the tasks file" in notice
        assert "of the plan" not in notice

    def test_e_review_tasks_names_its_own_snapshot_after_a_live_edit(self, tmp_path):
        phase, plan, _tasks, result = _review_tasks_round(
            tmp_path, big_doc("v1"), "# Tasks\n\nsmall\n"
        )
        write(plan, big_doc("v2"))
        task = result["batches"][0]
        snapshot = phase / PLAN_SNAPSHOT_NAME
        assert snapshot_path_in(task["prompt"]) == str(snapshot)
        assert content_hash(snapshot.read_text(encoding="utf-8")) == (
            result["inputs"]["plan_sha256"]
        )
        assert str(plan) not in task["prompt"]

    def test_e_review_code_names_its_own_snapshot_after_a_live_edit(self, tmp_path):
        plan = write(tmp_path / "plan.md", big_doc("v1"))
        phase = tmp_path / "plan" / "code-review"
        document = bind_document_file(str(plan), str(phase))
        write(plan, big_doc("v2"))
        result = prepare_batched_validation_tasks(
            groups=groups(), output_dir=str(phase), plan_file=str(plan),
            # What both review-code materialization sites pass: the bound
            # text alone, no `Plan file:` line naming the live plan.
            context=document.text,
            secondary_context="#### Implementation record\n\nnone",
            secondary_context_label="Reviewed code",
            reviewed_code="", context_document=document,
            secondary_document=bind_document_text(
                "#### Implementation record\n\nnone", label="reviewed-code"
            ),
        )
        task = result["batches"][0]
        snapshot = phase / PLAN_SNAPSHOT_NAME
        assert snapshot_path_in(task["prompt"]) == str(snapshot)
        assert content_hash(snapshot.read_text(encoding="utf-8")) == (
            result["inputs"]["plan_sha256"]
        )
        assert "# v2" not in task["prompt"]
        assert str(plan) not in task["prompt"]

    def test_f_review_code_dropped_windows_are_a_truncated_source(self, tmp_path):
        from code_review_orchestrator import reviewed_code_document

        plan = write(tmp_path / "plan.md", "# Plan\n\nshort\n")
        phase = tmp_path / "plan" / "code-review"
        secondary = "#### Code at the cited lines\n\n### a.py\n1: x\n"
        document = reviewed_code_document(secondary, {
            "record_cut": False, "dropped_files": ["b.py", "c.py"], "files_total": 3,
        })
        result = prepare_batched_validation_tasks(
            groups=groups(), context=plan.read_text(), output_dir=str(phase),
            plan_file=str(plan), secondary_context=secondary,
            secondary_context_label="Reviewed code", reviewed_code=secondary,
            context_document=bind_document_file(str(plan), str(phase)),
            secondary_document=document,
        )
        task = result["batches"][0]
        assert task["context_mode"] == "inline"
        assert "b.py" not in task["prompt"]  # no code path is named
        source = task["context_sources"][1]
        assert source["label"] == "reviewed-code" and source["mode"] == "truncated"
        context = status_file.read_status(str(phase))["validation_context"]
        from utils.validation_context import validation_context_notice

        assert validation_context_notice(context) == (
            "Validation saw 1 of 3 cited files of the reviewed code "
            "(dropped: b.py, c.py)"
        )

    def test_f_review_code_record_only_cut_names_the_cut(self, tmp_path):
        from code_review_orchestrator import (
            IMPLEMENTATION_RECORD_CHARS, reviewed_code_document,
        )
        from utils.validation_context import source_phrase, validation_context_notice

        plan = write(tmp_path / "plan.md", "# Plan\n\nshort\n")
        phase = tmp_path / "plan" / "code-review"
        secondary = "#### Implementation record\n\ncut record\n"
        raw = IMPLEMENTATION_RECORD_CHARS + 5000
        document = reviewed_code_document(secondary, {
            "record_cut": True, "record_chars": raw,
            "dropped_files": [], "files_total": 2,
        })
        prepare_batched_validation_tasks(
            groups=groups(), context=plan.read_text(), output_dir=str(phase),
            plan_file=str(plan), secondary_context=secondary,
            secondary_context_label="Reviewed code", reviewed_code="",
            context_document=bind_document_file(str(plan), str(phase)),
            secondary_document=document,
        )
        context = status_file.read_status(str(phase))["validation_context"]
        cut = f"implementation record cut at {IMPLEMENTATION_RECORD_CHARS} of {raw} chars"
        notice = validation_context_notice(context)
        assert notice == (
            f"Validation saw {len(secondary)} chars of the reviewed code ({cut})"
        )
        entry = next(e for e in context["sources"] if e["label"] == "reviewed-code")
        assert f"{len(secondary)} of {len(secondary)} chars" not in source_phrase(entry)
        assert cut in source_phrase(entry)


# ---------------------------------------------------------------------------
# 2.5 (d)/(f), 2.9, 2.10 through the real reaggregate
# ---------------------------------------------------------------------------

def _args(**over):
    base = dict(skip_validation=False, validation_model="auto", timeout=None,
                internal_validation=False, force=False, consolidation_threshold=None)
    base.update(over)
    return argparse.Namespace(**base)


def _distinct(names):
    return [
        {"title": n, "desc": f"{n}{n} {n}{n}", "importance": "MEDIUM",
         "reference": f"Step {n}", "type": f"kind-{n.lower()}"}
        for n in names
    ]


def _phase(tmp_path, plan_text):
    plan = write(tmp_path / "plan.md", plan_text)
    out_dir = tmp_path / "out"
    phase_dir = out_dir / "review-plan"
    phase_dir.mkdir(parents=True)
    (phase_dir / "claude-code_sonnet.json").write_text(
        json.dumps(_distinct(["Alpha", "Bravo", "Charlie", "Delta", "Echo"])),
        encoding="utf-8",
    )
    status_file.write_status(str(phase_dir), {
        "phase": "review-plan", "state": "models_complete",
        "models_requested": ["claude-code:sonnet"],
        "models_completed": ["claude-code:sonnet"], "models_failed": {},
    })
    return plan, out_dir, phase_dir


def _reaggregate(plan, out_dir, phase_dir, **args):
    return asyncio.run(rob.reaggregate_from_existing_files(
        prefix="plan", out_dir=str(out_dir), phase_dir=str(phase_dir),
        phase_name="review-plan", plan_path=str(plan), args=_args(**args),
        orchestrator_script="review_plan_orchestrator.py",
    ))


def _tasks(phase_dir):
    return json.loads((phase_dir / "validation_tasks.json").read_text(encoding="utf-8"))


def _complete(phase_dir, status="valid"):
    for batch in _tasks(phase_dir)["batches"]:
        verdicts = [
            {"group_index": gi, "group_hash": batch["group_ids"][li],
             "status": status, "reason": "ok", "confidence": 0.9}
            for li, gi in enumerate(batch["group_indices"])
        ]
        (phase_dir / batch["output_name"]).write_text(
            json.dumps({"groups": verdicts}), encoding="utf-8"
        )


class TestThroughTheReaggregate:
    def test_f_reaggregate_republishes_and_the_terminal_status_keeps_the_key(
        self, tmp_path
    ):
        plan, out_dir, phase_dir = _phase(tmp_path, big_doc())
        _reaggregate(plan, out_dir, phase_dir)
        tasks = _tasks(phase_dir)
        assert tasks["context_contract"] == CONTEXT_CONTRACT
        assert {t["context_mode"] for t in tasks["batches"]} == {"workspace-read"}
        assert status_file.read_status(str(phase_dir))["validation_context"]["mode"] == "path"
        # Wipe the key, then reaggregate the completed round: the merge
        # re-publishes it from the round's recorded sources — the live plan is
        # gone, so it cannot have come from there.
        status_file.update_status(str(phase_dir), {"validation_context": None})
        _complete(phase_dir)
        plan.unlink()
        write(plan, big_doc())  # same bytes again, so the inputs guard binds
        _reaggregate(plan, out_dir, phase_dir)
        final = status_file.read_status(str(phase_dir))
        assert final["state"] == "complete"
        assert final["validation_context"]["sources"][0]["path"] == str(
            phase_dir / PLAN_SNAPSHOT_NAME
        )
        report = (out_dir / "review-plan" / "report.md")
        if report.exists():
            assert "Validation context" in report.read_text(encoding="utf-8")
        validation = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert validation["metadata"]["context_contract"] == CONTEXT_CONTRACT

    def test_d_a_rerun_that_fits_inline_clears_the_key(self, tmp_path):
        plan, out_dir, phase_dir = _phase(tmp_path, big_doc())
        _reaggregate(plan, out_dir, phase_dir)
        assert status_file.read_status(str(phase_dir))["validation_context"]
        # A fresh round of the same phase whose plan now fits inline.
        rob.retire_stale_validation_round(str(phase_dir), reason="test rerun")
        write(plan, "# Plan\n\nsmall now\n")
        _reaggregate(plan, out_dir, phase_dir)
        # Deleted, not written null: a raw reader of .status.json sees no key.
        assert "validation_context" not in status_file.read_status(str(phase_dir))

    def test_2_9_a_copied_phase_dir_rebinds_to_its_own_snapshot(self, tmp_path, capsys):
        plan, out_dir, phase_dir = _phase(tmp_path, big_doc())
        _reaggregate(plan, out_dir, phase_dir)
        original = str(phase_dir / PLAN_SNAPSHOT_NAME)
        copy = tmp_path / "copy" / "review-plan"
        shutil.copytree(phase_dir, copy)
        # (a) original removed: prompts rebound to the copy, hash verified.
        shutil.rmtree(phase_dir)
        _reaggregate(plan, tmp_path / "copy", copy)
        tasks = _tasks(copy)
        for task in tasks["batches"]:
            assert original not in task["prompt"]
            assert str(copy / PLAN_SNAPSHOT_NAME) in task["prompt"]
            sidecar = Path(task["prompt_path"])
            assert sidecar.parent == copy
            assert sidecar.read_text(encoding="utf-8") == task["prompt"]
        assert "[VALIDATION_CONTEXT_REBOUND]" in capsys.readouterr().err

    def test_2_9_c_with_the_original_intact_the_copy_never_names_it(self, tmp_path):
        plan, out_dir, phase_dir = _phase(tmp_path, big_doc())
        _reaggregate(plan, out_dir, phase_dir)
        copy = tmp_path / "copy" / "review-plan"
        shutil.copytree(phase_dir, copy)
        _reaggregate(plan, tmp_path / "copy", copy)
        for task in _tasks(copy)["batches"]:
            assert str(phase_dir) + os.sep not in task["prompt"]

    def test_2_9_b_an_altered_snapshot_in_the_copy_is_refused(self, tmp_path, capsys):
        plan, out_dir, phase_dir = _phase(tmp_path, big_doc())
        _reaggregate(plan, out_dir, phase_dir)
        copy = tmp_path / "copy" / "review-plan"
        shutil.copytree(phase_dir, copy)
        write(copy / PLAN_SNAPSHOT_NAME, big_doc("edited"))
        with pytest.raises(SystemExit) as exit_info:
            _reaggregate(plan, tmp_path / "copy", copy)
        assert exit_info.value.code == 3
        assert SNAPSHOT_ERROR in capsys.readouterr().err

    def test_2_10_a_contract_1_truncated_round_is_retired_and_rematerialized(
        self, tmp_path, capsys
    ):
        plan, out_dir, phase_dir = _phase(tmp_path, big_doc())
        _reaggregate(plan, out_dir, phase_dir)
        # Forge what the 0.1.0 plugin left: no contract stamp, the bare cut in
        # every stored prompt, completed batches and a merged validation.json.
        tasks = _tasks(phase_dir)
        tasks.pop("context_contract")
        for task in tasks["batches"]:
            task.pop("context_sources", None)
            task["context_mode"] = "inline"
            task["prompt"] = task["prompt"].split("\n\n[... plan truncated")[0] + (
                f"\n\n{LEGACY_TRUNCATION_MARKER}\n"
            )
        (phase_dir / "validation_tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        _complete(phase_dir)
        (phase_dir / "validation.json").write_text(json.dumps({
            "groups": [], "metadata": {"model": "old", "schema_version": "2.1"},
        }), encoding="utf-8")
        capsys.readouterr()

        _reaggregate(plan, out_dir, phase_dir)

        out = capsys.readouterr().out
        assert "validation built from truncated context (contract 1); rematerializing" in out
        # The originals are kept, renamed.
        stale = {p.name for p in phase_dir.glob("*.stale.json")}
        assert any(n.startswith("validation_tasks.") for n in stale)
        assert any(n.startswith("validation.") for n in stale)
        assert any(n.startswith("validation_batch_0.") for n in stale)
        # And the round is rematerialized in the path form, awaiting validation.
        fresh = _tasks(phase_dir)
        assert fresh["context_contract"] == CONTEXT_CONTRACT
        assert {t["context_mode"] for t in fresh["batches"]} == {"workspace-read"}
        assert not (phase_dir / "validation.json").exists()
        assert not list(phase_dir.glob("validation_batch_*[0-9].json"))

    def test_7_1d_a_legacy_validation_json_is_retired_by_the_stale_check(self, tmp_path):
        from utils.validation import stale_validation_reason

        plan, out_dir, phase_dir = _phase(tmp_path, big_doc())
        _reaggregate(plan, out_dir, phase_dir)
        tasks = _tasks(phase_dir)
        tasks.pop("context_contract")
        tasks["batches"][0]["prompt"] += f"\n{LEGACY_TRUNCATION_MARKER}\n"
        (phase_dir / "validation_tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        (phase_dir / "validation.json").write_text(json.dumps({
            "groups": [], "metadata": {"model": "old", "schema_version": "2.1",
                                       "batch_sha256": {}},
        }), encoding="utf-8")
        reason = stale_validation_reason(
            phase_dir / "validation.json", phase_dir / "grouped.json"
        )
        assert reason and "contract 1" in reason

    def test_2_10_an_under_limit_contract_1_round_is_left_alone(self, tmp_path):
        from utils.validation import legacy_context_round_reason

        plan, out_dir, phase_dir = _phase(tmp_path, "# Plan\n\nsmall\n")
        _reaggregate(plan, out_dir, phase_dir)
        tasks = _tasks(phase_dir)
        tasks.pop("context_contract")
        (phase_dir / "validation_tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        assert legacy_context_round_reason(str(phase_dir)) is None
