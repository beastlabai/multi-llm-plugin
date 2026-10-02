"""``--reaggregate`` reads one file per logical model across the L1 anchor boundary (W3).

A phase dir that straddles the L1 anchor upgrade holds both ``<model>.json`` and
``<provider>_<model>.json``. L1 made ``select_models()`` return canonical
``provider:model`` specs and those specs feed ``sanitize_model_name``, so a run
started before L1 wrote ``sonnet.json``, and the resume after L1 looked for
``claude-code_sonnet.json``, did not find it, and **re-ran that model**. The dir
therefore holds two answers from one model, the anchored one newer than the legacy
one — and ``reaggregate_from_existing_files`` globs ``*.json``, so it used to count
them as two models and extend ``all_suggestions`` with both lists.

That is not cosmetic. ``source_model`` is what grouping attributes by, and
``SuggestionGroup._recalculate_priority`` scores ``len(self.models) + Σ importance``,
so one MEDIUM finding read twice scores 6 instead of 3 and the group renders as two
models agreeing when one model said it once — corroboration that did not happen.

Identity here is the **recorded canonical spec** in ``.status.json``, never the
filename: sanitization is lossy one way only, so a legacy stem cannot be
re-resolved into a spec, but it can be matched against the stems the recorded specs
generate. The precedence, one case each below: the anchored file wins
unconditionally and the legacy one is skipped by path; divergent payloads WARN
naming both paths and both counts and still aggregate (``--reaggregate`` is the
salvage path and must not be stricter than the run it salvages); and a legacy stem
matching two or more recorded specs is a genuine ambiguity that fails closed on
exit 3. The fourth case pins the fallback — no ``.status.json`` at all, where the
stem *is* the identity — which the fix must not disturb.

The ``_phase``/``_args``/``_reaggregate`` helpers below follow
``tests/test_task_file_resume_guard.py``, the only other direct caller of the base
helper against a real phase dir. ``.status.json`` is written through
``write_status`` so the fixture cannot drift from the checkpoint's real shape.
"""

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
from utils import progress_tracker  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import run_status  # noqa: E402
from utils import subagent_tasks as subagent_tasks_mod  # noqa: E402
from utils.status_file import write_status  # noqa: E402


def _suggestions(titles):
    return [
        {
            "title": title,
            "desc": f"Description for {title}",
            "importance": "MEDIUM",
            "reference": "Step 1",
            "type": "addition",
        }
        for title in titles
    ]


def _phase(tmp_path, files, models_requested=None):
    """A phase dir holding *files* (``stem -> suggestions``), ready to reaggregate.

    ``models_requested`` is written through ``write_status`` exactly as a real run
    writes its checkpoint; passing ``None`` writes no ``.status.json`` at all,
    which is the hand-copied / pre-checkpoint dir.
    """
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    phase_dir = out_dir / "review-plan"
    phase_dir.mkdir(parents=True)

    for stem, suggestions in files.items():
        (phase_dir / f"{stem}.json").write_text(
            json.dumps(suggestions), encoding="utf-8"
        )

    if models_requested is not None:
        write_status(str(phase_dir), {
            "phase": "review-plan",
            "state": "models_complete",
            "started_at": "2026-01-01T00:00:00",
            "models_requested": list(models_requested),
            "models_completed": list(models_requested),
            "models_failed": {},
            "error": None,
            "plan_hash": "deadbeefcafe",
        })

    return plan, out_dir, phase_dir


def _args(**over):
    base = dict(
        skip_validation=False,
        validation_model="auto",
        timeout=None,
        internal_validation=False,
        force=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _reaggregate(plan, out_dir, phase_dir):
    return asyncio.run(
        rob.reaggregate_from_existing_files(
            prefix="plan",
            out_dir=str(out_dir),
            phase_dir=str(phase_dir),
            phase_name="review-plan",
            plan_path=str(plan),
            args=_args(),
            orchestrator_script="review_plan_orchestrator.py",
        )
    )


def _aggregated_suggestions(phase_dir):
    """Every suggestion that reached grouping, flattened out of ``grouped.json``."""
    payload = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
    groups = payload["groups"] if isinstance(payload, dict) else payload
    return [s for group in groups for s in group.get("suggestions", [])]


class TestReaggregatePromptSidecars:
    def test_copied_phase_persists_rebound_pointers(self, tmp_path):
        plan, out_dir, phase_dir = _phase(
            tmp_path, {"sonnet": _suggestions(["Finding"])},
        )
        _reaggregate(plan, out_dir, phase_dir)
        copied_out = tmp_path / "copy"
        shutil.copytree(out_dir, copied_out)
        copied_phase = copied_out / "review-plan"
        original_tasks = (phase_dir / "validation_tasks.json").read_bytes()
        for sidecar in phase_dir.glob("*.prompt.txt"):
            sidecar.write_text("different round", encoding="utf-8")

        _reaggregate(plan, copied_out, copied_phase)

        tasks = json.loads((copied_phase / "validation_tasks.json").read_text())
        for batch in tasks["batches"]:
            assert Path(batch["prompt_path"]).parent == copied_phase
            assert Path(batch["prompt_path"]).read_bytes() == batch["prompt"].encode("utf-8")
        assert (phase_dir / "validation_tasks.json").read_bytes() == original_tasks
        assert all(p.read_text() == "different round" for p in phase_dir.glob("*.prompt.txt"))

    def test_failed_repair_persists_inline_fallback(self, tmp_path, monkeypatch):
        from utils import validation as validation_mod

        plan, out_dir, phase_dir = _phase(
            tmp_path, {"sonnet": _suggestions(["Finding"])},
        )
        _reaggregate(plan, out_dir, phase_dir)
        tasks_path = phase_dir / "validation_tasks.json"
        before = json.loads(tasks_path.read_text())
        for sidecar in phase_dir.glob("*.prompt.txt"):
            sidecar.write_text("incorrect prompt", encoding="utf-8")

        def cannot_write(*args):
            raise PermissionError("sidecar is locked")

        monkeypatch.setattr(validation_mod, "_write_one_sidecar", cannot_write)
        _reaggregate(plan, out_dir, phase_dir)

        after = subagent_tasks_mod.read_task_file(str(tasks_path))
        for original, repaired in zip(before["batches"], after["batches"]):
            assert "prompt_path" not in repaired
            assert "prompt_lines" not in repaired
            assert repaired["prompt"] == original["prompt"]
        assert all(p.read_text() == "incorrect prompt" for p in phase_dir.glob("*.prompt.txt"))

    def test_corrupt_sidecar_does_not_block_completed_batches(self, tmp_path):
        plan, out_dir, phase_dir = _phase(
            tmp_path, {"sonnet": _suggestions(["Finding"])},
        )
        _reaggregate(plan, out_dir, phase_dir)
        tasks = _complete_validation_batches(phase_dir)
        batch = tasks["batches"][0]
        Path(batch["prompt_path"]).write_bytes(b"\xff")

        _reaggregate(plan, out_dir, phase_dir)

        assert Path(batch["prompt_path"]).read_bytes() == batch["prompt"].encode("utf-8")
        validation = json.loads((phase_dir / "validation.json").read_text())
        assert all(group["status"] == "valid" for group in validation["groups"])

    def test_metadata_write_failure_preserves_the_task_file_and_stops_handoff(
        self, tmp_path, monkeypatch, capsys
    ):
        from utils import subagent_executor

        plan, out_dir, phase_dir = _phase(
            tmp_path, {"sonnet": _suggestions(["Finding"])},
        )
        _reaggregate(plan, out_dir, phase_dir)
        tasks_path = phase_dir / "validation_tasks.json"
        tasks = json.loads(tasks_path.read_text())
        tasks["batches"][0]["prompt_path"] = str(tmp_path / "old.prompt.txt")
        tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
        before = tasks_path.read_bytes()

        def cannot_write(*args):
            raise PermissionError("task file is locked")

        monkeypatch.setattr(subagent_executor, "_atomic_write_json", cannot_write)
        with patch.object(rob, "dispatch_subagent_tasks") as dispatch:
            with pytest.raises(SystemExit) as exc:
                _reaggregate(plan, out_dir, phase_dir)
        assert exc.value.code == 3
        dispatch.assert_not_called()
        assert tasks_path.read_bytes() == before
        assert "Could not persist repaired prompt metadata" in capsys.readouterr().err


class TestOneFilePerLogicalModel:
    def test_identical_duplicate_is_read_once_from_the_anchored_file(
        self, tmp_path, capsys
    ):
        """The straddling dir's two spellings are one model, read from one file.

        This is the case that fails on today's double count: 6 suggestions from 2
        models, with `sonnet` and `claude-code:sonnet` both in `models_found`.
        """
        payload = _suggestions(["Finding 0", "Finding 1", "Finding 2"])
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"sonnet": payload, "claude-code_sonnet": payload},
            models_requested=["claude-code:sonnet"],
        )

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "Re-aggregating 3 suggestions from 1 models" in out

        aggregated = _aggregated_suggestions(phase_dir)
        assert len(aggregated) == 3
        assert {s["source_model"] for s in aggregated} == {"claude-code:sonnet"}
        assert sorted(s["title"] for s in aggregated) == [
            "Finding 0", "Finding 1", "Finding 2",
        ]

        # Every group is one model agreeing with itself once, never two.
        payload_on_disk = json.loads(
            (phase_dir / "grouped.json").read_text(encoding="utf-8")
        )
        groups = payload_on_disk["groups"]
        for group in groups:
            assert group["models"] == ["claude-code:sonnet"]

        # The skip line names the legacy file it skipped, and the anchored file it
        # read instead — the legacy one is skipped, not merged.
        skip_lines = [ln for ln in out.splitlines()
                      if "Skipping pre-anchor duplicate" in ln]
        assert len(skip_lines) == 1
        assert str(phase_dir / "sonnet.json") in skip_lines[0]
        assert str(phase_dir / "claude-code_sonnet.json") in skip_lines[0]
        # Identical payloads are not a divergence.
        assert "WARN" not in out

    def test_divergent_stale_and_current_warns_and_uses_the_anchored_file(
        self, tmp_path, capsys
    ):
        """A stale pre-anchor answer never strands the current one — it is logged."""
        stale = _suggestions(["Stale A", "Stale B"])
        current = _suggestions(["Current A", "Current B", "Current C"])
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"sonnet": stale, "claude-code_sonnet": current},
            models_requested=["claude-code:sonnet"],
        )

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "Re-aggregating 3 suggestions from 1 models" in out

        aggregated = _aggregated_suggestions(phase_dir)
        assert sorted(s["title"] for s in aggregated) == [
            "Current A", "Current B", "Current C",
        ]
        assert {s["source_model"] for s in aggregated} == {"claude-code:sonnet"}

        warn_lines = [ln for ln in out.splitlines() if "WARN" in ln]
        assert len(warn_lines) == 1
        warn = warn_lines[0]
        # Both paths and both counts, so the divergence is inspectable.
        assert str(phase_dir / "sonnet.json") in warn
        assert str(phase_dir / "claude-code_sonnet.json") in warn
        assert "2 suggestions" in warn and "3 suggestions" in warn

    def test_ambiguous_legacy_stem_exits_3_naming_every_candidate_spec(
        self, tmp_path, capsys
    ):
        """`openai:gpt-5.2` and `openrouter:gpt-5.2` share the legacy stem `gpt-5_2`.

        There is no evidence on disk for which one wrote it: attributing it to
        either fabricates a `source_model`, and splitting it across both is the
        double count again. Exit 3 is the same needs-human code the resume reader
        already uses on this path.
        """
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {
                "gpt-5_2": _suggestions(["Legacy finding"]),
                "openai_gpt-5_2": _suggestions(["OpenAI finding"]),
                "openrouter_gpt-5_2": _suggestions(["OpenRouter finding"]),
            },
            models_requested=["openai:gpt-5.2", "openrouter:gpt-5.2"],
        )

        with pytest.raises(SystemExit) as excinfo:
            _reaggregate(plan, out_dir, phase_dir)
        assert excinfo.value.code == 3

        err = capsys.readouterr().err
        assert str(phase_dir / "gpt-5_2.json") in err
        assert "openai:gpt-5.2" in err
        assert "openrouter:gpt-5.2" in err
        # The repair is named, not left to the reader.
        assert "rename" in err.lower() or "remove" in err.lower()
        # Nothing was aggregated behind the refusal.
        assert not (phase_dir / "grouped.json").exists()

    def test_a_dir_with_no_checkpoint_still_reaggregates_by_stem(
        self, tmp_path, capsys
    ):
        """The fallback the fix must not disturb: no `.status.json`, stem is identity."""
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {
                "model-a": _suggestions(["A0", "A1", "A2"]),
                "model-b": _suggestions(["B0", "B1", "B2"]),
            },
            models_requested=None,
        )
        assert not (phase_dir / ".status.json").exists()

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "Re-aggregating 6 suggestions from 2 models" in out

        aggregated = _aggregated_suggestions(phase_dir)
        assert len(aggregated) == 6
        assert {s["source_model"] for s in aggregated} == {"model-a", "model-b"}
        assert "Skipping pre-anchor duplicate" not in out
        assert "WARN" not in out


class TestBatchedValidationAnnouncementIsHostNeutral:
    """F9: this handover used to hardcode "Claude Code subagent" even though
    the run driving it (opencode, codex, ...) is named correctly a few lines
    later in the same log — measured on an opencode-driven run whose log
    printed this beside 11 lines correctly naming opencode."""

    def test_the_batched_validation_prints_name_no_host(self, tmp_path, capsys):
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"model-a": _suggestions(["Finding 0"])},
            models_requested=["model-a"],
        )

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "Claude Code subagent" not in out
        assert "Preparing batched validation for subagent execution" in out
        assert "Validation prepared for subagent execution" in out


class TestMergedValidationIsAnnouncedOnce:
    """F10: ``save_validation_results`` already prints
    "[validation] Saved validation results to: ..." for every write it makes;
    ``merge_validation_batches`` used to print a second line
    ("Merged validation saved to: ...") for the very same file."""

    def test_merge_validation_batches_prints_the_write_exactly_once(
        self, tmp_path, capsys
    ):
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        (phase_dir / "validation_batch_0.json").write_text(
            json.dumps({
                "groups": [
                    {"group_index": 0, "status": "valid", "reason": "ok",
                     "confidence": 0.9},
                ],
                "metadata": {"model": "model-a"},
            }),
            encoding="utf-8",
        )
        batch_metadata = {"batches": [{"batch_index": 0, "group_indices": [0]}]}
        grouped = [{"theme": "Theme", "suggestions": []}]
        validation_path = phase_dir / "validation.json"

        rob.merge_validation_batches(
            str(phase_dir), batch_metadata, grouped, str(validation_path)
        )
        out = capsys.readouterr().out

        assert validation_path.exists()
        assert out.count(str(validation_path)) == 1
        assert "[validation] Saved validation results to:" in out
        assert "Merged validation saved to:" not in out


class TestOneMalformedElementCannotStrandTheOtherModels:
    """`s["source_model"] = model` is an item assignment, so a result file
    holding a bare scalar used to raise `TypeError` here and lose every model's
    suggestions — the review-plan twin of the 2026-09-20 code-review failure,
    where the stdout fallback scraped `[0]` out of one reviewer's prose.
    """

    def test_the_other_models_suggestions_still_reach_the_grouping(
        self, tmp_path, capsys
    ):
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {
                "cursor-agent_kimi-k3-high": [0],
                "claude-code_sonnet": _distinct_suggestions(["Alpha", "Bravo"]),
            },
            models_requested=["cursor-agent:kimi-k3-high", "claude-code:sonnet"],
        )

        asyncio.run(
            rob.reaggregate_from_existing_files(
                prefix="plan",
                out_dir=str(out_dir),
                phase_dir=str(phase_dir),
                phase_name="review-plan",
                plan_path=str(plan),
                args=_args(skip_validation=True),
                orchestrator_script="review_plan_orchestrator.py",
            )
        )
        captured = capsys.readouterr()

        titles = sorted(s["title"] for s in _aggregated_suggestions(phase_dir))
        assert titles == ["Alpha", "Bravo"]
        assert "cursor-agent:kimi-k3-high" in captured.err
        assert "0 (int)" in captured.err


# ---------------------------------------------------------------------------
# The code-review copy
# ---------------------------------------------------------------------------
#
# `code_review_orchestrator.reaggregate_from_existing_files` is a genuinely
# separate implementation of the same salvage path, reached from
# `code_review_orchestrator.py:1445`, and W3's Done-when is not met for
# code-review mode while it globs and reads both spellings. It accumulates into
# `all_issues`, `models_found` AND a `results` dict that feeds
# `generate_review_report`, so the "exactly one file per logical model" rule has
# to hold for the report surface too: a duplicate `results` entry re-renders the
# same model twice even if the counts were deduped.
#
# The no-checkpoint fallback for this copy is already pinned by
# `tests/test_code_review_orchestrator.py`, which patches `build_unsanitize_map`
# to `{}` and still expects the model resolved from its stem.


def _cro_phase(tmp_path, files, models_requested=None):
    """A code-review phase dir holding *files* (``stem -> issues``)."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    phase_dir = tmp_path / "out" / "code-review"
    phase_dir.mkdir(parents=True)

    for stem, issues in files.items():
        (phase_dir / f"{stem}.json").write_text(json.dumps(issues), encoding="utf-8")

    if models_requested is not None:
        write_status(str(phase_dir), {
            "phase": "code-review",
            "state": "models_complete",
            "started_at": "2026-01-01T00:00:00",
            "models_requested": list(models_requested),
            "models_completed": list(models_requested),
            "models_failed": {},
            "error": None,
            "plan_hash": "deadbeefcafe",
        })

    return plan, phase_dir


def _issues(titles):
    return [
        {
            "title": title,
            "desc": f"Description for {title}",
            "importance": "medium",
            "file": "src/a.py",
            "type": "style",
        }
        for title in titles
    ]


def _run_cro_reaggregate(
    plan, phase_dir, *, skip_validation=True, render_report=False, extra_patches=()
):
    """Drive the code-review reaggregate; return the patched report mock.

    Everything patched here is git/state/report machinery the identity question
    does not touch — ``build_unsanitize_map`` is deliberately NOT patched, since
    the recorded checkpoint is the whole point.

    ``render_report=True`` leaves the *real* ``generate_review_report`` in place,
    which is the only way to assert on what ``report.md`` actually says about a
    model; it is off by default so the identity tests keep asserting on the
    ``results`` dict handed to the renderer rather than on rendered markdown.
    ``extra_patches`` takes already-built ``patch`` objects (a spy on
    ``check_task_file_inputs``, a ``group_similar_suggestions`` rigged to raise)
    so a caller can pin what was *called* and not only what was written.
    """
    args = argparse.Namespace(
        plan_file=str(plan),
        models=["claude-code:sonnet"],
        interactive=False,
        quick=False,
        timeout=None,
        max_parallel=2,
        base_ref=None,
        skip_validation=skip_validation,
        validation_model="auto",
        apply_fixes=False,
        force=False,
        rerun_all=False,
        reaggregate=True,
        report_style="pr",
    )

    fake_state = MagicMock()
    fake_state.get = MagicMock(
        side_effect=lambda k, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_at_start": "abc123",
        }.get(k, default)
    )
    report = MagicMock(return_value="# report\n")

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(cro, "get_or_create_state", return_value=fake_state))
        stack.enter_context(
            patch.object(cro, "get_phase_dir", return_value=phase_dir))
        stack.enter_context(
            patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"))
        stack.enter_context(
            patch.object(cro, "intent_to_add_untracked",
                         side_effect=lambda files, **_: contextlib.nullcontext()))
        if not render_report:
            stack.enter_context(patch.object(cro, "generate_review_report", report))
        for extra in extra_patches:
            stack.enter_context(extra)
        asyncio.run(
            cro.reaggregate_from_existing_files(
                "plan", plan.parent / "out", plan, args
            )
        )
    return report


def _issues_json_path(plan):
    """Where the code-review phase writes its raw issue list."""
    return cro.get_output_paths(
        plan, "code_review_issues", phase="code-review"
    ).with_suffix(".json")


def _reported_results(report_mock):
    """The ``results`` dict code-review reaggregate hands the report generator."""
    return report_mock.call_args[0][1]


class TestCodeReviewCopyReadsOneFilePerLogicalModel:
    def test_identical_duplicate_is_read_once_from_the_anchored_file(
        self, tmp_path, capsys
    ):
        """Fails on today's double count: 6 issues, 2 models, 2 `results` entries."""
        payload = _issues(["Issue 0", "Issue 1", "Issue 2"])
        plan, phase_dir = _cro_phase(
            tmp_path,
            {"sonnet": payload, "claude-code_sonnet": payload},
            models_requested=["claude-code:sonnet"],
        )

        report = _run_cro_reaggregate(plan, phase_dir)
        out = capsys.readouterr().out

        assert "Re-aggregating 3 issues from 1 models" in out

        # The report surface: one entry per logical model, carrying one copy of
        # the issues — a second entry would render the model twice.
        results = _reported_results(report)
        assert list(results) == ["claude-code:sonnet"]
        ok, issues, err = results["claude-code:sonnet"]
        assert ok is True and err is None
        assert len(issues) == 3
        assert {i["source_model"] for i in issues} == {"claude-code:sonnet"}
        assert {i["model"] for i in issues} == {"claude-code:sonnet"}

        # And the issues.json the phase writes holds three, not six.
        issues_json = json.loads(_issues_json_path(plan).read_text(encoding="utf-8"))
        assert len(issues_json) == 3

        skip_lines = [ln for ln in out.splitlines()
                      if "Skipping pre-anchor duplicate" in ln]
        assert len(skip_lines) == 1
        assert str(phase_dir / "sonnet.json") in skip_lines[0]
        assert str(phase_dir / "claude-code_sonnet.json") in skip_lines[0]
        assert "WARN" not in out

    def test_divergent_stale_and_current_warns_and_uses_the_anchored_file(
        self, tmp_path, capsys
    ):
        stale = _issues(["Stale A", "Stale B"])
        current = _issues(["Current A", "Current B", "Current C"])
        plan, phase_dir = _cro_phase(
            tmp_path,
            {"sonnet": stale, "claude-code_sonnet": current},
            models_requested=["claude-code:sonnet"],
        )

        report = _run_cro_reaggregate(plan, phase_dir)
        out = capsys.readouterr().out

        assert "Re-aggregating 3 issues from 1 models" in out

        _ok, issues, _err = _reported_results(report)["claude-code:sonnet"]
        assert sorted(i["title"] for i in issues) == [
            "Current A", "Current B", "Current C",
        ]

        warn_lines = [ln for ln in out.splitlines() if "WARN" in ln]
        assert len(warn_lines) == 1
        warn = warn_lines[0]
        assert str(phase_dir / "sonnet.json") in warn
        assert str(phase_dir / "claude-code_sonnet.json") in warn
        assert "2 issues" in warn and "3 issues" in warn

    def test_ambiguous_legacy_stem_exits_3_naming_every_candidate_spec(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _cro_phase(
            tmp_path,
            {
                "gpt-5_2": _issues(["Legacy issue"]),
                "openai_gpt-5_2": _issues(["OpenAI issue"]),
                "openrouter_gpt-5_2": _issues(["OpenRouter issue"]),
            },
            models_requested=["openai:gpt-5.2", "openrouter:gpt-5.2"],
        )

        with pytest.raises(SystemExit) as excinfo:
            _run_cro_reaggregate(plan, phase_dir)
        assert excinfo.value.code == 3

        err = capsys.readouterr().err
        assert str(phase_dir / "gpt-5_2.json") in err
        assert "openai:gpt-5.2" in err
        assert "openrouter:gpt-5.2" in err
        assert "rename" in err.lower() or "remove" in err.lower()
        # Nothing was aggregated behind the refusal.
        assert not _issues_json_path(plan).exists()


# ---------------------------------------------------------------------------
# The checkpoint the code-review copy reads has to be one review-code writes
# ---------------------------------------------------------------------------
#
# Everything above hands `.status.json` to the reaggregate path ready-made, so it
# pins how the map is *used* and says nothing about whether the phase that owns
# the dir ever writes one. `review-code` did not: `code_review_orchestrator.main`
# crossed its detach boundary straight into the fan-out without the
# `write_status` the base performs at the same point, so `build_unsanitize_map`
# read no `models_requested`, returned `{}`, and every group was re-attributed to
# the sanitized *stem* — `cursor-agent_cursor-grok-4_6-xhigh` for
# `cursor-agent:cursor-grok-4.6-xhigh`, the colon and the dot gone for good.
#
# That is not a cosmetic relabel either. It rewrites `models` and
# `source_model` in `grouped.json`, which changes the file's bytes, which breaks
# the `grouped_sha256` binding `--reaggregate` checks against
# `validation_tasks.json` moments later (`subagent_tasks.check_task_file_inputs`)
# — so the documented validate-then-reaggregate path refused its own task file
# with "the inputs it was materialized from have changed".
#
# Byte-identity of `grouped.json` across a group → reaggregate round trip is
# therefore the invariant to hold, and it takes two things, which is why the
# second test asserts on bytes rather than on the specs it can read back. The
# recorded canonical specs are one. The other is **key order**: this phase is
# the only one that stamps *two* attribution fields on a suggestion, and its
# reaggregate copy used to stamp them `source_model`-then-`model` while its
# fan-out stamps them `model`-then-`source_model`. `to_dict` carries a
# suggestion's dict through unreordered and `grouped_sha256` hashes bytes, so
# with the checkpoint restored and every spec identical the round trip still
# moved the file. The base fan-out is immune to that half by construction — it
# sets `source_model` and nothing else, on both sides.
#
# These two drive `main()` rather than asserting on the checkpoint write
# directly: the write has to land *after* the detach boundary (so it never runs
# in the invoker) and against the phase dir the fan-out actually uses, and only
# the real entry point can be wrong about either.


def _fanout_args(plan, models, skip_validation=True):
    """`parse_args` output for a foreground `review-code` run.

    ``skip_validation=False`` is the leg the documented workflow actually uses:
    the fan-out materializes `validation_tasks.json`, prints
    `[VALIDATION_BATCHES_PENDING]` and returns, leaving the run to be completed
    by `--reaggregate`.
    """
    return argparse.Namespace(
        plan_file=str(plan),
        models=list(models),
        interactive=False,
        quick=False,
        timeout=None,
        max_parallel=2,
        base_ref=None,
        skip_validation=skip_validation,
        validation_model="auto",
        apply_fixes=False,
        force=False,
        rerun_all=False,
        reaggregate=False,
        report_style="pr",
    )


def _forced(args):
    args.force = True
    return args


def _fanout_state():
    fake_state = MagicMock()
    fake_state.get = MagicMock(
        side_effect=lambda k, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_before_implement": "abc123",
            "head_at_start": "abc123",
            "pre_existing_changes": [],
        }.get(k, default)
    )
    fake_state.is_phase_completed = MagicMock(return_value=False)
    fake_state.state = {"phases_completed": {}}
    return fake_state


def _run_cro_fanout(tmp_path, issues_by_model, *, skip_validation=True,
                    extra_patches=(), failed=None):
    """Drive `code_review_orchestrator.main()` with the providers mocked out.

    Only the model invocation and the report renderer are stubbed — the phase
    dir, `save_model_result`, grouping and `grouped.json` are the real ones, so
    what lands on disk is what a real fan-out leaves for `--reaggregate` to read.

    ``failed`` (``{spec: reason}``) adds slots the fan-out records as failed —
    no result file, an entry under ``.status.json``'s ``models_failed`` — which
    is the dir the documented salvage flow, and an out-of-band result, land in.

    Returns the plan, the phase dir and **what `main()` returned**, because two
    of the questions below are about the run surviving something: a checkpoint
    write that quietly does nothing must not turn into an early return or a
    non-zero exit, and `None` (the validation leg's early return, which
    `sys.exit` treats as 0) has to be distinguishable from `EXIT_OK` only in the
    sense that neither is a failure.
    """
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    models = list(issues_by_model) + list(failed or {})
    results = {m: (True, issues_by_model[m], None) for m in issues_by_model}
    for spec, reason in (failed or {}).items():
        results[spec] = (False, None, reason)
    args = _fanout_args(plan, models, skip_validation=skip_validation)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(cro, "parse_args", return_value=args))
        stack.enter_context(
            patch.object(cro, "get_or_create_state", return_value=_fanout_state()))
        stack.enter_context(patch.object(cro, "resolve_models", return_value=models))
        stack.enter_context(patch.object(cro, "is_model_valid", return_value=True))
        stack.enter_context(
            patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"))
        stack.enter_context(
            patch.object(cro, "intent_to_add_untracked",
                         side_effect=lambda files, **_: contextlib.nullcontext()))
        stack.enter_context(
            patch.object(cro, "run_all_reviews", new=AsyncMock(return_value=results)))
        stack.enter_context(
            patch.object(cro, "generate_review_report", return_value="# report\n"))
        for extra in extra_patches:
            stack.enter_context(extra)
        exit_code = asyncio.run(cro.main())

    return plan, cro.get_phase_dir(plan, "code-review"), exit_code


#: Two specs whose sanitized stems are lossy in both of the ways that matter: the
#: `:` that separates provider from model and the `.` inside a version number.
_REAL_SPECS = ["claude-code:sonnet-4.6", "cursor-agent:cursor-grok-4.6-xhigh"]

#: The reserved ``progress`` keys, as ``utils/status_file.py``'s module docstring
#: enumerates them. A foreground run writes this object now (``report_progress``
#: falls back to ``status_file.update_progress`` when no run handle owns the phase
#: dir), so the foreground assertion below pins its *shape*: the keys a reader of
#: the checkpoint is allowed to meet, and never a ``run`` object beside them.
_RESERVED_PROGRESS_KEYS = {
    "step", "batch_index", "batches_total", "tasks_done", "tasks_total",
    "failed_task_ids", "models", "models_done", "models_succeeded",
    "models_total", "interval_s",
    # Written straight through ``update_status`` by the two executor mirrors,
    # so they already reach a foreground checkpoint.
    "subagents", "write_subagents", "sandbox_bypassed",
    # Landlock rung (2026-09-22): the contained-rung slots, `{spec: rung}`.
    "sandbox_fallback",
}


class TestReviewCodeWritesTheCheckpointReaggregationReads:
    def test_the_fanout_records_the_canonical_specs(self, tmp_path):
        """`.status.json` exists, and it holds specs — not filename stems."""
        plan, phase_dir, _exit_code = _run_cro_fanout(
            tmp_path,
            {
                _REAL_SPECS[0]: _issues(["Issue A"]),
                _REAL_SPECS[1]: _issues(["Issue B"]),
            },
        )

        status_path = phase_dir / ".status.json"
        assert status_path.exists(), (
            "review-code left no checkpoint, so build_unsanitize_map has nothing "
            "to restore the colon-bearing specs from"
        )
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["phase"] == "code-review"
        assert status["models_requested"] == _REAL_SPECS
        assert status["models_completed"] == _REAL_SPECS

        # And the map the reaggregate path builds from it actually inverts the
        # sanitization, which is the whole point of recording them.
        unsanitize = cro.build_unsanitize_map(str(phase_dir))
        assert unsanitize == {
            "claude-code_sonnet-4_6": "claude-code:sonnet-4.6",
            "cursor-agent_cursor-grok-4_6-xhigh": "cursor-agent:cursor-grok-4.6-xhigh",
        }

    def test_reaggregating_reproduces_grouped_json_byte_for_byte(self, tmp_path):
        """The round trip `--reaggregate` promises, on the bytes `grouped_sha256` binds.

        Fails on the missing checkpoint: without it every `models` entry and
        every `source_model` comes back as the sanitized stem, the file's bytes
        change, and `check_task_file_inputs` refuses `validation_tasks.json`
        against a `grouped.json` that reaggregation itself just rewrote.
        """
        plan, phase_dir, _exit_code = _run_cro_fanout(
            tmp_path,
            {
                _REAL_SPECS[0]: _issues(["Shared issue", "Only A"]),
                _REAL_SPECS[1]: _issues(["Shared issue", "Only B"]),
            },
        )

        grouped_path = phase_dir / "grouped.json"
        before = grouped_path.read_bytes()

        _run_cro_reaggregate(plan, phase_dir)

        assert grouped_path.read_bytes() == before, (
            "reaggregating a review-code phase rewrote grouped.json, so the "
            "grouped_sha256 recorded in validation_tasks.json no longer matches"
        )

        payload = json.loads(before.decode("utf-8"))
        attributed = {m for group in payload["groups"] for m in group["models"]}
        assert attributed == set(_REAL_SPECS)
        assert all(":" in spec and "." in spec for spec in attributed)


# ---------------------------------------------------------------------------
# The bound `grouped.json` is reused, not rewritten
# ---------------------------------------------------------------------------
#
# `--reaggregate` is the command that *checks* `grouped.json`'s bytes: it builds
# `inputs.grouped_sha256` from the file and hands it to `read_task_file`, which
# refuses `validation_tasks.json` if the hash has moved
# (`utils/subagent_tasks.check_task_file_inputs`). While the code-review copy
# regrouped unconditionally, it broke that binding and then audited it — the file
# it hashed was the file it had just rewritten — so the operator's
#
#     ERROR: Refusing task file .../validation_tasks.json: the inputs it was
#     materialized from have changed (grouped_sha256: ...)
#
# was this command refusing itself, on the one workflow the phase documents.
#
# The byte-identity test above is the regression guard for that, and it stays:
# it catches a serializer or grouping drift at the point of divergence. But it is
# a guard over a property nothing enforces — that regrouping is perfectly
# reproducible out of two separate dict literals in two functions — and the class
# of defect is "the bound file was rewritten at all", not "the rewrite differed".
# `utils/review_orchestrator_base.reaggregate_from_existing_files` has never had
# the problem because it *loads* the file instead ("preserves validation
# alignment", in its own printed words); these pin the ported branch by what it
# does NOT do: no write, and no call into grouping.


def _fanout_with_validation(tmp_path, issues_by_model):
    """A phase dir left exactly where `[VALIDATION_BATCHES_PENDING]` leaves one.

    The real fan-out, with validation *enabled*: per-model results,
    `grouped.json`, and a `validation_tasks.json` whose `inputs.grouped_sha256`
    binds to the bytes of the `grouped.json` sitting beside it. `main()` returns
    at the marker, which is where the documented workflow hands over to the
    agent.
    """
    plan, phase_dir, exit_code = _run_cro_fanout(
        tmp_path, issues_by_model, skip_validation=False
    )
    return plan, phase_dir, exit_code


def _complete_validation_batches(phase_dir, status="valid"):
    """Write the batch outputs the agent would have written after the marker.

    Every group the task file names gets a verdict, joined by `group_hash` —
    which `merge_batched_validation_results` treats as authoritative — so the
    merge exercises the real join rather than positional luck.

    `grouped.json` is deliberately untouched: its bytes are what
    `inputs.grouped_sha256` binds to, and the flow under test is the one where
    reaggregation *checks* that binding rather than moving it.
    """
    tasks = json.loads(
        (phase_dir / "validation_tasks.json").read_text(encoding="utf-8")
    )
    for batch in tasks["batches"]:
        verdicts = [
            {
                "group_index": global_index,
                "group_hash": batch["group_ids"][local_index],
                "status": status,
                "reason": f"Confirmed against the diff (group {global_index}).",
                "confidence": 0.9,
            }
            for local_index, global_index in enumerate(batch["group_indices"])
        ]
        (phase_dir / f"validation_batch_{batch['batch_index']}.json").write_text(
            json.dumps({"groups": verdicts}, indent=2), encoding="utf-8"
        )
    return tasks


def _unrelated_issues(names):
    """Issues no two of which `group_similar_suggestions` will ever merge.

    `compute_similarity` (`utils/suggestion_processor.py`) scores shared
    *content tokens* — normalized, rarity-weighted, title above body — so
    `_issues`' shared "Description for ..." boilerplate can pull an arbitrary
    list together, which is fine for the identity questions above and useless
    here. These share no content token at all, so the group count is the name
    count, and the batch count that follows from it (`max_per_batch=4`) is
    something a test can rely on.
    """
    return [
        {
            "title": name,
            "desc": f"{name}{name} {name}{name}",
            "importance": "medium",
            "file": f"src/{name.lower()}.py",
            "type": f"kind-{name.lower()}",
        }
        for name in names
    ]


#: Six mutually dissimilar findings, the last three shared, so the fan-out
#: produces six groups — two validation batches at `max_per_batch=4` — and at
#: least one group carrying both specs.
_A_ONLY = ["Alpha", "Bravo", "Charlie"]
_SHARED = ["Delta"]
_B_ONLY = ["Echo", "Foxtrot"]


def _two_model_findings():
    return {
        _REAL_SPECS[0]: _unrelated_issues(_A_ONLY + _SHARED),
        _REAL_SPECS[1]: _unrelated_issues(_SHARED + _B_ONLY),
    }


def _raises_if_grouped(*_args, **_kwargs):
    raise AssertionError(
        "group_similar_suggestions was called on a phase dir whose grouped.json "
        "is bound by validation_tasks.json — reaggregation must load that file, "
        "not regenerate it"
    )


class TestReaggregateReusesTheGroupedFileValidationIsBoundTo:
    def test_a_phase_dir_carrying_validation_tasks_is_never_regrouped(
        self, tmp_path
    ):
        """Not "the rewrite matched" — no rewrite, and no call into grouping.

        `group_similar_suggestions` is patched to raise rather than merely
        counted, because a reuse branch that still calls it and throws the result
        away would satisfy a byte-compare and leave the defect one refactor from
        returning.
        """
        plan, phase_dir, _exit_code = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        _complete_validation_batches(phase_dir)

        grouped_path = phase_dir / "grouped.json"
        before = grouped_path.read_bytes()

        _run_cro_reaggregate(
            plan,
            phase_dir,
            skip_validation=False,
            extra_patches=[
                patch.object(cro, "group_similar_suggestions", _raises_if_grouped)
            ],
        )

        assert grouped_path.read_bytes() == before, (
            "reaggregation rewrote the grouped.json that validation_tasks.json's "
            "grouped_sha256 binds to"
        )

    def test_a_regroup_that_would_diverge_no_longer_breaks_the_binding(
        self, tmp_path, capsys
    ):
        """The case the byte-identical round trip cannot cover.

        An extra key is stamped on one model's issues after the fan-out — the
        stand-in for every drift the plan's Out of scope lists: a field added on
        one path, a reorder inside `SuggestionGroup.to_dict`, a changed argument
        to `json.dump`. Regrouping would now produce different bytes, so the old
        unconditional rewrite would move `grouped_sha256` and refuse its own task
        file. Loading the bound file makes the divergence irrelevant: the run
        completes, the verdicts are merged, and the report is written.
        """
        plan, phase_dir, _exit_code = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        _complete_validation_batches(phase_dir)

        stem = cro.sanitize_model_name(_REAL_SPECS[0])
        result_path = phase_dir / f"{stem}.json"
        drifted = json.loads(result_path.read_text(encoding="utf-8"))
        for issue in drifted:
            issue["serializer_drift"] = "a field a later release added"
        result_path.write_text(json.dumps(drifted, indent=2), encoding="utf-8")

        grouped_path = phase_dir / "grouped.json"
        before = grouped_path.read_bytes()

        _run_cro_reaggregate(plan, phase_dir, skip_validation=False)
        out, err = capsys.readouterr()

        assert grouped_path.read_bytes() == before
        assert "Refusing task file" not in out + err
        assert (phase_dir / "report.md").exists()
        assert (phase_dir / "validation.json").exists()

    def test_a_run_interrupted_before_grouping_still_groups_and_writes(
        self, tmp_path
    ):
        """The other half of the guard: nothing bound means nothing to preserve.

        `--reaggregate` is also the path `main` points a dir at when it holds
        per-model results and no `grouped.json` ("Found N per-model result
        file(s) but no grouped.json"). Reuse must not turn that into a run that
        aggregates nothing.
        """
        plan, phase_dir, _exit_code = _run_cro_fanout(
            tmp_path, _two_model_findings()
        )
        grouped_path = phase_dir / "grouped.json"
        grouped_path.unlink()
        assert not (phase_dir / "validation_tasks.json").exists()

        _run_cro_reaggregate(plan, phase_dir)

        assert grouped_path.exists(), (
            "a phase dir interrupted before grouping was not grouped"
        )
        payload = json.loads(grouped_path.read_text(encoding="utf-8"))
        attributed = {m for group in payload["groups"] for m in group["models"]}
        assert attributed == set(_REAL_SPECS)


class TestReaggregateAttributesTheExecutorsOwnRunner:
    """F3: this branch (`validation_tasks.json` already on disk, its batch
    files complete) used to merge and save validation inline rather than
    calling the shared `review_orchestrator_base.merge_validation_batches`,
    and owed itself the same runner attribution: a batch's own
    `metadata.model` is a subagent's free-hand self-report, and
    `subagent_batch_report.json` — written only when `--exec-subagents`
    dispatched the batch itself — is the executor's own record of what
    actually ran it. Since 2026-09-17 it calls the shared merge (the last
    test pins that); the attribution is still asserted end to end.
    """

    def test_metadata_names_the_executors_dispatch_record_not_the_self_report(
        self, tmp_path
    ):
        plan, phase_dir, _exit_code = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        tasks = _complete_validation_batches(phase_dir)

        (phase_dir / "subagent_batch_report.json").write_text(json.dumps({
            "report_version": 1, "invoker": "executor", "phase": "code-review",
            "kind": "validation", "task_file": None,
            "tasks": [
                {
                    "task_id": batch["task_id"], "status": "ok",
                    "runner": "opencode:openrouter/~deepseek/deepseek-flash-latest",
                }
                for batch in tasks["batches"]
            ],
        }), encoding="utf-8")

        _run_cro_reaggregate(plan, phase_dir, skip_validation=False)

        validation = json.loads(
            (phase_dir / "validation.json").read_text(encoding="utf-8")
        )
        assert validation["metadata"]["model"] == (
            "opencode:openrouter/~deepseek/deepseek-flash-latest"
        )
        assert validation["metadata"]["runner"] == (
            "opencode:openrouter/~deepseek/deepseek-flash-latest"
        )

    def test_the_merge_is_the_base_s_shared_one(self, tmp_path):
        """One implementation of "batch files -> validation.json -> groups":
        the branch re-implemented the three steps (importing the base's
        private runner lookup to do it) and could drift from the base's."""
        plan, phase_dir, _exit_code = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        _complete_validation_batches(phase_dir)
        calls = []
        real_merge = cro.merge_validation_batches

        def spy(phase_dir_arg, batch_metadata, grouped, validation_path):
            calls.append((phase_dir_arg, validation_path, len(grouped)))
            return real_merge(phase_dir_arg, batch_metadata, grouped, validation_path)

        _run_cro_reaggregate(
            plan, phase_dir, skip_validation=False,
            extra_patches=(patch.object(cro, "merge_validation_batches", spy),),
        )

        assert calls == [(str(phase_dir), str(phase_dir / "validation.json"),
                          len(_groups_on_disk(phase_dir)))]
        validation = json.loads(
            (phase_dir / "validation.json").read_text(encoding="utf-8")
        )
        assert len(validation["groups"]) == len(_groups_on_disk(phase_dir))

    def test_falls_back_to_the_self_report_with_no_executor_record(
        self, tmp_path
    ):
        """The host-native (Claude Code Task tool) path leaves no
        `subagent_batch_report.json` — nothing this process ran — so the
        best-effort derivation from the batches' own `metadata.model` is the
        only provenance available."""
        plan, phase_dir, _exit_code = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        _complete_validation_batches(phase_dir)
        assert not (phase_dir / "subagent_batch_report.json").exists()

        _run_cro_reaggregate(plan, phase_dir, skip_validation=False)

        validation = json.loads(
            (phase_dir / "validation.json").read_text(encoding="utf-8")
        )
        # `_complete_validation_batches` writes no `metadata.model` of its own
        # (it writes only `{"groups": verdicts}`), so the derivation falls
        # through to "unknown" — the pre-existing behaviour this test pins is
        # that nothing here raises and no `runner` key is invented.
        assert validation["metadata"]["model"] == "unknown"
        assert "runner" not in validation["metadata"]


# ---------------------------------------------------------------------------
# The documented validate-then-reaggregate workflow, end to end
# ---------------------------------------------------------------------------
#
# Every test above drives `cro.main()` with `skip_validation=True`. So
# `validation_tasks.json` is never materialized, no batch output is ever
# completed, and `check_task_file_inputs` — the function that printed the refusal
# this whole area exists to fix — is never called. Byte-identity of
# `grouped.json` is a proxy for the cause; it is one layer below the failure the
# operator actually met, which was a *task file* being rejected against the file
# reaggregation had just rewritten. Nothing in this suite would have gone red on
# that as reported.
#
# So this walks the whole documented path: fan out with validation on, take the
# `[VALIDATION_BATCHES_PENDING]` handover, write the batch outputs the agent
# would have written, and complete the run with `--reaggregate`. It asserts on
# the operator-visible outcome — accepted, merged, reported, exit 0 — and it
# asserts `check_task_file_inputs` actually *ran*, since "no refusal printed" is
# equally true of a check that was skipped.


class TestTheDocumentedValidateThenReaggregateFlowCompletes:
    def test_the_task_file_is_accepted_and_its_verdicts_reach_the_report(
        self, tmp_path, capsys
    ):
        plan, phase_dir, fanout_exit = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        fanout_out = capsys.readouterr().out

        # The handover the phase documents, and the artifacts it leaves behind.
        assert "[VALIDATION_BATCHES_PENDING]" in fanout_out
        assert fanout_exit is None or fanout_exit == cro.EXIT_OK
        tasks_path = phase_dir / "validation_tasks.json"
        assert tasks_path.exists()
        bound_hash = json.loads(tasks_path.read_text(encoding="utf-8"))["inputs"][
            "grouped_sha256"
        ]

        tasks = _complete_validation_batches(phase_dir)
        assert tasks["total_batches"] >= 2, (
            "the fixture is meant to exercise the *batched* handover"
        )
        # Left exactly as the fan-out bound it — the test would be vacuous if the
        # fixture re-bound the hash to whatever reaggregation produces.
        assert json.loads(tasks_path.read_text(encoding="utf-8"))["inputs"][
            "grouped_sha256"
        ] == bound_hash

        checked = []
        real_check = subagent_tasks_mod.check_task_file_inputs

        def spy(document, path, phase_dir_arg, current_inputs):
            checked.append((path, dict(current_inputs)))
            return real_check(document, path, phase_dir_arg, current_inputs)

        _run_cro_reaggregate(
            plan,
            phase_dir,
            skip_validation=False,
            render_report=True,
            extra_patches=[
                patch.object(subagent_tasks_mod, "check_task_file_inputs", spy)
            ],
        )
        out, err = capsys.readouterr()

        # Accepted — and the check that could have refused it actually ran.
        assert checked, (
            "check_task_file_inputs never ran, so this asserts nothing about the "
            "binding that produced the original refusal"
        )
        assert checked[0][0] == str(tasks_path)
        assert checked[0][1]["grouped_sha256"] == bound_hash
        assert "Refusing task file" not in out + err

        # The verdicts were consumed, not ignored.
        validation_path = phase_dir / "validation.json"
        assert validation_path.exists()
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Validation:** 6 valid, 0 invalid, 0 needs human review" in report
        assert "? Unknown" not in report

        # Attribution is the canonical spec, never the stem it sanitizes to.
        for spec in _REAL_SPECS:
            assert f"**Model:** {spec}" in report
            assert cro.sanitize_model_name(spec) not in report

    def test_the_same_flow_holds_with_grouping_rigged_to_raise(
        self, tmp_path, capsys
    ):
        """The proof that the bound file is not merely reproduced but reused.

        Same workflow, same assertions on the outcome, with
        `group_similar_suggestions` unavailable. Before the reuse branch this run
        could not exist: reaggregation had to regroup before it could hash.
        """
        plan, phase_dir, _fanout_exit = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        _complete_validation_batches(phase_dir)
        capsys.readouterr()

        _run_cro_reaggregate(
            plan,
            phase_dir,
            skip_validation=False,
            render_report=True,
            extra_patches=[
                patch.object(cro, "group_similar_suggestions", _raises_if_grouped)
            ],
        )
        out, err = capsys.readouterr()

        assert "Refusing task file" not in out + err
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Validation:** 6 valid, 0 invalid, 0 needs human review" in report
        for spec in _REAL_SPECS:
            assert f"**Model:** {spec}" in report


# ---------------------------------------------------------------------------
# Plan review-run-scope-salvage-and-failure-visibility §5 (T5.1): the
# reaggregate reads the failure map back
# ---------------------------------------------------------------------------
#
# The code-review reaggregate used to derive its failure map from `results`,
# which holds only models with a file on disk — so the map was always `{}` and
# a reviewer that timed out vanished from every report rebuilt after it. The
# facts were on disk all along: `.status.json.models_failed`, reconciled with
# the files by `record_models_present`.

_LOST_SPEC = _REAL_SPECS[1]
_LOST_REASON = "cursor-agent timed out after 1200s"


def _cro_report_md(plan):
    """`report.md` where the code-review reaggregate writes it: the plan-derived
    phase dir (`get_output_paths`), whatever `get_phase_dir` was patched to."""
    path = cro.get_output_paths(plan, "code_review", phase="code-review")
    return path.with_suffix(".md").read_text(encoding="utf-8")


def _line_after_models(report):
    lines = report.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("**Models:**"))
    return lines[index + 1]


#: Five specs whose stems invert through the recorded checkpoint.
_FIVE = [f"claude-code:model-{letter}" for letter in "abcde"]


def _five_spec_phase(tmp_path, checkpoint_extra=None, **fields):
    """A code-review dir with result files for a, b, c and a checkpoint."""
    plan, phase_dir = _cro_phase(
        tmp_path,
        {cro.sanitize_model_name(spec): _issues([f"Issue {spec}"]) for spec in _FIVE[:3]},
    )
    checkpoint = {
        "phase": "code-review",
        "state": "models_complete",
        "started_at": "2026-01-01T00:00:00",
        **fields,
        **(checkpoint_extra or {}),
    }
    write_status(str(phase_dir), checkpoint)
    return plan, phase_dir


class TestTheCodeReviewReaggregateNamesTheModelsThatProducedNothing:
    def test_the_documented_flow_names_the_recorded_failure(self, tmp_path, capsys):
        plan, phase_dir, _exit = _run_cro_fanout(
            tmp_path,
            {_REAL_SPECS[0]: _unrelated_issues(_A_ONLY + _SHARED)},
            skip_validation=False,
            failed={_LOST_SPEC: _LOST_REASON},
        )
        _complete_validation_batches(phase_dir)
        capsys.readouterr()

        _run_cro_reaggregate(
            plan, phase_dir, skip_validation=False, render_report=True,
        )
        out = capsys.readouterr().out

        report = _cro_report_md(plan)
        assert _line_after_models(report) == (
            f"**Models failed:** 1 of 2 requested — {_LOST_SPEC}: {_LOST_REASON}"
        )
        assert f"- **{_LOST_SPEC}**: {_LOST_REASON}" in report
        # T5.4a/T5.4b: named to the host, and the run reads partial.
        assert f"[MODEL_FAILED] {_LOST_SPEC}: {_LOST_REASON}" in out
        assert f"[PARTIAL_RESULTS: {_LOST_SPEC}]" in out
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["failedModels"] == {_LOST_SPEC: _LOST_REASON}
        assert data["requestedModels"] == 2

    def test_a_model_whose_file_appeared_is_not_reported(self, tmp_path, capsys):
        plan, phase_dir, _exit = _run_cro_fanout(
            tmp_path,
            {_REAL_SPECS[0]: _issues(["Issue A"])},
            failed={_LOST_SPEC: _LOST_REASON},
        )
        # A salvage (or an out-of-band result) lands before the reaggregate.
        (phase_dir / f"{cro.sanitize_model_name(_LOST_SPEC)}.json").write_text(
            json.dumps(_issues(["Issue B"])), encoding="utf-8"
        )
        capsys.readouterr()

        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        out = capsys.readouterr().out

        assert f"[{_LOST_SPEC}] result file present; recorded as completed" in out
        assert "[MODEL_FAILED]" not in out
        assert "[PARTIAL_RESULTS:" not in out
        report = _cro_report_md(plan)
        assert "**Models failed:**" not in report
        assert "## Failed Reviews" not in report

    def test_the_denominator_is_the_requested_set_not_success_plus_failure(
        self, tmp_path, capsys
    ):
        """Five requested, three with files, one recorded failed, one with
        neither: the fifth is named with the fixed gap reason, so N = M minus
        the models under **Models:**."""
        plan, phase_dir = _five_spec_phase(
            tmp_path,
            models_requested=list(_FIVE),
            models_completed=list(_FIVE[:3]),
            models_failed={_FIVE[3]: "exited with code 1"},
        )
        assert rob.recorded_requested_models(str(phase_dir), _FIVE[:3]) == _FIVE

        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        out = capsys.readouterr().out

        assert _line_after_models(_cro_report_md(plan)) == (
            f"**Models failed:** 2 of 5 requested — {_FIVE[3]}: exited with code 1; "
            f"{_FIVE[4]}: {rob.UNACCOUNTED_MODEL_REASON}"
        )
        assert f"[MODEL_FAILED] {_FIVE[4]}: no result file and no recorded failure" in out

    def test_a_dropped_progress_object_reads_the_same_five(self, tmp_path):
        """A detached reaggregate's claim drops `progress` wholesale."""
        fields = dict(
            models_requested=list(_FIVE),
            models_completed=list(_FIVE[:3]),
            models_failed={_FIVE[3]: "exited with code 1"},
        )
        with_progress = tmp_path / "with"
        without = tmp_path / "without"
        with_progress.mkdir()
        without.mkdir()
        _p1, dir_with = _five_spec_phase(with_progress, checkpoint_extra={
            "progress": {"models_total": 5, "models_done": 5,
                         "models": {spec: {"state": "done"} for spec in _FIVE}},
        }, **fields)
        _p2, dir_without = _five_spec_phase(without, **fields)

        assert rob.remaining_failed_models(str(dir_with), _FIVE[:3]) == (
            rob.remaining_failed_models(str(dir_without), _FIVE[:3])
        )
        failures, requested = rob.remaining_failed_models(str(dir_without), _FIVE[:3])
        assert requested == _FIVE
        assert failures == {
            _FIVE[3]: "exited with code 1",
            _FIVE[4]: rob.UNACCOUNTED_MODEL_REASON,
        }

    def test_no_models_requested_falls_back_to_the_union_of_the_rest(self, tmp_path):
        plan, phase_dir = _five_spec_phase(
            tmp_path,
            models_completed=list(_FIVE[:3]),
            models_failed={_FIVE[3]: "exited with code 1"},
        )
        failures, requested = rob.remaining_failed_models(str(phase_dir), _FIVE[:3])
        assert requested == _FIVE[:4]
        assert failures == {_FIVE[3]: "exited with code 1"}

    def test_an_old_file_from_a_dropped_model_counts_once_on_each_side(
        self, tmp_path, capsys
    ):
        """A model the latest run did not request, whose old result file is
        still in the dir: loaded, recorded completed, counted as succeeded and
        once in the denominator — so the numbers still add up."""
        plan, phase_dir = _five_spec_phase(
            tmp_path,
            models_requested=list(_FIVE[:2]) + [_FIVE[3]],
            models_completed=list(_FIVE[:2]),
            models_failed={_FIVE[3]: "exited with code 1"},
        )
        # _FIVE[2]'s file is on disk (written by the fixture) but unrequested.
        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        capsys.readouterr()

        report = _cro_report_md(plan)
        models_line = next(
            line for line in report.splitlines() if line.startswith("**Models:**")
        )
        succeeded = [spec for spec in _FIVE if spec in models_line]
        assert len(succeeded) == 2  # a, b by spec; c by its stem
        assert cro.sanitize_model_name(_FIVE[2]) in models_line
        assert _line_after_models(report) == (
            f"**Models failed:** 1 of 4 requested — {_FIVE[3]}: exited with code 1"
        )

    def test_a_checkpoint_with_none_of_the_fields_writes_no_line(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _cro_phase(
            tmp_path, {"claude-code_sonnet": _issues(["Issue 0"])},
        )
        assert rob.recorded_requested_models(str(phase_dir), ["claude-code_sonnet"]) is None

        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        capsys.readouterr()

        assert "**Models failed:**" not in _cro_report_md(plan)
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert "requestedModels" not in data
        assert data["failedModels"] == {}


# ---------------------------------------------------------------------------
# A checkpoint that could not be written stops being a silent degradation
# ---------------------------------------------------------------------------
#
# Every test above drives a writable `tmp_path`, so `write_status` always
# succeeds and the failure path is never entered — the suite would look identical
# if the call had been left out on an unwritable dir. And there is nothing to
# catch: `write_status` is best-effort *by contract*, wrapping the temp write and
# the `os.replace` in a bare `except Exception: pass` (`utils/status_file.py`),
# unlinking the temp file and returning `None`. A read-only or full filesystem, a
# phase dir made unwritable between the mkdir and the write, a checkpoint lock it
# could not take — each leaves the run proceeding exactly as it did before the
# write existed, and the operator meets the consequence days later as stems in a
# report.
#
# Patching it to a no-op is the honest simulation of that swallow. The posture
# under test is verify-and-warn: the fan-out below this line is the expensive
# half of the run and sits under the detach boundary where nobody is watching, so
# a checkpoint that could not be written must never become a fan-out that never
# ran. Fail-closed belongs at `resolve_models`, where refusing costs a correct
# run nothing.

#: The sentence both warning sites share, and the thing the negatives assert the
#: absence of. Matching on the consequence rather than on "WARNING" keeps these
#: from passing on some unrelated warning that happens to reach stderr.
_STEM_WARNING = "attributes issues by filename stem instead of by model spec"


def _no_op_write_status(*_args, **_kwargs):
    """`write_status`, exactly as it behaves when its write throws."""
    return None


def _half_landing_write_status(phase_dir, data):
    """A checkpoint that lands carrying `phase` and `state` and nothing else.

    What a partially merged, truncated or hand-edited `.status.json` looks like.
    `build_unsanitize_map` reads only `models_requested`/`models_completed`, so
    this file is worth exactly as much as no file — which is why the probe is on
    the field and not on `os.path.exists`.
    """
    write_status(phase_dir, {"phase": data.get("phase"), "state": data.get("state")})


class TestAFanOutWhoseCheckpointWriteFailedSaysSoAndCarriesOn:
    def test_a_write_that_silently_does_nothing_neither_raises_nor_returns_early(
        self, tmp_path
    ):
        """The whole point of warning instead of raising: the run still happens."""
        plan, phase_dir, exit_code = _run_cro_fanout(
            tmp_path,
            {
                _REAL_SPECS[0]: _issues(["Issue A"]),
                _REAL_SPECS[1]: _issues(["Issue B"]),
            },
            extra_patches=[
                patch.object(cro, "write_status", _no_op_write_status)
            ],
        )

        assert exit_code == cro.EXIT_OK
        for spec in _REAL_SPECS:
            assert (phase_dir / f"{cro.sanitize_model_name(spec)}.json").exists(), (
                "the fan-out did not dispatch, so a lost checkpoint cost the run "
                "the expensive half rather than the identity map"
            )
        assert (phase_dir / "report.md").exists()

    def test_the_warning_names_the_consequence_and_both_recoveries_on_stderr(
        self, tmp_path, capsys
    ):
        """One warning, on stderr, in the operator's terms rather than the mechanism's."""
        _plan, _phase_dir, _exit_code = _run_cro_fanout(
            tmp_path,
            {
                _REAL_SPECS[0]: _issues(["Issue A"]),
                _REAL_SPECS[1]: _issues(["Issue B"]),
            },
            extra_patches=[
                patch.object(cro, "write_status", _no_op_write_status)
            ],
        )
        out, err = capsys.readouterr()

        assert _STEM_WARNING in err
        # stdout is this phase's machine-readable channel; a degradation nobody
        # chose does not belong in it.
        assert _STEM_WARNING not in out
        assert err.count(_STEM_WARNING) == 1, "one warning, not one per model"

        # The consequence, spelled as the thing that will go wrong later.
        assert "--reaggregate" in err
        assert "cursor-agent_cursor-grok-4_6-xhigh" in err
        assert "cursor-agent:cursor-grok-4.6-xhigh" in err

        # Both recoveries, and the specs the hand-written one needs — a stem in
        # an old report is exactly what cannot be inverted, so the warning has to
        # be the source.
        assert "--force" in err
        assert ".status.json" in err
        for spec in _REAL_SPECS:
            assert spec in err

    def test_a_checkpoint_that_lands_without_models_requested_is_warned_identically(
        self, tmp_path, capsys
    ):
        """The probe is on the field `build_unsanitize_map` needs, not on the file."""
        _plan, phase_dir, exit_code = _run_cro_fanout(
            tmp_path,
            {_REAL_SPECS[0]: _issues(["Issue A"])},
            extra_patches=[
                patch.object(cro, "write_status", _half_landing_write_status)
            ],
        )
        err = capsys.readouterr().err

        assert (phase_dir / ".status.json").exists(), (
            "this variant is about a checkpoint that exists and is useless"
        )
        assert _STEM_WARNING in err
        assert exit_code == cro.EXIT_OK

    def test_an_ordinary_run_on_a_writable_dir_is_not_warned_about(
        self, tmp_path, capsys
    ):
        """The negative that keeps the read-back from becoming noise."""
        _plan, phase_dir, exit_code = _run_cro_fanout(
            tmp_path,
            {
                _REAL_SPECS[0]: _issues(["Issue A"]),
                _REAL_SPECS[1]: _issues(["Issue B"]),
            },
        )
        out, err = capsys.readouterr()

        assert _STEM_WARNING not in err + out
        assert exit_code == cro.EXIT_OK
        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        assert status["models_requested"] == _REAL_SPECS


# ---------------------------------------------------------------------------
# ...and reaggregation says the same thing from the other end
# ---------------------------------------------------------------------------
#
# The fan-out's read-back covers runs from here on. It does nothing for the
# artifacts already on disk: a dir whose fan-out predates the checkpoint write, a
# dir copied to another machine or carried across a branch, a dir whose
# `.status.json` was hand-merged without the field. `build_unsanitize_map`
# returns `{}` for all three, `resolve_result_files_by_model` falls back to "the
# stem IS the identity", and on the `--skip-validation` leg the run completes and
# writes a report attributed to stems with nothing said about it.
#
# Warning-only, deliberately, and not a gate. Refusing would break a path that
# completes today, which is a change that owes an override flag; and it is the
# same posture as the fan-out's read-back, so the operator meets one story about
# stem attribution rather than two. The wording is literally shared
# (`warn_stem_attribution`), which is what keeps them from drifting into two
# paraphrases.


def _stemmed_phase(tmp_path, checkpoint):
    """A code-review phase dir with results and *checkpoint* written verbatim.

    `checkpoint=None` writes no `.status.json` at all — the pre-checkpoint or
    hand-copied dir. Anything else is written through `write_status`, so the
    fixture cannot drift from the real document's shape.
    """
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")
    phase_dir = cro.get_phase_dir(plan, "code-review")

    for spec, titles in ((_REAL_SPECS[0], ["Alpha"]), (_REAL_SPECS[1], ["Bravo"])):
        (phase_dir / f"{cro.sanitize_model_name(spec)}.json").write_text(
            json.dumps(_unrelated_issues(titles)), encoding="utf-8"
        )

    if checkpoint is not None:
        write_status(str(phase_dir), dict(checkpoint))
    return plan, phase_dir


class TestReaggregatingADirWithNoRecordedModelsWarnsAndStillCompletes:
    def test_a_dir_with_no_status_json_still_writes_the_report(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _stemmed_phase(tmp_path, checkpoint=None)
        assert not (phase_dir / ".status.json").exists()

        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        out, err = capsys.readouterr()

        # Warned, not refused: the report is still there and nothing exited.
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "# Code Review Report" in report
        assert _STEM_WARNING in err
        assert _STEM_WARNING not in out
        assert err.count(_STEM_WARNING) == 1
        assert "--force" in err and ".status.json" in err

        # And the misattribution the warning is about is really happening, which
        # is what makes it worth printing.
        for spec in _REAL_SPECS:
            assert cro.sanitize_model_name(spec) in report

    def test_a_checkpoint_without_models_requested_is_warned_about_identically(
        self, tmp_path, capsys
    ):
        """A file that exists and inverts nothing is the same defect as no file."""
        plan, phase_dir = _stemmed_phase(
            tmp_path,
            checkpoint={"phase": "code-review", "state": "models_complete"},
        )
        assert (phase_dir / ".status.json").exists()

        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        err = capsys.readouterr().err

        assert _STEM_WARNING in err
        assert (phase_dir / "report.md").exists()

    def test_a_dir_with_a_proper_checkpoint_is_not_warned_about(
        self, tmp_path, capsys
    ):
        """The negative, and the positive it implies: specs, not stems."""
        plan, phase_dir = _stemmed_phase(
            tmp_path,
            checkpoint={
                "phase": "code-review",
                "state": "models_complete",
                "models_requested": list(_REAL_SPECS),
                "models_completed": list(_REAL_SPECS),
            },
        )

        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        out, err = capsys.readouterr()

        assert _STEM_WARNING not in err + out
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        for spec in _REAL_SPECS:
            assert f"**Model:** {spec}" in report
            assert cro.sanitize_model_name(spec) not in report


# ---------------------------------------------------------------------------
# The gate that keeps the identity map injective in the first place (D1 amended)
# ---------------------------------------------------------------------------

#: A pair that is distinct as specs and identical as filenames. `sanitize_model_name`
#: flattens every separator a spec carries, so `.` and `_` both become `_` and these
#: two collapse onto `p_a_b`. The `--models` leg is the one used here deliberately:
#: it is the only leg that does not filter on `get_provider(name).is_available()`,
#: so a fictitious provider reaches the gate instead of dying earlier on "all
#: providers unavailable" — which would pass this test for the wrong reason. The
#: config legs are covered as unit tests in `tests/test_provider_registry.py`.
_COLLIDING_SPECS = ["p:a.b", "p:a_b"]
_COLLIDING_STEM = "p_a_b"


def _run_cro_fanout_resolving_models(tmp_path, models):
    """`_run_cro_fanout`, minus the one patch that would hide the gate.

    Every other fan-out driver in this file patches `resolve_models` to hand back
    a fixed list, because every other question here is about what happens *after*
    selection. This one is about selection itself, so the real function has to
    run — patching it is exactly the mistake that would make a green test mean
    nothing.
    """
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")
    args = _fanout_args(plan, models)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(cro, "parse_args", return_value=args))
        stack.enter_context(
            patch.object(cro, "get_or_create_state", return_value=_fanout_state()))
        stack.enter_context(patch.object(cro, "is_model_valid", return_value=True))
        stack.enter_context(
            patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"))
        stack.enter_context(
            patch.object(cro, "intent_to_add_untracked",
                         side_effect=lambda files, **_: contextlib.nullcontext()))
        stack.enter_context(
            patch.object(cro, "run_all_reviews",
                         new=AsyncMock(side_effect=AssertionError(
                             "the fan-out dispatched despite a stem collision"))))
        stack.enter_context(
            patch.object(cro, "generate_review_report", return_value="# report\n"))
        return plan, asyncio.run(cro.main())


class TestTwoSpecsSharingOneFilenameStemNeverReachTheFanOut:
    """The end-to-end shape of D1's collision gate.

    `tests/test_provider_registry.py` pins `resolve_models`' refusal directly, on
    all four legs. What that cannot show is the consequence the refusal buys:
    that the run stops while stopping is still free. Both silent failures a
    collision causes are downstream of a per-model *file* — one model's result
    overwriting the other's under the shared stem, and `skip_existing_result`
    mis-skipping the second because the first one's file is already there — so
    the property worth pinning here is that **no per-model result exists at all**
    when the run ends. The overwrite and the mis-skip are then unreachable rather
    than merely unlikely.
    """

    def test_a_colliding_pair_stops_the_run_before_any_result_is_written(
        self, tmp_path, capsys
    ):
        """Exit 3, both specs and the stem named, and nothing on disk to misread."""
        with pytest.raises(SystemExit) as exit_info:
            _run_cro_fanout_resolving_models(tmp_path, _COLLIDING_SPECS)

        assert exit_info.value.code == 3

        err = capsys.readouterr().err
        assert "share one filename stem" in err
        for spec in _COLLIDING_SPECS:
            assert repr(spec).replace('"', "'") in err or spec in err
        assert _COLLIDING_STEM in err

        # `get_phase_dir` mkdirs, so asking it where the phase dir *would* be
        # would create the very directory this asserts is absent. Sweep the
        # tmp tree for the artifact instead: the collision refusal fires above
        # `out_dir.mkdir()`, so not one per-model file — under either spec's
        # stem — can have been written.
        written = [p.name for p in tmp_path.rglob("*.json")]
        assert f"{_COLLIDING_STEM}.json" not in written
        assert not any(name.startswith(_COLLIDING_STEM) for name in written)

    def test_a_pair_whose_stems_differ_is_not_stopped(self, tmp_path):
        """The negative: the gate refuses a collision, not a separator.

        `_REAL_SPECS` carry both lossy characters — the `:` between provider and
        model and the `.` inside a version — and still sanitize apart. A gate
        that fired on them would refuse every realistic run this plan exists to
        make work.
        """
        with pytest.raises(AssertionError, match="dispatched despite"):
            _run_cro_fanout_resolving_models(tmp_path, list(_REAL_SPECS))


# ---------------------------------------------------------------------------
# A stale validation.json never outranks the batch files that superseded it (D7)
# ---------------------------------------------------------------------------
#
# The resume of a real review re-grouped 48 groups into 61, wrote a fresh
# `validation_tasks.json`, and left the first run's `validation.json` where it
# was. With all 21 batch files complete, `--reaggregate` trusted that file first
# ("Found existing validation.json, using it"), and the 13 groups it had never
# seen each took whatever record sat at their *index* — another group's verdict.
# 52 / 3 / 6 on screen, 56 / 4 / 1 on disk, exit 0.
#
# Three rules close it, each pinned below against the base reaggregator over a
# real phase dir: a `validation.json` records the grouping it belongs to
# (`metadata.grouped_sha256`) and is honoured only when that is the grouping on
# disk; a re-group retires the file it orphans (renamed, never deleted); and the
# index fallback in `apply_validation_to_groups` is for hash-less records only.


def _distinct_suggestions(names):
    """Review-plan suggestions no two of which grouping will merge.

    Same construction as `_unrelated_issues` — no shared word, a distinct
    `type` each — so the group count is the name count and a second model's
    findings *add* groups rather than joining existing ones.
    """
    return [
        {
            "title": name,
            "desc": f"{name}{name} {name}{name}",
            "importance": "MEDIUM",
            "reference": f"Step {name}",
            "type": f"kind-{name.lower()}",
        }
        for name in names
    ]


def _validated_round(plan, out_dir, phase_dir, status):
    """Take a phase dir from per-model results to a merged `validation.json`.

    The first reaggregate groups and materializes `validation_tasks.json` (the
    marker handover); the batches are completed with *status*; the second
    reaggregate merges them, which is the write that produces `validation.json`
    — bound, since D7, to the `grouped.json` it was merged against.
    """
    _reaggregate(plan, out_dir, phase_dir)
    assert (phase_dir / "validation_tasks.json").exists()
    _complete_validation_batches(phase_dir, status=status)
    _reaggregate(plan, out_dir, phase_dir)
    return json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))


def _grouped_hash(phase_dir):
    return subagent_tasks_mod.file_content_hash(str(phase_dir / "grouped.json"))


def _groups_on_disk(phase_dir):
    payload = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
    return payload["groups"] if isinstance(payload, dict) else payload


def _stale_files(phase_dir):
    return sorted(p.name for p in phase_dir.glob("validation.*.stale.json"))


def _validation_summary(phase_dir):
    report = (phase_dir / "report.md").read_text(encoding="utf-8")
    line = next(ln for ln in report.splitlines() if ln.startswith("**Validation:**"))
    return line


def _regrouped_after_a_second_model(plan, out_dir, phase_dir, first_validation):
    """Reproduce the dir the defective resume left behind.

    The first grouping was validated (`first_validation`). Then a second model's
    result lands and the phase is re-materialized: a fresh grouping, a fresh
    `validation_tasks.json`, and fresh batches completed as *invalid* so a
    verdict from the old file is distinguishable from one out of the batches.
    Pre-D7 the resume left the first `validation.json` where it was — the same
    bytes are put back after the regroup to reproduce exactly that dir.
    """
    stale_bytes = json.dumps(first_validation, indent=2).encode("utf-8")
    for name in ("validation.json", "validation_tasks.json"):
        (phase_dir / name).unlink()
    for batch in phase_dir.glob("validation_batch_*.json"):
        batch.unlink()
    (phase_dir / "codex_gpt-6.json").write_text(
        json.dumps(_distinct_suggestions(["Delta", "Echo"])), encoding="utf-8"
    )

    _reaggregate(plan, out_dir, phase_dir)
    assert _grouped_hash(phase_dir) != first_validation["metadata"]["grouped_sha256"], (
        "the fixture is meant to change the grouping"
    )
    _complete_validation_batches(phase_dir, status="invalid")
    (phase_dir / "validation.json").write_bytes(stale_bytes)
    return stale_bytes


class TestAStaleValidationFileYieldsToTheBatches:
    def test_a_hashed_file_from_an_older_grouping_is_retired_and_the_batches_merge(
        self, tmp_path, capsys
    ):
        """The defect as reported: hashed `validation.json` from grouping A, fresh
        batches for grouping B. The batches win, the reason is printed, and the
        old file is renamed rather than read or overwritten."""
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"claude-code_sonnet": _distinct_suggestions(["Alpha", "Bravo", "Charlie"])},
            models_requested=["claude-code:sonnet", "codex:gpt-6"],
        )
        first = _validated_round(plan, out_dir, phase_dir, status="valid")
        assert first["metadata"]["grouped_sha256"] == _grouped_hash(phase_dir)
        assert _validation_summary(phase_dir) == (
            "**Validation:** 3 valid, 0 invalid, 0 needs human review, "
            "0 validation failed"
        )

        stale_bytes = _regrouped_after_a_second_model(plan, out_dir, phase_dir, first)
        capsys.readouterr()

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "Found existing validation.json" not in out
        assert (
            f"Not using validation.json: validation.json belongs to grouping "
            f"{first['metadata']['grouped_sha256']}, current grouped.json is "
            f"{_grouped_hash(phase_dir)}"
        ) in out
        assert "batch files found, merging" in out

        # Renamed, never deleted, and byte-identical to what was left behind.
        stale = _stale_files(phase_dir)
        assert len(stale) == 1
        assert (phase_dir / stale[0]).read_bytes() == stale_bytes
        assert "[VALIDATION_RETIRED] " + stale[0] in out

        # The merged file is bound to the grouping it was merged against, and
        # every verdict came out of the batches — none out of the old file.
        merged = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert merged["metadata"]["grouped_sha256"] == _grouped_hash(phase_dir)
        assert [g["status"] for g in merged["groups"]] == ["invalid"] * 5
        assert _validation_summary(phase_dir) == (
            "**Validation:** 0 valid, 5 invalid, 0 needs human review, "
            "0 validation failed"
        )

    def test_a_hashed_file_applied_to_a_regrouped_set_yields_exactly_the_unmatched_default(
        self, tmp_path, capsys
    ):
        """`apply_validation_to_groups` over the same two groupings, in memory.

        The new groups are put *first* so they sit at the indices the old
        records carry — the arrangement the index fallback used to misattribute.
        Every old group keeps its own verdict, every new group gets the default,
        and the count of the latter is printed. Zero foreign verdicts.
        """
        from utils import validation as validation_mod

        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"claude-code_sonnet": _distinct_suggestions(["Alpha", "Bravo", "Charlie"])},
            models_requested=["claude-code:sonnet", "codex:gpt-6"],
        )
        first = _validated_round(plan, out_dir, phase_dir, status="valid")
        old_ids = {g["group_id"] for g in first["groups"]}
        assert len(old_ids) == 3

        _regrouped_after_a_second_model(plan, out_dir, phase_dir, first)
        regrouped = _groups_on_disk(phase_dir)
        assert len(regrouped) == 5
        new_first = sorted(
            regrouped,
            key=lambda g: validation_mod.generate_group_id(g) in old_ids,
        )
        assert validation_mod.generate_group_id(new_first[0]) not in old_ids
        assert validation_mod.generate_group_id(new_first[1]) not in old_ids

        records = validation_mod.load_validation_results(phase_dir / "validation.json")
        capsys.readouterr()
        applied = validation_mod.apply_validation_to_groups(new_first, records)
        out = capsys.readouterr().out

        assert "[validation] 2 of 5 groups have no validation result" in out
        for group in applied:
            if validation_mod.generate_group_id(group) in old_ids:
                assert group["validation_status"] == "valid"
                assert group["validation_reason"].startswith("Confirmed against the diff")
            else:
                assert group["validation_status"] == "needs-human-decision"
                assert group["validation_reason"] == "No validation result"
                assert group["validation_confidence"] == 0.0
        assert sum(g["validation_status"] == "valid" for g in applied) == 3


class TestARegroupRetiresTheVerdictsItOrphans:
    def test_the_base_regroup_renames_validation_json_in_the_same_step(
        self, tmp_path, capsys
    ):
        """`reaggregate_from_existing_files`' own regroup branch — the one that
        rewrites `grouped.json` — retires the file it orphans."""
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"claude-code_sonnet": _distinct_suggestions(["Alpha", "Bravo", "Charlie"])},
            models_requested=["claude-code:sonnet"],
        )
        first = _validated_round(plan, out_dir, phase_dir, status="valid")
        stale_bytes = (phase_dir / "validation.json").read_bytes()
        # No task file, `--skip-validation`: nothing binds the grouping, so this
        # reaggregate regroups and rewrites `grouped.json`.
        (phase_dir / "validation_tasks.json").unlink()
        capsys.readouterr()

        asyncio.run(
            rob.reaggregate_from_existing_files(
                prefix="plan",
                out_dir=str(out_dir),
                phase_dir=str(phase_dir),
                phase_name="review-plan",
                plan_path=str(plan),
                args=_args(skip_validation=True),
                orchestrator_script="review_plan_orchestrator.py",
            )
        )
        out = capsys.readouterr().out

        assert "Grouped suggestions saved to" in out
        assert not (phase_dir / "validation.json").exists()
        stale = _stale_files(phase_dir)
        assert len(stale) == 1
        assert (phase_dir / stale[0]).read_bytes() == stale_bytes
        assert f"[VALIDATION_RETIRED] {stale[0]} (grouped.json rewritten by reaggregate)" in out
        assert first["groups"]  # the retired verdicts are still on disk, intact

    def test_the_code_review_fanout_renames_validation_json_when_it_regroups(
        self, tmp_path, capsys
    ):
        """The main path's write, driven end to end on `code_review_orchestrator`
        (`review_orchestrator_base`'s fan-out calls the same helper at the same
        step). A `validation.json` from a previous run is in the phase dir when
        the fan-out regroups; afterwards it is the `.stale.json` sibling."""
        phase_dir = cro.get_phase_dir(tmp_path / "auth-feature.md", "code-review")
        previous = {
            "format_version": 2,
            "groups": [{"group_index": 0, "group_id": "0123456789abcdef",
                        "status": "valid", "reason": "From an earlier run",
                        "confidence": 0.9}],
            "metadata": {"model": "unknown", "schema_version": "2.1",
                         "grouped_sha256": "feedfacecafebeef"},
        }
        stale_bytes = json.dumps(previous, indent=2).encode("utf-8")
        (phase_dir / "validation.json").write_bytes(stale_bytes)

        _run_cro_fanout(tmp_path, _two_model_findings())
        out = capsys.readouterr().out

        assert (phase_dir / "grouped.json").exists()
        assert not (phase_dir / "validation.json").exists()
        stale = _stale_files(phase_dir)
        assert len(stale) == 1
        assert (phase_dir / stale[0]).read_bytes() == stale_bytes
        assert f"[VALIDATION_RETIRED] {stale[0]} (grouped.json rewritten by a re-run)" in out

    def test_the_code_review_fanout_retires_the_previous_rounds_batch_files(
        self, tmp_path, capsys, monkeypatch
    ):
        """#25: `--force` regroups; a `validation_batch_<n>.json` from the previous
        round must leave with `validation.json`, or the executor skips it as
        already complete and the merge discards it as a foreign group_hash."""
        phase_dir = cro.get_phase_dir(tmp_path / "auth-feature.md", "code-review")
        (phase_dir / "validation_batch_0.json").write_text('{"groups": []}', encoding="utf-8")
        (phase_dir / "validation_tasks.json").write_text("{}", encoding="utf-8")
        # `--force`, as in the measured run: without it the partial-completion
        # guard refuses a dir that already carries a batch file.
        real_args = _fanout_args
        monkeypatch.setattr(
            sys.modules[__name__], "_fanout_args",
            lambda *a, **k: _forced(real_args(*a, **k)),
        )

        _run_cro_fanout(tmp_path, _two_model_findings())
        out = capsys.readouterr().out

        assert not (phase_dir / "validation_batch_0.json").exists()
        assert any(
            n.startswith("validation_batch_0.") and n.endswith(".stale.json")
            for n in os.listdir(phase_dir)
        )
        assert "[VALIDATION_RETIRED] validation_batch_0." in out


class TestAModelDroppedFromTheSelectionIsRetired:
    """#30: stale detection is keyed by slot, so a respelled model
    (`codex:gpt-6-luna` -> `codex:gpt-6-luna:high`) left its old result and log
    beside the new slot's, for the next `--reaggregate` to read back."""

    def _seed(self, phase_dir):
        (phase_dir / "codex_gpt-6-luna.json").write_text('[{"title": "old"}]', encoding="utf-8")
        (phase_dir / "log_codex_gpt-6-luna.txt").write_text("old log", encoding="utf-8")
        (phase_dir / "codex_gpt-6-luna_high.json").write_text("[]", encoding="utf-8")
        (phase_dir / "grouped.json").write_text('{"groups": []}', encoding="utf-8")
        (phase_dir / "weird.json").write_text('{"not": "a result"}', encoding="utf-8")

    def test_helper_retires_only_unselected_list_results_and_their_logs(self, tmp_path, capsys):
        self._seed(tmp_path)
        retired = rob.retire_unselected_results(
            str(tmp_path), ["codex:gpt-6-luna:high"]
        )
        assert [os.path.basename(r) for r in retired] == [
            "codex_gpt-6-luna.json.stale-unselected"]
        assert (tmp_path / "codex_gpt-6-luna.json.stale-unselected").exists()
        assert (tmp_path / "log_codex_gpt-6-luna.txt.stale-unselected").exists()
        assert not (tmp_path / "codex_gpt-6-luna.json").exists()
        # selected slot, bookkeeping and an unidentifiable file all stay.
        for kept in ("codex_gpt-6-luna_high.json", "grouped.json", "weird.json"):
            assert (tmp_path / kept).exists()
        assert "[SELECTION_CHANGED] codex_gpt-6-luna" in capsys.readouterr().out
        assert rob.retire_unselected_results(str(tmp_path), ["codex:gpt-6-luna:high"]) == []

    def test_the_code_review_fanout_retires_them(self, tmp_path, monkeypatch):
        phase_dir = cro.get_phase_dir(tmp_path / "auth-feature.md", "code-review")
        (phase_dir / "codex_gpt-6-luna.json").write_text("[]", encoding="utf-8")
        (phase_dir / "log_codex_gpt-6-luna.txt").write_text("x", encoding="utf-8")
        real_args = _fanout_args
        monkeypatch.setattr(
            sys.modules[__name__], "_fanout_args",
            lambda *a, **k: _forced(real_args(*a, **k)),
        )
        _run_cro_fanout(tmp_path, _two_model_findings())
        assert not (phase_dir / "codex_gpt-6-luna.json").exists()
        assert (phase_dir / "codex_gpt-6-luna.json.stale-unselected").exists()
        assert (phase_dir / "log_codex_gpt-6-luna.txt.stale-unselected").exists()


def _as_legacy_file(phase_dir, status):
    """Strip `metadata.grouped_sha256` and set every verdict to *status*.

    A pre-D7 `validation.json`: hashed records, no binding to any grouping —
    and (§7.1) none to the batch bytes either, which is what puts it on the
    legacy mtime test.
    """
    path = phase_dir / "validation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for group in data["groups"]:
        group["status"] = status
        group["reason"] = "From the legacy file"
    del data["metadata"]["grouped_sha256"]
    for key in ("batch_sha256", "batch_manifest", "runners"):
        data["metadata"].pop(key, None)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _set_mtime(path, seconds):
    import os
    os.utime(path, (seconds, seconds))


class TestALegacyFileIsHonouredOnlyIfNoBatchIsNewer:
    def _legacy_phase(self, tmp_path):
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"claude-code_sonnet": _distinct_suggestions(["Alpha", "Bravo", "Charlie"])},
            models_requested=["claude-code:sonnet"],
        )
        _validated_round(plan, out_dir, phase_dir, status="invalid")
        legacy = _as_legacy_file(phase_dir, status="valid")
        batches = sorted(phase_dir.glob("validation_batch_*.json"))
        assert batches
        base = 1_700_000_000
        for batch in batches:
            _set_mtime(batch, base)
        _set_mtime(legacy, base + 60)
        return plan, out_dir, phase_dir, batches

    def test_honoured_when_it_postdates_every_batch(self, tmp_path, capsys):
        plan, out_dir, phase_dir, _batches = self._legacy_phase(tmp_path)
        capsys.readouterr()

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert (
            "Using validation.json (legacy: no batch_sha256, mtime test) "
            "for report generation"
        ) in out
        assert "Not using validation.json" not in out
        assert _stale_files(phase_dir) == []
        # Left as it was: still key-less, still the legacy verdicts.
        data = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert "grouped_sha256" not in data["metadata"]
        assert _validation_summary(phase_dir) == (
            "**Validation:** 3 valid, 0 invalid, 0 needs human review, "
            "0 validation failed"
        )

    def test_not_honoured_when_a_batch_is_newer(self, tmp_path, capsys):
        plan, out_dir, phase_dir, batches = self._legacy_phase(tmp_path)
        newer = batches[0]
        _set_mtime(newer, 1_700_000_000 + 120)
        stale_bytes = (phase_dir / "validation.json").read_bytes()
        capsys.readouterr()

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "Found existing validation.json" not in out
        assert (
            "Not using validation.json: validation.json carries no batch_sha256 "
            f"and 1 batch file(s) are newer than it ({newer.name}) "
            "(legacy: no batch_sha256, mtime test)"
        ) in out
        stale = _stale_files(phase_dir)
        assert len(stale) == 1
        assert (phase_dir / stale[0]).read_bytes() == stale_bytes
        # Merged from the batches, and bound from now on.
        data = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert data["metadata"]["grouped_sha256"] == _grouped_hash(phase_dir)
        assert [g["status"] for g in data["groups"]] == ["invalid"] * 3
        assert _validation_summary(phase_dir) == (
            "**Validation:** 0 valid, 3 invalid, 0 needs human review, "
            "0 validation failed"
        )


# ---------------------------------------------------------------------------
# A result file that landed after grouping is folded in, not silently excluded
# ---------------------------------------------------------------------------
#
# The reuse branch above ("preserves validation alignment") presumes the bound
# `grouped.json` still describes the results on disk. The documented salvage
# flow falsifies that by construction — the fan-out groups without the failed
# model, materializes validation and returns at the marker; the salvage lands
# afterwards; `--reaggregate` found a bound grouping and reused it — and so does
# a result obtained out of band for a slot the fan-out could not run. Measured
# 2026-09-17 on a real `review-code --quick`: codex's three findings would have
# reached no group while `**Models:**` listed it, and folding them in took a
# hand-rolled rename of `validation.json`, `validation_tasks.json` and every
# `validation_batch_*.json` to `*.<stamp>.stale.json` plus two reaggregates.
# Reaggregation now detects the gap by content, retires the whole round under
# D7's naming, regroups, and re-materializes validation for the grouping that
# does cover the dir.

_LATE_SPEC = "codex:gpt-6-astra"
_LATE_STEM = "codex_gpt-6-astra"


def _late_issues():
    """One finding that joins the shared `Delta` group, one that is new."""
    return _unrelated_issues(["Delta", "Golf"])


def _completed_round_with_a_failed_slot(tmp_path):
    """A validated, reported review-code phase whose third slot failed.

    Fan-out (validation on) → batches completed → reaggregate merges them and
    writes the report. `.status.json` carries the third spec under
    `models_failed`, exactly as the sandbox-refused codex slot was recorded.
    """
    plan, phase_dir, _exit_code = _run_cro_fanout(
        tmp_path,
        _two_model_findings(),
        skip_validation=False,
        failed={_LATE_SPEC: "[SANDBOX_UNAVAILABLE] codex's sandbox cannot start here"},
    )
    _complete_validation_batches(phase_dir)
    _run_cro_reaggregate(plan, phase_dir, skip_validation=False)
    assert (phase_dir / "validation.json").exists()
    status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
    assert _LATE_SPEC in status["models_failed"]
    assert _LATE_SPEC not in status["models_completed"]
    return plan, phase_dir


def _retired_round_files(phase_dir):
    return sorted(p.name for p in phase_dir.glob("*.stale.json"))


def _stamp_of(stale_name):
    """`validation_batch_0.20260917-025330.stale.json` -> `20260917-025330`."""
    return stale_name[: -len(".stale.json")].rsplit(".", 1)[1]


class TestAResultFileThatLandedAfterGroupingIsFoldedIn:
    def test_the_bound_round_is_retired_and_the_dir_is_regrouped(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _completed_round_with_a_failed_slot(tmp_path)
        old_grouped = phase_dir / "grouped.json"
        old_hash = subagent_tasks_mod.file_content_hash(str(old_grouped))
        old_validation = (phase_dir / "validation.json").read_bytes()
        old_tasks = (phase_dir / "validation_tasks.json").read_bytes()

        # The result obtained out of band, written where the fan-out would
        # have written it.
        (phase_dir / f"{_LATE_STEM}.json").write_text(
            json.dumps(_late_issues()), encoding="utf-8"
        )
        capsys.readouterr()

        _run_cro_reaggregate(plan, phase_dir, skip_validation=False)
        out = capsys.readouterr().out

        # Named, by content, before anything is moved.
        assert "grouped.json does not cover the model results on disk" in out
        assert (
            f"  {_LATE_SPEC}: 2 issue(s) on disk not in grouped.json, "
            "0 in grouped.json not on disk"
        ) in out
        assert "Using existing grouped.json" not in out

        # The whole round retired under one stamp — renamed, byte-identical —
        # and nothing of it left under its live name.
        retired = _retired_round_files(phase_dir)
        assert [n.split(".")[0] for n in retired] == [
            "validation", "validation_batch_0", "validation_batch_1",
            "validation_tasks",
        ]
        assert len({_stamp_of(n) for n in retired}) == 1
        for name in retired:
            assert (
                f"[VALIDATION_RETIRED] {name} "
                "(grouped.json did not cover the results on disk)"
            ) in out
        assert (phase_dir / retired[0]).read_bytes() == old_validation
        assert (phase_dir / retired[-1]).read_bytes() == old_tasks
        assert not (phase_dir / "validation.json").exists()
        assert not [
            p for p in phase_dir.glob("validation_batch_*.json")
            if not p.name.endswith(".stale.json")
        ]

        # Regrouped over all three models: Delta now has three members, and
        # Golf is a group of its own.
        groups = _groups_on_disk(phase_dir)
        assert subagent_tasks_mod.file_content_hash(str(old_grouped)) != old_hash
        attributed = {m for g in groups for m in g["models"]}
        assert attributed == set(_REAL_SPECS) | {_LATE_SPEC}
        by_theme = {g["theme"]: g for g in groups}
        assert set(by_theme["Delta"]["models"]) == set(_REAL_SPECS) | {_LATE_SPEC}
        assert by_theme["Golf"]["models"] == [_LATE_SPEC]
        assert len(groups) == 7

        # A fresh round, bound to the grouping that covers the dir, handed to
        # the host at the marker the fan-out itself uses.
        tasks = json.loads(
            (phase_dir / "validation_tasks.json").read_text(encoding="utf-8")
        )
        assert tasks["inputs"]["grouped_sha256"] == (
            subagent_tasks_mod.file_content_hash(str(old_grouped))
        )
        assert "[VALIDATION_BATCHES_PENDING]" in out or "[VALIDATION_PENDING]" in out

    def test_the_checkpoint_records_the_late_model_as_completed(
        self, tmp_path, capsys
    ):
        """`.status.json` is what `status` and `build_unsanitize_map` read, and it
        used to keep the sandbox error under `models_failed` beside a report that
        listed the model."""
        plan, phase_dir = _completed_round_with_a_failed_slot(tmp_path)
        (phase_dir / f"{_LATE_STEM}.json").write_text(
            json.dumps(_late_issues()), encoding="utf-8"
        )
        capsys.readouterr()

        _run_cro_reaggregate(plan, phase_dir, skip_validation=False)
        out = capsys.readouterr().out

        assert (
            f"[{_LATE_SPEC}] result file present; recorded as completed (was failed)"
        ) in out
        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        assert status["models_completed"] == _REAL_SPECS + [_LATE_SPEC]
        assert status["models_failed"] == {}
        # `report_progress` gained a foreground write path, so this reaggregate
        # publishes a `progress` object of its own — the same reserved object,
        # in the same shape, a detached run writes, which is what lets one
        # renderer serve both. The invariant that survives is the narrower one:
        # a foreground `.status.json` gains `progress` and still no `run`
        # object, because only a detached worker's `claim_run` writes process
        # state. The detached leg's `models_done` follow-through is pinned in
        # `TestRecordModelsPresent`.
        #
        # The step is `validation_pending`, not `complete` (2026-09-21, the
        # finding `TestTheHostValidationHandOffIsTerminal` in
        # test_code_review_orchestrator.py pins: neither review-code hand-off
        # said the phase had ended). This reaggregate runs *with* validation, so
        # it writes the fresh round, hands it to the host at the marker its
        # sibling `test_the_bound_round_is_retired_and_the_dir_is_regrouped`
        # asserts on the same fixture (`[VALIDATION_BATCHES_PENDING]` /
        # `[VALIDATION_PENDING]`), and returns. A run whose validation the host
        # still owes is not complete, so `complete` was exactly the inaccurate
        # word that finding removed. `validation_pending` IS in
        # `run_status.CHECKPOINT_TERMINAL_STEPS`, so the checkpoint still reads
        # settled rather than in flight — which is the whole point of the fix.
        progress = status.get("progress") or {}
        assert progress.get("step") == "validation_pending"
        assert progress.get("models_total") == len(status["models_requested"])
        assert progress.get("models_done") == progress["models_total"]
        assert set(progress) <= _RESERVED_PROGRESS_KEYS, progress
        assert "run" not in status, (
            "a foreground reaggregate claims no run; `progress` is the only "
            "reserved object it writes"
        )

    def test_the_documented_flow_then_completes_with_every_model_in_the_report(
        self, tmp_path, capsys
    ):
        """Validate the fresh round, reaggregate: the report's numbers reconcile."""
        plan, phase_dir = _completed_round_with_a_failed_slot(tmp_path)
        (phase_dir / f"{_LATE_STEM}.json").write_text(
            json.dumps(_late_issues()), encoding="utf-8"
        )
        _run_cro_reaggregate(plan, phase_dir, skip_validation=False)
        _complete_validation_batches(phase_dir)
        capsys.readouterr()

        _run_cro_reaggregate(
            plan, phase_dir, skip_validation=False, render_report=True
        )
        out = capsys.readouterr().out

        assert "does not cover" not in out
        assert "Using existing grouped.json" in out
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        models_line = next(
            ln for ln in report.splitlines() if ln.startswith("**Models:**")
        )
        assert _LATE_SPEC in models_line
        assert "**Total Issues:** 9" in report
        assert "**Groups:** 7 " in report
        assert (
            "**Validation:** 7 valid, 0 invalid, 0 needs human review, "
            "0 validation failed (of 7 groups)"
        ) in report
        data = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert data["metadata"]["grouped_sha256"] == _grouped_hash(phase_dir)

    def test_a_round_that_covers_the_dir_is_never_retired(self, tmp_path, capsys):
        """The regression guard on the check: the plain validate-then-reaggregate
        round trip moves nothing and prints no retirement."""
        plan, phase_dir, _exit_code = _fanout_with_validation(
            tmp_path, _two_model_findings()
        )
        _complete_validation_batches(phase_dir)
        capsys.readouterr()

        _run_cro_reaggregate(plan, phase_dir, skip_validation=False)
        out = capsys.readouterr().out

        assert "[VALIDATION_RETIRED]" not in out
        assert "does not cover" not in out
        assert _retired_round_files(phase_dir) == []

    def test_the_base_lane_folds_a_late_model_in_the_same_way(
        self, tmp_path, capsys
    ):
        """review-plan/review-tasks reuse the bound grouping through the same
        branch, and the salvage flow lands the late file there the same way."""
        plan, out_dir, phase_dir = _phase(
            tmp_path,
            {"claude-code_sonnet": _distinct_suggestions(["Alpha", "Bravo", "Charlie"])},
            models_requested=["claude-code:sonnet", "codex:gpt-6"],
        )
        first = _validated_round(plan, out_dir, phase_dir, status="valid")
        assert first["metadata"]["grouped_sha256"] == _grouped_hash(phase_dir)
        (phase_dir / "codex_gpt-6.json").write_text(
            json.dumps(_distinct_suggestions(["Delta", "Echo"])), encoding="utf-8"
        )
        capsys.readouterr()

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "grouped.json does not cover the model results on disk" in out
        assert (
            "  codex:gpt-6: 2 issue(s) on disk not in grouped.json, "
            "0 in grouped.json not on disk"
        ) in out
        retired = _retired_round_files(phase_dir)
        assert [n.split(".")[0] for n in retired] == [
            "validation", "validation_batch_0", "validation_tasks",
        ]
        assert len(_groups_on_disk(phase_dir)) == 5
        assert _grouped_hash(phase_dir) != first["metadata"]["grouped_sha256"]
        tasks = json.loads(
            (phase_dir / "validation_tasks.json").read_text(encoding="utf-8")
        )
        assert tasks["inputs"]["grouped_sha256"] == _grouped_hash(phase_dir)
        assert "[VALIDATION_BATCHES_PENDING]" in out or "[VALIDATION_PENDING]" in out

        # And the round then completes: verdicts for all five, bound to the
        # grouping they were made against.
        _complete_validation_batches(phase_dir, status="invalid")
        _reaggregate(plan, out_dir, phase_dir)
        merged = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert merged["metadata"]["grouped_sha256"] == _grouped_hash(phase_dir)
        assert [g["status"] for g in merged["groups"]] == ["invalid"] * 5
        assert _validation_summary(phase_dir) == (
            "**Validation:** 0 valid, 5 invalid, 0 needs human review, "
            "0 validation failed"
        )


class TestGroupedCoverageGap:
    """The pure comparison behind the branch, one case per way it must not lie."""

    def _grouped(self, issues_by_model):
        from utils import group_similar_suggestions
        from utils.state_manager import stamp_stable_ids

        flat = []
        for model, issues in issues_by_model.items():
            for issue in issues:
                flat.append({**issue, "source_model": model, "model": model})
        grouped = [g.to_dict() for g in group_similar_suggestions(flat)]
        stamp_stable_ids(grouped)
        return grouped

    def test_agreement_is_no_gap(self):
        findings = _two_model_findings()
        assert rob.grouped_coverage_gap(self._grouped(findings), findings) == []

    def test_a_model_named_by_stem_on_disk_still_agrees(self):
        """No checkpoint: `resolve_result_files_by_model` names the model by
        its stem while `grouped.json` carries the spec. Same model."""
        findings = _two_model_findings()
        by_stem = {cro.sanitize_model_name(m): v for m, v in findings.items()}
        assert rob.grouped_coverage_gap(self._grouped(findings), by_stem) == []

    def test_a_key_a_later_release_stamped_is_not_a_gap(self):
        findings = _two_model_findings()
        grouped = self._grouped(findings)
        drifted = {
            m: [{**issue, "serializer_drift": "added later"} for issue in issues]
            for m, issues in findings.items()
        }
        assert rob.grouped_coverage_gap(grouped, drifted) == []

    def test_a_model_with_no_group_is_named_with_its_count(self):
        findings = _two_model_findings()
        on_disk = {**findings, _LATE_SPEC: _late_issues()}
        assert rob.grouped_coverage_gap(self._grouped(findings), on_disk) == [
            f"{_LATE_SPEC}: 2 issue(s) on disk not in grouped.json, "
            "0 in grouped.json not on disk"
        ]

    def test_a_file_that_lost_or_changed_an_issue_is_named_too(self):
        findings = _two_model_findings()
        grouped = self._grouped(findings)
        edited = {m: list(issues) for m, issues in findings.items()}
        edited[_REAL_SPECS[0]] = edited[_REAL_SPECS[0]][:-1]
        edited[_REAL_SPECS[1]] = [
            {**edited[_REAL_SPECS[1]][0], "desc": "a rewritten description"},
            *edited[_REAL_SPECS[1]][1:],
        ]
        assert rob.grouped_coverage_gap(grouped, edited) == [
            f"{_REAL_SPECS[0]}: 0 issue(s) on disk not in grouped.json, "
            "1 in grouped.json not on disk",
            f"{_REAL_SPECS[1]}: 1 issue(s) on disk not in grouped.json, "
            "1 in grouped.json not on disk",
        ]

    def test_a_grouping_with_no_attribution_is_inconclusive(self):
        """Older than anything this can reason about: keep today's behaviour."""
        findings = _two_model_findings()
        grouped = self._grouped(findings)
        for group in grouped:
            for suggestion in group["suggestions"]:
                suggestion.pop("source_model")
                suggestion.pop("model")
        on_disk = {**findings, _LATE_SPEC: _late_issues()}
        assert rob.grouped_coverage_gap(grouped, on_disk) == []


class TestRetireStaleValidationRound:
    def test_renames_every_artifact_of_the_round_under_one_stamp(
        self, tmp_path, capsys
    ):
        for name in (
            "validation.json", "validation_tasks.json",
            "validation_batch_0.json", "validation_batch_1.json",
            "validation_batch_10.json",
        ):
            (tmp_path / name).write_text(name, encoding="utf-8")
        # An earlier retirement, which must be neither re-retired nor clobbered.
        (tmp_path / "validation_batch_0.20260101-000000.stale.json").write_text(
            "older", encoding="utf-8"
        )

        retired = rob.retire_stale_validation_round(str(tmp_path), reason="why")
        out = capsys.readouterr().out

        names = [Path(p).name for p in retired]
        assert [n.split(".")[0] for n in names] == [
            "validation", "validation_tasks",
            "validation_batch_0", "validation_batch_1", "validation_batch_10",
        ]
        assert len({_stamp_of(n) for n in names}) == 1
        for name in names:
            assert name.endswith(".stale.json")
            assert f"[VALIDATION_RETIRED] {name} (why)" in out
        # Content travels with the rename.
        assert (tmp_path / names[0]).read_text(encoding="utf-8") == "validation.json"
        assert (tmp_path / names[-1]).read_text(encoding="utf-8") == (
            "validation_batch_10.json"
        )
        assert (
            tmp_path / "validation_batch_0.20260101-000000.stale.json"
        ).read_text(encoding="utf-8") == "older"
        live = [p.name for p in tmp_path.glob("*.json") if not p.name.endswith(".stale.json")]
        assert live == []


    def test_retire_round_takes_the_sidecars_with_it(self, tmp_path, capsys):
        """A prompt sidecar belongs to the grouping its batch does.

        Left behind, `validation_batch_0.prompt.txt` would sit beside the fresh
        round describing the retired one, and the only reader that ever finds it
        is a host following `prompt_path` — which would hand a subagent the old
        round's prompt with nothing to say otherwise. It retires under the same
        stamp, keeping its real extension: `.txt` stays `.txt`, because
        `<stem>.<stamp>.stale.json` on a text file would be a lie an operator
        (and any `*.json` glob) reads as a JSON reply.
        """
        for name in (
            "validation.json", "validation_tasks.json",
            "validation_batch_0.json", "validation_batch_1.json",
            "validation_batch_0.prompt.txt", "validation_batch_1.prompt.txt",
        ):
            (tmp_path / name).write_text(name, encoding="utf-8")

        retired = rob.retire_stale_validation_round(str(tmp_path), reason="regrouped")
        out = capsys.readouterr().out

        names = [Path(p).name for p in retired]
        stamps = {n[: -len(".stale.json")].rsplit(".", 1)[1] for n in names[:4]}
        stamps |= {n[: -len(".stale.txt")].rsplit(".", 1)[1] for n in names[4:]}
        assert len(stamps) == 1, f"one round, one stamp: {names}"
        stamp = stamps.pop()

        # Order: merged file, task file, batch replies, then the sidecars.
        assert names == [
            f"validation.{stamp}.stale.json",
            f"validation_tasks.{stamp}.stale.json",
            f"validation_batch_0.{stamp}.stale.json",
            f"validation_batch_1.{stamp}.stale.json",
            f"validation_batch_0.prompt.{stamp}.stale.txt",
            f"validation_batch_1.prompt.{stamp}.stale.txt",
        ]
        for name in names:
            assert f"[VALIDATION_RETIRED] {name} (regrouped)" in out
        # Content travels with every rename, and nothing live is left behind.
        for original, name in zip(
            (
                "validation.json", "validation_tasks.json",
                "validation_batch_0.json", "validation_batch_1.json",
                "validation_batch_0.prompt.txt", "validation_batch_1.prompt.txt",
            ),
            names,
        ):
            assert (tmp_path / name).read_text(encoding="utf-8") == original
            assert not (tmp_path / original).exists()
        assert list(tmp_path.glob("validation_batch_*.prompt.txt")) == []
        # The sidecars keep their own extension rather than wearing the batch
        # files': a `.txt` named `.stale.json` would be read back as a reply.
        assert sorted(p.name for p in tmp_path.glob("*.stale.txt")) == sorted(names[4:])
        assert sorted(p.name for p in tmp_path.glob("*.stale.json")) == sorted(names[:4])

    def test_nothing_on_disk_is_nothing_retired(self, tmp_path, capsys):
        assert rob.retire_stale_validation_round(str(tmp_path), reason="why") == []
        assert capsys.readouterr().out == ""

    def test_the_single_file_helper_still_retires_only_validation_json(self, tmp_path):
        (tmp_path / "validation.json").write_text("v", encoding="utf-8")
        (tmp_path / "validation_tasks.json").write_text("t", encoding="utf-8")

        target = rob.retire_stale_validation(str(tmp_path), reason="why")

        assert Path(target).name.startswith("validation.")
        assert target.endswith(".stale.json")
        assert (tmp_path / "validation_tasks.json").exists()


class TestARetiredBatchIsNotValidationInProgress:
    def test_the_legacy_mtime_check_ignores_a_retired_batch(self, tmp_path):
        """`stale_validation_reason`'s legacy leg counts batch files newer than
        a key-less `validation.json`; a retired one keeps its mtime and belongs
        to no round that file could have come from."""
        from utils import validation as validation_mod

        grouped = tmp_path / "grouped.json"
        grouped.write_text("[]", encoding="utf-8")
        legacy = tmp_path / "validation.json"
        legacy.write_text(json.dumps({"groups": [], "metadata": {}}), encoding="utf-8")
        retired = tmp_path / "validation_batch_0.20260917-025330.stale.json"
        retired.write_text("{}", encoding="utf-8")
        _set_mtime(legacy, 1_700_000_000)
        _set_mtime(retired, 1_700_000_000 + 120)

        assert validation_mod.stale_validation_reason(legacy, grouped) is None


def _terminal(state, **over):
    """One frozen `progress.models` entry, as the fan-out's sweep left it."""
    entry = {
        "state": state,
        "started_at": "2026-09-17T10:00:00Z",
        "updated_at": "2026-09-17T10:04:00Z",
        "ended_at": "2026-09-17T10:04:00Z",
        "elapsed_s": 240,
        "timeout_s": 1800,
        "attempt": 1,
        "source": "file" if state == "done" else None,
    }
    if state == "failed":
        entry["error"] = "c exited with code 1"
    entry.update(over)
    return entry


class TestRecordModelsPresent:
    def _checkpoint(self, tmp_path, **over):
        data = {
            "phase": "code-review",
            "state": "complete",
            "models_requested": ["a:x", "b:y", "c:z"],
            "models_completed": ["a:x", "b:y"],
            "models_failed": {"c:z": "c exited with code 1"},
            "progress": {
                "models_total": 3,
                "models_done": 3,
                "models_succeeded": 2,
                "step": "complete",
                "models": {
                    "a:x": _terminal("done"),
                    "b:y": _terminal("done"),
                    "c:z": _terminal("failed"),
                },
            },
        }
        data.update(over)
        write_status(str(tmp_path), data)
        return tmp_path

    def test_a_present_failed_model_moves_to_completed(self, tmp_path):
        phase_dir = self._checkpoint(tmp_path)

        moved = rob.record_models_present(str(phase_dir), ["a:x", "b:y", "c:z"])

        assert moved == ["c:z"]
        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        assert status["models_completed"] == ["a:x", "b:y", "c:z"]
        assert status["models_failed"] == {}
        progress = status["progress"]
        # `models_succeeded`, NOT `models_done`. This writer counts completed
        # RESULTS; `models_done` counts finished slots, and the salvaged model's
        # slot finished long ago — as `failed`. So the terminal count is left
        # exactly where the fan-out left it and only the success count moves.
        assert progress["models_succeeded"] == 3
        assert progress["models_done"] == 3
        assert progress["step"] == "complete"
        # The map is flipped in the same write, so a reader never meets a model
        # the report calls completed while the table still calls it failed.
        recovered = progress["models"]["c:z"]
        assert recovered["state"] == "done"
        assert recovered["source"] == "existing_file"
        assert "error" not in recovered
        # Frozen: the slot really did end when it ended, and this is a later
        # reading of the same slot's outcome, not a second run of it.
        assert recovered["ended_at"] == "2026-09-17T10:04:00Z"
        assert recovered["elapsed_s"] == 240
        # The two it did not touch are byte-identical.
        assert progress["models"]["a:x"] == _terminal("done")

    def test_a_completed_model_whose_file_is_gone_keeps_its_entry(self, tmp_path):
        phase_dir = self._checkpoint(tmp_path)

        assert rob.record_models_present(str(phase_dir), ["a:x"]) == []

        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        assert status["models_completed"] == ["a:x", "b:y"]
        assert status["models_failed"] == {"c:z": "c exited with code 1"}
        assert status["progress"]["models_done"] == 3
        assert status["progress"]["models_succeeded"] == 2
        assert status["progress"]["models"]["c:z"]["state"] == "failed"

    def test_a_checkpoint_without_outcomes_gains_nothing(self, tmp_path):
        write_status(str(tmp_path), {"phase": "code-review", "state": "complete"})
        before = (tmp_path / ".status.json").read_text(encoding="utf-8")

        assert rob.record_models_present(str(tmp_path), ["a:x"]) == []

        assert (tmp_path / ".status.json").read_text(encoding="utf-8") == before

    def test_no_checkpoint_at_all_writes_none(self, tmp_path):
        assert rob.record_models_present(str(tmp_path), ["a:x"]) == []
        assert not (tmp_path / ".status.json").exists()


class TestASalvagedModelIsNotRenderedAsSkipped:
    """2026-09-21: `flip_map_entries_to_done` serves two events and only one of
    them is "this slot never dispatched".

    `record_models_present` (the `--reaggregate` path) stamps `existing_file`,
    which is in `run_status.SKIPPED_SOURCES` / `progress_tracker.RESUME_SOURCES`,
    so `_model_state_label` renders that row **skipped** — true there, the slot
    really never ran on this run. The helper stamped the same word for
    `record_models_salvaged`, whose slot dispatched, ran, failed to parse and was
    repaired in-process, so every salvaged model was announced as skipped in the
    `models:` table and on `just finished:` — the opposite of what happened.
    The source word is now the caller's to supply, and salvage supplies one that
    is in neither set.
    """

    SPEC = "c:z"
    OTHER = "a:x"

    def _checkpoint(self, tmp_path, *, still_failed):
        """The checkpoint each event actually meets.

        Salvage runs *after* the fan-out's own `del failed_models[spec]`, so the
        report side already reads completed and only the map is behind. A
        reaggregate meets the model still under `models_failed`.
        """
        write_status(str(tmp_path), {
            "phase": "code-review",
            "state": "complete",
            "models_requested": [self.OTHER, self.SPEC],
            "models_completed": (
                [self.OTHER] if still_failed else [self.OTHER, self.SPEC]
            ),
            "models_failed": (
                {self.SPEC: "c exited with code 1"} if still_failed else {}
            ),
            "progress": {
                "models_total": 2,
                "models_done": 2,
                "models_succeeded": 1 if still_failed else 2,
                "step": "complete",
                "models": {
                    self.OTHER: _terminal("done"),
                    self.SPEC: _terminal("failed"),
                },
            },
        })
        return tmp_path

    def _progress(self, phase_dir):
        return json.loads(
            (phase_dir / ".status.json").read_text(encoding="utf-8")
        )["progress"]

    def _view(self, phase_dir):
        return {"phase_dir": str(phase_dir), "progress": self._progress(phase_dir)}

    def _row(self, lines, spec):
        rows = [line for line in lines if line.strip().startswith(f"{spec} ")]
        assert len(rows) == 1, lines
        return rows[0]

    def test_salvage_stamps_a_source_that_is_in_neither_skip_set(self, tmp_path):
        phase_dir = self._checkpoint(tmp_path, still_failed=False)

        rob.record_models_salvaged(str(phase_dir), [self.SPEC])

        entry = self._progress(phase_dir)["models"][self.SPEC]
        assert entry["state"] == "done", entry
        assert entry["source"] == "salvage", entry
        assert entry["source"] not in run_status.SKIPPED_SOURCES
        assert entry["source"] not in progress_tracker.RESUME_SOURCES
        # Everything the shared helper already guaranteed still holds: the
        # pre-salvage error is gone and the freeze is not re-stamped.
        assert "error" not in entry, entry
        assert entry["ended_at"] == "2026-09-17T10:04:00Z"
        assert entry["elapsed_s"] == 240

    def test_the_models_table_calls_a_salvaged_model_done(self, tmp_path):
        phase_dir = self._checkpoint(tmp_path, still_failed=False)

        rob.record_models_salvaged(str(phase_dir), [self.SPEC])

        view = self._view(phase_dir)
        assert run_status._model_state_label(
            view["progress"]["models"][self.SPEC]
        ) == "done"
        row = self._row(run_status._model_table_lines(view), self.SPEC)
        assert "done" in row, row
        assert "skipped" not in row, row

    def test_just_finished_calls_a_salvaged_model_done(self, tmp_path):
        phase_dir = self._checkpoint(tmp_path, still_failed=False)

        rob.record_models_salvaged(str(phase_dir), [self.SPEC])

        view = self._view(phase_dir)
        # An empty baseline for this dir: every terminal row in it went terminal
        # during the wait, which is what a poll that opened before the salvage
        # landed sees.
        run_status._attach_finished_during_wait({str(phase_dir): {}}, [view])
        text = run_status._just_finished_text(view)
        assert self.SPEC in text, text
        assert "skipped" not in text, text
        assert f"{self.SPEC} (" in text and "done" in text, text

    def test_reaggregate_still_stamps_existing_file_and_still_reads_skipped(
        self, tmp_path
    ):
        """The other caller is deliberately unchanged: that slot never ran."""
        phase_dir = self._checkpoint(tmp_path, still_failed=True)

        assert rob.record_models_present(str(phase_dir), [self.OTHER, self.SPEC]) == [
            self.SPEC
        ]

        view = self._view(phase_dir)
        entry = view["progress"]["models"][self.SPEC]
        assert entry["state"] == "done", entry
        assert entry["source"] == "existing_file", entry
        assert entry["source"] in run_status.SKIPPED_SOURCES
        assert entry["source"] in progress_tracker.RESUME_SOURCES
        assert run_status._model_state_label(entry) == "skipped"
        assert "skipped" in self._row(
            run_status._model_table_lines(view), self.SPEC
        )


class TestFlippingALiveRowStopsItsClock:
    """2026-09-21: the flip froze the verdict and left the clock running.

    ``entry_elapsed_s`` decides at *read* time which branch a row is on, and
    the discriminator is ``ended_at``: present -> the stored ``elapsed_s``,
    absent -> ``now - started_at`` recomputed on every read and clamped only at
    the slot's own budget. ``flip_map_entries_to_done`` stamped neither, which
    is right for the entry it was written for (a ``failed`` row, already frozen
    by the fan-out's own terminal transition) and wrong for the one it can also
    meet: a ``--reaggregate`` over a checkpoint whose map still holds a live
    row — a foreground run killed with no close-out reaches neither the
    fan-out's ``finally`` sweep nor ``retire_live_entries`` — flipped it to
    ``done`` with no ending at all, so the table rendered a finished model
    whose elapsed kept climbing until it pinned at the budget and stayed there.
    """

    SPEC = "c:z"
    OTHER = "a:x"

    def _live(self, state, *, ago_s, timeout_s=1800):
        """A row as a killed fan-out leaves it: live, with no ending."""
        started = (datetime.now() - timedelta(seconds=ago_s)).isoformat()
        return {
            "state": state,
            "started_at": started,
            "updated_at": started,
            "timeout_s": timeout_s,
            "attempt": 1,
            "source": None,
        }

    def test_a_live_row_is_given_an_ending_when_it_is_flipped(self):
        models_map = {self.SPEC: self._live("running", ago_s=90)}

        assert rob.flip_map_entries_to_done(models_map, [self.SPEC]) is True

        entry = models_map[self.SPEC]
        assert entry["state"] == "done", entry
        assert entry["ended_at"], entry
        frozen = entry["elapsed_s"]
        assert 85 <= frozen <= 150, entry
        # The read-time rule now takes its frozen branch, so the row reads the
        # same duration however long afterwards the table is rendered.
        assert progress_tracker.entry_elapsed_s(entry) == frozen
        assert progress_tracker.entry_elapsed_s(entry, time.time() + 3600) == frozen

    def test_a_queued_row_never_dispatched_is_closed_out_too(self):
        """The other live state a killed run leaves behind.

        Ten hours against a 30-minute budget is what the pre-fix row rendered
        forever: the live branch clamped at ``timeout_s``, so the clock stopped
        moving but the number was the budget rather than anything the slot did.
        It used to freeze at that budget (1800). A slot that never dispatched
        ran for no time at all, so it now freezes ``elapsed_s: 0`` — a bare
        int, like every other duration in the checkpoint — and **no**
        ``queued_s``: a ``--reaggregate`` can run hours after the kill, so "now"
        is not when the wait ended and its length is unknown (plan
        review-run-scope-salvage-and-failure-visibility §6.2). The rule reads
        ``state``, so this row — which, like a pre-upgrade map's, carries no
        ``dispatched_at`` — lands at 0 too.
        """
        models_map = {self.SPEC: self._live("queued", ago_s=10 * 3600)}

        rob.flip_map_entries_to_done(models_map, [self.SPEC])

        entry = models_map[self.SPEC]
        assert entry["elapsed_s"] == 0, entry
        assert isinstance(entry["elapsed_s"], int), entry
        assert "queued_s" not in entry, entry
        assert entry["ended_at"], entry
        assert progress_tracker.entry_elapsed_s(entry, time.time() + 86400) == 0
        assert progress_tracker.entry_queued_s(entry, time.time() + 86400) is None

    def test_a_dispatched_row_freezes_its_wait_from_the_two_exact_stamps(self):
        """Both stamps are exact, so the wait is ``dispatched_at - started_at``
        whenever the reaggregate runs; the run time counts from the dispatch."""
        entry = self._live("running", ago_s=300)
        started = datetime.fromisoformat(entry["started_at"])
        entry["dispatched_at"] = (started + timedelta(seconds=120)).isoformat()
        models_map = {self.SPEC: entry}

        rob.flip_map_entries_to_done(models_map, [self.SPEC])

        assert entry["state"] == "done", entry
        assert entry["queued_s"] == 120, entry
        # 300 s since admission, 120 of them queued: about 180 s of run time,
        # not the 300 the clock read when it counted from admission.
        assert 175 <= entry["elapsed_s"] <= 240, entry
        later = time.time() + 3600
        assert progress_tracker.entry_elapsed_s(entry, later) == entry["elapsed_s"]
        assert progress_tracker.entry_queued_s(entry, later) == 120

    def test_a_dispatched_row_keeps_the_wait_its_fanout_measured(self):
        """The monotonic figure the first dispatch stored is the same quantity,
        measured more exactly, so it is what freezes."""
        entry = self._live("running", ago_s=300)
        started = datetime.fromisoformat(entry["started_at"])
        entry["dispatched_at"] = (started + timedelta(seconds=120)).isoformat()
        entry["queued_s"] = 119.6
        models_map = {self.SPEC: entry}

        rob.flip_map_entries_to_done(models_map, [self.SPEC])

        assert entry["queued_s"] == 119.6, entry

    def test_an_already_terminal_row_keeps_the_ending_the_fanout_gave_it(self):
        """The freeze rule the helper was written for is untouched."""
        models_map = {
            self.SPEC: _terminal("failed"),
            # A slot an abort retired before it ever dispatched: no run time,
            # its whole wait in `queued_s` (plan review-run-scope-salvage-and-
            # failure-visibility §6.3). A later flip changes the verdict only.
            self.OTHER: _terminal("failed", elapsed_s=0, queued_s=300,
                                  error="cancelled"),
        }

        rob.flip_map_entries_to_done(models_map, [self.SPEC, self.OTHER])

        entry = models_map[self.SPEC]
        assert entry["state"] == "done", entry
        assert entry["ended_at"] == "2026-09-17T10:04:00Z", entry
        assert entry["elapsed_s"] == 240, entry
        assert "queued_s" not in entry, entry

        retired = models_map[self.OTHER]
        assert retired["state"] == "done", retired
        assert retired["ended_at"] == "2026-09-17T10:04:00Z", retired
        assert (retired["elapsed_s"], retired["queued_s"]) == (0, 300), retired

    def test_a_reaggregate_over_an_unclosed_map_writes_a_finished_row(
        self, tmp_path
    ):
        """End to end through the caller that meets this map in the wild."""
        write_status(str(tmp_path), {
            "phase": "code-review",
            "state": "running",
            "models_requested": [self.OTHER, self.SPEC],
            "models_completed": [self.OTHER],
            "models_failed": {},
            "progress": {
                "models_total": 2,
                "models_done": 1,
                "models_succeeded": 1,
                "step": "fanout",
                "models": {
                    self.OTHER: _terminal("done"),
                    self.SPEC: self._live("running", ago_s=300),
                },
            },
        })

        # Nothing is "recovered" here — a run killed mid-fan-out never wrote
        # this model under `models_failed`, so the return is empty and the
        # whole repair is the checkpoint it leaves behind.
        assert rob.record_models_present(str(tmp_path), [self.OTHER, self.SPEC]) == []

        status = json.loads((tmp_path / ".status.json").read_text(encoding="utf-8"))
        assert status["models_completed"] == [self.OTHER, self.SPEC]
        progress = status["progress"]
        entry = progress["models"][self.SPEC]
        assert entry["state"] == "done", entry
        assert entry["ended_at"], entry
        assert progress_tracker.entry_elapsed_s(entry, time.time() + 7200) == (
            entry["elapsed_s"]
        )
        # And the row renders as a finished one rather than a running clock.
        view = {"phase_dir": str(tmp_path), "progress": progress}
        rows = [
            ln for ln in run_status._model_table_lines(view)
            if ln.strip().startswith(f"{self.SPEC} ")
        ]
        assert len(rows) == 1, rows
        assert "running" not in rows[0], rows


# ---------------------------------------------------------------------------
# `**Files Changed:**` on a reaggregated report counts the same files the
# fan-out counted
# ---------------------------------------------------------------------------
#
# The fan-out resolves its file list with a fallback — `tracked_files` empty
# (no implement phase behind the plan) → `git diff` from the base ref, minus the
# `pre_existing_changes` baseline — and printed `Files to review: 144` on the
# 2026-09-17 run. The reaggregate path read `tracked_files` alone, so every
# `report.md` that run's reaggregates wrote said `**Files Changed:** 0` over an
# empty `## Changed Files` section. Nothing persists the list between the two
# entry points; both now resolve it through `resolve_changed_files`.


def _state_without_tracked_files(pre_existing=()):
    fake_state = MagicMock()
    fake_state.get = MagicMock(
        side_effect=lambda k, default=None: {
            "tracked_files": [],
            "head_at_start": "abc123",
            "pre_existing_changes": list(pre_existing),
        }.get(k, default)
    )
    return fake_state


def _rendered_report(plan):
    """`report.md` where the code-review reaggregate writes it: the plan-derived
    phase dir (`get_output_paths`), not the fixture's patched `get_phase_dir`."""
    path = cro.get_output_paths(plan, "code_review", phase="code-review").with_suffix(".md")
    return path.read_text(encoding="utf-8")


class TestReaggregateCountsTheFilesTheFanOutCounted:
    def test_a_plan_with_no_implement_phase_falls_back_to_the_diff(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _cro_phase(
            tmp_path,
            {cro.sanitize_model_name(_REAL_SPECS[0]): _issues(["Issue A"])},
            models_requested=[_REAL_SPECS[0]],
        )
        diffed = ["apps/web/a.tsx", "apps/web/b.tsx", "plans/x.md"]
        capsys.readouterr()

        _run_cro_reaggregate(
            plan,
            phase_dir,
            render_report=True,
            extra_patches=[
                patch.object(
                    cro, "get_or_create_state",
                    return_value=_state_without_tracked_files(pre_existing=["plans/"]),
                ),
                patch.object(cro, "get_files_changed_since_ref", return_value=diffed),
            ],
        )
        out = capsys.readouterr().out

        assert "No tracked files in state, falling back to git diff from abc123" in out
        assert "Filtered out 1 pre-existing files (not part of implementation)" in out
        assert "Files to review: 2" in out
        report = _rendered_report(plan)
        assert "**Files Changed:** 2" in report
        assert "- `apps/web/a.tsx`" in report
        assert "- `apps/web/b.tsx`" in report
        assert "plans/x.md" not in report

    def test_tracked_files_still_win_when_the_implement_phase_recorded_them(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _cro_phase(
            tmp_path,
            {cro.sanitize_model_name(_REAL_SPECS[0]): _issues(["Issue A"])},
            models_requested=[_REAL_SPECS[0]],
        )
        capsys.readouterr()

        _run_cro_reaggregate(
            plan,
            phase_dir,
            render_report=True,
            extra_patches=[
                patch.object(cro, "get_files_changed_since_ref", side_effect=AssertionError(
                    "the diff must not be consulted when tracked_files is populated"
                )),
            ],
        )
        out = capsys.readouterr().out

        assert "Using 1 tracked files from implementation phase" in out
        report = _rendered_report(plan)
        assert "**Files Changed:** 1" in report
        assert "- `src/a.py`" in report


class TestResolveChangedFiles:
    def test_tracked_entries_are_deduplicated_paths(self, capsys):
        state = MagicMock()
        state.get = MagicMock(side_effect=lambda k, d=None: {
            "tracked_files": [{"path": "a.py"}, {"path": "b.py"}, {"path": "a.py"}, {}],
        }.get(k, d))

        assert sorted(cro.resolve_changed_files(state, "abc123")) == ["a.py", "b.py"]
        assert "Using 4 tracked files from implementation phase" in capsys.readouterr().out

    def test_no_tracked_files_and_no_base_ref_is_an_empty_list_with_a_warning(
        self, capsys
    ):
        assert cro.resolve_changed_files(_state_without_tracked_files(), "") == []
        assert "cannot discover changed files via git diff" in capsys.readouterr().err

    def test_the_diff_minus_the_baseline_otherwise(self, capsys):
        with patch.object(
            cro, "get_files_changed_since_ref", return_value=["x.py", "plans/p.md", "y.py"]
        ):
            files = cro.resolve_changed_files(
                _state_without_tracked_files(pre_existing=["plans/"]), "abc123"
            )
        assert files == ["x.py", "y.py"]
        assert "Filtered out 1 pre-existing files" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# `resolve_review_scope`: the same resolution, labelled by the branch that
# picked the paths (review-scope plan §2 D2.1, T2.1)
# ---------------------------------------------------------------------------


def _dict_state(values):
    """A state stand-in answering `.get` from *values* (absent keys → default)."""
    state = MagicMock()
    state.get = MagicMock(side_effect=lambda k, d=None: values.get(k, d))
    return state


class TestResolveReviewScopeLabelsItsBranch:
    def test_the_tracked_files_branch_carries_each_entrys_task_id(self, capsys):
        state = _dict_state({
            "tracked_files": [
                {"path": "a.py", "task_id": "T001"},
                {"path": "b.py", "task_id": "implementation"},
                {"path": "a.py", "task_id": "T009"},   # first entry wins
                {"path": "c.py"},                      # no id → no attribution
                {},
            ],
        })

        scope = cro.resolve_review_scope(state, "abc123")

        assert scope.source == "tracked_files"
        assert sorted(scope.paths) == sorted(cro.resolve_changed_files(state, "abc123"))
        assert sorted(scope.paths) == ["a.py", "b.py", "c.py"]
        assert scope.task_ids == {"a.py": "T001", "b.py": "implementation"}

    def test_the_git_diff_branch(self, capsys):
        state = _state_without_tracked_files(pre_existing=["plans/"])
        with patch.object(
            cro, "get_files_changed_since_ref", return_value=["x.py", "plans/p.md", "y.py"]
        ), patch.object(cro, "list_worktree_untracked_files", return_value=[]):
            scope = cro.resolve_review_scope(state, "abc123")
            paths = cro.resolve_changed_files(state, "abc123")

        assert scope.source == "git_diff"
        assert scope.paths == paths == ["x.py", "y.py"]
        assert scope.task_ids == {}

    def test_the_none_branch(self, capsys):
        state = _state_without_tracked_files()
        scope = cro.resolve_review_scope(state, "")
        assert scope == cro.ReviewScope([], "none", {})
        assert scope.paths == cro.resolve_changed_files(state, "")

    def test_a_recorded_files_scope_wins_over_tracked_files(self, capsys):
        """`--files` (decision 11.1) is checked first, whatever else state holds."""
        state = _dict_state({
            "review_scope_files": ["b.py", "a.py", "b.py"],
            "tracked_files": [{"path": "src/other.py", "task_id": "T001"}],
        })
        with patch.object(cro, "get_files_changed_since_ref", side_effect=AssertionError(
            "the diff must not be consulted when --files named the scope"
        )):
            scope = cro.resolve_review_scope(state, "abc123")

        assert scope == cro.ReviewScope(["b.py", "a.py"], "files_flag", {})
        assert "Using 2 files given with --files" in capsys.readouterr().out
        assert cro.resolve_changed_files(state, "abc123") == ["b.py", "a.py"]

    def test_an_empty_recorded_files_scope_falls_through(self, capsys):
        state = _dict_state({
            "review_scope_files": [],
            "tracked_files": [{"path": "src/a.py", "task_id": "T001"}],
        })
        assert cro.resolve_review_scope(state, "abc123").source == "tracked_files"


# ---------------------------------------------------------------------------
# The git-diff fallback is loud and bounded (review-scope plan §2 D2.5, T2.5)
# ---------------------------------------------------------------------------


def _fallback_output(capsys, diffed, untracked, *, pre_existing=None, base_ref="abc123"):
    """Resolve a fallback scope over *diffed*; return (scope, stdout)."""
    values = {"tracked_files": [], "head_at_start": base_ref}
    if pre_existing is not None:
        values["pre_existing_changes"] = list(pre_existing)
    capsys.readouterr()
    with patch.object(cro, "get_files_changed_since_ref", return_value=list(diffed)), \
         patch.object(cro, "list_worktree_untracked_files", return_value=untracked):
        scope = cro.resolve_review_scope(_dict_state(values), base_ref)
    return scope, capsys.readouterr().out


class TestTheFallbackScopeIsLoud:
    def test_scope_fallback_names_the_ref_the_count_the_groups_and_the_baseline(
        self, capsys
    ):
        diffed = [
            "src/a.py", "src/b.py", "tests/test_a.py",
            "tests/fixtures/opencode/one.json", "tests/fixtures/opencode/two.json",
        ]
        scope, out = _fallback_output(capsys, diffed, [])

        assert scope.source == "git_diff"
        line = next(l for l in out.splitlines() if l.startswith("[SCOPE_FALLBACK] no implement"))
        assert "`git diff --name-only abc123`" in line
        assert "5 files (source 2, tests 1, fixtures 2, generated 0)" in line
        assert "baseline: none recorded" in line
        assert "[SCOPE_LARGE]" not in out

    def test_a_recorded_baseline_is_named_with_what_it_subtracted(self, capsys):
        _scope, out = _fallback_output(
            capsys, ["src/a.py", "plans/p.md"], [], pre_existing=["plans/"]
        )
        assert (
            "baseline: pre_existing_changes recorded (1 entries), 1 path(s) subtracted"
            in out
        )

    def test_the_untracked_sample_is_capped_worktree_wide_and_leads_with_neighbours(
        self, capsys
    ):
        diffed = ["pkg/mod.py", "README.md"]
        # 24 unrelated untracked files that sort BEFORE the neighbour, plus the
        # one untracked file that sits beside a diffed file.
        untracked = [f"aaa/build/out_{i:02d}.o" for i in range(24)] + ["pkg/new_helper.py"]

        scope, out = _fallback_output(capsys, diffed, untracked)

        assert "[SCOPE_FALLBACK] untracked in worktree, not reviewed: 25" in out
        assert "worktree-wide" in out
        named = [l for l in out.splitlines() if l.startswith("  - ")]
        assert len(named) == 10
        assert named[0] == "  - pkg/new_helper.py (beside a diffed file)"
        assert named[1:] == [f"  - aaa/build/out_{i:02d}.o" for i in range(9)]
        assert "  … and 15 more" in out.splitlines()
        # Decision 11.2: untracked files are counted, never added to the scope.
        assert not set(untracked) & set(scope.paths)

    def test_an_unknown_untracked_count_says_so(self, capsys):
        _scope, out = _fallback_output(capsys, ["src/a.py"], None)
        assert "untracked in worktree, not reviewed: unknown" in out

    @pytest.mark.parametrize("reviewed, fires", [(200, False), (201, True)])
    def test_scope_large_fires_above_200_source_and_test_files(
        self, capsys, reviewed, fires
    ):
        source = [f"src/m_{i:03d}.py" for i in range(reviewed - 1)]
        diffed = source + ["tests/test_m.py"] + [
            f"tests/fixtures/f_{i:03d}.json" for i in range(300)
        ]
        _scope, out = _fallback_output(capsys, diffed, [])

        large = [l for l in out.splitlines() if l.startswith("[SCOPE_LARGE]")]
        assert bool(large) is fires
        if fires:
            text = large[0]
            assert text.startswith("[SCOPE_LARGE] 201 source and test files")
            assert "fixtures and generated files not counted" in text
            assert "nothing is dropped" in text
            # Both remedies, each with its condition.
            assert "If the implementation is uncommitted, re-run with `--base-ref HEAD`" in text
            assert "Otherwise" in text
            assert "finalize-tracking" in text and "--only-touched --touched" in text
            assert "`git diff HEAD` sees only uncommitted changes" in text

    def test_the_tracked_files_branch_prints_no_fallback_marker(self, capsys):
        state = _dict_state({"tracked_files": [{"path": "a.py", "task_id": "T1"}]})
        cro.resolve_review_scope(state, "abc123")
        assert "[SCOPE_FALLBACK]" not in capsys.readouterr().out

    def test_the_reaggregate_prints_the_same_lines(self, tmp_path, capsys):
        plan, phase_dir = _cro_phase(
            tmp_path,
            {cro.sanitize_model_name(_REAL_SPECS[0]): _issues(["Issue A"])},
            models_requested=[_REAL_SPECS[0]],
        )
        capsys.readouterr()

        _run_cro_reaggregate(
            plan,
            phase_dir,
            extra_patches=[
                patch.object(
                    cro, "get_or_create_state",
                    return_value=_state_without_tracked_files(),
                ),
                patch.object(
                    cro, "get_files_changed_since_ref",
                    return_value=["apps/web/a.tsx", "apps/web/b.tsx"],
                ),
                patch.object(
                    cro, "list_worktree_untracked_files",
                    return_value=["apps/web/c.tsx"],
                ),
            ],
        )
        out = capsys.readouterr().out

        assert "[SCOPE_FALLBACK] no implement phase recorded the files under review" in out
        assert "2 files (source 2, tests 0, fixtures 0, generated 0)" in out
        assert "[SCOPE_FALLBACK] untracked in worktree, not reviewed: 1" in out
        assert "  - apps/web/c.tsx (beside a diffed file)" in out


class TestNoUntrackedPathEntersTheFallbackDiff:
    """Real git: the fallback's paths are the diff, and an untracked file — even
    one beside a changed file — is counted and named, never reviewed."""

    def test_in_a_real_repository(self, tmp_path, monkeypatch, capsys):
        import subprocess

        repo = tmp_path / "repo"
        repo.mkdir()
        for args in (
            ["init", "--quiet"],
            ["config", "user.email", "t@example.com"],
            ["config", "user.name", "T"],
            ["config", "commit.gpgsign", "false"],
        ):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        (repo / "src").mkdir()
        (repo / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "--quiet", "-m", "init"], cwd=repo, check=True,
            capture_output=True,
        )
        (repo / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
        (repo / "src" / "new.py").write_text("n = 1\n", encoding="utf-8")
        monkeypatch.chdir(repo)
        capsys.readouterr()

        scope = cro.resolve_review_scope(_state_without_tracked_files(), "HEAD")
        out = capsys.readouterr().out

        assert scope == cro.ReviewScope(["src/a.py"], "git_diff", {})
        assert "untracked in worktree, not reviewed: 1" in out
        assert "  - src/new.py (beside a diffed file)" in out


# ---------------------------------------------------------------------------
# `--reaggregate` after a `--files` fan-out (decision 11.1, T2.6)
# ---------------------------------------------------------------------------


class TestReaggregateResolvesTheFilesScopeTheFanOutRecorded:
    def test_the_recorded_scope_wins_over_tracked_files(self, tmp_path, capsys):
        plan, phase_dir = _cro_phase(
            tmp_path,
            {cro.sanitize_model_name(_REAL_SPECS[0]): _issues(["Issue A"])},
            models_requested=[_REAL_SPECS[0]],
        )
        state = _dict_state({
            "review_scope_files": ["lib/one.py", "lib/two.py"],
            "tracked_files": [{"path": "src/a.py", "task_id": "implementation"}],
            "head_at_start": "abc123",
        })
        capsys.readouterr()

        _run_cro_reaggregate(
            plan,
            phase_dir,
            render_report=True,
            extra_patches=[
                patch.object(cro, "get_or_create_state", return_value=state),
                patch.object(cro, "get_files_changed_since_ref", side_effect=AssertionError(
                    "a recorded --files scope must not fall back to the diff"
                )),
            ],
        )
        out = capsys.readouterr().out

        assert "Using 2 files given with --files" in out
        assert "Files to review: 2" in out
        report = _rendered_report(plan)
        assert "**Files Changed:** 2" in report
        assert "- `lib/one.py`" in report
        assert "- `lib/two.py`" in report


# ---------------------------------------------------------------------------
# The reaggregate reports the scope that was actually reviewed (review-scope
# plan §2, D2.2/T2.2)
# ---------------------------------------------------------------------------
#
# The reaggregate never enters the fan-out block, so it writes `review-scope.md`
# and `diff/` only when the pair is missing, and then from the current tree;
# otherwise it reads the header back. Each loaded result is checked against the
# scope ledger (`.review-scope.json`). The harness's base ref (`abc123`)
# resolves nowhere, so every set here is the degraded one — which is also what
# makes a regenerated set's id reproducible from outside.

_SCOPE_SPECS = ["claude-code:sonnet", "cursor-agent:auto", "gemini:flash"]
_REGENERATED = "reaggregate (regenerated from the current tree)"


def _scope_phase(tmp_path):
    return _cro_phase(
        tmp_path,
        {
            cro.sanitize_model_name(spec): _issues([f"Issue {index}"])
            for index, spec in enumerate(_SCOPE_SPECS)
        },
        models_requested=_SCOPE_SPECS,
    )


def _report_data(phase_dir):
    return json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))


def _write_ledger(phase_dir, entries):
    (phase_dir / cro.REVIEW_SCOPE_LEDGER).write_text(json.dumps(entries), encoding="utf-8")


class TestTheReaggregateReportsTheScopeThatWasReviewed:
    def test_a_missing_pair_is_regenerated_and_the_report_says_so(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _scope_phase(tmp_path)

        _run_cro_reaggregate(plan, phase_dir, render_report=True)

        out = capsys.readouterr().out
        assert "[REVIEW_SCOPE_REGENERATED]" in out
        manifest = (phase_dir / "review-scope.md").read_text(encoding="utf-8")
        assert f"- Written by: {_REGENERATED}" in manifest
        assert (phase_dir / "diff" / ".gitignore").is_file()
        report = _cro_report_md(plan)
        assert f"written by {_REGENERATED}" in report
        assert "not a record of what the reviewers read" in report
        data = _report_data(phase_dir)
        assert data["reviewScope"]["writtenBy"] == _REGENERATED
        assert data["reviewScope"]["regenerated"] is True
        assert data["diffCoverage"]["coverage"] == "partial"

    def test_a_whole_pair_is_read_back_and_left_untouched(self, tmp_path, capsys):
        plan, phase_dir = _scope_phase(tmp_path)
        written = cro.write_review_diff("abc123", ["src/a.py"], phase_dir)
        manifest = phase_dir / "review-scope.md"
        before = (manifest.read_bytes(), manifest.stat().st_mtime_ns)
        capsys.readouterr()

        _run_cro_reaggregate(plan, phase_dir, render_report=True)

        assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
        assert "[REVIEW_SCOPE_REGENERATED]" not in capsys.readouterr().out
        data = _report_data(phase_dir)
        assert data["reviewScope"]["scopeId"] == written.scope_id
        assert data["reviewScope"]["writtenBy"] == "fan-out"
        assert data["reviewScope"]["regenerated"] is False
        assert data["diffCoverage"]["coverage"] == written.coverage
        assert data["diffCoverage"]["omitted"] == [
            {"path": "src/a.py", "reason": "git diff failed"}
        ]

    def test_each_result_is_named_by_what_the_ledger_records(self, tmp_path, capsys):
        plan, phase_dir = _scope_phase(tmp_path)
        written = cro.write_review_diff("abc123", ["src/a.py"], phase_dir)
        stems = [cro.sanitize_model_name(spec) for spec in _SCOPE_SPECS]
        _write_ledger(phase_dir, {stems[0]: written.scope_id, stems[1]: "0123456789ab"})

        _run_cro_reaggregate(plan, phase_dir, render_report=True)

        out = capsys.readouterr().out
        # Their findings stay: three models, three issues.
        assert "Re-aggregating 3 issues from 3 models" in out
        report = _cro_report_md(plan)
        assert "`cursor-agent:auto` reviewed another scope (0123456789ab)" in report
        assert "`gemini:flash` scope unknown" in report
        assert "`claude-code:sonnet` reviewed" not in report
        assert _report_data(phase_dir)["reviewScope"]["results"] == {
            "cursor-agent:auto": "reviewed another scope (0123456789ab)",
            "gemini:flash": "scope unknown",
        }

    def test_a_regenerated_set_every_ledger_entry_matches_is_the_one_reviewed(
        self, tmp_path
    ):
        plan, phase_dir = _scope_phase(tmp_path)
        # The id is content-derived, so the same inputs elsewhere give the id a
        # regeneration of this phase dir will compute.
        expected = cro.write_review_diff("abc123", ["src/a.py"], tmp_path / "scratch")
        _write_ledger(phase_dir, {
            cro.sanitize_model_name(spec): expected.scope_id for spec in _SCOPE_SPECS
        })

        _run_cro_reaggregate(plan, phase_dir, render_report=True)

        report = _cro_report_md(plan)
        assert "every result's recorded scope matches it" in report
        data = _report_data(phase_dir)
        assert data["reviewScope"]["scopeId"] == expected.scope_id
        assert data["reviewScope"]["resultsMatch"] is True
        assert data["reviewScope"]["results"] == {}

    def test_the_ledger_is_never_loaded_as_a_model_result(self, tmp_path, capsys):
        plan, phase_dir = _scope_phase(tmp_path)
        _write_ledger(phase_dir, {
            cro.sanitize_model_name(spec): "0123456789ab" for spec in _SCOPE_SPECS
        })

        report = _run_cro_reaggregate(plan, phase_dir)

        assert sorted(_reported_results(report)) == sorted(_SCOPE_SPECS)
        assert "Re-aggregating 3 issues from 3 models" in capsys.readouterr().out
