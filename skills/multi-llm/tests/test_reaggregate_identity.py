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
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
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
                         side_effect=lambda files: contextlib.nullcontext()))
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
                    extra_patches=()):
    """Drive `code_review_orchestrator.main()` with the providers mocked out.

    Only the model invocation and the report renderer are stubbed — the phase
    dir, `save_model_result`, grouping and `grouped.json` are the real ones, so
    what lands on disk is what a real fan-out leaves for `--reaggregate` to read.

    Returns the plan, the phase dir and **what `main()` returned**, because two
    of the questions below are about the run surviving something: a checkpoint
    write that quietly does nothing must not turn into an early return or a
    non-zero exit, and `None` (the validation leg's early return, which
    `sys.exit` treats as 0) has to be distinguishable from `EXIT_OK` only in the
    sense that neither is a failure.
    """
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    models = list(issues_by_model)
    results = {m: (True, issues_by_model[m], None) for m in models}
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
                         side_effect=lambda files: contextlib.nullcontext()))
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
                         side_effect=lambda files: contextlib.nullcontext()))
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
            "**Validation:** 3 valid, 0 invalid, 0 needs human review"
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
            "**Validation:** 0 valid, 5 invalid, 0 needs human review"
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


def _as_legacy_file(phase_dir, status):
    """Strip `metadata.grouped_sha256` and set every verdict to *status*.

    A pre-D7 `validation.json`: hashed records, no binding to any grouping.
    """
    path = phase_dir / "validation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for group in data["groups"]:
        group["status"] = status
        group["reason"] = "From the legacy file"
    del data["metadata"]["grouped_sha256"]
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

        assert "Found existing validation.json, using it for report generation" in out
        assert "Not using validation.json" not in out
        assert _stale_files(phase_dir) == []
        # Left as it was: still key-less, still the legacy verdicts.
        data = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert "grouped_sha256" not in data["metadata"]
        assert _validation_summary(phase_dir) == (
            "**Validation:** 3 valid, 0 invalid, 0 needs human review"
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
            "Not using validation.json: validation.json carries no grouped_sha256 "
            f"and 1 batch file(s) are newer than it ({newer.name})"
        ) in out
        stale = _stale_files(phase_dir)
        assert len(stale) == 1
        assert (phase_dir / stale[0]).read_bytes() == stale_bytes
        # Merged from the batches, and bound from now on.
        data = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        assert data["metadata"]["grouped_sha256"] == _grouped_hash(phase_dir)
        assert [g["status"] for g in data["groups"]] == ["invalid"] * 3
        assert _validation_summary(phase_dir) == (
            "**Validation:** 0 valid, 3 invalid, 0 needs human review"
        )
