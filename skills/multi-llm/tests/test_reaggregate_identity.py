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


def _run_cro_reaggregate(plan, phase_dir):
    """Drive the code-review reaggregate; return the patched report mock.

    Everything patched here is git/state/report machinery the identity question
    does not touch — ``build_unsanitize_map`` is deliberately NOT patched, since
    the recorded checkpoint is the whole point.
    """
    args = argparse.Namespace(
        plan_file=str(plan),
        models=["claude-code:sonnet"],
        interactive=False,
        quick=False,
        timeout=None,
        max_parallel=2,
        base_ref=None,
        skip_validation=True,
        validation_model=None,
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

    with patch.object(cro, "get_or_create_state", return_value=fake_state), \
         patch.object(cro, "get_phase_dir", return_value=phase_dir), \
         patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
         patch.object(cro, "intent_to_add_untracked",
                      side_effect=lambda files: contextlib.nullcontext()), \
         patch.object(cro, "generate_review_report", report):
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


def _fanout_args(plan, models):
    """`parse_args` output for a foreground `review-code` run with no validation."""
    return argparse.Namespace(
        plan_file=str(plan),
        models=list(models),
        interactive=False,
        quick=False,
        timeout=None,
        max_parallel=2,
        base_ref=None,
        skip_validation=True,
        validation_model=None,
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


def _run_cro_fanout(tmp_path, issues_by_model):
    """Drive `code_review_orchestrator.main()` with the providers mocked out.

    Only the model invocation and the report renderer are stubbed — the phase
    dir, `save_model_result`, grouping and `grouped.json` are the real ones, so
    what lands on disk is what a real fan-out leaves for `--reaggregate` to read.
    """
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    models = list(issues_by_model)
    results = {m: (True, issues_by_model[m], None) for m in models}

    with patch.object(cro, "parse_args", return_value=_fanout_args(plan, models)), \
         patch.object(cro, "get_or_create_state", return_value=_fanout_state()), \
         patch.object(cro, "resolve_models", return_value=models), \
         patch.object(cro, "is_model_valid", return_value=True), \
         patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
         patch.object(cro, "intent_to_add_untracked",
                      side_effect=lambda files: contextlib.nullcontext()), \
         patch.object(cro, "run_all_reviews", new=AsyncMock(return_value=results)), \
         patch.object(cro, "generate_review_report", return_value="# report\n"):
        asyncio.run(cro.main())

    return plan, cro.get_phase_dir(plan, "code-review")


#: Two specs whose sanitized stems are lossy in both of the ways that matter: the
#: `:` that separates provider from model and the `.` inside a version number.
_REAL_SPECS = ["claude-code:sonnet-4.6", "cursor-agent:cursor-grok-4.6-xhigh"]


class TestReviewCodeWritesTheCheckpointReaggregationReads:
    def test_the_fanout_records_the_canonical_specs(self, tmp_path):
        """`.status.json` exists, and it holds specs — not filename stems."""
        plan, phase_dir = _run_cro_fanout(
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
        plan, phase_dir = _run_cro_fanout(
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
