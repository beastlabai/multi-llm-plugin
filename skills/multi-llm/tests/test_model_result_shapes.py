"""Each review lane's ``save_model_result`` reads any reply without ending the round.

After a fan-out, the save loop calls ``save_model_result`` once per model, and
an exception from one call ends the loop: no model after it is saved and the
round fails. These replies did that, left a partial ``{model}.json``, or were
saved and then ended the round further on:

* valid JSON of an unexpected shape on the review-plan / review-tasks lane
  (``[1]``, ``[null]``, ``{"result": [...]}``, a ``type`` that is not a
  string) raised ``TypeError`` or ``AttributeError`` from the suggestion
  checks. Such entries are invalid suggestions now, dropped with a warning, as
  review-code drops them (``schema_validator.drop_non_record_entries``);
* a reply file nested deeper than the parser's stack raised ``RecursionError``
  out of ``json_extractor.parse_file_content``, which both lanes read result
  files with. It is a reply that does not parse now;
* a reply that parses but nests too deep to encode: review-code wrote with
  ``json.dump`` straight into the open file, which left a truncated result and
  raised out of the loop; and where the save's encoder managed it (Python
  3.14's ``json.dumps``), the round's later ``json.dump`` of ``grouped.json``
  or the issues file did not. Both lanes now refuse a result nested deeper than
  ``MAX_RESULT_DEPTH`` and encode before opening the file
  (``review_orchestrator_base.write_model_result``);
* the same reply, when a file-capable provider had already written it to
  ``{model}.json``: the refused file stayed, and ``--reaggregate`` read it,
  recorded the model completed and ended in ``json.dump`` of ``grouped.json``.
  A refused result's file is set aside as ``{model}.json.rejected`` now, and
  both reaggregates refuse such a file themselves.

Every round test puts the odd replies before an ordinary one, so the ordinary
one is saved only if the loop got past them, and runs grouping and the report
for real.
"""

import asyncio
import contextlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import status_file  # noqa: E402
from utils.json_extractor import sanitize_model_name  # noqa: E402
from utils.llm_client import ERROR_PARSE_ERROR, invoke_with_file_output  # noqa: E402
from utils.review_orchestrator_base import MAX_RESULT_DEPTH, write_model_result  # noqa: E402
from utils.result_files import REJECTED_RESULT_SUFFIX  # noqa: E402
from tests.test_code_review_orchestrator import _make_args  # noqa: E402
from tests.test_reaggregate_identity import _run_cro_reaggregate  # noqa: E402

#: The title that makes :func:`_failing_encoder` refuse a record list.
ENCODER_FAILS = "The encoder cannot write this one"

SUGGESTION = {
    "title": "Validate the email field",
    "desc": "The plan adds the field but never says how it is validated.",
    "importance": "HIGH",
    "reference": "Overview",
    "type": "addition",
}

ISSUE = {
    "title": "Email is never validated",
    "desc": "The handler stores the field as typed.",
    "importance": "MEDIUM",
    "file": "src/a.py",
    "line_range": [1, 2],
    "type": "bug",
    "anchor_text": "store(email)",
}

#: Deeper than any Python's encoder, shallower than 3.12's and later decoders.
PARSED_TOO_DEEP = 1100


def _nested(depth):
    """A list nested *depth* lists deep, built without recursion."""
    value = []
    for _ in range(depth - 1):
        value = [value]
    return value


def _at_depth(record, depth):
    """``[record]`` with an extra field that makes the list *depth* levels deep
    (the list is 1, the record 2)."""
    return [dict(record, x=_nested(depth - 2))]


def _deep_reply(depth, record=SUGGESTION):
    """:func:`_at_depth` of *record* as reply text, written without the
    encoder (which may not manage it)."""
    inner = depth - 2
    return json.dumps([record])[:-2] + ', "x": ' + "[" * inner + "]" * inner + "}]"


def _parses(text):
    try:
        json.loads(text)
    except RecursionError:
        return False
    return True


@contextlib.contextmanager
def _failing_encoder():
    """``json.dump`` and ``json.dumps`` fail as the real ones do on a record
    list holding :data:`ENCODER_FAILS`: ``dump`` after writing a prefix to the
    file, ``dumps`` before returning anything. Every other value is encoded."""
    real_dump, real_dumps = json.dump, json.dumps

    def refused(obj):
        return isinstance(obj, list) and any(
            isinstance(item, dict) and item.get("title") == ENCODER_FAILS
            for item in obj)

    def dump(obj, fp, *args, **kwargs):
        if refused(obj):
            fp.write('[\n  {\n    "title": "The encoder')
            raise RecursionError("maximum recursion depth exceeded")
        return real_dump(obj, fp, *args, **kwargs)

    def dumps(obj, *args, **kwargs):
        if refused(obj):
            raise RecursionError("maximum recursion depth exceeded")
        return real_dumps(obj, *args, **kwargs)

    with patch.object(json, "dump", dump), patch.object(json, "dumps", dumps):
        yield


def _files(phase_dir, model):
    """The names of *model*'s artifacts in *phase_dir*."""
    stem = sanitize_model_name(model)
    return sorted(p.name for p in Path(phase_dir).iterdir() if stem in p.name)


def _read(phase_dir, name):
    return (Path(phase_dir) / name).read_text(encoding="utf-8")


# --- write_model_result: the writer both lanes share ------------------------------


class TestWriteModelResultBoundsTheNesting:

    def test_a_result_at_the_bound_is_written(self, tmp_path):
        records = _at_depth(ISSUE, MAX_RESULT_DEPTH)

        assert write_model_result("m:x", records, tmp_path, noun="issues") is True
        assert json.loads(_read(tmp_path, "m_x.json")) == records

    def test_one_level_deeper_is_refused_with_no_file(self, tmp_path, capsys):
        records = _at_depth(ISSUE, MAX_RESULT_DEPTH + 1)

        assert write_model_result(
            "m:x", records, tmp_path, noun="issues", raw_output="the reply") is False
        assert sorted(p.name for p in tmp_path.iterdir()) == ["error_m_x.log"]
        log = _read(tmp_path, "error_m_x.log")
        assert (
            "Encode error: the reply parsed, but it is nested too deep to write "
            f"back out as m_x.json: more than {MAX_RESULT_DEPTH} levels"
        ) in log
        assert "Raw output:\nthe reply" in log
        assert "nested too deep to save, raw output saved to:" in capsys.readouterr().out

    def test_a_file_the_provider_wrote_is_set_aside_whole(self, tmp_path, capsys):
        """A file-capable provider writes ``{model}.json`` before the reply is
        saved; left there, every later reader takes it for the result."""
        written = _deep_reply(MAX_RESULT_DEPTH + 1, ISSUE)
        (tmp_path / "m_x.json").write_text(written, encoding="utf-8")

        assert write_model_result(
            "m:x", json.loads(written), tmp_path, noun="issues") is False
        assert not (tmp_path / "m_x.json").exists()
        assert _read(tmp_path, f"m_x.json.{REJECTED_RESULT_SUFFIX}") == written
        assert (
            f"The m_x.json already in the phase dir is set aside as "
            f"m_x.json.{REJECTED_RESULT_SUFFIX}."
        ) in _read(tmp_path, "error_m_x.log")
        note = (
            "m_x.json is not m:x's result: the reply is nested too deep to write "
            f"back out (more than {MAX_RESULT_DEPTH} levels of lists and objects); "
            f"set aside as m_x.json.{REJECTED_RESULT_SUFFIX} (nothing was deleted)"
        )
        assert note in capsys.readouterr().out
        assert status_file.read_status(str(tmp_path))["result_notes"] == [note]

    def test_an_earlier_set_aside_file_is_not_overwritten(self, tmp_path):
        (tmp_path / f"m_x.json.{REJECTED_RESULT_SUFFIX}").write_text("first", encoding="utf-8")
        (tmp_path / "m_x.json").write_text("second", encoding="utf-8")

        assert write_model_result(
            "m:x", _at_depth(ISSUE, MAX_RESULT_DEPTH + 1), tmp_path, noun="issues") is False
        set_aside = sorted(
            p.read_text(encoding="utf-8") for p in tmp_path.iterdir()
            if p.name.startswith(f"m_x.json.{REJECTED_RESULT_SUFFIX}"))
        assert set_aside == ["first", "second"]
        assert not (tmp_path / "m_x.json").exists()

    def test_a_deep_object_counts_like_a_deep_list(self, tmp_path):
        value = {}
        for _ in range(MAX_RESULT_DEPTH):
            value = {"k": value}
        assert write_model_result("m:x", [value], tmp_path, noun="issues") is False
        assert not (tmp_path / "m_x.json").exists()


# --- review-plan / review-tasks ---------------------------------------------------


def _save_reply(tmp_path, output, model="m:x"):
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir(exist_ok=True)
    saved = base.save_model_result(
        "p", model, True, output, None, str(tmp_path), str(phase_dir), "review_plan")
    return saved, phase_dir


def _changed(**changes):
    return dict(SUGGESTION, **changes)


class TestTheReviewBaseDropsEntriesOfAnotherShape:
    """An entry that is not a suggestion object is an invalid suggestion: it is
    dropped with a warning and the rest of the reply is saved. A reply whose
    entries are all dropped is saved as ``[]``, as a reply of malformed
    suggestion objects always was."""

    @pytest.mark.parametrize("output, kept, dropped", [
        ("[1]", [], 1),
        ("[null]", [], 1),
        (json.dumps([1, SUGGESTION, None]), [SUGGESTION], 2),
        (json.dumps(["title desc importance reference type"]), [], 1),
        (json.dumps([[SUGGESTION]]), [], 1),
        (json.dumps([_changed(type=None)]), [], 1),
        (json.dumps([_changed(type=5)]), [], 1),
        (json.dumps([_changed(type=["addition"])]), [], 1),
        (json.dumps([_changed(importance=None)]), [], 1),
        (json.dumps({"result": [SUGGESTION]}), [SUGGESTION], 0),
        (json.dumps({"result": [1]}), [], 1),
        (json.dumps({"result": {"suggestions": [1, SUGGESTION]}}), [SUGGESTION], 1),
    ], ids=[
        "int", "null", "mixed", "string-entry", "nested-list", "type-null",
        "type-int", "type-list", "importance-null", "result-list",
        "result-list-of-int", "result-object",
    ])
    def test_an_odd_entry_is_dropped_and_the_reply_saved(
        self, tmp_path, capsys, output, kept, dropped
    ):
        saved, phase_dir = _save_reply(tmp_path, output)

        assert saved is True
        assert _files(phase_dir, "m:x") == ["m_x.json"]
        assert json.loads(_read(phase_dir, "m_x.json")) == kept
        assert capsys.readouterr().out.count(" invalid - ") == dropped

    def test_a_non_object_entry_is_named_in_the_warning(self, tmp_path, capsys):
        _save_reply(tmp_path, "[1]")
        out = capsys.readouterr().out
        assert "[m:x] Warning: Suggestion 0 invalid - expected object, got int" in out
        assert "[m:x] Warning: All suggestions were invalid" in out

    @pytest.mark.parametrize("value", [5, None, True, "", "no array here"])
    def test_a_result_without_an_array_does_not_parse(self, tmp_path, value):
        """``{"result": 5}`` holds no array: the reply does not parse, the model
        fails, and nothing is salvaged (there is no JSON answer in it)."""
        saved, phase_dir = _save_reply(tmp_path, json.dumps({"result": value}))

        assert saved is False
        assert _files(phase_dir, "m:x") == ["error_m_x.log"]
        assert "No JSON in response" in _read(phase_dir, "error_m_x.log")

    def test_a_cli_envelope_is_still_read_as_its_answer_text(self, tmp_path):
        envelope = {"type": "result", "subtype": "success",
                    "result": "Here it is:\n```json\n" + json.dumps([SUGGESTION]) + "\n```"}
        saved, phase_dir = _save_reply(tmp_path, json.dumps(envelope))

        assert saved is True
        assert json.loads(_read(phase_dir, "m_x.json")) == [SUGGESTION]


class TestTheReviewBaseRoundSurvivesEveryShape:

    def test_the_model_after_the_odd_replies_is_saved(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\n## Overview\n\nAdd an email field.\n", encoding="utf-8")
        parsed_too_deep = _deep_reply(PARSED_TOO_DEEP)
        replies = {
            "cursor-agent:int": "[1]",
            "cursor-agent:null": "[null]",
            "cursor-agent:wrapped": json.dumps({"result": [1]}),
            "cursor-agent:typed": json.dumps([_changed(type=None)]),
            "cursor-agent:unparsed": "[" * (1 << 20),
            "cursor-agent:too-deep": parsed_too_deep,
            "cursor-agent:encoder": json.dumps([_changed(title=ENCODER_FAILS)]),
            "cursor-agent:at-the-bound": _deep_reply(MAX_RESULT_DEPTH),
            "cursor-agent:good": json.dumps([SUGGESTION]),
        }

        async def fake_run_all_models(*_args, **_kwargs):
            return {
                spec: {
                    "success": True, "output": output, "error": None,
                    "prompt": "", "stdout": output, "stderr": "",
                    "duration_seconds": 0.1, "model_spec": spec, "source": "file",
                }
                for spec, output in replies.items()
            }

        argv = ["--plan-file", str(plan), "--models", *replies,
                "--skip-validation", "--timeout", "5", "--max-parallel", "2"]
        with patch.object(base, "run_all_models", new=fake_run_all_models), \
                _failing_encoder():
            exit_code = asyncio.run(rpo.main(argv))

        assert exit_code == 0
        phase_dir = base.get_phase_dir(plan, "review-plan")

        def saved(spec):
            return json.loads(_read(phase_dir, f"{sanitize_model_name(spec)}.json"))

        assert saved("cursor-agent:good") == [SUGGESTION]
        assert saved("cursor-agent:at-the-bound")[0]["title"] == SUGGESTION["title"]
        for spec in ("cursor-agent:int", "cursor-agent:null",
                     "cursor-agent:wrapped", "cursor-agent:typed"):
            assert saved(spec) == []
        # The ones that failed left a log and no result file, partial or whole.
        failed = {
            "cursor-agent:unparsed": "Parse error",
            "cursor-agent:too-deep": (
                "Encode error: the reply parsed" if _parses(parsed_too_deep)
                else "Parse error"),
            "cursor-agent:encoder": "Encode error: the reply parsed",
        }
        for spec, line in failed.items():
            stem = sanitize_model_name(spec)
            assert not (phase_dir / f"{stem}.json").exists()
            assert line in _read(phase_dir, f"error_{stem}.log")
        status = status_file.read_status(str(phase_dir))
        assert set(status["models_failed"]) == set(failed)
        assert {"cursor-agent:good", "cursor-agent:at-the-bound"} <= set(
            status["models_completed"])
        assert SUGGESTION["title"] in _read(phase_dir, "report.md")


# --- review-code --------------------------------------------------------------------


def _save_issues(tmp_path, issues, model="m:x"):
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir(exist_ok=True)
    return cro.save_model_result("p", model, True, issues, None, phase_dir), phase_dir


class TestReviewCodeWritesTheResultWholeOrNotAtAll:

    @pytest.mark.parametrize("issues", [[1], [None], [1, ISSUE]], ids=["int", "null", "mixed"])
    def test_entries_of_another_shape_are_kept_for_aggregation_to_drop(self, tmp_path, issues):
        saved, phase_dir = _save_issues(tmp_path, issues)

        assert saved is True
        assert json.loads(_read(phase_dir, "m_x.json")) == issues

    def test_an_exhausted_encoder_leaves_no_partial_file(self, tmp_path, capsys):
        with _failing_encoder():
            saved, phase_dir = _save_issues(tmp_path, [dict(ISSUE, title=ENCODER_FAILS)])

        assert saved is False
        assert sorted(p.name for p in phase_dir.iterdir()) == ["error_m_x.log"]
        log = _read(phase_dir, "error_m_x.log")
        assert "Encode error: the reply parsed" in log
        assert "maximum recursion depth exceeded" in log
        out = capsys.readouterr().out
        assert "nested too deep to save, error logged to:" in out
        assert "Saved" not in out

    def test_a_reply_too_deep_to_write_leaves_no_partial_file(self, tmp_path):
        """``json.dump`` into the file stopped at about 1,000 levels on every
        Python, after writing part of the file, and raised out of the loop."""
        saved, phase_dir = _save_issues(tmp_path, _at_depth(ISSUE, PARSED_TOO_DEEP))

        assert saved is False
        assert sorted(p.name for p in phase_dir.iterdir()) == ["error_m_x.log"]
        assert "Encode error: the reply parsed" in _read(phase_dir, "error_m_x.log")

    def test_a_failed_run_still_logs_its_error(self, tmp_path):
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        assert cro.save_model_result("p", "m:x", False, None, "boom", phase_dir) is False
        assert "Error: boom" in _read(phase_dir, "error_m_x.log")
        assert not (phase_dir / "m_x.json").exists()


class TestReviewCodeReadsADeepFileAsUnparsed:
    """The result file is read by ``invoke_with_file_output``; one nested
    deeper than the parser's stack raised ``RecursionError`` out of the reader,
    and the ladder ended the slot as an exception. It is a file that does not
    parse now, and the reader goes on to stdout as for any other."""

    @pytest.mark.parametrize("text", [
        "[" * (1 << 20),
        "[" * (1 << 20) + "]" * (1 << 20),
    ], ids=["unclosed", "closed"])
    def test_a_deep_file_does_not_raise(self, tmp_path, text):
        model = "cursor-agent:auto"
        phase_dir = tmp_path / "code-review"

        def provider(*_args, **_kwargs):
            (phase_dir / f"{sanitize_model_name(model)}.json").write_text(text, encoding="utf-8")
            return {"success": False, "error": "No valid JSON in stdout",
                    "error_code": ERROR_PARSE_ERROR, "data": None,
                    "details": {"provider": "cursor-agent"}}

        with patch("utils.llm_client.invoke_with_provider", side_effect=provider):
            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec=model, prompt_context={}, output_dir=tmp_path,
                phase="code_review", timeout=600,
            )

        assert result["success"] is False
        assert result["error"] == "JSON parsing failed from both file and stdout"
        assert result["file_error"] == "No valid JSON found in text"


class TestTheReviewCodeRoundSurvivesEveryShape:

    def test_the_model_after_the_odd_replies_is_saved(self, tmp_path, capsys):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        at_the_bound = dict(ISSUE, title="At the bound")
        results = {
            "cursor-agent:int": (True, [1], None),
            "cursor-agent:null": (True, [None], None),
            "cursor-agent:too-deep": (True, _at_depth(ISSUE, PARSED_TOO_DEEP), None),
            "cursor-agent:encoder": (True, [dict(ISSUE, title=ENCODER_FAILS)], None),
            "cursor-agent:at-the-bound": (True, _at_depth(at_the_bound, MAX_RESULT_DEPTH), None),
            "cursor-agent:good": (True, [dict(ISSUE)], None),
        }
        state = MagicMock()
        state.get = MagicMock(side_effect=lambda k, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_before_implement": "abc123",
            "head_at_start": "abc123",
            "pre_existing_changes": [],
        }.get(k, default))
        state.state = {"phases_completed": {}}
        state.is_phase_completed = MagicMock(return_value=False)

        with patch.object(cro, "parse_args",
                          return_value=_make_args(plan, models=list(results))), \
                patch.object(cro, "get_or_create_state", return_value=state), \
                patch("utils.interactive.resolve_models", return_value=list(results)), \
                patch("utils.interactive.is_model_valid", return_value=True), \
                patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
                patch.object(cro, "run_all_reviews", new=AsyncMock(return_value=results)), \
                patch.object(cro, "get_phase_dir", return_value=phase_dir), \
                _failing_encoder():
            exit_code = asyncio.run(cro.main())

        assert exit_code == 0
        stem = {spec: sanitize_model_name(spec) for spec in results}
        saved = json.loads(_read(phase_dir, f"{stem['cursor-agent:good']}.json"))
        assert [issue["title"] for issue in saved] == [ISSUE["title"]]
        for spec in ("cursor-agent:too-deep", "cursor-agent:encoder"):
            assert not (phase_dir / f"{stem[spec]}.json").exists()
            assert "Encode error: the reply parsed" in _read(
                phase_dir, f"error_{stem[spec]}.log")
        status = status_file.read_status(str(phase_dir))
        assert set(status["models_failed"]) == {"cursor-agent:too-deep", "cursor-agent:encoder"}
        assert {"cursor-agent:good", "cursor-agent:at-the-bound"} <= set(
            status["models_completed"])
        # The round is aggregated from memory: a model whose result was not
        # saved contributes no issue to the issues file the report is built on.
        out = capsys.readouterr().out
        issues_path = Path(out.split("Issues JSON saved to: ", 1)[1].splitlines()[0])
        aggregated = json.loads(issues_path.read_text(encoding="utf-8"))
        assert sorted(issue["model"] for issue in aggregated) == [
            "cursor-agent:at-the-bound", "cursor-agent:good"]


# --- a refused result a provider had already written, then --reaggregate -----------

DEEP = "cursor-agent:deep"
GOOD = "cursor-agent:good"

#: Depths a reply is refused at: one past the bound, which every Python reads
#: and writes, and one the round's ``json.dump`` of ``grouped.json`` cannot
#: write (where this Python's decoder reads it at all).
REFUSED_DEPTHS = [
    pytest.param(MAX_RESULT_DEPTH + 1, id="past-the-bound"),
    pytest.param(PARSED_TOO_DEEP, id="past-the-encoder", marks=pytest.mark.skipif(
        not _parses(_deep_reply(PARSED_TOO_DEEP)),
        reason="this Python's decoder stops before the encoder does")),
]

#: A result file an older run left: one of the refused depths, or one nested
#: deeper than any decoder reads.
OLD_FILES = [
    pytest.param(MAX_RESULT_DEPTH + 1, id="past-the-bound"),
    REFUSED_DEPTHS[1],
    pytest.param(None, id="past-the-decoder"),
]


def _old_file(depth, record):
    if depth is None:
        return "[" * (1 << 20) + "]" * (1 << 20)
    return _deep_reply(depth, record)


def _set_aside_texts(phase_dir, stem):
    """What the files *stem*'s result was set aside as hold."""
    prefix = f"{stem}.json.{REJECTED_RESULT_SUFFIX}"
    return sorted(
        p.read_text(encoding="utf-8") for p in Path(phase_dir).iterdir()
        if p.name.startswith(prefix))


def _plan_round(tmp_path, replies, *, provider_writes=(), reaggregate=False):
    """One review-plan run over *replies*; a model in *provider_writes* also
    writes its reply to its own ``{model}.json``, as a file-capable provider
    does before the round saves it."""
    plan = tmp_path / "plan.md"
    if not plan.exists():
        plan.write_text("# Plan\n\n## Overview\n\nAdd an email field.\n", encoding="utf-8")
    phase_dir = Path(base.get_phase_dir(plan, "review-plan"))

    async def fake_run_all_models(*_args, **_kwargs):
        for spec in provider_writes:
            (phase_dir / f"{sanitize_model_name(spec)}.json").write_text(
                replies[spec], encoding="utf-8")
        return {
            spec: {
                "success": True, "output": output, "error": None,
                "prompt": "", "stdout": output, "stderr": "",
                "duration_seconds": 0.1, "model_spec": spec, "source": "file",
            }
            for spec, output in replies.items()
        }

    argv = ["--plan-file", str(plan), "--models", *replies,
            "--skip-validation", "--timeout", "5", "--max-parallel", "2"]
    if reaggregate:
        argv.append("--reaggregate")
    with patch.object(base, "run_all_models", new=fake_run_all_models):
        exit_code = asyncio.run(rpo.main(argv))
    return exit_code, phase_dir


def _assert_the_reviewer_stayed_failed(phase_dir, grouped_text, title):
    stem = sanitize_model_name(DEEP)
    assert not (phase_dir / f"{stem}.json").exists()
    status = status_file.read_status(str(phase_dir))
    assert DEEP in status["models_failed"]
    assert DEEP not in status["models_completed"]
    assert GOOD in status["models_completed"]
    # grouped.json was written whole, from the ordinary reviewer alone.
    grouped = json.loads(grouped_text)
    assert title in json.dumps(grouped)
    assert DEEP not in grouped_text


class TestTheReviewBaseSetsARefusedProviderFileAside:

    @pytest.mark.parametrize("depth", REFUSED_DEPTHS)
    def test_the_reviewer_stays_failed_through_a_reaggregate(self, tmp_path, depth):
        reply = _deep_reply(depth)
        replies = {DEEP: reply, GOOD: json.dumps([SUGGESTION])}

        exit_code, phase_dir = _plan_round(tmp_path, replies, provider_writes=[DEEP])
        assert exit_code == 0
        stem = sanitize_model_name(DEEP)
        assert not (phase_dir / f"{stem}.json").exists()
        assert _set_aside_texts(phase_dir, stem) == [reply]

        exit_code, _ = _plan_round(tmp_path, replies, reaggregate=True)

        assert exit_code == 0
        _assert_the_reviewer_stayed_failed(
            phase_dir, _read(phase_dir, "grouped.json"), SUGGESTION["title"])
        assert f"set aside as {stem}.json.{REJECTED_RESULT_SUFFIX}" in _read(
            phase_dir, "report.md")

    @pytest.mark.parametrize("depth", OLD_FILES)
    def test_a_reaggregate_refuses_a_file_an_older_run_left(self, tmp_path, depth):
        """A run before the set-aside left the provider's file where it was."""
        replies = {DEEP: _deep_reply(MAX_RESULT_DEPTH + 1), GOOD: json.dumps([SUGGESTION])}
        _plan_round(tmp_path, replies)
        stem = sanitize_model_name(DEEP)
        phase_dir = Path(base.get_phase_dir(tmp_path / "plan.md", "review-plan"))
        old = _old_file(depth, SUGGESTION)
        (phase_dir / f"{stem}.json").write_text(old, encoding="utf-8")

        exit_code, _ = _plan_round(tmp_path, replies, reaggregate=True)

        assert exit_code == 0
        _assert_the_reviewer_stayed_failed(
            phase_dir, _read(phase_dir, "grouped.json"), SUGGESTION["title"])
        assert _set_aside_texts(phase_dir, stem) == [old]


class TestReviewCodeSetsARefusedProviderFileAside:

    @staticmethod
    def _round(tmp_path, results, provider_writes):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        state = MagicMock()
        state.get = MagicMock(side_effect=lambda k, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_before_implement": "abc123",
            "head_at_start": "abc123",
            "pre_existing_changes": [],
        }.get(k, default))
        state.state = {"phases_completed": {}}
        state.is_phase_completed = MagicMock(return_value=False)

        async def fake_run_all_reviews(*_args, **_kwargs):
            for spec, text in provider_writes.items():
                (phase_dir / f"{sanitize_model_name(spec)}.json").write_text(
                    text, encoding="utf-8")
            return results

        with patch.object(cro, "parse_args",
                          return_value=_make_args(plan, models=list(results))), \
                patch.object(cro, "get_or_create_state", return_value=state), \
                patch("utils.interactive.resolve_models", return_value=list(results)), \
                patch("utils.interactive.is_model_valid", return_value=True), \
                patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
                patch.object(cro, "run_all_reviews", new=fake_run_all_reviews), \
                patch.object(cro, "get_phase_dir", return_value=phase_dir):
            assert asyncio.run(cro.main()) == 0
        return plan, phase_dir

    @staticmethod
    def _reaggregate(plan, phase_dir, capsys):
        capsys.readouterr()
        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        out = capsys.readouterr().out
        grouped_path = Path(out.split("Grouped issues saved to: ", 1)[1].splitlines()[0])
        return out, grouped_path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("depth", REFUSED_DEPTHS)
    def test_the_reviewer_stays_failed_through_a_reaggregate(self, tmp_path, capsys, depth):
        written = _deep_reply(depth, ISSUE)
        results = {
            DEEP: (True, _at_depth(ISSUE, depth), None),
            GOOD: (True, [dict(ISSUE)], None),
        }

        plan, phase_dir = self._round(tmp_path, results, {DEEP: written})
        stem = sanitize_model_name(DEEP)
        assert not (phase_dir / f"{stem}.json").exists()
        assert _set_aside_texts(phase_dir, stem) == [written]

        out, grouped_text = self._reaggregate(plan, phase_dir, capsys)

        assert "recorded as completed (was failed)" not in out
        _assert_the_reviewer_stayed_failed(phase_dir, grouped_text, ISSUE["title"])

    @pytest.mark.parametrize("depth", OLD_FILES)
    def test_a_reaggregate_refuses_a_file_an_older_run_left(self, tmp_path, capsys, depth):
        results = {
            DEEP: (True, _at_depth(ISSUE, MAX_RESULT_DEPTH + 1), None),
            GOOD: (True, [dict(ISSUE)], None),
        }
        plan, phase_dir = self._round(tmp_path, results, {})
        stem = sanitize_model_name(DEEP)
        old = _old_file(depth, ISSUE)
        (phase_dir / f"{stem}.json").write_text(old, encoding="utf-8")

        out, grouped_text = self._reaggregate(plan, phase_dir, capsys)

        assert f"{stem}.json is nested too deep to aggregate; not read" in out
        _assert_the_reviewer_stayed_failed(phase_dir, grouped_text, ISSUE["title"])
        assert _set_aside_texts(phase_dir, stem) == [old]


def test_an_unrecorded_file_too_deep_to_decode_is_left_unread(tmp_path, capsys):
    """Both reaggregates check every file no recorded model owns for a JSON
    list before reading anything; one deeper than the decoder's stack raised
    ``RecursionError`` out of that check."""
    from utils import result_files

    status_file.update_status(str(tmp_path), {"models_requested": [GOOD]})
    stray = tmp_path / "stray.json"
    stray.write_text(_old_file(None, ISSUE), encoding="utf-8")
    good = tmp_path / f"{sanitize_model_name(GOOD)}.json"
    good.write_text("[]", encoding="utf-8")

    kept = result_files.set_aside_unrecorded_results(tmp_path, [str(good), str(stray)])

    assert kept == [str(good)]
    assert stray.exists()
    assert "[RESULT_NOT_READ] stray.json is not a result file" in capsys.readouterr().out
