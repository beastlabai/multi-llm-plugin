"""A reviewer's answer whose JSON is broken is mended or salvaged, never lost.

Measured 2026-11-18, codex ``gpt-6-luna`` (2 of about 70 single-reviewer runs):
the reply was a normal, useful review, only its JSON was broken, and it came in
on stdout (``stdout_fallback``). The shared extractor could not parse the
findings array, served the last finding object inside it (its clause 1, with
``enclosing_syntax_error``), ``invoke_with_file_output`` passed that object on
as the answer, review-code failed its only model with "Expected JSON array, got
dict" and wrote no salvage request, and the run stopped "All models failed",
exit 3, even under ``--exec-subagents``. review-plan re-read the object's text
and, finding a ``[..]`` inside it, saved an empty review.

The two replies are kept verbatim under ``tests/fixtures/malformed_review_json/``
(the final ``agent_message`` event of each run's codex stream):

* ``stray_empty_key``: one finding ended ``"scope_reference": "…",\\n ""\\n }``,
  an empty key with no value (and another repeated ``scope_reference``, which
  parses: Python keeps the last).
* ``trailing_commas``: three findings ended ``"scope_reference": "…",\\n }``.

What holds now (docs/development/review-results.md, "Malformed result entries"):
an unambiguous break is mended (``json_extractor.repair_document``) and the
repair is said (``[JSON_REPAIRED]``, ``result_notes``, the log's ``Result
source:``); anything else fails the attempt with the whole reply attached, so
the ladder's retry and both lanes' salvage gates see all of it; and a whole
answer of the wrong shape on review-code goes to salvage as on the base.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
from utils import llm_client  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.json_extractor import extract_json_from_text, repair_document  # noqa: E402
from utils.provider_outcome import Capture, outcome_from_capture  # noqa: E402
from utils.providers.claude_code import ClaudeCodeProvider  # noqa: E402
from utils.providers.codex import CodexProvider  # noqa: E402
from utils.result_files import RESULT_NOTES_KEY  # noqa: E402
from utils.retry_ladder import STATUS_UNPARSEABLE  # noqa: E402
from utils.status_file import read_status, write_status  # noqa: E402

from tests.support.reaggregate_world import (  # noqa: E402
    _cro_phase, _reported_results, _run_cro_reaggregate,
)

FIXTURES = Path(__file__).parent / "fixtures" / "malformed_review_json"
CODEX = "codex:gpt-6-luna:high"
STEM = "codex_gpt-6-luna_high"

#: The findings each real reply holds, in order.
EMPTY_KEY_TITLES = [
    "Bulk discount misses exactly ten units",
    "Customers without a tier raise KeyError",
    "Line totals use floats and banker rounding",
    "Coupon expiry check is reversed",
    "Releasing stock increases reservations",
    "Long SKUs break invoice column alignment",
]
TRAILING_COMMA_TITLES = [
    "Repeated SKU replaces the existing line",
    "Bulk discount skips quantity ten",
    "Coupon amount is still taxed",
    "Reviewer-directed instruction in source",
]

#: The line on which the parser says each real reply breaks. The empty key
#: breaks at its closing ``}`` (line 14) on every Python. A trailing comma is
#: placed by the interpreter: Python 3.13 and later name the comma's own line
#: ("Illegal trailing comma before end of object", line 12), earlier versions
#: the closer after it ("Expecting property name enclosed in double quotes",
#: line 13).
SYNTAX_ERROR_LINES = {
    "stray_empty_key": 14,
    "trailing_commas": 12 if sys.version_info >= (3, 13) else 13,
}


def _stream(name: str) -> str:
    """The codex stdout of one real run: its final ``agent_message`` event."""
    return (FIXTURES / f"{name}.codex.jsonl").read_text(encoding="utf-8")


def _answer_text(name: str) -> str:
    return json.loads(_stream(name))["item"]["text"]


def _stream_of(text: str) -> str:
    """A codex stdout whose final ``agent_message`` is *text*."""
    return json.dumps({"type": "item.completed", "item": {
        "id": "item_9", "type": "agent_message", "text": text}}) + "\n"


def _unrepairable(name: str = "stray_empty_key") -> str:
    """The real reply with its empty key named: a key whose value is missing
    may be a lost field, so nothing mends it."""
    text = _answer_text(name)
    assert text.count('\n    ""\n  }') == 1
    return _stream_of(text.replace('\n    ""\n  }', '\n    "severity"\n  }'))


def _codex_exit_zero(stdout: str):
    """``invoke_with_provider`` for a codex run that exited 0 printing
    *stdout*, read by the real adapter and the real outcome reader."""
    def invoke(_prompt, _model_spec, timeout=None, _log_file=None, **kwargs):
        return outcome_from_capture(
            CodexProvider(), Capture(stdout=stdout, stderr="", exit_code=0),
            provider_name="codex", model="gpt-6-luna", timeout=timeout,
            defer_log=bool(kwargs.get("_defer_parse_failure_log")),
        )
    return invoke


def _invoke(tmp_path: Path, stdout: str, *, writes: str | None = None):
    """One ``invoke_with_file_output`` call over a codex exit 0; *writes* is
    what the model wrote to the requested file during the run."""
    phase_dir = tmp_path / "code-review"
    provider = _codex_exit_zero(stdout)

    def run(*args, **kwargs):
        if writes is not None:
            (phase_dir / f"{STEM}.json").write_text(writes, encoding="utf-8")
        return provider(*args, **kwargs)

    log = tmp_path / "log.txt"
    with patch.object(llm_client, "invoke_with_provider", side_effect=run):
        result = llm_client.invoke_with_file_output(
            prompt_template="Write to {output_json_path}", model_spec=CODEX,
            prompt_context={}, output_dir=tmp_path, phase="code_review",
            timeout=600, log_file=str(log),
        )
    return result, log


def _titles(findings):
    return [finding["title"] for finding in findings]


# ---------------------------------------------------------------------------
# The repair itself
# ---------------------------------------------------------------------------

class TestRepairDocument:
    @pytest.mark.parametrize("broken, mended", [
        ('{"a": 1,}', {"a": 1}),
        ('[1, 2, ]', [1, 2]),
        ('{"a": [1,], "b": {"c": 2,},}', {"a": [1], "b": {"c": 2}}),
        ('{"a": 1,\n  ""\n}', {"a": 1}),
        ('{""}', {}),
        ('[{"a": "x"},\n]', [{"a": "x"}]),
    ])
    def test_an_unambiguous_break_is_mended(self, broken, mended):
        text, notes = repair_document(broken)
        assert json.loads(text) == mended
        assert len(text) == len(broken), "every other character keeps its place"
        assert notes and all(note.startswith("line ") for note in notes)

    @pytest.mark.parametrize("broken", [
        '[,]',  # a comma after no value
        '{"a",}',  # after a key
        '[1,,]',  # a doubled comma
        '{"a": 1, "b"}',  # a named key with no value: a lost field?
        '{"a": 1, "", "b": 2}',  # an empty key before another member
        '{"a": "say "hi"",}',  # an unescaped quote breaks the string walk
        '{"a": 1,]',  # a closer that does not match
        '{"a": "unterminated,}',
    ])
    def test_anything_else_is_left_alone(self, broken):
        mended = repair_document(broken)
        if mended is not None:
            with pytest.raises(json.JSONDecodeError):
                json.loads(mended[0])

    def test_nothing_inside_a_string_is_touched(self):
        broken = '{"a": "x,}", "b": "\\",]",}'
        text, notes = repair_document(broken)
        assert json.loads(text) == {"a": "x,}", "b": '",]'}
        assert notes == ["line 1, column 25: removed a trailing comma before '}'"]

    def test_a_valid_document_needs_no_repair(self):
        assert repair_document('[{"a": [1, 2], "b": ""}]') is None

    def test_the_notes_name_line_and_column(self):
        _, notes = repair_document('[\n  {"a": 1,\n  },\n  {"b": 2,}\n]')
        assert notes == [
            "line 2, column 10: removed a trailing comma before '}'",
            "line 4, column 10: removed a trailing comma before '}'",
        ]


class TestTheExtractorOffersTheRepairedReading:
    @pytest.mark.parametrize("name, titles, changes", [
        ("stray_empty_key", EMPTY_KEY_TITLES, 1),
        ("trailing_commas", TRAILING_COMMA_TITLES, 3),
    ])
    def test_both_real_replies(self, name, titles, changes):
        result = extract_json_from_text(_answer_text(name))

        # What every caller read before is unchanged: the last finding,
        # with the document's own syntax error beside it.
        assert result["data"]["title"] == titles[-1]
        assert result["enclosing_syntax_error"]["line"] == SYNTAX_ERROR_LINES[name]
        # And the reply read whole, with what it took.
        assert _titles(result["repaired"]["data"]) == titles
        assert len(result["repaired"]["repairs"]) == changes

    def test_a_valid_reply_has_no_repaired_reading(self):
        result = extract_json_from_text('Here: [{"title": "x", "line_range": [1, 2]}]')
        assert "repaired" not in result and "enclosing_syntax_error" not in result

    def test_a_document_broken_otherwise_has_none(self):
        text = '[{"title": "A", "desc": "say "hi" twice"}, {"title": "B"}]'
        result = extract_json_from_text(text)
        assert result["enclosing_syntax_error"]
        assert "repaired" not in result

    def test_the_mended_reply_is_read_by_the_same_rules(self):
        """A wrapper mended in place serves what a valid wrapper serves."""
        text = 'Done.\n{"issues": [{"title": "A"}], "summary": "two",}'
        repaired = extract_json_from_text(text)["repaired"]
        assert repaired["data"] == extract_json_from_text(
            'Done.\n{"issues": [{"title": "A"}], "summary": "two"}')["data"]


# ---------------------------------------------------------------------------
# The fan-out's reader: invoke_with_file_output
# ---------------------------------------------------------------------------

class TestTheFanOutReader:
    @pytest.mark.parametrize("name, titles", [
        ("stray_empty_key", EMPTY_KEY_TITLES),
        ("trailing_commas", TRAILING_COMMA_TITLES),
    ])
    def test_a_mended_stdout_answer_is_the_whole_review(self, tmp_path, name, titles):
        result, log = _invoke(tmp_path, _stream(name))

        assert result["success"] is True
        assert result["source"] == "stdout_fallback"
        assert _titles(result["data"]) == titles
        assert result["json_repairs"]
        text = log.read_text(encoding="utf-8")
        assert "Result source: stdout_fallback (JSON repaired: line " in text

    def test_an_answer_nothing_can_mend_fails_with_the_whole_reply(self, tmp_path):
        result, log = _invoke(tmp_path, _unrepairable())

        assert result["success"] is False
        assert result["error"].startswith(
            "JSON parsing failed from both file and stdout (stdout: malformed JSON at line 14"
        )
        details = result["details"]
        # Unparseable: one retry on the ladder, then the slot's salvage gate,
        # which reads the reply's text, never the one finding served from it.
        assert details["failure_category"] == STATUS_UNPARSEABLE
        assert rob.attempt_ladder_status(result, details) == STATUS_UNPARSEABLE
        source = rob.salvage_source(details)
        assert all(title in source for title in EMPTY_KEY_TITLES)
        assert "Result source" not in log.read_text(encoding="utf-8")

    def test_a_scalar_piece_of_a_broken_array_is_not_an_answer(self, tmp_path):
        """Every finding broken beyond repair: the only piece that parses is a
        ``line_range``, which used to be served as the review."""
        text = (
            '[{"title": "A", "desc": "say "hi"", "line_range": [39, 39]},\n'
            ' {"title": "B", "desc": "say "no"", "line_range": [40, 41]}]'
        )
        assert extract_json_from_text(text)["data"] == [40, 41]
        result, _ = _invoke(tmp_path, _stream_of(text))
        assert result["success"] is False

    def test_a_broken_file_is_mended_too(self, tmp_path):
        result, log = _invoke(
            tmp_path, _stream_of("Written."), writes=_answer_text("trailing_commas"),
        )
        assert result["success"] is True
        assert result["source"] == "file_extracted"
        assert _titles(result["data"]) == TRAILING_COMMA_TITLES
        assert len(result["json_repairs"]) == 3
        assert "Result source: file_extracted (JSON repaired:" in log.read_text(
            encoding="utf-8")

    def test_a_broken_file_nothing_mends_leaves_stdout_to_answer(self, tmp_path):
        broken = json.loads(_unrepairable())["item"]["text"]
        result, _ = _invoke(tmp_path, _stream("trailing_commas"), writes=broken)
        assert result["success"] is True
        assert result["source"] == "stdout_fallback"
        assert _titles(result["data"]) == TRAILING_COMMA_TITLES
        assert result["file_error"].startswith(f"{STEM}.json holds malformed JSON at line 14")

    def test_a_valid_answer_carries_no_repair_key(self, tmp_path):
        good = json.dumps([{"title": "A", "line_range": [1, 2]}])
        result, _ = _invoke(tmp_path, _stream_of(good))
        assert result["success"] is True
        assert "json_repairs" not in result


# ---------------------------------------------------------------------------
# Both lanes, one slot each
# ---------------------------------------------------------------------------

SANDBOXED = rob.SandboxVerdict(rob.SANDBOX_DISPATCH, None)


@pytest.fixture
def lane_env(tmp_path, monkeypatch):
    """A scratch home for codex, and the sandbox gate pinned to a sandboxed
    dispatch, so a slot runs offline and touches nothing outside *tmp_path*."""
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: SANDBOXED)
    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n", encoding="utf-8")
    return plan


def _writing(stdout: str, result_file: Path, writes: str | None):
    """``invoke_with_provider`` for a codex exit 0 that wrote *writes* to the
    requested *result_file* during the run (nothing when ``None``)."""
    provider = _codex_exit_zero(stdout)

    def run(*args, **kwargs):
        if writes is not None:
            result_file.write_text(writes, encoding="utf-8")
        return provider(*args, **kwargs)
    return run


def _review_code(plan: Path, stdout: str, *, writes: str | None = None):
    """review-code's slot for one codex model over *stdout* (and *writes* in
    its result file); also returns its phase dir."""
    result_file = plan.parent / "out" / "code-review" / f"{STEM}.json"
    with patch.object(llm_client, "invoke_with_provider",
                      side_effect=_writing(stdout, result_file, writes)), \
         patch.object(cro, "get_project_root", lambda _p: str(plan.parent)):
        outcome = asyncio.run(cro.run_single_review(
            CODEX, plan, plan.parent / "out", timeout=600, run_model_specs=[CODEX],
        ))
    return outcome, Path(cro.get_phase_dir(plan, "code-review"))


def _review_plan(plan: Path, stdout: str):
    """review-plan's slot for the same, then the save the fan-out makes;
    returns the slot's record, the save's verdict and the phase dir."""
    with patch.object(llm_client, "invoke_with_provider",
                      side_effect=_codex_exit_zero(stdout)), \
         patch.object(rob, "get_project_root", lambda _p: str(plan.parent)), \
         patch.object(rob, "get_provider_timeout", lambda *_a, **_k: 600):
        record = asyncio.run(rob.run_single_model(
            model_spec=CODEX, plan_path=str(plan),
            prompt_template="{prefix}{output_json_path}", prompt_context={},
            phase_name="review-plan", invoke_phase="plan_review", timeout=600,
            out_dir=str(plan.parent), prefix="plan",
        ))
    phase_dir = Path(rob.get_phase_dir(plan, "review-plan"))
    saved = rob.save_model_result(
        "plan", CODEX, record["success"], record["output"], record["error"],
        str(plan.parent), str(phase_dir), "review_plan",
        failure_output=rob.failure_output_from_result(record),
        failure=record.get("failure"),
    )
    return record, saved, phase_dir


def _plan_reply(*, broken: bool) -> str:
    """A review-plan answer in codex's stream, its first suggestion ending in
    the empty key of the 2026-11-18 reply when *broken*."""
    tail = ',\n    ""\n  }' if broken else "\n  }"
    return _stream_of(
        '[\n  {\n    "title": "Name the retry budget",\n    "desc": "d",\n'
        '    "importance": "high",\n    "reference": "Goal",\n'
        f'    "type": "clarification"{tail},\n'
        '  {"title": "Drop the cache", "desc": "d", "importance": "low",'
        ' "reference": "Step 2", "type": "deletion"}\n]'
    )


class TestReviewCodeSlot:
    @pytest.mark.parametrize("name, titles", [
        ("stray_empty_key", EMPTY_KEY_TITLES),
        ("trailing_commas", TRAILING_COMMA_TITLES),
    ])
    def test_the_real_reply_completes_with_every_finding(
        self, lane_env, capsys, name, titles,
    ):
        (success, issues, error), phase_dir = _review_code(lane_env, _stream(name))

        assert (success, error) == (True, None)
        assert _titles(issues) == titles
        out = capsys.readouterr().out
        assert f"[JSON_REPAIRED] {CODEX}'s answer was not valid JSON" in out
        notes = read_status(str(phase_dir))[RESULT_NOTES_KEY]
        assert any(note.startswith(f"{CODEX}'s answer was not valid JSON") for note in notes)
        assert not list(phase_dir.glob("salvage_*.json"))

    def test_a_reply_nothing_mends_goes_to_salvage_whole(self, lane_env, capsys):
        (success, issues, error), phase_dir = _review_code(lane_env, _unrepairable())

        assert success is False and issues is None
        assert "malformed JSON at line 14" in error
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert all(title in request["raw_output"] for title in EMPTY_KEY_TITLES)
        assert (phase_dir / f"salvage_task_{STEM}.json").exists()
        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out

    def test_a_whole_answer_of_another_shape_goes_to_salvage(self, lane_env, capsys):
        """One finding where an array was asked for: a whole document, so it
        is not mended, and salvage repackages it instead of the slot failing
        with nothing requested."""
        one = {"title": "Only finding", "desc": "d", "importance": "high",
               "file": "src/a.py", "line_range": [3, 4], "type": "bug"}
        (success, _issues, error), phase_dir = _review_code(
            lane_env, _stream_of(json.dumps(one)))

        assert success is False
        assert error == "Expected JSON array, got dict"
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert json.loads(request["raw_output"]) == one
        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out

    def test_an_object_holding_nothing_answer_shaped_is_not_salvaged(self, lane_env, capsys):
        (success, _issues, error), phase_dir = _review_code(
            lane_env, _stream_of('{"status": "done"}'))

        assert success is False
        assert error == "Expected JSON array, got dict"
        assert not list(phase_dir.glob("salvage_*.json"))
        assert "nothing to salvage" in capsys.readouterr().out


class TestReviewPlanSlot:
    """The base lane's same two endings, through its slot and its save."""

    def test_a_mended_reply_saves_every_suggestion(self, lane_env, capsys):
        record, saved, phase_dir = _review_plan(lane_env, _plan_reply(broken=True))

        assert record["success"] is True and saved is True
        stored = json.loads((phase_dir / f"{STEM}.json").read_text(encoding="utf-8"))
        assert _titles(stored) == ["Name the retry budget", "Drop the cache"]
        assert f"[JSON_REPAIRED] {CODEX}'s answer" in capsys.readouterr().out
        assert read_status(str(phase_dir))[RESULT_NOTES_KEY]

    def test_a_reply_nothing_mends_goes_to_salvage_whole(self, lane_env):
        broken = json.loads(_plan_reply(broken=True))["item"]["text"].replace(
            '\n    ""\n  }', '\n    "severity"\n  }')
        record, saved, phase_dir = _review_plan(lane_env, _stream_of(broken))

        assert record["success"] is False and saved is False
        # Never the empty review the re-read piece used to save.
        assert not (phase_dir / f"{STEM}.json").exists()
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert "Name the retry budget" in request["raw_output"]
        assert "Drop the cache" in request["raw_output"]

    def test_a_valid_reply_is_unchanged(self, lane_env, capsys):
        record, saved, phase_dir = _review_plan(lane_env, _plan_reply(broken=False))

        assert record["success"] is True and saved is True
        assert "[JSON_REPAIRED]" not in capsys.readouterr().out
        assert RESULT_NOTES_KEY not in read_status(str(phase_dir))


# ---------------------------------------------------------------------------
# Several reviewers: the broken one is mended or salvaged, the others untouched
# ---------------------------------------------------------------------------

GOOD = "codex:gpt-5.5"
GOOD_ISSUES = [{"title": "Sound finding", "desc": "d", "importance": "LOW",
                "file": "src/a.py", "line_range": [1, 2], "type": "style"}]


def _by_spec(streams):
    """``invoke_with_provider`` answering each model spec with its own stream."""
    readers = {spec: _codex_exit_zero(stdout) for spec, stdout in streams.items()}

    def invoke(prompt, model_spec, *args, **kwargs):
        return readers[model_spec](prompt, model_spec, *args, **kwargs)
    return invoke


@pytest.mark.parametrize("broken, mended", [
    (lambda: _stream("trailing_commas"), True),
    (_unrepairable, False),
], ids=["mended", "salvaged"])
def test_a_fan_out_of_two_keeps_the_valid_reviewer_as_it_was(lane_env, capsys, broken, mended):
    plan = lane_env
    streams = {CODEX: broken(), GOOD: _stream_of(json.dumps(GOOD_ISSUES))}
    with patch.object(llm_client, "invoke_with_provider", side_effect=_by_spec(streams)), \
         patch.object(cro, "get_project_root", lambda _p: str(plan.parent)), \
         patch("utils.concurrency.PROVIDER_STAGGER_DELAY", 0.0):
        results = asyncio.run(cro.run_all_reviews(
            [CODEX, GOOD], plan, plan.parent / "out", 600, 2,
        ))

    assert results[GOOD] == (True, GOOD_ISSUES, None)
    phase_dir = Path(cro.get_phase_dir(plan, "code-review"))
    out = capsys.readouterr().out
    if mended:
        assert results[CODEX][0] is True
        assert _titles(results[CODEX][1]) == TRAILING_COMMA_TITLES
        assert f"[JSON_REPAIRED] {CODEX}'s answer" in out
        assert f"[JSON_REPAIRED] {GOOD}" not in out
        assert not list(phase_dir.glob("salvage_*.json"))
    else:
        assert results[CODEX][0] is False
        assert "malformed JSON" in results[CODEX][2]
        assert sorted(p.name for p in phase_dir.glob("salvage_*.json")) == [
            f"salvage_{STEM}.json", f"salvage_task_{STEM}.json",
        ]
        assert "[JSON_REPAIRED]" not in out


# ---------------------------------------------------------------------------
# Without --exec-subagents: the host salvages, then --reaggregate
# ---------------------------------------------------------------------------

def test_a_host_salvaged_single_reviewer_reaggregates(tmp_path):
    """The run itself exits 3 "All models failed" (no in-process salvage
    without ``--exec-subagents``), leaving the request on disk; the host
    runs the salvage task, which writes the model's own result file, and the
    documented ``--reaggregate`` then reads it as the review's one result."""
    plan, phase_dir = _cro_phase(tmp_path, {})
    error = ("JSON parsing failed from both file and stdout "
             "(stdout: malformed JSON at line 14, column 3: Expecting ':' delimiter)")
    write_status(str(phase_dir), {
        "phase": "code-review",
        "state": "models_running",
        "started_at": "2026-11-18T20:35:02",
        "models_requested": [CODEX],
        "models_completed": [],
        "models_failed": {CODEX: error},
        "error": None,
    })
    salvaged = extract_json_from_text(_answer_text("stray_empty_key"))["repaired"]["data"]
    (phase_dir / f"{STEM}.json").write_text(json.dumps(salvaged), encoding="utf-8")
    outcome = {}

    report = _run_cro_reaggregate(plan, phase_dir, outcome=outcome)

    assert outcome["exit"] == 0
    results = _reported_results(report)
    assert results[CODEX][0] is True
    assert _titles(results[CODEX][1]) == EMPTY_KEY_TITLES
    status = read_status(str(phase_dir))
    assert status["models_completed"] == [CODEX]
    assert CODEX not in status.get("models_failed", {})


# ---------------------------------------------------------------------------
# Fable review 2026-11-19: a truncated reply, a broken result file, a nested
# array, and a reply in which nothing parses
# ---------------------------------------------------------------------------

ALPHA = {"title": "Alpha", "desc": "first", "importance": "high",
         "file": "src/a.py", "line_range": [1, 2], "type": "bug"}
BETA = {"title": "Beta", "desc": "second", "importance": "low",
        "file": "src/b.py", "line_range": [3, 4], "type": "style"}
#: Cut off inside the third finding, as a reply that hit its output limit.
TRUNCATED = (
    "[\n" + json.dumps(ALPHA) + ",\n" + json.dumps(BETA)
    + ',\n{"title": "Gamma", "desc": "cut off he'
)
#: Two findings and no comma between them; the first carries a nested array
#: of objects, the only thing in the reply that parses as a list.
REFS = {**ALPHA, "refs": [{"file": "a.py"}, {"file": "b.py"}]}
MISSING_COMMA = "[\n" + json.dumps(REFS) + "\n" + json.dumps(BETA) + "\n]"
#: The two findings written broken beyond repair (unescaped quotes), so no
#: piece of the document parses at all.
UNESCAPED = (
    '[{"title": "Alpha", "desc": "say "hi"", "importance": "high", '
    '"file": "src/a.py", "type": "bug"},\n'
    ' {"title": "Beta", "desc": "say "no"", "importance": "low", '
    '"file": "src/b.py", "type": "style"}]'
)


#: The two findings with no nested array: a ``line_range`` parses on its own,
#: and a reply holding one is the fragment path's, not this one's.
FLAT = [{key: value for key, value in finding.items() if key != "line_range"}
        for finding in (ALPHA, BETA)]


def _trailing_comma_in_every_finding() -> str:
    """The two flat findings, each ending in a trailing comma: no piece of
    the reply parses at all."""
    return "[\n" + ",\n".join(json.dumps(finding)[:-1] + ",}" for finding in FLAT) + "\n]"


class TestTheExtractorSeesATruncatedDocument:
    def test_a_piece_of_a_document_that_never_closes_says_so(self):
        result = extract_json_from_text(TRUNCATED)

        # Served as before, for the readers that want something...
        assert result["data"] == BETA
        # ...but no longer as if it stood alone.
        error = result["enclosing_syntax_error"]
        assert "never closed" in error["message"]
        assert "repaired" not in result

    def test_a_whole_array_before_a_cut_off_trailer_is_served_clean(self):
        text = json.dumps([ALPHA, BETA]) + '\nNotes: {"extra": "cut'
        result = extract_json_from_text(text)
        assert result["data"] == [ALPHA, BETA]
        assert "enclosing_syntax_error" not in result

    def test_a_bracket_from_a_log_line_or_an_escape_is_not_a_document(self):
        text = "\x1b[0m[INFO] done\n" + json.dumps([ALPHA])
        result = extract_json_from_text(text)
        assert result["data"] == [ALPHA]
        assert "enclosing_syntax_error" not in result

    def test_a_truncation_is_never_mended_into_a_shorter_review(self):
        """A trailing comma in the first finding is repairable; the cut-off
        array around it is not, so no repaired reading is offered."""
        text = '[{"title": "A", "desc": "x", "importance": "high",}, {"title": "B", "de'
        result = extract_json_from_text(text)
        assert "repaired" not in result


class TestTheExtractorMendsAReplyInWhichNothingParses:
    @pytest.mark.parametrize("text, titles", [
        ('[{"title": "A", "desc": "x",}]', ["A"]),
        ('[{"title": "A", "desc": "x",\n  ""\n}]', ["A"]),
        (_trailing_comma_in_every_finding(), ["Alpha", "Beta"]),
    ], ids=["one-finding", "one-finding-empty-key", "every-finding"])
    def test_the_failure_carries_the_mended_reading(self, text, titles):
        result = extract_json_from_text(text)

        # Every caller that does not ask reads the failure it read before:
        # the executor lane still retries such a reply.
        assert result["success"] is False and result["data"] is None
        assert _titles(result["repaired"]["data"]) == titles
        assert result["repaired"]["repairs"]

    def test_a_reply_nothing_mends_carries_no_repaired_reading(self):
        assert "repaired" not in extract_json_from_text(UNESCAPED)


class TestTheFanOutReaderOnTheReview:
    def test_a_truncated_reply_fails_with_its_whole_text(self, tmp_path):
        result, _ = _invoke(tmp_path, _stream_of(TRUNCATED))

        assert result["success"] is False
        assert "never closed" in result["error"]
        details = result["details"]
        assert rob.attempt_ladder_status(result, details) == STATUS_UNPARSEABLE
        source = rob.salvage_source(details)
        assert all(word in source for word in ("Alpha", "Beta", "Gamma"))

    def test_a_truncated_wrapper_still_serves_its_whole_findings_array(self, tmp_path):
        text = '{"issues": ' + json.dumps([ALPHA, BETA]) + ', "summary": "cut off'
        result, _ = _invoke(tmp_path, _stream_of(text))
        assert result["success"] is True
        assert result["data"] == [ALPHA, BETA]

    def test_a_nested_array_inside_a_broken_reply_is_not_the_review(self, tmp_path):
        assert extract_json_from_text(MISSING_COMMA)["data"] == REFS["refs"]
        result, _ = _invoke(tmp_path, _stream_of(MISSING_COMMA))

        assert result["success"] is False
        source = rob.salvage_source(result["details"])
        assert "Alpha" in source and "Beta" in source

    def test_a_broken_finding_does_not_give_its_refs_as_a_wrapper_value(self, tmp_path):
        text = (
            '{"title": "Alpha", "desc": "say "hi"", "importance": "high", '
            '"refs": [{"file": "a.py"}, {"file": "b.py"}]}'
        )
        result, _ = _invoke(tmp_path, _stream_of(text))
        assert result["success"] is False

    def test_a_broken_wrapper_around_whole_findings_is_still_served(self, tmp_path):
        text = '{"issues": ' + json.dumps([ALPHA, BETA]) + ', "summary": "say "hi""}'
        result, _ = _invoke(tmp_path, _stream_of(text))
        assert result["success"] is True
        assert result["data"] == [ALPHA, BETA]

    def test_a_trailing_comma_in_every_finding_is_mended(self, tmp_path):
        result, log = _invoke(tmp_path, _stream_of(_trailing_comma_in_every_finding()))

        assert result["success"] is True
        assert result["source"] == "stdout_fallback"
        assert result["data"] == FLAT
        assert len(result["json_repairs"]) == 2
        assert "Result source: stdout_fallback (JSON repaired: line " in log.read_text(
            encoding="utf-8")

    def test_a_file_with_a_trailing_comma_in_every_finding_is_mended(self, tmp_path):
        result, log = _invoke(
            tmp_path, _stream_of("Written."), writes=_trailing_comma_in_every_finding(),
        )
        assert result["success"] is True
        assert result["source"] == "file_extracted"
        assert result["data"] == FLAT
        assert "Result source: file_extracted (JSON repaired:" in log.read_text(
            encoding="utf-8")


class TestABrokenResultFileReachesSalvage:
    """The model wrote its review to the requested file, broken, and printed
    only prose: the file's text is what salvage reads."""

    @pytest.mark.parametrize("broken", [
        UNESCAPED,  # nothing in it parses
        MISSING_COMMA,  # a piece parses, and is refused
    ], ids=["no-piece", "refused-piece"])
    def test_the_file_text_is_the_salvage_text(self, tmp_path, broken):
        result, _ = _invoke(
            tmp_path, _stream_of("I wrote the review to the file."), writes=broken,
        )

        assert result["success"] is False
        assert result["file_error"]
        details = result["details"]
        source = rob.salvage_source(details)
        assert source == broken
        assert "I wrote the review" in details["stdout_excerpt"]

    def test_the_file_wins_over_a_damaged_stdout_answer(self, tmp_path):
        """Both channels hold a broken answer: the file is read first, so it
        is salvaged, as it is served first when both parse."""
        result, _ = _invoke(tmp_path, _stream_of(TRUNCATED), writes=UNESCAPED)

        assert result["success"] is False
        assert rob.salvage_source(result["details"]) == UNESCAPED

    def test_the_file_wins_over_a_clean_result_envelope(self, tmp_path):
        """An adapter with no salvage hook leaves its whole stdout as the
        excerpt; claude-code's clean envelope carries ``permission_denials:
        []``, which reads as an answer to repair around a prose result."""
        envelope = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "I wrote the review to the requested file.",
            "permission_denials": [], "usage": {"iterations": []},
        })
        spec, stem = "claude-code:claude-sonnet-5", "claude-code_claude-sonnet-5"
        phase_dir = tmp_path / "code-review"

        def run(_prompt, _model_spec, timeout=None, _log_file=None, **kwargs):
            (phase_dir / f"{stem}.json").write_text(UNESCAPED, encoding="utf-8")
            return outcome_from_capture(
                ClaudeCodeProvider(), Capture(stdout=envelope, stderr="", exit_code=0),
                provider_name="claude-code", model="claude-sonnet-5", timeout=timeout,
                defer_log=bool(kwargs.get("_defer_parse_failure_log")),
            )

        with patch.object(llm_client, "invoke_with_provider", side_effect=run):
            result = llm_client.invoke_with_file_output(
                prompt_template="Write to {output_json_path}", model_spec=spec,
                prompt_context={}, output_dir=tmp_path, phase="code_review",
                timeout=600, log_file=str(tmp_path / "log.txt"),
            )

        assert result["success"] is False
        details = result["details"]
        assert rob.salvage_source(details) == UNESCAPED
        assert "permission_denials" in details["stdout_excerpt"]

    def test_a_file_holding_no_answer_leaves_the_stdout_text(self, tmp_path):
        result, _ = _invoke(
            tmp_path, _stream_of("I could not finish."), writes="Nothing here yet.",
        )
        assert result["success"] is False
        assert rob.salvage_source(result["details"]) == "I could not finish."

    def test_review_code_requests_salvage_over_the_file(self, lane_env, capsys):
        (success, _issues, _error), phase_dir = _review_code(
            lane_env, _stream_of("Written to the requested file."), writes=UNESCAPED,
        )

        assert success is False
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert request["raw_output"] == UNESCAPED
        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out


def _plan_suggestion(title: str) -> dict:
    return {"title": title, "desc": "d", "importance": "high",
            "reference": "Goal", "type": "clarification"}


class TestBothLanesSalvageTheWholeReply:
    """A truncated reply, a nested array and a reply of two objects: each slot
    fails, and its salvage request holds every finding the reply did."""

    def test_review_code_truncated(self, lane_env, capsys):
        (success, issues, error), phase_dir = _review_code(lane_env, _stream_of(TRUNCATED))

        assert success is False and issues is None
        assert "never closed" in error
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert all(word in request["raw_output"] for word in ("Alpha", "Beta", "Gamma"))
        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out

    def test_review_code_nested_array(self, lane_env, capsys):
        (success, issues, _error), phase_dir = _review_code(
            lane_env, _stream_of(MISSING_COMMA))

        assert success is False and issues is None
        assert "Found 2 issues" not in capsys.readouterr().out
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert "Alpha" in request["raw_output"] and "Beta" in request["raw_output"]

    def test_review_code_two_objects(self, lane_env):
        """No array at all: the last object is served, whole and valid, so it
        reaches the wrong-shape ending; the request carries both objects."""
        text = json.dumps(ALPHA) + "\n" + json.dumps(BETA)
        (success, _issues, error), phase_dir = _review_code(lane_env, _stream_of(text))

        assert (success, error) == (False, "Expected JSON array, got dict")
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert request["raw_output"] == text

    def test_review_code_a_mended_reply_with_nothing_parsing(self, lane_env, capsys):
        (success, issues, error), phase_dir = _review_code(
            lane_env, _stream_of(_trailing_comma_in_every_finding()))

        assert (success, error) == (True, None)
        assert _titles(issues) == ["Alpha", "Beta"]
        assert f"[JSON_REPAIRED] {CODEX}'s answer" in capsys.readouterr().out
        assert not list(phase_dir.glob("salvage_*.json"))

    def test_review_plan_truncated(self, lane_env):
        text = (
            "[\n" + json.dumps(_plan_suggestion("Name the retry budget")) + ",\n"
            + json.dumps(_plan_suggestion("Drop the cache")) + ',\n{"title": "Split'
        )
        record, saved, phase_dir = _review_plan(lane_env, _stream_of(text))

        assert record["success"] is False and saved is False
        # Never the one suggestion the piece held.
        assert not (phase_dir / f"{STEM}.json").exists()
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        for title in ("Name the retry budget", "Drop the cache", "Split"):
            assert title in request["raw_output"]

    def test_review_plan_two_objects(self, lane_env):
        text = (
            json.dumps(_plan_suggestion("Name the retry budget")) + "\n"
            + json.dumps(_plan_suggestion("Drop the cache"))
        )
        record, saved, phase_dir = _review_plan(lane_env, _stream_of(text))

        assert record["success"] is True and saved is False
        assert not (phase_dir / f"{STEM}.json").exists()
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert request["raw_output"] == text


# ---------------------------------------------------------------------------
# Fable review 2, N1 and N5: a bracket in prose before a clean answer, and a
# second broken document beside the mended one
# ---------------------------------------------------------------------------

#: Prose whose ``[[`` opens like a structure and never closes.
CHAR_CLASS = 'The class "[[a-z]" is wrong, nothing else.\n'
#: Two findings each ending in a trailing comma: a ``line_range`` in each
#: still parses, so the extractor serves a piece (the fragment path).
TWO_BROKEN_ARRAYS = (
    "[\n" + json.dumps(ALPHA)[:-1] + ",}\n]\nAnd again:\n[\n"
    + json.dumps(BETA)[:-1] + ",}\n]"
)
#: The same with no nested array, so nothing parses at all.
TWO_BROKEN_FLAT_ARRAYS = (
    "[\n" + json.dumps(FLAT[0])[:-1] + ",}\n]\nAnd again:\n[\n"
    + json.dumps(FLAT[1])[:-1] + ",}\n]"
)
#: Prose brackets that are no answer: ``[see above]`` and ``lines [12-15]``
#: open like no structure, ``[[:alpha:]]`` and the code literal
#: ``{"error": err}`` name no answer field, ``{and so on}`` is prose in braces.
PROSE_BRACKETS = (
    'See [see above], lines [12-15], [[:alpha:]] {and so on}.\n'
    'It returns {"error": err} on failure.\n'
)


class TestABracketInProseIsNoDocument:
    @pytest.mark.parametrize("answer", [
        [],
        [{"title": "A", "desc": "x"}],  # no importance: not records to _holds_records
        {"issues": []},
    ], ids=["empty", "not-records", "object"])
    def test_a_clean_answer_after_it_is_served_clean(self, answer):
        result = extract_json_from_text(CHAR_CLASS + json.dumps(answer))

        assert result["data"] == answer
        assert "enclosing_syntax_error" not in result

    @pytest.mark.parametrize("prose", [
        CHAR_CLASS,
        "The class [[a-z] is wrong.\n",
        'Use ["a", "b" in the call.\n',
        'The config {"debug" flag is ignored.\n',
        "Wiki style [[Page] link.\n",
        # A bracket in quotes: the closing quote opens a string nothing closes,
        # straight after the opener, where no value stands.
        'The token "{" is unexpected.\n',
        'Expected "[" here.\n',
        'A stray "[{" opens it.\n',
        'Expected "[" or ",".\n',
    ])
    def test_prose_of_any_bracket_leaves_the_empty_review_whole(self, prose):
        result = extract_json_from_text(prose + "[]")
        assert result["data"] == [] and "enclosing_syntax_error" not in result

    def test_the_fan_out_takes_the_clean_answer(self, tmp_path):
        result, _ = _invoke(tmp_path, _stream_of(CHAR_CLASS + "[]"))
        assert result["success"] is True
        assert result["data"] == []

    def test_a_lone_reviewer_completes_with_no_findings(self, lane_env):
        (success, issues, error), phase_dir = _review_code(
            lane_env, _stream_of(CHAR_CLASS + "[]"))

        assert (success, issues, error) == (True, [], None)
        assert not list(phase_dir.glob("salvage_*.json"))

    def test_a_truncation_after_such_prose_still_never_closes(self):
        """The prose opener is passed over for the document's own ``[``."""
        result = extract_json_from_text(CHAR_CLASS + TRUNCATED)

        assert result["data"] == BETA
        error = result["enclosing_syntax_error"]
        assert "never closed" in error["message"]
        assert error["line"] == 4, "read from the document's opener, not the prose's"

    def test_a_string_the_truncation_cut_is_not_prose(self):
        result = extract_json_from_text(
            '[{"title": "B", "desc": "see rows[0] and then it was cut')

        assert result["data"] == [0]
        assert "never closed" in result["enclosing_syntax_error"]["message"]

    def test_an_unescaped_quote_in_the_cut_off_finding_is_not_prose(self, tmp_path):
        """``"say "hi" there"`` reads as one string, so its ``line_range`` is
        still a piece of a truncated finding, never the review."""
        text = '[{"title": "B", "desc": "say "hi" there", "line_range": [3, 4], "impor'
        result = extract_json_from_text(text)
        assert result["data"] == [3, 4]
        assert "never closed" in result["enclosing_syntax_error"]["message"]

        answer, _ = _invoke(tmp_path, _stream_of(text))
        assert answer["success"] is False

    def test_the_documented_miss_an_unescaped_quote_before_a_space(self):
        """docs/development/review-results.md names it: a quote a space
        follows reads as the string's end, ``screen`` as prose, and the
        ``line_range`` is served clean, as before the never-closed rule."""
        text = '[{"title": "B", "desc": "a 5" screen", "line_range": [3, 4], "impor'
        result = extract_json_from_text(text)
        assert result["data"] == [3, 4]
        assert "enclosing_syntax_error" not in result

    def test_a_missing_comma_is_damage_not_prose(self):
        text = (
            "[\n" + json.dumps(ALPHA) + "\n" + json.dumps(BETA)
            + ',\n{"title": "Gamma", "desc": "cut'
        )
        result = extract_json_from_text(text)
        assert result["data"] == BETA
        assert "never closed" in result["enclosing_syntax_error"]["message"]


class TestAnotherBrokenDocumentRefusesTheMend:
    @pytest.mark.parametrize("text", [TWO_BROKEN_ARRAYS, TWO_BROKEN_FLAT_ARRAYS],
                             ids=["piece-served", "nothing-parses"])
    def test_two_broken_arrays_offer_no_mended_reading(self, text):
        assert "repaired" not in extract_json_from_text(text)

    @pytest.mark.parametrize("text, unmendable", [
        (TWO_BROKEN_ARRAYS, MISSING_COMMA),
        (TWO_BROKEN_FLAT_ARRAYS, UNESCAPED),
    ], ids=["piece-served", "nothing-parses"])
    def test_the_fan_out_fails_with_both_documents_for_salvage(
        self, tmp_path, text, unmendable,
    ):
        """It ends as a reply nothing mends ends: a refused piece is
        unparseable, a reply in which nothing parses is the runner's."""
        result, _ = _invoke(tmp_path / "two", _stream_of(text))
        reference, _ = _invoke(tmp_path / "one", _stream_of(unmendable))

        assert result["success"] is False
        details = result["details"]
        assert rob.attempt_ladder_status(result, details) == rob.attempt_ladder_status(
            reference, reference["details"])
        source = rob.salvage_source(details)
        assert "Alpha" in source and "Beta" in source

    def test_review_code_requests_salvage_over_both(self, lane_env):
        (success, issues, _error), phase_dir = _review_code(
            lane_env, _stream_of(TWO_BROKEN_ARRAYS))

        assert success is False and issues is None
        request = json.loads(
            (phase_dir / f"salvage_{STEM}.json").read_text(encoding="utf-8"))
        assert "Alpha" in request["raw_output"] and "Beta" in request["raw_output"]

    @pytest.mark.parametrize("findings, expected", [
        ((ALPHA, BETA), [ALPHA, BETA]),
        (tuple(FLAT), FLAT),
    ], ids=["piece-served", "nothing-parses"])
    def test_one_broken_document_among_prose_brackets_is_still_mended(
        self, findings, expected,
    ):
        text = (
            PROSE_BRACKETS + "[\n"
            + ",\n".join(json.dumps(finding)[:-1] + ",}" for finding in findings)
            + "\n]\nDone."
        )
        repaired = extract_json_from_text(text)["repaired"]
        assert repaired["data"] == expected
        assert len(repaired["repairs"]) == 2

    def test_a_cut_off_trailer_does_not_refuse_the_mend(self):
        """A trailer that never closes is no document beside the mended one,
        as a whole array before it is served clean."""
        text = _trailing_comma_in_every_finding() + '\nNotes: {"extra": "cut'
        assert extract_json_from_text(text)["repaired"]["data"] == FLAT


class TestTheOpenerSearchStaysLinear:
    """Every opener turned down is passed for the next one after the prose
    that turned it down, so the readings never overlap. Each input reaches
    the search through a served ``[]`` after thousands of unclosed openers;
    re-reading from each would take minutes."""

    CEILING_S = 5.0

    @pytest.mark.parametrize("text", [
        '{"a":' * 10_000 + " x\n[]",
        "[ " + '"[{" , ' * 20_000 + "x\n[]",
        "[ " + '"a"b"[{" , ' * 15_000 + "x\n[]",
        CHAR_CLASS * 3_000 + "[]",
        '[{"a": 1, ' * 10_000 + "x\n[]",
    ], ids=["unclosed-objects", "openers-in-strings", "glued-quotes",
            "prose-openers", "unclosed-records"])
    def test_a_served_value_after_many_unclosed_openers(self, text):
        started = time.perf_counter()
        result = extract_json_from_text(text)
        elapsed = time.perf_counter() - started

        assert result["data"] == [] and "enclosing_syntax_error" not in result
        assert elapsed < self.CEILING_S, (
            f"{len(text)} characters took {elapsed:.1f}s: the opener search "
            f"re-reads the reply (see _never_closed_opener)")
