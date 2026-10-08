"""A write reply with prose before its JSON object is read as that object
(fix round 28, live run 12 #6).

Codex decline batch 0 (`/tmp/mllm-race-r12/decline-codex/`,
`failed/apply-fix-batch-0.attempt1.json`): the final agent message opened
with two prose lines ("Fix 1: … Fix 2: …"), then the report object. The codex
adapter's fallback extraction prefers arrays, and an array holding an object
ranks as an answer, so it served the report's own `declined` array; the
batch read "<root>: expected type object, got list" and cost a repair
re-ask that restated the same report. The extractor now names the reply's
one JSON object outside every other document (`only_object`), and both
invokers read a kind whose result is an object through it
(`subagent_executor.the_reply_object`). A reply holding two such objects
names none, so it keeps its re-ask, as a validation reply holding two does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.json_extractor import extract_json_from_text  # noqa: E402
from utils.providers.codex import CodexProvider  # noqa: E402

REASON = "The implementation record says: “the catch-up boost may exceed it.”"
REPORT = {
    "status": "applied",
    "ids": ["c2e33cbae17e32bc"],
    "declined": [{"id": "c8fe719158c8d2be", "reason": REASON}],
    "files_touched": ["src/opponent.js"],
    "summary": "Updated standings labels to use opponent.driver; left the planned "
               "catch-up speed behavior unchanged.",
    "notes": "Fix 1: Left unchanged per the implementation record’s catch-up speed "
             "decision. Fix 2: Changed opponentLabel to read opponent.driver.",
    "check": {"passed": True, "findings": [],
              "ran": {"node --check src/opponent.js": "passed", "npm test": "passed"},
              "listed": {"node --check src/opponent.js": "passed", "npm test": "passed"}},
}
PROSE = (
    "Fix 1: Left unchanged because the implementation record says the catch-up "
    "boost may exceed `MAX_SPEED`.\n"
    "Fix 2: Changed the standings label to use `opponent.driver`.\n\n"
)
#: The run's final agent message, word for word.
FINAL_TEXT = PROSE + json.dumps(REPORT, ensure_ascii=False, separators=(",", ":"))


def _codex_stream(*messages: str) -> str:
    events = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"}]
    events += [
        {"type": "item.completed",
         "item": {"id": f"item_{n}", "type": "agent_message", "text": text}}
        for n, text in enumerate(messages)
    ]
    events.append({"type": "turn.completed",
                   "usage": {"input_tokens": 1, "output_tokens": 1}})
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _parse(kind: str, stdout: str):
    capture = executor._Capture(stdout=stdout, stderr="", exit_code=0, duration=1.0)
    return executor._parse_and_validate(
        CodexProvider(), capture, kind=kind, prefer_arrays=kind == "salvage",
    )


class TestTheExtractorNamesTheReplysOneObject:

    def test_the_run_s_reply_serves_the_declined_array_and_names_the_report(self):
        extracted = extract_json_from_text(FINAL_TEXT, prefer_arrays=True)
        # The trap, as measured: an array holding an object outranks the
        # object around it under shape preference.
        assert extracted["data"] == REPORT["declined"]
        assert extracted["only_object"] == REPORT

    def test_a_list_in_the_prose_does_not_make_two_objects(self):
        text = 'I kept ["c2e33cbae17e32bc"] applied.\n\n' + json.dumps(REPORT)
        assert extract_json_from_text(text, prefer_arrays=True)["only_object"] == REPORT

    def test_a_brace_in_the_prose_around_the_object_does_not_hide_it(self):
        text = "Report {see below}:\n" + json.dumps(REPORT) + "\n{end}"
        assert extract_json_from_text(text, prefer_arrays=True)["only_object"] == REPORT

    def test_two_report_objects_name_none(self):
        draft = dict(REPORT, summary="draft")
        text = json.dumps(draft) + "\n\nCorrected:\n" + json.dumps(REPORT)
        assert "only_object" not in extract_json_from_text(text, prefer_arrays=True)

    def test_a_served_object_carries_no_second_reading(self):
        extracted = extract_json_from_text(json.dumps(REPORT), prefer_arrays=False)
        assert extracted["data"] == REPORT and "only_object" not in extracted

    def test_an_array_reply_with_no_object_outside_it_names_none(self):
        text = "Findings:\n" + json.dumps([{"title": "a"}, {"title": "b"}])
        assert "only_object" not in extract_json_from_text(text, prefer_arrays=True)


class TestTheExecutorReadsTheReportObject:

    def test_the_codex_decline_reply_validates_at_once(self):
        stdout = _codex_stream(
            "I'll inspect the current file and the requested diff first.", FINAL_TEXT)
        status, error, data = _parse("apply_code_fixes", stdout)
        assert (status, error) == (executor.STATUS_OK, "")
        assert data == REPORT

    @pytest.mark.parametrize("kind", ["apply_suggestions", "apply_task_suggestions", "implement"])
    def test_every_write_kind_reads_it_so(self, kind):
        report = dict(REPORT, ids=["s1"], declined=[{"id": "s0", "reason": "no"}])
        if kind == "implement":
            report.pop("declined")
            report["check"] = {"passed": True, "findings": [], "ran": [{"command": "npm test"}]}
        text = PROSE + json.dumps(report)
        assert _parse(kind, _codex_stream(text))[0] == executor.STATUS_OK

    def test_two_report_objects_stay_a_re_ask(self):
        draft = dict(REPORT, summary="draft")
        text = json.dumps(draft) + "\n\nCorrected:\n" + json.dumps(REPORT)
        status, error, _data = _parse("apply_code_fixes", _codex_stream(text))
        assert status == executor.STATUS_SCHEMA_INVALID
        assert "expected type object, got list" in error

    @pytest.mark.parametrize("kind", ["validation", "revalidation", "salvage"])
    def test_a_kind_whose_reply_may_be_an_array_keeps_the_array(self, kind):
        whole = {"groups": []}
        listed = [{"group_hash": "a"}]
        assert executor.the_reply_object(kind, listed, {"only_object": whole}) is listed

    def test_the_object_kinds_are_the_ones_whose_schema_root_is_an_object(self):
        assert executor.OBJECT_RESULT_KINDS == {
            "task_generation", "consolidation", "human_decision_judge",
            *executor.WRITE_KINDS,
        }


from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401 - the fixture


def _two_item_task(env):
    return env.apply_task(0, expected_ids=["s0", "s1"])


def _prose_reply() -> str:
    return PROSE + json.dumps({
        "status": "applied", "ids": ["s1"],
        "declined": [{"id": "s0", "reason": "the plan decides it"}],
        "files_touched": ["plans/p.md"], "summary": "applied s1",
    })


class TestBothInvokersApplyTheBatchWithNoReAsk:

    def test_the_executor_leg(self, env):
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)])
        env.monkeypatch.setenv("MOCK_LLM_WRITE_RESULT", _prose_reply())

        result = env.scheduler([_two_item_task(env)]).run()

        outcome = result.outcomes[0]
        assert outcome.ok and outcome.write_outcome == executor.WRITE_APPLIED, outcome.reason
        assert outcome.declined == {"s0": "the plan decides it"}
        assert [a.status for a in outcome.attempts] == [executor.STATUS_OK]
        assert len(log.read_text().splitlines()) == 1
        assert not list((env.phase_dir / "failed").glob("apply-batch-0.attempt*"))
        assert result.exit_code == 0

    def test_the_native_leg(self, env):
        """The native leg reads a text reply objects-first, so the report
        object is served when the prose comes before it; a recap list after
        it ends last and was served instead."""
        def respond(task, prompt):
            env.plan.write_text(env.plan.read_text() + "edit\n")
            return (_prose_reply()
                    + '\n\nDeclined: [{"id": "s0", "reason": "the plan decides it"}]')

        result = env.scheduler([_two_item_task(env)], respond=respond).run()

        outcome = result.outcomes[0]
        assert outcome.ok and outcome.write_outcome == executor.WRITE_APPLIED, outcome.reason
        assert outcome.declined == {"s0": "the plan decides it"}
