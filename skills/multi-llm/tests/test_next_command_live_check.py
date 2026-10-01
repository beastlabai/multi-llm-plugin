"""T10's graders and its two entrypoint helpers, offline (plans/next-phase-command.md).

`TestTheNextCommandLiveCheck` in `test_container_host_lane.py` only ever answers
inside a credentialed Docker run, and every one of its rows is real spend. So
the readers it asserts with — `tests/container/next_command_check.py` — are
debugged here first, over streams in each host's recorded shape:

* codex: `codex exec --json` NDJSON (`item.completed` items; the shapes of
  `tests/fixtures/codex/0.156.1/mcp_tool_call.jsonl`);
* opencode: `opencode run --format json` events (`tool_use` / `text` parts; the
  shapes of `tests/fixtures/opencode/run_review.1.18.30.jsonl`);
* claude-code: the `--output-format json` result envelope;
* receipts: what `wrapper._write_receipt` writes.

The entrypoint's fixture-plan copy and its typed follow-up session are driven
here against a fake host binary, so the polling, the early stop, the scrub and
the record are proven before a container ever runs them.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from .container import entrypoint
from .container import harness
from .container import next_command_check as ncc

SKILL = "/opt/mllm/skill"
WORKSPACE = "/work/sample-project"
FIXTURE = "plans/next phase.md"

TEMPLATES = {
    "codex": {"default": "Use the multi-llm skill with exactly these arguments: {args}",
              "package": "$multi-llm:multi-llm {args}"},
    "opencode": {"default": "/multi-llm {args}", "package": "/multi-llm {args}"},
    "claude-code": {"default": "/multi-llm:multi-llm {args}",
                    "skill": "/multi-llm {args}"},
}


def _step(prefix: str, plan: str = '"plans/next phase.md"') -> dict:
    return {
        "phase": "implement",
        "command": f"{prefix}--implement {plan}",
        "optional_before": [{
            "phase": "review-tasks",
            "command": f"{prefix}--review-tasks {plan}",
            "why": "optional: have the models review the tasks before implementing",
        }],
        "complete": False,
    }


def _receipt(seq: int, words: list[str], result: dict | None = None) -> dict:
    return {
        "seq": seq,
        "argv": ["run", "--frozen", "--no-sync", "--project", SKILL, "--",
                 "python", f"{SKILL}/multi_llm.py", *words],
        "result": result,
        "exit": 0,
    }


def _report(step: dict, *, fence_primary: bool = True, fence_optional: bool = True,
            mangle: str | None = None) -> str:
    primary = mangle or step["command"]
    optional = step["optional_before"][0]
    lines = ["### Summary", "Generated 5 tasks.", "", "### Next Steps",
             "- Start the next phase in a fresh context"]
    lines += (["", "```", primary, "```"] if fence_primary else ["", primary])
    lines += ["", "Optional, before that:", optional["why"]]
    lines += (["", "```bash", optional["command"], "```"] if fence_optional
              else ["", optional["command"]])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Fences
# --------------------------------------------------------------------------- #

class TestFencedLines:
    def test_backtick_and_tilde_fences_with_and_without_a_language(self):
        text = "a\n```\nfirst\n```\nb\n~~~text\nsecond\n~~~\n```bash\nthird\n```\n"
        assert ncc.fenced_lines(text) == ["first", "second", "third"]

    def test_an_indented_fence_in_a_list_item_strips_the_layout_only(self):
        text = "- Next:\n  ```\n  /multi-llm --implement \"plans/x y.md\"\n  ```\n"
        assert ncc.fenced_lines(text) == ['/multi-llm --implement "plans/x y.md"']

    def test_text_outside_any_fence_is_not_a_fenced_line(self):
        assert ncc.fenced_lines("/multi-llm --implement x\n") == []

    def test_an_unclosed_fence_still_yields_its_lines(self):
        assert ncc.fenced_lines("```\nonly\n") == ["only"]

    def test_a_tilde_line_inside_a_backtick_fence_does_not_close_it(self):
        assert ncc.fenced_lines("```\n~~~\nx\n```\n") == ["~~~", "x"]


# --------------------------------------------------------------------------- #
# Final messages
# --------------------------------------------------------------------------- #

def _codex_stream(items: list[dict]) -> str:
    events = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"}]
    for index, item in enumerate(items):
        item = dict(item, id=f"item_{index}")
        events.append({"type": "item.started", "item": item})
        events.append({"type": "item.completed", "item": item})
    events.append({"type": "turn.completed", "usage": {"input_tokens": 1}})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _opencode_stream(parts: list[tuple[str, dict]]) -> str:
    events = []
    for kind, part in parts:
        events.append({"type": kind, "timestamp": 1, "sessionID": "ses",
                       "part": part})
    return "\n".join(json.dumps(e) for e in events) + "\n"


class TestFinalMessage:
    def test_codex_joins_the_agent_messages_after_the_last_tool_item(self):
        stream = _codex_stream([
            {"type": "agent_message", "text": "Reading the skill."},
            {"type": "command_execution", "command": "cat SKILL.md",
             "aggregated_output": "...", "exit_code": 0, "status": "completed"},
            {"type": "reasoning", "text": "I should not appear"},
            {"type": "agent_message", "text": "### Summary"},
            {"type": "agent_message", "text": "```\n/x\n```"},
        ])
        assert ncc.final_message("codex", stream) == "### Summary\n```\n/x\n```"

    def test_codex_with_no_agent_message_after_the_last_tool_has_none(self):
        stream = _codex_stream([
            {"type": "agent_message", "text": "early"},
            {"type": "mcp_tool_call", "server": "multi_llm", "tool": "multi_llm",
             "arguments": {}, "result": None, "status": "completed"},
        ])
        assert ncc.final_message("codex", stream) is None

    def test_opencode_takes_the_text_parts_after_the_last_tool_use(self):
        stream = _opencode_stream([
            ("text", {"type": "text", "text": "narration"}),
            ("tool_use", {"type": "tool", "tool": "bash",
                          "state": {"status": "completed", "input": {"command": "ls"},
                                    "output": "x"}}),
            ("text", {"type": "text", "text": "final report"}),
        ])
        assert ncc.final_message("opencode", stream) == "final report"

    def test_claude_code_reads_the_envelope_result(self):
        envelope = {"type": "result", "subtype": "success", "is_error": False,
                    "result": "the report", "session_id": "s"}
        assert ncc.final_message("claude-code", json.dumps(envelope)) == "the report"
        assert ncc.final_message("claude-code", "not json") is None


# --------------------------------------------------------------------------- #
# Tool calls
# --------------------------------------------------------------------------- #

class TestToolCalls:
    def test_codex_mcp_and_shell_calls_carry_their_modes(self):
        outcome = {"ok": True, "state": "done", "user_next_step": _step("$multi-llm:multi-llm ")}
        stream = _codex_stream([
            {"type": "command_execution",
             "command": "/bin/bash -lc 'uv run --project \"/s\" -- python \"/s/multi_llm.py\" "
                        "util instructions generate-tasks'",
             "aggregated_output": "", "status": "completed"},
            {"type": "mcp_tool_call", "server": "multi_llm", "tool": "multi_llm",
             "arguments": {"mode": "generate-tasks", "plan": "plans/next phase.md"},
             "result": {"content": [{"type": "text", "text": json.dumps(outcome)}],
                        "structured_content": None},
             "status": "completed"},
            {"type": "command_execution", "command": "ls plans",
             "aggregated_output": "next phase.md", "status": "completed"},
        ])
        calls = ncc.tool_calls("codex", stream)
        assert [(c["kind"], c["mode"]) for c in calls] == [
            ("shell", "util instructions"), ("tool", "generate-tasks"), ("shell", None)]
        assert ncc.carries_next_step(calls[1])

    def test_opencode_tool_use_parts_name_the_tool_and_mode(self):
        outcome = {"ok": True, "user_next_step": _step("/multi-llm ")}
        stream = _opencode_stream([
            ("tool_use", {"type": "tool", "tool": "multi_llm", "callID": "c1",
                          "state": {"status": "completed",
                                    "input": {"mode": "generate-tasks",
                                              "plan": "plans/next phase.md"},
                                    "output": json.dumps(outcome)}}),
            ("tool_use", {"type": "tool", "tool": "bash", "callID": "c2",
                          "state": {"status": "completed",
                                    "input": {"command": "uv run -- python /s/multi_llm.py implement x"},
                                    "output": ""}}),
            ("tool_use", {"type": "tool", "tool": "read", "callID": "c3",
                          "state": {"status": "completed", "input": {"filePath": "/x"},
                                    "output": ""}}),
        ])
        calls = ncc.tool_calls("opencode", stream)
        assert [(c["kind"], c["name"], c["mode"]) for c in calls] == [
            ("tool", "multi_llm", "generate-tasks"), ("shell", "bash", "implement"),
            ("other", "read", None)]
        assert ncc.carries_next_step(calls[0])
        assert not ncc.carries_next_step(calls[2])

    def test_a_shell_result_carrying_the_block_is_a_terminal_result(self):
        call = {"kind": "shell", "mode": "status",
                "output": "[NEXT_COMMAND] /multi-llm --implement x\nMULTI_LLM_RESULT: {}"}
        assert ncc.carries_next_step(call)

    def test_claude_code_has_no_tool_calls_in_its_envelope(self):
        assert ncc.tool_calls("claude-code", json.dumps({"result": "x"})) == []


# --------------------------------------------------------------------------- #
# Receipts
# --------------------------------------------------------------------------- #

class TestReceipts:
    def test_mode_and_plan_argument_of_a_spaced_positional(self):
        receipt = _receipt(1, ["implement", "plans/next phase.md", "--host", "codex"])
        assert ncc.receipt_mode(receipt) == "implement"
        assert ncc.receipt_plan_arg(receipt) == "plans/next phase.md"

    def test_the_plan_file_flag_form_is_read_too(self):
        assert ncc.receipt_plan_arg(_receipt(1, ["implement", "--plan-file",
                                                 "/w/plans/a b.md"])) == "/w/plans/a b.md"
        assert ncc.receipt_plan_arg(_receipt(1, ["implement",
                                                 "--plan-file=/w/p.md"])) == "/w/p.md"

    def test_a_util_receipt_names_its_subcommand(self):
        receipt = _receipt(1, ["util", "next-command", "plans/x.md", "--after",
                               "generate-tasks"])
        assert ncc.receipt_mode(receipt) == "util next-command"
        assert ncc.receipt_plan_arg(receipt) == "plans/x.md"

    def test_a_quoted_plan_argument_is_returned_with_its_quotes(self):
        receipt = _receipt(1, ["implement", '"plans/next phase.md"'])
        assert ncc.receipt_plan_arg(receipt) == '"plans/next phase.md"'

    def test_the_last_user_next_step_wins_and_its_position_is_kept(self):
        first = _step("/multi-llm ")
        second = dict(first, command="/multi-llm --implement y")
        receipts = [_receipt(1, ["generate-tasks", "x"], {"user_next_step": first}),
                    _receipt(2, ["status", "x"], {"state": "running"}),
                    _receipt(3, ["status", "x"], {"user_next_step": second}),
                    _receipt(4, ["status", "x"], None)]
        assert ncc.last_user_next_step(receipts) == (second, 2)
        assert ncc.last_user_next_step([]) == (None, None)

    def test_read_receipts_orders_by_seq_and_drops_the_staging_call(self, tmp_path):
        for seq, words in ((2, ["status", "x"]), (1, ["install-host", "codex"]),
                           (3, ["implement", "x"])):
            receipt = _receipt(seq, words)
            if words[0] == "install-host":
                receipt["argv"] = ["install-host", "codex"]
            (tmp_path / f"{seq:03d}.receipt.json").write_text(json.dumps(receipt))
        assert [r["seq"] for r in ncc.read_receipts(tmp_path)] == [2, 3]


# --------------------------------------------------------------------------- #
# Criterion 4: the copy
# --------------------------------------------------------------------------- #

class TestCopyCheck:
    PREFIXES = ["/multi-llm "]

    def test_both_commands_fenced_byte_for_byte_passes(self):
        step = _step("/multi-llm ")
        verdict = ncc.copy_check(_report(step), step, self.PREFIXES)
        assert verdict["passes"] and verdict["command_copied"]
        assert verdict["optional_copied"] == [True]

    def test_an_unfenced_primary_fails_and_says_it_was_present(self):
        step = _step("/multi-llm ")
        verdict = ncc.copy_check(_report(step, fence_primary=False), step, self.PREFIXES)
        assert not verdict["passes"]
        assert verdict["command_anywhere"] and not verdict["command_copied"]

    def test_a_missing_optional_command_fails(self):
        step = _step("/multi-llm ")
        verdict = ncc.copy_check(_report(step, fence_optional=False), step, self.PREFIXES)
        assert not verdict["passes"] and verdict["optional_copied"] == [False]

    def test_a_requoted_command_is_a_near_miss_not_a_copy(self):
        step = _step("/multi-llm ")
        mangled = "/multi-llm --implement 'plans/next phase.md'"
        verdict = ncc.copy_check(_report(step, mangle=mangled), step, self.PREFIXES)
        assert not verdict["passes"]
        assert verdict["near_misses"] == [mangled]

    def test_complete_needs_the_sentence_and_no_fenced_command(self):
        done = {"phase": None, "command": None, "complete": True}
        assert ncc.copy_check("### Next Steps\nWorkflow complete.\n", done,
                              self.PREFIXES)["passes"]
        stray = "Workflow complete.\n```\n/multi-llm --review-code x\n```\n"
        verdict = ncc.copy_check(stray, done, self.PREFIXES)
        assert not verdict["passes"] and verdict["stray_commands"]

    def test_nothing_printed_is_unobservable_never_a_pass(self):
        verdict = ncc.copy_check("anything", None, self.PREFIXES)
        assert verdict["observable"] is False and "passes" not in verdict


# --------------------------------------------------------------------------- #
# Criterion 5: the no-call
# --------------------------------------------------------------------------- #

def _call(index: int, kind: str, mode: str | None, carries: bool = False) -> dict:
    output = json.dumps({"ok": True, "user_next_step": _step("/multi-llm ")}) if carries \
        else json.dumps({"ok": True, "state": "running"})
    return {"index": index, "kind": kind, "name": "multi_llm" if kind == "tool" else "bash",
            "mode": mode, "arguments": {"mode": mode}, "output": output}


class TestNoCallCheck:
    STEP = _step("/multi-llm ")

    def test_one_terminal_call_and_nothing_after_holds(self):
        calls = [_call(0, "tool", "generate-tasks", carries=True)]
        verdict = ncc.no_call_check(calls, self.STEP, [], None)
        assert verdict["observable"] and verdict["holds"]

    def test_a_status_poll_before_the_terminal_result_is_fine(self):
        calls = [_call(0, "tool", "generate-tasks"), _call(1, "tool", "status", True),
                 {"index": 2, "kind": "other", "name": "read", "mode": None,
                  "arguments": {}, "output": ""}]
        assert ncc.no_call_check(calls, self.STEP, [], None)["holds"]

    def test_any_dispatcher_call_after_the_terminal_result_breaks_it(self):
        calls = [_call(0, "tool", "generate-tasks", True), _call(1, "tool", "status")]
        verdict = ncc.no_call_check(calls, self.STEP, [], None)
        assert not verdict["holds"]
        assert verdict["calls_after_terminal"] == [{"index": 1, "mode": "status"}]

    def test_running_the_next_phase_anywhere_breaks_it(self):
        calls = [_call(0, "shell", "implement"), _call(1, "tool", "generate-tasks", True)]
        verdict = ncc.no_call_check(calls, self.STEP, [], None)
        assert not verdict["holds"]
        assert verdict["calls_naming_next_phase"] == [{"index": 0, "kind": "shell"}]

    def test_a_receipt_for_the_next_phase_breaks_it_whoever_made_it(self):
        calls = [_call(0, "tool", "generate-tasks", True)]
        receipts = [_receipt(1, ["generate-tasks", "x"]), _receipt(2, ["implement", "x"])]
        verdict = ncc.no_call_check(calls, self.STEP, receipts, 0)
        assert not verdict["holds"]
        assert verdict["receipts_naming_next_phase"] == ["implement"]

    def test_no_terminal_result_is_not_observable(self):
        verdict = ncc.no_call_check([_call(0, "tool", "generate-tasks")], self.STEP, [], None)
        assert verdict["observable"] is False and verdict["holds"] is False


# --------------------------------------------------------------------------- #
# The typed template, the lookup, the route
# --------------------------------------------------------------------------- #

class TestTypedCheck:
    RUN = {"exercised": True, "phase": "implement", "typed": "x", "surface": "s",
           "stop_reason": "the named mode dispatched; session stopped"}

    def test_the_named_mode_with_the_unquoted_fixture_passes(self):
        receipts = [_receipt(1, ["util", "instructions", "implement"]),
                    _receipt(2, ["implement", FIXTURE, "--host", "codex"])]
        verdict = ncc.typed_check(self.RUN, receipts, FIXTURE, WORKSPACE)
        assert verdict["passes"] and verdict["plan_args"] == [FIXTURE]

    def test_an_absolute_fixture_path_names_it_too(self):
        receipts = [_receipt(1, ["implement", f"{WORKSPACE}/{FIXTURE}"])]
        assert ncc.typed_check(self.RUN, receipts, FIXTURE, WORKSPACE)["passes"]

    def test_a_quote_surviving_into_argv_fails(self):
        receipts = [_receipt(1, ["implement", f'"{FIXTURE}"'])]
        verdict = ncc.typed_check(self.RUN, receipts, FIXTURE, WORKSPACE)
        assert not verdict["passes"] and verdict["quote_free"] is False

    def test_a_split_path_names_a_different_plan_and_fails(self):
        receipts = [_receipt(1, ["implement", "plans/next", "phase.md"])]
        verdict = ncc.typed_check(self.RUN, receipts, FIXTURE, WORKSPACE)
        assert not verdict["passes"] and verdict["names_fixture"] is False

    def test_another_mode_is_not_a_dispatch(self):
        receipts = [_receipt(1, ["review-tasks", FIXTURE])]
        verdict = ncc.typed_check(self.RUN, receipts, FIXTURE, WORKSPACE)
        assert verdict["dispatched"] is False and not verdict["passes"]

    def test_an_unexercised_session_never_passes(self):
        verdict = ncc.typed_check({"exercised": False, "reason": "nothing printed"},
                                  [], FIXTURE, WORKSPACE)
        assert verdict == {"exercised": False, "passes": False,
                           "reason": "nothing printed"}


class TestLookupAndRoute:
    def test_a_util_next_command_after_the_last_phase_step_is_the_lookup(self):
        receipts = [_receipt(1, ["generate-tasks", "x"], {"next": "reagg"}),
                    _receipt(2, ["generate-tasks", "x", "--reaggregate"], {"state": "done"}),
                    _receipt(3, ["util", "next-command", "x", "--after", "generate-tasks"],
                             {"user_next_step": _step("/multi-llm ")})]
        assert ncc.lookup_check(receipts, "generate-tasks")["lookup"] == "util-next-command"

    def test_a_phase_result_carrying_the_key_counts(self):
        receipts = [_receipt(1, ["generate-tasks", "x", "--reaggregate"],
                             {"user_next_step": _step("/multi-llm ")})]
        assert ncc.lookup_check(receipts, "generate-tasks")["lookup"] == "phase-result"

    def test_a_lookup_before_the_last_phase_step_does_not_count(self):
        receipts = [_receipt(1, ["util", "next-command", "x"], {}),
                    _receipt(2, ["generate-tasks", "x"], {"state": "done"})]
        assert ncc.lookup_check(receipts, "generate-tasks")["lookup"] == "none"

    def test_the_hook_is_not_installed_off_the_plugin_route(self):
        assert ncc.stop_hook_state("report-check decision block", False) == "not-installed"
        assert ncc.stop_hook_state("", True) == "silent"

    def test_the_longest_matching_prefix_names_the_route(self):
        claude = TEMPLATES["claude-code"]
        assert ncc.route_of("/multi-llm:multi-llm --implement x", claude) == "default"
        assert ncc.route_of("/multi-llm --implement x", claude) == "skill"
        assert ncc.route_of("$multi-llm:multi-llm --implement x",
                            TEMPLATES["codex"]) == "package"
        assert ncc.route_of(None, claude) is None


# --------------------------------------------------------------------------- #
# One synthetic bundle, end to end, and the record it produces
# --------------------------------------------------------------------------- #

def _bundle(tmp_path: Path, host: str, row_id: str, step: dict, stdout: str,
            typed_receipts: list[dict], typed_run: dict) -> Path:
    receipts = tmp_path / "receipts" / row_id
    receipts.mkdir(parents=True)
    for receipt in [_receipt(1, ["util", "instructions", "generate-tasks"]),
                    _receipt(2, ["generate-tasks", FIXTURE, "--detach"], {"state": "done"}),
                    _receipt(3, ["status", FIXTURE], {"user_next_step": step})]:
        (receipts / f"{receipt['seq']:03d}.receipt.json").write_text(json.dumps(receipt))
    typed = tmp_path / "receipts" / f"{row_id}-typed"
    typed.mkdir(parents=True)
    for receipt in typed_receipts:
        (typed / f"{receipt['seq']:03d}.receipt.json").write_text(json.dumps(receipt))
    (tmp_path / "host-stdout.txt").write_text(stdout)
    (tmp_path / "typed-run.json").write_text(json.dumps(typed_run))
    return tmp_path


MATRIX_ROW = {"id": "x", "fixture_plan": {"source": "plans/auth-feature.md", "path": FIXTURE},
              "next_command_check": {"phase": "generate-tasks", "typed_followup": True}}


class TestObserveOneBundle:
    def test_an_opencode_package_bundle_passes_all_three_checks(self, tmp_path):
        step = _step("/multi-llm ", plan=f'"{WORKSPACE}/{FIXTURE}"')
        outcome = {"ok": True, "state": "done", "user_next_step": step}
        stdout = _opencode_stream([
            ("tool_use", {"type": "tool", "tool": "multi_llm",
                          "state": {"status": "completed",
                                    "input": {"mode": "generate-tasks", "plan": FIXTURE},
                                    "output": json.dumps(outcome)}}),
            ("text", {"type": "text", "text": _report(step)}),
        ])
        bundle = _bundle(tmp_path, "opencode", "x", step, stdout,
                         [_receipt(1, ["implement", f"{WORKSPACE}/{FIXTURE}"])],
                         {"exercised": True, "phase": "implement", "typed": "t",
                          "surface": "s", "stop_reason": "stopped"})
        (bundle / "credential-census.json").write_text(json.dumps(
            {"model": "deepseek/deepseek-v4.1-flash", "row": "x"}))
        observed = ncc.observe(bundle, "opencode", "x", MATRIX_ROW,
                               TEMPLATES["opencode"], route="package")
        assert observed["model"] == "deepseek/deepseek-v4.1-flash"
        assert observed["route_observed"] == "default|package"
        assert observed["copy"]["passes"]
        assert observed["no_call"]["holds"]
        assert observed["typed"]["passes"]
        block = ncc.record_block(observed)
        assert block["copied"] is True and block["no_call"] is True
        assert block["model"] == "deepseek/deepseek-v4.1-flash"
        assert block["typed"]["passes"] is True

    def test_a_claude_code_bundle_records_the_lookup_and_the_hook(self, tmp_path):
        step = _step("/multi-llm ")
        stdout = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                             "result": _report(step, fence_optional=False)})
        bundle = _bundle(tmp_path, "claude-code", "x", step, stdout, [],
                         {"exercised": False, "reason": "nothing to type"})
        observed = ncc.observe(bundle, "claude-code", "x", MATRIX_ROW,
                               TEMPLATES["claude-code"], route="file")
        assert observed["route_observed"] == "skill"
        assert observed["copy"]["passes"] is False
        assert observed["lookup"]["lookup"] == "none"
        assert observed["stop_hook"] == "not-installed"
        assert "no_call" not in observed
        block = ncc.record_block(observed)
        assert block["copied"] is False and block["lookup"] == "none"
        assert block["typed"] == {"exercised": False, "surface": None,
                                  "dispatched": None, "plan_args": [],
                                  "quote_free": None, "names_fixture": None,
                                  "passes": False, "stop_reason": "nothing to type"}

    def test_the_record_block_validates_against_the_record_schema(self, tmp_path):
        import jsonschema

        schema = harness.load_record_schema()
        row_schema = schema["properties"]["rows"]["items"]["properties"]["next_command"]
        step = _step("Use the multi-llm skill with exactly these arguments: ")
        stdout = _codex_stream([{"type": "agent_message", "text": _report(step)}])
        bundle = _bundle(tmp_path, "codex", "x", step, stdout,
                         [_receipt(1, ["implement", FIXTURE])],
                         {"exercised": True, "phase": "implement", "typed": "t",
                          "surface": "s", "stop_reason": "stopped"})
        block = ncc.record_block(ncc.observe(bundle, "codex", "x", MATRIX_ROW,
                                             TEMPLATES["codex"]))
        jsonschema.validate(block, row_schema)
        assert block["route_observed"] == "default"


# --------------------------------------------------------------------------- #
# The entrypoint's two helpers, against a fake host
# --------------------------------------------------------------------------- #

FAKE_HOST = """#!{python}
import json, os, sys, time
receipts = os.environ["MLLM_RECEIPT_DIR"]
os.makedirs(receipts, exist_ok=True)
prompt = sys.argv[-1]
print("echo:" + prompt + " key=" + os.environ.get("FAKE_SECRET", ""), flush=True)
def receipt(seq, words):
    with open(os.path.join(receipts, "%03d.receipt.json" % seq), "w") as h:
        json.dump({{"seq": seq, "argv": ["run", "--", "python", "/s/multi_llm.py", *words]}}, h)
receipt(1, ["util", "instructions", "implement"])
time.sleep(0.2)
if os.environ.get("FAKE_DISPATCH") == "1":
    receipt(2, ["implement", "plans/next phase.md", "--host", "x"])
time.sleep(float(os.environ.get("FAKE_HANG", "60")))
"""


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    out = tmp_path / "out"
    workspace = tmp_path / "work"
    real = tmp_path / "real"
    for directory in (out, workspace / "plans", real):
        directory.mkdir(parents=True)
    host = real / "codex"
    host.write_text(FAKE_HOST.format(python=sys.executable))
    host.chmod(host.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(entrypoint, "OUT", out)
    monkeypatch.setattr(entrypoint, "WORKSPACE", workspace)
    monkeypatch.setattr(entrypoint, "REAL_DIR", real)
    monkeypatch.setattr(entrypoint, "log", lambda message: None)
    return out, workspace, tmp_path / "home"


def _seed_session_receipt(out: Path, row_id: str, step: dict | None) -> None:
    directory = out / "receipts" / row_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "001.receipt.json").write_text(json.dumps(
        _receipt(1, ["status", FIXTURE], {"user_next_step": step} if step else None)))


CODEX_ENTRY = {"invoke_argv": ["codex", "exec", "--json", "--model", "<model>", "<prompt>"],
               "model": {"slug": "m"},
               "prompt_template": "Use the multi-llm skill with exactly these arguments: {args}\nFollow."}
ROW = {"id": "codex-next-command", "args": "--generate-tasks x",
       "next_command_check": {"phase": "generate-tasks", "typed_followup": True,
                              "typed_wall_clock_s": 20, "typed_grace_s": 0.5}}


class TestTheTypedFollowup:
    def test_it_types_the_printed_command_verbatim_and_stops_at_the_dispatch(
            self, rig, monkeypatch):
        out, _workspace, home = rig
        home.mkdir()
        (home / ".profile").write_text("export MLLM_RECEIPT_DIR=/old\n")
        step = _step("Use the multi-llm skill with exactly these arguments: ")
        _seed_session_receipt(out, ROW["id"], step)
        monkeypatch.setenv("FAKE_DISPATCH", "1")
        monkeypatch.setenv("FAKE_SECRET", "sekrit-value")
        record = entrypoint.run_typed_followup(
            "codex", CODEX_ENTRY, ROW, home, {}, "sekrit-value")
        assert record["exercised"] and record["typed"] == step["command"]
        assert record["argv"][-1] == step["command"]
        assert record["stop_reason"] == "the named mode dispatched; session stopped"
        assert record["wall_clock_s"] < 15
        stdout = (out / "host-typed-stdout.txt").read_text()
        assert "sekrit-value" not in stdout and "<redacted:openrouter-credential>" in stdout
        typed = ncc.read_receipts(out / "receipts" / f"{ROW['id']}-typed")
        assert [ncc.receipt_mode(r) for r in typed] == ["util instructions", "implement"]
        assert (home / ".profile").read_text().rstrip().endswith(
            f"export MLLM_RECEIPT_DIR={out}/receipts/{ROW['id']}-typed")
        assert json.loads((out / "typed-run.json").read_text())["phase"] == "implement"

    def test_a_session_that_never_dispatches_is_stopped_at_its_bound(
            self, rig, monkeypatch):
        out, _workspace, home = rig
        home.mkdir()
        _seed_session_receipt(out, ROW["id"], _step("/x "))
        row = dict(ROW, next_command_check=dict(ROW["next_command_check"],
                                                typed_wall_clock_s=2))
        record = entrypoint.run_typed_followup("codex", CODEX_ENTRY, row, home, {}, "")
        assert record["stop_reason"] == "wall-clock" and record["timed_out"]

    def test_nothing_printed_means_nothing_typed(self, rig):
        out, _workspace, home = rig
        _seed_session_receipt(out, ROW["id"], None)
        record = entrypoint.run_typed_followup("codex", CODEX_ENTRY, ROW, home, {}, "")
        assert record == {"host": "codex", "row": ROW["id"], "exercised": False,
                          "reason": record["reason"]}
        assert not (out / "host-typed-stdout.txt").exists()

    def test_opencode_types_the_arguments_after_its_command_name(self):
        typed, surface = entrypoint._typed_prompt(
            "opencode", {}, '/multi-llm --implement "plans/next phase.md"')
        assert typed == '--implement "plans/next phase.md"'
        assert "--command multi-llm" in surface
        with pytest.raises(ValueError):
            entrypoint._typed_prompt("opencode", {}, "$multi-llm:multi-llm --implement x")
        assert entrypoint._typed_prompt("claude-code", {}, "/multi-llm --x")[0] == "/multi-llm --x"


class TestTheFixturePlan:
    def test_it_is_a_byte_copy_at_the_spaced_path(self, rig):
        _out, workspace, _home = rig
        (workspace / "plans" / "auth-feature.md").write_bytes(b"# plan\n")
        target = entrypoint.install_fixture_plan(
            {"source": "plans/auth-feature.md", "path": FIXTURE})
        assert target == workspace / FIXTURE
        assert target.read_bytes() == b"# plan\n"

    def test_its_output_dir_is_the_products_sanitized_one(self, rig):
        """Measured on the first T10 run: the product writes `plans/next_phase/`,
        and a poll/export keyed on the bare stem found no run and exported
        nothing."""
        _out, workspace, _home = rig
        assert entrypoint.plan_output_dir(workspace / FIXTURE) == (
            workspace / "plans" / "next_phase")
        assert entrypoint.plan_output_dir(workspace / "plans" / "auth-feature.md") == (
            workspace / "plans" / "auth-feature")

    def test_a_missing_source_or_an_empty_path_is_refused(self, rig):
        with pytest.raises(SystemExit):
            entrypoint.install_fixture_plan({"source": "plans/none.md", "path": FIXTURE})
        with pytest.raises(SystemExit):
            entrypoint.install_fixture_plan({"source": "plans/none.md", "path": ""})


class TestTheMatrixRows:
    """Every T10 row carries what its graders read, on the host it claims."""

    ROWS = [(host, row) for host, entry in harness.load_host_matrix().items()
            for row in entry.get("rows") or [] if row.get("next_command_check")]

    def test_one_row_per_route_the_lane_can_run(self):
        assert sorted((host, row.get("route", "file")) for host, row in self.ROWS) == [
            ("claude-code", "file"), ("codex", "file"), ("codex", "package"),
            ("opencode", "file"), ("opencode", "package")]

    @pytest.mark.parametrize("host,row", ROWS, ids=[row["id"] for _h, row in ROWS])
    def test_each_row_runs_generate_tasks_on_a_spaced_fixture(self, host, row):
        path = row["fixture_plan"]["path"]
        assert " " in path and f'"{path}"' in row["args"]
        assert row["args"].startswith("--generate-tasks ")
        assert row["next_command_check"]["phase"] == "generate-tasks"
        assert row["next_command_check"]["typed_followup"] is True
        assert row["expect"]["phases"] == ["generate-tasks"]
