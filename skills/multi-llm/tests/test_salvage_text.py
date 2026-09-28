"""``salvage_text`` for codex, opencode, copilot and muse, graded fixture by fixture.

Plan: review-run-scope-salvage-and-failure-visibility §4, task T4.3.

**Why.** The salvage gate (``has_salvageable_json``) used to judge a failed
attempt's raw stdout. For these four adapters stdout is an event stream, and
the arrays inside its transport records — copilot's echoed prompt and
bookkeeping records, a codex ``web_search`` item, an opencode ``tool_use``
part, muse's lifecycle records — pass the gate as if they were findings. So an
auth failure, a rate limit or a reviewer killed mid-tool-call was sent to a
JSON repair. Each adapter now reduces its stream to the assistant's answer text
(:meth:`LLMProvider.salvage_text`), and the gate judges that instead.

**What is graded.** One sweep per adapter, over every stdout capture it has.
Each capture carries a label saying whether its bytes hold answer text a
salvage could recover — judged from the content, **never** from whether the
run succeeded (``tests/test_provider_fixture_rows.py`` owns the vocabulary):

- ``none``: nothing to recover. The reduced text must be *rejected* by
  ``has_salvageable_json``. This covers transport-only streams and a closing
  message that is prose with no JSON document in it (the capture rig's
  ``FINAL ANSWER``, a ``Wrote x to m.txt``): salvage repairs a JSON answer and
  has nothing to repair there.
- ``partial``: a cut answer. The reduced text must *pass* the gate.
- ``answer``: a whole answer. The reduced text passes the gate and
  ``extract_json_from_text`` recovers it — the row's pinned ``data`` where the
  manifest pins one, else the document ``parse_output`` itself reads off the
  whole stream (the hook and the parser share one walk, so the text salvage
  reads is the text the parser treats as the answer).

"The reduced text" is the hook's string, or — when the hook returns ``None``
(not this adapter's stream, so no reducer applies) — the raw stdout, which is
what the gate falls back to.

**Where the labels live.** copilot and muse have manifests, so each row's label
is its ``expect.salvage``; a row with files and no label fails. codex and
opencode have no manifest (both sit in ``MANIFEST_PENDING``), so their labels
are the two tables below, keyed by path, and a coverage test fails on any
stdout capture a table does not name — a new capture is labelled when it is
committed, not when it bites.

**The baseline.** Plan §4.1 measured the gate over the raw captures: copilot 11
of 12, codex 14 of 18, opencode 187 of 312 passed. The ``none`` rows among them
are what this sweep turns red on the old input and green on the reduced one;
:class:`TestTheBaselineMoved` names the incidents.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_provider_fixture_rows import MANIFESTS, SALVAGE_LABELS  # noqa: E402
from utils.json_extractor import extract_json_from_text, has_salvageable_json  # noqa: E402
from utils.providers.codex import CodexProvider  # noqa: E402
from utils.providers.copilot import CopilotProvider  # noqa: E402
from utils.providers.muse import MuseProvider  # noqa: E402
from utils.providers.opencode import OpenCodeProvider  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

ADAPTERS = {
    "codex": CodexProvider,
    "copilot": CopilotProvider,
    "muse": MuseProvider,
    "opencode": OpenCodeProvider,
}

#: Sentinel: the row pins no ``data`` of its own.
_UNPINNED = object()


def _read(relative: str) -> str:
    return (FIXTURES / relative).read_bytes().decode("utf-8")


def _reduced(provider: str, stdout: str) -> str:
    """The gate's input after the hook: its text, or the raw stream on ``None``."""
    text = ADAPTERS[provider]().salvage_text(stdout)
    assert text is None or isinstance(text, str)
    return stdout if text is None else text


def _assert_label(
    provider: str, stdout: str, label: str, pinned: Any = _UNPINNED
) -> None:
    """The rule for one label (see the module docstring)."""
    assert label in SALVAGE_LABELS, f"unknown salvage label {label!r}"
    reduced = _reduced(provider, stdout)
    if label == "none":
        assert not has_salvageable_json(reduced), (
            f"labelled none, but the reduced text passes the gate: {reduced[:300]!r}"
        )
        return
    assert has_salvageable_json(reduced), (
        f"labelled {label}, but the gate rejects the reduced text: {reduced[:300]!r}"
    )
    if label == "partial":
        return
    recovered = extract_json_from_text(reduced, prefer_arrays=True)
    assert recovered.get("success"), (
        f"labelled answer, but nothing was recovered: {recovered.get('error')}"
    )
    if pinned is not _UNPINNED:
        assert recovered["data"] == pinned
        return
    # No pinned document: the adapter's own parser is the reference. On a
    # stream it answers from, salvage must recover the same document.
    parsed = ADAPTERS[provider]().parse_output(stdout, "")
    if parsed.get("success"):
        assert recovered["data"] == parsed["data"]


# --------------------------------------------------------------------------- #
# copilot and muse: the label is the manifest row's `expect.salvage`.
# --------------------------------------------------------------------------- #

MANIFEST_ROWS = [
    (provider, row)
    for provider in ("copilot", "muse")
    for row, body in (MANIFESTS[provider].get("rows") or {}).items()
    if "not_applicable" not in body
]


@pytest.mark.parametrize(
    "provider,row", MANIFEST_ROWS, ids=[f"{p}/{r}" for p, r in MANIFEST_ROWS]
)
def test_manifest_row_meets_its_salvage_label(provider: str, row: str) -> None:
    body = MANIFESTS[provider]["rows"][row]
    expect = body.get("expect") or {}
    assert "salvage" in expect, (
        f"{provider}/{row} has captured bytes and no expect.salvage label; "
        f"label it {list(SALVAGE_LABELS)} from what the stdout holds"
    )
    stdout = _read(f"{provider}/{body['stdout']}") if body.get("stdout") else ""
    pinned = expect["data"] if "data" in expect else _UNPINNED
    _assert_label(provider, stdout, expect["salvage"], pinned)


def test_both_manifests_have_rows_to_sweep() -> None:
    """A manifest that lost its rows would make the sweep above vacuous."""
    assert {p for p, _ in MANIFEST_ROWS} == {"copilot", "muse"}


# --------------------------------------------------------------------------- #
# codex: which files are stdout, and their labels.
#
# A codex stdout capture is a `.jsonl` (or `.stdout.txt`) under
# `tests/fixtures/codex/`, except:
#   * `*.rollout.jsonl` — codex's own session file under `~/.codex/sessions`,
#     a different schema (`response_item` payloads) that never reaches stdout;
#   * the rig directories (`capture/` scripts, `schemas/`, `plugin-route/`);
#   * `web_search_hosted.sse.json` needs no rule: it is the Responses SSE
#     `capture/proxy.py` recorded on the provider's side of the wire, and the
#     suffix already leaves it out.
# --------------------------------------------------------------------------- #

_CODEX_RIG_DIRS = ("capture", "schemas", "plugin-route")

CODEX_LABELS: Dict[str, str] = {
    # The stub-driven 0.156.1 captures answer `[]`: an empty review is an answer.
    "0.156.1/collab_spawn_agent.jsonl": "answer",
    "0.156.1/exec_session.jsonl": "answer",
    "0.156.1/mcp_tool_call.jsonl": "answer",
    "0.156.1/quoting_render.jsonl": "answer",
    "0.156.1/read_only_clean.jsonl": "answer",
    "0.156.1/web_search_hosted_replay.jsonl": "answer",
    "0.156.1/web_search_probe_replay.jsonl": "answer",
    "0.156.1/writes_patches_sessions.jsonl": "answer",
    # Live captures whose closing message is prose with no JSON in it.
    "0.156.1/code_mode_mcp.jsonl": "none",
    "0.156.1/code_mode_writes.jsonl": "none",
    "0.156.1/web_search_code_mode.jsonl": "none",
    "0.156.1/web_search_hosted.jsonl": "none",
    # Items only: the run never wrote an agent_message. It passed the gate
    # today on a `web_search` item's arrays.
    "0.156.1/run_0c477c4d_partial.jsonl": "none",
    # Real answers — and the raw streams FAILED the gate: the answer array is
    # a JSON-escaped string inside `item.text`, so salvage could not see it.
    "agentic_reasoning_multi_message.jsonl": "answer",
    "gpt5_sol_single_message.jsonl": "answer",
}


def _codex_stdout_captures() -> Set[str]:
    root = FIXTURES / "codex"
    found: Set[str] = set()
    for path in list(root.rglob("*.jsonl")) + list(root.rglob("*.stdout.txt")):
        relative = path.relative_to(root)
        if relative.parts[0] in _CODEX_RIG_DIRS:
            continue
        if path.name.endswith(".rollout.jsonl"):
            continue
        found.add(relative.as_posix())
    return found


# --------------------------------------------------------------------------- #
# opencode: which files are stdout, and their labels.
#
# An opencode stdout capture is a `.jsonl` or `.stdout.txt` under
# `tests/fixtures/opencode/`, except `*.requests.jsonl`: those are the capture
# stub's record of every HTTP request opencode SENT to the provider (our prompt,
# the tool schemas, the tool results going back), not anything `opencode run`
# printed. They were half of §4.1's 312 and are not a salvage input. The
# sibling `.config.json`, `.debug_config.json`, `.stderr.txt`, `.exit.txt`,
# `auth_list.*.json` and `summary.tsv` are inputs, other commands' output, or
# rig bookkeeping, and the suffixes already leave them out.
#
# Every capture is `none`. The `capture/<version>/` permission probes are driven
# by `capture/stub.py`, whose scripted reply is the prose `FINAL ANSWER` — or no
# text at all when the turn ended on a denied tool call.
# --------------------------------------------------------------------------- #

#: The T4.0 subset every one of 1.18.18, 1.18.23 and 1.18.30 ran.
_T40_SUBSET = (
    "q2_ustar_agentlayer_build", "q2_ustar_agentlayer_plan", "t40_block_build",
    "t40_block_plan", "t40_blockcont_build", "t40_blockcont_plan", "t40_cont_build",
    "t40_cont_plan", "t40_none_build", "t40_none_plan",
)
#: Every case 1.17.15 (`min_version`) ran; 1.18.32 ran these too.
_MIN_VERSION_CASES = (
    "builtin_plansdir_build", "builtin_plansdir_ctrl_build",
    "builtin_plansdir_ctrl_plan", "builtin_plansdir_plan", "builtin_skilldir_build",
    "builtin_skilldir_ctrl_build", "builtin_skilldir_ctrl_plan",
    "builtin_skilldir_plan", "builtin_tmp_build", "builtin_tmp_ctrl_build",
    "builtin_tmp_ctrl_plan", "builtin_tmp_plan", "builtin_toolout_build",
    "builtin_toolout_plan", "envperm_plan", "managed_agent_plan",
    "managed_toplevel_plan", "org_agent_dbmem_plan", "org_agent_plan",
    "q2_ustar_agentlayer_build", "q2_ustar_agentlayer_plan",
    "q2_ustar_toplevel_build", "q2_ustar_toplevel_plan", "s41_ask_bash",
    "s41_ask_read", "s41_cont_read", "s41_deny_bash", "s41_deny_read",
    "s41_denyallow_file_read", "s41_denyallow_read", "t40_badexp_plan",
    "t40_badperm_plan", "t40_badpermname_plan", "t40_block_build", "t40_block_plan",
    "t40_blockcont_build", "t40_blockcont_plan", "t40_cont_build", "t40_cont_plan",
    "t40_env_block_build", "t40_env_block_plan", "t40_env_none_build",
    "t40_env_none_plan", "t40_none_build", "t40_none_plan", "t47_private_build",
    "t47_private_plan", "t47_shared_build", "t47_shared_plan",
    "t47_shared_toplevel_build", "t47_shared_toplevel_plan", "useragent_dir_build",
    "useragent_dir_plan", "useragent_star_build", "useragent_star_plan",
    "write_outofcwd_build",
)
#: The cases only 1.18.32 ran.
_CURRENT_ONLY_CASES = (
    "t47_inherited_star_build", "t47_inherited_star_ctrl_build",
    "useragent_relist_build", "useragent_relist_ctrl_build",
    "useragent_relist_ctrl_plan", "useragent_relist_plan",
    "useragent_relist_private_build", "useragent_relist_private_plan",
    "useragent_ustar_build", "useragent_ustar_ctrl_build",
    "useragent_ustar_ctrl_plan", "useragent_ustar_plan",
)

OPENCODE_LABELS: Dict[str, str] = {
    # One `error` event each (UnknownError / ProviderAuthError): no text part.
    "invalid-model.stdout.txt": "none",
    "provider_auth_error.jsonl": "none",
    # A finished review whose closing text is "JSON written to …": prose.
    "run_review.1.18.30.jsonl": "none",
    # Exit 0 on an auto-rejected permission. Its text parts are only narration
    # ("I'll read the exact plan rows …"), and the raw stream passed the gate
    # on its tool parts.
    "1.18.32/denied_permission_turn_end.jsonl": "none",
    "1.18.32/continue_loop_on_deny.jsonl": "none",
    **{f"capture/1.17.15/{case}.jsonl": "none" for case in _MIN_VERSION_CASES},
    **{
        f"capture/{version}/{case}.jsonl": "none"
        for version in ("1.18.18", "1.18.23", "1.18.30")
        for case in _T40_SUBSET
    },
    **{
        f"capture/1.18.32/{case}.jsonl": "none"
        for case in _MIN_VERSION_CASES + _CURRENT_ONLY_CASES
    },
}


def _opencode_stdout_captures() -> Set[str]:
    root = FIXTURES / "opencode"
    found: Set[str] = set()
    for path in list(root.rglob("*.jsonl")) + list(root.rglob("*.stdout.txt")):
        if path.name.endswith(".requests.jsonl"):
            continue
        found.add(path.relative_to(root).as_posix())
    return found


TABLES: Dict[str, Tuple[Dict[str, str], Any]] = {
    "codex": (CODEX_LABELS, _codex_stdout_captures),
    "opencode": (OPENCODE_LABELS, _opencode_stdout_captures),
}

TABLE_ROWS = [
    (provider, path)
    for provider, (table, _discover) in TABLES.items()
    for path in sorted(table)
]


@pytest.mark.parametrize("provider", sorted(TABLES))
def test_the_table_names_every_stdout_capture(provider: str) -> None:
    """A capture committed without a label fails here, by name."""
    table, discover = TABLES[provider]
    on_disk = discover()
    assert sorted(on_disk - set(table)) == [], "unlabelled stdout captures"
    assert sorted(set(table) - on_disk) == [], "labels for captures that are gone"


@pytest.mark.parametrize("provider", sorted(TABLES))
def test_every_table_label_is_in_the_vocabulary(provider: str) -> None:
    table, _discover = TABLES[provider]
    assert set(table.values()) <= set(SALVAGE_LABELS)


@pytest.mark.parametrize(
    "provider,path", TABLE_ROWS, ids=[f"{p}/{f}" for p, f in TABLE_ROWS]
)
def test_table_capture_meets_its_salvage_label(provider: str, path: str) -> None:
    table, _discover = TABLES[provider]
    _assert_label(provider, _read(f"{provider}/{path}"), table[path])


# --------------------------------------------------------------------------- #
# The baseline: what moved.
# --------------------------------------------------------------------------- #

#: Captures whose raw stdout passed the gate and hold nothing to recover — the
#: §4.1 shape. Each must pass on the raw bytes (so the pin is live) and be
#: rejected once reduced.
TRANSPORT_THAT_PASSED = (
    ("copilot", "copilot/auth-error.jsonl"),
    ("copilot", "copilot/rate-limit.jsonl"),
    ("copilot", "copilot/truncated.jsonl"),
    ("copilot", "copilot/truncated-mid-echo.jsonl"),
    ("muse", "muse/auth-error.stdout.txt"),
    ("muse", "muse/truncated.stdout.txt"),
    ("codex", "codex/0.156.1/run_0c477c4d_partial.jsonl"),
    ("codex", "codex/0.156.1/code_mode_mcp.jsonl"),
    ("opencode", "opencode/1.18.32/denied_permission_turn_end.jsonl"),
    ("opencode", "opencode/run_review.1.18.30.jsonl"),
)

#: Real answers the raw gate turned DOWN (the array hides in a JSON string).
ANSWERS_THAT_FAILED = (
    ("codex", "codex/gpt5_sol_single_message.jsonl"),
    ("codex", "codex/agentic_reasoning_multi_message.jsonl"),
)


class TestTheBaselineMoved:
    @pytest.mark.parametrize("provider,path", TRANSPORT_THAT_PASSED)
    def test_transport_no_longer_reaches_salvage(self, provider: str, path: str) -> None:
        stdout = _read(path)
        assert has_salvageable_json(stdout), "the raw bytes no longer pass: re-pin"
        assert not has_salvageable_json(_reduced(provider, stdout))

    @pytest.mark.parametrize("provider,path", ANSWERS_THAT_FAILED)
    def test_a_hidden_answer_now_reaches_salvage(self, provider: str, path: str) -> None:
        stdout = _read(path)
        assert not has_salvageable_json(stdout), "the raw bytes now pass: re-pin"
        assert has_salvageable_json(_reduced(provider, stdout))

    @pytest.mark.parametrize("provider", sorted(ADAPTERS))
    def test_every_adapter_has_a_none_row_the_raw_gate_passed(self, provider: str) -> None:
        """The sweep is not vacuous: each adapter has a row that moved."""
        if provider in TABLES:
            table, _discover = TABLES[provider]
            rows = [
                _read(f"{provider}/{path}")
                for path, label in table.items() if label == "none"
            ]
        else:
            rows = [
                _read(f"{provider}/{body['stdout']}")
                for body in MANIFESTS[provider]["rows"].values()
                if (body.get("expect") or {}).get("salvage") == "none"
                and body.get("stdout")
            ]
        assert any(has_salvageable_json(stdout) for stdout in rows)


# --------------------------------------------------------------------------- #
# The hook's contract, beyond the committed captures.
# --------------------------------------------------------------------------- #

def _lines(relative: str) -> List[str]:
    return _read(relative).splitlines(keepends=True)


def _prefix_through(lines: List[str], predicate, occurrence: int = -1) -> str:
    """A byte prefix of a capture, ending after the chosen matching line.

    What a kill does to a stream: every byte is the CLI's own, at its original
    offset, and nothing after the cut exists.
    """
    hits = [index for index, line in enumerate(lines) if predicate(line)]
    assert hits, "the capture no longer has the line this cut is anchored on"
    return "".join(lines[: hits[occurrence] + 1])


def _event_type(line: str, key: str = "type") -> Optional[str]:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event.get(key) if isinstance(event, dict) else None


class TestOnlyTheLastMessageCounts:
    """D4.2: earlier messages are dropped, not joined — narration can quote an
    example array — and an unfinished last message is the one that counts."""

    def test_muse_cut_after_the_tool_result_drops_the_draft(self) -> None:
        # multi-assistant's first turn prints a DRAFT array, then calls
        # read_file. Cut right after the tool result: the closing turn has
        # written nothing, and the draft must not stand in for it.
        prefix = _prefix_through(
            _lines("muse/multi-assistant.stdout.txt"),
            lambda line: _event_type(line, "payload_type") == "tool.result",
        )
        assert "FIRST ANSWER" in prefix
        assert MuseProvider().salvage_text(prefix) == ""

    def test_muse_cut_before_the_terminal_record_keeps_the_closing_turn(self) -> None:
        lines = _lines("muse/multi-assistant.stdout.txt")
        prefix = _prefix_through(
            lines,
            lambda line: _event_type(line, "payload_type") == "run.output.delta",
        )
        assert '"payload_type":"run.terminal' not in prefix
        text = MuseProvider().salvage_text(prefix)
        assert text and "S001" in text and "FIRST ANSWER" not in text

    def test_muse_cut_mid_answer_keeps_the_partial_text(self) -> None:
        """Deltas are token fragments, so a cut inside the closing turn leaves
        its first half — the case salvage exists for.

        SYNTHETIC: the capture's closing turn is one delta, so its record is
        split in two (same envelope, the text halved) and the stream cut after
        the first half.
        """
        lines = _lines("muse/multi-assistant.stdout.txt")
        closing = max(
            index for index, line in enumerate(lines)
            if _event_type(line, "payload_type") == "run.output.delta"
        )
        record = json.loads(lines[closing])
        answer = record["payload"]["text"]
        record["payload"]["text"] = answer[: len(answer) // 2]
        cut = "".join(lines[:closing]) + json.dumps(record) + "\n"
        text = MuseProvider().salvage_text(cut)
        assert text == answer[: len(answer) // 2]
        assert has_salvageable_json(text)

    def test_opencode_cut_at_a_new_step_drops_the_earlier_narration(self) -> None:
        # run_review: step 2 says "Now let me explore the codebase …", and
        # step 3 starts. Killed there, the last message has no text yet.
        lines = _lines("opencode/run_review.1.18.30.jsonl")
        prefix = _prefix_through(
            lines, lambda line: _event_type(line) == "step_start", occurrence=2
        )
        assert "Now let me explore" in prefix
        assert OpenCodeProvider().salvage_text(prefix) == ""

    def test_opencode_takes_the_closing_step_not_the_first(self) -> None:
        text = OpenCodeProvider().salvage_text(_read("opencode/run_review.1.18.30.jsonl"))
        assert text.startswith("JSON written to")
        assert "Now let me explore" not in text

    def test_opencode_joins_one_message_split_across_parts(self) -> None:
        # A document split mid-token across two text parts of ONE message is
        # reassembled without a separator, the way parse_output's exact pass does.
        def text_part(message_id: str, text: str) -> str:
            return json.dumps({
                "type": "text",
                "part": {"type": "text", "messageID": message_id, "text": text},
            })

        stream = "\n".join([
            text_part("msg_1", 'Example: [{"title": "draft"}]'),
            text_part("msg_2", '[{"title": "real", "desc'),
            text_part("msg_2", '": "d", "importance": "HIGH"}]'),
        ])
        text = OpenCodeProvider().salvage_text(stream)
        assert text == '[{"title": "real", "desc": "d", "importance": "HIGH"}]'

    def test_codex_takes_the_last_agent_message(self) -> None:
        def message(text: str) -> str:
            return json.dumps({
                "type": "item.completed",
                "item": {"id": "item_0", "type": "agent_message", "text": text},
            })

        stream = "\n".join([
            json.dumps({"type": "turn.started"}),
            message('I will report findings like [{"title": "example"}].'),
            json.dumps({
                "type": "item.completed",
                "item": {"id": "item_1", "type": "command_execution",
                         "command": "rg x", "aggregated_output": "[1, 2]",
                         "exit_code": 0, "status": "completed"},
            }),
            message("Still reading the plan."),
        ])
        assert CodexProvider().salvage_text(stream) == "Still reading the plan."
        assert not has_salvageable_json(CodexProvider().salvage_text(stream))

    def test_copilot_cut_mid_stream_keeps_the_open_message(self) -> None:
        # success-streamed under `--stream on`, killed after the delta and
        # before the closing assistant.message: the deltas are the answer.
        lines = _lines("copilot/success-streamed.jsonl")
        prefix = _prefix_through(
            lines, lambda line: _event_type(line) == "assistant.message_delta"
        )
        assert '"type":"assistant.message"' not in prefix.replace(" ", "")
        text = CopilotProvider().salvage_text(prefix)
        recovered = extract_json_from_text(text, prefer_arrays=True)
        assert recovered["success"]
        assert [record["id"] for record in recovered["data"]] == ["S001", "S002"]

    def test_copilot_a_closed_message_outranks_its_own_deltas(self) -> None:
        stdout = _read("copilot/success-streamed.jsonl")
        text = CopilotProvider().salvage_text(stdout)
        assert text == CopilotProvider().salvage_text(_read("copilot/success.jsonl"))


#: Not the adapter's stream: prose, a bare JSON answer, an empty stdout.
NOT_A_STREAM = (
    "",
    "   \n",
    "I could not finish the review.",
    '[{"title": "t", "desc": "d", "importance": "HIGH"}]',
    '{"title": "t", "desc": "d", "importance": "HIGH"}',
)

#: Inputs a hook must survive without raising.
GARBAGE = (
    "\x00\xff�",
    "x" * 40000,
    "{" * 20000,
    '{"type": ["x"]}\n{"type": {"k": 1}, "part": {}}',
    "[" * 5000 + "]" * 5000,
    "null\ntrue\n42\n\"s\"",
    '{"type":"item.completed","item":{"type":"agent_message","text":5}}',
    '{"type":"assistant.message","data":{"content":["x"],"messageId":{}}}',
    '{"type":"assistant.message_delta","data":{"messageId":["x"],"deltaContent":1}}',
    '{"type":"text","part":{"type":"text","text":"x","messageID":["m"]}}',
    '{"type":"text","part":"not a dict"}',
    '{"payload_type":7,"payload":{}}',
    '{"payload_type":"run.output.delta","payload":{"text":["x"]}}',
    '{"payload_type":"run.terminal.completed","payload":{"text":5}}',
)


class TestTheHookContract:
    @pytest.mark.parametrize("provider", sorted(ADAPTERS))
    @pytest.mark.parametrize("stdout", NOT_A_STREAM)
    def test_not_this_adapters_stream_is_no_opinion(self, provider: str, stdout: str) -> None:
        assert ADAPTERS[provider]().salvage_text(stdout) is None

    @pytest.mark.parametrize("provider", sorted(ADAPTERS))
    @pytest.mark.parametrize("stdout", GARBAGE)
    def test_never_raises(self, provider: str, stdout: str) -> None:
        text = ADAPTERS[provider]().salvage_text(stdout)
        assert text is None or isinstance(text, str)

    @pytest.mark.parametrize("provider", ["codex", "copilot", "opencode"])
    @pytest.mark.parametrize("stdout", [
        '{"type": ["x"]}\n{"type": {"k": 1}, "part": {}}',
        '{"type":"assistant.message","data":{"content":"a","messageId":{}}}',
        '{"type":"assistant.message_delta","data":{"messageId":["x"],"deltaContent":"d"}}',
        '{"type":"text","part":{"type":"text","text":"x","messageID":["m"]}}',
    ])
    def test_the_shared_walk_keeps_parse_output_total(self, provider: str, stdout: str) -> None:
        """``parse_output`` now reads through the same walk; an id or a type of
        the wrong shape must not make it raise where it never did."""
        result = ADAPTERS[provider]().parse_output(stdout, "")
        assert isinstance(result, dict) and "success" in result

    @pytest.mark.parametrize("provider", sorted(ADAPTERS))
    def test_none_input_is_not_a_stream(self, provider: str) -> None:
        assert ADAPTERS[provider]().salvage_text(None) is None  # type: ignore[arg-type]

    def test_an_empty_answer_is_a_final_empty_verdict(self) -> None:
        # copilot's exit-0 failure: one assistant message whose content is "".
        text = CopilotProvider().salvage_text(_read("copilot/exit-zero-failure.jsonl"))
        assert text == ""
