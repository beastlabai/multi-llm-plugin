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

**Six more adapters** (plans/ask-answer-recovery-and-false-success.md T2.4,
D2.5): cline, goose, grok and vibe, labelled in their manifests, and kimi and
qwen-code, labelled by the tables below (both sit in ``MANIFEST_PENDING``).
Their hooks also feed the ask lane's ``answer_text``, and unlike the four
above they never answer ``None``: each CLI is always run in its structured
mode, so a stdout that does not hold the adapter's answer shape is a transport
that carried none, and ``""`` keeps that verdict final. Every committed answer
of theirs is JSON; the prose answers the gate cannot grade are pinned in
``tests/test_answer_text.py``.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_provider_fixture_rows import MANIFESTS, SALVAGE_LABELS  # noqa: E402
from tests.opencode_fixtures import PIN as OC_PIN  # noqa: E402
from utils.json_extractor import extract_json_from_text, has_salvageable_json  # noqa: E402
from utils.providers.aider import AiderProvider  # noqa: E402
from utils.providers.cline import ClineProvider, cline_answer_text  # noqa: E402
from utils.providers.codex import CodexProvider  # noqa: E402
from utils.providers.copilot import CopilotProvider  # noqa: E402
from utils.providers.goose import GooseProvider  # noqa: E402
from utils.providers.grok import GrokProvider  # noqa: E402
from utils.providers.kimi import KimiProvider  # noqa: E402
from utils.providers.muse import MuseProvider  # noqa: E402
from utils.providers.opencode import OpenCodeProvider  # noqa: E402
from utils.providers.qwen_code import QwenCodeProvider  # noqa: E402
from utils.providers.vibe import VibeProvider  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

ADAPTERS = {
    # plans/ask-answer-recovery-and-false-success.md T4.1 (D4.1): labelled in
    # its manifest; the prose answers are pinned in tests/test_aider_provider.py.
    "aider": AiderProvider,
    "cline": ClineProvider,
    "codex": CodexProvider,
    "copilot": CopilotProvider,
    "goose": GooseProvider,
    "grok": GrokProvider,
    "kimi": KimiProvider,
    "muse": MuseProvider,
    "opencode": OpenCodeProvider,
    "qwen-code": QwenCodeProvider,
    "vibe": VibeProvider,
}

#: The hooks that answer ``None`` ("not my stream") off their own stream shape.
NO_OPINION_ADAPTERS = ("codex", "copilot", "muse", "opencode")
#: The D2.5 hooks (T2.4, and aider's T4.1): ``""`` for anything that is not
#: their answer shape.
DECISIVE_ADAPTERS = ("aider", "cline", "goose", "grok", "kimi", "qwen-code", "vibe")

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
    # stream it answers from, salvage must recover the same document — or the
    # reduced text must BE that document whole: qwen's structured-output answer
    # `{"suggestions": [...]}` is served whole by its parser, while the
    # extractor's array preference lifts the inner array out of it.
    parsed = ADAPTERS[provider]().parse_output(stdout, "")
    if parsed.get("success"):
        try:
            whole = json.loads(reduced)
        except ValueError:
            whole = _UNPINNED
        assert parsed["data"] in (recovered["data"], whole)


# --------------------------------------------------------------------------- #
# The adapters with a manifest: the label is the row's `expect.salvage`.
# --------------------------------------------------------------------------- #

#: Every swept adapter that has a manifest; the others are labelled by TABLES.
MANIFEST_ADAPTERS = tuple(sorted(p for p in ADAPTERS if p in MANIFESTS))

MANIFEST_ROWS = [
    (provider, row)
    for provider in MANIFEST_ADAPTERS
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


def test_every_manifest_has_rows_to_sweep() -> None:
    """A manifest that lost its rows would make the sweep above vacuous."""
    assert {"cline", "copilot", "goose", "grok", "muse", "vibe"} <= set(MANIFEST_ADAPTERS)
    assert {p for p, _ in MANIFEST_ROWS} == set(MANIFEST_ADAPTERS)


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
    "0.160.0/collab_spawn_agent.jsonl": "answer",
    "0.160.0/quoting_render.jsonl": "answer",
    "0.160.0/read_only_clean.jsonl": "answer",
    "0.160.1/collab_spawn_agent.jsonl": "answer",
    "0.160.1/quoting_render.jsonl": "answer",
    "0.160.1/read_only_clean.jsonl": "answer",
    "0.161.0/collab_spawn_agent.jsonl": "answer",
    "0.161.0/quoting_render.jsonl": "answer",
    "0.161.0/read_only_clean.jsonl": "answer",
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
    # Plan codex-sandbox-unavailable (2026-09-30): the preflight's stub-driven
    # `jail_writes` turn inside the plugin-landlock jail, closing on `[]`.
    "landlock/jail_writes.out.jsonl": "answer",
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

#: The rig's cases (`capture/capture.py:_cases()`), all run on the pin.
_RIG_CASES = (
    "auth_error", "builtin_plansdir_build", "builtin_plansdir_ctrl_build",
    "builtin_plansdir_ctrl_plan", "builtin_plansdir_plan", "builtin_skilldir_build",
    "builtin_skilldir_ctrl_build", "builtin_skilldir_ctrl_plan",
    "builtin_skilldir_plan", "builtin_tmp_build", "builtin_tmp_ctrl_build",
    "builtin_tmp_ctrl_plan", "builtin_tmp_plan", "builtin_toolout_build",
    "builtin_toolout_plan", "denied_turn", "envperm_plan", "invalid_model",
    "managed_agent_plan", "managed_toplevel_plan", "org_agent_dbmem_plan",
    "org_agent_plan", "q2_ustar_agentlayer_build", "q2_ustar_agentlayer_plan",
    "q2_ustar_toplevel_build", "q2_ustar_toplevel_plan", "run_review",
    "s41_ask_bash", "s41_ask_read", "s41_cont_read", "s41_deny_bash", "s41_deny_read",
    "s41_denyallow_file_read", "s41_denyallow_read", "t40_badexp_plan",
    "t40_badperm_plan", "t40_badpermname_plan", "t40_block_build", "t40_block_plan",
    "t40_blockcont_build", "t40_blockcont_plan", "t40_cont_build", "t40_cont_plan",
    "t40_env_block_build", "t40_env_block_plan", "t40_env_none_build",
    "t40_env_none_plan", "t40_none_build", "t40_none_plan",
    "t47_inherited_star_build", "t47_inherited_star_ctrl_build", "t47_private_build",
    "t47_private_plan", "t47_shared_build", "t47_shared_plan",
    "t47_shared_toplevel_build", "t47_shared_toplevel_plan", "useragent_dir_build",
    "useragent_dir_plan", "useragent_relist_build", "useragent_relist_ctrl_build",
    "useragent_relist_ctrl_plan", "useragent_relist_plan",
    "useragent_relist_private_build", "useragent_relist_private_plan",
    "useragent_star_build", "useragent_star_plan", "useragent_ustar_build",
    "useragent_ustar_ctrl_build", "useragent_ustar_ctrl_plan", "useragent_ustar_plan",
    "write_outofcwd_build",
)

#: Every one is `none`: `auth_error` and `invalid_model` are one `error` event
#: (ProviderAuthError / UnknownError) with no text part; `run_review` closes on
#: the prose "JSON written to …"; `denied_turn` exits 0 on an auto-rejected
#: permission whose text parts are only narration, and the raw stream passed
#: the gate on its tool parts.
OPENCODE_LABELS: Dict[str, str] = {
    f"capture/{OC_PIN}/{case}.jsonl": "none" for case in _RIG_CASES
}


def _opencode_stdout_captures() -> Set[str]:
    root = FIXTURES / "opencode"
    found: Set[str] = set()
    for path in list(root.rglob("*.jsonl")) + list(root.rglob("*.stdout.txt")):
        if path.name.endswith(".requests.jsonl"):
            continue
        found.add(path.relative_to(root).as_posix())
    return found


# --------------------------------------------------------------------------- #
# kimi and qwen-code: which files are stdout, and their labels (T2.4).
#
# A kimi stdout capture is a `<row>.stdout.txt` under `tests/fixtures/kimi/`
# (the `.md` files are provenance notes, `.stderr.txt` the other stream). Only
# `success` and the synthesized `multi-assistant` carry an assistant record;
# the other eight are meta records only (5xx retries, auth, …) or, for
# `success-text`, the text-mode output (`• OK`) the adapter never asks for.
#
# A qwen-code stdout capture is a `<row>.stdout.json` (the `-o json` array) or
# `.stdout.txt` (`wall-time-exceeded`, an abort that printed nothing).
# --------------------------------------------------------------------------- #

KIMI_LABELS: Dict[str, str] = {
    "success.stdout.txt": "answer",
    # SYNTHESIZED (multi-assistant.SYNTHESIZED.md): the last assistant record
    # wins over an earlier one.
    "multi-assistant.stdout.txt": "answer",
    # Meta records only; `success` also ends on the `session.resume_hint`
    # record whose `content` is not an answer.
    "auth-error.stdout.txt": "none",
    "context-overflow.stdout.txt": "none",
    "ephemeral-only.stdout.txt": "none",
    "error-envelope.stdout.txt": "none",
    "invalid-model.stdout.txt": "none",
    "rate-limit.stdout.txt": "none",
    "truncated.stdout.txt": "none",
    "success-text.stdout.txt": "none",
}

QWEN_CODE_LABELS: Dict[str, str] = {
    # JSON answers the gate accepts.
    "success-document.stdout.json": "answer",
    "structured-output-success.stdout.json": "answer",
    # Real answers, but prose (`OK`, `LAST-ASSISTANT-TEXT`): the gate can grade
    # neither, so the hook's text is pinned by the tests below instead.
    "success.stdout.json": "none",
    "multi-assistant.stdout.json": "none",
    # `[API Error: …]` inside a record marked clean — two of them after partial
    # text (`OKOKOKOKOK[API Error: …]`) — and one error-shaped record.
    "auth-error.stdout.json": "none",
    "context-overflow.stdout.json": "none",
    "empty-stream.stdout.json": "none",
    "exit-zero-failure.stdout.json": "none",
    "invalid-model.stdout.json": "none",
    "no-finish-reason.stdout.json": "none",
    "truncated.stdout.json": "none",
    "stdout-error.stdout.json": "none",
    "wall-time-exceeded.stdout.txt": "none",
}


def _kimi_stdout_captures() -> Set[str]:
    return {path.name for path in (FIXTURES / "kimi").glob("*.stdout.txt")}


def _qwen_code_stdout_captures() -> Set[str]:
    root = FIXTURES / "qwen-code"
    return {
        path.name
        for path in list(root.glob("*.stdout.json")) + list(root.glob("*.stdout.txt"))
    }


TABLES: Dict[str, Tuple[Dict[str, str], Any]] = {
    "codex": (CODEX_LABELS, _codex_stdout_captures),
    "kimi": (KIMI_LABELS, _kimi_stdout_captures),
    "opencode": (OPENCODE_LABELS, _opencode_stdout_captures),
    "qwen-code": (QWEN_CODE_LABELS, _qwen_code_stdout_captures),
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
    ("opencode", f"opencode/capture/{OC_PIN}/denied_turn.jsonl"),
    ("opencode", f"opencode/capture/{OC_PIN}/run_review.jsonl"),
    # T2.4: goose's envelope, vibe's transcript and qwen's record array are
    # all answer-shaped to the gate (an object with an array-valued key, or an
    # array), whatever they carry.
    ("goose", "goose/auth-error.stdout.txt"),
    ("goose", "goose/truncated.stdout.txt"),
    ("goose", "goose/zero-body.stdout.txt"),
    ("vibe", "vibe/ephemeral-only.json"),
    ("qwen-code", "qwen-code/auth-error.stdout.json"),
    ("qwen-code", "qwen-code/truncated.stdout.json"),
)

#: Real answers the raw gate turned DOWN (the array hides in a JSON string).
ANSWERS_THAT_FAILED = (
    ("codex", "codex/gpt5_sol_single_message.jsonl"),
    ("codex", "codex/agentic_reasoning_multi_message.jsonl"),
    # T2.4: inside a `run_result`'s `text`, a wrapper's `text`, and an
    # assistant record's `content`.
    ("cline", "cline/success.stdout.txt"),
    ("grok", "grok/success.stdout.txt"),
    ("kimi", "kimi/success.stdout.txt"),
)

#: Adapters whose raw gate already rejected every ``none`` row (event lines and
#: error bodies with no array in them): what moved for them is the answer rows,
#: which the raw gate turned down.
MOVED_BY_AN_ANSWER = ("cline", "grok", "kimi")


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

    @staticmethod
    def _labelled(provider: str, wanted: str) -> List[str]:
        if provider in TABLES:
            table, _discover = TABLES[provider]
            return [
                _read(f"{provider}/{path}")
                for path, label in table.items() if label == wanted
            ]
        return [
            _read(f"{provider}/{body['stdout']}")
            for body in MANIFESTS[provider]["rows"].values()
            if (body.get("expect") or {}).get("salvage") == wanted
            and body.get("stdout")
        ]

    @pytest.mark.parametrize(
        "provider", sorted(set(ADAPTERS) - set(MOVED_BY_AN_ANSWER))
    )
    def test_every_adapter_has_a_none_row_the_raw_gate_passed(self, provider: str) -> None:
        """The sweep is not vacuous: each adapter has a row that moved."""
        rows = self._labelled(provider, "none")
        assert any(has_salvageable_json(stdout) for stdout in rows)

    @pytest.mark.parametrize("provider", MOVED_BY_AN_ANSWER)
    def test_the_rest_have_an_answer_row_the_raw_gate_turned_down(self, provider: str) -> None:
        """Their raw gate rejected every ``none`` row already; the answer rows
        are what moved, from rejected on the raw bytes to passing once reduced."""
        rows = self._labelled(provider, "answer")
        assert not any(has_salvageable_json(stdout) for stdout in rows)
        assert rows and all(has_salvageable_json(_reduced(provider, s)) for s in rows)


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
        lines = _lines(f"opencode/capture/{OC_PIN}/run_review.jsonl")
        prefix = _prefix_through(
            lines, lambda line: _event_type(line) == "step_start", occurrence=2
        )
        assert "Now let me explore" in prefix
        assert OpenCodeProvider().salvage_text(prefix) == ""

    def test_opencode_takes_the_closing_step_not_the_first(self) -> None:
        text = OpenCodeProvider().salvage_text(_read(f"opencode/capture/{OC_PIN}/run_review.jsonl"))
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
    # T2.4's shapes, each with a field of the wrong type.
    '{"type":"agent_event","event":{"type":"content_start","contentType":"text","accumulated":5}}',
    '{"type":"run_result","finishReason":["x"],"text":{"k":1}}',
    '{"messages":[{"role":"assistant","content":null}]}',
    '{"messages":[{"role":"assistant","content":[{"type":"text","text":5}]}]}',
    '{"messages":"x"}',
    '{"text":["x"],"type":{"k":1}}',
    '[{"role":"assistant","type":"message","content":[{"type":"text","text":5}]}]',
    '[{"role":"assistant","type":"message","content":"x"}]',
    '{"role":"assistant","content":["x"]}',
    '[{"type":"result","is_error":false,"subtype":"success","result":["x"]}]',
    '[{"type":"result","subtype":["x"],"result":"x"}]',
    '[{"type":"result","terminal_reason":7,"result":"x"}]',
)


class TestTheHookContract:
    @pytest.mark.parametrize("provider", NO_OPINION_ADAPTERS)
    @pytest.mark.parametrize("stdout", NOT_A_STREAM)
    def test_not_this_adapters_stream_is_no_opinion(self, provider: str, stdout: str) -> None:
        assert ADAPTERS[provider]().salvage_text(stdout) is None

    @pytest.mark.parametrize("provider", DECISIVE_ADAPTERS)
    @pytest.mark.parametrize("stdout", NOT_A_STREAM + (None,))
    def test_not_this_adapters_shape_is_a_final_empty_verdict(
        self, provider: str, stdout: Optional[str]
    ) -> None:
        """D2.5: these CLIs always run in their structured mode, so prose, a
        bare JSON value or no bytes at all hold no answer — ``""``, never
        ``None``, which would hand the whole stdout to the generic plain-text
        rule (vibe's config diagnostic) or the gate to stderr."""
        assert ADAPTERS[provider]().salvage_text(stdout) == ""  # type: ignore[arg-type]

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

    @pytest.mark.parametrize("provider", NO_OPINION_ADAPTERS)
    def test_none_input_is_not_a_stream(self, provider: str) -> None:
        assert ADAPTERS[provider]().salvage_text(None) is None  # type: ignore[arg-type]

    def test_an_empty_answer_is_a_final_empty_verdict(self) -> None:
        # copilot's exit-0 failure: one assistant message whose content is "".
        text = CopilotProvider().salvage_text(_read("copilot/exit-zero-failure.jsonl"))
        assert text == ""


# --------------------------------------------------------------------------- #
# T2.4: what each of the six D2.5 hooks reads, beyond the label sweep.
# --------------------------------------------------------------------------- #

def _manifest_stdouts(provider: str) -> List[Tuple[str, str]]:
    return [
        (row, _read(f"{provider}/{body['stdout']}") if body.get("stdout") else "")
        for row, body in (MANIFESTS[provider].get("rows") or {}).items()
        if "not_applicable" not in body
    ]


class TestTheClineHookIsTheReducer:
    """T2.4 wires T4.4's reducer in as the hook and writes no second one."""

    @pytest.mark.parametrize(
        "row,stdout", _manifest_stdouts("cline"),
        ids=[row for row, _ in _manifest_stdouts("cline")],
    )
    def test_on_every_fixture_row(self, row: str, stdout: str) -> None:
        assert ClineProvider().salvage_text(stdout) == cline_answer_text(stdout)

    def test_on_the_incident_tail(self) -> None:
        from tests.test_cline_reducer import incident_tail

        stdout = incident_tail()
        assert ClineProvider().salvage_text(stdout) == cline_answer_text(stdout) == ""

    def test_a_cut_mid_answer_is_a_partial(self) -> None:
        """SYNTHETIC ``partial`` row: no committed capture of the six is a cut
        before its answer ended (T2.4). ``success`` minus its ``done`` and
        ``run_result``, the text event's ``accumulated`` halved and nothing
        after it — a stream killed while the answer printed. cline is the one
        of the six whose answer streams in events, so the only one where a cut
        can leave half an answer behind."""
        events = [
            json.loads(line) for line in _read("cline/success.stdout.txt").splitlines()
            if line.strip()
        ]
        kept = []
        for event in events:
            inner = event.get("event") or {}
            if inner.get("type") == "content_start" and inner.get("contentType") == "text":
                answer = inner["accumulated"]
                inner["text"] = inner["accumulated"] = answer[: len(answer) // 2]
                kept.append(event)
                break
            kept.append(event)
        stdout = "\n".join(json.dumps(event) for event in kept) + "\n"
        text = ClineProvider().salvage_text(stdout)
        assert text and text == cline_answer_text(stdout)
        assert text == answer[: len(answer) // 2].strip()
        _assert_label("cline", stdout, "partial")
        # And the parser says so: no terminal event, so the answer is not whole.
        assert ClineProvider().parse_output(stdout, "")["incomplete"] is True


class TestKimiReadsOnlyAssistantRecords:
    def test_a_stream_whose_only_content_is_the_resume_hint_gives_nothing(self) -> None:
        records = [
            json.loads(line)
            for line in _read("kimi/success.stdout.txt").splitlines() if line.strip()
        ]
        kept = [record for record in records if record.get("role") != "assistant"]
        hints = [r for r in kept if r.get("type") == "session.resume_hint"]
        # The trap is live: the hint carries `content`, and it is all that does.
        assert hints and all(isinstance(r.get("content"), str) for r in hints)
        assert [r for r in kept if "content" in r] == hints
        stdout = "\n".join(json.dumps(record) for record in kept) + "\n"
        assert KimiProvider().salvage_text(stdout) == ""

    def test_a_line_separator_inside_the_answer_does_not_tear_its_record(self) -> None:
        # JSON leaves U+2028 unescaped inside a string; `str.splitlines` would
        # split the record there and read the stream as having no answer.
        answer = "First line second line"
        stdout = "\n".join([
            '{"role":"meta","type":"system.version","version":"0.36.1"}',
            json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
        ]) + "\n"
        assert KimiProvider().salvage_text(stdout) == answer
        assert KimiProvider().parse_output(stdout, "")["raw"] == answer

    def test_the_last_assistant_record_wins(self) -> None:
        text = KimiProvider().salvage_text(_read("kimi/multi-assistant.stdout.txt"))
        assert KimiProvider().parse_output(
            _read("kimi/multi-assistant.stdout.txt"), ""
        )["data"] == json.loads(text)


class TestGooseDropsItsOwnErrorProse:
    @pytest.mark.parametrize("row", ["truncated", "zero-body"])
    def test_network_error_text_is_not_an_answer(self, row: str) -> None:
        stdout = _read(f"goose/{row}.stdout.txt")
        from utils.providers.goose import last_assistant_text

        # The pin is live: the assistant text is goose's un-prefixed prose.
        assert last_assistant_text(json.loads(stdout)).strip().startswith("Network error")
        assert GooseProvider().salvage_text(stdout) == ""

    def test_the_compaction_error_is_not_an_answer_either(self) -> None:
        # "Ran into this error trying to compact:" — no colon after "error".
        stdout = _read("goose/context-overflow.stdout.txt")
        assert "Ran into this error trying to compact" in stdout
        assert GooseProvider().salvage_text(stdout) == ""

    def test_the_echoed_prompt_is_never_the_answer(self) -> None:
        text = GooseProvider().salvage_text(_read("goose/prompt-echo.stdout.txt"))
        assert extract_json_from_text(text, prefer_arrays=True)["success"]
        assert "timeout" not in text


class TestGrokNeverServesItsErrorBody:
    #: The 219-byte body grok printed on `runs/20260922T084502Z`, which the ask
    #: lane saved as grok's answer (the same bytes as
    #: `test_stray_write_paths.GROK_ERROR_BODY`).
    RECORDED = (
        '{"type":"error","message":"Not signed in. To authenticate without a '
        'browser, run:\\n  grok login --device-code\\n\\nAlternatively, set the '
        'XAI_API_KEY environment variable or run `grok login` on a machine with '
        'a browser."}'
    )

    def test_the_recorded_error_body_gives_nothing(self) -> None:
        assert GrokProvider().salvage_text(self.RECORDED) == ""

    def test_the_answer_is_the_wrappers_text(self) -> None:
        stdout = _read("grok/success.stdout.txt")
        assert GrokProvider().salvage_text(stdout) == json.loads(stdout)["text"].strip()


class TestVibeRefusesItsPlainTextDiagnostic:
    def test_the_config_diagnostic_on_stdout_gives_nothing(self) -> None:
        stdout = _read("vibe/stdout-error.stdout.txt")
        assert stdout.startswith("Invalid configuration")
        assert VibeProvider().salvage_text(stdout) == ""


class TestQwenCodeReadsOnlyCleanRecords:
    #: Every capture whose `result` carries qwen's `[API Error:` marker.
    API_ERROR_ROWS = (
        "auth-error", "context-overflow", "empty-stream", "exit-zero-failure",
        "invalid-model", "no-finish-reason", "truncated",
    )
    ANSWER_ROWS = (
        "success", "success-document", "structured-output-success", "multi-assistant",
    )

    @staticmethod
    def _result(row: str) -> Dict[str, Any]:
        records = json.loads(_read(f"qwen-code/{row}.stdout.json"))
        return [r for r in records if r.get("type") == "result"][-1]

    @pytest.mark.parametrize("row", API_ERROR_ROWS)
    def test_an_api_error_row_gives_nothing(self, row: str) -> None:
        record = self._result(row)
        # The pin is live: qwen marked the record clean around its error.
        assert not record.get("is_error") and record.get("subtype") == "success"
        assert "[API Error:" in record["result"]
        assert QwenCodeProvider().salvage_text(_read(f"qwen-code/{row}.stdout.json")) == ""

    @pytest.mark.parametrize("row", ANSWER_ROWS)
    def test_an_answer_row_gives_its_answer(self, row: str) -> None:
        text = QwenCodeProvider().salvage_text(_read(f"qwen-code/{row}.stdout.json"))
        assert text and text == self._result(row)["result"]

    def test_the_prose_answers_are_themselves(self) -> None:
        assert QwenCodeProvider().salvage_text(_read("qwen-code/success.stdout.json")) == "OK"
        assert QwenCodeProvider().salvage_text(
            _read("qwen-code/multi-assistant.stdout.json")
        ) == "LAST-ASSISTANT-TEXT"

    def test_an_error_shaped_record_gives_nothing(self) -> None:
        assert QwenCodeProvider().salvage_text(_read("qwen-code/stdout-error.stdout.json")) == ""
