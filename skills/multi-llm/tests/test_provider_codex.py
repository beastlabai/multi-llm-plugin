"""Tests for the Codex adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.retry_ladder import STATUS_AUTH_FAILED, STATUS_RATE_LIMITED
from utils.providers.codex import CodexProvider
from tests.provider_test_support import C1_FIXTURES, _c1_adapter, _c1_capture, _c1_without_the_hook


class TestCodexProvider:
    """Tests for the CodexProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a CodexProvider instance."""
        return CodexProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "codex"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_codex_reasoning_efforts_constant(self):
        """REASONING_EFFORTS is the seven values codex EMITS (L11 #1, §6.1).

        `max` was missing, so `gpt-5.6-sol:max` — a combination the codex
        catalog advertises — had its suffix glued onto the model name. `ultra`
        is deliberately NOT here: it is an inbound alias the harness rewrites,
        never a value this adapter puts on argv.
        """
        from utils.providers.codex import EFFORT_ALIASES, REASONING_EFFORTS

        assert REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh", "max",
        }
        assert EFFORT_ALIASES == {"ultra": "max"}

    def test_codex_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: -c pair after --model <base>, prompt last."""
        cmd = provider.build_command("p", "gpt-5.5:high")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.5",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_with_a_schema_exact(self, provider, tmp_path):
        """§4.2/§4.2.2's exact argv — codex's half is a FILE, not inline JSON."""
        from utils.providers.base import StructuredSchema

        path = tmp_path / "probe.schema.json"
        path.write_text('{"type": "object"}')
        schema = StructuredSchema(
            document={"type": "object"}, path=str(path), name="probe.schema.json"
        )

        cmd = provider.build_command("p", "gpt-5.5:high", schema=schema)

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--output-schema",
            str(path),
            "--model",
            "gpt-5.5",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the -c model_reasoning_effort form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            cmd = provider.build_command("Review this code", f"gpt-5.5:{effort}")

            assert cmd == [
                "codex",
                "exec",
                "--sandbox",
                "workspace-write",
                "--json",
                "-c", "features.plugins=false",
                "--ephemeral",
                "--skip-git-repo-check",
                "--model",
                "gpt-5.5",
                "-c",
                f"model_reasoning_effort={effort}",
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_codex_build_command_max_effort_reaches_the_flag(self, provider):
        """L11 #1's live defect, stated as the argv it broke.

        The codex catalog advertises `max` for `gpt-5.6-sol/-terra/-luna`
        (`codex debug models` → `supported_reasoning_levels`). Before L11 the
        adapter's whitelist stopped at `xhigh`, so `split_reasoning_effort`
        found no known suffix and emitted `--model gpt-5.6-sol:max` — a model id
        that does not exist — with no `-c` at all.
        """
        cmd = provider.build_command("p", "gpt-5.6-sol:max")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.6-sol",
            "-c",
            "model_reasoning_effort=max",
            "p",
        ]

    def test_codex_build_command_ultra_is_normalized_to_max(self, provider):
        """`ultra` is accepted at the harness boundary and EMITTED as `max` (§6.1).

        codex normalizes `ultra` down to `max` client-side, but that backstop is
        deliberately not relied on: `ultra` is not a canonical rung, so the
        rewrite happens here and argv carries `max`.
        """
        cmd = provider.build_command("p", "gpt-5.6-sol:ultra")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.6-sol",
            "-c",
            "model_reasoning_effort=max",
            "p",
        ]

    def test_codex_build_command_multi_colon_splits_on_last(self, provider):
        """Multi-colon model rsplits on the LAST colon: base foo:bar, effort high."""
        cmd = provider.build_command("p", "foo:bar:high")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "foo:bar",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_passthrough_no_colon(self, provider):
        """A model without a colon is passed verbatim with no -c argument."""
        cmd = provider.build_command("p", "gpt-5.5")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5", "p"]

    def test_codex_build_command_passthrough_unknown_suffix(self, provider):
        """An unknown suffix is not an effort: model passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:turbo")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5:turbo", "p"]

    def test_codex_build_command_passthrough_wrong_case_suffix(self, provider):
        """Effort matching is case-sensitive: :HIGH is passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:HIGH")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5:HIGH", "p"]

    def test_codex_build_command_passthrough_empty_suffix(self, provider):
        """A trailing colon with no effort is passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5:", "p"]

    def test_codex_build_command_passthrough_bare_effort_word(self, provider):
        """A bare effort word as the full model name (no colon) is passed verbatim."""
        cmd = provider.build_command("p", "high")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "high", "p"]

    def test_codex_build_command_passthrough_empty_base(self, provider):
        """An empty base before the colon (:high) is passed verbatim, no -c."""
        cmd = provider.build_command("p", ":high")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", ":high", "p"]

    # --- M2: --ephemeral + --skip-git-repo-check, unconditional (§3.2) -------
    #
    # Both were re-verified FUNCTIONALLY at the declared floor (`min_version`
    # 0.144.0) as well as at 0.146.0, which is why there is no version gate to
    # test: every codex dispatch carries them, on every profile and every
    # transport. `--ephemeral` stops each fan-out call writing a rollout file
    # into `$CODEX_HOME/sessions` that multi-llm never resumes (a four-model,
    # ten-phase run left 40 dead sessions); `--skip-git-repo-check` opens the
    # trust gate that otherwise makes codex exit 1 with ZERO requests outside a
    # git repo — the measured discriminator.

    M2_FLAGS = ("--ephemeral", "--skip-git-repo-check")

    def test_codex_m2_flags_on_every_profile_and_transport(self, provider):
        """No profile and no transport may drop either flag."""
        from utils.providers.base import PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE

        for transport in ("argv", "stdin"):
            argv, _stdin = provider.build_command_for_transport(
                "p", "gpt-5.5", transport)
            for flag in self.M2_FLAGS:
                assert flag in argv, (
                    f"generate/{transport} lost {flag}: {argv}")
            for profile in (PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
                argv, _stdin = provider.build_agent_command_for_transport(
                    "p", "gpt-5.5", profile, transport)
                for flag in self.M2_FLAGS:
                    assert flag in argv, (
                        f"{profile}/{transport} lost {flag}: {argv}")

    def test_codex_m2_flags_are_valueless_and_precede_the_prompt(self, provider):
        """Both are boolean flags, and neither may sit where it swallows argv[-1].

        A boolean flag placed last would take the prompt as its value on the
        real CLI, and `build_command_for_transport`'s stdin leg — which drops
        `cmd[-1]` — would then drop the wrong element.
        """
        cmd = provider.build_command("the prompt", "gpt-5.5:high")

        assert cmd[-1] == "the prompt"
        for flag in self.M2_FLAGS:
            index = cmd.index(flag)
            assert cmd[index + 1].startswith("-"), (
                f"{flag} is followed by {cmd[index + 1]!r}, which a boolean flag "
                "would swallow as its value")
        assert cmd.count("--ephemeral") == 1
        assert cmd.count("--skip-git-repo-check") == 1

    def test_codex_agent_readonly_keeps_m2_flags_and_rewrites_sandbox(self, provider):
        """The read-only rewrite is positional; M2's flags must survive it intact."""
        from utils.providers.base import PROFILE_AGENT_READONLY

        cmd = provider.build_agent_command("p", "gpt-5.5", PROFILE_AGENT_READONLY)

        assert cmd == ["codex", "exec", "--sandbox", "read-only", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5", "p"]

    @patch("shutil.which")
    def test_codex_is_available_true(self, mock_which, provider):
        """Test is_available returns True when codex is found."""
        mock_which.return_value = "/usr/bin/codex"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("codex")

    @patch("shutil.which")
    def test_codex_is_available_false(self, mock_which, provider):
        """Test is_available returns False when codex is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("codex")


def _codex_stream(*events) -> str:
    """Join events into the NDJSON stream shape `codex exec --json` writes."""
    return "\n".join(json.dumps(event) for event in events)


def _agent_message(text: str, item_id: str = "item_1") -> dict:
    return {"type": "item.completed",
            "item": {"id": item_id, "type": "agent_message", "text": text}}


class TestCodexParseOutput:
    """Parsing of the `codex exec --json` NDJSON event stream.

    Event shapes here are taken from live codex-cli 0.144.0 output. The reply
    is nested in `item.completed` -> `item.type == "agent_message"`; there is
    no top-level "text"/"message"/"content" event type (issue #2).
    """

    @pytest.fixture
    def provider(self):
        return CodexProvider()

    def test_agent_message_item_is_parsed(self, provider):
        """The advertised happy path: reply nested in item.completed."""
        stdout = _codex_stream(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            _agent_message('[{"task":"ship","verdict":"ready"}]'),
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship", "verdict": "ready"}]

    def test_reasoning_item_text_never_leaks(self, provider):
        """`reasoning` items also carry `text` — they must not reach the payload."""
        stdout = _codex_stream(
            {"type": "item.completed",
             "item": {"id": "rs_1", "type": "reasoning", "text": "SECRET_CHAIN_OF_THOUGHT"}},
            _agent_message('[{"task":"ship"}]', item_id="item_2"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]
        assert "SECRET_CHAIN_OF_THOUGHT" not in json.dumps(result["data"])

    def test_last_agent_message_wins_over_prose(self, provider):
        """Prose then JSON: take the final message, don't concatenate."""
        stdout = _codex_stream(
            _agent_message("Here is my analysis of the tasks.", item_id="item_1"),
            _agent_message('[{"task":"ship"}]', item_id="item_2"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_streaming_item_events_are_ignored(self, provider):
        """item.started/item.updated are deltas; only item.completed counts."""
        stdout = _codex_stream(
            {"type": "item.started",
             "item": {"id": "item_1", "type": "agent_message", "text": '[{"a":'}},
            {"type": "item.updated",
             "item": {"id": "item_1", "type": "agent_message", "text": '[{"a":1}]'}},
            _agent_message('[{"a":1}]'),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"a": 1}]

    def test_benign_item_error_does_not_fail_a_good_run(self, provider):
        """Nested item errors are routinely non-fatal (unknown model metadata)."""
        stdout = _codex_stream(
            {"type": "item.completed",
             "item": {"id": "item_0", "type": "error",
                      "message": "Model metadata for `x` not found. Defaulting to fallback metadata"}},
            _agent_message('[{"task":"ship"}]', item_id="item_1"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_top_level_error_is_surfaced(self, provider):
        """A quota failure must not be reported as 'No text events found'."""
        stdout = _codex_stream(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "error", "message": "Quota exceeded. Check your plan and billing details."},
            {"type": "turn.failed",
             "error": {"message": "Quota exceeded. Check your plan and billing details."}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "Quota exceeded" in result["error"]

    def test_turn_failed_alone_is_surfaced(self, provider):
        stdout = _codex_stream({"type": "turn.failed", "error": {"message": "upstream 503"}})

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "upstream 503" in result["error"]

    def test_benign_item_error_surfaced_when_nothing_else_parsed(self, provider):
        """With no reply at all, the nested error is better than a generic message."""
        stdout = _codex_stream(
            {"type": "item.completed",
             "item": {"id": "item_0", "type": "error", "message": "sandbox denied"}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "sandbox denied" in result["error"]

    def test_no_events_reports_no_text(self, provider):
        stdout = _codex_stream(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.completed", "usage": {}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["error"] == "No text events found in output"
        assert result["data"] is None

    def test_empty_agent_message_reports_empty(self, provider):
        result = provider.parse_output(_codex_stream(_agent_message("   ")), "")

        assert result["success"] is False
        assert result["error"] == "Empty text response"

    def test_malformed_and_non_dict_lines_are_skipped(self, provider):
        stdout = "\n".join([
            "not json at all",
            json.dumps(["a", "list", "not", "an", "object"]),
            json.dumps(_agent_message('[{"task":"ship"}]')),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_prose_wrapped_json_is_extracted(self, provider):
        """Non-JSON-leading replies still go through the extractor."""
        stdout = _codex_stream(
            _agent_message('Sure! Here you go:\n```json\n[{"task":"ship"}]\n```'))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_falls_back_to_joined_messages(self, provider):
        """If the final message alone yields nothing, try all of them joined."""
        stdout = _codex_stream(
            _agent_message('[{"task":"ship"}]', item_id="item_1"),
            _agent_message("Let me know if you need anything else.", item_id="item_2"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_legacy_text_events_still_parse(self, provider):
        """Pre-dotted-namespace shapes remain supported as a fallback."""
        stdout = _codex_stream({"type": "text", "part": {"type": "text", "text": '[{"a":1}]'}})

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"a": 1}]

    def test_agent_message_beats_legacy_events(self, provider):
        stdout = _codex_stream(
            {"type": "text", "text": '[{"from":"legacy"}]'},
            _agent_message('[{"from":"agent_message"}]'),
        )

        result = provider.parse_output(stdout, "")

        assert result["data"] == [{"from": "agent_message"}]


class TestCodexParseRealCaptures:
    """Regression tests against streams captured from a real codex CLI.

    Both fixtures are verbatim `codex exec --json` output from codex-cli
    0.144.0. They are the streams the pre-fix parser rejected outright.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "codex"

    @pytest.fixture
    def provider(self):
        return CodexProvider()

    def test_real_single_message_capture(self, provider):
        """gpt-5.6-sol at xhigh effort — the model this repo ships by default."""
        stdout = (self.FIXTURES / "gpt5_sol_single_message.jsonl").read_text(encoding="utf-8")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert result["data"], "expected a non-empty findings array"
        assert all("severity" in item for item in result["data"])

    def test_real_agentic_capture_excludes_reasoning(self, provider):
        """Agentic run with real chain-of-thought and two agent messages."""
        stdout = (self.FIXTURES / "agentic_reasoning_multi_message.jsonl").read_text(encoding="utf-8")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [
            {"entry": "directory listing blocked by sandbox"},
            {"entry": "directory listing blocked by sandbox"},
            {"entry": "directory listing blocked by sandbox"},
        ]

    def test_real_agentic_capture_leaks_no_reasoning_text(self, provider):
        """Every reasoning item's text must be absent from the parsed payload."""
        stdout = (self.FIXTURES / "agentic_reasoning_multi_message.jsonl").read_text(encoding="utf-8")
        reasoning = [
            json.loads(line)["item"]["text"]
            for line in stdout.splitlines()
            if line.strip() and json.loads(line).get("type") == "item.completed"
            and json.loads(line)["item"].get("type") == "reasoning"
        ]
        assert reasoning, "fixture must contain reasoning items to be meaningful"

        payload = json.dumps(provider.parse_output(stdout, "")["data"])

        for text in reasoning:
            assert text[:40] not in payload


#: Every codex capture, graded through `classify_failure` on the pre-T3.1
#: haystack (`before`) and through the hook (`after`), at exit 1 — the exit a
#: failed attempt reaches `_classify_failed_run` with. **This tuple is T3.1's
#: acceptance table**: exactly one cell moves. `agentic_reasoning_multi_message`
#: drew `permission_denied` from the model's own prose ("directory listing
#: blocked by sandbox", in a `reasoning` and an `agent_message` item); nothing
#: codex itself wrote says so. Graded 2026-09-25 while landing T3.1.
T3_CODEX_CORPUS = (
    # capture,                                     before,               after
    ("agentic_reasoning_multi_message.jsonl",      "permission_denied",  None),
    ("gpt5_sol_single_message.jsonl",              None,                 None),
    ("0.156.1/code_mode_mcp.jsonl",                None,                 None),
    ("0.156.1/code_mode_mcp.rollout.jsonl",        None,                 None),
    ("0.156.1/code_mode_writes.jsonl",             None,                 None),
    ("0.156.1/code_mode_writes.rollout.jsonl",     None,                 None),
    ("0.156.1/collab_spawn_agent.jsonl",           None,                 None),
    ("0.160.0/collab_spawn_agent.jsonl",           None,                 None),
    ("0.160.0/quoting_render.jsonl",               None,                 None),
    ("0.160.0/read_only_clean.jsonl",              None,                 None),
    ("0.160.1/collab_spawn_agent.jsonl",           None,                 None),
    ("0.160.1/quoting_render.jsonl",               None,                 None),
    ("0.160.1/read_only_clean.jsonl",              None,                 None),
    ("0.161.0/collab_spawn_agent.jsonl",           None,                 None),
    ("0.161.0/quoting_render.jsonl",               None,                 None),
    ("0.161.0/read_only_clean.jsonl",              None,                 None),
    ("0.163.0/exec_session.jsonl",                 None,                 None),
    ("0.163.0/quoting_render.jsonl",               None,                 None),
    ("0.163.0/read_only_clean.jsonl",              None,                 None),
    ("0.163.0/writes_patches_sessions.jsonl",      None,                 None),
    ("0.156.1/exec_session.jsonl",                 None,                 None),
    ("0.156.1/mcp_tool_call.jsonl",                None,                 None),
    ("0.156.1/quoting_render.jsonl",               None,                 None),
    ("0.156.1/read_only_clean.jsonl",              None,                 None),
    ("0.156.1/run_0c477c4d_partial.jsonl",         None,                 None),
    ("0.156.1/web_search_code_mode.jsonl",         None,                 None),
    ("0.156.1/web_search_code_mode.rollout.jsonl", None,                 None),
    ("0.156.1/web_search_hosted.jsonl",            None,                 None),
    ("0.156.1/web_search_hosted.sse.json",         None,                 None),
    ("0.156.1/web_search_hosted_replay.jsonl",     None,                 None),
    ("0.156.1/web_search_probe_replay.jsonl",      None,                 None),
    ("0.156.1/writes_patches_sessions.jsonl",      None,                 None),
    # Plan codex-sandbox-unavailable (2026-09-30): the preflight's `jail_writes`
    # turn on 0.158.0 inside the plugin-landlock jail — every write denied, so
    # the commands' own output is full of `Permission denied`, none of it the
    # vendor's. Graded `permission_denied` without the hook, nothing with it.
    ("landlock/jail_writes.out.jsonl",             "permission_denied",  None),
)


#: Subdirectories of `tests/fixtures/codex/` that hold rig scripts, schemas and
#: plugin-route data rather than a codex stdout capture.
_T3_CODEX_NOT_CAPTURES = ("capture", "schemas", "plugin-route", "rig")


def _t3_codex_item(item_type, **fields):
    item = {"id": f"item_{item_type}", "type": item_type, **fields}
    return json.dumps({"type": "item.completed", "item": item})


#: The case the plan names: a validator that grepped a plan discussing rate
#: limits and model names, on an attempt that failed for some other reason.
T3_CODEX_REPO_ECHO = "\n".join([
    json.dumps({"type": "thread.started", "thread_id": "0000"}),
    json.dumps({"type": "turn.started"}),
    _t3_codex_item(
        "command_execution",
        command="/bin/bash -lc \"rg -n '429|model' plan.md\"",
        aggregated_output=(
            "12: the client backs off on 429 Too Many Requests\n"
            "31: a typo in the id is reported as model not found\n"
            "40: 401 Unauthorized means the key was revoked\n"
        ),
        exit_code=0,
        status="completed",
    ),
    _t3_codex_item("reasoning", text="The plan says rate limit exceeded is retried."),
    _t3_codex_item("agent_message", text="Section 3 covers the 429 Too Many Requests path."),
]) + "\n"


class TestCodexVendorTextGradesEveryCapture:
    """T3.1 — T2.4's technique: every committed capture, before and after."""

    @pytest.mark.parametrize("capture,before,after", T3_CODEX_CORPUS)
    def test_after(self, capture, before, after):
        stdout = _c1_capture("codex", capture)
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == after

    @pytest.mark.parametrize("capture,before,after", T3_CODEX_CORPUS)
    def test_before(self, capture, before, after, monkeypatch):
        stdout = _c1_capture("codex", capture)
        _c1_without_the_hook(monkeypatch, "codex")
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == before

    def test_the_table_covers_every_capture(self):
        """A new capture is graded when it is committed, not when it bites."""
        root = C1_FIXTURES / "codex"
        on_disk = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in (".jsonl", ".json", ".txt")
            and not set(path.relative_to(root).parts) & set(_T3_CODEX_NOT_CAPTURES)
        }
        assert on_disk == {row[0] for row in T3_CODEX_CORPUS}

    def test_every_capture_is_only_ever_reduced(self):
        """Whole lines out, nothing in, stderr untouched (the base contract)."""
        adapter = _c1_adapter("codex")
        for capture, _before, _after in T3_CODEX_CORPUS:
            raw = _c1_capture("codex", capture)
            reduced, stderr = adapter.vendor_text(raw, "a stderr diagnosis")
            assert len(reduced) <= len(raw), capture
            remaining = iter(raw)
            assert all(char in remaining for char in reduced), capture
            assert stderr == "a stderr diagnosis", capture


class TestCodexVendorText:
    """T3.1 — what the hook drops, what it keeps, and how it degrades."""

    def test_repo_text_in_a_failed_attempt_classifies_as_nothing(self):
        """The plan's synthetic stream: `429` and `model not found` quoted."""
        adapter = _c1_adapter("codex")
        assert adapter.classify_failure(T3_CODEX_REPO_ECHO, "", 1) is None
        assert adapter.classify_failure(T3_CODEX_REPO_ECHO, "", 0) is None

    def test_and_that_is_a_change(self, monkeypatch):
        """On the pre-T3.1 haystack the quoted 401 outranked everything."""
        _c1_without_the_hook(monkeypatch, "codex")
        assert _c1_adapter("codex").classify_failure(
            T3_CODEX_REPO_ECHO, "", 1
        ) == STATUS_AUTH_FAILED

    @pytest.mark.parametrize("item_type", [
        "command_execution", "agent_message", "reasoning", "file_change",
        "mcp_tool_call", "web_search", "collab_tool_call",
    ])
    @pytest.mark.parametrize("event_type", [
        "item.started", "item.updated", "item.completed",
    ])
    def test_model_and_repo_items_are_dropped(self, item_type, event_type):
        line = json.dumps({"type": event_type, "item": {
            "id": "item_1", "type": item_type, "text": "rate limit exceeded",
        }})
        stdout = f'{{"type":"turn.started"}}\n{line}\n'
        reduced, _stderr = _c1_adapter("codex").vendor_text(stdout, "")
        assert reduced == '{"type":"turn.started"}\n'

    @pytest.mark.parametrize("line", [
        '{"type":"thread.started","thread_id":"0000"}',
        '{"type":"turn.started"}',
        '{"type":"turn.completed","usage":{"input_tokens":1}}',
        '{"type":"error","message":"exceeded retry limit, last status: 429"}',
        '{"type":"turn.failed","error":{"message":"unexpected status 401"}}',
        # codex's own non-fatal notice is an item, but it is codex's words.
        _t3_codex_item("error", message="Model metadata for `x` not found."),
        # Uncaptured item and event types are kept: the conservative direction.
        _t3_codex_item("todo_list", items=[]),
        '{"type":"some.future.event","message":"x"}',
        "stream error: exceeded retry limit, last status: 429 Too Many Requests",
    ])
    def test_codex_own_events_are_kept(self, line):
        stdout = f"{line}\n" + _t3_codex_item("agent_message", text="hi") + "\n"
        reduced, _stderr = _c1_adapter("codex").vendor_text(stdout, "")
        assert reduced == f"{line}\n"

    def test_a_real_429_is_not_outranked_by_a_quoted_401(self):
        """Auth is asked before rate limit, so a quoted 401 used to win."""
        stdout = T3_CODEX_REPO_ECHO + (
            '{"type":"turn.failed","error":{"message":"stream error: exceeded '
            'retry limit, last status: 429 Too Many Requests"}}\n'
        )
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == \
            STATUS_RATE_LIMITED

    def test_a_real_401_still_reaches_the_tables(self):
        stdout = T3_CODEX_REPO_ECHO + (
            '{"type":"turn.failed","error":{"message":"unexpected status 401 '
            'Unauthorized: Missing authentication in header"}}\n'
        )
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == \
            STATUS_AUTH_FAILED

    def test_stderr_is_never_reduced(self):
        stderr = "ERROR: 429 Too Many Requests"
        reduced, kept = _c1_adapter("codex").vendor_text(T3_CODEX_REPO_ECHO, stderr)
        assert kept == stderr
        assert _c1_adapter("codex").classify_failure(
            T3_CODEX_REPO_ECHO, stderr, 1
        ) == STATUS_RATE_LIMITED

    def test_a_cut_final_repo_item_is_dropped(self):
        """A stream killed mid-line: textual, and order-tolerant."""
        adapter = _c1_adapter("codex")
        head = '{"type":"turn.started"}\n'
        for cut in (
            '{"type":"item.completed","item":{"id":"item_7",'
            '"type":"command_execution","aggregated_output":"429 Too Many',
            # `type` is not the first key of either object.
            '{"item":{"id":"item_7","type":"agent_message","text":"x"},'
            '"type":"item.updated","extra":"rate lim',
        ):
            reduced, _stderr = adapter.vendor_text(head + cut, "")
            assert reduced == head, cut

    def test_a_cut_final_error_line_is_kept(self):
        stdout = '{"type":"turn.started"}\n{"type":"turn.failed","error":{"mess'
        assert _c1_adapter("codex").vendor_text(stdout, "") == (stdout, "")

    def test_a_cut_line_that_is_not_last_is_kept(self):
        """Only the END of a stream is cut by a kill; a beginning cut is kept."""
        cut = '"type":"command_execution","aggregated_output":"429 Too Many"}}'
        stdout = cut + '\n{"type":"turn.started"}\n'
        assert _c1_adapter("codex").vendor_text(stdout, "") == (stdout, "")

    @pytest.mark.parametrize("junk", [
        "", "   ", "not json at all", '{"unclosed": ', "\x00\x01",
        # A lone cut repo item: no line parsed, so not an event stream.
        '{"type":"item.completed","item":{"id":"i","type":"reasoning","text":"x',
    ])
    def test_no_parseable_line_is_identity(self, junk):
        assert _c1_adapter("codex").vendor_text(junk, "err") == (junk, "err")

    def test_a_raised_exception_is_identity(self, monkeypatch):
        from utils.providers import codex as codex_module

        def _boom(stdout):
            raise RuntimeError("reduction failed")

        monkeypatch.setattr(codex_module, "_vendor_stdout", _boom)
        assert _c1_adapter("codex").vendor_text(T3_CODEX_REPO_ECHO, "err") == (
            T3_CODEX_REPO_ECHO, "err"
        )

    def test_the_readonly_audit_still_sees_every_item(self):
        """The audit is separate: it judges the commands the hook drops."""
        from utils.providers import codex_stream_audit

        source = Path(codex_stream_audit.__file__).read_text(encoding="utf-8")
        assert "vendor_text" not in source
        assert "_vendor_stdout" not in source


class TestCodexSlotsStartWithoutThePlugins:
    """2026-10-02: a nested ``codex exec`` booted the user's installed plugins, so
    every review slot started multi-llm's own MCP server, whose venv Python wrote
    ``__pycache__/*.pyc`` into the watched ``$CODEX_HOME/plugins`` tree and fired
    ``[CODEX_HOME_CHECK]`` on every slot. ``-c features.plugins=false`` stops it."""

    def test_every_codex_argv_leg_carries_the_plugins_off_pair(self):
        from utils.providers.codex import PLUGINS_OFF_PAIR, CodexProvider

        provider = CodexProvider()
        cmd = provider.build_command("p", "gpt-5.5")
        assert PLUGINS_OFF_PAIR == ("-c", "features.plugins=false")
        at = cmd.index("-c")
        assert tuple(cmd[at:at + 2]) == PLUGINS_OFF_PAIR
        assert at < cmd.index("--ephemeral") < cmd.index("--skip-git-repo-check")
        assert cmd[2:4] == ["--sandbox", cmd[3]], "--sandbox keeps its fixed offset"
