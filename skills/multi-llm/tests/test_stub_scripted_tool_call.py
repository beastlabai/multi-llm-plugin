"""The scripted-call stub (plans/codex-native-plugin.md, Phase 1 item 1a), offline.

``tests/container/stub_server.py`` grew exactly one scripted shape: a scenario
carrying ``scripted_tool_call:`` makes the Responses surface answer a request
that ADVERTISES the named tool with one ``function_call``, and answer the request
carrying the matching ``function_call_output`` with a plain terminal message.

These tests drive the REAL server process on loopback — the parity module's
reason applies here too: the claim is about the server the container runs, whose
script is armed at startup by ``install_resolver`` and rendered by its own SSE
framing, and an in-process call would compare the module against itself. No
Docker, no codex, no network beyond 127.0.0.1, so plain CI runs them.

What they pin:

* the call is served only on the named scenario, only on the Responses surface,
  and only to a request whose ``tools[]`` carries the tool — namespace form
  (codex's measured shape) and flat ``<namespace>__<name>`` form both;
* the output round-trips: the request carrying ``function_call_output`` for the
  script's ``call_id`` gets the terminal message, whatever the turn counter says;
* the frames served, the declaration seen and the output returned are recorded
  in the log;
* every other scenario, and a scenario-less stub, never emits a tool call even
  to a request that advertises the tool — the text path is unchanged;
* the chat-completions leg (plans/opencode-validator-false-rate-limit-and-
  denied-turn.md, T4.10), served only when ``surfaces:`` names ``chat``: one
  assistant ``tool_calls`` message to a request advertising the tool, the
  terminal message to the request carrying the matching ``role: "tool"``
  message, and the call again for a tool message with another id.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

TESTS_DIR = Path(__file__).resolve().parent
STUB_SERVER = TESTS_DIR / "container" / "stub_server.py"
SCENARIOS = TESTS_DIR / "fixtures" / "e2e" / "scenarios"
SCRIPTED = SCENARIOS / "scripted_codex_plugin_tool_call.yaml"
TEXT_ONLY = SCENARIOS / "container_full_chain.yaml"
#: T4.9's deny rows (plans/opencode-validator-false-rate-limit-and-denied-
#: turn.md): the chain's rules plus one chat-surface `read` of `/etc/hostname`.
DENY = SCENARIOS / "scripted_opencode_read_deny.yaml"

#: The tools[] entry codex 0.155.1 sent for a plugin MCP server (the plan's
#: Appendix B capture), trimmed to the keys the stub reads plus the schema.
NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "mcp__multi_llm",
    "tools": [{
        "type": "function",
        "name": "multi_llm",
        "description": "probe. This tool is part of plugin `Multi-LLM`.",
        "parameters": {"type": "object",
                       "properties": {"mode": {"type": "string"}},
                       "required": ["mode"]},
    }],
}
BUILTIN_TOOL = {"type": "function", "name": "exec_command",
                "parameters": {"type": "object", "properties": {}}}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Stub:
    def __init__(self, tmp: Path, scenario: Path | None):
        self.port = _free_port()
        self.log = tmp / f"stub-{self.port}.jsonl"
        argv = [sys.executable, str(STUB_SERVER), "--port", str(self.port),
                "--log", str(self.log)]
        if scenario is not None:
            argv += ["--scenario", str(scenario)]
        env = os.environ.copy()
        # The stub reaches for `tests.scenario_matcher`; the skill dir is the
        # parent of this tests package (the parity module's arrangement).
        env["PYTHONPATH"] = os.pathsep.join(
            [str(TESTS_DIR.parent), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, env=env)
        self.base = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"stub exited {self.proc.returncode}: {self.proc.stderr.read()}")
            try:
                with urllib.request.urlopen(self.base + "/__health", timeout=1):
                    return
            except OSError:
                time.sleep(0.1)
        raise AssertionError("stub never answered /__health")

    def post(self, path: str, body: dict, stream: bool) -> tuple[int, str]:
        body = dict(body, stream=stream)
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()

    def entries(self) -> list[dict]:
        return [json.loads(line) for line in
                self.log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()


@pytest.fixture
def stub(tmp_path):
    started: list[_Stub] = []

    def factory(scenario: Path | None) -> _Stub:
        server = _Stub(tmp_path, scenario)
        started.append(server)
        return server

    yield factory
    for server in started:
        server.close()


def _sse_events(text: str) -> list[tuple[str | None, dict]]:
    events = []
    for block in text.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if data is not None and data != "[DONE]":
            events.append((name, json.loads(data)))
    return events


def _request(*, tools=(), extra_input=()) -> dict:
    return {
        "model": "stub-model",
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text",
                                "text": "Use the multi-llm tool."}]},
                  *extra_input],
        "tools": list(tools),
    }


def _output_item(call_id: str, output) -> dict:
    return {"type": "function_call_output", "call_id": call_id,
            "output": output}


SCRIPT = yaml.safe_load(SCRIPTED.read_text(encoding="utf-8"))["scripted_tool_call"]


class TestTheScenarioFile:
    def test_it_declares_the_plugin_tool_as_codex_presents_it(self):
        assert SCRIPT["namespace"] == "mcp__multi_llm"
        assert SCRIPT["name"] == "multi_llm"
        assert SCRIPT["surfaces"] == ["responses"]

    def test_its_arguments_are_zero_quota_and_relative(self):
        # `status` spawns no provider; the plan path is relative so the anchor
        # (D3f) is exercised by every row that drives this scenario.
        assert SCRIPT["arguments"]["mode"] == "status"
        assert not Path(SCRIPT["arguments"]["plan"]).is_absolute()

    def test_no_other_scenario_carries_a_script(self):
        carriers = sorted(
            path.name for path in SCENARIOS.glob("*.yaml")
            if "scripted_tool_call" in (yaml.safe_load(
                path.read_text(encoding="utf-8")) or {}))
        assert carriers == sorted([SCRIPTED.name, DENY.name]), (
            f"{carriers}: the scripted shape is reached only through these "
            f"named scenarios (codex's plugin call; T4.9's opencode deny rows), "
            f"so no existing row can start seeing tool calls")


class TestTheCallIsServed:
    @pytest.mark.parametrize("stream", [True, False], ids=["sse", "json"])
    def test_an_advertising_request_gets_one_function_call(self, stub, stream):
        server = stub(SCRIPTED)
        status, text = server.post(
            "/v1/responses", _request(tools=[BUILTIN_TOOL, NAMESPACE_TOOL]), stream)
        assert status == 200
        if stream:
            events = _sse_events(text)
            names = [name for name, _ in events]
            assert names[0] == "response.created"
            assert names[-1] == "response.completed"
            assert "response.function_call_arguments.done" in names
            assert "response.output_text.delta" not in names
            output = events[-1][1]["response"]["output"]
        else:
            output = json.loads(text)["output"]
        assert len(output) == 1, "exactly one output item: the call"
        call = output[0]
        assert call["type"] == "function_call"
        assert call["namespace"] == "mcp__multi_llm"
        assert call["name"] == "multi_llm"
        assert call["call_id"] == SCRIPT["call_id"]
        assert json.loads(call["arguments"]) == SCRIPT["arguments"]

    def test_a_flat_declaration_is_answered_in_its_own_spelling(self, stub):
        server = stub(SCRIPTED)
        flat = {"type": "function", "name": "mcp__multi_llm__multi_llm",
                "parameters": {"type": "object"}}
        _status, text = server.post("/v1/responses", _request(tools=[flat]), False)
        call = json.loads(text)["output"][0]
        assert call["name"] == "mcp__multi_llm__multi_llm"
        assert "namespace" not in call

    @pytest.mark.parametrize("output", [
        "MULTI_LLM_RESULT: {\"exit\": 2}",
        [{"type": "input_text", "text": "MULTI_LLM_RESULT: {\"exit\": 2}"}],
    ], ids=["string", "content-items"])
    def test_the_output_round_trips_to_a_terminal_message(self, stub, output):
        server = stub(SCRIPTED)
        call = {"type": "function_call", "call_id": SCRIPT["call_id"],
                "name": "multi_llm", "namespace": "mcp__multi_llm",
                "arguments": json.dumps(SCRIPT["arguments"])}
        status, text = server.post(
            "/v1/responses",
            _request(tools=[NAMESPACE_TOOL],
                     extra_input=[call, _output_item(SCRIPT["call_id"], output)]),
            True)
        assert status == 200
        events = _sse_events(text)
        final = events[-1][1]["response"]["output"]
        assert [item["type"] for item in final] == ["message"]
        assert final[0]["content"][0]["text"] == SCRIPT["terminal"]
        entry = [e for e in server.entries() if e["surface"] == "responses"][-1]
        assert entry["scripted"]["phase"] == "terminal"
        assert "MULTI_LLM_RESULT" in entry["scripted"]["output"]
        assert entry["scripted"]["output_truncated"] is False

    def test_an_output_for_another_call_id_is_not_ours(self, stub):
        server = stub(SCRIPTED)
        _status, text = server.post(
            "/v1/responses",
            _request(tools=[NAMESPACE_TOOL],
                     extra_input=[_output_item("call_somebody_else", "x")]),
            False)
        # Not our output, and the tool is still advertised: the call again.
        assert json.loads(text)["output"][0]["type"] == "function_call"


class TestTheFramesAreRecorded:
    def test_the_log_carries_frames_declaration_and_arguments(self, stub):
        server = stub(SCRIPTED)
        server.post("/v1/responses", _request(tools=[NAMESPACE_TOOL]), True)
        entry = [e for e in server.entries() if e["surface"] == "responses"][-1]
        scripted = entry["scripted"]
        assert scripted["phase"] == "call"
        assert scripted["declaration"] == NAMESPACE_TOOL, (
            "the declaration codex advertised is the loader row's tools[] "
            "capture, and it must be recorded verbatim")
        assert scripted["arguments"] == SCRIPT["arguments"]
        assert [f["event"] for f in scripted["frames"]][-1] == "response.completed"
        assert entry["fixture"] == "scripted_tool_call:call"
        assert entry["unmatched"] is False


class TestTheTextPathIsUnchanged:
    def test_a_request_that_does_not_advertise_the_tool_gets_text(self, stub):
        server = stub(SCRIPTED)
        _status, text = server.post("/v1/responses",
                                    _request(tools=[BUILTIN_TOOL]), False)
        output = json.loads(text)["output"]
        assert [item["type"] for item in output] == ["message"]
        assert output[0]["content"][0]["text"] == "OK"
        entry = [e for e in server.entries() if e["surface"] == "responses"][-1]
        assert entry["scripted"]["phase"] == "not-advertised"

    def test_the_chat_surface_never_sees_a_call(self, stub):
        server = stub(SCRIPTED)
        body = {"model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [NAMESPACE_TOOL]}
        _status, text = server.post("/v1/chat/completions", body, False)
        message = json.loads(text)["choices"][0]["message"]
        assert "tool_calls" not in message
        assert message["content"] == "OK"

    @pytest.mark.parametrize("scenario", [None, TEXT_ONLY],
                             ids=["no-scenario", "container_full_chain"])
    def test_no_other_scenario_emits_a_call_even_when_advertised(
            self, stub, scenario):
        server = stub(scenario)
        _status, text = server.post(
            "/v1/responses", _request(tools=[NAMESPACE_TOOL]), True)
        types = {item["type"] for _n, data in _sse_events(text)
                 for item in (data.get("response") or {}).get("output") or []}
        assert "function_call" not in types
        for entry in server.entries():
            assert "scripted" not in entry, (
                "a stub whose scenario declares no script must log exactly the "
                "shape it always did")


class TestAMalformedScriptIsFatal:
    def test_a_script_without_arguments_refuses_to_start(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({
            "name": "bad",
            "scripted_tool_call": {"namespace": "mcp__multi_llm",
                                   "name": "multi_llm"},
        }), encoding="utf-8")
        with pytest.raises(AssertionError, match="stub exited"):
            _Stub(tmp_path, bad)


# --------------------------------------------------------------------------- #
# The chat-completions leg (T4.10). The script lives in a tmp scenario, not in
# SCENARIOS, so `test_no_other_scenario_carries_a_script` still holds.
# --------------------------------------------------------------------------- #

#: opencode's `read` tool as an openai-compatible host advertises it.
CHAT_READ_TOOL = {
    "type": "function",
    "function": {"name": "read", "description": "Read a file.",
                 "parameters": {"type": "object",
                                "properties": {"filePath": {"type": "string"}},
                                "required": ["filePath"]}},
}
CHAT_BASH_TOOL = {"type": "function",
                  "function": {"name": "bash", "parameters": {"type": "object"}}}
CHAT_SCRIPT = {"surfaces": ["chat"], "name": "read",
               "call_id": "call_mllm_chat_1",
               "arguments": {"filePath": "/etc/hostname"},
               "terminal": "chat scripted turn complete"}


def _chat_scenario(tmp_path: Path, script: dict = CHAT_SCRIPT) -> Path:
    path = tmp_path / "chat_scripted.yaml"
    path.write_text(yaml.safe_dump({
        "name": "chat_scripted", "description": "T4.10 chat leg",
        "scripted_tool_call": script,
        "prompts": [{"pattern": "read the file", "content": "OK"}],
        "fallback": {"content": "OK"},
    }), encoding="utf-8")
    return path


def _chat_request(*, tools=(), extra_messages=()) -> dict:
    return {"model": "stub-model",
            "messages": [{"role": "user", "content": "read the file"},
                         *extra_messages],
            "tools": list(tools)}


def _chat_follow_up(call_id: str, content) -> list[dict]:
    """The assistant turn that made the call, then the tool's result."""
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": "read",
                                      "arguments": json.dumps(
                                          CHAT_SCRIPT["arguments"])}}]},
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def _last_entry(server: "_Stub", surface: str, count: int = 1) -> dict:
    """The newest log line for ``surface``, waiting for it to land.

    The stub writes its log line after the response (`_dispatch`'s
    ``finally``), so a JSON answer can reach the client before the line does.
    """
    deadline = time.time() + 5
    while True:
        lines = [e for e in server.entries() if e["surface"] == surface]
        if len(lines) >= count or time.time() > deadline:
            assert lines, f"no {surface} log line"
            return lines[-1]
        time.sleep(0.05)


def _chat_stream_result(text: str) -> tuple[str, list[dict], str | None]:
    """Reassemble a chat SSE body: (content, tool calls, finish_reason)."""
    assert text.rstrip().endswith("data: [DONE]")
    content, calls, finish = "", {}, None
    for name, chunk in _sse_events(text):
        assert name is None, "chat chunks carry no `event:` line"
        choice = chunk["choices"][0]
        delta = choice["delta"]
        content += delta.get("content") or ""
        for part in delta.get("tool_calls") or []:
            call = calls.setdefault(part["index"],
                                    {"function": {"name": "", "arguments": ""}})
            for key in ("id", "type"):
                if key in part:
                    call[key] = part[key]
            function = part.get("function") or {}
            call["function"]["name"] += function.get("name") or ""
            call["function"]["arguments"] += function.get("arguments") or ""
        finish = choice["finish_reason"] or finish
    return content, [calls[i] for i in sorted(calls)], finish


class TestTheChatLeg:
    @pytest.mark.parametrize("stream", [True, False], ids=["sse", "json"])
    def test_an_advertising_request_gets_one_tool_call(
            self, stub, tmp_path, stream):
        server = stub(_chat_scenario(tmp_path))
        status, text = server.post(
            "/v1/chat/completions",
            _chat_request(tools=[CHAT_BASH_TOOL, CHAT_READ_TOOL]), stream)
        assert status == 200
        if stream:
            content, calls, finish = _chat_stream_result(text)
            assert content == ""
        else:
            choice = json.loads(text)["choices"][0]
            calls, finish = choice["message"]["tool_calls"], choice["finish_reason"]
            assert choice["message"]["content"] is None
        assert finish == "tool_calls"
        assert len(calls) == 1, "exactly one tool call"
        call = calls[0]
        assert call["id"] == CHAT_SCRIPT["call_id"]
        assert call["type"] == "function"
        assert call["function"]["name"] == "read"
        assert json.loads(call["function"]["arguments"]) == CHAT_SCRIPT["arguments"]
        entry = _last_entry(server, "chat")
        assert entry["scripted"]["phase"] == "call"
        assert entry["scripted"]["declaration"] == CHAT_READ_TOOL
        assert entry["fixture"] == "scripted_tool_call:call"

    @pytest.mark.parametrize("stream", [True, False], ids=["sse", "json"])
    @pytest.mark.parametrize("content", [
        "error: permission denied for /etc/hostname",
        [{"type": "text", "text": "error: permission denied for /etc/hostname"}],
    ], ids=["string", "content-parts"])
    def test_the_matching_tool_message_gets_the_terminal(
            self, stub, tmp_path, stream, content):
        server = stub(_chat_scenario(tmp_path))
        status, text = server.post(
            "/v1/chat/completions",
            _chat_request(tools=[CHAT_READ_TOOL],
                          extra_messages=_chat_follow_up(
                              CHAT_SCRIPT["call_id"], content)),
            stream)
        assert status == 200
        if stream:
            answer, calls, finish = _chat_stream_result(text)
        else:
            choice = json.loads(text)["choices"][0]
            answer = choice["message"]["content"]
            calls = choice["message"].get("tool_calls") or []
            finish = choice["finish_reason"]
        assert answer == CHAT_SCRIPT["terminal"]
        assert calls == []
        assert finish == "stop"
        entry = _last_entry(server, "chat")
        assert entry["scripted"]["phase"] == "terminal"
        assert "permission denied" in entry["scripted"]["output"]
        assert entry["scripted"]["output_truncated"] is False
        assert entry["fixture"] == "scripted_tool_call:terminal"

    def test_a_tool_message_for_another_call_id_is_not_ours(
            self, stub, tmp_path):
        server = stub(_chat_scenario(tmp_path))
        _status, text = server.post(
            "/v1/chat/completions",
            _chat_request(tools=[CHAT_READ_TOOL],
                          extra_messages=_chat_follow_up(
                              "call_somebody_else", "x")),
            False)
        # Not our output, and the tool is still advertised: the call again.
        choice = json.loads(text)["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["id"] == CHAT_SCRIPT["call_id"]
        entry = _last_entry(server, "chat")
        assert entry["scripted"]["phase"] == "call"

    def test_a_request_that_does_not_advertise_the_tool_gets_text(
            self, stub, tmp_path):
        server = stub(_chat_scenario(tmp_path))
        _status, text = server.post(
            "/v1/chat/completions", _chat_request(tools=[CHAT_BASH_TOOL]), False)
        message = json.loads(text)["choices"][0]["message"]
        assert "tool_calls" not in message
        assert message["content"] == "OK"
        entry = _last_entry(server, "chat")
        assert entry["scripted"]["phase"] == "not-advertised"

    def test_a_chat_only_script_leaves_responses_untouched(
            self, stub, tmp_path):
        server = stub(_chat_scenario(tmp_path))
        flat = {"type": "function", "name": "read",
                "parameters": {"type": "object"}}
        _status, text = server.post("/v1/responses", _request(tools=[flat]),
                                    False)
        output = json.loads(text)["output"]
        assert [item["type"] for item in output] == ["message"]
        entry = _last_entry(server, "responses")
        assert "scripted" not in entry

    def test_then_shell_on_the_chat_leg_refuses_to_start(self, tmp_path):
        bad = _chat_scenario(tmp_path, dict(CHAT_SCRIPT, then_shell="true"))
        with pytest.raises(AssertionError, match="stub exited"):
            _Stub(tmp_path, bad)

    def test_an_unknown_surface_refuses_to_start(self, tmp_path):
        bad = _chat_scenario(tmp_path, dict(CHAT_SCRIPT, surfaces=["anthropic"]))
        with pytest.raises(AssertionError, match="stub exited"):
            _Stub(tmp_path, bad)

    def test_an_unknown_after_output_refuses_to_start(self, tmp_path):
        bad = _chat_scenario(tmp_path, dict(CHAT_SCRIPT, after_output="judge"))
        with pytest.raises(AssertionError, match="stub exited"):
            _Stub(tmp_path, bad)

    def test_a_terminal_beside_after_output_text_refuses_to_start(self, tmp_path):
        # `terminal` would never be served, so declaring both is a mistake.
        bad = _chat_scenario(tmp_path, dict(CHAT_SCRIPT, after_output="text"))
        with pytest.raises(AssertionError, match="stub exited"):
            _Stub(tmp_path, bad)


# --------------------------------------------------------------------------- #
# `after_output: text` (T4.9): the output's turn goes back to the text path.
# --------------------------------------------------------------------------- #

TEXT_SCRIPT = {key: value for key, value in CHAT_SCRIPT.items()
               if key != "terminal"}
TEXT_SCRIPT["after_output"] = "text"

#: What opencode 1.18.32 hands the model for a `deny` rule (T4.6 rig,
#: `tests/fixtures/opencode/capture/SCHEMA.md`), trimmed.
DENY_TEXT = ("The user has specified a rule which prevents you from using this "
             "specific tool call. Here are some of the relevant rules "
             '[{"permission":"external_directory","pattern":"*","action":"deny"}]')


class TestAfterOutputText:
    @pytest.mark.parametrize("stream", [True, False], ids=["sse", "json"])
    def test_the_call_is_still_served_first(self, stub, tmp_path, stream):
        server = stub(_chat_scenario(tmp_path, TEXT_SCRIPT))
        _status, text = server.post(
            "/v1/chat/completions", _chat_request(tools=[CHAT_READ_TOOL]),
            stream)
        if stream:
            _content, calls, finish = _chat_stream_result(text)
        else:
            choice = json.loads(text)["choices"][0]
            calls, finish = choice["message"]["tool_calls"], choice["finish_reason"]
        assert finish == "tool_calls"
        assert [c["function"]["name"] for c in calls] == ["read"]

    @pytest.mark.parametrize("stream", [True, False], ids=["sse", "json"])
    def test_the_output_turn_is_answered_by_the_scenario_rules(
            self, stub, tmp_path, stream):
        server = stub(_chat_scenario(tmp_path, TEXT_SCRIPT))
        status, text = server.post(
            "/v1/chat/completions",
            _chat_request(tools=[CHAT_READ_TOOL],
                          extra_messages=_chat_follow_up(
                              TEXT_SCRIPT["call_id"], DENY_TEXT)),
            stream)
        assert status == 200
        if stream:
            answer, calls, finish = _chat_stream_result(text)
        else:
            choice = json.loads(text)["choices"][0]
            answer = choice["message"]["content"]
            calls = choice["message"].get("tool_calls") or []
            finish = choice["finish_reason"]
        # The `read the file` rule's answer, not a terminal string.
        assert answer == "OK"
        assert calls == []
        assert finish == "stop"
        entry = _last_entry(server, "chat")
        assert entry["scripted"]["phase"] == "output-to-text"
        assert entry["scripted"]["output"] == DENY_TEXT
        assert entry["scripted"]["output_truncated"] is False
        assert entry["fixture"] != "scripted_tool_call:terminal"

    def test_the_responses_leg_hands_back_the_same_way(self, stub, tmp_path):
        script = dict(TEXT_SCRIPT, surfaces=["responses"])
        server = stub(_chat_scenario(tmp_path, script))
        flat = {"type": "function", "name": "read",
                "parameters": {"type": "object"}}
        body = {"model": "stub-model", "tools": [flat],
                "input": [{"type": "message", "role": "user",
                           "content": [{"type": "input_text",
                                        "text": "read the file"}]},
                          _output_item(script["call_id"], DENY_TEXT)]}
        _status, text = server.post("/v1/responses", body, False)
        output = json.loads(text)["output"]
        assert [item["type"] for item in output] == ["message"]
        entry = _last_entry(server, "responses")
        assert entry["scripted"]["phase"] == "output-to-text"

    def test_then_shell_with_after_output_text_refuses_to_start(self, tmp_path):
        bad = _chat_scenario(tmp_path, dict(TEXT_SCRIPT, surfaces=["responses"],
                                            then_shell="true"))
        with pytest.raises(AssertionError, match="stub exited"):
            _Stub(tmp_path, bad)


class TestTheDenyScenario:
    """`scripted_opencode_read_deny.yaml`, the T4.9 rows' scenario."""

    def _load(self, path: Path) -> dict:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_it_scripts_one_out_of_workspace_read_on_chat_only(self):
        script = self._load(DENY)["scripted_tool_call"]
        assert script["surfaces"] == ["chat"]
        assert script["name"] == "read"
        assert not script.get("namespace")
        assert script["arguments"] == {"filePath": "/etc/hostname"}
        assert script["after_output"] == "text"
        assert "terminal" not in script

    def test_its_rules_are_the_chain_scenarios_verbatim(self):
        deny, chain = self._load(DENY), self._load(TEXT_ONLY)
        assert deny["prompts"] == chain["prompts"]
        assert deny.get("on_unmatched") == chain.get("on_unmatched") == "fail"
        assert set(deny) - set(chain) == {"scripted_tool_call"}

    def test_a_follow_up_resolves_to_the_same_fixture_as_the_bare_turn(
            self, stub):
        """The refusal text leaves the rule choice alone: the nested-exec
        probe's own fixture answers its follow-up, as it answers the bare turn."""
        server = stub(DENY)
        prompt = ("You are opencode, an interactive CLI tool.\n"
                  "Reply with exactly this token and nothing else: "
                  "MULTI_LLM_NESTED_EXEC_OK")
        bare = {"model": "stub-model",
                "messages": [{"role": "user", "content": prompt}],
                "tools": [CHAT_READ_TOOL]}
        _status, text = server.post("/v1/chat/completions", bare, False)
        call = json.loads(text)["choices"][0]["message"]["tool_calls"][0]
        assert call["id"] == "call_mllm_read_deny"
        follow = dict(bare, messages=[
            *bare["messages"],
            {"role": "assistant", "content": None,
             "tool_calls": [dict(call)]},
            {"role": "tool", "tool_call_id": call["id"], "content": DENY_TEXT}])
        _status, text = server.post("/v1/chat/completions", follow, False)
        answer = json.loads(text)["choices"][0]["message"]["content"]
        entry = _last_entry(server, "chat", count=2)
        assert entry["scripted"]["phase"] == "output-to-text"
        assert entry["fixture"].endswith("nested_exec/probe_not_measured.txt")
        assert answer and not entry["unmatched"]
