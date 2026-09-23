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
  to a request that advertises the tool — the text path is unchanged.
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
        assert carriers == [SCRIPTED.name], (
            f"{carriers}: the scripted shape is reached only through ONE named "
            f"scenario, so no existing row can start seeing tool calls")


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
