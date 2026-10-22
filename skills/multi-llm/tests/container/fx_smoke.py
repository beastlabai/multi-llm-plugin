#!/usr/bin/env python3
"""Zero-credit fx wrapper smoke, sent on stdin to an isolated test container.

Uses the real adapter, PATH wrapper, seeded settings, and repository stub. Fixed
tool frames test protocol plumbing; they do not claim a model chose a tool.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer

sys.path[:0] = ["/opt/mllm/skill", "/opt/mllm/rig"]

import stub_server  # noqa: E402
from utils.providers.fx import FxProvider  # noqa: E402


def main():
    work = Path("/tmp/fx-wrapper-smoke")
    work.mkdir()
    os.chdir(work)
    os.environ.update({
        "MLLM_HOMES_ROOT": "/opt/mllm/rig/home_seed",
        "MLLM_INVOCATION_ROOT": str(work / "invocations"),
        "MLLM_CALL_LOG": str(work / "calls.jsonl"),
        "OPENROUTER_API_KEY": "host-sentinel-must-be-replaced",
    })
    canary = work / "bin/fx-wrapper-path-canary"
    canary.parent.mkdir()
    canary.write_text("#!/bin/sh\nprintf path-ok\n")
    canary.chmod(0o755)
    os.environ["PATH"] = str(canary.parent) + ":" + os.environ["PATH"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), stub_server.Handler)
    os.environ["MLLM_STUB_PORT"] = str(server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = FxProvider()
    results = []
    tools = [
        ("happy", None, None),
        ("read", "read_file", {"path": str(work / "read.txt")}),
        ("write", "write_file", {"path": str(work / "write.txt"), "content": "write-ok"}),
        ("shell", "shell", {"request": {"action": "run", "command":
            "printf fx-shell-; fx-wrapper-path-canary"}}),
    ]
    (work / "read.txt").write_text("read-ok")
    try:
        for label, name, arguments in tools:
            stub_server._log_path = work / f"{label}.stub.jsonl"
            stub_server._tool_script = None
            if name:
                stub_server._tool_script = stub_server.ScriptedToolCall({
                    "name": name, "arguments": arguments, "surfaces": ["chat"],
                    "terminal": '["ok"]',
                    **({"fx_permission_decision": "clear"} if label == "shell" else {}),
                })
            # A strict text fixture for the no-tool control; scripted tool
            # requests are owned by their advertised declaration and call id.
            stub_server._resolver = lambda *_: '["ok"]'
            with provider.hermetic_dispatch("generate", "stub-model"):
                env = dict(os.environ)
                env.update(provider.resolve_env("stub-model", base_env=env))
                for key in provider.get_remove_env(base_env=env):
                    env.pop(key, None)
                home = Path(env["HOME"])
                argv, stdin = provider.build_command_for_transport(
                    "Return the fixed probe result.", "stub-model", "stdin")
                process = subprocess.run(argv, input=stdin, text=True,
                                         capture_output=True, env=env, timeout=12)
                assert process.returncode == 0, (label, process.stdout, process.stderr)
                envelope = json.loads(process.stdout)
                assert provider.parse_output(process.stdout, process.stderr)["success"], envelope
                assert envelope["final_output"] == '["ok"]', envelope
                calls = [json.loads(line) for line in stub_server._log_path.read_text().splitlines()]
                reviews = [c for c in calls if c.get("fixture") == "fx_permission_review"]
                assert len(reviews) == (1 if label == "shell" else 0), (label, calls)
                assert envelope["usage"] == {
                    "input_tokens": 2 if name else 1, "output_tokens": 2 if name else 1,
                }, envelope
                assert all(c["raw_path"].startswith("/s/") for c in calls), calls
                assert all(not c.get("unmatched") and not c.get("unhandled") for c in calls)
                advertised = set(calls[0]["tool_names"])
                assert not advertised & {"skill", "subagent", "mcp_select_tool", "mcp_features"}
                assert "web_fetch" in advertised  # allowed policy; this smoke has no egress
                if name:
                    [tool] = envelope["tool_calls"]
                    assert tool["name"] == name and tool.get("status") == "success", envelope
                if label == "shell":
                    assert "fx-shell-path-ok" in process.stderr, process.stderr
                results.append({"case": label, "requests": len(calls), "reviewers": len(reviews),
                                "usage": envelope["usage"], "exit": process.returncode})
            assert not home.exists(), home
        assert (work / "write.txt").read_text() == "write-ok"
        assert not list((work / "invocations").glob("fx-*"))
        print(json.dumps({"ok": True, "results": results}, sort_keys=True))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
