"""Credential-free captures of the production junie adapter against the real binary.

Run inside the pinned image (``--network none``, a non-root uid), with this skill
mounted at ``$SKILL`` and the output directory as the only argument::

    python capture_probe.py /work/out

Each row builds its argv and environment through ``JunieProvider`` exactly as a
dispatch does (``build_command_for_transport(..., "stdin")``, ``resolve_env``,
``build_child_env``); the only change is the profile's ``baseUrl``, pointed at a
server on 127.0.0.1. Success-shaped rows use the scripted chat server below,
which answers the way a model drives Junie: through its ``submit`` tool. Failure
rows use ``tests/container/failserver.py``. Every row writes
``<row>.stdout.txt``, ``<row>.stderr.txt`` and a summary line in
``captures.json`` (exit code, wall time, request count, what the server saw).
"""
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

SKILL = Path(os.environ.get("SKILL", "/skill"))
sys.path.insert(0, str(SKILL))
from utils.providers import junie  # noqa: E402
from utils.providers.base import build_child_env  # noqa: E402

FINDINGS = (
    '[\n  {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},\n'
    '  {"id": "S002", "title": "Strip the vendor token", "severity": "medium"}\n]'
)
DECOY = 'FIRST ANSWER (must NOT be selected)\n[{"id": "D999", "title": "decoy"}]'
MODEL = "openai/gpt-6-luna"
PROMPT = (Path(__file__).resolve().parent.parent / "row_capture" / "prompt.txt").read_text(encoding="utf-8")
WORKSPACE = Path("/tmp/junie-work")


def _sse(chunks):
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _completion(body, content=None, call=None):
    """A chat completion: text, or one tool call. Streamed when the request asks."""
    model = body.get("model", MODEL)
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    message = {"role": "assistant", "content": content}
    finish = "stop"
    if call is not None:
        name, args = call
        message["tool_calls"] = [{
            "index": 0, "id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
        finish = "tool_calls"
    if not body.get("stream"):
        return "application/json", json.dumps({
            "id": "c1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage,
        })
    delta = {"role": "assistant", "content": content}
    if call is not None:
        delta["tool_calls"] = message["tool_calls"]
    return "text/event-stream", _sse([
        {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
         "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
    ])


class Script(http.server.BaseHTTPRequestHandler):
    """Junie's chat surface, scripted per row. Tool-less requests are Junie's own
    helpers (the task title, the closing summary) and get a short text."""

    protocol_version = "HTTP/1.1"
    row = ""
    log = []

    def log_message(self, *_):
        pass

    def do_GET(self):  # noqa: N802
        self.log.append({"method": "GET", "path": self.path})
        payload = b'{"object":"list","data":[]}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        tools = [(t.get("function") or t).get("name") for t in body.get("tools") or []]
        tool_results = sum(1 for m in body.get("messages", []) if m.get("role") == "tool")
        self.log.append({"method": "POST", "path": self.path, "tools": tools,
                         "tool_results": tool_results, "stream": body.get("stream"),
                         "reasoning_effort": body.get("reasoning_effort")})
        row = self.row
        if not tools:
            kind, text = _completion(body, content="Review the plan")
        elif row == "ephemeral-only":
            time.sleep(60)
            return
        elif row == "stdout-error":
            kind, text = _completion(body, content=FINDINGS)
        elif row == "exit-zero-failure":
            kind, text = _completion(body, call=("grep_search", {"path": ".", "pattern": f"needle{len(self.log)}"}))
        elif row == "multi-assistant" and tool_results == 0:
            kind, text = _completion(body, content=DECOY, call=("open", {"path": "README.md"}))
        elif row == "empty":
            kind, text = _completion(body, call=("submit", {"solution_summary": ""}))
        else:
            kind, text = _completion(body, call=("submit", {"solution_summary": FINDINGS}))
        payload = text.encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _script_server(row, port):
    handler = type(f"Script_{row}", (Script,), {"row": row, "log": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler.log


def _failserver(mode, port, log):
    proc = subprocess.Popen(
        [sys.executable, str(SKILL / "tests" / "container" / "failserver.py"),
         "--port", str(port), "--mode", mode, "--log", str(log)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            import socket
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"failserver {mode} did not start on {port}")


def run(row, port, *, mode=None, deadline=420.0, term_after=None, drop_key=False):
    original = junie.render_profile
    log_path = Path(f"/tmp/junie-{row}.server.jsonl")
    server = proc = None
    if mode:
        proc = _failserver(mode, port, log_path)
    else:
        server, log = _script_server(row, port)
    junie.render_profile = lambda model, effort=None: original(
        model, effort, f"http://127.0.0.1:{port}/v1/chat/completions")
    provider = junie.JunieProvider()
    workdir = WORKSPACE / row
    workdir.mkdir(parents=True)
    (workdir / "README.md").write_text("fixture contents\n", encoding="utf-8")
    try:
        env = build_child_env(provider, provider_env=provider.resolve_env(MODEL))
    finally:
        junie.render_profile = original
    if drop_key:
        env.pop(junie.CREDENTIAL_ENV, None)
    root = Path(env["JUNIE_MODEL_LOCATIONS"]).parent
    argv, stdin = provider.build_command_for_transport(PROMPT, MODEL, "stdin")
    start = time.monotonic()
    child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env, cwd=workdir)
    killed = False
    try:
        if term_after is not None:
            try:
                out, err = child.communicate(stdin, timeout=term_after)
            except subprocess.TimeoutExpired:
                child.send_signal(signal.SIGTERM)
                out, err = child.communicate(timeout=30)
        else:
            out, err = child.communicate(stdin, timeout=deadline)
    except subprocess.TimeoutExpired:
        killed = True
        child.kill()
        out, err = child.communicate()
    wall = round(time.monotonic() - start, 3)
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=10)
        log = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    if server is not None:
        server.shutdown()
    leaked = sorted(str(p.relative_to(workdir)) for p in workdir.rglob("*") if p.name != "README.md")
    return {
        "row": row, "server": mode or "script", "exit_code": child.returncode, "wall_s": wall,
        "killed": killed, "requests": sum(1 for r in log if r.get("method") == "POST"),
        "log": log, "stdout": out, "stderr": err, "root": str(root),
        "workspace_writes": leaked, "parsed": provider.parse_output(out, err),
        "usage": provider.parse_usage(out, err), "argv": argv,
    }


ROWS = [
    ("success", {}),
    ("multi-assistant", {}),
    ("empty", {}),
    ("stdout-error", {}),
    ("exit-zero-failure", {}),
    ("ephemeral-only", {"term_after": 8}),
    ("missing-key", {"drop_key": True}),
    ("truncated", {"mode": "trunc"}),
    ("error-envelope", {"mode": "errbody"}),
    ("auth-error", {"mode": "401"}),
    ("invalid-model", {"mode": "404"}),
    ("context-overflow", {"mode": "ctx"}),
    ("zero-body", {"mode": "empty"}),
    ("server-error", {"mode": "500"}),
    ("rate-limit", {"mode": "429"}),
]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    results = []
    threads = []

    def one(index, row, opts):
        result = run(row, 18900 + index, **opts)
        (out / f"{row}.stdout.txt").write_text(result["stdout"], encoding="utf-8")
        (out / f"{row}.stderr.txt").write_text(result["stderr"], encoding="utf-8")
        results.append({k: v for k, v in result.items() if k not in ("stdout", "stderr")})

    for index, (row, opts) in enumerate(ROWS):
        if only and row not in only:
            continue
        thread = threading.Thread(target=one, args=(index, row, opts))
        thread.start()
        threads.append(thread)
        time.sleep(0.5)
    for thread in threads:
        thread.join()
    results.sort(key=lambda r: [name for name, _ in ROWS].index(r["row"]))
    (out / "captures.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    for r in results:
        print(r["row"], r["exit_code"], r["wall_s"], r["requests"], r["killed"],
              r["parsed"].get("success"), str(r["parsed"].get("error"))[:120], r["workspace_writes"])


if __name__ == "__main__":
    main()
