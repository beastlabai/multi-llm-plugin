"""Capture-only chat-completions server for hermes's tool-turn rows.

``--mode multi``: the first turn answers a decoy text AND one tool call, the
turn carrying the tool's result gets the findings (two assistant messages, the
first a decoy document). ``write`` / ``terminal`` force a ``write_file`` or
``terminal`` call the toolless lane must refuse; ``loop`` calls a tool on every
turn, so the run ends on ``--max-turns``. Catalog GETs answer one model with a
256k context. Every request is appended to ``--log`` as one JSON line.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DECOY = 'FIRST ANSWER (must NOT be selected)\n[\n  {"id": "D999", "title": "Decoy", "severity": "low"}\n]'
FINDINGS = ('[\n  {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},\n'
            '  {"id": "S002", "title": "Strip the vendor token", "severity": "medium"}\n]')
CALLS = {
    "multi": ("read_file", {"path": "plans/auth-feature.md"}),
    "write": ("write_file", {"path": "/tmp/work/hermes-write-canary.txt", "content": "unexpected write"}),
    "terminal": ("terminal", {"command": "touch /tmp/work/hermes-terminal-canary"}),
    "loop": ("read_file", {"path": "plans/auth-feature.md"}),
}
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--mode", required=True, choices=sorted(CALLS))
ARGS = ap.parse_args()
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _log(self, body):
        with LOCK, open(ARGS.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"method": self.command, "path": self.path, "body": body}, sort_keys=True) + "\n")

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._log(None)
        self._json({"data": [{"id": "stub-model", "object": "model", "context_length": 256000}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        self._log(body)
        if "chat/completions" not in self.path:
            self._json({"error": "not found"}, 404)
            return
        answered = any(m.get("role") == "tool" for m in body.get("messages", []))
        name, arguments = CALLS[ARGS.mode]
        if answered and ARGS.mode != "loop":
            content, calls = (FINDINGS if ARGS.mode == "multi" else "[]"), None
        else:
            content = DECOY if ARGS.mode == "multi" else "Calling a tool.\n"
            calls = [{"index": 0, "id": f"call_{len(body.get('messages', []))}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments)}}]
        delta = {"role": "assistant", "content": content}
        if calls:
            delta["tool_calls"] = calls
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "chatcmpl-hermes", "object": "chat.completion.chunk", "model": body.get("model")}
        for chunk in ({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                      {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}):
            self.wfile.write(("data: " + json.dumps({**base, **chunk}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
