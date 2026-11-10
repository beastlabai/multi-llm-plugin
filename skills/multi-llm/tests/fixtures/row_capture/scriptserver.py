#!/usr/bin/env python3
"""Capture-only server for the ``multi-assistant`` row (README.md here).

The committed stub's ``scripted_tool_call`` answers the tool turn with a bare
call and no text, so the run it drives has ONE assistant message with text,
which says nothing about last-wins. This server answers that turn with a decoy
text AND the call, then answers the request carrying the tool's result with the
real findings: two assistant messages with text, the first a decoy document.
It imports stub_server (mounted at /rig) for the catalog and the Responses
rendering, so every other byte a CLI sees is the rig's own.

  request with tools, no tool result in it  -> DECOY text + one tool call
  request carrying the tool result          -> FINDINGS
  request without tools (a title call)      -> "Plan review"

Chat-completions (streamed) and Responses (streamed, codex) only. Every request
is appended to ``--log`` as one JSON line.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, "/rig")
import stub_server  # noqa: E402

DECOY = ("FIRST ANSWER (must NOT be selected)\n"
         "[\n  {\"id\": \"D999\", \"title\": \"Decoy\", \"severity\": \"low\"}\n]")
FINDINGS = ("[\n  {\"id\": \"S001\", \"title\": \"Bound the retry ladder\", \"severity\": \"high\"},\n"
            "  {\"id\": \"S002\", \"title\": \"Strip the vendor token\", \"severity\": \"medium\"}\n]")
TITLE = "Plan review"
CALL_ID = "call_rowcap_1"

ARGS = None
LOCK = threading.Lock()


def log(entry):
    with LOCK, open(ARGS.log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def chat_frames(model, content, tool=None):
    base = {"id": "chatcmpl-rowcap", "object": "chat.completion.chunk", "created": 0,
            "model": model}
    out = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant"},
                                "finish_reason": None}]),
           dict(base, choices=[{"index": 0, "delta": {"content": content},
                                "finish_reason": None}])]
    if tool:
        out.append(dict(base, choices=[{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": CALL_ID, "type": "function",
            "function": {"name": tool, "arguments": ARGS.args}}]},
            "finish_reason": None}]))
    out.append(dict(base, choices=[{"index": 0, "delta": {},
                                    "finish_reason": "tool_calls" if tool else "stop"}],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}))
    return [(None, frame) for frame in out]


def responses_frames(model, content, tool):
    """A message item carrying ``content``, then one function_call item."""
    message = {"id": "msg_rowcap", "type": "message", "role": "assistant",
               "status": "completed",
               "content": [{"type": "output_text", "text": content, "annotations": []}]}
    call = {"id": "fc_rowcap", "type": "function_call", "status": "completed",
            "call_id": CALL_ID, "name": tool, "arguments": ARGS.args}
    done = {"id": "resp_rowcap", "object": "response", "created_at": 0,
            "status": "completed", "model": model, "output": [message, call],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
    started = dict(done, status="in_progress", output=[])
    events = [
        ("response.created", {"type": "response.created", "response": started}),
        ("response.output_item.added", {"type": "response.output_item.added",
         "output_index": 0, "item": dict(message, status="in_progress", content=[])}),
        ("response.content_part.added", {"type": "response.content_part.added",
         "item_id": "msg_rowcap", "output_index": 0, "content_index": 0,
         "part": {"type": "output_text", "text": "", "annotations": []}}),
        ("response.output_text.delta", {"type": "response.output_text.delta",
         "item_id": "msg_rowcap", "output_index": 0, "content_index": 0,
         "delta": content}),
        ("response.output_text.done", {"type": "response.output_text.done",
         "item_id": "msg_rowcap", "output_index": 0, "content_index": 0,
         "text": content}),
        ("response.content_part.done", {"type": "response.content_part.done",
         "item_id": "msg_rowcap", "output_index": 0, "content_index": 0,
         "part": message["content"][0]}),
        ("response.output_item.done", {"type": "response.output_item.done",
         "output_index": 0, "item": message}),
        ("response.output_item.added", {"type": "response.output_item.added",
         "output_index": 1, "item": dict(call, status="in_progress", arguments="")}),
        ("response.function_call_arguments.done", {
         "type": "response.function_call_arguments.done", "item_id": "fc_rowcap",
         "output_index": 1, "arguments": ARGS.args}),
        ("response.output_item.done", {"type": "response.output_item.done",
         "output_index": 1, "item": call}),
        ("response.completed", {"type": "response.completed", "response": done}),
    ]
    return [(name, dict(event, sequence_number=n)) for n, (name, event) in enumerate(events)]


def carries_result(body):
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "tool":
            return True
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _sse(self, frames, done_marker):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for name, frame in frames:
            head = f"event: {name}\n" if name else ""
            self.wfile.write(f"{head}data: {json.dumps(frame)}\n\n".encode())
            self.wfile.flush()
        if done_marker:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def do_GET(self):  # noqa: N802
        path = stub_server.strip_session(self.path.split("?")[0])
        if path.endswith("/__health") or path == "/":
            return self._json({"ok": True})
        log({"method": "GET", "path": path, "served": "catalog"})
        return self._json(stub_server.catalog("openai"))

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        path = stub_server.strip_session(self.path.split("?")[0])
        model = body.get("model") or "stub-model"
        names = stub_server._tool_names(body)
        entry = {"method": "POST", "path": path, "model": model,
                 "has_tools": bool(body.get("tools")), "tool_names": names}
        responses = path.endswith("/responses")
        if not (responses or path.endswith("/chat/completions")):
            entry["served"] = "501"
            log(entry)
            return self._json({"error": {"message": "unhandled"}}, 501)
        if not body.get("tools"):
            entry["served"] = "title"
            answer, tool = TITLE, None
        elif carries_result(body):
            entry["served"] = "findings"
            answer, tool = FINDINGS, None
        elif ARGS.tool in names:
            entry["served"] = f"decoy+{ARGS.tool}"
            answer, tool = DECOY, ARGS.tool
        else:
            entry["served"] = "tool-not-advertised"
            answer, tool = FINDINGS, None
        log(entry)
        if responses:
            frames = (responses_frames(model, answer, tool) if tool
                      else list(stub_server.responses_stream(model, answer)))
            return self._sse(frames, done_marker=False)
        return self._sse(chat_frames(model, answer, tool), done_marker=True)


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--tool", required=True, help="the tool the decoy turn calls")
    parser.add_argument("--args", required=True, help="its arguments, a JSON string")
    ARGS = parser.parse_args()
    stub_server.refuse_probe_port(ARGS.port, "row_capture scriptserver")
    server = stub_server.LoopbackHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"row_capture scriptserver listening on 127.0.0.1:{ARGS.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
