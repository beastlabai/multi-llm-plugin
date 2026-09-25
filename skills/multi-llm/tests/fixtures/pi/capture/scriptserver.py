#!/usr/bin/env python3
"""THROWAWAY capture server for the pi json-mode / omp compaction rows.

Chat-completions only. It IMPORTS stub_server (/rig) for the catalog and
failserver's ctx wording so the bytes a CLI sees are the rig's own; it adds only
what neither has a mode for:

  length-full       every turn: a cut answer, finish_reason length, usage.output
                    == the model's maxTokens (16384) -> pi's non-recoverable length
  length-short      every turn: the same cut with usage.output 7 -> pi's
                    "recoverable length" (compact-and-retry); summarization
                    requests (no tools) get a summary
  overflow-compact  tool turns 1..4: a `read` tool call of big/partN.txt; tool
                    turn 5: the failserver `ctx` 400; no-tools requests
                    (summarization): a summary with usage; tool turns after the
                    400: the findings answer

Every request is logged (--log, one JSON line) with has_tools, the tool-message
count, the head of the system prompt and what was served.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/rig")
import stub_server  # noqa: E402

CTX_MESSAGE = ("This model's maximum context length is 8192 tokens. However, your "
               "messages resulted in 214537 tokens. Please reduce the length of the messages.")
FINDINGS = ("[\n  {\"id\": \"S001\", \"title\": \"Bound the retry ladder\", \"severity\": \"high\"},\n"
            "  {\"id\": \"S002\", \"title\": \"Strip the vendor token\", \"severity\": \"medium\"}\n]")
CUT = "[\n  {\"id\": \"S001\", \"title\": \"Bound the retry"
SUMMARY = "## Goal\nReview the plan.\n\n## Progress\nRead four parts of big/.\n"

ARGS = None
LOCK = threading.Lock()
STATE = {"seq": 0, "overflowed": False, "ctx": 0}


def log(entry):
    with LOCK:
        STATE["seq"] += 1
        entry["seq"] = STATE["seq"]
        with open(ARGS.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


def frames(model, *, content=None, tool_call=None, finish="stop", usage=None):
    base = {"id": "chatcmpl-script", "object": "chat.completion.chunk", "created": 0,
            "model": model}
    out = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant"},
                                "finish_reason": None}])]
    if content is not None:
        out.append(dict(base, choices=[{"index": 0, "delta": {"content": content},
                                        "finish_reason": None}]))
    if tool_call is not None:
        out.append(dict(base, choices=[{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": tool_call["id"], "type": "function",
            "function": {"name": tool_call["name"], "arguments": ""}}]},
            "finish_reason": None}]))
        out.append(dict(base, choices=[{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": tool_call["arguments"]}}]},
            "finish_reason": None}]))
    out.append(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}]))
    if usage is not None:
        out.append(dict(base, choices=[], usage=usage))
    return out


def usage(prompt, completion):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def read_tool(body):
    for tool in body.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name") == "read":
            props = ((fn.get("parameters") or {}).get("properties") or {})
            for key in ("path", "file_path", "filePath"):
                if key in props:
                    return key
            return "path"
    return None


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

    def _sse(self, objs):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for obj in objs:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path.endswith("/__health") or path == "/":
            log({"method": "GET", "path": path, "health": True})
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
        path = self.path.split("?")[0]
        model = body.get("model") or "stub-model"
        messages = body.get("messages") or []
        has_tools = bool(body.get("tools"))
        tool_msgs = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        system = next((m.get("content") for m in messages
                       if isinstance(m, dict) and m.get("role") in ("system", "developer")), "")
        if isinstance(system, list):
            system = " ".join(str(p.get("text", "")) for p in system if isinstance(p, dict))
        entry = {"method": "POST", "path": path, "model": model, "has_tools": has_tools,
                 "tool_msgs": tool_msgs, "n_messages": len(messages),
                 "system_head": str(system)[:120], "max_tokens": body.get("max_tokens")
                 or body.get("max_completion_tokens"), "stream": bool(body.get("stream"))}
        if not path.endswith("/chat/completions"):
            entry["served"] = "501"
            log(entry)
            return self._json({"error": {"message": "unhandled"}}, 501)

        mode = ARGS.mode
        if not has_tools:
            entry["served"] = "summary"
            log(entry)
            return self._sse(frames(model, content=SUMMARY, usage=usage(900, 40)))

        if mode == "length-full":
            entry["served"] = "length-full"
            log(entry)
            return self._sse(frames(model, content=CUT, finish="length",
                                    usage=usage(1200, 16384)))
        if mode == "length-short":
            entry["served"] = "length-short"
            log(entry)
            return self._sse(frames(model, content=CUT, finish="length",
                                    usage=usage(1200, 7)))
        if mode.startswith("overflow-compact"):
            want = int(mode.split(":")[1]) if ":" in mode else 1
            if STATE["overflowed"] and STATE["ctx"] < want:
                STATE["ctx"] += 1
                entry["served"] = f"ctx-400#{STATE['ctx']}"
                log(entry)
                return self._json({"error": {"message": CTX_MESSAGE,
                                             "type": "invalid_request_error", "param": None,
                                             "code": "context_length_exceeded"}}, 400)
            if STATE["overflowed"]:
                entry["served"] = "answer"
                log(entry)
                return self._sse(frames(model, content=FINDINGS, usage=usage(3000, 60)))
            if tool_msgs < 4:
                key = read_tool(body) or "path"
                n = tool_msgs + 1
                call = {"id": f"call_read_{n}", "name": "read",
                        "arguments": json.dumps({key: f"big/part{n}.txt"})}
                entry["served"] = f"tool_call:read:part{n}"
                log(entry)
                return self._sse(frames(model, tool_call=call, finish="tool_calls",
                                        usage=usage(2000 + 10000 * tool_msgs, 20)))
            STATE["overflowed"] = True
            STATE["ctx"] = 1
            entry["served"] = "ctx-400#1"
            log(entry)
            return self._json({"error": {"message": CTX_MESSAGE,
                                         "type": "invalid_request_error", "param": None,
                                         "code": "context_length_exceeded"}}, 400)
        entry["served"] = "unknown-mode"
        log(entry)
        return self._json({"error": {"message": "bad mode"}}, 500)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--log", required=True)
    ARGS = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"scriptserver[{ARGS.mode}] on {ARGS.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
