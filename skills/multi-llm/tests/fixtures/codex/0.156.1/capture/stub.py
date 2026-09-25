#!/usr/bin/env python3
"""Scripted Responses stub: step N = number of function_call_output items in the request input."""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = json.load(open(sys.argv[2]))  # list of steps; each step = list of output items; final step gets message
LOG = open(sys.argv[3], "a")

def outputs_in(body):
    n = 0
    for it in body.get("input") or []:
        if isinstance(it, dict) and it.get("type") in ("function_call_output", "custom_tool_call_output", "mcp_tool_call_output"):
            n += 1
    return n

def last_session(body):
    import re
    for it in reversed(body.get("input") or []):
        if isinstance(it, dict) and it.get("type") == "function_call_output":
            m = re.search(r"session ID (\d+)", json.dumps(it.get("output")))
            if m:
                return int(m.group(1))
    return 0

def sse(handler, frames):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()
    for ev, data in frames:
        handler.wfile.write(f"event: {ev}\ndata: {json.dumps(data)}\n\n".encode())
    handler.wfile.flush()

def frames_for(items, model):
    resp = {"id": "resp_stub", "object": "response", "created_at": 0, "status": "completed", "model": model,
            "output": items, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
    inprog = dict(resp, status="in_progress", output=[])
    out = [("response.created", {"type": "response.created", "sequence_number": 0, "response": inprog})]
    seq = 1
    for i, it in enumerate(items):
        if it.get("type") == "web_search_call":
            # Hosted web search, as the live 2026-09-25 capture sent it
            # (../web_search_hosted.sse.json): the added item carries no action
            # yet, then three progress events, then the done item with it.
            added = {k: v for k, v in it.items() if k not in ("action", "results")}
            out.append(("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": i, "item": dict(added, status="in_progress")})); seq += 1
            for phase in ("in_progress", "searching", "completed"):
                ev = "response.web_search_call." + phase
                out.append((ev, {"type": ev, "item_id": it["id"], "output_index": i, "sequence_number": seq})); seq += 1
            out.append(("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": i, "item": it})); seq += 1
            continue
        out.append(("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": i, "item": dict(it, status="in_progress")})); seq += 1
        out.append(("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": i, "item": it})); seq += 1
    out.append(("response.completed", {"type": "response.completed", "sequence_number": seq, "response": resp}))
    return out

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        LOG.write(json.dumps({"GET": self.path}) + "\n"); LOG.flush()
        body = json.dumps({"models": [], "data": []}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        step = outputs_in(body)
        LOG.write(json.dumps({"POST": self.path, "step": step, "tools": [ (t.get("type"), t.get("name")) for t in body.get("tools") or [] if isinstance(t, dict)],
                              "input_tail": (body.get("input") or [])[-2:], "fulltools": body.get("tools")}) + "\n"); LOG.flush()
        script = SCRIPT
        if "SUBAGENT-TASK" in json.dumps(body.get("input") or []) and not any(isinstance(i, dict) and i.get("name") == "spawn_agent" for i in body.get("input") or []):
            script = json.load(open(sys.argv[2] + ".sub"))
        if step < len(script):
            items = json.loads(json.dumps(script[step]).replace("__SESSION_ID__", str(last_session(body))))
        else:
            items = [{"id": "msg_final", "type": "message", "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text", "text": "[]", "annotations": []}]}]
        sse(self, frames_for(items, body.get("model", "stub-model")))

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
