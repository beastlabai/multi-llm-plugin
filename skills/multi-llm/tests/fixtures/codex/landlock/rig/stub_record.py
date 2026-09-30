#!/usr/bin/env python3
"""Scripted Responses stub that also records every request body.

Origin (2026-09-30): a copy of plan codex-sandbox-unavailable-findings-and-fixes
step 1c's rig stub (`scratch/codex-sandbox-step1c-2026-09-30/rig/stub_record.py`),
itself a copy of the committed capture stub
`tests/fixtures/codex/0.156.1/capture/stub.py`. The committed stub logs only the
last two input items of each request, and tier 3 (`tests/test_codex_jail_rig.py`)
must read the WHOLE first request: the `-c developer_instructions=…` developer
message and the advertised tool list. So this copy:

* saves every request body as `<log>.body<N>.json` (N = 1-based request index,
  so `.body1.json` is the first request codex sent) and keeps the rig's
  `.body0.json` (the last request that carried no tool output);
* counts `tool_search_output` items as step outputs, as the rig did;
* accepts port `0`: it binds a free port and prints it on stdout as the first
  line (`port=<n>`), so the test never races a probe-then-bind port pick;
* drops the committed stub's hosted-web-search frames and its `.sub`
  sub-agent script, which no jail turn uses.

Usage: stub_record.py PORT SCRIPT.json LOG
Step N of SCRIPT = the list of output items served when the request carries N
tool outputs; past the end, a final assistant message "[]" ends the turn.
Zero spend: nothing here talks to a real model.
"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = json.load(open(sys.argv[2]))  # list of steps; each step = list of output items; final step gets message
LOG = open(sys.argv[3], "a")

def outputs_in(body):
    n = 0
    for it in body.get("input") or []:
        if isinstance(it, dict) and it.get("type") in ("function_call_output", "custom_tool_call_output", "mcp_tool_call_output", "tool_search_output"):
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
        out.append(("response.output_item.added", {"type": "response.output_item.added", "sequence_number": seq, "output_index": i, "item": dict(it, status="in_progress")})); seq += 1
        out.append(("response.output_item.done", {"type": "response.output_item.done", "sequence_number": seq, "output_index": i, "item": it})); seq += 1
    out.append(("response.completed", {"type": "response.completed", "sequence_number": seq, "response": resp}))
    return out

NREQ = 0
NREQ_LOCK = threading.Lock()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        LOG.write(json.dumps({"GET": self.path}) + "\n"); LOG.flush()
        body = json.dumps({"models": [], "data": []}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        global NREQ
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        step = outputs_in(body)
        with NREQ_LOCK:
            NREQ += 1
            n = NREQ
        if step == 0:
            open(sys.argv[3] + ".body0.json", "wb").write(raw)
        open(sys.argv[3] + f".body{n}.json", "wb").write(raw)
        LOG.write(json.dumps({"POST": self.path, "n": n, "step": step, "tools": [(t.get("type"), t.get("name")) for t in body.get("tools") or [] if isinstance(t, dict)],
                              "input_tail": (body.get("input") or [])[-2:]}) + "\n"); LOG.flush()
        if step < len(SCRIPT):
            items = json.loads(json.dumps(SCRIPT[step]).replace("__SESSION_ID__", str(last_session(body))))
        else:
            items = [{"id": "msg_final", "type": "message", "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text", "text": "[]", "annotations": []}]}]
        sse(self, frames_for(items, body.get("model", "stub-model")))

SERVER = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H)
print(f"port={SERVER.server_address[1]}", flush=True)
SERVER.serve_forever()
