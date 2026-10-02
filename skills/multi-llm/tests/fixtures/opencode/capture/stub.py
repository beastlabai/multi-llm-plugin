#!/usr/bin/env python3
"""Scripted chat-completions stub for the opencode capture rig (plan §4, T4.6).

usage: stub.py PORT SPEC.json LOG.jsonl

SPEC keys:
  tool       opencode tool name the first turn calls ("read" or "bash")
  args       that tool's arguments
  org_config optional; served as {"config": ...} on GET /api/config, the
             account org-config endpoint opencode fetches for an active org
  turns      optional; a scripted multi-step turn that replaces the one-call
             logic below. A tool-bearing request is answered with
             ``turns[N]``, N being the assistant messages it already carries
             (one per finished step). Each entry may hold ``text`` (streamed
             before the calls), ``calls`` (``[{"tool", "args"}]``) and
             ``usage`` (``{"input", "output", "cached"}``). Past the end of the
             list a tool-bearing request gets ``final`` (with ``final_usage``).
  final      optional; the closing text (default ``FINAL ANSWER``)

Turn logic (the same one the investigation used):
  * a request that advertises tools and carries no ``role: "tool"`` message
    gets one streamed ``tool_calls`` delta (finish ``tool_calls``);
  * every other request (the follow-up after a tool result, or the title
    request, which carries no tools) gets the text ``FINAL ANSWER`` (finish
    ``stop``).
So a turn that reaches a second tool-bearing request is a turn whose tool
result went back to the model. Every POST is logged to LOG with the tool
results it carried, which is how "the deny returned to the model" is read.
"""
import itertools
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
SPEC = json.load(open(sys.argv[2]))
LOG = sys.argv[3]
_n = itertools.count()


def _log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/config"):
            _log({"GET": self.path, "x-org-id": self.headers.get("x-org-id"),
                  "bearer": bool(self.headers.get("authorization"))})
            if "org_config" in SPEC:
                return self._json({"config": SPEC["org_config"]})
            return self._json({}, 404)
        _log({"GET": self.path})
        self._json({"data": [{"id": "m", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        has_tools = bool(body.get("tools"))
        msgs = body.get("messages", [])
        tool_results = [m for m in msgs if m.get("role") == "tool"]
        i = next(_n)
        _log({"i": i, "path": self.path, "tools": has_tools, "nmsgs": len(msgs),
              "tool_results": [str(m.get("content")) for m in tool_results]})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def ev(d):
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            self.wfile.flush()

        base = {"id": "c%d" % i, "object": "chat.completion.chunk", "created": 1, "model": "m"}
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        final = SPEC.get("final", "FINAL ANSWER")
        if "turns" in SPEC:
            step = sum(1 for m in msgs if m.get("role") == "assistant")
            if not has_tools:  # the session-title request
                turn = {"text": final}
            elif step < len(SPEC["turns"]):
                turn = SPEC["turns"][step]
            else:
                turn = {"text": final, "usage": SPEC.get("final_usage")}
            u = turn.get("usage")
            if u:
                usage = {"prompt_tokens": u["input"] + u.get("cached", 0), "completion_tokens": u["output"],
                         "total_tokens": u["input"] + u.get("cached", 0) + u["output"],
                         "prompt_tokens_details": {"cached_tokens": u.get("cached", 0)}}
            if turn.get("text"):
                ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": turn["text"]},
                                         "finish_reason": None}]})
            calls = [{"index": k, "id": "call_%d_%d" % (step, k), "type": "function",
                      "function": {"name": c["tool"], "arguments": json.dumps(c["args"])}}
                     for k, c in enumerate(turn.get("calls") or [])]
            if calls:
                ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": calls},
                                         "finish_reason": None}]})
            ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": usage})
        elif has_tools and not tool_results:
            call = {"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": SPEC["tool"], "arguments": json.dumps(SPEC["args"])}}
            ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [call]}, "finish_reason": None}]})
            ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": usage})
        else:
            ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": final}, "finish_reason": None}]})
            ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
