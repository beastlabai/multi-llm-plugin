#!/usr/bin/env python3
import json, sys, os
def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    try:
        m = json.loads(line)
    except Exception:
        continue
    mid = m.get("id"); meth = m.get("method")
    if meth == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": m["params"].get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}}, "serverInfo": {"name": "capfs", "version": "0.0.1"}}})
    elif meth == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{"name": "write_file", "description": "Write a file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}, "required": ["path", "text"]}}]}})
    elif meth == "tools/call":
        a = m["params"].get("arguments") or {}
        with open(a["path"], "w") as f:
            f.write(a["text"])
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": "wrote " + a["path"]}]}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
