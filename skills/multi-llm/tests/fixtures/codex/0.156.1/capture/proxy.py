#!/usr/bin/env python3
"""Logging passthrough: localhost -> https://chatgpt.com/backend-api/codex (§6.2 live capture, 2026-09-25).

usage: proxy.py PORT LOGFILE, then point codex at it with its own ChatGPT login:
  codex exec --json -m gpt-5.5 -c model_provider=cap -c model_providers.cap.name=cap \
    -c model_providers.cap.base_url=http://127.0.0.1:PORT -c model_providers.cap.wire_api=responses \
    -c model_providers.cap.requires_openai_auth=true '<prompt>'
Logs response bodies (the raw SSE) and header NAMES only (never auth values);
each POST body is saved next to LOGFILE as LOGFILE.req.<ms>.json. Keep both out
of the repo: they hold the account's instructions and ids. Code-mode models
(gpt-6-astra) get no web tool through a custom provider, so web__run cannot be
captured this way.
"""
import http.client, json, sys, ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
UP_HOST = "chatgpt.com"; UP_PREFIX = "/backend-api/codex"
LOG = open(sys.argv[2], "ab")
SKIP = {"host", "accept-encoding", "connection", "content-length"}
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _fwd(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in SKIP}
        hdrs["Accept-Encoding"] = "identity"
        if body is not None: hdrs["Content-Length"] = str(len(body))
        LOG.write(("\n=== REQ %s %s hdrs=%s ce=%s len=%d\n" % (method, self.path,
            sorted(k for k in hdrs), self.headers.get("Content-Encoding"), len(body or b""))).encode()); LOG.flush()
        if body and method == "POST":
            import os, time
            fn = sys.argv[2] + ".req.%d.json" % int(time.time()*1000)
            open(fn, "wb").write(body)
        c = http.client.HTTPSConnection(UP_HOST, context=ssl.create_default_context(), timeout=600)
        c.request(method, UP_PREFIX + self.path, body=body, headers=hdrs)
        r = c.getresponse()
        LOG.write(("=== RESP %d %s\n" % (r.status, r.getheader("Content-Type"))).encode()); LOG.flush()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length", "content-encoding"): continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk: break
            LOG.write(chunk); LOG.flush()
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n"); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
    def do_POST(self): self._fwd("POST")
    def do_GET(self): self._fwd("GET")
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
