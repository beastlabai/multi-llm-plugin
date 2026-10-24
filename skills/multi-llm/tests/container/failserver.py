#!/usr/bin/env python3
"""Error injection over the stub's route surface, for §6.2's failure fixtures.

**Deliberately NOT the committed ``stub_server.py``**, which has no
error-injection mode and must not grow one at a probe: that server is what every
green container row resolves its answers through, and a mode switch able to turn
it into a 500 machine is one typo away from a lane that passes because nothing
was ever asked. This file is the second binary instead, committed so the next
capture session does not begin by rewriting it from a scratch directory again
(the previous copy lived in an untracked ``scratch/`` and was lost).

**It mirrors ``stub_server.py``'s routes rather than re-deriving them**, and
imports that module for the catalog bodies so the two cannot drift. Only the
*completion* surfaces fail; ``/__health`` and every catalog GET answer exactly as
they do under the stub. That asymmetry is the whole point: a CLI that 501s on its
own catalog GET never reaches the completion where the error is being injected,
so the capture would record a routing accident instead of the vendor failure the
fixture claims to pin.

Modes (``--mode``):

======== ==================================================================
500      HTTP 500, vendor server-error body
429      HTTP 429, no ``Retry-After``
429ra    HTTP 429 with ``Retry-After: <--retry-after>`` (default 1)
401      HTTP 401, invalid-credential body
404      HTTP 404, model-not-found body **naming the requested model**
ctx      HTTP 400, context-length-exceeded body
empty    accept the request and close writing **zero bytes**
stall    accept the request and never respond (``--stall-s``, default 3600)
trunc    200 headers, then a PREFIX of the requesting surface's own success
         body, then close. Streamed clients get a cut SSE stream, non-streamed
         ones a Content-Length promising bytes that never arrive.
errbody  HTTP **200** whose body is an error object — a provider error
         delivered on the success channel (§6.2's ``error-envelope``)
======== ==================================================================

``ok`` is *not* a mode. A success capture is ``stub_server.py``'s job.

Every request is appended to ``--log`` as one JSON line before the response is
written, so a fail-forever session doubles as §6.1.1's request-count measurement
(the count is the evidence; "the docs say 3" is not admissible).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stub_server import (  # noqa: E402
    LoopbackHTTPServer,
    catalog,
    refuse_probe_port,
    chat_body,
    chat_stream,
    extract_prompt,
    messages_body,
    messages_stream,
    responses_body,
    responses_stream,
    strip_session,
)

#: status + the vendor wording each surface renders. The OpenAI messages are the
#: upstream strings verbatim, because the parsers under test match on them.
MODES = {
    "500": (500, "The server had an error while processing your request. Sorry about that!",
            "server_error", None, "api_error", "INTERNAL"),
    "429": (429, "Rate limit reached for stub-model in organization org-stub on "
                 "requests per min (RPM): Limit 3, Used 3. Please try again in 20s.",
            "rate_limit_exceeded", "rate_limit_exceeded", "rate_limit_error", "RESOURCE_EXHAUSTED"),
    "429ra": (429, "Rate limit reached for stub-model in organization org-stub on "
                   "requests per min (RPM): Limit 3, Used 3. Please try again in 20s.",
              "rate_limit_exceeded", "rate_limit_exceeded", "rate_limit_error", "RESOURCE_EXHAUSTED"),
    "401": (401, "Incorrect API key provided: sk-stub***. You can find your API key at "
                 "https://platform.openai.com/account/api-keys.",
            "invalid_request_error", "invalid_api_key", "authentication_error", "UNAUTHENTICATED"),
    "404": (404, "The model `{model}` does not exist or you do not have access to it.",
            "invalid_request_error", "model_not_found", "not_found_error", "NOT_FOUND"),
    "ctx": (400, "This model's maximum context length is 8192 tokens. However, your "
                 "messages resulted in 214537 tokens. Please reduce the length of the messages.",
            "invalid_request_error", "context_length_exceeded", "invalid_request_error",
            "INVALID_ARGUMENT"),
    "errbody": (200, "Incorrect API key provided: sk-stub***. You can find your API key at "
                     "https://platform.openai.com/account/api-keys.",
                "invalid_request_error", "invalid_api_key", "authentication_error",
                "UNAUTHENTICATED"),
    "empty": (0, "", "", None, "", ""),
    "stall": (0, "", "", None, "", ""),
    "trunc": (200, "", "", None, "", ""),
}

_lock = threading.Lock()
_seq = 0
ARGS = None


def _record(entry: dict) -> None:
    global _seq
    with _lock:
        _seq += 1
        entry["seq"] = _seq
    if not ARGS.log:
        return
    with open(ARGS.log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()


def error_body(surface: str, model: str) -> dict:
    status, message, oa_type, oa_code, an_type, g_status = MODES[ARGS.mode]
    message = message.format(model=model)
    if surface == "anthropic":
        return {"type": "error", "error": {"type": an_type, "message": message}}
    if surface in ("gemini", "gemini-stream"):
        return {"error": {"code": status, "message": message, "status": g_status}}
    return {"error": {"message": message, "type": oa_type, "param": None, "code": oa_code}}


def gemini_body(answer: str) -> dict:
    """The one success document ``stub_server`` renders inline, not in a helper.

    Every other surface has an importable renderer (``chat_body``,
    ``responses_body``, ``messages_body``); the gemini shape is built in place
    inside ``stub_server.Handler._route``, so there is nothing to import for it.
    It is restated here rather than refactored out of that module, because
    ``stub_server.py`` is the server every green container row resolves its
    answers through and a probe does not edit it (this file's opening
    paragraph). The duplication is three keys wide and is the reason `trunc`
    below asserts nothing about the gemini body's contents — a drift shows up as
    a capture that no longer matches the committed one, which is what the
    fixtures are for.
    """
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": answer}]},
                        "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1,
                          "totalTokenCount": 2},
    }


def success_body(surface: str, model: str, answer: str) -> dict:
    """The non-streamed success document the REQUESTING surface would have got."""
    if surface == "chat":
        return chat_body(model, answer)
    if surface == "responses":
        return responses_body(model, answer)
    if surface == "anthropic":
        return messages_body(model, answer)
    return gemini_body(answer)


def success_frames(surface: str, model: str, answer: str) -> tuple[list, bool]:
    """The streamed frame sequence, plus whether this surface takes ``[DONE]``.

    ``data: [DONE]`` is an OpenAI-ism: gemini-cli ``JSON.parse``s every frame
    and dies on it, and the named-event surfaces terminate with their own final
    event. Same rule as ``stub_server._send_sse``, and for the same reason.
    """
    if surface == "chat":
        return list(chat_stream(model, answer)), True
    if surface == "responses":
        return list(responses_stream(model, answer)), False
    if surface == "anthropic":
        return list(messages_stream(model, answer)), False
    return [(None, gemini_body(answer))], False


def sse_payload(frames: list, done_sentinel: bool) -> bytes:
    """``stub_server._send_sse``'s framing, rendered to bytes instead of written.

    `trunc` needs the whole stream up front so it can cut it at a byte offset,
    which is the one thing a frame-at-a-time writer cannot do.
    """
    out = []
    emitted_named = False
    for name, obj in frames:
        out.append(((f"event: {name}\n" if name else "")
                    + f"data: {json.dumps(obj)}\n\n").encode("utf-8"))
        if name:
            emitted_named = True
    if done_sentinel and not emitted_named:
        out.append(b"data: [DONE]\n\n")
    return b"".join(out)


#: `/v1beta/models/<model>:generateContent`. The gemini surface is the one that
#: carries the model in the PATH and never in the body, so `body["model"]` is
#: absent for every gemini request — and the 404 body, whose whole job is to NAME
#: the model, would otherwise name the `stub-model` default instead of the model
#: the CLI actually asked for. Measured: gemini-cli surfaces the vendor sentence
#: verbatim, so an unnamed model there makes the one failure §6.2 says must be
#: findable unfindable.
_GEMINI_PATH_MODEL = re.compile(r"/models/([^/:]+):")


def model_of(path: str, body: dict) -> str:
    """The model this request is for, from the body or — for gemini — the path."""
    match = _GEMINI_PATH_MODEL.search(path)
    if match:
        return match.group(1)
    return body.get("model") or "stub-model"


def surface_of(method: str, path: str, query: str) -> str | None:
    """The stub's own route table, mirrored. `None` means "not a completion"."""
    if method != "POST":
        return None
    if path.endswith("/chat/completions"):
        return "chat"
    if path.endswith("/responses"):
        return "responses"
    if path.endswith("/messages"):
        return "anthropic"
    if ":streamGenerateContent" in path:
        return "gemini-stream"
    if ":generateContent" in path:
        return "gemini"
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mllm-failserver"

    def log_message(self, *_args):
        pass

    def _send_json(self, obj: dict, status: int) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if ARGS.mode == "429ra":
            self.send_header("Retry-After", str(ARGS.retry_after))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, ValueError):
            body = {}
        path, _, query = self.path.partition("?")
        path = strip_session(path)
        model = model_of(path, body)
        surface = surface_of(method, path, query)
        # Streaming is a ROUTING input here, not decoration: `trunc` has to cut
        # the shape the client actually asked for. Same three signals
        # ``stub_server`` reads, plus `alt=sse` — which is how gemini asks, on the
        # URL rather than in the body or an Accept header.
        streamed = (
            bool(body.get("stream"))
            or "text/event-stream" in (self.headers.get("accept") or "")
            or "alt=sse" in query
        )
        entry = {"method": method, "path": path, "query": query, "model": model,
                 "surface": surface, "mode": ARGS.mode, "injected": bool(surface),
                 "stream": streamed,
                 "prompt_len": len(extract_prompt(body)), "body_bytes": len(raw),
                 "t": round(time.time(), 3)}
        _record(entry)

        # Health and catalogs always succeed — see the module docstring.
        if path in ("/__health", "/health"):
            return self._send_json({"ok": True}, 200)
        if surface is None:
            if method == "GET" and path.endswith("/muse-code/models"):
                return self._send_json(catalog("muse"), 200)
            if method == "GET" and (path.endswith("/models") or "/models/" in path):
                kind = ("anthropic" if self.headers.get("anthropic-version")
                        else "gemini" if "v1beta" in path else "openai")
                return self._send_json(catalog(kind), 200)
            if method == "POST" and path.endswith("/count_tokens"):
                return self._send_json({"input_tokens": 1}, 200)
            return self._send_json(
                {"error": {"type": "unhandled_route", "method": method, "path": path,
                           "message": "the multi-llm failserver does not serve this route"}},
                501)

        if ARGS.mode == "stall":
            time.sleep(ARGS.stall_s)
            self.close_connection = True
            return None
        if ARGS.mode == "empty":
            self.close_connection = True
            return None
        if ARGS.mode == "trunc":
            # THE BODY IS THE REQUESTING SURFACE'S OWN. This branch used to build
            # an OpenAI `chat.completion` object unconditionally, whatever surface
            # asked — so a cut of it delivered to claude-code's `/v1/messages` or
            # gemini's `:streamGenerateContent` was a MALFORMED-RESPONSE capture,
            # not a truncated one, and §6.2's `truncated` row would have been
            # mislabelled for every non-chat CLI. Rendering through
            # ``stub_server``'s own renderers is the same reuse the error path
            # already makes of ``error_body(surface, model)``: what a `trunc`
            # capture pins is a prefix of the exact bytes the SUCCESS capture
            # carried, and nothing else.
            #
            # Chat + non-streamed is byte-identical to the previous
            # implementation (``chat_body`` is where that literal came from), so
            # aider's committed `truncated.stdout.txt` still grades.
            if streamed:
                frames, sentinel = success_frames(surface, model, ARGS.trunc_answer)
                full = sse_payload(frames, sentinel)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # No Content-Length on a stream, and `Connection: close` for the
                # same measured reason ``stub_server._send_sse`` carries it: a
                # keep-alive SSE response with no framing has no defined end, and
                # claude-code sat on one for 90 s writing zero bytes.
                self.send_header("Connection", "close")
                self.end_headers()
            else:
                full = json.dumps(
                    success_body(surface, model, ARGS.trunc_answer)
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                # The Content-Length still promises the WHOLE body: that promise,
                # broken by the close below, is what makes this a truncation
                # rather than a short answer.
                self.send_header("Content-Length", str(len(full)))
                self.end_headers()
            self.wfile.write(full[:max(0, ARGS.trunc_bytes)])
            self.wfile.flush()
            self.close_connection = True
            return None

        status = MODES[ARGS.mode][0]
        return self._send_json(error_body(surface, model), status)


def main(argv=None) -> int:
    global ARGS
    parser = argparse.ArgumentParser(description="error-injection twin of stub_server.py")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--log", default=None, help="JSONL request log (the retry evidence)")
    parser.add_argument("--retry-after", type=int, default=1, help="429ra header value")
    parser.add_argument("--stall-s", type=int, default=3600)
    parser.add_argument("--trunc-bytes", type=int, default=60,
                        help="bytes of the success body `trunc` writes before closing")
    parser.add_argument("--trunc-answer", default="OK")
    ARGS = parser.parse_args(argv)
    if ARGS.log:
        os.makedirs(os.path.dirname(os.path.abspath(ARGS.log)) or ".", exist_ok=True)
        open(ARGS.log, "a", encoding="utf-8").close()

    # 127.0.0.1 only, like the stub: under `--network none` loopback is the only
    # address there is, and binding another is how a probe server eventually
    # answers something it should not.
    # Same refusal as the stub's, from the same table: a failserver on one of
    # omp's local-model discovery ports would be found by a CLI nobody pointed
    # at it, and a capture session would record the wrong server's error.
    refuse_probe_port(ARGS.port, "failserver")
    server = LoopbackHTTPServer(("127.0.0.1", ARGS.port), Handler)
    server.daemon_threads = True
    print(f"failserver[{ARGS.mode}] listening on 127.0.0.1:{ARGS.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
