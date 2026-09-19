#!/usr/bin/env python3
"""The stub: three protocol surfaces, scenario-resolved answers, total logging (C4).

**It runs inside the container**, started by the entrypoint before the first CLI
and torn down with it — not by the pytest harness on the host. That is forced by
the lane's own rule: under ``--network none`` the container has loopback and
nothing else, so a host-side process has no address a redirected CLI could reach.

**It never emits tool calls.** Provider-role invocations run the `generate`
profile: one prompt in, one answer out. Making the stub drive a tool loop would be
simulating an agent, which is the host lane's job and needs a real model. Stated
here so nobody later "improves" this into a fake agent.

**Response selection reuses the existing matcher** rather than inventing a second
one, so the fake binary (`tests/mocks/mock_llm.py`) and this server can never
disagree about what a scenario means.

**Two things fail loudly rather than defaulting**, and both are the same rule:
an unknown route answers `501` with `unhandled: true` in the log, and an unmatched
prompt answers a machine-readable error with `unmatched: true`. The harness fails
the row on either record's presence, independently of the CLI's exit code — a
client that shrugs off a missing catalog and hangs on a retry loop would otherwise
surface only as an unexplained timeout, and a prompt falling through to a generic
answer would mean a CLI never exercising the fixture its row claims to cover.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_lock = threading.Lock()
_seq = 0
_turns: dict[str, int] = {}
_log_path: Path | None = None
_scenario_path: Path | None = None
_resolver = None

#: `[(fact_name, compiled_regex)]` — the OPT-IN prompt-fact probe (`--facts`).
#:
#: The log records a prompt's sha256 and byte length, never its body, and that
#: rule is not relaxed here: a fact is a *boolean over a pattern the rig itself
#: wrote*, so what lands in the log is a list of the rig's own fact names and
#: nothing the model was sent. It exists because one question cannot be answered
#: from a digest — *did the host expand its own command shim into the prompt, and
#: did the user's argument survive its parser* — and the alternative (dumping the
#: request body) would put tens of KB of third-party text into an artifact this
#: repo publishes.
#:
#: Absent `--facts` the log shape is byte-identical to what it always was: the
#: key is written only when at least one pattern is configured.
_facts: list[tuple[str, "re.Pattern[str]"]] = []

DEFAULT_ANSWER = "OK"

# Every model id any row may pin, served by the catalogs. A CLI that fetches a
# catalog before its first completion (goose, kilocode) must find its model in it.
CATALOG_MODELS = [
    "stub-model", "openai/stub-model", "stub/stub-model",
    "gpt-5.5", "gpt-4o", "gpt-4o-mini", "o3",
    "claude-haiku-5", "claude-sonnet-5", "claude-3-5-haiku-20241022",
    "gemini-3-flash", "gemini-2.5-flash",
    "deepseek/deepseek-v4-flash-0731",
]


#: Ports this server must never bind, whatever it is asked for.
#:
#: MEASURED (omp 18.2.2, `plans/add-omp-provider.md` §1.8): every `omp -p` run
#: probes 127.0.0.1:11434, :8080 and :1234 for a local ollama / llama.cpp /
#: lm-studio before it dispatches, logging `model discovery failed for provider`
#: when nothing answers. A stub listening on one of them would be discovered as a
#: local model server by a CLI that was never redirected there — answering
#: requests the row does not declare, on a session key nothing minted, and
#: turning a redirection failure into a green row. It is a refusal rather than a
#: comment because the port is an argument: the day somebody passes one of these,
#: the failure should name the reason.
LOCAL_MODEL_PROBE_PORTS = (8080, 1234, 11434)


def refuse_probe_port(port: int, who: str) -> None:
    if port in LOCAL_MODEL_PROBE_PORTS:
        raise SystemExit(
            f"{who}: refusing to listen on 127.0.0.1:{port} — omp (and any CLI "
            f"with the same local-model discovery) probes "
            f"{LOCAL_MODEL_PROBE_PORTS} on every run and would find this server "
            f"without having been redirected to it. Pick another port."
        )


def _record(entry: dict) -> None:
    """One JSON object per line, flushed, so a killed container still leaves a log."""
    global _seq
    with _lock:
        _seq += 1
        entry["seq"] = _seq
    assert _log_path is not None
    with open(_log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------- #
# Sessions
#
# HTTP carries no session identity and the fan-out is parallel, so a global turn
# counter would interleave nondeterministically across concurrent CLIs and quietly
# destroy both the determinism claim and the zero-diff goldens. This server
# therefore never keeps one.
# --------------------------------------------------------------------------- #

_SESSION_IN_PATH = re.compile(r"^/s/([^/]+)")


def session_key(path: str, body: dict, headers) -> str:
    """Per-exec id from the base URL path, else a coarse tuple.

    Precedence order, and rule 1 is what the wrapper arranges: an id minted per
    `exec` and baked into the base URL is exact, needs no cooperation from the
    client, and costs one path segment stripped before dispatch. Phase 1 measured
    it working on every verified CLI. Rule 2 is the fallback for a CLI that
    rejects a path prefix; note muse keeps the prefix on its completion POST but
    rebuilds its catalog GET from the origin, so a vendor session header is the
    tiebreaker before the coarse tuple.
    """
    match = _SESSION_IN_PATH.match(path)
    if match:
        return match.group(1)
    for header in ("x-tbh-session-id", "x-session-id", "x-session-affinity"):
        value = headers.get(header)
        if value:
            return f"hdr:{value}"
    return f"model:{body.get('model', '?')}"


def strip_session(path: str) -> str:
    return _SESSION_IN_PATH.sub("", path, count=1) or "/"


def next_turn(key: str) -> int:
    """Turn counters are per session key and incremented under a lock.

    Requests within one session are sequential by construction, so concurrent
    sessions cannot perturb each other's answers.
    """
    with _lock:
        _turns[key] = _turns.get(key, 0) + 1
        return _turns[key]


# --------------------------------------------------------------------------- #
# Prompt extraction and resolution
# --------------------------------------------------------------------------- #

def extract_prompt(body: dict) -> str:
    """Recover the user-visible prompt text from any of the three wire shapes."""
    parts: list[str] = []

    def walk(value):
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for key in ("text", "content", "input_text", "parts"):
                if key in value:
                    walk(value[key])

    for key in ("messages", "input", "contents", "prompt", "system"):
        if key in body:
            walk(body[key])
    return "\n".join(p for p in parts if p)


class Unmatched(Exception):
    """No pattern owned this prompt and the scenario declares on_unmatched: fail."""


def resolve(prompt: str, turn: int) -> str:
    """Answer text for this prompt, via the shared matcher when one is configured."""
    if _resolver is None:
        return DEFAULT_ANSWER
    return _resolver(prompt, turn)


# --------------------------------------------------------------------------- #
# Rendering: one resolved answer, two shapes, so they cannot disagree.
# --------------------------------------------------------------------------- #

def chat_body(model: str, answer: str) -> dict:
    return {
        "id": "chatcmpl-stub", "object": "chat.completion", "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def chat_stream(model: str, answer: str):
    base = {"id": "chatcmpl-stub", "object": "chat.completion.chunk",
            "created": 0, "model": model}
    yield None, dict(base, choices=[{"index": 0, "delta": {"role": "assistant"},
                                     "finish_reason": None}])
    yield None, dict(base, choices=[{"index": 0, "delta": {"content": answer},
                                     "finish_reason": None}])
    yield None, dict(base, choices=[{"index": 0, "delta": {},
                                     "finish_reason": "stop"}])


def responses_body(model: str, answer: str) -> dict:
    return {
        "id": "resp_stub", "object": "response", "created_at": 0,
        "status": "completed", "model": model,
        "output": [{"id": "msg_stub", "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": answer,
                                 "annotations": []}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def responses_stream(model: str, answer: str):
    """The full Responses SSE sequence, named events included.

    codex is SSE-only on this surface and reads the `event:` line as well as the
    data, so the short four-event form is not enough.
    """
    resp = responses_body(model, answer)
    item = resp["output"][0]
    part = item["content"][0]
    in_progress = dict(resp, status="in_progress", output=[])
    yield "response.created", {"type": "response.created", "sequence_number": 0,
                               "response": in_progress}
    yield "response.in_progress", {"type": "response.in_progress",
                                   "sequence_number": 1, "response": in_progress}
    yield "response.output_item.added", {
        "type": "response.output_item.added", "sequence_number": 2,
        "output_index": 0,
        "item": {"id": item["id"], "type": "message", "role": "assistant",
                 "status": "in_progress", "content": []}}
    yield "response.content_part.added", {
        "type": "response.content_part.added", "sequence_number": 3,
        "item_id": item["id"], "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []}}
    yield "response.output_text.delta", {
        "type": "response.output_text.delta", "sequence_number": 4,
        "item_id": item["id"], "output_index": 0, "content_index": 0,
        "delta": answer}
    yield "response.output_text.done", {
        "type": "response.output_text.done", "sequence_number": 5,
        "item_id": item["id"], "output_index": 0, "content_index": 0,
        "text": answer}
    yield "response.content_part.done", {
        "type": "response.content_part.done", "sequence_number": 6,
        "item_id": item["id"], "output_index": 0, "content_index": 0, "part": part}
    yield "response.output_item.done", {
        "type": "response.output_item.done", "sequence_number": 7,
        "output_index": 0, "item": item}
    yield "response.completed", {"type": "response.completed",
                                 "sequence_number": 8, "response": resp}


def messages_body(model: str, answer: str) -> dict:
    return {
        "id": "msg_stub", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": answer}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def messages_stream(model: str, answer: str):
    start = messages_body(model, answer)
    start["content"] = []
    start["stop_reason"] = None
    start["usage"] = {"input_tokens": 1, "output_tokens": 0}
    yield "message_start", {"type": "message_start", "message": start}
    yield "content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}
    yield "ping", {"type": "ping"}
    yield "content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": answer}}
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield "message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": "end_turn",
                                      "stop_sequence": None},
                            "usage": {"output_tokens": 1}}
    yield "message_stop", {"type": "message_stop"}


def catalog(surface: str) -> dict:
    if surface == "anthropic":
        return {"data": [{"id": m, "type": "model", "display_name": m,
                          "created_at": "2026-01-01T00:00:00Z"}
                         for m in CATALOG_MODELS],
                "has_more": False,
                "first_id": CATALOG_MODELS[0], "last_id": CATALOG_MODELS[-1]}
    if surface == "gemini":
        return {"models": [{"name": f"models/{m}", "displayName": m,
                            "supportedGenerationMethods":
                                ["generateContent", "streamGenerateContent"]}
                           for m in CATALOG_MODELS]}
    if surface == "muse":
        # `data`, not `models`: muse decodes this with a typed deserializer and
        # fails the whole run with `missing field \`data\`` on the wrong envelope.
        # Serving a route is not the same as serving it in the shape its client
        # parses, which is why the stub self-test replays each recorded route
        # rather than only asserting a 200.
        return {"data": [{"id": m, "name": m} for m in CATALOG_MODELS]}
    return {"object": "list",
            "data": [{"id": m, "object": "model", "created": 0, "owned_by": "stub"}
                     for m in CATALOG_MODELS]}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mllm-stub"

    def log_message(self, *_args):
        pass  # the JSONL log is the record; stderr noise is not

    # -- plumbing ---------------------------------------------------------- #
    def _read_body(self) -> tuple[bytes, dict]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return raw, (json.loads(raw.decode("utf-8")) if raw else {})
        except (UnicodeDecodeError, ValueError):
            return raw, {}

    def _send_json(self, obj: dict, status: int = 200) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _send_sse(self, frames, done_sentinel: bool = True) -> None:
        """text/event-stream, a flush per frame, no response buffering.

        **`Connection: close`, measured rather than assumed.** Under HTTP/1.1 a
        response with neither `Content-Length` nor chunked framing has no defined
        end, so a client reading until the socket closes waits forever on a
        keep-alive connection. codex only survived that in the spike because it
        stops reading at `response.completed`; claude-code retried 16 times over
        90 s writing zero bytes, which is indistinguishable from "redirection
        failed". Closing the connection is the form all eleven verified CLIs were
        measured against.

        **`data: [DONE]` is an OpenAI-ism, not an SSE one.** gemini-cli
        `JSON.parse`s every frame and dies with `Unexpected token 'D'`, so the
        sentinel is emitted only on the surfaces whose clients expect it.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        emitted_named = False
        for name, obj in frames:
            chunk = (f"event: {name}\n" if name else "") + \
                    f"data: {json.dumps(obj)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
            if name:
                emitted_named = True
        # Chat Completions terminates with the sentinel; the named-event surfaces
        # (Responses, Anthropic Messages) terminate with their own final event and
        # must not get one.
        if done_sentinel and not emitted_named:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    # -- routing ------------------------------------------------------------ #
    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        raw, body = self._read_body()
        raw_path = self.path
        # SPLIT THE QUERY STRING BEFORE ROUTING. Claude Code posts
        # `/v1/messages?beta=true` and gemini signals streaming with `?alt=sse`,
        # so suffix-matching the raw path 501s a route we do serve — and a 501
        # there reads to the client as an error worth retrying, which is how one
        # missing `split("?")` became 16 retries over 90 s with zero bytes
        # written, indistinguishable from "redirection failed".
        path, _, query = raw_path.partition("?")
        path = strip_session(path)
        key = session_key(raw_path, body, self.headers)
        turn = next_turn(key)
        model = body.get("model") or "stub-model"
        prompt = extract_prompt(body)
        # Streaming is not optional: these clients ask for a stream by default, so
        # a stub that only returns one JSON body gets rejected or hung.
        streamed = bool(body.get("stream")) or \
            "text/event-stream" in (self.headers.get("accept") or "")

        entry = {
            "method": method, "path": path, "raw_path": raw_path,
            "session": key, "turn": turn, "model": model, "stream": streamed,
            # The prompt SHA, never the body. And never request headers: the
            # Authorization a redirected CLI sends is a fake key, but logging
            # headers as a habit is how the real one eventually gets logged.
            "query": query,
            "prompt_sha": _sha(prompt), "prompt_len": len(prompt),
            "body_bytes": len(raw), "has_tools": bool(body.get("tools")),
            "unhandled": False, "unmatched": False, "fixture": None,
            "surface": None, "t": round(time.time(), 3),
        }
        if _facts:
            # Names only, and only the ones that matched — see `_facts`.
            entry["facts"] = [name for name, rx in _facts if rx.search(prompt)]

        try:
            self._route(method, path, query, body, model, prompt, turn,
                        streamed, entry)
        finally:
            _record(entry)

    def _resolve_or_422(self, prompt, turn, entry) -> tuple[bool, str]:
        """Resolve for any surface, or answer 422 and flag the log line.

        Every completion surface resolves through here, gemini included. This is
        the response-selection twin of the unknown-route rule, and it bites
        harder: a generic fallback is shaped enough to look like a passing phase
        while proving nothing. An `Unmatched` escaping `_route` would be worse
        still — the handler thread dies, the client sees a dropped connection
        rather than the machine-readable error, and the log line is written with
        `unmatched: false`, so the fail-on-unmatched assertion the provider lane
        rests on never fires.

        Returns `(matched, answer)`; on `False` the 422 has already been sent and
        the caller must return without writing a second response.
        """
        try:
            answer = resolve(prompt, turn)
        except Unmatched as exc:
            entry["unmatched"] = True
            self._send_json(
                {"error": {"type": "unmatched_prompt", "message": str(exc),
                           "prompt_sha": entry["prompt_sha"]}}, status=422)
            return False, ""
        entry["fixture"] = getattr(resolve, "last_fixture", None)
        return True, answer

    def _route(self, method, path, query, body, model, prompt, turn, streamed,
               entry):
        # `query` is a routing input, not just a logged one: the gemini surface
        # splits on `alt=sse`. It is threaded in like `path` and `prompt` rather
        # than read back out of `entry`, which is the log record.
        if path in ("/__health", "/health"):
            entry["surface"] = "health"
            return self._send_json({"ok": True})

        # THE BASE-URL ROOT, and it is a reachability probe rather than a model
        # call. MEASURED 2026-09-19 on the grok 0.2.93 -> 1.0.34 bump: 1.0.x opens
        # with `GET /` against whatever `GROK_MODELS_BASE_URL` names, before any
        # completion. 0.2.93 did not, so the route was neither served nor declared
        # and the row recorded it unserved. It is answered as health, not routed
        # to a surface: there is no prompt to resolve and no scenario that could
        # match one, so sending it down `_resolve_or_422` would 422 a request the
        # CLI is entitled to make. Declared in redirect.yaml's grok `http_log`
        # beside the completion call, so it stays a stated fact rather than a
        # silently-tolerated extra request.
        if method == "GET" and path in ("", "/"):
            entry["surface"] = "health"
            return self._send_json({"ok": True})

        surface = None
        if method == "POST" and path.endswith("/chat/completions"):
            surface = "chat"
        elif method == "POST" and path.endswith("/responses"):
            surface = "responses"
        elif method == "POST" and path.endswith("/messages"):
            surface = "anthropic"

        if surface:
            entry["surface"] = surface
            matched, answer = self._resolve_or_422(prompt, turn, entry)
            if not matched:
                return None
            renderers = {
                "chat": (chat_body, chat_stream),
                "responses": (responses_body, responses_stream),
                "anthropic": (messages_body, messages_stream),
            }[surface]
            if streamed:
                return self._send_sse(renderers[1](model, answer))
            return self._send_json(renderers[0](model, answer))

        # Token counting: a deterministic count derived from the body, never a
        # call to a real tokenizer.
        if method == "POST" and path.endswith("/count_tokens"):
            entry["surface"] = "count_tokens"
            return self._send_json({"input_tokens": max(1, len(prompt) // 4)})

        if method == "POST" and ":streamGenerateContent" not in path \
                and ":generateContent" in path and "alt=sse" not in query:
            entry["surface"] = "gemini"
            matched, answer = self._resolve_or_422(prompt, turn, entry)
            if not matched:
                return None
            return self._send_json({
                "candidates": [{"content": {"role": "model",
                                            "parts": [{"text": answer}]},
                                "finishReason": "STOP", "index": 0}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1,
                                  "totalTokenCount": 2}})

        if method == "POST" and ":streamGenerateContent" in path:
            entry["surface"] = "gemini-stream"
            matched, answer = self._resolve_or_422(prompt, turn, entry)
            if not matched:
                return None
            # No [DONE]: gemini-cli JSON.parses every frame and dies with
            # `Unexpected token 'D'`.
            return self._send_sse([(None, {
                "candidates": [{"content": {"role": "model",
                                            "parts": [{"text": answer}]},
                                "finishReason": "STOP", "index": 0}]})],
                done_sentinel=False)

        # Catalogs. Phase 1 measured which CLIs fetch one before their first
        # completion (goose, kilocode) and the vendor spellings they use — muse's
        # `/muse-code/models` only worked in the spike by accident of a generic
        # suffix match, so it is served explicitly here.
        if method == "GET":
            if path.endswith("/muse-code/models"):
                entry["surface"] = "catalog:muse"
                return self._send_json(catalog("muse"))
            if path.endswith("/models") or "/models/" in path:
                if self.headers.get("anthropic-version"):
                    kind = "anthropic"
                elif "v1beta" in path:
                    kind = "gemini"
                else:
                    kind = "openai"
                entry["surface"] = f"catalog:{kind}"
                return self._send_json(catalog(kind))

        # An unknown request fails loudly rather than 404-ing quietly.
        entry["unhandled"] = True
        return self._send_json(
            {"error": {"type": "unhandled_route", "method": method, "path": path,
                       "message": "the multi-llm stub does not serve this route"}},
            status=501)


def install_resolver(scenario: Path | None):
    """Wire the shared scenario matcher, when a scenario was supplied.

    The matcher is imported rather than reimplemented: the fake binary and this
    server resolving a scenario differently is the seam most likely to rot, and a
    plain-CI parity test asserts they do not.
    """
    global _resolver
    if scenario is None:
        return
    sys.path.insert(0, "/opt/mllm/skill")
    from tests.scenario_matcher import ScenarioMatcher, UnmatchedPrompt  # type: ignore

    matcher = ScenarioMatcher.load(scenario)

    def _resolve(prompt: str, turn: int) -> str:
        try:
            resolved = matcher.resolve(prompt, turn=turn)
        except UnmatchedPrompt as exc:
            raise Unmatched(str(exc)) from exc
        _resolve.last_fixture = getattr(resolved, "fixture", None)
        if resolved is None:
            # "This scenario has nothing to say" — only reachable under a
            # LENIENT scenario, since a strict one raises above. It has to mean
            # the same thing here as it does in the fake binary, which falls
            # through to its own default: `str(resolved)` would answer the
            # literal four characters "None", which is not a default, it is a
            # wrong answer that every parser downstream would then try to read.
            return DEFAULT_ANSWER
        return resolved.content

    _resolve.last_fixture = None
    globals()["resolve"] = _resolve
    _resolver = _resolve


def install_facts(path: Path | None) -> None:
    """Load the opt-in prompt-fact probe: `{"<fact name>": "<regex>"}`.

    Compiled case-insensitively, to match :meth:`PromptRule.matches`, so a
    scenario pattern and a fact pattern cannot mean two different things. A
    pattern spanning lines writes its own `(?s)`.

    An unparseable file or an invalid pattern is fatal rather than ignored: a
    silently empty probe would report "the host expanded nothing" for every row,
    which is the one verdict this probe must never fabricate.
    """
    global _facts
    if path is None:
        return
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"stub: --facts {path} carries no {{name: regex}} map")
    _facts = [(str(name), re.compile(str(pattern), re.IGNORECASE))
              for name, pattern in sorted(data.items())]
    print(f"stub: prompt-fact probe armed with {len(_facts)} pattern(s): "
          f"{[name for name, _ in _facts]}", flush=True)


def main(argv=None) -> int:
    global _log_path, _scenario_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--facts", default=None,
                        help="JSON {name: regex}; each request's log entry gains "
                             "`facts: [name…]` for the patterns that matched its "
                             "prompt. Names only — never the prompt body.")
    args = parser.parse_args(argv)

    _log_path = Path(args.log)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_path.touch()
    _scenario_path = Path(args.scenario) if args.scenario else None
    install_resolver(_scenario_path)
    install_facts(Path(args.facts) if args.facts else None)

    # 127.0.0.1 explicitly, never 0.0.0.0: there is no other interface to serve
    # and binding one is how a stub eventually answers something it shouldn't.
    #
    # ThreadingHTTPServer, not the single-threaded default: the orchestrator fans
    # out to several provider CLIs concurrently and clients with connection
    # pooling hold keep-alive sockets open. A single-threaded server serializes
    # that fan-out at best and deadlocks a pooling client at worst.
    refuse_probe_port(args.port, "stub")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"stub listening on 127.0.0.1:{args.port}", flush=True)
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
