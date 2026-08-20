#!/usr/bin/env python3
"""An allowlisting DNS resolver, so the credentialed provider lane's egress is a
*measurement* rather than a hope (harness-expansion.md §7.4).

WHY DNS AND NOT A PROXY. Three of the CLIs this lane drives have a vendor home to
phone (GitHub for copilot, Moonshot for kimi, DashScope for qwen) plus an updater
and telemetry apiece, and the lane hands them a live credit-spending key on a
bridge network. The obvious containment — ``HTTPS_PROXY`` with a CONNECT
allowlist — does not bind: Node's global ``fetch`` ignores proxy environment
variables by default, and every one of these CLIs is a Node program. A proxy
allowlist is therefore *additive* where a CLI honours it and worth nothing where
it does not. Name resolution is the one chokepoint every one of them goes
through, so that is where the allowlist lives.

WHAT IT IS NOT. An IP-literal request bypasses DNS entirely, and this cannot see
it. That is a documented limit of the measure, not a reason to skip it: the lane
also pins the key's own credit limit (§7.4's spend bound, part 4), which is the
control that still holds when every other one has a bug.

THE LOG IS THE POINT. Enforcement without observation would answer "did anything
leak?" with silence. Every query is appended to a JSONL log — allowed and refused
alike — and the lane's assertion 5 reads it back: ``openrouter.ai`` and nothing
else. A query for ``*.github.com``, ``ampcode.com``, ``*.moonshot.cn``,
``dashscope*``, ``*.mistral.ai`` or ``*.deepseek.com`` is not a curiosity, it is a
finding — it means a shipped adapter makes a request users did not ask for.

PORT 53 UNDER A DROPPED CAPABILITY SET. The container runs as uid 10001 with
``--cap-drop ALL``, so binding the privileged DNS port would normally be refused.
The lane's ``docker run`` therefore passes
``--sysctl net.ipv4.ip_unprivileged_port_start=53`` — a namespaced sysctl, so it
widens nothing outside this container and costs no capability. If a daemon
refuses that sysctl the containment control cannot be applied, and §7.4's skip
policy applies: the row **skips with the reason recorded**, never runs weakened.

Stdlib only, like the rest of the rig — the image's system ``python3`` runs it.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import struct
import threading
import time
from pathlib import Path

#: The only name this lane is allowed to resolve. A suffix match, so
#: ``openrouter.ai`` and ``api.openrouter.ai`` both pass and
#: ``openrouter.ai.evil.test`` does not.
DEFAULT_ALLOWLIST = ("openrouter.ai",)

#: Where an allowlisted query is forwarded. The container's own resolver has been
#: replaced by this process, so the upstream has to be stated rather than read
#: from /etc/resolv.conf — which now points back here.
DEFAULT_UPSTREAM = "1.1.1.1"

_RCODE_NOERROR = 0
_RCODE_NXDOMAIN = 3


def parse_question(packet: bytes) -> tuple[str, int]:
    """``(qname, qtype)`` from a DNS query, or ``("", 0)`` if it will not parse.

    Deliberately forgiving: a malformed packet is logged and refused, never
    raised. This process sits in front of the CLI under test, and a traceback
    here would look exactly like the network failure it is meant to explain.
    """
    if len(packet) < 12:
        return "", 0
    offset = 12
    labels = []
    while offset < len(packet):
        length = packet[offset]
        if length == 0:
            offset += 1
            break
        # A compression pointer has no business in a question section; treat it
        # as unparseable rather than chasing it.
        if length & 0xC0:
            return "", 0
        offset += 1
        labels.append(packet[offset:offset + length].decode("ascii", "replace"))
        offset += length
    qtype = 0
    if offset + 2 <= len(packet):
        qtype = struct.unpack("!H", packet[offset:offset + 2])[0]
    return ".".join(labels), qtype


def is_allowed(qname: str, allowlist) -> bool:
    """Exact or dot-suffix match, case-insensitively. Never a bare substring."""
    name = qname.lower().rstrip(".")
    for allowed in allowlist:
        allowed = allowed.lower().rstrip(".")
        if name == allowed or name.endswith("." + allowed):
            return True
    return False


def nxdomain_response(packet: bytes) -> bytes:
    """The refusal, built from the query so the client matches it to its request.

    NXDOMAIN rather than REFUSED or a timeout on purpose: a resolver that hangs
    turns a blocked call-home into a row that times out at the outer cap, and the
    evidence then reads as "the CLI is slow" instead of "the CLI tried to reach a
    host this lane does not permit".
    """
    if len(packet) < 12:
        return b""
    ident = packet[:2]
    # QR=1 (response), RD copied from the query, RA=1, RCODE=NXDOMAIN.
    flags = 0x8180 | _RCODE_NXDOMAIN
    header = ident + struct.pack("!HHHHH", flags, 1, 0, 0, 0)
    return header + packet[12:]


def forward(packet: bytes, upstream: str, timeout: float = 5.0) -> bytes:
    """Hand an allowlisted query to the real resolver, verbatim."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(packet, (upstream, 53))
        data, _ = sock.recvfrom(65535)
    return data


class QueryLog:
    """Append-only JSONL, flushed per record and safe across both handlers."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def record(self, **fields) -> None:
        if self.path is None:
            return
        fields.setdefault("ts", time.time())
        line = json.dumps(fields, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()


class _Resolver:
    """The decision, shared by the UDP and TCP servers so they cannot diverge."""

    def __init__(self, allowlist, upstream: str, log: QueryLog) -> None:
        self.allowlist = tuple(allowlist)
        self.upstream = upstream
        self.log = log

    def answer(self, packet: bytes, client: str, transport: str) -> bytes:
        qname, qtype = parse_question(packet)
        if qname and is_allowed(qname, self.allowlist):
            try:
                reply = forward(packet, self.upstream)
                self.log.record(qname=qname, qtype=qtype, decision="allow",
                                client=client, transport=transport,
                                upstream=self.upstream)
                return reply
            except (OSError, socket.timeout) as exc:  # noqa: BLE001
                # An allowlisted name we could not resolve is still recorded as
                # allowed — the decision was ours, the failure was upstream's,
                # and conflating the two would read as a containment breach.
                self.log.record(qname=qname, qtype=qtype, decision="allow",
                                client=client, transport=transport,
                                upstream=self.upstream,
                                error=f"{type(exc).__name__}: {exc}")
                return nxdomain_response(packet)
        self.log.record(qname=qname or "<unparseable>", qtype=qtype,
                        decision="nxdomain", client=client, transport=transport)
        return nxdomain_response(packet)


class _UDPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        packet, sock = self.request
        reply = self.server.resolver.answer(packet, self.client_address[0], "udp")
        if reply:
            sock.sendto(reply, self.client_address)


class _TCPHandler(socketserver.BaseRequestHandler):
    """glibc falls back to TCP on a truncated answer, so both are served.

    A UDP-only allowlist looks airtight until the first response over 512 bytes,
    at which point the client retries over TCP, finds nothing listening, and the
    failure surfaces as an unexplained resolver error.
    """

    def handle(self) -> None:
        header = self.request.recv(2)
        if len(header) < 2:
            return
        (length,) = struct.unpack("!H", header)
        packet = b""
        while len(packet) < length:
            chunk = self.request.recv(length - len(packet))
            if not chunk:
                return
            packet += chunk
        reply = self.server.resolver.answer(packet, self.client_address[0], "tcp")
        if reply:
            self.request.sendall(struct.pack("!H", len(reply)) + reply)


class _UDPServer(socketserver.ThreadingUDPServer):
    allow_reuse_address = True


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def serve(allowlist=DEFAULT_ALLOWLIST, upstream: str = DEFAULT_UPSTREAM,
          log_path: Path | None = None, host: str = "127.0.0.1",
          port: int = 53) -> tuple[_UDPServer, _TCPServer, QueryLog]:
    """Start both servers on background threads and return them for shutdown.

    Raises ``PermissionError``/``OSError`` if the port cannot be bound — which the
    caller must treat as "this containment control cannot be applied", i.e. a
    §7.4 **skip**, never a weakened run.
    """
    log = QueryLog(log_path)
    resolver = _Resolver(allowlist, upstream, log)

    udp = _UDPServer((host, port), _UDPHandler)
    udp.resolver = resolver
    tcp = _TCPServer((host, port), _TCPHandler)
    tcp.resolver = resolver

    for server in (udp, tcp):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    return udp, tcp, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow", action="append", default=None,
                        help="a name to permit (repeatable); defaults to openrouter.ai")
    parser.add_argument("--upstream", default=os.environ.get(
        "MLLM_EGRESS_UPSTREAM", DEFAULT_UPSTREAM))
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--port", type=int, default=53)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    udp, tcp, _log = serve(args.allow or DEFAULT_ALLOWLIST, args.upstream,
                           args.log, args.host, args.port)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        udp.shutdown()
        tcp.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
