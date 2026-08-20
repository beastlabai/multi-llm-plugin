#!/usr/bin/env python3
"""The credentialed lane's egress allowlist, tested without a container.

The resolver is the one containment control in harness-expansion.md §7.4 that is
pure logic: given a DNS query it either forwards it or refuses it, and it writes
down which. That makes it testable on any machine, which matters — the lane it
guards runs rarely, costs money, and is exactly the kind of thing whose guard
rots unobserved between runs.

What is deliberately NOT tested here: binding port 53. That needs the lane's
``--sysctl net.ipv4.ip_unprivileged_port_start=53``, and a daemon that refuses it
leaves the control unavailable — which §7.4 turns into a recorded **skip**, never
a weakened run. These tests bind a high port instead and exercise the decision.
"""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path

import pytest

from .container.egress_resolver import (
    DEFAULT_ALLOWLIST,
    QueryLog,
    is_allowed,
    nxdomain_response,
    parse_question,
    serve,
)

_RCODE_MASK = 0x000F
_RCODE_NXDOMAIN = 3


def _query(name: str, ident: int = 0x1234) -> bytes:
    """A minimal well-formed A query for ``name``."""
    header = struct.pack("!HHHHHH", ident, 0x0100, 1, 0, 0, 0)
    labels = b"".join(
        bytes([len(part)]) + part.encode("ascii") for part in name.split("."))
    return header + labels + b"\x00" + struct.pack("!HH", 1, 1)


class TestTheDecision:
    def test_the_allowlist_is_a_suffix_match_not_a_substring(self):
        assert is_allowed("openrouter.ai", DEFAULT_ALLOWLIST)
        assert is_allowed("api.openrouter.ai", DEFAULT_ALLOWLIST)
        assert is_allowed("OpenRouter.AI", DEFAULT_ALLOWLIST)
        # The whole reason it is not `in`: an attacker-controlled or merely
        # careless name that merely CONTAINS the allowed one must not pass.
        assert not is_allowed("openrouter.ai.evil.test", DEFAULT_ALLOWLIST)
        assert not is_allowed("notopenrouter.ai", DEFAULT_ALLOWLIST)

    @pytest.mark.parametrize("name", [
        "api.github.com", "raw.githubusercontent.com", "ampcode.com",
        "api.moonshot.cn", "dashscope.aliyuncs.com", "api.mistral.ai",
        "api.deepseek.com",
    ])
    def test_every_vendor_home_the_lane_forbids_is_refused(self, name):
        """These are the call-homes assertion 5 reports as findings."""
        assert not is_allowed(name, DEFAULT_ALLOWLIST)

    def test_a_question_round_trips(self):
        qname, qtype = parse_question(_query("api.openrouter.ai"))
        assert qname == "api.openrouter.ai"
        assert qtype == 1

    def test_a_malformed_packet_is_refused_rather_than_raised(self):
        """A traceback here would look exactly like the network failure it explains."""
        assert parse_question(b"") == ("", 0)
        assert parse_question(b"\x00" * 4) == ("", 0)

    def test_the_refusal_is_nxdomain_and_echoes_the_query_id(self):
        """NXDOMAIN, not a hang: a blocked call-home must not read as a slow CLI."""
        packet = _query("api.github.com", ident=0xABCD)
        reply = nxdomain_response(packet)
        assert reply[:2] == packet[:2]
        (flags,) = struct.unpack("!H", reply[2:4])
        assert flags & 0x8000, "QR bit not set — this is not a response"
        assert flags & _RCODE_MASK == _RCODE_NXDOMAIN


class TestTheLog:
    def test_every_query_is_recorded_allowed_and_refused_alike(self, tmp_path: Path):
        """Enforcement without observation answers 'did anything leak?' with silence."""
        log_path = tmp_path / "dns.jsonl"
        udp, tcp, _log = serve(DEFAULT_ALLOWLIST, upstream="127.0.0.1",
                               log_path=log_path, port=0)
        try:
            port = udp.server_address[1]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(5)
                for name in ("api.github.com", "telemetry.example.test"):
                    sock.sendto(_query(name), ("127.0.0.1", port))
                    reply, _ = sock.recvfrom(65535)
                    (flags,) = struct.unpack("!H", reply[2:4])
                    assert flags & _RCODE_MASK == _RCODE_NXDOMAIN
        finally:
            udp.shutdown()
            tcp.shutdown()

        records = [json.loads(line) for line in
                   log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert {r["qname"] for r in records} == {
            "api.github.com", "telemetry.example.test"}
        assert {r["decision"] for r in records} == {"nxdomain"}

    def test_a_log_with_no_path_is_inert(self):
        QueryLog(None).record(qname="x", decision="nxdomain")  # must not raise
