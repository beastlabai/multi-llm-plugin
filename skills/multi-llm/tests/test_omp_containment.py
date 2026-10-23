"""Two things omp must not do, both graded rather than asserted in prose.

**What its committed bytes must not carry.** ``tests/fixtures/omp/`` is the one
fixture set in this repo whose rows were captured on a developer's host rather
than in the container image, because three of them need a tool loop and the
committed ``stub_server.py`` forbids itself from driving one. Host captures
carry things a probe-image capture cannot: a home directory naming the user, a
spawn cwd naming the repository, and — had any row ever reached the live
service — an OpenRouter generation id, which resolves to the account in its
dashboard. None of those is key-shaped, so ``tests/test_no_committed_secrets``'
gitleaks pass over ``git ls-files`` does not look for them. This module does,
and it plants its own vectors so the sweep cannot go vacuous.

**What its dispatches must not mount.** ``--approval-mode yolo`` is bounded by
the ``--tools`` allow-list and by nothing else, and upstream's own loader
appends a freshly mounted ``mcp__<server>__<tool>`` to the enabled set
*regardless* of ``--tools`` — so an MCP server that mounts on a host dispatch is
an auto-approved tool the harness never declared and cannot see. The overlay's
``mcp.enableProjectConfig: false`` is the lever that stops it. That it is *in*
the overlay is graded by ``TestOmpProvider``; that it *works* can only be
measured, so the second class here runs a real dispatch with two populated
``mcp.json`` files on the paths omp reads and asserts nothing starts.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packaging"))

# The vendor-key regexes have ONE owner. `packaging/common/payload.py` is the
# secret sweep every distribution tree already goes through, and importing it
# here — the way `tests/test_opencode_package.py` does — is what stops this
# module from becoming a second, quietly diverging copy of that table.
from common import payload  # noqa: E402

from utils.providers.base import PROFILE_GENERATE, build_child_env  # noqa: E402
from utils.providers.omp import (  # noqa: E402
    OMP_CREDENTIAL_ENV,
    OMP_TOOLS,
    OmpProvider,
)

FIXTURES = SKILL_ROOT / "tests" / "fixtures" / "omp"
CHAIN_BUNDLE = SKILL_ROOT / "tests" / "golden" / "container" / "omp" / "bundle.json"

#: A home directory that is not the container's own. `/home/e2e` is the probe
#: image's HOME and the scrub's replacement for a host one, so it is the only
#: `/home/` prefix a committed row may carry.
_FOREIGN_HOME = re.compile(r"/home/(?!e2e/|e2e\b)")

#: This session's scratch root. A capture taken through a coding agent's own
#: temp dir carries it, and it names both the user and the repository.
_AGENT_SCRATCH = re.compile(r"/tmp/claude-")

#: An OpenRouter **generation id**. Not key-shaped, and not what gitleaks looks
#: for, but it resolves to the account that made the request in the dashboard.
_GENERATION_ID = re.compile(r"\bgen-\d{10}-[A-Za-z0-9]{20,}")

#: The redaction marker itself. Its presence in a committed row would mean the
#: credential had been on a channel it must never be on — the capture is then
#: discarded and the key rotated, never committed with the marker in it.
_REDACTED = re.compile(r"<redacted>")


def _identity_literals():
    """The developer's own name and email, read at test time and never spelled.

    Writing either literal into this file would make the guard the one committed
    place they appear.
    """
    out = []
    for key in ("user.name", "user.email"):
        proc = subprocess.run(
            ["git", "config", "--get", key],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        value = (proc.stdout or "").strip()
        if value and len(value) > 3:
            out.append((key, value))
    return out


def _findings(name, body):
    """Every rule this sweep enforces, over one file's text."""
    found = []
    for label, pattern in payload._SECRET_PATTERNS:
        match = pattern.search(body)
        if match:
            found.append(f"{name}: {label} ({match.group(0)[:12]}…)")
    for label, pattern in (
        ("a home directory that is not /home/e2e", _FOREIGN_HOME),
        ("an agent scratch path", _AGENT_SCRATCH),
        ("an OpenRouter generation id", _GENERATION_ID),
        ("a <redacted> marker", _REDACTED),
    ):
        match = pattern.search(body)
        if match:
            found.append(f"{name}: {label} ({match.group(0)[:40]}…)")
    for key, value in _identity_literals():
        if value in body:
            found.append(f"{name}: the committer's {key}")
    return found


def _committed_texts():
    """Every byte this module is responsible for, as (name, text) pairs."""
    paths = sorted(p for p in FIXTURES.iterdir() if p.is_file())
    if CHAIN_BUNDLE.exists():
        paths.append(CHAIN_BUNDLE)
    return [
        (str(p.relative_to(SKILL_ROOT)), p.read_text(encoding="utf-8", errors="replace"))
        for p in paths
    ]


class TestTheCommittedFixtureBytesCarryNothingPersonal:
    """The Stage A scrub gate, over ``tests/fixtures/omp/`` and the chain bundle."""

    def test_the_directory_is_not_empty(self):
        """A sweep over nothing passes trivially, so say what it swept."""
        texts = _committed_texts()
        assert len(texts) >= 24, [name for name, _ in texts]
        assert any(name.endswith("manifest.yaml") for name, _ in texts)

    def test_no_committed_file_carries_a_credential_or_an_identity(self):
        found = []
        for name, body in _committed_texts():
            found.extend(_findings(name, body))
        assert found == [], "\n".join(found)

    def test_every_manifest_row_that_was_scrubbed_says_so(self):
        """The module's own rule: bytes that are not raw CLI output say so.

        A row that needed a replacement carries ``synthesized: true`` and a
        ``provenance:`` naming what moved — either a ``SCRUB ONLY:`` list of the
        path substitutions, or, for the five rows composed from the ``success``
        capture, the fields that were edited.
        """
        import yaml

        manifest = yaml.safe_load((FIXTURES / "manifest.yaml").read_text(encoding="utf-8"))
        for name, row in manifest["rows"].items():
            if not row.get("synthesized"):
                continue
            provenance = (row.get("provenance") or "").strip()
            assert provenance, f"{name}: synthesized with no provenance"
            assert provenance.startswith(("SCRUB ONLY:", "COMPOSED")), (
                f"{name}: provenance must name the scrub or the composition, "
                f"got {provenance[:60]!r}"
            )

    @pytest.mark.parametrize("label,vector", [
        ("an OpenRouter key", "sk-or-v1-" + "a1b2c3d4e5f6a7b8" * 2),
        ("a generation id", "gen-1789673379-" + "Ab3" * 8),
        ("a foreign home", "/home/somebodyelse/.omp/agent/.env"),
        ("an agent scratch path", "/tmp/claude-1000/-home-x/scratchpad/probe"),
        ("a redaction marker", "<redacted>"),
    ])
    def test_the_sweep_catches_a_planted_vector(self, tmp_path, label, vector):
        """The sweep cannot go vacuous.

        Each vector is assembled at run time and never committed — the same rule
        ``tests/test_no_committed_secrets.py`` follows for its own planted
        credentials, because a guard whose test data is a real-looking secret in
        the tree is the one file nobody scans.
        """
        clean = (FIXTURES / "success.stdout.txt").read_text(encoding="utf-8")
        assert _findings("clean", clean) == [], "the control is not clean"
        planted = clean.replace('"cwd":"/work"', f'"cwd":"{vector}"', 1)
        if planted == clean:  # the capture's cwd spelling moved
            planted = clean + json.dumps({"type": "notice", "text": vector}) + "\n"
        assert _findings("planted", planted), f"the sweep missed {label}"


# --------------------------------------------------------------------------- #
# MCP suppression — measured, not assumed
# --------------------------------------------------------------------------- #

_MCP_SERVER = """#!/usr/bin/env python3
import json, os, sys
LOG = os.environ["MCPSRV_LOG"]
with open(LOG, "a") as log:
    log.write("started\\n")
    log.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        method, mid = req.get("method"), req.get("id")
        if method == "initialize":
            res = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                   "serverInfo": {"name": os.environ["MCPSRV_NAME"], "version": "0.0.1"}}
        elif method == "tools/list":
            res = {"tools": [{"name": os.environ["MCPSRV_TOOL"],
                              "description": "probe tool",
                              "inputSchema": {"type": "object", "properties": {}}}]}
        elif method in ("resources/list", "prompts/list"):
            res = {"resources": [], "prompts": []}
        else:
            res = {}
        if mid is not None:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\\n")
            sys.stdout.flush()
"""


class _StubHandler(BaseHTTPRequestHandler):
    """One SSE chat-completion, and a record of every request body."""

    requests = []
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # pragma: no cover - silence the default logger
        pass

    def do_GET(self):  # noqa: N802
        self._json({"object": "list", "data": [{"id": "stub-model", "object": "model"}]})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(body)
        frames = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"content": "pong"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        payload_bytes = b"".join(
            b"data: " + json.dumps(dict(f, id="c", object="chat.completion.chunk",
                                        created=0, model="stub-model")).encode() + b"\n\n"
            for f in frames
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(payload_bytes)

    def _json(self, obj):
        blob = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


@pytest.mark.skipif(shutil.which("omp") is None, reason="needs the omp binary")
class TestNoMcpServerMountsOnADispatch:
    """§9.2's ship gate, measured on this machine at dispatch time.

    The allow-list is the *entire* bound on the accepted ``--approval-mode
    yolo``, and MCP tools do not pass through it. So this runs the adapter's own
    argv against a loopback stub with a populated ``.cursor/mcp.json`` on both
    paths omp reads, and asserts the same three things branch (a) of the plan
    asks for: no server process started, no ``notice`` naming one, and no
    ``mcp__``-prefixed name in the tool list the request carried.

    It skips where omp is not installed, which is every CI machine; it is not
    marked ``live`` because it spends nothing and reaches no network but
    loopback.
    """

    @pytest.fixture
    def rig(self, tmp_path):
        home = tmp_path / "home"
        work = tmp_path / "work"
        for path in (home / ".cursor", work / ".cursor", home / ".omp" / "agent"):
            path.mkdir(parents=True, exist_ok=True)

        server_py = tmp_path / "mcpsrv.py"
        server_py.write_text(_MCP_SERVER, encoding="utf-8")

        logs = {}
        for scope, root in (("user", home), ("project", work)):
            log = tmp_path / f"mcp-{scope}.log"
            logs[scope] = log
            (root / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {
                f"{scope}level": {
                    "command": sys.executable,
                    "args": [str(server_py)],
                    "env": {"MCPSRV_NAME": f"{scope}level",
                            "MCPSRV_TOOL": f"{scope}_probe",
                            "MCPSRV_LOG": str(log)},
                },
            }}), encoding="utf-8")

        _StubHandler.requests = []
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        from tests.container.stub_server import LoopbackHTTPServer

        server = LoopbackHTTPServer(("127.0.0.1", port), _StubHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        (home / ".omp" / "agent" / "models.yml").write_text(json.dumps({"providers": {
            "stub": {
                "baseUrl": f"http://127.0.0.1:{port}/v1",
                "api": "openai-completions",
                "apiKey": "stub-probe-key",
                "models": [{"id": "stub-model", "name": "Stub", "contextWindow": 128000,
                            "maxTokens": 4096}],
            },
        }}), encoding="utf-8")

        # The natives cache is 346 MB and is extracted per HOME on the first
        # AGENT run. Reuse the developer's own rather than paying for it here.
        real_natives = Path(os.path.expanduser("~")) / ".omp" / "natives"
        if real_natives.is_dir():
            (home / ".omp" / "natives").symlink_to(real_natives)

        try:
            yield home, work, logs
        finally:
            server.shutdown()

    def _dispatch(self, home, work, extra_env=None):
        provider = OmpProvider()
        argv, stdin_text = provider.build_command_for_transport(
            "say pong", "stub/stub-model", "stdin", timeout_s=60, cwd=str(work),
        )
        env = build_child_env(
            provider,
            provider_env=provider.resolve_env(
                "stub/stub-model", PROFILE_GENERATE, cwd=str(work)
            ),
        )
        env["HOME"] = str(home)
        env.update(extra_env or {})
        return subprocess.run(
            argv, input=stdin_text, capture_output=True, text=True,
            env=env, cwd=str(work), timeout=180,
        )

    @staticmethod
    def _events(stdout):
        out = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def test_the_control_shows_the_lever_is_load_bearing(self, rig):
        """Without the overlay a project server really does start.

        A suppression test whose positive control never fires is a test that
        would pass against a machine where MCP was simply unavailable.
        """
        home, work, logs = rig
        provider = OmpProvider()
        argv = provider.build_command(
            "say pong", "stub/stub-model", timeout_s=60, cwd=str(work),
        )
        # The same argv minus the overlay, and minus the trailing `-- <prompt>`
        # so the prompt can travel on stdin as it does in production.
        i = argv.index("--config")
        bare = argv[:i] + argv[i + 2:-2]
        env = build_child_env(
            provider,
            provider_env=provider.resolve_env(
                "stub/stub-model", PROFILE_GENERATE, cwd=str(work)
            ),
        )
        env["HOME"] = str(home)
        subprocess.run(bare, input="say pong", capture_output=True, text=True,
                       env=env, cwd=str(work), timeout=180)

        assert logs["project"].exists(), (
            "the project .cursor/mcp.json did not mount even WITHOUT the "
            "overlay, so this machine cannot measure the suppression at all"
        )

    def test_nothing_mounts_under_the_shipped_overlay(self, rig):
        home, work, logs = rig
        result = self._dispatch(home, work)

        assert result.returncode == 0, result.stderr[:400]
        events = self._events(result.stdout)
        assert events, "the dispatch produced no stream to read"

        # 1. No server process started, on either path omp reads.
        for scope, log in logs.items():
            assert not log.exists(), f"the {scope}-level MCP server started"

        # 2. Nothing announced itself in the stream.
        notices = [e for e in events if e.get("type") == "notice"]
        assert notices == [], notices

        # 3. No `mcp__`-prefixed tool reached the request the model was sent,
        #    which is the check that matters: the allow-list filters built-ins
        #    only, so a mounted MCP tool would be an unbounded `yolo`.
        tools = []
        for body in _StubHandler.requests:
            for tool in body.get("tools") or []:
                tools.append((tool.get("function") or {}).get("name"))
        assert tools, "the request carried no tools at all"
        assert not [t for t in tools if (t or "").startswith("mcp")], tools
        # And the built-ins really are the allow-list's — six names, six
        # schemas, since `find` (an ALIAS of `glob` at 18.2.2, a separate
        # default-off tool at 18.2.9) left the list.
        assert set(t for t in tools if t) <= set(OMP_TOOLS), tools

    def test_the_user_level_config_does_not_mount_even_from_home(self, rig):
        """`--allow-home` is the flag that mounts it, and is never rendered.

        The measurement that put this hazard in the plan saw the user-level
        `~/.cursor/mcp.json` mount with `--allow-home` from `cwd = ~`. This
        adapter renders `--cwd` instead, so the case reachable from here is the
        cwd BEING the home directory — and it still does not mount.
        """
        home, _work, logs = rig
        result = self._dispatch(home, home)

        assert result.returncode == 0, result.stderr[:400]
        assert not logs["user"].exists(), "the user-level MCP server started"
        # `--cwd` is also what keeps the run out of /tmp: without it, omp
        # relocates silently when its cwd is the user's home.
        session = next(e for e in self._events(result.stdout) if e.get("type") == "session")
        assert session["cwd"] == str(home), session


#: The one marker omp's `--tools` validation leaves on stderr. It is MATCHED and
#: never parsed: the "Valid tools:" list that same sentence carries is
#: incomplete and changes between runs (plans/add-omp-provider.md §1.4), so the
#: only question it can answer reliably is "was THIS name rejected?" — which is
#: why the sweep below asks it one name at a time.
_UNKNOWN_TOOL = "Unknown tool"

#: A model selector no catalog can resolve. `--tools` is validated BEFORE model
#: resolution, so every run below dies at the gate being measured and never
#: reaches a provider; the credential is dropped from the child as well, so a
#: name that PASSES the gate cannot spend either.
_UNRESOLVABLE_MODEL = "multi-llm-no-such-provider/no-such-model"


@pytest.mark.skipif(shutil.which("omp") is None, reason="needs the omp binary")
class TestTheAllowListIsWhatTheInstalledBinaryAccepts:
    """`OMP_TOOLS` against the omp installed on this machine, one name per spawn.

    omp auto-updates and its tool registry moves with it. `--tools` is validated
    before model resolution and before the session opens, so a name the
    installed build does not know is **0 B on stdout, zero requests** and a
    refusal on stderr (exit 1 and a `CliUsageError` at 18.2.2, exit 2 and
    `Error: Unknown tool in --tools: …` at 18.8.3, measured 2026-10-08) — which
    is to say *every* omp dispatch on
    that machine is dead at launch, at zero spend, recorded as `spawn_failed`
    (D3k rung 1: fails the whole batch, never retried). That is the right
    disposition and the wrong-looking failure: the run never started, so nothing
    about containment was exercised, yet `TestNoMcpServerMountsOnADispatch`
    above goes red exactly as if a server had mounted.

    18.2.2 → 18.2.9 did precisely that. `find` was an alias of `glob` (six
    schemas under seven names) and became a separate, default-off
    (`find.enabled: false`), model-backed semantic-grep tool that `--tools`
    rejects outright — a minor-version bump of an auto-updating CLI that broke
    every dispatch and read, from the test names, like a containment breach.

    So the vocabulary is checked the only way that is honest: the adapter's own
    production argv, with the `--tools` value replaced by the single name under
    test, run against the real binary. The check is the free pre-flight itself —
    nothing resolves a model, the child carries no `OPENROUTER_API_KEY`, and no
    model call is possible whatever the verdict.
    """

    def _preflight(self, tools, cwd):
        """The production argv with `--tools` narrowed to `tools`."""
        provider = OmpProvider()
        argv, stdin_text = provider.build_command_for_transport(
            "x", _UNRESOLVABLE_MODEL, "stdin", timeout_s=60, cwd=str(cwd),
        )
        argv = list(argv)
        argv[argv.index("--tools") + 1] = tools
        env = build_child_env(
            provider,
            provider_env=provider.resolve_env(
                _UNRESOLVABLE_MODEL, PROFILE_GENERATE, cwd=str(cwd)
            ),
        )
        env.pop(OMP_CREDENTIAL_ENV, None)
        return subprocess.run(
            argv, input=stdin_text, capture_output=True, text=True,
            env=env, cwd=str(cwd), timeout=120,
        )

    @pytest.mark.parametrize("tool", OMP_TOOLS)
    def test_every_name_on_the_allow_list_is_one_this_build_knows(
        self, tool, tmp_path
    ):
        result = self._preflight(tool, tmp_path)

        assert _UNKNOWN_TOOL not in result.stderr, (
            f"the installed omp rejects `{tool}`, which OMP_TOOLS "
            f"(utils/providers/omp.py) puts on every dispatch's --tools: this "
            f"machine's omp dispatches all die at launch, before containment is "
            f"reached. Re-measure the vocabulary name by name and fix the "
            f"constant plus the prose in providers.yaml. "
            f"{result.stderr.strip()[-300:]}"
        )

    def test_a_name_this_build_does_not_know_is_rejected(self, tmp_path):
        """The positive control, so the sweep above cannot go vacuous.

        A machine where `--tools` stopped being validated at all would pass
        every row above while granting `yolo` over whatever omp's registry holds
        that day — which is the one thing the allow-list exists to prevent.
        """
        result = self._preflight("read,multi_llm_not_a_tool", tmp_path)

        assert _UNKNOWN_TOOL in result.stderr, (
            "omp accepted a tool name that does not exist, so `--tools` is no "
            f"longer the bound on `--approval-mode yolo`: {result.stderr[-400:]}"
        )
