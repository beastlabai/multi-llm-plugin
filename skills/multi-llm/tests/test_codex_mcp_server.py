"""The codex plugin's stdio MCP server, driven over its real wire.

``packaging/codex/mcp_server.py`` is the whole tool route on codex (plan
``plans/codex-native-plugin.md`` D3–D3f). Everything that *executes* it here
drives the **copy** inside a fixture plugin root — ``<root>/mcp/server.py`` beside
``<root>/.mcp.json``, ``<root>/dist/cli-surface.json`` and
``<root>/skills/multi-llm/`` — because the server resolves all three from its own
``__file__`` (D2), never from the process cwd, and each test starts it from an
unrelated cwd to prove that.

The dispatcher is a scripted fake ``uv`` (the ``test_opencode_package.py`` stub,
ported): it records its argv, cwd and environment, replays one scripted outcome
per child, and answers ``util instructions`` from a runbook fixture without
consuming a step. A small set of rows at the end runs the real skill through a
real ``uv`` (``--no-sync``, so the developer's ``.venv`` is never re-synced), and
skips with a reason when ``uv`` is absent.

Nothing here is ``package``/``container``/``live``: this file runs in plain CI.
Serial only (the suite is not xdist-safe).
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
SERVER_SRC = REPO_ROOT / "packaging" / "codex" / "mcp_server.py"

sys.path.insert(0, str(REPO_ROOT / "packaging"))
from common import surface as surface_mod  # noqa: E402

SURFACE = surface_mod.cli_surface("codex")
DISPATCHER = SURFACE["dispatcher"]
STATES = DISPATCHER["states"]
CODES = DISPATCHER["exit_codes"]
MARKER = DISPATCHER["result_marker"]
#: Schema v5's route signal, read from the surface like every other word here.
ROUTE_ENV = DISPATCHER["route_env"]

#: A2b's allowlist, as the fixture .mcp.json carries it.
ENV_VARS = [
    "CODEX_HOME", "MULTI_LLM_PROVIDERS_CONFIG", "MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE",
    "MULTI_LLM_GIT_CREDENTIAL_HELPER", "MULTI_LLM_HOST", "MULTI_LLM_UV",
    "MULTI_LLM_UV_RUN_ARGS", "MULTI_LLM_SKILL_DIR", "MULTI_LLM_MAX_DETACHED",
    "MULTI_LLM_HEARTBEAT_INTERVAL_S", "MULTI_LLM_HANDSHAKE_TIMEOUT_S",
    "MULTI_LLM_REPLAY_FLAGS", "MULTI_LLM_LEGACY_INSTRUCTIONS",
    "MULTI_LLM_CODEX_UNSANDBOXED_WRITES", "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR",
    "UV_OFFLINE", "PWD",
]
ENVELOPE_KEYS = {"ok", "state", "message", "exit", "anchor", "anchor_source",
                 "environment", "relays", "progress_lines"}
ENVIRONMENT_KEYS = {"platform", "shell", "login_shell", "profile_shell",
                    "interpreter", "missing"}
RELAY_KEYS = {"attempted", "delivered", "rejected", "unavailable", "logged"}


def _server_module_constants() -> dict:
    """The server's module-level literal constants, read by AST (no import)."""
    tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


CONST = _server_module_constants()
LOGIN_SHELL_SCRIPT = CONST["LOGIN_SHELL_SCRIPT"]
CANCEL_LATENCY_BOUND_S = CONST["CANCEL_LATENCY_BOUND_S"]
RUNNABLE_MARKER = CONST["RUNNABLE_MARKER"]
NEXT_MARKER = CONST["NEXT_MARKER"]


# ===========================================================================
# Fixture plugin roots
# ===========================================================================

def _mcp_json(cap=60, env_vars=None) -> str:
    entry = {"command": "python3", "args": ["./mcp/server.py"], "cwd": ".",
             "env_vars": list(ENV_VARS if env_vars is None else env_vars)}
    if cap is not None:
        entry["tool_timeout_sec"] = cap
    return json.dumps({"mcpServers": {"multi_llm": entry}}, indent=2)


def make_plugin_root(base: Path, *, cap=60, mcp_json=None, surface_text=None,
                     skill_dir: Path | None = None) -> Path:
    """A plugin root laid out as A2b builds one, with a fake payload by default."""
    root = base
    (root / ".codex-plugin").mkdir(parents=True, exist_ok=True)
    (root / ".codex-plugin" / "plugin.json").write_text(
        json.dumps({"name": "multi-llm", "version": "0.0.0-test"}), encoding="utf-8")
    (root / "mcp").mkdir(exist_ok=True)
    shutil.copyfile(SERVER_SRC, root / "mcp" / "server.py")
    if mcp_json is not False:
        (root / ".mcp.json").write_text(
            _mcp_json(cap) if mcp_json is None else mcp_json, encoding="utf-8")
    (root / "dist").mkdir(exist_ok=True)
    if surface_text is not False:
        (root / "dist" / "cli-surface.json").write_text(
            surface_mod.cli_surface_json("codex") if surface_text is None else surface_text,
            encoding="utf-8")
    skills = root / "skills"
    skills.mkdir(exist_ok=True)
    if skill_dir is not None:
        (skills / "multi-llm").symlink_to(skill_dir)
    else:
        payload = skills / "multi-llm"
        (payload / "hosts").mkdir(parents=True, exist_ok=True)
        (payload / DISPATCHER["script"]).write_text(
            "raise SystemExit('the fake payload is never run directly')\n",
            encoding="utf-8")
        for name in CONST["PAYLOAD_INVENTORY"]:
            (payload / name).write_text("# fake payload file\n", encoding="utf-8")
        shutil.copyfile(SKILL_ROOT / SURFACE["uv"]["missing_message_file"],
                        payload / SURFACE["uv"]["missing_message_file"])
    return root


@pytest.fixture(scope="session")
def plugin_root(tmp_path_factory) -> Path:
    return make_plugin_root(tmp_path_factory.mktemp("plugin-root"))


# ===========================================================================
# The scripted uv
# ===========================================================================

UV_STUB = r'''#!{python}
"""Replay one scripted dispatcher outcome per child, recording every argv."""
import json, os, signal, subprocess, sys, time

log = os.environ["STUB_LOG"]
argv = sys.argv[1:]
record = {{"argv": argv, "cwd": os.getcwd(), "pid": os.getpid(),
          "env": {{k: os.environ.get(k) for k in (
              "MULTI_LLM_UV", "MULTI_LLM_SKILL_DIR", "STUB_PROFILE_MARK",
              "PYTHONDONTWRITEBYTECODE", "MULTI_LLM_HOST_SESSION_MODEL",
              "MULTI_LLM_HOST_SANDBOX_POSTURE", {route_env_name!r})}}}}
dargv = argv[argv.index("--") + 3:] if "--" in argv else []
if dargv[:2] == ["util", "instructions"]:
    with open(log + ".instructions", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    text = open(os.environ["STUB_RUNBOOK"], encoding="utf-8").read()
    sys.stdout.write(text.replace("{{MODE}}", dargv[2]))
    sys.exit(0)
if dargv[:2] == {resume_argv!r}:
    # `util next-command <plan> --blocked <mode> --host <id>`: answered from the
    # environment, never from the script, so it takes no step of the run's own.
    with open(log + ".resume", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if os.environ.get("STUB_RESUME_FAIL"):
        sys.exit(1)
    # The real lookup answers only for a phase with something to resume:
    # `STUB_RESUME_ONLY` names the modes this one answers for (unset: all).
    only = os.environ.get("STUB_RESUME_ONLY")
    if only is not None and dargv[4] not in only.split(","):
        print({marker!r} + " " + json.dumps({{
            "mode": "util next-command", "state": "done", "exit": 0,
            "artifacts": [], "next": None}}))
        sys.exit(0)
    command = os.environ.get("STUB_RESUME_COMMAND") or (
        "$multi-llm:multi-llm --" + dargv[4] + " " + dargv[2] + " --resume")
    print({marker!r} + " " + json.dumps({{
        "mode": "util next-command", "state": "done", "exit": 0, "artifacts": [],
        "next": None, "user_resume": {{"phase": dargv[4], "command": command}}}}))
    sys.exit(0)
with open(log, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
with open(log, encoding="utf-8") as handle:
    index = sum(1 for _ in handle) - 1
script = json.load(open(os.environ["STUB_SCRIPT_FILE"], encoding="utf-8"))
step = script[index] if index < len(script) else script[-1]
if step.get("read_stdin"):
    data = sys.stdin.buffer.read()
    with open(log + ".stdin", "a", encoding="utf-8") as handle:
        handle.write(repr(data) + "\n")
if step.get("ignore_term"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if step.get("grandchild"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(log + ".grandchild", "a", encoding="utf-8") as handle:
        handle.write(str(child.pid) + "\n")
for line in step.get("stdout", []):
    print(line, flush=True)
if step.get("bulk"):
    chunk = "progress " + "x" * 100 + "\n"
    for _ in range(step["bulk"] // len(chunk) + 1):
        sys.stdout.write(chunk)
    sys.stdout.flush()
for line in step.get("after", []):
    print(line, flush=True)
for line in step.get("stderr", []):
    print(line, file=sys.stderr, flush=True)
time.sleep(step.get("sleep", 0))
sys.exit(step.get("exit", 0))
'''


def runbook_text(*, sentinel=True) -> str:
    """A ``util instructions`` stdout shaped like the dispatcher's own."""
    opening = SURFACE["runbook"]["banner_opening"]
    end = SURFACE["runbook"]["core_end_marker"]
    lines = [
        opening + " · multi-llm core instructions · {MODE} ===",
        "The core below is 999 characters. Its last line is exactly:",
        "",
        "    " + end + " {MODE}",
        "",
        "=== the core begins on the next line ===",
        "# Core",
        "WORKFLOW-STEP: read the validation tasks and report.",
        '   uv run --frozen --no-dev --project "/x/skill" -- python '
        '"/x/skill/multi_llm.py" status "$PLAN" --view run --wait \\',
        "     --run-id ABC",
        "3. Runs: uv run --project /x -- python /x/multi_llm.py review-plan p.md",
    ]
    if sentinel:
        lines.append(end + " {MODE}")
    lines += ["", "## Capability verbs", "VERB-APPENDIX: spawn, ask, detach.",
              MARKER + ' {"mode": "util instructions", "state": "done", "exit": 0, '
              '"artifacts": [], "next": null}']
    return "\n".join(lines) + "\n"


def result_line(**payload) -> str:
    body = {"mode": "status", "state": STATES["ok"], "exit": 0, "artifacts": [],
            "next": None}
    body.update(payload)
    return MARKER + " " + json.dumps(body)


def launch_step(run_id="RID-1", log="/tmp/rid.log", **extra) -> dict:
    step = {"exit": 0, "stdout": [
        DISPATCHER["banner"]["headline"],
        "run_id: " + run_id, "pid: 4242", "log: " + log,
        result_line(mode="review-plan", next=None)]}
    step.update(extra)
    return step


def poll_step(code, *, sleep=0, **payload) -> dict:
    state = {CODES["ok"]: STATES["ok"], CODES["nothing_to_do"]: STATES["running"],
             CODES["needs_human"]: STATES["needs_human"],
             CODES["error"]: STATES["error"]}[code]
    return {"exit": code, "sleep": sleep,
            "stdout": [result_line(state=state, exit=code, **payload)]}


ABORT_CONFIRMED = {"exit": 0, "stdout": ["aborted run", result_line(exit=0)]}

REVIEW = {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []}


def dispatcher_argv(call: dict) -> list:
    argv = call["argv"]
    return argv[argv.index("--") + 3:]


_ABSENT = object()


def turn_meta(workspaces=_ABSENT, sandbox_mode=_ABSENT, model="stub-model") -> dict:
    """A ``tools/call`` ``_meta`` shaped like codex 0.156.1's (row 7's tap).

    ``workspaces`` is a list of paths (each becomes a key, in order) or any
    raw value to put there as is; either argument left out is left out.
    """
    metadata = {"codex_version": "0.156.1", "sandbox": "none", "session_id": "s",
                "thread_id": "s", "turn_id": "t", "turn_trigger": "exec"}
    if model is not _ABSENT:
        metadata["model"] = model
    if sandbox_mode is not _ABSENT:
        metadata["sandbox_mode"] = sandbox_mode
    if workspaces is not _ABSENT:
        if isinstance(workspaces, list):
            workspaces = {str(path): {"has_changes": False,
                                      "latest_git_commit_hash": "c42d568a"}
                          for path in workspaces}
        metadata["workspaces"] = workspaces
    return {"plugin_id": "multi-llm@beastlabai", "x-codex-turn-metadata": metadata}


# ===========================================================================
# The stdio client
# ===========================================================================

class Client:
    """A line-framed JSON-RPC peer: requests, notifications, and roots/list."""

    def __init__(self, root: Path, env: dict, cwd: Path, roots=None):
        self.roots = roots
        self.raw_lines = []
        self.bad_lines = []
        self.notifications = []
        self.server_requests = []
        self.responses = {}
        self.cond = threading.Condition()
        self.stderr_chunks = []
        self.next_id = 0
        self.proc = subprocess.Popen(
            [sys.executable, str(root / "mcp" / "server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(cwd), env=env)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._err = threading.Thread(target=self._read_err, daemon=True)
        self._err.start()

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            self.raw_lines.append(raw)
            try:
                message = json.loads(raw)
            except ValueError:
                self.bad_lines.append(raw)
                continue
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                self.bad_lines.append(raw)
                continue
            if "method" in message and "id" in message:
                self.server_requests.append(message)
                self._answer_server(message)
            elif "method" in message:
                with self.cond:
                    self.notifications.append(message)
                    self.cond.notify_all()
            else:
                with self.cond:
                    self.responses[json.dumps(message.get("id"))] = message
                    self.cond.notify_all()

    def _read_err(self):
        for chunk in iter(lambda: self.proc.stderr.read1(4096), b""):
            self.stderr_chunks.append(chunk)

    @property
    def stderr(self) -> str:
        return b"".join(self.stderr_chunks).decode("utf-8", "replace")

    def _answer_server(self, message):
        if message["method"] == "roots/list" and self.roots is not None:
            roots = [{"uri": Path(p).as_uri(), "name": "ws"} for p in self.roots]
            self.send({"jsonrpc": "2.0", "id": message["id"], "result": {"roots": roots}})
        else:
            self.send({"jsonrpc": "2.0", "id": message["id"],
                       "error": {"code": -32601, "message": "Method not found"}})

    def send_raw(self, data: bytes):
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def send(self, message: dict):
        self.send_raw((json.dumps(message) + "\n").encode("utf-8"))

    def request(self, method, params=None, request_id=None):
        if request_id is None:
            self.next_id += 1
            request_id = self.next_id
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        return request_id

    def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def response(self, request_id, timeout=60):
        key = json.dumps(request_id)
        with self.cond:
            ok = self.cond.wait_for(lambda: key in self.responses, timeout)
            assert ok, "no response to {} within {}s; stderr:\n{}".format(
                key, timeout, self.stderr)
            return self.responses[key]

    def has_response(self, request_id) -> bool:
        with self.cond:
            return json.dumps(request_id) in self.responses

    def initialize(self, version="2025-06-18"):
        caps = {"roots": {"listChanged": True}} if self.roots is not None else {}
        rid = self.request("initialize", {"protocolVersion": version,
                                          "capabilities": caps,
                                          "clientInfo": {"name": "t", "version": "0"}})
        answer = self.response(rid)
        self.notify("notifications/initialized")
        return answer

    def start_call(self, args, token=None, request_id=None, meta=None):
        params = {"name": SURFACE["tool_name"], "arguments": args}
        if token is not None or meta is not None:
            params["_meta"] = dict(meta or {})
        if token is not None:
            params["_meta"]["progressToken"] = token
        return self.request("tools/call", params, request_id=request_id)

    def envelope(self, request_id, timeout=60):
        answer = self.response(request_id, timeout)
        result = answer["result"]
        assert len(result["content"]) == 1 and result["content"][0]["type"] == "text"
        envelope = json.loads(result["content"][0]["text"])
        assert_envelope(envelope, result)
        return envelope, result

    def close(self, timeout=60):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait(timeout)
        self._reader.join(10)
        self._err.join(10)
        assert not self.bad_lines, "fd 1 carried non-JSON-RPC: {!r}".format(
            self.bad_lines)
        return code


def assert_envelope(envelope: dict, result: dict) -> None:
    """Rule 5's one shape, asserted on every call this file makes."""
    assert ENVELOPE_KEYS <= set(envelope), sorted(ENVELOPE_KEYS - set(envelope))
    assert result["isError"] is (not envelope["ok"])
    environment = envelope["environment"]
    assert set(environment) == ENVIRONMENT_KEYS
    assert isinstance(environment["platform"], str)
    assert isinstance(environment["interpreter"], str)
    assert isinstance(environment["login_shell"], bool)
    assert environment["profile_shell"] is None or isinstance(
        environment["profile_shell"], str)
    assert environment["missing"] == sorted(environment["missing"])
    if environment["shell"] is None:
        assert environment["login_shell"] is False
    else:
        assert all(isinstance(part, str) for part in environment["shell"])
        assert environment["shell"][1:3] == ["-lc", LOGIN_SHELL_SCRIPT]
    assert set(envelope["relays"]) == RELAY_KEYS
    if envelope["anchor"] is None:
        assert envelope["anchor_source"] is None
    else:
        assert os.path.isabs(envelope["anchor"])
        source = envelope["anchor_source"]
        # D3f's spellings: the MCP roots rung, codex's per-call turn metadata,
        # and an environment name; `property` is the rung held in reserve.
        assert source in ("roots", "turn-metadata", "property") or source.startswith(
            "env:")


# ===========================================================================
# The harness
# ===========================================================================

class Harness:
    def __init__(self, tmp_path: Path, root: Path):
        self.tmp = tmp_path
        self.root = root
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.cwd = tmp_path / "elsewhere"  # the server's own, unrelated cwd
        self.cwd.mkdir()
        self.stub = tmp_path / "uv-stub"
        self.stub.write_text(UV_STUB.format(
            python=sys.executable, route_env_name=ROUTE_ENV["name"],
            resume_argv=list(DISPATCHER["resume"]["argv"]), marker=MARKER),
            encoding="utf-8")
        self.stub.chmod(0o755)
        self.log = tmp_path / "argv.log"
        self.script = tmp_path / "script.json"
        self.runbook = tmp_path / "runbook.txt"
        self.runbook.write_text(runbook_text(), encoding="utf-8")
        self.clients = []

    def env(self, extra=None, *, pwd=True) -> dict:
        env = {
            "PATH": os.environ.get("PATH", os.defpath), "HOME": str(self.home),
            "LANG": "C.UTF-8", "SHELL": "/bin/bash", "TERM": "dumb",
            "USER": os.environ.get("USER", "u"), "LOGNAME": os.environ.get("USER", "u"),
            "MULTI_LLM_UV": str(self.stub), "STUB_LOG": str(self.log),
            "STUB_SCRIPT_FILE": str(self.script), "STUB_RUNBOOK": str(self.runbook),
        }
        if pwd:
            env["PWD"] = str(self.workspace)
        for key, value in (extra or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return env

    def client(self, script=(), *, extra_env=None, roots=None, root=None, pwd=True,
               initialize=True) -> Client:
        self.log.write_text("", encoding="utf-8")
        Path(str(self.log) + ".instructions").write_text("", encoding="utf-8")
        Path(str(self.log) + ".resume").write_text("", encoding="utf-8")
        self.script.write_text(json.dumps(list(script) or [{"exit": 0}]), encoding="utf-8")
        client = Client(root or self.root, self.env(extra_env, pwd=pwd), self.cwd,
                        roots=roots)
        self.clients.append(client)
        if initialize:
            client.initialize()
        return client

    def run(self, args, script=(), *, meta=None, **kwargs):
        """One call on a fresh server: ``(envelope, result, calls, client)``."""
        client = self.client(script, **kwargs)
        envelope, result = client.envelope(client.start_call(args, meta=meta))
        assert client.close() == 0
        return envelope, result, self.calls(), client

    def calls(self) -> list:
        return [json.loads(line) for line in
                self.log.read_text(encoding="utf-8").splitlines() if line]

    def instruction_calls(self) -> list:
        path = Path(str(self.log) + ".instructions")
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def resume_calls(self) -> list:
        path = Path(str(self.log) + ".resume")
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def cleanup(self):
        for client in self.clients:
            if client.proc.poll() is None:
                client.proc.kill()
                client.proc.wait(10)


@pytest.fixture
def harness(tmp_path, plugin_root):
    h = Harness(tmp_path, plugin_root)
    yield h
    h.cleanup()


def _pid_gone(pid: int, within: float = 5.0) -> bool:
    """True once *pid* is gone or a zombie."""
    until = time.monotonic() + within
    while time.monotonic() < until:
        try:
            state = Path("/proc/{}/status".format(pid)).read_text()
        except OSError:
            return True
        if re.search(r"^State:\s+Z", state, re.M):
            return True
        time.sleep(0.05)
    return False


def _wait_for_calls(harness, count: int, timeout: float = 20.0) -> None:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if len(harness.calls()) >= count:
            time.sleep(0.2)  # let that child print past its argv record
            return
        time.sleep(0.02)
    raise AssertionError("the stub never reached call {}".format(count))


# ===========================================================================
# Source-level rows (src)
# ===========================================================================

def _string_constants(tree, *, keys=True) -> list:
    """Every string literal in the server except docstrings.

    ``keys=False`` also leaves out the literals used only as a mapping key — a
    subscript (``surface["dispatcher"]["abort"]``), a ``.get()`` argument, or
    the key paths in ``_REQUIRED_SURFACE_PATHS`` — because reading the
    surface's ``abort`` *key* is not holding the word.
    """
    docstrings = set()
    if not keys:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "_REQUIRED_SURFACE_PATHS"
                    for t in node.targets):
                docstrings.update(id(c) for c in ast.walk(node.value)
                                  if isinstance(c, ast.Constant))
            if isinstance(node, ast.Subscript):
                inner = node.slice
                inner = getattr(inner, "value", inner) if not isinstance(
                    inner, ast.Constant) else inner
                if isinstance(inner, ast.Constant):
                    docstrings.add(id(inner))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get" and node.args
                  and isinstance(node.args[0], ast.Constant)):
                docstrings.add(id(node.args[0]))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings]


def _prose_fragments(clause: str, minimum: int = 24) -> list:
    stripped = re.sub(r"\{[a-z_]+\}", " ", clause)
    return [piece.strip() for piece in re.split(r"(?<=[.;:,])\s+", stripped)
            if len(piece.strip()) >= minimum]


#: The named exemptions (D3e): JSON-RPC vocabulary and log levels, which share
#: the word "error" with the dispatcher's state vocabulary; the degraded
#: definition's tool name; and the degraded state word, which cannot come from
#: a surface that could not be read.
EXEMPT_EQUAL = {
    "error": "JSON-RPC's error member and MCP's `error` log level",
    SURFACE["tool_name"]: "the degraded definition's literal name (rule 7)",
    STATES["needs_human"]: "DEGRADED_STATE: the degraded call's state word",
}


def test_server_source_parses_at_the_python_floor():
    src = SERVER_SRC.read_text(encoding="utf-8")
    ast.parse(src, feature_version=(3, 9))
    # Stdlib only: every import is a top-level stdlib module.
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib:
        assert imported - {"__future__"} <= set(stdlib), imported - set(stdlib)


def test_surface_schema_version_matches():
    assert CONST["SUPPORTED_SURFACE_VERSION"] == surface_mod.SURFACE_SCHEMA_VERSION
    assert SURFACE["schema_version"] == surface_mod.SURFACE_SCHEMA_VERSION
    assert isinstance(SURFACE["tool_name"], str) and SURFACE["tool_name"]
    assert isinstance(SURFACE["tool_description"], str) and SURFACE["tool_description"]
    assert set(SURFACE["argument_descriptions"]) == {"mode", "plan", "question",
                                                     "extra_flags"}
    assert all(SURFACE["argument_descriptions"].values())


def test_named_constants():
    assert CONST["CANCEL_LATENCY_BOUND_S"] == 10
    assert CONST["DEFAULT_TOOL_TIMEOUT_S"] == 900
    assert LOGIN_SHELL_SCRIPT == 'cd -- "$1" && shift && exec "$@"'


class TestNoWordOfItsOwn:
    def test_server_holds_no_word_of_its_own(self, harness):
        """(src) no surface word is a literal; (build) tools/list is the surface."""
        tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
        literals = _string_constants(tree)
        literal_set = set(_string_constants(tree, keys=False))

        words = set()
        for mode in SURFACE["modes"]:
            words.add(mode["name"])
            words.update(mode["aliases"])
        words.update(STATES.values())
        words.update(SURFACE["positional_kinds"].values())
        for word in sorted(words):
            if word in EXEMPT_EQUAL:
                continue
            assert word not in literal_set, "surface word {!r} is a literal".format(word)

        flags = set(SURFACE["baked_flags"][::2]) | {SURFACE["detach_flag"]}
        reserved = SURFACE["reserved_flags"]
        flags.update(reserved["exact"], reserved["named_exception"], reserved["prefixes"])
        flags.update(t for t in DISPATCHER["poll"] + DISPATCHER["abort"]
                     + DISPATCHER["instructions"] if t.startswith("-"))
        flags.update([DISPATCHER["host_option"], DISPATCHER["run_id_option"],
                      DISPATCHER["review_gate"]["flag"]])
        flags.update(DISPATCHER["mechanical_continuations"])
        flags.update(SURFACE["uv"]["default_run_args"])
        for flag in sorted(flags):
            pattern = re.compile(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])")
            hits = [lit for lit in literals if pattern.search(lit)]
            assert not hits, "flag {!r} is spelled in the server: {!r}".format(flag, hits)

        # The host id, as a word (``.codex-plugin`` is the host's layout, not it).
        assert not any(re.search(r"(?<![\w.-])" + re.escape(SURFACE["host"]) + r"(?![\w-])",
                                 lit) for lit in literals)
        # Whole identifiers the surface owns.
        owned = [
            DISPATCHER["script"], MARKER,
            DISPATCHER["banner"]["headline"], SURFACE["runbook"]["core_end_marker"],
            SURFACE["runbook"]["banner_opening"], SURFACE["uv"]["env"],
            SURFACE["uv"]["run_args_env"], SURFACE["uv"]["missing_message_file"],
            SURFACE["write_posture_refusal"]["lever_env"],
            DISPATCHER["session_model_env"],
            # Schema v5: neither the route variable's name nor its value.
            ROUTE_ENV["name"], ROUTE_ENV["value"],
            *SURFACE["uv"]["candidates"], *SURFACE["nested_exec_reasons"],
        ]
        # Routing words, by equality once surface-key reads are set aside.
        for word in [*DISPATCHER["poll"][:1], *DISPATCHER["instructions"],
                     *DISPATCHER["resume"]["argv"]]:
            assert word not in literal_set, "{!r} is spelled in the server".format(word)
        for word in owned:
            assert not any(word in lit for lit in literals if word not in EXEMPT_EQUAL), (
                "{!r} is spelled in the server".format(word))

        # Message templates and prose, by fragment.
        prose = [
            *SURFACE["needs_human_message"].values(), SURFACE["needs_human_option"],
            SURFACE["tool_description"], *SURFACE["argument_descriptions"].values(),
            SURFACE["write_posture_refusal"]["template"],
            SURFACE["write_posture_refusal"]["posture_unreadable"],
            SURFACE["anchor_unresolved"], SURFACE["shell_missing"],
            SURFACE["uninstall_command"], SURFACE["outcomes_line"],
            *(mode["summary"] for mode in SURFACE["modes"]),
            (SKILL_ROOT / SURFACE["uv"]["missing_message_file"]).read_text(),
        ]
        joined = "\n".join(literals)
        for clause in prose:
            for fragment in _prose_fragments(clause):
                assert fragment not in joined, "prose {!r} is a literal".format(fragment)

        # Exit codes: never compared against a literal 1/2/3 (lengths aside).
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                sides = [node.left, *node.comparators]
                if any(isinstance(side, ast.Call) and isinstance(side.func, ast.Name)
                       and side.func.id == "len" for side in sides):
                    continue
                for side in sides:
                    if isinstance(side, ast.Constant) and side.value in (1, 2, 3) and \
                            not isinstance(side.value, bool):
                        pytest.fail("an exit code is compared as a literal at line "
                                    "{}".format(node.lineno))

        # (build) tools/list, byte for byte.
        client = harness.client()
        tools = client.response(client.request("tools/list"))["result"]["tools"]
        assert len(tools) == 1
        tool = tools[0]
        assert tool["name"] == SURFACE["tool_name"]
        assert tool["description"] == SURFACE["tool_description"]
        props = tool["inputSchema"]["properties"]
        for name, text in SURFACE["argument_descriptions"].items():
            assert props[name]["description"] == text
        assert client.close() == 0

    def test_exemptions_are_used_exactly_as_named(self):
        """The degraded words each appear once, in their named constants."""
        assert CONST["DEGRADED_TOOL_NAME"] == SURFACE["tool_name"]
        assert CONST["DEGRADED_STATE"] == STATES["needs_human"]
        tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
        literals = _string_constants(tree)
        assert literals.count(SURFACE["tool_name"]) == 1
        assert literals.count(STATES["needs_human"]) == 1


class TestToolDefinition:
    def test_tool_definition_matches_the_generated_surface(self, harness):
        client = harness.client()
        page = client.response(client.request("tools/list"))["result"]
        assert "nextCursor" not in page
        schema = page["tools"][0]["inputSchema"]
        assert schema["type"] == "object"
        assert schema["required"] == ["mode", "plan", "question", "extra_flags"]
        assert schema["properties"]["extra_flags"]["type"] == "array"
        assert schema["properties"]["extra_flags"]["items"] == {"type": "string"}
        for name in ("mode", "plan", "question"):
            assert schema["properties"][name]["type"] == "string"
        # A cursor this server never issued gets the same single-tool page.
        again = client.response(client.request("tools/list", {"cursor": "abc"}))
        assert again["result"] == page
        assert client.close() == 0


# ===========================================================================
# The wire contract (build)
# ===========================================================================

def test_server_speaks_the_protocol(harness, tmp_path):
    client = harness.client(initialize=False, extra_env={"MULTI_LLM_UV": None,
                                                         "PATH": str(tmp_path)})
    for version in ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"):
        answer = client.response(client.request("initialize", {
            "protocolVersion": version, "capabilities": {}}))
        assert answer["result"]["protocolVersion"] == version
        assert set(answer["result"]["capabilities"]) >= {"tools", "logging"}
    answer = client.response(client.request("initialize", {
        "protocolVersion": "1999-01-01", "capabilities": {}}))
    assert answer["result"]["protocolVersion"] == "2025-06-18"
    client.notify("notifications/initialized")
    missing = client.response(client.request("discover"))
    assert missing["error"]["code"] == -32601
    assert client.response(client.request("ping"))["result"] == {}
    # A uv-missing call is a CallToolResult with isError, never a JSON-RPC error.
    answer = client.response(client.start_call(REVIEW))
    assert "error" not in answer
    assert answer["result"]["isError"] is True
    envelope = json.loads(answer["result"]["content"][0]["text"])
    assert envelope["ok"] is False
    assert client.close() == 0


class TestWireContract:
    def test_ids_keep_their_json_type(self, harness):
        client = harness.client()
        assert client.response(client.request("ping", request_id="7"))["id"] == "7"
        assert client.response(client.request("ping", request_id=7))["id"] == 7
        assert client.close() == 0

    def test_malformed_lines_are_answered_and_survived(self, harness):
        client = harness.client()
        client.send_raw(b"{not json\n")
        client.send_raw(b"Content-Length: 40\r\n")
        client.send_raw(json.dumps({"id": 3, "method": "ping"}).encode() + b"\n")
        client.send_raw(json.dumps({"jsonrpc": "2.0", "id": 4}).encode() + b"\n")
        client.send_raw(json.dumps({"jsonrpc": "1.0", "id": 5, "method": "ping"}).encode()
                        + b"\n")
        # A response object the server never asked for is dropped in silence.
        client.send({"jsonrpc": "2.0", "id": "never-asked", "result": {}})
        assert client.response(client.request("ping", request_id=6))["result"] == {}
        parse_errors = client.responses.get("null")
        assert parse_errors and parse_errors["error"]["code"] == -32700
        for rid in (3, 4, 5):
            assert client.response(rid)["error"]["code"] == -32600
        assert not client.has_response("never-asked")
        assert client.close() == 0

    def test_a_cancel_for_an_unknown_id_is_a_noop(self, harness):
        client = harness.client()
        client.notify("notifications/cancelled", {"requestId": 999})
        assert client.response(client.request("ping", request_id=1000))["result"] == {}
        assert not client.has_response(999)
        assert client.close() == 0

    def test_logging_set_level(self, harness):
        client = harness.client([launch_step(), poll_step(2), poll_step(0)])
        assert client.response(client.request("logging/setLevel", {"level": "error"}))[
            "result"] == {}
        bad = client.response(client.request("logging/setLevel", {"level": "loud"}))
        assert bad["error"]["code"] == -32602
        envelope, _ = client.envelope(client.start_call(REVIEW))
        assert envelope["relays"]["attempted"] == 2
        assert envelope["relays"]["logged"] == 0
        assert not [n for n in client.notifications
                    if n["method"] == "notifications/message"]
        assert client.close() == 0

    def test_unknown_tool_is_a_protocol_error(self, harness):
        client = harness.client()
        answer = client.response(client.request("tools/call", {"name": "nope",
                                                              "arguments": {}}))
        assert answer["error"]["code"] == -32602
        assert harness.calls() == []
        assert client.close() == 0

    def test_a_parked_call_answers_ping_and_refuses_a_second_call(self, harness):
        client = harness.client([launch_step(), poll_step(0, sleep=2)])
        first = client.start_call(REVIEW, request_id="first")
        _wait_for_calls(harness, 2)
        started = time.monotonic()
        assert client.response(client.request("ping"))["result"] == {}
        assert time.monotonic() - started < 1.5
        second = client.start_call(REVIEW, request_id="second")
        refused = client.response(second)
        assert refused["error"]["code"] == -32000
        assert '"first"' in refused["error"]["message"]
        # Cancelling the refused call's id leaves the parked call untouched.
        client.notify("notifications/cancelled", {"requestId": "second"})
        envelope, _ = client.envelope(first)
        assert envelope["ok"] is True and envelope["state"] == STATES["ok"]
        assert not [c for c in harness.calls() if DISPATCHER["abort"][1]
                    in dispatcher_argv(c)]
        assert client.close() == 0

    def test_a_cancel_mid_launch_poll_kills_the_group_and_aborts_once(self, harness):
        client = harness.client([launch_step(run_id="RID-C"),
                                 {"exit": 2, "sleep": 60, "grandchild": True},
                                 ABORT_CONFIRMED])
        rid = client.start_call(REVIEW)
        _wait_for_calls(harness, 2)
        poll_pid = harness.calls()[1]["pid"]
        grandchild = int(Path(str(harness.log) + ".grandchild").read_text().split()[0])
        started = time.monotonic()
        client.notify("notifications/cancelled", {"requestId": rid})
        envelope, result = client.envelope(rid)
        elapsed = time.monotonic() - started
        # Acted on well within the bound: the worker watches the cancel event
        # and kills at once (the elapsed time includes the abort dispatch).
        assert elapsed < CANCEL_LATENCY_BOUND_S / 2, elapsed
        assert _pid_gone(poll_pid) and _pid_gone(grandchild)
        aborts = [dispatcher_argv(c) for c in harness.calls()
                  if DISPATCHER["abort"][1] in dispatcher_argv(c)]
        assert len(aborts) == 1
        assert aborts[0][aborts[0].index(DISPATCHER["run_id_option"]) + 1] == "RID-C"
        assert envelope["abort_dispatched"] is True
        assert envelope["state"] == STATES["needs_human"] and result["isError"]
        assert harness.instruction_calls() == []  # a cancelled call fetches nothing
        assert client.close() == 0

    def test_a_cancel_mid_reattach_poll_sends_no_abort(self, harness):
        live = {"exit": 2, "stdout": ["log: /tmp/r.log", result_line(
            state=STATES["nothing_to_do"], exit=2,
            next="uv run -- python multi_llm.py status p.md --view run --wait "
                 "--run-id RID-77")]}
        client = harness.client([live, {"exit": 2, "sleep": 60}, ABORT_CONFIRMED])
        rid = client.start_call({"mode": "status", "plan": "p.md", "question": "",
                                 "extra_flags": ["--view", "run", "--run-id", "RID"]})
        _wait_for_calls(harness, 2)
        client.notify("notifications/cancelled", {"requestId": rid})
        envelope, _ = client.envelope(rid)
        assert envelope["abort_dispatched"] is False
        assert envelope["run_id"] == "RID-77"
        assert len(harness.calls()) == 2
        assert client.close() == 0

    def test_stdin_eof_cancels_the_call_in_flight_then_exits(self, harness):
        client = harness.client([launch_step(run_id="RID-E"), {"exit": 2, "sleep": 60},
                                 ABORT_CONFIRMED])
        client.start_call(REVIEW)
        _wait_for_calls(harness, 2)
        started = time.monotonic()
        assert client.close(timeout=CANCEL_LATENCY_BOUND_S + 20) == 0
        assert time.monotonic() - started < CANCEL_LATENCY_BOUND_S + 5
        aborts = [c for c in harness.calls() if DISPATCHER["abort"][1] in dispatcher_argv(c)]
        assert len(aborts) == 1

    def test_sigterm_with_stdin_closed_cancels_the_call_in_flight(self, harness):
        # Codex's own shape, measured by Phase 1 row 10 (codex 0.156.1): an
        # interrupted session closes stdin and SIGTERMs the server in the same
        # millisecond, with no notifications/cancelled. Before the handler the
        # default SIGTERM exit won that race and orphaned the poll child's
        # group; now it takes the EOF branch.
        client = harness.client([launch_step(run_id="RID-T"),
                                 {"exit": 2, "sleep": 60, "grandchild": True},
                                 ABORT_CONFIRMED])
        client.start_call(REVIEW)
        _wait_for_calls(harness, 2)
        poll_pid = harness.calls()[1]["pid"]
        grandchild = int(Path(str(harness.log) + ".grandchild").read_text().split()[0])
        started = time.monotonic()
        client.proc.send_signal(signal.SIGTERM)
        assert client.close(timeout=CANCEL_LATENCY_BOUND_S + 20) == 0
        assert time.monotonic() - started < CANCEL_LATENCY_BOUND_S + 5
        assert _pid_gone(poll_pid) and _pid_gone(grandchild)
        aborts = [c for c in harness.calls() if DISPATCHER["abort"][1] in dispatcher_argv(c)]
        assert len(aborts) == 1

    def test_the_login_shell_s_noise_never_reaches_fd_1(self, harness):
        (harness.home / ".profile").write_text(
            "echo PROFILE-STDOUT-NOISE\necho PROFILE-STDERR-NOISE >&2\n"
            "export STUB_PROFILE_MARK=profile-ran\n", encoding="utf-8")
        envelope, _, calls, client = harness.run(
            {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line()]}])
        assert "PROFILE-STDOUT-NOISE" in envelope["output"]
        assert calls[0]["env"]["STUB_PROFILE_MARK"] == "profile-ran"
        assert envelope["environment"]["login_shell"] is True
        assert "PROFILE-STDERR-NOISE" in envelope["stderr"]
        # harness.run's close() already asserted every fd-1 line was JSON-RPC:
        # the planted lines reached the client only inside the result text.
        assert not any(line.startswith(b"PROFILE-") for line in client.raw_lines)

    def test_the_child_s_stdin_is_devnull(self, harness):
        client = harness.client([{"exit": 0, "read_stdin": True, "sleep": 1,
                                  "stdout": [result_line()]}])
        rid = client.start_call({"mode": "status", "plan": "p.md", "question": "",
                                 "extra_flags": []})
        _wait_for_calls(harness, 1)
        assert client.response(client.request("ping"), timeout=5)["result"] == {}
        envelope, _ = client.envelope(rid)
        assert envelope["ok"] is True
        assert Path(str(harness.log) + ".stdin").read_text().strip() == "b''"
        assert client.close() == 0


# ===========================================================================
# Bad reads of the surface and the cap (D3e, rule 7)
# ===========================================================================

def _previous_version_surface() -> dict:
    """The current surface as the previous schema version shipped it.

    One version back, and without the keys that version bump added
    (``dispatcher.route_env`` and each mode row's ``optional_why``).
    """
    dispatcher = dict(DISPATCHER)
    dispatcher.pop("route_env")
    rows = [{k: v for k, v in row.items() if k != "optional_why"}
            for row in SURFACE["modes"]]
    return dict(SURFACE, schema_version=CONST["SUPPORTED_SURFACE_VERSION"] - 1,
                dispatcher=dispatcher, modes=rows)


class TestSurfaceBadReads:
    @pytest.mark.parametrize("spelling", ["wrong-version", "unparseable", "absent",
                                          "descriptions-array", "descriptions-string",
                                          "previous-version", "no-route-env",
                                          "empty-route-name", "no-route-value",
                                          "route-env-string"])
    def test_a_bad_surface_degrades_but_stays_listed(self, harness, tmp_path, spelling):
        if spelling == "wrong-version":
            bad = dict(SURFACE, schema_version=CONST["SUPPORTED_SURFACE_VERSION"] + 1)
            text = json.dumps(bad)
        elif spelling == "previous-version":
            # A v4 surface: the previous version, without schema v5's keys. A
            # v5 server refuses it rather than spawn with no route signal.
            text = json.dumps(_previous_version_surface())
        elif spelling in ("no-route-env", "empty-route-name", "no-route-value",
                          "route-env-string"):
            # The current version with an unusable route signal: refused whole,
            # never half-used (a spawn without it prints the file-shim spelling).
            dispatcher = dict(DISPATCHER)
            if spelling == "no-route-env":
                del dispatcher["route_env"]
            elif spelling == "empty-route-name":
                dispatcher["route_env"] = dict(ROUTE_ENV, name="")
            elif spelling == "no-route-value":
                dispatcher["route_env"] = {"name": ROUTE_ENV["name"]}
            else:
                dispatcher["route_env"] = ROUTE_ENV["name"]
            text = json.dumps(dict(SURFACE, dispatcher=dispatcher))
        elif spelling.startswith("descriptions-"):
            # A container of the wrong type passes the key-presence checks; it
            # must still degrade rather than crash Server construction.
            shape = (list(SURFACE["argument_descriptions"].values())
                     if spelling == "descriptions-array" else "mode plan question")
            text = json.dumps(dict(SURFACE, argument_descriptions=shape))
        elif spelling == "unparseable":
            text = "{not json"
        else:
            text = False
        root = make_plugin_root(tmp_path / "bad-root", surface_text=text)
        client = harness.client(root=root)
        tools = client.response(client.request("tools/list"))["result"]["tools"]
        assert [t["name"] for t in tools] == [SURFACE["tool_name"]]
        assert tools[0]["description"] == CONST["DEGRADED_DESCRIPTION"]
        assert all("description" not in p for p in tools[0]["inputSchema"][
            "properties"].values())
        envelope, result = client.envelope(client.start_call(REVIEW))
        assert result["isError"] and envelope["ok"] is False
        assert envelope["state"] == STATES["needs_human"]
        assert envelope["exit"] is None and harness.calls() == []
        if spelling == "wrong-version":
            assert str(CONST["SUPPORTED_SURFACE_VERSION"] + 1) in envelope["message"]
            assert str(CONST["SUPPORTED_SURFACE_VERSION"]) in envelope["message"]
        if spelling in ("no-route-env", "empty-route-name", "no-route-value",
                        "route-env-string"):
            assert "route_env" in envelope["message"]
        assert client.close() == 0
        assert "cli-surface.json" in client.stderr
        warnings = [n for n in client.notifications
                    if n["method"] == "notifications/message"
                    and n["params"]["level"] == "warning"]
        assert warnings

    @pytest.mark.parametrize("spelling", ["missing", "unparseable", "keyless"])
    def test_a_bad_cap_falls_back_to_900_with_a_warning(self, harness, tmp_path, spelling):
        mcp = {"missing": False, "unparseable": "{",
               "keyless": _mcp_json(cap=None)}[spelling]
        root = make_plugin_root(tmp_path / "cap-root", mcp_json=mcp)
        envelope, _, _, client = harness.run(
            {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line()]}], root=root)
        assert envelope["ok"] is True
        assert str(CONST["DEFAULT_TOOL_TIMEOUT_S"]) in client.stderr


# ===========================================================================
# Arguments (ported TestArguments)
# ===========================================================================

def _refused(harness, args, **kwargs):
    envelope, result, calls, _ = harness.run(args, **kwargs)
    assert result["isError"] and envelope["ok"] is False
    assert envelope["exit"] is None
    assert calls == [] and harness.instruction_calls() == []
    assert envelope["environment"]["shell"] is None
    return envelope


class TestArguments:
    @pytest.mark.parametrize("token", [
        "--host", "--ho", "--host=opencode", "--det", "--yes=1",
        "--if-not-running", "--max-det", "--internal-worker", "--internal-run-id=x"])
    def test_reserved_flags_are_refused_by_option_head(self, harness, token):
        """Every composed flag a call restates with another meaning — a bare or
        foreign `--host`, `--detach`, a detach control, `--internal-*` — is
        refused; the no-op restatements are `TestRestatedComposedFlags`'."""
        envelope = _refused(harness, dict(REVIEW, extra_flags=[token]))
        assert envelope["state"] == STATES["error"]
        assert token in envelope["message"]

    @pytest.mark.parametrize("token", ["--interactive", "-i", "--inter"])
    def test_interactive_is_refused_never_substituted(self, harness, token):
        envelope = _refused(harness, dict(REVIEW, extra_flags=[token]))
        assert "interactive" in envelope["message"]

    @pytest.mark.parametrize("args", [
        None, [], {"mode": 42, "plan": "p", "question": "", "extra_flags": []},
        {"mode": "", "plan": "p", "question": "", "extra_flags": []},
        {"mode": "reviewplan", "plan": "p", "question": "", "extra_flags": []},
        {"mode": "review-plan", "plan": None, "question": "", "extra_flags": []},
        {"mode": "review-plan", "plan": "p", "question": "", "extra_flags": "--force"},
        {"mode": "review-plan", "plan": "p", "question": "", "extra_flags": [1]},
        {"mode": "review-plan", "plan": " ", "question": "", "extra_flags": []},
        {"mode": "init", "plan": "p", "question": "", "extra_flags": []},
        {"mode": "review-plan", "plan": "p", "question": "why", "extra_flags": []},
        {"mode": "ask", "plan": "p", "question": "", "extra_flags": []},
        {"mode": "ask", "plan": "p", "question": "-x", "extra_flags": []},
        {"mode": "review-plan", "plan": "-p", "question": "", "extra_flags": []},
    ])
    def test_arguments_are_validated_before_any_spawn(self, harness, args):
        envelope = _refused(harness, args)
        assert envelope["state"] == STATES["error"]
        assert envelope["anchor"] == os.path.realpath(harness.workspace)

    def test_composed_argv_order_and_empties(self, harness):
        _, _, calls, _ = harness.run(
            {"mode": "init", "plan": "", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line(mode="init")]}])
        assert dispatcher_argv(calls[0]) == ["init"] + SURFACE["baked_flags"]
        _, _, calls, _ = harness.run(
            {"mode": "ask", "plan": "p.md", "question": "why?", "extra_flags": ["--x"]},
            [launch_step(), poll_step(0)])
        # The user's own spelling, not the anchor-resolved path (issue #15): the
        # child runs in the anchor, so a relative plan names the same file there.
        plan = "p.md"
        assert dispatcher_argv(calls[0]) == (
            ["ask", plan, "why?"] + SURFACE["baked_flags"] + [SURFACE["detach_flag"],
                                                             "--x"])

    @pytest.mark.parametrize("alias", ["review", "abort"])
    def test_an_alias_reaches_the_dispatcher_unexpanded(self, harness, alias):
        _, _, calls, _ = harness.run(
            {"mode": alias, "plan": "p.md", "question": "", "extra_flags": []},
            [launch_step(), poll_step(0)] if alias == "review"
            else [{"exit": 0, "stdout": [result_line()]}])
        assert dispatcher_argv(calls[0])[0] == alias

    def test_detach_only_for_detach_capable_modes(self, harness):
        _, _, calls, _ = harness.run(
            {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line()]}])
        assert SURFACE["detach_flag"] not in dispatcher_argv(calls[0])

    def test_argv_survives_shell_metacharacters(self, harness):
        nasty = "we ird'$(touch PWNED1)`touch PWNED2`;touch PWNED3\n\"x"
        args = {"mode": "ask", "plan": nasty + ".md", "question": "q " + nasty,
                "extra_flags": ["--flag with space", "$(touch PWNED4)"]}
        _, _, calls, _ = harness.run(args, [launch_step(), poll_step(0)])
        plan = nasty + ".md"
        assert dispatcher_argv(calls[0]) == (
            ["ask", plan, "q " + nasty] + SURFACE["baked_flags"]
            + [SURFACE["detach_flag"], "--flag with space", "$(touch PWNED4)"])
        for place in (harness.workspace, harness.cwd, harness.home, harness.tmp):
            assert not list(Path(place).glob("PWNED*"))

    def test_child_is_spawned_through_one_login_shell_at_one_site(self, harness, tmp_path):
        """Inverted from the reference: exactly one shell, one site, a constant -c."""
        src = SERVER_SRC.read_text(encoding="utf-8")
        tree = ast.parse(src)
        assert "shlex" not in src
        popens = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
        assert len(popens) == 1
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "shell":
                assert isinstance(node.value, ast.Constant) and node.value.value is False
        # And at run time: a recording bash first on PATH sees the pinned argv.
        bindir = tmp_path / "bin"
        bindir.mkdir()
        record = tmp_path / "bash-argv.log"
        wrapper = bindir / "bash"
        wrapper.write_text(
            "#!{}\nimport json, os, sys\n"
            "open({!r}, 'a').write(json.dumps(sys.argv) + '\\n')\n"
            "os.execv('/bin/bash', ['bash'] + sys.argv[1:])\n".format(
                sys.executable, str(record)), encoding="utf-8")
        wrapper.chmod(0o755)
        path = "{}:{}".format(bindir, os.environ.get("PATH", ""))
        envelope, _, _, _ = harness.run(
            REVIEW, [launch_step(), poll_step(2), poll_step(0)],
            extra_env={"PATH": path})
        anchor = os.path.realpath(harness.workspace)
        spawns = [json.loads(line) for line in record.read_text().splitlines() if line]
        assert len(spawns) == 3
        for argv in spawns:
            assert argv[1:5] == ["-lc", LOGIN_SHELL_SCRIPT, "bash", anchor]
        assert envelope["environment"]["shell"] == [
            str(wrapper), "-lc", LOGIN_SHELL_SCRIPT, "bash"]


# ===========================================================================
# uv preflight (ported TestUvPreflight)
# ===========================================================================

class TestUvPreflight:
    def test_resolution_order(self, harness, tmp_path):
        status = {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []}
        step = [{"exit": 0, "stdout": [result_line()]}]
        # An override that names a file wins.
        _, _, calls, _ = harness.run(status, step)
        assert calls[0]["env"]["MULTI_LLM_UV"] == str(harness.stub)
        # An override that is not a file falls through to PATH.
        bindir = tmp_path / "pathbin"
        bindir.mkdir()
        shutil.copyfile(harness.stub, bindir / "uv")
        (bindir / "uv").chmod(0o755)
        _, _, calls, _ = harness.run(status, step, extra_env={
            "MULTI_LLM_UV": str(tmp_path / "nope"),
            "PATH": "{}:{}".format(bindir, os.environ["PATH"])})
        assert calls[0]["env"]["MULTI_LLM_UV"] == str(bindir / "uv")
        # A PATH miss falls through to the surface's candidates.
        local = harness.home / ".local" / "bin"
        local.mkdir(parents=True)
        shutil.copyfile(harness.stub, local / "uv")
        (local / "uv").chmod(0o755)
        _, _, calls, _ = harness.run(status, step, extra_env={
            "MULTI_LLM_UV": None, "PATH": "/usr/bin:/bin"})
        assert calls[0]["env"]["MULTI_LLM_UV"] == str(local / "uv")
        assert calls[0]["env"]["MULTI_LLM_SKILL_DIR"] == str(
            harness.root / "skills" / "multi-llm")

    @pytest.mark.parametrize("override,expected", [
        (None, None), ("--no-sync  --offline", ["--no-sync", "--offline"]),
        ("   ", None), ("\t--no-sync\n", ["--no-sync"])])
    def test_the_run_args_override_replaces_the_surface_list(self, harness, override,
                                                             expected):
        _, _, calls, _ = harness.run(
            {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line()]}],
            extra_env={SURFACE["uv"]["run_args_env"]: override})
        argv = calls[0]["argv"]
        between = argv[1:argv.index("--project")]
        assert argv[0] == "run"
        assert between == (expected or SURFACE["uv"]["default_run_args"])

    def test_a_missing_uv_refuses_with_the_payload_s_message(self, harness, tmp_path):
        envelope = _refused(harness, REVIEW, extra_env={
            "MULTI_LLM_UV": None, "PATH": str(tmp_path)})
        expected = (SKILL_ROOT / SURFACE["uv"]["missing_message_file"]).read_text(
            encoding="utf-8").rstrip("\n")
        assert envelope["message"] == expected
        assert envelope["state"] == STATES["needs_human"]


def test_server_starts_without_uv_and_lists_the_tool(harness, tmp_path):
    client = harness.client(extra_env={"MULTI_LLM_UV": None, "PATH": str(tmp_path)})
    tools = client.response(client.request("tools/list"))["result"]["tools"]
    assert [t["name"] for t in tools] == [SURFACE["tool_name"]]
    answer = client.response(client.start_call(REVIEW))
    assert answer["result"]["isError"] is True
    envelope = json.loads(answer["result"]["content"][0]["text"])  # JSON, not prose
    assert envelope["message"] == (SKILL_ROOT / SURFACE["uv"][
        "missing_message_file"]).read_text(encoding="utf-8").rstrip("\n")
    assert client.close() == 0


class TestPartialPayload:
    """D5's health contract row 2 and rule 7, through the server: each required
    payload file deleted in turn is a pre-spawn tool error naming its absolute
    path, and ``tools/list`` still answers with the one tool every time."""

    @pytest.mark.parametrize("name", [DISPATCHER["script"],
                                      *CONST["PAYLOAD_INVENTORY"]])
    def test_partial_payload_fails_by_name(self, harness, tmp_path, name):
        root = make_plugin_root(tmp_path / "partial-root")
        target = root / "skills" / "multi-llm" / name
        target.unlink()
        client = harness.client(root=root)
        tools = client.response(client.request("tools/list"))["result"]["tools"]
        assert [t["name"] for t in tools] == [SURFACE["tool_name"]]
        assert tools[0]["description"] == SURFACE["tool_description"]
        envelope, result = client.envelope(client.start_call(REVIEW))
        assert result["isError"] is True and envelope["ok"] is False
        assert envelope["state"] == STATES["needs_human"]
        assert envelope["exit"] is None
        assert os.path.abspath(str(target)) in envelope["message"]
        assert harness.calls() == [] and harness.instruction_calls() == []
        # The tool is still listed after the refusal, too.
        tools = client.response(client.request("tools/list"))["result"]["tools"]
        assert [t["name"] for t in tools] == [SURFACE["tool_name"]]
        assert client.close() == 0

    def test_a_missing_lockfile_is_refused_even_under_no_sync(self, harness, tmp_path):
        """The ``MULTI_LLM_UV_RUN_ARGS`` override cannot route around the check."""
        root = make_plugin_root(tmp_path / "nosync-root")
        target = root / "skills" / "multi-llm" / "uv.lock"
        target.unlink()
        envelope, result, calls, _ = harness.run(
            REVIEW, root=root, extra_env={SURFACE["uv"]["run_args_env"]: "--no-sync"})
        assert result["isError"] and envelope["exit"] is None and calls == []
        assert str(target) in envelope["message"]

    def test_the_two_exemptions(self, harness, tmp_path):
        """``dist/cli-surface.json`` degrades (TestSurfaceBadReads owns the
        wording); ``mcp/server.py`` absent means nothing of ours runs at all."""
        root = make_plugin_root(tmp_path / "no-surface", surface_text=False)
        client = harness.client(root=root)
        tools = client.response(client.request("tools/list"))["result"]["tools"]
        assert [t["name"] for t in tools] == [SURFACE["tool_name"]]
        envelope, _ = client.envelope(client.start_call(REVIEW))
        assert envelope["ok"] is False and envelope["state"] == STATES["needs_human"]
        assert envelope["exit"] is None and harness.calls() == []
        assert client.close() == 0

        root = make_plugin_root(tmp_path / "no-server")
        (root / "mcp" / "server.py").unlink()
        proc = subprocess.run([sys.executable, str(root / "mcp" / "server.py")],
                              input=b"", capture_output=True, timeout=30,
                              cwd=str(harness.cwd), env=harness.env())
        assert proc.returncode != 0 and proc.stdout == b""


def test_server_resolves_its_skill_dir(harness, tmp_path):
    cached = tmp_path / "codex-home" / "plugins" / "cache" / "beastlabai" / "multi-llm" / "9.9.9"
    root = make_plugin_root(cached)
    _, _, calls, _ = harness.run(
        {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []},
        [{"exit": 0, "stdout": [result_line()]}], root=root)
    argv = calls[0]["argv"]
    project = argv[argv.index("--project") + 1]
    assert project == str(cached / "skills" / "multi-llm")
    assert (Path(project) / DISPATCHER["script"]).is_file()
    assert argv[argv.index("--") + 2] == str(Path(project) / DISPATCHER["script"])


# ===========================================================================
# Pre-spawn refusals: posture, shell
# ===========================================================================

class TestChildEnvironment:
    def test_the_child_writes_no_bytecode(self, harness):
        """D9 layer 2: no ``__pycache__`` in the plugin cache (row 7)."""
        _, _, calls, _ = harness.run(REVIEW, [launch_step(), poll_step(0)])
        assert calls and {c["env"]["PYTHONDONTWRITEBYTECODE"] for c in calls} == {"1"}

    def test_a_user_setting_is_left_alone(self, harness):
        _, _, calls, _ = harness.run(REVIEW, [launch_step(), poll_step(0)],
                                     extra_env={"PYTHONDONTWRITEBYTECODE": "x"})
        assert {c["env"]["PYTHONDONTWRITEBYTECODE"] for c in calls} == {"x"}

    SESSION_ENV = DISPATCHER["session_model_env"]

    def test_the_turn_s_model_reaches_the_launch_and_every_poll(self, harness):
        """The ladder's `session` rung; the detached worker inherits the launch's env."""
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(CODES["nothing_to_do"]), poll_step(0)],
            meta=turn_meta(model="gpt-5.5"))
        assert len(calls) == 3
        assert [c["env"][self.SESSION_ENV] for c in calls] == ["gpt-5.5"] * 3
        assert envelope["session_model"] == "gpt-5.5"

    @pytest.mark.parametrize("value", [
        _ABSENT, None, "", 7, ["gpt-5.5"], {"id": "gpt-5.5"}, "gpt-oss:20b",
        "gpt 5.5", " gpt-5.5", "gpt-5.5\n", "-gpt-5.5", "x" * 201,
    ], ids=["absent", "null", "empty", "number", "list", "object", "colon",
            "space", "leading-space", "newline", "leading-dash", "too-long"])
    def test_an_unusable_model_sets_nothing(self, harness, value):
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0)], meta=turn_meta(model=value))
        assert calls and {c["env"][self.SESSION_ENV] for c in calls} == {None}
        assert "session_model" not in envelope

    REFUSED = "refused (not a plain model id)"

    def test_a_refused_model_is_named_in_one_log_line(self, harness):
        """A colon id (``codex --oss``) falls back visibly, not silently."""
        _, _, _, client = harness.run(REVIEW, [launch_step(), poll_step(0)],
                                      meta=turn_meta(model="gpt-oss:20b"))
        lines = [line for line in client.stderr.splitlines() if self.REFUSED in line]
        assert len(lines) == 1
        assert '"gpt-oss:20b"' in lines[0]
        assert "falls back past the session rung" in lines[0]
        assert any('"gpt-oss:20b"' in n["params"]["data"] for n in client.notifications
                   if n["method"] == "notifications/message")
        _, _, _, client = harness.run(REVIEW, [launch_step(), poll_step(0)],
                                      meta=turn_meta(model="gpt-oss:20b"),
                                      extra_env={self.SESSION_ENV: "from-env"})
        assert "keeps the forwarded " + self.SESSION_ENV in client.stderr

    @pytest.mark.parametrize("value", [_ABSENT, None, "gpt-5.5"],
                             ids=["absent", "null", "usable"])
    def test_an_absent_or_usable_model_logs_nothing(self, harness, value):
        _, _, _, client = harness.run(REVIEW, [launch_step(), poll_step(0)],
                                      meta=turn_meta(model=value))
        assert self.REFUSED not in client.stderr

    def test_malformed_turn_metadata_sets_nothing(self, harness):
        _, _, calls, _ = harness.run(REVIEW, [launch_step(), poll_step(0)],
                                     meta={"x-codex-turn-metadata": "not json"})
        assert calls and {c["env"][self.SESSION_ENV] for c in calls} == {None}

    def test_the_live_model_beats_a_forwarded_one_and_absence_keeps_it(self, harness):
        forwarded = {self.SESSION_ENV: "from-env"}
        _, _, calls, _ = harness.run(REVIEW, [launch_step(), poll_step(0)],
                                     meta=turn_meta(model="gpt-5.5"), extra_env=forwarded)
        assert {c["env"][self.SESSION_ENV] for c in calls} == {"gpt-5.5"}
        _, _, calls, _ = harness.run(REVIEW, [launch_step(), poll_step(0)],
                                     meta=turn_meta(model=_ABSENT), extra_env=forwarded)
        assert {c["env"][self.SESSION_ENV] for c in calls} == {"from-env"}

    def test_the_model_is_read_per_call_and_never_outlives_its_turn(self, harness):
        client = harness.client([launch_step(), poll_step(0)] * 2)
        client.envelope(client.start_call(REVIEW, meta=turn_meta(model="gpt-5.5")))
        client.envelope(client.start_call(REVIEW, meta=turn_meta(model=_ABSENT)))
        assert client.close() == 0
        assert [c["env"][self.SESSION_ENV] for c in harness.calls()] == [
            "gpt-5.5", "gpt-5.5", None, None]


class TestTheChatPostureReachesTheDispatcher:
    """Plan codex-sandbox-unavailable step 6 (F5, 2026-09-30): the turn's own
    sandbox posture is exported per call as ``MULTI_LLM_HOST_SANDBOX_POSTURE``,
    beside the session model, so the dispatcher may treat a
    ``danger-full-access`` chat as consent for an uncontained write — below
    the plugin's Landlock jail, never instead of it. Only codex's own posture
    words are exported, and the variable never outlives its call."""

    POSTURE_ENV = CONST["HOST_SANDBOX_POSTURE_ENV"]

    def test_the_name_matches_the_dispatcher_s(self):
        from utils.providers.base import HOST_SANDBOX_POSTURE_ENV

        assert self.POSTURE_ENV == HOST_SANDBOX_POSTURE_ENV

    @pytest.mark.parametrize("mode", ["read-only", "workspace-write", "danger-full-access"])
    def test_the_turn_s_posture_reaches_the_launch_and_every_poll(self, harness, mode):
        _, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0)], meta=turn_meta(sandbox_mode=mode))
        assert calls and {c["env"][self.POSTURE_ENV] for c in calls} == {mode}

    @pytest.mark.parametrize("value", [
        _ABSENT, None, "", "DANGER-FULL-ACCESS", "external-sandbox", ["danger-full-access"], 1,
    ], ids=["absent", "null", "empty", "case", "unknown-word", "list", "number"])
    def test_an_unknown_posture_exports_nothing(self, harness, value):
        _, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0)], meta=turn_meta(sandbox_mode=value))
        assert calls and {c["env"][self.POSTURE_ENV] for c in calls} == {None}

    def test_a_forwarded_value_is_never_consent(self, harness):
        """Only the chat's own posture is consent: a value codex forwarded
        from its environment is removed when the turn names none."""
        _, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0)], meta=turn_meta(sandbox_mode=_ABSENT),
            extra_env={self.POSTURE_ENV: "danger-full-access"})
        assert calls and {c["env"][self.POSTURE_ENV] for c in calls} == {None}

    def test_the_posture_is_read_per_call_and_never_carried_over(self, harness):
        client = harness.client([launch_step(), poll_step(0)] * 2)
        client.envelope(client.start_call(
            REVIEW, meta=turn_meta(sandbox_mode="danger-full-access")))
        client.envelope(client.start_call(REVIEW, meta=turn_meta(sandbox_mode=_ABSENT)))
        assert client.close() == 0
        assert [c["env"][self.POSTURE_ENV] for c in harness.calls()] == [
            "danger-full-access", "danger-full-access", None, None]


class TestRefusals:
    def test_a_write_mode_is_refused_until_the_user_sets_the_lever(self, harness):
        lever = SURFACE["write_posture_refusal"]
        args = {"mode": "implement", "plan": "p.md", "question": "", "extra_flags": []}
        envelope = _refused(harness, args)
        assert envelope["message"] == lever["template"].replace(
            "{mode}", "implement").replace("{posture}", lever["posture_unreadable"])
        _refused(harness, args, extra_env={lever["lever_env"]: "true"})
        envelope, _, calls, _ = harness.run(
            args, [launch_step(), poll_step(0)],
            extra_env={lever["lever_env"]: lever["lever_value"]})
        assert envelope["ok"] is True and calls

    WRITE = {"mode": "implement", "plan": "p.md", "question": "", "extra_flags": []}

    def test_a_read_only_posture_refuses_and_names_it(self, harness):
        lever = SURFACE["write_posture_refusal"]
        envelope = _refused(harness, self.WRITE,
                            meta=turn_meta(sandbox_mode="read-only"))
        assert envelope["state"] == STATES["needs_human"]
        assert envelope["message"] == lever["template"].replace(
            "{mode}", "implement").replace("{posture}", "read-only")

    @pytest.mark.parametrize("mode", ["workspace-write", "danger-full-access"])
    def test_a_posture_that_may_write_allows_the_write_mode(self, harness, mode):
        envelope, _, calls, _ = harness.run(
            self.WRITE, [launch_step(), poll_step(0)],
            meta=turn_meta(sandbox_mode=mode))
        assert envelope["ok"] is True and calls

    @pytest.mark.parametrize("value", [
        _ABSENT, None, "", "READ-ONLY", "external-sandbox", ["read-only"], 1,
    ], ids=["absent", "null", "empty", "case", "unknown-word", "list", "number"])
    def test_an_unreadable_posture_fails_closed(self, harness, value):
        lever = SURFACE["write_posture_refusal"]
        envelope = _refused(harness, self.WRITE, meta=turn_meta(sandbox_mode=value))
        assert envelope["message"] == lever["template"].replace(
            "{mode}", "implement").replace("{posture}", lever["posture_unreadable"])

    @pytest.mark.parametrize("raw", ["not json", "[1, 2]", 7, None, ["x"]],
                             ids=["bad-json", "json-list", "number", "null", "list"])
    def test_malformed_turn_metadata_fails_closed_and_never_crashes(self, harness, raw):
        lever = SURFACE["write_posture_refusal"]
        envelope = _refused(harness, self.WRITE, meta={"x-codex-turn-metadata": raw})
        assert lever["posture_unreadable"] in envelope["message"]
        # And a read-only mode still runs, anchored on the next rung.
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0)], meta={"x-codex-turn-metadata": raw})
        assert envelope["ok"] is True and envelope["anchor_source"] == "env:PWD"

    def test_a_json_string_turn_metadata_is_read(self, harness):
        meta = turn_meta(sandbox_mode="read-only")
        meta["x-codex-turn-metadata"] = json.dumps(meta["x-codex-turn-metadata"])
        envelope = _refused(harness, self.WRITE, meta=meta)
        assert "is read-only." in envelope["message"]

    def test_the_lever_lifts_a_read_only_refusal(self, harness):
        lever = SURFACE["write_posture_refusal"]
        envelope, _, calls, _ = harness.run(
            self.WRITE, [launch_step(), poll_step(0)],
            meta=turn_meta(sandbox_mode="read-only"),
            extra_env={lever["lever_env"]: lever["lever_value"]})
        assert envelope["ok"] is True and calls
        _refused(harness, self.WRITE, meta=turn_meta(sandbox_mode="read-only"),
                 extra_env={lever["lever_env"]: "true"})

    def test_a_read_only_posture_never_refuses_a_read_only_mode(self, harness):
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0)],
            meta=turn_meta(sandbox_mode="read-only"))
        assert envelope["ok"] is True and calls

    def test_no_bash_is_a_named_refusal(self, harness, tmp_path):
        envelope = _refused(harness, REVIEW, extra_env={
            "PATH": str(tmp_path), "MLLM_TEST_FALLBACK_SHELL": str(tmp_path / "nobash")})
        assert envelope["message"] == (
            SURFACE["shell_missing"].replace("{shell}", "bash")
            .replace("{platform}", sys.platform).replace("{path}", str(tmp_path)))
        assert envelope["environment"]["login_shell"] is False


# ===========================================================================
# The state machine (ported TestExecuteStateMachine + D3c's new rows)
# ===========================================================================

class TestExecuteStateMachine:
    def test_run_id_comes_from_the_banner(self, harness):
        launch = launch_step(run_id="RID-42")
        launch["stdout"][-1] = result_line(
            mode="review-plan",
            next="uv run -- python multi_llm.py status p.md --view run --wait")
        envelope, _, calls, _ = harness.run(REVIEW, [launch, poll_step(0)])
        assert envelope["run_id"] == "RID-42"
        poll = dispatcher_argv(calls[1])
        assert poll[:1] == DISPATCHER["poll"][:1]
        assert poll[poll.index(DISPATCHER["run_id_option"]) + 1] == "RID-42"
        assert all(flag in poll for flag in DISPATCHER["poll"][1:])
        assert "--timeout" not in poll
        assert SURFACE["baked_flags"][0] in poll

    def test_launch_result_is_not_the_answer(self, harness):
        launch = launch_step()
        launch["stdout"][-1] = result_line(mode="review-plan",
                                           artifacts=["LAUNCH-ARTIFACT"])
        envelope, _, _, _ = harness.run(
            REVIEW, [launch, poll_step(0, artifacts=["REAL-ARTIFACT"])])
        assert envelope["artifacts"] == ["REAL-ARTIFACT"]
        assert envelope["launch_state"] == STATES["ok"]

    def test_poll_loop_exit_map(self, harness):
        envelope, result, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(2), poll_step(2), poll_step(0)])
        assert envelope["state"] == STATES["ok"] and not result["isError"]
        assert len(calls) == 4 and envelope["polls"] == 3
        assert envelope["exit"] == 0

        failed = {"exit": 3, "stdout": ["the run failed"], "stderr": ["trace"]}
        envelope, result, _, _ = harness.run(REVIEW, [launch_step(), poll_step(2), failed])
        assert envelope["state"] == STATES["needs_human"] and result["isError"]
        assert "the run failed" in envelope["output"] and "trace" in envelope["stderr"]
        assert envelope["options"][-1] == {"human": SURFACE["needs_human_option"]}
        assert envelope["exit"] == 3

        broken = {"exit": 1, "stderr": ["status: bad arguments"]}
        envelope, result, _, _ = harness.run(REVIEW, [launch_step(run_id="RID"), broken])
        assert envelope["state"] == STATES["error"] and result["isError"]
        assert envelope["run_id"] == "RID" and envelope["log"] == "/tmp/rid.log"
        assert envelope["abort_command"]["mode"] == DISPATCHER["abort"][0]
        assert envelope["abort_command"]["extra_flags"][-1] == "RID"

    def test_exit_2_on_a_foreground_child_is_nothing_to_do(self, harness):
        envelope, result, calls, _ = harness.run(
            {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []},
            [{"exit": 2, "stdout": [result_line(state=STATES["nothing_to_do"], exit=2)]}])
        assert envelope["ok"] is True and result["isError"] is False
        assert envelope["state"] == STATES["nothing_to_do"]
        assert len(calls) == 1

    def test_a_live_reattach_polls_to_the_end(self, harness):
        live = {"exit": 2, "stdout": ["log: /tmp/r.log", result_line(
            state=STATES["nothing_to_do"], exit=2,
            next="uv run -- python multi_llm.py status p.md --view run --wait "
                 "--run-id RID-77")]}
        envelope, _, calls, _ = harness.run(
            {"mode": "status", "plan": "p.md", "question": "",
             "extra_flags": ["--view", "run", "--wait", "--run-id", "RID"]},
            [live, poll_step(2), poll_step(0)])
        assert envelope["state"] == STATES["ok"] and envelope["run_id"] == "RID-77"
        assert len(calls) == 3
        poll = dispatcher_argv(calls[1])
        assert poll[poll.index(DISPATCHER["run_id_option"]) + 1] == "RID-77"

    @pytest.mark.parametrize("code", [1, 3])
    def test_a_run_id_matching_no_run_is_a_tool_error(self, harness, code):
        answer = {"exit": code, "stdout": ["no run matches RID-X",
                                           result_line(state=STATES["error"], exit=code)]}
        envelope, result, calls, _ = harness.run(
            {"mode": "status", "plan": "p.md", "question": "",
             "extra_flags": ["--run-id", "RID-X"]}, [answer])
        assert result["isError"] and len(calls) == 1

    def test_a_mechanical_next_is_launched_claimed_and_polled(self, harness):
        done = poll_step(0, next="uv run -- python multi_llm.py review-plan p.md "
                                 "--reaggregate --host codex --yes")
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(run_id="R1"), done, launch_step(run_id="R2"),
                     poll_step(0)])
        assert envelope["continuations"] == 1 and envelope["run_id"] == "R2"
        relaunch = dispatcher_argv(calls[2])
        assert "--reaggregate" in relaunch and SURFACE["detach_flag"] in relaunch
        assert relaunch.count(SURFACE["baked_flags"][0]) == 1

    def test_an_unfollowed_next_comes_back_as_tool_call_args(self, harness):
        done = poll_step(0, next="uv run --project /x -- python /x/multi_llm.py "
                                 "generate-tasks /w/p.md --host codex --yes")
        envelope, _, calls, _ = harness.run(REVIEW, [launch_step(), done])
        assert envelope["next"] == {"mode": "generate-tasks", "plan": "/w/p.md",
                                    "question": "", "extra_flags": []}
        assert len(calls) == 2
        assert "runbook" in envelope  # unfinished: a next the loop did not follow

    def test_notice_is_surfaced_and_never_followed(self, harness):
        done = poll_step(0, notice="consider: uv run -- python /x/multi_llm.py "
                                   "review-plan p.md --consolidate",
                         consolidation_recommended=True)
        envelope, _, calls, _ = harness.run(REVIEW, [launch_step(), done])
        assert len(calls) == 2
        assert envelope["result"]["consolidation_recommended"] is True
        assert RUNNABLE_MARKER in envelope["result"]["notice"]

    def test_validation_counts_reach_the_tool_result(self, harness):
        """§6 T6.1: the host's summary reports these, so the envelope must
        carry the poll line's counts through untouched."""
        counts = {"valid": 53, "invalid": 1,
                  "needs_human_decision": 3, "validation_failed": 0}
        done = poll_step(0, validation_counts=counts)
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), done])
        assert envelope["result"]["validation_counts"] == counts

    def test_a_launch_failing_after_its_banner_aborts_the_claimed_run(self, harness):
        launch = launch_step(run_id="RID-F", exit=1)
        envelope, _, calls, _ = harness.run(REVIEW, [launch, ABORT_CONFIRMED])
        assert envelope["abort_dispatched"] is True
        assert DISPATCHER["abort"][1] in dispatcher_argv(calls[1])

    def test_no_banner_is_refused_rather_than_adopting_the_newest(self, harness):
        launch = {"exit": 0, "stdout": [result_line(mode="review-plan")]}
        envelope, result, calls, _ = harness.run(REVIEW, [launch])
        assert result["isError"] and len(calls) == 1

    def test_nested_exec_reasons_are_tool_errors(self, harness):
        code = next(iter(SURFACE["nested_exec_reasons"]))
        refused = {"exit": 3, "stdout": ["[{}] refused".format(code)]}
        envelope, result, _, _ = harness.run(REVIEW, [launch_step(), refused])
        assert result["isError"]
        assert envelope["nested_exec"]["reason"] == code

    def test_a_pipe_buffer_of_output_is_drained(self, harness):
        bulky = {"exit": 0, "bulk": 300_000, "after": [result_line(artifacts=["x"])]}
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), bulky])
        assert envelope["ok"] is True and envelope["artifacts"] == ["x"]
        assert envelope["exit"] == 0 and envelope["stdout_truncated"] is True

    def test_the_whole_stream_is_scanned_for_the_last_result_line(self, harness):
        noisy = {"exit": 0, "stdout": [result_line(artifacts=["first"])] + [
            "noise"] * 50 + [result_line(artifacts=["last"])]}
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), noisy])
        assert envelope["artifacts"] == ["last"]

    def test_a_child_ignoring_sigterm_is_killed(self, harness):
        client = harness.client([launch_step(run_id="RID-T"),
                                 {"exit": 2, "sleep": 60, "ignore_term": True},
                                 ABORT_CONFIRMED])
        rid = client.start_call(REVIEW)
        _wait_for_calls(harness, 2)
        pid = harness.calls()[1]["pid"]
        client.notify("notifications/cancelled", {"requestId": rid})
        envelope, _ = client.envelope(rid, timeout=30)
        assert _pid_gone(pid) and envelope["abort_dispatched"] is True
        assert client.close() == 0


class TestBudgetHandoff:
    def test_the_handoff_keeps_the_run_alive_and_the_reattach_finishes_it(self, harness):
        running = poll_step(2, sleep=0.2)
        started = time.monotonic()
        envelope, result, calls, _ = harness.run(
            REVIEW, [launch_step(run_id="RID-H", log="/tmp/h.log")] + [running] * 50,
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1500"})
        assert time.monotonic() - started < 10
        # ok: the run is alive and re-attachable, so the result is no error.
        assert envelope["state"] == STATES["running"]
        assert envelope["ok"] is True and result["isError"] is False
        assert envelope["run_id"] == "RID-H" and envelope["log"] == "/tmp/h.log"
        assert envelope["abort_dispatched"] is False
        assert not [c for c in calls if DISPATCHER["abort"][1] in dispatcher_argv(c)]
        reattach = envelope["next"]
        assert reattach["mode"] == DISPATCHER["poll"][0] and reattach["question"] == ""
        flags = reattach["extra_flags"]
        assert flags[flags.index(DISPATCHER["run_id_option"]) + 1] == "RID-H"
        assert "runbook" in envelope

        # The re-attach, driven by exactly those arguments, polls THAT run.
        live = {"exit": 2, "stdout": ["log: /tmp/h.log", result_line(
            state=STATES["nothing_to_do"], exit=2,
            next="uv run -- python multi_llm.py status p.md --view run --wait "
                 "--run-id RID-H")]}
        envelope, _, calls, _ = harness.run(reattach, [live, poll_step(2), poll_step(0)])
        assert envelope["state"] == STATES["ok"] and envelope["run_id"] == "RID-H"
        for call in calls[1:]:
            argv = dispatcher_argv(call)
            assert argv[argv.index(DISPATCHER["run_id_option"]) + 1] == "RID-H"

    def test_a_chain_of_reattaches_never_aborts(self, harness):
        live = {"exit": 2, "stdout": [result_line(
            state=STATES["nothing_to_do"], exit=2,
            next="uv run -- python multi_llm.py status p.md --view run --wait "
                 "--run-id RID-K")]}
        args = {"mode": "status", "plan": "p.md", "question": "",
                "extra_flags": ["--view", "run", "--wait", "--run-id", "RID-K"]}
        for _ in range(2):
            envelope, _, calls, _ = harness.run(
                args, [live] + [poll_step(2, sleep=0.2)] * 20,
                extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1000"})
            assert envelope["state"] == STATES["running"]
            assert not [c for c in calls if DISPATCHER["abort"][1] in dispatcher_argv(c)]
            args = envelope["next"]

    def test_a_reattach_whose_first_wait_eats_the_budget_hands_off(self, harness):
        args = {"mode": "status", "plan": "p.md", "question": "",
                "extra_flags": ["--view", "run", "--wait", "--run-id", "RID-W"]}
        envelope, result, calls, _ = harness.run(
            args, [{"exit": 2, "sleep": 60}],
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1000"})
        assert envelope["state"] == STATES["running"]
        assert envelope["ok"] is True and result["isError"] is False
        assert envelope["run_id"] == "RID-W" and envelope["abort_dispatched"] is False
        flags = envelope["next"]["extra_flags"]
        assert flags[flags.index(DISPATCHER["run_id_option"]) + 1] == "RID-W"
        assert not [c for c in calls if DISPATCHER["abort"][1] in dispatcher_argv(c)]

    def test_the_cap_is_read_from_the_sibling_mcp_json(self, harness, tmp_path):
        root = make_plugin_root(tmp_path / "short-cap", cap=2)
        started = time.monotonic()
        envelope, _, _, _ = harness.run(
            REVIEW, [launch_step()] + [poll_step(2, sleep=0.2)] * 50, root=root,
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "60000"})  # only ever shortens
        elapsed = time.monotonic() - started
        assert envelope["state"] == STATES["running"]
        assert elapsed < 2 + 3, elapsed

    def test_a_poll_is_never_cut_at_the_cancellation_bound(self, harness):
        """A poll longer than CANCEL_LATENCY_BOUND_S runs to its own answer.

        Codex's ceiling is 0, so the dispatcher's --wait is 90 s; a poll killed
        at the cancellation bound would never return and never relay. The bound
        is met by the cancel-event watch instead (the cancel rows above).
        """
        slow = poll_step(2, sleep=CANCEL_LATENCY_BOUND_S + 1,
                         progress_lines=["SLOW-POLL-NEWS"])
        envelope, _, calls, client = harness.run(
            REVIEW, [launch_step(), slow, poll_step(0)])
        assert envelope["state"] == STATES["ok"] and envelope["polls"] == 2
        assert len(calls) == 3  # the slow poll was not killed and re-spawned
        messages = [n["params"]["data"] for n in client.notifications
                    if n["method"] == "notifications/message"]
        assert "SLOW-POLL-NEWS" in messages
        # Structurally: the bound times no child.
        tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
                and n.id == "CANCEL_LATENCY_BOUND_S" and isinstance(n.ctx, ast.Load)]
        assert uses == []

    def test_a_cancel_after_the_handoff_does_nothing(self, harness):
        client = harness.client([launch_step(run_id="RID-P")] + [poll_step(2, sleep=0.2)] * 50,
                                extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1000"})
        rid = client.start_call(REVIEW)
        envelope, _ = client.envelope(rid)
        count = len(harness.calls())
        client.notify("notifications/cancelled", {"requestId": rid})
        assert client.response(client.request("ping"))["result"] == {}
        time.sleep(0.3)
        assert len(harness.calls()) == count
        assert client.close() == 0


# ===========================================================================
# Progress (D3d)
# ===========================================================================

class TestProgressRelay:
    def _script(self):
        launch = launch_step()
        launch["stdout"][-1] = result_line(mode="review-plan", progress_lines=["ROSTER"])
        return [launch,
                poll_step(2, progress_lines=["ROSTER", "just finished: a"]),
                poll_step(2, progress_lines=["ROSTER", "just finished: b"]),
                poll_step(0, progress_lines=["ROSTER", "TERMINAL"])]

    def test_one_update_per_relay_point_on_both_channels(self, harness):
        client = harness.client(self._script())
        envelope, _ = client.envelope(client.start_call(REVIEW, token="tok"))
        progress = [n["params"] for n in client.notifications
                    if n["method"] == "notifications/progress"]
        logs = [n["params"] for n in client.notifications
                if n["method"] == "notifications/message"]
        assert [p["message"] for p in progress] == [
            "ROSTER", "just finished: a", "just finished: b"]
        assert [p["progress"] for p in progress] == [1, 2, 3]
        assert all(p["progressToken"] == "tok" for p in progress)
        assert [entry["data"] for entry in logs] == [p["message"] for p in progress]
        # Codex 0.156.1 forwards no notifications/progress (row 10), so the log
        # channel is the one that reaches it: `info`, under the default floor
        # (this client never sends logging/setLevel).
        assert {entry["level"] for entry in logs} == {"info"}
        assert {entry["logger"] for entry in logs} == {SURFACE["tool_name"]}
        assert envelope["relays"] == {"attempted": 3, "delivered": 3, "rejected": 0,
                                      "unavailable": 0, "logged": 3}
        assert envelope["progress_lines"] == ["ROSTER", "just finished: b"]
        assert client.close() == 0

    def test_no_token_is_counted_as_unavailable(self, harness):
        client = harness.client(self._script())
        envelope, _ = client.envelope(client.start_call(REVIEW))
        assert envelope["relays"] == {"attempted": 3, "delivered": 0, "rejected": 0,
                                      "unavailable": 3, "logged": 3}
        # Without a token the log channel still fires at every relay point.
        logs = [n["params"] for n in client.notifications
                if n["method"] == "notifications/message"]
        assert [(entry["level"], entry["data"]) for entry in logs] == [
            ("info", "ROSTER"), ("info", "just finished: a"),
            ("info", "just finished: b")]
        assert not [n for n in client.notifications
                    if n["method"] == "notifications/progress"]
        assert client.close() == 0

    def test_a_command_in_progress_is_redacted(self, harness):
        script = self._script()
        script[1] = poll_step(2, progress={"models": {"m": {"error": (
            "run uv run -- python /x/multi_llm.py review-plan p.md")}}})
        client = harness.client(script)
        client.envelope(client.start_call(REVIEW, token=1))
        text = json.dumps(client.notifications)
        assert "multi_llm.py review-plan" not in text and RUNNABLE_MARKER in text
        assert client.close() == 0


# ===========================================================================
# The runbook (TestInstructionsSurface)
# ===========================================================================

class TestInstructionsSurface:
    def test_an_unfinished_outcome_gets_the_scrubbed_runbook(self, harness):
        failed = {"exit": 3, "stdout": ["stopped"]}
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), failed])
        asked = harness.instruction_calls()
        assert len(asked) == 1
        assert dispatcher_argv(asked[0]) == list(DISPATCHER["instructions"]) + [
            "review-plan", DISPATCHER["host_option"], SURFACE["host"]]
        runbook = envelope["runbook"]
        assert SURFACE["runbook"]["banner_opening"] not in runbook
        assert "999 characters" not in runbook
        sentinel = SURFACE["runbook"]["core_end_marker"] + " review-plan"
        lines = runbook.split("\n")
        assert lines.count(sentinel) == 1
        assert lines.index(sentinel) < lines.index("VERB-APPENDIX: spawn, ask, detach.")
        assert "WORKFLOW-STEP: read the validation tasks and report." in lines
        assert "uv run" not in runbook and MARKER not in runbook
        assert "--run-id ABC" not in runbook
        assert "3. Runs: " + RUNNABLE_MARKER in lines

    def test_a_finished_run_gets_no_runbook(self, harness):
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), poll_step(0)])
        assert "runbook" not in envelope and harness.instruction_calls() == []

    def test_a_runbook_without_its_sentinel_leaves_the_run_result_alone(self, harness):
        harness.runbook.write_text(runbook_text(sentinel=False), encoding="utf-8")
        envelope, result, _, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0, next="uv run -- python x/multi_llm.py "
                                                       "generate-tasks p.md")])
        assert not result["isError"] and envelope["ok"] is True
        assert "runbook" not in envelope
        assert SURFACE["runbook"]["core_end_marker"] in envelope["runbook_error"]

    def test_a_refusal_fetches_nothing(self, harness):
        _refused(harness, dict(REVIEW, extra_flags=["--host"]))


# ===========================================================================
# Redaction (the redactCommand port, one row per call site)
# ===========================================================================

class TestRedaction:
    def test_no_write_safety_reuse(self):
        src = SERVER_SRC.read_text(encoding="utf-8")
        assert "write_safety" not in src and "redact_text" not in src

    def test_next_the_parsed_result_and_the_tails(self, harness):
        command = ('uv run --frozen --project "C:\\skill" -- python '
                   '"C:\\skill\\multi_llm.py" generate-tasks /w/p.md')
        escaped = json.dumps(command)[1:-1]
        done = poll_step(0, next=command, artifacts=["see " + command],
                         notice={"nested": "echo " + escaped})
        done["stdout"].insert(0, "human line: " + command)
        done["stderr"] = ["uv run -- python /x/multi_llm.py review-plan p.md"]
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), done])
        assert isinstance(envelope["next"], dict)
        assert envelope["next"]["mode"] == "generate-tasks"
        text = json.dumps({k: envelope[k] for k in ("result", "output", "stderr",
                                                    "artifacts")})
        assert "generate-tasks" not in text
        assert "multi_llm.py review-plan" not in text
        assert NEXT_MARKER in envelope["result"]["next"]
        assert NEXT_MARKER in envelope["output"]
        assert NEXT_MARKER in envelope["result"]["notice"]["nested"]
        assert MARKER not in envelope["output"]  # dropped: it parsed

    def test_the_exit_3_option_block_and_the_banner_lines(self, harness):
        launch = launch_step(run_id="RID-B")
        launch["stdout"][4:4] = [
            "status: uv run -- python /x/multi_llm.py status /w/p.md --run-id RID-B",
            "abort: uv run -- python /x/multi_llm.py status /w/p.md --abort --run-id RID-B"]
        blocked = {"exit": 3, "stdout": [
            "1) re-invoke with `uv run -- python /x/multi_llm.py review-plan /w/p.md "
            "--reaggregate` to retry", result_line(state=STATES["needs_human"], exit=3,
                                                  error="blocked")]}
        envelope, _, _, _ = harness.run(REVIEW, [launch, blocked])
        text = json.dumps(envelope)
        assert "multi_llm.py review-plan" not in text
        assert envelope["options"][0] == {"mode": "review-plan", "plan": "/w/p.md",
                                          "question": "", "extra_flags": ["--reaggregate"]}
        assert envelope["options"][-1] == {"human": SURFACE["needs_human_option"]}
        assert '"blocked"' in envelope["message"]

    def test_an_unparsed_result_line_stays_in_the_output(self, harness):
        broken = {"exit": 0, "stdout": [MARKER + " {not json"]}
        envelope, result, _, _ = harness.run(REVIEW, [launch_step(), broken])
        assert result["isError"] and MARKER in envelope["output"]


# ===========================================================================
# The route signal and the lifted next step (plans/next-phase-command.md T2, T8)
# ===========================================================================

sys.path.insert(0, str(SKILL_ROOT))
from utils import host_capabilities as _caps  # noqa: E402

#: Every next-phase spelling the capability data carries, read through the
#: accessor (never retyped): six ``(host, route)`` cells.
NEXT_COMMAND_CELLS = [
    (host_id, route)
    for host_id in _caps.host_ids()
    for route in _caps.next_command_routes(host_id)
]
PHASE_ARGS = "--generate-tasks /w/plans/p.md"


def next_step(command=None, *, complete=False, optional=()):
    """A ``user_next_step`` value shaped like the dispatcher's (§2.1)."""
    if complete:
        return {"phase": None, "command": None, "complete": True}
    step = {"phase": "generate-tasks", "command": command, "complete": False}
    if optional:
        step["optional_before"] = [dict(entry) for entry in optional]
    return step


class TestRouteSignal:
    def test_the_route_reaches_the_launch_every_poll_and_the_runbook(self, harness):
        """Every child carries ``route_env.name=route_env.value``, from the surface."""
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(CODES["nothing_to_do"]),
                     poll_step(CODES["needs_human"], error="blocked")])
        assert len(calls) == 3
        assert [c["env"][ROUTE_ENV["name"]] for c in calls] == [ROUTE_ENV["value"]] * 3
        runbook = harness.instruction_calls()
        assert runbook, "the unfinished outcome fetched no runbook"
        assert {c["env"][ROUTE_ENV["name"]] for c in runbook} == {ROUTE_ENV["value"]}
        assert envelope["state"] == STATES["needs_human"]

    def test_a_foreground_mode_carries_it_too(self, harness):
        _, _, calls, _ = harness.run(
            {"mode": "init", "plan": "", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line(mode="init")]}])
        assert [c["env"][ROUTE_ENV["name"]] for c in calls] == [ROUTE_ENV["value"]]

    def test_the_server_spells_neither_the_name_nor_the_value(self):
        tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
        literals = _string_constants(tree)
        for word in (ROUTE_ENV["name"], ROUTE_ENV["value"]):
            assert not any(word in lit for lit in literals), (
                "{!r} is spelled in the server; it must be read from the "
                "surface's dispatcher.route_env".format(word))

    def test_the_route_is_required_at_load(self):
        """``("dispatcher", "route_env")`` is a required path (T2)."""
        assert ("dispatcher", "route_env") in CONST["_REQUIRED_SURFACE_PATHS"]


class TestUserNextStep:
    """§2.4: lifted to the top level on exit 0 and exit 2, never followed."""

    def test_exit_0_lifts_it_beside_the_payload_copy(self, harness):
        why = "optional: have the models review the tasks before implementing"
        step = next_step("/multi-llm --implement /w/plans/p.md", optional=[{
            "phase": "review-tasks", "command": "/multi-llm --review-tasks /w/plans/p.md",
            "why": why}])
        envelope, result, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0, user_next_step=step)])
        assert not result["isError"]
        assert envelope["user_next_step"] == step
        assert envelope["result"]["user_next_step"] == step
        assert envelope.get("next") is None  # the envelope drops a null key
        assert "next_phase" not in json.dumps(envelope)
        assert len(calls) == 2  # launch + terminal poll: nothing followed

    def test_the_completed_form_is_lifted_too(self, harness):
        envelope, _, _, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0, user_next_step=next_step(complete=True))])
        assert envelope["user_next_step"] == {
            "phase": None, "command": None, "complete": True}

    def test_exit_2_of_a_phase_already_complete_lifts_it(self, harness):
        """The phase's own guard exits 2 before any detach: no banner, a result."""
        step = next_step("/multi-llm --generate-tasks /w/plans/p.md")
        already = {"exit": CODES["nothing_to_do"], "stdout": [result_line(
            mode="review-plan", state=STATES["nothing_to_do"],
            exit=CODES["nothing_to_do"], user_next_step=step)]}
        envelope, result, calls, _ = harness.run(REVIEW, [already])
        assert not result["isError"] and envelope["ok"] is True
        assert envelope["exit"] == CODES["nothing_to_do"]
        assert envelope["user_next_step"] == step
        assert "next_phase" not in json.dumps(envelope)
        assert len(calls) == 1

    def test_a_still_running_poll_carries_none(self, harness):
        """A poll's exit 2 means "still moving": no key arrives, none is lifted."""
        status = {"mode": DISPATCHER["poll"][0], "plan": "/w/p.md", "question": "",
                  "extra_flags": []}
        running = {"exit": CODES["nothing_to_do"], "stdout": [result_line(
            state=STATES["running"], exit=CODES["nothing_to_do"])]}
        envelope, _, _, _ = harness.run(status, [running])
        assert envelope["exit"] == CODES["nothing_to_do"]
        assert "user_next_step" not in envelope
        assert "user_next_step" not in envelope["result"]

    def test_an_absent_key_lifts_nothing(self, harness):
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), poll_step(0)])
        assert "user_next_step" not in envelope

    def test_exit_3_and_1_lift_nothing(self, harness):
        """Nothing is lifted from a failed payload: the top-level key is the
        explicit null a failed result carries (`TestNoCommandToCompose`)."""
        step = next_step("/multi-llm --implement /w/plans/p.md")
        for code in (CODES["needs_human"], CODES["error"]):
            envelope, _, _, _ = harness.run(
                REVIEW, [launch_step(), poll_step(code, user_next_step=step)])
            assert "user_next_step" in envelope and envelope["user_next_step"] is None, code

    def test_the_continuation_loop_never_follows_it(self, harness):
        """Zero follows for a result carrying only ``user_next_step`` — even one
        whose command carries a mechanical-continuation flag."""
        marker = DISPATCHER["mechanical_continuations"][0]
        step = next_step("/multi-llm --review-plan /w/plans/p.md " + marker)
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(), poll_step(0, user_next_step=step),
                     launch_step(run_id="RID-FOLLOWED")])
        assert len(calls) == 2
        assert "continuations" not in envelope
        assert envelope["user_next_step"] == step

    @pytest.mark.parametrize("host_id, route", NEXT_COMMAND_CELLS)
    def test_redaction_leaves_every_spelling_intact(self, harness, host_id, route):
        """The command is the host's own spelling, never a ``uv run`` line, so
        ``redact`` passes it through byte for byte: in the lifted copy, the
        payload copy and the human output."""
        command = _caps.next_command_template(host_id, route).replace(
            "{args}", PHASE_ARGS)
        done = poll_step(0, user_next_step=next_step(command))
        done["stdout"].insert(0, "[NEXT_COMMAND] " + command)
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), done])
        assert envelope["user_next_step"]["command"] == command
        assert envelope["result"]["user_next_step"]["command"] == command
        assert "[NEXT_COMMAND] " + command in envelope["output"].split("\n")
        assert RUNNABLE_MARKER not in json.dumps(envelope)


# ===========================================================================
# A blocked write phase hands the user a command they can type (issue #5)
# ===========================================================================

IMPLEMENT = {"mode": "implement", "plan": "plans/p.md", "question": "",
             "extra_flags": []}


class TestResumeCommand:
    """2026-10-02: a needs-human `implement` ended with the tool's JSON arguments
    and nothing the user could type. The dispatcher builds the host-spelled
    resume command (`util next-command --blocked`); this server only asks."""

    def _blocked(self, harness, args=IMPLEMENT, **kwargs):
        kwargs.setdefault("meta", turn_meta(sandbox_mode="workspace-write"))
        return harness.run(
            args, [launch_step(), poll_step(CODES["needs_human"], error="batch 0 refused")],
            **kwargs)

    def test_the_result_carries_it_and_the_message_says_how_to_use_it(self, harness):
        envelope, _, _, _ = self._blocked(harness)
        assert envelope["resume_command"] == \
            "$multi-llm:multi-llm --implement plans/p.md --resume"
        assert envelope["message"].endswith(SURFACE["needs_human_message"]["resume"])
        assert envelope.get("next") is None

    def test_it_asks_with_the_users_spelling_the_word_and_the_host(self, harness):
        _, _, _, _ = self._blocked(harness)
        (asked,) = harness.resume_calls()
        assert dispatcher_argv(asked) == list(DISPATCHER["resume"]["argv"]) + [
            "plans/p.md", DISPATCHER["resume"]["option"], "implement",
            DISPATCHER["host_option"], SURFACE["host"]]
        assert asked["env"][ROUTE_ENV["name"]] == ROUTE_ENV["value"]
        assert asked["cwd"] == os.path.realpath(harness.workspace)

    def test_it_is_never_followed_and_never_a_call(self, harness):
        envelope, _, calls, _ = self._blocked(harness)
        assert len(calls) == 2  # the launch and the poll: nothing re-ran
        assert "uv run" not in envelope["resume_command"]
        assert RUNNABLE_MARKER not in envelope["resume_command"]
        assert all("resume_command" not in option for option in envelope["options"])

    def test_the_agent_facing_option_does_not_read_as_a_command(self, harness):
        envelope, _, _, _ = self._blocked(harness)
        human = envelope["options"][-1]["human"]
        assert human == SURFACE["needs_human_option"]
        assert human.startswith("(a note to you, not a command")

    @pytest.mark.parametrize("code", [CODES["ok"], CODES["error"]])
    def test_only_an_exit_3_asks(self, harness, code):
        harness.run(IMPLEMENT, [launch_step(), poll_step(code)],
                    meta=turn_meta(sandbox_mode="workspace-write"))
        assert harness.resume_calls() == []

    def test_a_failed_lookup_leaves_the_result_as_it_was(self, harness):
        envelope, _, _, _ = self._blocked(harness, extra_env={"STUB_RESUME_FAIL": "1"})
        assert "resume_command" not in envelope
        assert not envelope["message"].endswith(SURFACE["needs_human_message"]["resume"])
        assert envelope["state"] == STATES["needs_human"]

    def test_the_server_spells_none_of_the_lookup(self):
        source = SERVER_SRC.read_text()
        for word in [*DISPATCHER["resume"]["argv"], DISPATCHER["resume"]["option"]]:
            assert '"{}"'.format(word) not in source, word

    def test_a_continuation_keeps_the_typed_spelling_of_the_plan(self, harness):
        """The dispatcher's `next` names the plan absolutely; the continuation of
        the same plan is launched with the spelling the user typed."""
        absolute = os.path.join(os.path.realpath(harness.workspace), "plans", "p.md")
        done = poll_step(0, next="uv run -- python multi_llm.py review-plan {} "
                                 "--reaggregate --host codex --yes".format(absolute))
        _, _, calls, _ = harness.run(
            dict(REVIEW, plan="plans/p.md"),
            [launch_step(run_id="R1"), done, launch_step(run_id="R2"), poll_step(0)])
        launches = [c for c in calls if dispatcher_argv(c)[0] != DISPATCHER["poll"][0]]
        assert [dispatcher_argv(c)[1] for c in launches] == ["plans/p.md", "plans/p.md"]
        polls = [c for c in calls if dispatcher_argv(c)[0] == DISPATCHER["poll"][0]]
        assert all(dispatcher_argv(c)[1] == absolute for c in polls)


# ===========================================================================
# Every phase, both ways a failure reaches the user: the plan as typed
# ===========================================================================

#: Every phase the surface knows, read from the surface (never retyped), so a
#: phase added later is covered the day it lands.
PHASE_ROWS = [row for row in SURFACE["modes"] if row.get("phase")]
TYPED_PLAN = "plans/p.md"


def phase_args(row) -> dict:
    """A call on *row* with the plan typed relative (and a question for ask)."""
    takes_question = SURFACE["positional_kinds"]["question"] in row["positionals"]
    return {"mode": row["name"], "plan": TYPED_PLAN,
            "question": "why?" if takes_question else "", "extra_flags": []}


class TestTheTypedPlanSurvivesEveryFailure:
    """Racing re-run 3 (2026-10-05, codex 06): implement outlived the call's
    budget, the re-attach `next` named the plan absolutely, the model sent it
    back as given, and the run's failure came back with
    `resume_command: $multi-llm:multi-llm --implement /tmp/…/plans/racing-game.md
    --resume` for a plan the user typed `plans/racing-game.md`. Every phase, the
    failure in the launching call and the failure in the re-attach."""

    META = turn_meta(sandbox_mode="workspace-write")

    def _absolute(self, harness) -> str:
        return os.path.join(os.path.realpath(harness.workspace), "plans", "p.md")

    def _failed_poll(self, harness, row) -> dict:
        """A failed run's terminal poll: status's own `resume with:` line names
        the plan absolutely, as `run_status` prints it."""
        return {"exit": CODES["needs_human"], "stdout": [
            "      resume with: {} {} {} --host {} --detach --yes --resume".format(
                UV_PREFIX, row["name"], self._absolute(harness), SURFACE["host"]),
            result_line(mode=row["name"], phase=row["phase"],
                        state=STATES["needs_human"], exit=CODES["needs_human"])]}

    def _assert_typed(self, harness, envelope, row):
        absolute = self._absolute(harness)
        assert TYPED_PLAN in envelope["resume_command"]
        assert absolute not in envelope["resume_command"]
        assert envelope["user_next_step"] is None
        (asked,) = harness.resume_calls()
        assert dispatcher_argv(asked)[2] == TYPED_PLAN
        for option in envelope["options"]:
            if "plan" in option:
                assert option["plan"] == TYPED_PLAN, option
        assert absolute not in envelope["output"]

    @pytest.mark.parametrize("row", PHASE_ROWS, ids=[row["name"] for row in PHASE_ROWS])
    def test_a_failure_in_the_launching_call(self, harness, row):
        envelope, _, _, _ = harness.run(
            phase_args(row), [launch_step(), self._failed_poll(harness, row)],
            meta=self.META)
        self._assert_typed(harness, envelope, row)

    @pytest.mark.parametrize("row", PHASE_ROWS, ids=[row["name"] for row in PHASE_ROWS])
    def test_a_failure_after_the_handoff_and_the_reattach(self, harness, row):
        handoff, _, _, _ = harness.run(
            phase_args(row),
            [launch_step(run_id="RID-T")] + [poll_step(2, sleep=0.2)] * 50,
            meta=self.META, extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1000"})
        assert handoff["state"] == STATES["running"]
        assert handoff["next"]["plan"] == TYPED_PLAN
        assert handoff["abort_command"]["plan"] == TYPED_PLAN

        live = {"exit": 2, "stdout": [result_line(
            state=STATES["nothing_to_do"], exit=2,
            next="uv run -- python multi_llm.py status {} --view run --wait "
                 "--run-id RID-T".format(self._absolute(harness)))]}
        envelope, _, calls, _ = harness.run(
            handoff["next"], [live, self._failed_poll(harness, row)], meta=self.META)
        # The re-attach still polls by the absolute path inside the server.
        assert dispatcher_argv(calls[-1])[1] == self._absolute(harness)
        self._assert_typed(harness, envelope, row)

    def test_an_unfollowed_next_names_the_plan_as_typed(self, harness):
        done = poll_step(0, next="uv run -- python multi_llm.py generate-tasks {}".format(
            self._absolute(harness)))
        envelope, _, _, _ = harness.run(
            dict(REVIEW, plan=TYPED_PLAN), [launch_step(), done])
        assert envelope["next"]["mode"] == "generate-tasks"
        assert envelope["next"]["plan"] == TYPED_PLAN


# ===========================================================================
# What a failed result says the user can run (racing re-run 2026-10-04)
# ===========================================================================

GENERATE = {"mode": "generate-tasks", "plan": "plans/racing-game.md", "question": "",
            "extra_flags": []}
UV_PREFIX = "uv run --frozen --no-dev --project /skill -- python /skill/multi_llm.py"


def failed_generate_tasks(harness) -> dict:
    """The poll a failed detached generate-tasks answers with: the worker's own
    retry hint (`--yes`) and status's `resume with:` line (the launcher's
    `--detach`), both naming the plan absolutely (codex #4)."""
    absolute = os.path.join(os.path.realpath(harness.workspace), "plans",
                            "racing-game.md")
    host = SURFACE["host"]
    return {"exit": CODES["needs_human"], "stdout": [
        "Re-invoke to retry only the failures (task_generation):",
        "  {} generate-tasks {} --host {} --yes".format(UV_PREFIX, absolute, host),
        "      resume with: {} generate-tasks {} --host {} --detach --yes".format(
            UV_PREFIX, absolute, host),
        result_line(mode="generate-tasks", state=STATES["needs_human"],
                    exit=CODES["needs_human"])]}


class TestNoCommandToCompose:
    """Codex #4 and OpenCode #3/#4: the same re-invoke option twice, its plan
    absolute, the dispatcher's own lines replaced by a removal note and no
    command key at all — after which a host model composed one."""

    def _run(self, harness, **kwargs):
        return harness.run(GENERATE, [launch_step(), failed_generate_tasks(harness)],
                           **kwargs)

    def test_one_option_per_call_with_the_plan_as_typed(self, harness):
        envelope, _, _, _ = self._run(harness, extra_env={"STUB_RESUME_FAIL": "1"})
        calls = [option for option in envelope["options"] if "human" not in option]
        assert calls == [dict(GENERATE)]
        assert envelope["options"][-1] == {"human": SURFACE["needs_human_option"]}

    def test_the_redacted_lines_name_the_hosts_own_command(self, harness):
        envelope, _, _, _ = self._run(harness, extra_env={"STUB_RESUME_FAIL": "1"})
        spelled = "$multi-llm:multi-llm --generate-tasks plans/racing-game.md"
        lines = envelope["output"].split("\n")
        assert "  " + spelled in lines
        assert "      resume with: " + spelled in lines
        blob = json.dumps({key: envelope.get(key) for key in
                           ("output", "result", "options", "stderr", "message")})
        assert "uv run" not in blob and "multi_llm.py" not in blob
        assert RUNNABLE_MARKER not in blob

    def test_no_resume_means_an_explicit_no_command(self, harness):
        envelope, _, _, _ = self._run(harness, extra_env={"STUB_RESUME_FAIL": "1"})
        assert "user_next_step" in envelope and envelope["user_next_step"] is None
        assert envelope["no_command_reason"] == SURFACE["no_command_reason"]
        assert "resume_command" not in envelope
        assert envelope["message"].endswith(SURFACE["needs_human_message"]["no_command"])

    def test_a_resume_command_replaces_the_no_command(self, harness):
        command = "$multi-llm:multi-llm --generate-tasks plans/racing-game.md"
        envelope, _, _, _ = self._run(harness, extra_env={"STUB_RESUME_COMMAND": command})
        assert envelope["resume_command"] == command
        assert envelope["user_next_step"] is None
        assert "no_command_reason" not in envelope
        assert envelope["message"].endswith(SURFACE["needs_human_message"]["resume"])

    def test_an_error_result_says_so_too(self, harness):
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), poll_step(CODES["error"])])
        assert envelope["ok"] is False
        assert envelope["user_next_step"] is None
        assert envelope["no_command_reason"] == SURFACE["no_command_reason"]

    def test_a_needs_human_refusal_says_so_and_an_argument_error_does_not(self, harness):
        refused = _refused(harness, dict(IMPLEMENT),
                           meta=turn_meta(sandbox_mode="read-only"))
        assert refused["user_next_step"] is None
        assert refused["no_command_reason"] == SURFACE["no_command_reason"]
        bad = _refused(harness, dict(REVIEW, extra_flags=["--detach"]))
        assert "user_next_step" not in bad and "no_command_reason" not in bad

    def test_a_finished_run_carries_neither(self, harness):
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), poll_step(CODES["ok"])])
        assert "user_next_step" not in envelope and "no_command_reason" not in envelope

    def test_an_older_surface_keeps_the_pointer_and_drops_only_the_sentences(
        self, tmp_path_factory, monkeypatch
    ):
        """Additive within schema v5, through the module's own functions."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("mcp_server_under_test", SERVER_SRC)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bare = json.loads(json.dumps(SURFACE))
        del bare["dispatcher"]["user_command"]
        assert module.respeller_for(bare, {"plan": "/w/p.md", "plan_arg": "p.md"}) is None
        text = "  {} generate-tasks /w/p.md --yes".format(UV_PREFIX)
        assert module.redact(text, None, bare, None).strip() == RUNNABLE_MARKER
        respell = module.respeller_for(SURFACE, {"plan": "/w/p.md", "plan_arg": "p.md"})
        assert module.redact(text, None, SURFACE, respell).strip() == \
            "$multi-llm:multi-llm --generate-tasks p.md"
        # Without `reserved_flags.redundant`, a restated `--yes` is refused as before.
        del bare["reserved_flags"]["redundant"]
        assert module.split_extra_flags(["--yes"], bare) == ([], [], ("--yes", "--yes"))
        assert module.split_extra_flags(["--yes"], SURFACE) == ([], ["--yes"], None)


def _server_module(name="mcp_server_under_test"):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, SERVER_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATUS = {"mode": "status", "plan": "plans/x.md", "question": "", "extra_flags": []}
SPELLED_RESUME = "$multi-llm:multi-llm --implement plans/x.md --resume"


def failed_implement_status(harness, **body) -> dict:
    """A status poll of a failed implement run (exit 3), plan absolute."""
    absolute = os.path.join(os.path.realpath(harness.workspace), "plans", "x.md")
    payload = {"mode": "implement", "phase": "implement",
               "state": STATES["needs_human"], "exit": CODES["needs_human"]}
    payload.update(body)
    return {"exit": CODES["needs_human"], "stdout": [
        "implement: failed (batch 2 of 4)",
        "      resume with: {} implement {} --host {} --detach --resume".format(
            UV_PREFIX, absolute, SURFACE["host"]),
        result_line(**payload)]}


class TestAReportCallOnAFailedRun:
    """Adversarial review 2026-10-05: a direct `status` (or `abort`) call on a
    failed write run re-spelled status's `resume with:` line into the host's
    command, then asked `--blocked status` — which answers nothing, status is no
    phase — and so added `user_next_step: null`, `no_command_reason` and "give
    the user no command to type" beside output that visibly named one."""

    @staticmethod
    def _asked(harness):
        return [dispatcher_argv(call)[4] for call in harness.resume_calls()]

    def test_it_asks_about_the_runs_own_phase(self, harness):
        envelope, _, _, _ = harness.run(
            STATUS, [failed_implement_status(harness)],
            extra_env={"STUB_RESUME_ONLY": "implement"})
        assert self._asked(harness) == ["implement"]
        assert envelope["resume_command"] == SPELLED_RESUME
        assert "resume with: " + SPELLED_RESUME in envelope["output"]
        assert "no_command_reason" not in envelope
        assert envelope["message"].endswith(SURFACE["needs_human_message"]["resume"])

    def test_an_unpinned_listing_names_it_through_its_one_option(self, harness):
        envelope, _, _, _ = harness.run(
            STATUS, [failed_implement_status(harness, mode="status", phase=None)],
            extra_env={"STUB_RESUME_ONLY": "implement"})
        assert self._asked(harness) == ["implement"]
        assert envelope["resume_command"] == SPELLED_RESUME

    def test_no_lookup_answer_beside_a_visible_one_claims_no_absence(self, harness):
        envelope, _, _, _ = harness.run(
            STATUS, [failed_implement_status(harness)],
            extra_env={"STUB_RESUME_ONLY": "review-plan"})
        assert "resume with: " + SPELLED_RESUME in envelope["output"]
        assert "resume_command" not in envelope
        assert envelope["user_next_step"] is None
        assert "no_command_reason" not in envelope
        assert not envelope["message"].endswith(
            SURFACE["needs_human_message"]["no_command"])

    def test_a_phase_call_still_says_there_is_none(self, harness):
        """On a phase's own exit 3 the lookup is authoritative: a visible retry
        it does not back is one that would repeat the stop."""
        envelope, _, _, _ = harness.run(
            dict(STATUS, mode="implement"),
            [launch_step(), failed_implement_status(harness)],
            meta=turn_meta(sandbox_mode="workspace-write"),
            extra_env={"STUB_RESUME_ONLY": "review-plan"})
        assert self._asked(harness) == ["implement"]
        assert envelope["no_command_reason"] == SURFACE["no_command_reason"]


class TestARunStillAliveIsNoFinishedStop:
    """Adversarial review 2026-10-05: `stop_report` is `ok: false` with no
    `exit` — the run outlived the call, or its abort is unconfirmed — and still
    got "this phase did not finish … nothing names a re-run" as its
    `no_command_reason`."""

    def test_the_continuation_cap_stop_carries_no_reason(self, harness):
        done = poll_step(0, next="uv run -- python multi_llm.py review-plan p.md "
                                 "--reaggregate --host codex --yes")
        envelope, _, calls, _ = harness.run(
            REVIEW, [launch_step(run_id="R1"), done, launch_step(run_id="R2"), done,
                     ABORT_CONFIRMED],
            extra_env={"MLLM_TEST_NEXT_CAP": "1"})
        assert envelope["ok"] is False and envelope["run_id"] == "R2"
        assert envelope["exit"] is None
        assert [c for c in calls if DISPATCHER["abort"][1] in dispatcher_argv(c)]
        assert "no_command_reason" not in envelope
        assert envelope["user_next_step"] is None


class TestARedactedCommandCutAtAQuote:
    """Adversarial review 2026-10-05: the runnable pattern stops at a `"`, and
    `shlex.quote` spells an apostrophe `'"'"'`, so an `ask` retry whose question
    held one was matched up to the first `"`, parsed as question `what`, and
    re-spelled — and offered as an option — mangled."""

    PLAN = "plans/racing game.md"

    def _probe(self, question):
        import types

        from utils import cli_common

        module = _server_module()
        absolute = "/work/plans/racing game.md"
        command = cli_common.retry_command("ask", absolute, question, "--reaggregate",
                                           flags=[])
        values = {"word": "ask", "plan": absolute, "plan_arg": self.PLAN,
                  "question": "", "extra": []}
        child = types.SimpleNamespace(result=None, stdout="ask: " + command + "\n",
                                      stderr="")
        server = types.SimpleNamespace(surface=SURFACE)
        return (module.redact(child.stdout, None, SURFACE,
                              module.respeller_for(SURFACE, values)),
                module.Server.tool_calls_for(server, child, values))

    def test_an_apostrophe_is_re_spelled_whole(self):
        output, calls = self._probe("what's next")
        assert output.strip() == \
            'ask: $multi-llm:multi-llm --ask "{}" "what\'s next" --reaggregate'.format(
                self.PLAN)
        (call,) = calls
        assert call["question"] == "what's next"

    def test_a_double_quote_keeps_the_pointer(self):
        output, calls = self._probe('what\'s "up"')
        assert RUNNABLE_MARKER in output
        assert "$multi-llm:multi-llm --ask" not in output and "uv run" not in output
        assert all(call.get("question") != "what" for call in calls)
        assert [call.get("unparseable") for call in calls] == [True]

    def test_a_cut_after_a_closing_quote_is_refused(self):
        import types

        module = _server_module()
        values = {"word": "ask", "plan": "/w/p.md", "plan_arg": "p.md"}
        text = "uv run -- python /s/multi_llm.py ask /w/p.md 'a'\"x\"\n"
        output = module.redact(text, None, SURFACE, module.respeller_for(SURFACE, values))
        assert output.startswith(RUNNABLE_MARKER)
        child = types.SimpleNamespace(result=None, stdout=text, stderr="")
        calls = module.Server.tool_calls_for(
            types.SimpleNamespace(surface=SURFACE), child, values)
        assert [call.get("unparseable") for call in calls] == [True]


class TestASymlinkedPlanDir:
    """Adversarial review 2026-10-05: the call's plan is joined to the anchor
    without following symlinks, the dispatcher prints `Path(…).resolve()`, so
    under a symlinked plan directory the two never compared equal and every
    re-spelled command and option carried the absolute real path instead of
    the plan as typed."""

    def test_the_real_path_is_put_back_as_typed(self, harness, tmp_path):
        shared = tmp_path / "shared-plans"
        shared.mkdir()
        (shared / "x.md").write_text("# x\n", encoding="utf-8")
        (harness.workspace / "plans").symlink_to(shared, target_is_directory=True)
        real = os.path.realpath(shared / "x.md")
        failed = {"exit": CODES["needs_human"], "stdout": [
            "  {} generate-tasks {} --host {} --yes".format(
                UV_PREFIX, real, SURFACE["host"]),
            result_line(mode="generate-tasks", state=STATES["needs_human"],
                        exit=CODES["needs_human"])]}
        envelope, _, _, _ = harness.run(
            dict(GENERATE, plan="plans/x.md"), [launch_step(), failed],
            extra_env={"STUB_RESUME_FAIL": "1"})
        assert "  $multi-llm:multi-llm --generate-tasks plans/x.md" in \
            envelope["output"].split("\n")
        calls = [option for option in envelope["options"] if "human" not in option]
        assert [call["plan"] for call in calls] == ["plans/x.md"]
        assert real not in json.dumps(envelope)


#: Values the plan quoting must agree on with `next_phase.quote_plan_arg`.
QUOTE_CASES = [
    "plans/x.md", "plans/my plan.md", "plans/it's.md", 'plans/a"b.md',
    "plans/a\"b'c.md", "C:\\skill\\x.md", "tab\there", "--quick", "a,b", "",
]


class TestRestatedComposedFlags:
    """OpenCode #1 (and codex #6): a restated `--yes` was refused as reserved.
    A composed flag restated with the same meaning is now dropped with a note."""

    @pytest.mark.parametrize("flags, shown", [
        (["--yes"], "--yes"), (["-y"], "-y"), (["--non-interactive"], "--non-interactive"),
        (["--exec-subagents"], "--exec-subagents"), (["--ye"], "--ye"),
        (["--host", "codex"], "--host codex"), (["--host=codex"], "--host=codex"),
    ])
    def test_a_restated_flag_runs_and_is_named_in_a_note(self, harness, flags, shown):
        envelope, result, calls, _ = harness.run(
            dict(REVIEW, extra_flags=[*flags, "--quick"]),
            [launch_step(), poll_step(CODES["ok"])])
        assert not result["isError"] and envelope["ok"] is True, envelope
        launch = dispatcher_argv(calls[0])
        for baked in SURFACE["baked_flags"]:
            if baked.startswith("-"):
                assert launch.count(baked) == 1, launch
        assert "--quick" in launch
        note = SURFACE["reserved_flags"]["redundant"]["note"]
        assert envelope["extra_flags_note"] == note.replace("{flags}", shown)

    def test_the_quoting_agrees_with_the_dispatchers(self):
        import importlib.util

        sys.path.insert(0, str(SKILL_ROOT))
        from utils import next_phase

        spec = importlib.util.spec_from_file_location("mcp_server_quote", SERVER_SRC)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert [module.quote_for_reader(v) for v in QUOTE_CASES] == [
            next_phase.quote_plan_arg(v) for v in QUOTE_CASES]


# ===========================================================================
# The anchor (D3f)
# ===========================================================================

class TestAnchor:
    def test_the_env_rung_anchors_every_child(self, harness):
        client = harness.client([launch_step(run_id="RID-A"), poll_step(2),
                                 {"exit": 2, "sleep": 60}, ABORT_CONFIRMED])
        rid = client.start_call(REVIEW)
        _wait_for_calls(harness, 3)
        client.notify("notifications/cancelled", {"requestId": rid})
        envelope, _ = client.envelope(rid)
        assert client.close() == 0
        anchor = os.path.realpath(harness.workspace)
        assert envelope["anchor"] == anchor and envelope["anchor_source"] == "env:PWD"
        calls = harness.calls()
        assert DISPATCHER["abort"][1] in dispatcher_argv(calls[-1])
        assert {call["cwd"] for call in calls} == {anchor}
        assert anchor != str(harness.root) and anchor != str(harness.cwd)
        # `missing` is a forwarding report over the .mcp.json allowlist.
        environment = envelope["environment"]
        assert environment["missing"] == sorted(set(ENV_VARS) - {"MULTI_LLM_UV", "PWD"})
        assert environment["profile_shell"] == "/bin/bash"
        assert environment["interpreter"].startswith(sys.executable)

    def test_roots_win_over_the_environment(self, harness, tmp_path):
        other = tmp_path / "rooted"
        other.mkdir()
        envelope, _, calls, client = harness.run(
            {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line()]}], roots=[str(other)])
        assert envelope["anchor"] == os.path.realpath(other)
        assert envelope["anchor_source"] == "roots"
        assert calls[0]["cwd"] == os.path.realpath(other)
        assert [r["method"] for r in client.server_requests] == ["roots/list"]

    def test_list_changed_rereads_the_roots(self, harness, tmp_path):
        first, second = tmp_path / "r1", tmp_path / "r2"
        first.mkdir()
        second.mkdir()
        status = {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []}
        client = harness.client([{"exit": 0, "stdout": [result_line()]}],
                                roots=[str(first)])
        envelope, _ = client.envelope(client.start_call(status))
        assert envelope["anchor"] == os.path.realpath(first)
        client.roots = [str(second)]
        client.notify("notifications/roots/list_changed")
        envelope, _ = client.envelope(client.start_call(status))
        assert envelope["anchor"] == os.path.realpath(second)
        assert client.close() == 0

    def test_a_relative_plan_keeps_the_users_spelling_and_runs_in_the_anchor(self, harness):
        """Issue #15 (2026-10-02): the dispatcher receives the plan as typed, so the
        `plan_arg` it records and the next command it prints are the user's
        spelling; the child's cwd is the anchor, so it names the same file."""
        _, _, calls, _ = harness.run(
            {"mode": "status", "plan": "plans/../plans/x.md", "question": "",
             "extra_flags": []}, [{"exit": 0, "stdout": [result_line()]}])
        anchor = os.path.realpath(harness.workspace)
        assert dispatcher_argv(calls[0])[1] == "plans/../plans/x.md"
        assert calls[0]["cwd"] == anchor
        _, _, calls, _ = harness.run(
            {"mode": "status", "plan": "~/x.md", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [result_line()]}])
        assert dispatcher_argv(calls[0])[1] == str(harness.home / "x.md")

    def test_no_candidate_is_a_refusal_that_spawns_nothing(self, harness):
        envelope = _refused(harness, REVIEW, pwd=False)
        assert envelope["anchor"] is None and envelope["anchor_source"] is None
        assert envelope["state"] == STATES["needs_human"]
        head = SURFACE["anchor_unresolved"].split("{probed}")[0]
        assert envelope["message"].startswith(head)
        assert "env:PWD (unset)" in envelope["message"]
        assert "{probed}" not in envelope["message"]

    def test_the_plugin_s_own_directory_is_never_the_anchor(self, harness):
        envelope = _refused(harness, REVIEW, extra_env={"PWD": str(harness.root)})
        assert envelope["anchor"] is None

    def test_the_anchor_is_echoed_on_a_refusal_too(self, harness):
        envelope = _refused(harness, dict(REVIEW, mode="nope"))
        assert envelope["anchor"] == os.path.realpath(harness.workspace)

    STATUS = {"mode": "status", "plan": "p.md", "question": "", "extra_flags": []}

    def _dirs(self, tmp_path, *names):
        made = []
        for name in names:
            path = tmp_path / name
            path.mkdir(parents=True, exist_ok=True)
            made.append(path)
        return made

    def test_turn_metadata_anchors_when_pwd_is_absent(self, harness, tmp_path):
        (ws,) = self._dirs(tmp_path, "ws")
        envelope, _, calls, _ = harness.run(
            self.STATUS, [{"exit": 0, "stdout": [result_line()]}],
            meta=turn_meta([ws]), pwd=False)
        assert envelope["anchor"] == os.path.realpath(ws)
        assert envelope["anchor_source"] == "turn-metadata"
        assert calls[0]["cwd"] == os.path.realpath(ws)

    def test_the_dash_c_case_turn_metadata_beats_pwd(self, harness, tmp_path):
        """Row 7: under `codex exec -C <wt>` PWD still names the launching
        shell's checkout, and the turn metadata names `<wt>`."""
        (wt,) = self._dirs(tmp_path, "anchor-worktree")
        envelope, _, calls, client = harness.run(
            {"mode": "status", "plan": "plans/x.md", "question": "",
             "extra_flags": []}, [{"exit": 0, "stdout": [result_line()]}],
            meta=turn_meta([wt], sandbox_mode="danger-full-access"))
        anchor = os.path.realpath(wt)
        assert envelope["anchor"] == anchor != os.path.realpath(harness.workspace)
        assert envelope["anchor_source"] == "turn-metadata"
        assert calls[0]["cwd"] == anchor
        assert dispatcher_argv(calls[0])[1] == "plans/x.md"  # as typed; cwd is the anchor
        assert "anchor from turn-metadata" not in client.stderr  # one key: no choice

    def test_roots_still_win_over_turn_metadata(self, harness, tmp_path):
        rooted, ws = self._dirs(tmp_path, "rooted", "ws")
        envelope, _, _, _ = harness.run(
            self.STATUS, [{"exit": 0, "stdout": [result_line()]}],
            meta=turn_meta([ws]), roots=[str(rooted)])
        assert envelope["anchor_source"] == "roots"
        assert envelope["anchor"] == os.path.realpath(rooted)

    def test_several_workspaces_prefer_the_one_holding_pwd(self, harness, tmp_path):
        other, = self._dirs(tmp_path, "other")
        parent = harness.workspace.parent
        # Keys: an unrelated one first, then the workspace's parent (holds PWD),
        # then the workspace itself (equals PWD, more specific: it wins).
        envelope, _, calls, client = harness.run(
            self.STATUS, [{"exit": 0, "stdout": [result_line()]}],
            meta=turn_meta([other, parent, harness.workspace]))
        anchor = os.path.realpath(harness.workspace)
        assert envelope["anchor"] == anchor
        assert envelope["anchor_source"] == "turn-metadata"
        assert calls[0]["cwd"] == anchor
        assert "3 workspaces; chose {}".format(json.dumps(str(harness.workspace))) \
            in client.stderr and "it holds PWD" in client.stderr
        logs = [n["params"] for n in client.notifications
                if n["method"] == "notifications/message"]
        assert any("anchor from turn-metadata" in entry["data"]
                   and entry["level"] == "info" for entry in logs)

    def test_several_workspaces_none_holding_pwd_take_the_first(self, harness, tmp_path):
        first, second = self._dirs(tmp_path, "first", "second")
        envelope, _, _, client = harness.run(
            self.STATUS, [{"exit": 0, "stdout": [result_line()]}],
            meta=turn_meta([first, second]))
        assert envelope["anchor"] == os.path.realpath(first)
        assert "none holds PWD" in client.stderr

    def test_an_unusable_preferred_workspace_falls_to_the_next(self, harness, tmp_path):
        good, = self._dirs(tmp_path, "good")
        gone = tmp_path / "gone"
        envelope, _, _, client = harness.run(
            self.STATUS, [{"exit": 0, "stdout": [result_line()]}],
            meta=turn_meta([gone, good]), pwd=False)
        assert envelope["anchor"] == os.path.realpath(good)
        assert "skipped {} (not a directory)".format(json.dumps(str(gone))) \
            in client.stderr

    @pytest.mark.parametrize("workspaces", [
        "relative", "missing", "plugin-root", "a-file", "list", "string", "empty",
        "null", "nul-byte", "non-dict-meta",
    ])
    def test_unusable_turn_metadata_falls_through_to_pwd(self, harness, tmp_path,
                                                         workspaces):
        afile = tmp_path / "a-file"
        afile.write_text("x")
        meta = turn_meta({
            "relative": ["plans"], "missing": [tmp_path / "nope"],
            "plugin-root": [harness.root / "skills"], "a-file": [afile],
            "list": [], "string": "/tmp", "empty": {}, "null": None,
            "nul-byte": ["/tmp/\x00x"], "non-dict-meta": None,
        }[workspaces])
        if workspaces == "list":
            meta["x-codex-turn-metadata"]["workspaces"] = [str(tmp_path)]
        if workspaces == "non-dict-meta":
            meta = "not a dict"
        client = harness.client([{"exit": 0, "stdout": [result_line()]}])
        client.send({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {
            "name": SURFACE["tool_name"], "arguments": self.STATUS, "_meta": meta}})
        envelope, _ = client.envelope(9)
        assert client.close() == 0
        assert envelope["anchor"] == os.path.realpath(harness.workspace)
        assert envelope["anchor_source"] == "env:PWD"

    def test_no_rung_names_turn_metadata_in_the_refusal(self, harness, tmp_path):
        envelope = _refused(harness, REVIEW, pwd=False)
        assert "turn-metadata (absent)" in envelope["message"]
        envelope = _refused(harness, REVIEW, pwd=False,
                            meta=turn_meta([tmp_path / "nope"]))
        assert "turn-metadata (skipped" in envelope["message"]
        assert envelope["anchor_source"] is None

    def test_the_anchor_is_resolved_per_call(self, harness, tmp_path):
        first, second = self._dirs(tmp_path, "one", "two")
        client = harness.client([{"exit": 0, "stdout": [result_line()]}])
        envelope, _ = client.envelope(client.start_call(self.STATUS,
                                                        meta=turn_meta([first])))
        assert envelope["anchor"] == os.path.realpath(first)
        envelope, _ = client.envelope(client.start_call(self.STATUS,
                                                        meta=turn_meta([second])))
        assert envelope["anchor"] == os.path.realpath(second)
        envelope, _ = client.envelope(client.start_call(self.STATUS))
        assert envelope["anchor_source"] == "env:PWD"
        assert client.close() == 0

    def test_getcwd_is_never_a_fallback(self):
        src = SERVER_SRC.read_text(encoding="utf-8")
        assert "getcwd" not in src and "Path.cwd" not in src


# ===========================================================================
# The protocol channel's guard (src) and its live half
# ===========================================================================

class TestProtocolChannel:
    def test_nothing_prints_and_sys_stdout_lives_in_the_guard(self):
        tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", "print() at line {}".format(node.lineno)
        guard = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_guard_protocol_fd")
        inside = {id(n) for n in ast.walk(guard)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "stdout" and isinstance(
                    node.value, ast.Name) and node.value.id == "sys":
                assert id(node) in inside, "sys.stdout at line {}".format(node.lineno)

    def test_the_mismatch_warning_is_on_fd_2_only(self, harness, tmp_path):
        bad = dict(SURFACE, schema_version=CONST["SUPPORTED_SURFACE_VERSION"] - 1)
        root = make_plugin_root(tmp_path / "old", surface_text=json.dumps(bad))
        client = harness.client(root=root)
        client.response(client.request("tools/list"))
        assert client.close() == 0  # asserts every fd-1 line was JSON-RPC
        assert "schema_version" in client.stderr
        assert not any(b"schema_version" in line and b"notifications/message" not in line
                       for line in client.raw_lines)


# ===========================================================================
# Against the real skill, through a real uv (--no-sync: never re-syncs .venv)
# ===========================================================================

def _real_uv():
    found = shutil.which("uv")
    if not found:
        pytest.skip("uv is not on PATH; the real-dispatcher rows need it")
    if not (SKILL_ROOT / ".venv").is_dir():
        pytest.skip("the skill has no .venv to run --no-sync against")
    return found


@pytest.fixture
def real_root(tmp_path):
    _real_uv()
    return make_plugin_root(tmp_path / "real-root", skill_dir=SKILL_ROOT)


def _real_env(harness, uv: str) -> dict:
    return {"MULTI_LLM_UV": uv, SURFACE["uv"]["run_args_env"]: "--no-sync",
            "HOME": os.environ.get("HOME", str(harness.home)),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR")}


class TestRealDispatcher:
    def test_init_scaffolds_in_the_workspace_not_the_plugin(self, harness, real_root):
        uv = _real_uv()
        subprocess.run(["git", "init", "-q", str(harness.workspace)], check=True)
        envelope, result, _, _ = harness.run(
            {"mode": "init", "plan": "", "question": "", "extra_flags": []},
            root=real_root, extra_env=_real_env(harness, uv))
        assert envelope["ok"] is True, envelope
        assert envelope["exit"] == 0 and not result["isError"]
        assert (harness.workspace / ".multi-llm" / "providers.yaml").is_file()
        assert not (real_root / ".multi-llm").exists()
        assert not (SKILL_ROOT / ".multi-llm").exists()

    def test_the_real_runbook_passes_the_scrub(self, harness, real_root):
        uv = _real_uv()
        out = subprocess.run(
            [uv, "run", "--no-sync", "--project", str(SKILL_ROOT), "--", "python",
             str(SKILL_ROOT / DISPATCHER["script"])]
            + list(DISPATCHER["instructions"]) + [
                "review-plan", DISPATCHER["host_option"], SURFACE["host"]],
            capture_output=True, text=True, timeout=120, check=True)
        harness.runbook.write_text(out.stdout, encoding="utf-8")
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), {"exit": 3}])
        runbook = envelope["runbook"]
        sentinel = SURFACE["runbook"]["core_end_marker"] + " review-plan"
        assert runbook.split("\n").count(sentinel) == 1
        assert SURFACE["runbook"]["banner_opening"] not in runbook
        for line in runbook.split("\n"):
            assert not (re.search(r"\buv\s+run\b", line)
                        and DISPATCHER["script"] in line), line
        assert runbook.index(sentinel) < runbook.index("## Capability verbs")
