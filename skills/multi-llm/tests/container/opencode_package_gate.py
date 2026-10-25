"""Row 8 of plans/opencode-plugin-publish.md: the INSTALLED opencode package, offline.

The in-container half of the five `opencode-package-gate-*` rows. The host-side
half — the graders, the `gate-grader.json` each bundle carries, and the record —
is `tests/test_container_opencode_package_gate.py`. This file decides nothing:
it drives, measures and writes evidence, exactly as the codex package probes do.

WHAT IS UNDER TEST, AND WHAT IS NOT. The package route as a user installs it:
opencode's own config declares `plugin: ["/opt/mllm/opencode-plugin"]` (the
payload Dockerfile §10b staged), opencode loads it, the plugin injects
`/multi-llm` and registers the `multi_llm` tool, and the tool composes, launches
and supervises the dispatcher **from the payload's own skill tree**. No file
shim is staged, and `/opt/mllm/skill` — the checkout copy every other row runs —
is never the dispatcher here; a grader that found it would fail the row.

THE HOST'S MODEL IS A SCRIPT, NOT A MODEL. A second stub, on its own port,
answers the host's turn with ONE `multi_llm` call carrying the scenario's
arguments (`scripted_tool_call`, chat surface), and the turn carrying the tool's
result with a fixed terminal line. That grades the mechanism — the plugin's
argv, the detach claim, the supervision loop, the abort — and can never evidence
a model choosing the tool, which is not this row's question. The fan-out the
dispatcher runs is the provider lane's own: `opencode:stub/stub-model` through
`/opt/mllm/bin`, answered by the row's main stub with `container_full_chain.yaml`.

THE HOST PROVIDER IS NOT CALLED `stub`, deliberately. The fan-out's
`opencode:stub/stub-model` dispatch is hermetic, and `OpenCodeProvider` refuses a
slot whose provider is "defined only in your own opencode config" — the HOST's
config, `$HOME/.config/opencode/opencode.json`, is that config here. Measured
2026-10-25: a host provider named `stub` failed the fan-out with exactly that
refusal. So the host's provider is `hoststub`, and the fan-out resolves the way
it does on every provider-lane row.

The prompt is the command a user types — `opencode run --command multi-llm --
<args>` — so the host also expands the payload's `/multi-llm` template into the
turn; the host stub's fact probe records that it did.

Two rig accommodations, both recorded in the row's evidence rather than hidden:

* **A slow provider.** A run that finishes inside one `status --wait` has no
  non-terminal poll to supervise and nothing to abort. `gate-bin/opencode`, first
  on the host's PATH, sleeps before the fan-out's own `opencode run` (never
  before `--help`, `--version`, `auth` or `models`), then execs the ordinary
  wrapper. Same mechanism as the codex probes' `_slow_codex_profile`. It changes
  when the provider answers, never what it is asked or what it says.
* **A tool-result tap, cancellation row only.** opencode records an aborted tool
  call as `error: "Tool execution aborted"` and DROPS whatever `execute` returns
  afterwards (`completeToolCall` only writes a part still `running`; measured on
  1.19.4). The plugin's stop report — the run id, the log, whether the abort was
  confirmed — is exactly that late return, so a rig plugin
  (`plugin/mllm-gate-tap.js`) records every `multi_llm` result from opencode's
  own `tool.execute.after` hook, which still fires. It observes; it composes and
  dispatches nothing.

THE ABORT IS OPENCODE'S OWN. Plain `opencode run` serves its API in-process
(`http://opencode.internal`, no listener), so nothing outside it can abort its
session. The cancellation row therefore runs `opencode serve` and drives it with
`opencode run --attach`, and the abort is `POST /session/<id>/abort` — the call
opencode's TUI and SDK (`session.abort`) make. The row records that as
`abort_method`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

#: The five row-8 scenarios, as the `--probe` words the harness passes. The row
#: ids are the release-gate contract's (`opencode-package-gate-<scenario>`).
GATE_PROBES = ("gate-init-status", "gate-detach", "gate-supervision",
               "gate-completion", "gate-cancellation")

#: The three rows that run the same detached flow, each graded on its own aspect.
DETACHED_PROBES = ("gate-detach", "gate-supervision", "gate-completion")

#: The host's own provider and model, served by the HOST stub. Not `stub`: see
#: the module docstring.
HOST_PROVIDER = "hoststub"
HOST_MODEL = f"{HOST_PROVIDER}/stub-model"
#: Not 8097 (the codex scripted legs), 8098 (codex anchor), 8099 (the row's main
#: stub), nor a port `stub_server.LOCAL_MODEL_PROBE_PORTS` refuses.
HOST_STUB_PORT = 8096
#: `opencode serve` for the cancellation row, loopback only.
SERVE_PORT = 4196

PLAN_REL = "plans/auth-feature.md"
QUESTION = "What does this plan change?"
#: `progress.poll_s`'s floor (`run_status.WAIT_POLL_INTERVAL_S`), so each
#: `status --wait` returns within seconds and a run of SLOW_PROVIDER_S spans
#: several of them.
POLL_S = 5
#: How long the fan-out's provider is held before it runs. Two-plus polls on
#: the detached rows; on the cancellation row, longer than the row ever lives.
SLOW_PROVIDER_S = {"detached": 12, "cancellation": 900}

#: The tap's output, and the env var that hands the path to the tap plugin.
TAP_FILE = "gate-tool-tap.jsonl"
TAP_ENV = "MLLM_GATE_TAP"

#: The title rule every opencode turn needs (session-title generator), as the
#: chain scenario words it.
TITLE_PATTERN = (
    "Generate a concise, sentence-case title|You are a title generator\\. "
    "You output ONLY a thread title|Generate a short title \\(four words or "
    "less\\)|You are tasked with generating the session title")

_TAP_SOURCE = """\
// Rig observer for the row-8 cancellation scenario
// (skills/multi-llm/tests/container/opencode_package_gate.py). It records what
// the `multi_llm` tool returned, from opencode's own `tool.execute.after` hook,
// because an aborted call's part is closed before that value exists. It
// composes nothing, dispatches nothing and never throws.
import fs from "node:fs"

export const MllmGateTap = async () => ({
  "tool.execute.after": async (input, output) => {
    const target = process.env.%(env)s
    if (!target || input?.tool !== "multi_llm") return
    try {
      fs.appendFileSync(target, JSON.stringify({
        at: Date.now(),
        tool: input.tool,
        sessionID: input.sessionID ?? null,
        callID: input.callID ?? null,
        args: input.args ?? null,
        title: output?.title ?? null,
        output: output?.output ?? null,
        metadata: output?.metadata ?? null,
      }) + "\\n")
    } catch {
      // evidence, never a failure path
    }
  },
})
""" % {"env": TAP_ENV}


class GateContext:
    """What the entrypoint hands this leg: its paths and three of its helpers.

    Passed in rather than imported, because `entrypoint.py` runs as `__main__`
    in the image and an `import entrypoint` here would load a second copy.
    """

    def __init__(self, *, out: Path, home: Path, workspace: Path, real_dir: Path,
                 plugin_dir: Path, plan: Path, stub_cls, log, census,
                 plan_output_dir):
        self.out = out
        self.home = home
        self.workspace = workspace
        self.real_dir = real_dir
        self.plugin_dir = plugin_dir
        self.skill_dir = plugin_dir / "skills" / "multi-llm"
        self.plan = plan
        self.stub_cls = stub_cls
        self.log = log
        self.census = census
        self.plan_output_dir = plan_output_dir


# --------------------------------------------------------------------------- #
# The process table — `/proc`, with the session id, sampled on a thread
# --------------------------------------------------------------------------- #

def process_rows() -> list[dict]:
    """Every process: pid, ppid, pgid, sid, state, start time and argv."""
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            head, _, tail = stat.rpartition(")")
            fields = tail.split()
            raw = (entry / "cmdline").read_bytes()
        except (OSError, IndexError):
            continue
        try:
            row = {"pid": int(entry.name), "state": fields[0],
                   "ppid": int(fields[1]), "pgid": int(fields[2]),
                   "sid": int(fields[3]), "start": int(fields[19])}
        except (IndexError, ValueError):
            continue
        args = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        # A zombie has no cmdline left; its comm is all that names it.
        row["args"] = (args or f"[{head.split('(', 1)[-1]}]")[:600]
        rows.append(row)
    return sorted(rows, key=lambda r: r["pid"])


def ps_snapshot(out: Path, name: str) -> str:
    """The literal `ps -eo pid,ppid,pgid,sid,stat,args` table, kept in the bundle."""
    try:
        done = subprocess.run(["ps", "-eo", "pid,ppid,pgid,sid,stat,args"],
                              capture_output=True, text=True, timeout=20)
        text = done.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        text = f"ps unavailable: {exc}\n"
    path = out / f"gate-ps-{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path.name


class Sampler:
    """Samples `/proc` every `interval` seconds for the whole leg.

    Keeps one entry per process (keyed by pid AND start time, so a reused pid is
    a different process), with when it was first and last seen relative to the
    sampler's start. The literal `ps` table is also kept at the moment the
    detached worker is first seen — the "while it runs" evidence scenario 2 asks
    for — and at whatever moments the leg names.
    """

    def __init__(self, out: Path, interval: float = 0.1):
        self.out = out
        self.interval = interval
        self.started = time.time()
        self.seen: dict[str, dict] = {}
        self.snapshots: dict[str, str] = {}
        self.samples = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def sample(self) -> list[dict]:
        rows = process_rows()
        now = round(time.time() - self.started, 3)
        worker_new = False
        with self._lock:
            self.samples += 1
            for row in rows:
                key = f"{row['pid']}:{row['start']}"
                entry = self.seen.get(key)
                if entry is None:
                    # `first_ppid` survives a reparent: an orphan's ppid turns
                    # to 1 when its parent dies, and the grader follows the
                    # parent it was STARTED under to find the worker's tree.
                    entry = dict(row, first_seen=now, first_ppid=row["ppid"])
                    self.seen[key] = entry
                    worker_new = worker_new or "--internal-worker" in row["args"]
                entry.update(state=row["state"], ppid=row["ppid"],
                             pgid=row["pgid"], sid=row["sid"], last_seen=now)
                if row["state"] != "Z":
                    entry["last_alive"] = now
        if worker_new and "worker-live" not in self.snapshots:
            self.snapshot("worker-live")
        return rows

    def snapshot(self, name: str) -> None:
        self.snapshots[name] = ps_snapshot(self.out, name)

    def __enter__(self) -> "Sampler":
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(5)
        self.sample()

    def processes(self) -> list[dict]:
        with self._lock:
            return sorted(self.seen.values(),
                          key=lambda e: (e["first_seen"], e["pid"]))

    def write(self, name: str = "gate-processes.json") -> str:
        (self.out / name).write_text(json.dumps({
            "interval_s": self.interval,
            "samples": self.samples,
            "snapshots": self.snapshots,
            "processes": self.processes(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return name


# --------------------------------------------------------------------------- #
# Wiring: the host's config, the payload's own command, the slow provider
# --------------------------------------------------------------------------- #

def expected_command(plugin_dir: Path) -> dict:
    """What the payload's `config` hook injects as `command["multi-llm"]`.

    A transliteration of `injectCommand` in `packaging/opencode/src/index.mjs`:
    `dist/command.md`, every `{SKILL_DIR}` resolved to the payload's own skill
    tree, split at its frontmatter. Recomputed from the INSTALLED bytes, so a
    template equal to it can only have come from this payload's hook.
    """
    skill_dir = plugin_dir / "skills" / "multi-llm"
    raw = (plugin_dir / "dist" / "command.md").read_text(encoding="utf-8")
    body = raw.replace("{SKILL_DIR}", str(skill_dir))
    match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", body)
    if not match:
        return {"template": body, "description": "", "skill_dir": str(skill_dir)}
    described = re.search(r"^description:\s*(.*)$", match.group(1), re.M)
    description = described.group(1).strip() if described else ""
    if len(description) >= 2 and description[0] == description[-1] and \
            description[0] in "\"'":
        description = description[1:-1]
    return {"template": match.group(2), "description": description,
            "skill_dir": str(skill_dir)}


def wire_host(ctx: GateContext, *, tap: bool = False) -> dict:
    """Point the host's model at the HOST stub; keep the package declaration.

    The entrypoint's wiring step has already written opencode.json through
    `_write_opencode_config(..., route="package")` — the plugin spec and the
    `external_directory` grants every package row carries. Only the `provider`
    block is replaced, so the plugin route is the credentialed rows' own.
    """
    config_path = ctx.home / ".config" / "opencode" / "opencode.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["provider"] = {HOST_PROVIDER: {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Package gate host stub",
        # A placeholder, not a credential: the stub takes anything, and
        # @ai-sdk/openai-compatible insists on some key being set.
        "options": {"baseURL": f"http://127.0.0.1:{HOST_STUB_PORT}/v1",
                    "apiKey": "stub-not-a-credential"},
        "models": {"stub-model": {"name": "Package gate host model"}},
    }}
    config_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    wired = {"config": str(config_path), "plugin": document.get("plugin"),
             "provider": HOST_PROVIDER, "model": HOST_MODEL}
    if tap:
        plugin_dir = ctx.home / ".config" / "opencode" / "plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "mllm-gate-tap.js").write_text(_TAP_SOURCE, encoding="utf-8")
        wired["tap"] = str(plugin_dir / "mllm-gate-tap.js")
    (ctx.out / "gate-opencode.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return wired


def file_shims(ctx: GateContext) -> list[str]:
    """Any FILE-route `/multi-llm` command beside the plugin — there must be none."""
    candidates = [ctx.home / ".config" / "opencode" / sub / "multi-llm.md"
                  for sub in ("command", "commands")]
    candidates += [ctx.workspace / ".opencode" / sub / "multi-llm.md"
                   for sub in ("command", "commands")]
    return [str(path) for path in candidates if path.exists()]


def slow_provider(ctx: GateContext, seconds: int) -> Path:
    """`gate-bin/opencode`: hold the fan-out's model turn, then run it unchanged.

    Under the bundle, not the home: the tmpfs mounts are `noexec`. Only a
    `run` that is not `--help` sleeps — the adapter's own `run --help`,
    `--version`, `auth list` and `models` probes go straight through.
    """
    directory = ctx.out / "gate-bin"
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "opencode"
    shim.write_text(
        "#!/bin/sh\n"
        "# Row-8 rig accommodation (opencode_package_gate.slow_provider): the\n"
        "# fan-out provider's model turn is held so the run outlives a poll.\n"
        'case " $* " in\n'
        '  *" --help "*) ;;\n'
        f'  " run "*) sleep {int(seconds)} ;;\n'
        "esac\n"
        'exec /opt/mllm/bin/opencode "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    return directory


def with_poll_cadence(ctx: GateContext) -> None:
    """`progress.poll_s` at its floor, appended to the seeded workspace config."""
    path = ctx.workspace / ".multi-llm" / "providers.yaml"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + f"progress:\n  poll_s: {POLL_S}\n", encoding="utf-8")


def host_env(ctx: GateContext, path_prefix: Path | None = None,
             extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["HOME"] = str(ctx.home)
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}:{env.get('PATH', '')}"
    env.update(extra or {})
    return env


def debug_config(ctx: GateContext, env: dict) -> dict:
    """`opencode debug config`, and whether its `/multi-llm` is the payload's."""
    try:
        done = subprocess.run([str(ctx.real_dir / "opencode"), "debug", "config"],
                              cwd=str(ctx.workspace), env=env, capture_output=True,
                              text=True, timeout=120, stdin=subprocess.DEVNULL)
        stdout, stderr, code = done.stdout, done.stderr, done.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, code = str(exc.stdout or ""), str(exc.stderr or ""), None
    (ctx.out / "gate-debug-config.json").write_text(stdout, encoding="utf-8")
    (ctx.out / "gate-debug-config.stderr.txt").write_text(stderr, encoding="utf-8")
    try:
        config = json.loads(stdout)
    except ValueError:
        return {"exit": code, "parsed": False}
    command = (config.get("command") or {}).get("multi-llm") or {}
    expected = expected_command(ctx.plugin_dir)
    matches = (command.get("template") == expected["template"]
               and command.get("description") == expected["description"])
    return {
        "exit": code,
        "parsed": True,
        "command_keys": sorted(config.get("command") or {}),
        "plugin": config.get("plugin"),
        "plugin_origins": config.get("plugin_origins"),
        "template_matches_payload": command.get("template") == expected["template"],
        "description_matches_payload": (command.get("description")
                                        == expected["description"]),
        # The skill dir the injected template was resolved against — the
        # payload's, and only when the template is the payload's byte for byte.
        "command_skill_dir": expected["skill_dir"] if matches else None,
        "payload_command_md": str(ctx.plugin_dir / "dist" / "command.md"),
    }


def command_facts(ctx: GateContext) -> Path:
    """The stub's fact probe: did the host send the payload's `/multi-llm` text?

    The template's longest line, escaped — a sentence only that command carries.
    Names only reach the log, never the prompt.
    """
    template = expected_command(ctx.plugin_dir)["template"]
    line = max((ln.strip() for ln in template.splitlines()), key=len, default="")
    path = ctx.out / "gate-facts.json"
    path.write_text(json.dumps({"multi_llm_command": re.escape(line)}, indent=2)
                    + "\n", encoding="utf-8")
    return path


def call_scenario(ctx: GateContext, label: str, arguments: dict) -> Path:
    """A one-call scripted scenario (JSON is YAML): `multi_llm`, chat surface."""
    path = ctx.out / f"gate-scenario-{label}.json"
    path.write_text(json.dumps({
        "name": f"opencode_package_gate_{label.replace('-', '_')}",
        "description": f"row-8 package gate, leg {label}: one scripted "
                       f"multi_llm call, then a terminal line",
        "scripted_tool_call": {
            "surfaces": ["chat"],
            "name": "multi_llm",
            "call_id": f"call_mllm_gate_{label.replace('-', '_')}",
            "arguments": arguments,
            "terminal": f"The multi_llm {label} call returned.",
        },
        "prompts": [{"pattern": TITLE_PATTERN, "content": "Package gate",
                     "description": "session-title generator"}],
        "fallback": {"content": "OK"},
    }, indent=2) + "\n", encoding="utf-8")
    return path


def run_argv(ctx: GateContext, typed: list[str], attach: str | None = None) -> list[str]:
    argv = [str(ctx.real_dir / "opencode"), "run", "--command", "multi-llm",
            "--format", "json", "--model", HOST_MODEL]
    if attach:
        argv += ["--attach", attach]
    # `--` is mandatory: the typed arguments start with a flag, which yargs
    # would otherwise swallow (host_matrix.yaml, opencode `invoke_note`).
    return argv + ["--", *typed]


# --------------------------------------------------------------------------- #
# Reading back: the tool parts, the run records
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _parsed(text) -> dict | None:
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def tool_parts(events: list[dict]) -> list[dict]:
    """Every `multi_llm` tool part opencode reported, with its result parsed."""
    parts = []
    for event in events:
        part = event.get("part") or {}
        if event.get("type") != "tool_use" or part.get("tool") != "multi_llm":
            continue
        state = part.get("state") or {}
        parts.append({
            "call_id": part.get("callID"),
            "status": state.get("status"),
            "input": state.get("input"),
            "title": state.get("title"),
            "error": state.get("error"),
            "metadata": state.get("metadata"),
            "result": _parsed(state.get("output")),
        })
    return parts


def run_records(ctx: GateContext) -> list[dict]:
    """Every `.status.json` under the plan's output dir that carries a run."""
    root = Path(ctx.plan_output_dir(ctx.plan))
    found = []
    for status in sorted(root.rglob(".status.json")) if root.is_dir() else []:
        try:
            document = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run = document.get("run") if isinstance(document, dict) else None
        if not isinstance(run, dict):
            continue
        found.append({
            "dir": str(status.parent.relative_to(ctx.workspace)),
            "run": {k: run.get(k) for k in (
                "run_id", "state", "exit_code", "error", "pid", "pgid",
                "detached", "host", "mode", "log_path", "route")},
            "models": sorted(((document.get("progress") or {}).get("models")
                              or {}).items()),
            "files": sorted(p.name for p in status.parent.iterdir()
                            if p.is_file()),
        })
    return found


def census_delta(before: dict, after: dict) -> dict:
    return {name: [before.get(name), after.get(name)]
            for name in sorted(set(before) | set(after))
            if before.get(name) != after.get(name)}


def wait_for(predicate, timeout: float, interval: float = 0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------- #
# One foreground leg: a fresh `opencode run`, its own host stub
# --------------------------------------------------------------------------- #

def foreground_leg(ctx: GateContext, label: str, typed: list[str],
                   arguments: dict, env: dict, timeout: float = 420.0) -> dict:
    """`opencode run --command multi-llm -- <typed>`, the host stub scripted.

    Launched with stdin at /dev/null (`opencode run` reads a non-TTY stdin to
    EOF before it starts) and never under a `timeout` wrapper, which wedges
    opencode before session creation; the bound is `Popen.wait`'s.
    """
    scenario = call_scenario(ctx, label, arguments)
    stub_log = ctx.out / f"gate-stub-{label}.jsonl"
    events_path = ctx.out / f"gate-host-{label}.jsonl"
    stderr_path = ctx.out / f"gate-host-{label}.stderr.txt"
    argv = run_argv(ctx, typed)
    ctx.log(f"package gate: leg {label}: {' '.join(argv[1:])}")
    timed_out = False
    with ctx.stub_cls(HOST_STUB_PORT, stub_log, scenario,
                      facts=command_facts(ctx)):
        started = time.time()
        with events_path.open("w", encoding="utf-8") as out, \
                stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.Popen(argv, cwd=str(ctx.workspace), env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait(timeout=30)
                code = None
        wall = round(time.time() - started, 2)
    events = read_jsonl(events_path)
    return {
        "label": label,
        "argv": [Path(argv[0]).name, *argv[1:]],
        "typed": typed,
        "arguments": arguments,
        "pid": proc.pid,
        "exit": code,
        "timed_out": timed_out,
        "wall_clock_s": wall,
        "events": events_path.name,
        "stderr": stderr_path.name,
        "stub_log": stub_log.name,
        "scenario": scenario.name,
        "session_ids": sorted({e.get("sessionID") for e in events
                               if e.get("sessionID")}),
        "error_events": [e.get("error") for e in events if e.get("type") == "error"],
        "tool_parts": tool_parts(events),
    }


# --------------------------------------------------------------------------- #
# The scenarios
# --------------------------------------------------------------------------- #

def init_status(ctx: GateContext, record: dict, env: dict) -> None:
    """Scenario 1: `/multi-llm --init`, then `/multi-llm --status <plan>`.

    `init` scaffolds `.multi-llm/providers.yaml` and refuses when one exists —
    and `seed_workspace.py` always writes one. So the seeded copy is set aside
    first (recorded), which is the state of a project that has never run
    multi-llm: init then has its project-local files to create, and the census
    around the leg is what shows it created them.
    """
    seeded = ctx.workspace / ".multi-llm"
    removed = sorted(str(p.relative_to(ctx.workspace))
                     for p in seeded.rglob("*") if p.is_file()) \
        if seeded.is_dir() else []
    shutil.rmtree(seeded, ignore_errors=True)
    record["workspace_set_aside"] = removed

    before = ctx.census()
    init = foreground_leg(ctx, "init", ["--init"],
                          {"mode": "init", "plan": "", "question": "",
                           "extra_flags": []}, env)
    after_init = ctx.census()
    status = foreground_leg(ctx, "status", ["--status", PLAN_REL],
                            {"mode": "status", "plan": PLAN_REL, "question": "",
                             "extra_flags": []}, env)
    after = ctx.census()
    init["census_delta"] = census_delta(before, after_init)
    status["census_delta"] = census_delta(after_init, after)
    record["legs"] = [init, status]


def detached(ctx: GateContext, record: dict, env: dict) -> None:
    """Scenarios 2-4: one `ask --quick` call the tool detaches and supervises."""
    before = ctx.census()
    with Sampler(ctx.out) as sampler:
        leg = foreground_leg(ctx, "ask", ["--ask", PLAN_REL, QUESTION, "--quick"],
                             {"mode": "ask", "plan": PLAN_REL, "question": QUESTION,
                              "extra_flags": ["--quick"]}, env)
        # The tool returns only once its poll saw a terminal run, so the worker
        # is finishing; give its exit a moment to land in the table.
        wait_for(lambda: not any("--internal-worker" in r["args"]
                                 and r["state"] != "Z" for r in process_rows()),
                 timeout=30)
        sampler.snapshot("after")
    leg["census_delta"] = census_delta(before, ctx.census())
    record["legs"] = [leg]
    record["host_pid"] = leg["pid"]
    record["processes"] = sampler.write()


def _http(method: str, url: str, timeout: float = 20.0) -> tuple[int | None, str]:
    request = urllib.request.Request(
        url, method=method,
        data=b"{}" if method == "POST" else None,
        headers={"content-type": "application/json"} if method == "POST" else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _supervising(serve_pid: int) -> dict | None:
    """The tool's live `status --wait --run-id <id>` poll, if one is running.

    The uv wrapper the plugin spawns is the serve process's direct child, so
    `ppid == serve_pid` is what makes it the TOOL's poll and nothing else's.
    """
    for row in process_rows():
        args = row["args"]
        if (row["ppid"] == serve_pid and row["state"] != "Z"
                and "multi_llm.py status" in args and " --wait" in args):
            match = re.search(r"--run-id (\S+)", args)
            if match:
                return {"pid": row["pid"], "run_id": match.group(1), "args": args}
    return None


def _provider_running(run_id: str) -> dict | None:
    """The fan-out's provider turn, live inside the claimed run's worker group.

    The slow provider's shim (`gate-bin/opencode run …`) or the `sleep` it holds
    the turn with, in the process group of the worker whose argv names
    `run_id`. Waiting for it is what makes the abort land MID-RUN: a worker
    with a live grandchild, not a launch that has not reached its provider.
    """
    rows = process_rows()
    groups = {r["pgid"] for r in rows if r["state"] != "Z"
              and f"--internal-run-id {run_id}" in r["args"]}
    for row in rows:
        if (row["pgid"] in groups and row["state"] != "Z"
                and ("gate-bin/opencode run" in row["args"]
                     or row["args"].startswith("sleep "))):
            return {"pid": row["pid"], "pgid": row["pgid"], "args": row["args"]}
    return None


def cancellation(ctx: GateContext, record: dict, env: dict) -> None:
    """Scenario 5: abort the session while the tool supervises its claimed run."""
    serve_out = ctx.out / "gate-serve.stdout.txt"
    serve_err = ctx.out / "gate-serve.stderr.txt"
    base = f"http://127.0.0.1:{SERVE_PORT}"
    label = "ask"
    scenario = call_scenario(ctx, label, {
        "mode": "ask", "plan": PLAN_REL, "question": QUESTION,
        "extra_flags": ["--quick"]})
    stub_log = ctx.out / f"gate-stub-{label}.jsonl"
    events_path = ctx.out / f"gate-host-{label}.jsonl"
    stderr_path = ctx.out / f"gate-host-{label}.stderr.txt"
    typed = ["--ask", PLAN_REL, QUESTION, "--quick"]
    argv = run_argv(ctx, typed, attach=base)
    abort: dict = {"method": "POST /session/{id}/abort",
                   "via": f"opencode serve on {base}, driven by `opencode run "
                          f"--attach {base}`",
                   "sent": False}
    leg: dict = {"label": label, "argv": [Path(argv[0]).name, *argv[1:]],
                 "typed": typed, "events": events_path.name,
                 "stderr": stderr_path.name, "stub_log": stub_log.name,
                 "scenario": scenario.name}
    before = ctx.census()

    with ctx.stub_cls(HOST_STUB_PORT, stub_log, scenario,
                      facts=command_facts(ctx)), Sampler(ctx.out) as sampler:
        handles = [serve_out.open("w", encoding="utf-8"),
                   serve_err.open("w", encoding="utf-8"),
                   events_path.open("w", encoding="utf-8"),
                   stderr_path.open("w", encoding="utf-8")]
        serve = subprocess.Popen(
            [str(ctx.real_dir / "opencode"), "serve", "--port", str(SERVE_PORT),
             "--hostname", "127.0.0.1"],
            cwd=str(ctx.workspace), env=env, stdin=subprocess.DEVNULL,
            stdout=handles[0], stderr=handles[1])
        record["serve_pid"] = serve.pid
        try:
            # A request sent before the "listening" line can hang for good
            # (measured: an unbounded GET /session never returned), so the
            # readiness gate is the line, and every request below is bounded.
            listening = wait_for(
                lambda: "listening on" in serve_out.read_text(encoding="utf-8"),
                timeout=90)
            leg["serve_listening"] = bool(listening)
            ready = wait_for(lambda: _http("GET", f"{base}/path", 10)[0] == 200,
                             timeout=60, interval=1.0)
            leg["serve_ready"] = bool(ready)

            started = time.time()
            client = subprocess.Popen(argv, cwd=str(ctx.workspace), env=env,
                                      stdin=subprocess.DEVNULL, stdout=handles[2],
                                      stderr=handles[3])
            leg["pid"] = client.pid

            poll = wait_for(lambda: _supervising(serve.pid), timeout=240,
                            interval=0.2)
            abort["supervising_poll"] = poll
            if poll:
                abort["provider_at_abort"] = wait_for(
                    lambda: _provider_running(poll["run_id"]), timeout=120)
                abort["supervising_at_abort"] = _supervising(serve.pid)
                code, body = _http("GET", f"{base}/session", 20)
                try:
                    sessions = json.loads(body) if code == 200 else []
                except ValueError:
                    sessions = []
                listed = [s for s in (sessions if isinstance(sessions, list) else [])
                          if isinstance(s, dict) and not s.get("parentID")]
                abort["sessions_listed"] = [s.get("id") for s in listed]
                session_id = listed[0].get("id") if len(listed) == 1 else None
                abort["session_id"] = session_id
                sampler.snapshot("at-abort")
                abort["run_records_at_abort"] = run_records(ctx)
                if session_id:
                    abort["sent_at_s"] = round(time.time() - started, 3)
                    status, reply = _http(
                        "POST", f"{base}/session/{session_id}/abort", 20)
                    abort.update(sent=True, http_status=status,
                                 response=reply[:500])
            try:
                leg["exit"] = client.wait(timeout=120)
            except subprocess.TimeoutExpired:
                leg["timed_out"] = True
                client.kill()
                leg["exit"] = client.wait(timeout=30)
            leg["wall_clock_s"] = round(time.time() - started, 2)

            # The plugin's abort dispatch is bounded at 30 s plus its 5 s kill
            # grace; the tap line is written once `execute` has returned.
            tap = ctx.out / TAP_FILE
            abort["tap_seen"] = bool(wait_for(lambda: read_jsonl(tap), timeout=90))
            # And the claimed worker's group has to be gone, not just signalled.
            run_id = (poll or {}).get("run_id")
            wait_for(lambda: all(r["run"].get("state") in ("done", "failed",
                                                           "aborted")
                                 for r in run_records(ctx)
                                 if r["run"].get("run_id") == run_id),
                     timeout=60)
            wait_for(lambda: not any("--internal-worker" in r["args"]
                                     and r["state"] != "Z"
                                     for r in process_rows()), timeout=60)
            time.sleep(1.0)
            sampler.snapshot("after")
            session_id = abort.get("session_id")
            if session_id:
                code, body = _http("GET", f"{base}/session/{session_id}/message", 20)
                (ctx.out / "gate-session-messages.json").write_text(
                    body, encoding="utf-8")
                abort["messages_status"] = code
        finally:
            serve.send_signal(signal.SIGTERM)
            try:
                serve.wait(timeout=15)
            except subprocess.TimeoutExpired:
                serve.kill()
                serve.wait(timeout=15)
            for handle in handles:
                handle.close()

    events = read_jsonl(events_path)
    leg.update(
        session_ids=sorted({e.get("sessionID") for e in events if e.get("sessionID")}),
        error_events=[e.get("error") for e in events if e.get("type") == "error"],
        tool_parts=tool_parts(events),
        census_delta=census_delta(before, ctx.census()))
    record["legs"] = [leg]
    record["abort"] = abort
    record["tap"] = TAP_FILE
    record["processes"] = sampler.write()


def run(probe: str, ctx: GateContext) -> int:
    """Drive one row-8 scenario and write `opencode-gate.json`. 0 unless it broke.

    The verdict is the host-side grader's. A non-zero return means the LEG could
    not be driven at all (a missing payload, an unparseable config), which is a
    rig failure and is reported as one rather than graded.
    """
    if probe not in GATE_PROBES:
        raise SystemExit(f"opencode package gate: unknown probe {probe!r}")
    if not (ctx.plugin_dir / "package.json").is_file():
        raise SystemExit(
            f"opencode package gate: no installed payload at {ctx.plugin_dir} — "
            f"the image was built without Dockerfile §10b's staged package")
    scenario = probe.removeprefix("gate-")
    record: dict = {
        "probe": probe,
        "scenario": scenario,
        "installed_root": str(ctx.plugin_dir),
        "payload_skill_dir": str(ctx.skill_dir),
        "plan": PLAN_REL,
        "host_model": HOST_MODEL,
        "poll_s": None,
        "slow_provider_s": None,
    }
    try:
        record["opencode_version"] = subprocess.run(
            [str(ctx.real_dir / "opencode"), "--version"], capture_output=True,
            text=True, timeout=60, env=host_env(ctx),
            stdin=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        record["opencode_version"] = None

    record["wiring"] = wire_host(ctx, tap=scenario == "cancellation")
    extra = {TAP_ENV: str(ctx.out / TAP_FILE)} if scenario == "cancellation" else None
    path_prefix = None
    if probe != "gate-init-status":
        with_poll_cadence(ctx)
        record["poll_s"] = POLL_S
        seconds = SLOW_PROVIDER_S["cancellation" if scenario == "cancellation"
                                  else "detached"]
        path_prefix = slow_provider(ctx, seconds)
        record["slow_provider_s"] = seconds
    env = host_env(ctx, path_prefix, extra)
    record["file_shims"] = file_shims(ctx)
    record["debug_config"] = debug_config(ctx, env)

    started = time.time()
    try:
        if probe == "gate-init-status":
            init_status(ctx, record, env)
        elif probe in DETACHED_PROBES:
            detached(ctx, record, env)
        else:
            cancellation(ctx, record, env)
    finally:
        record["wall_clock_s"] = round(time.time() - started, 2)
        record["run_records"] = run_records(ctx)
        record["plan_output_dir"] = str(
            Path(ctx.plan_output_dir(ctx.plan)).relative_to(ctx.workspace))
        (ctx.out / "opencode-gate.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctx.log(f"package gate: {probe} driven in {record['wall_clock_s']}s")
    return 0
