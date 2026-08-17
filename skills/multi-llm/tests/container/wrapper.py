#!/usr/bin/env python3
"""The PATH wrapper: redirection, environment scrub, call log, receipts (C2).

``/opt/mllm/bin/<cli>`` is this file under another name; the real binary lives at
``/opt/mllm/real/<cli>``. Every invocation:

    1. appends a REDACTED argv + cwd + an env ALLOWLIST to ``$MLLM_CALL_LOG``
    2. exports this CLI's backend config (env and/or a written config file)
    3. runs the genuine binary, untouched, and removes the per-invocation config
       root once it exits (``run_child`` — the one reason this is a fork rather
       than an ``execve``)

**Why not just set env vars in the container?** Three reasons, and the first is
fatal on its own:

1. **Role collision.** In a host-lane run, ``claude-code`` is the *host* (its brain
   must reach the real API) while ``claude-code`` may also be dispatched as a
   *provider* (its brain must reach the stub). Both read ``ANTHROPIC_BASE_URL``. A
   process-wide env var cannot express "real for the parent, stub for the child".
   A wrapper can, because it applies per-exec.
2. **Inheritance.** The orchestrator's provider subprocesses inherit the host's
   environment. ``utils/detach.py`` already curates that env; adding backend
   redirection there would put test-only concerns into shipping code.
3. **Not every CLI takes an env var.** ``codex`` wants a ``[model_providers.*]``
   block, ``opencode`` a provider block in its JSON config, ``goose`` a config
   key. A wrapper normalizes all three shapes behind one interface.

**Step 2 is a scrub, not an overlay**, and that is a requirement of the host lane
rather than hygiene. Everything above the wrapper runs with the live
``OPENROUTER_API_KEY`` in its environment — that is the point of the host lane —
on a network that (unlike the provider lane's) can reach openrouter.ai. Overlaying
a base URL leaves the credential sitting in the child's environment, so a CLI that
falls back past the stub, reads a variable we never anticipated, or is compromised
has both a valid key and a route. The child's environment is therefore **built
from scratch**: an allowlisted empty base, the fake per-exec stub key in place of
any vendor credential, and a per-invocation ``HOME``/XDG/CLI config root. The worst
case of a mis-redirected provider is then an authentication failure against a real
endpoint — loud, free, and recorded.

**One class of argv is deliberately left alone**: the executor's credential probes
(``codex login status``, ``opencode auth list``). They are local reads of a
credential store and make no HTTP request at all, so redirecting them is not a fix
and cannot be one — pointed at the stub they produce a logged 501 and a probe that
still reports logged-out. What makes them pass is the synthesized credential file
in the per-CLI home, which rule 3 seeds the per-invocation root from.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REAL_DIR = Path(os.environ.get("MLLM_REAL_DIR", "/opt/mllm/real"))
HOMES_ROOT = Path(os.environ.get("MLLM_HOMES_ROOT", "/work/homes"))
REDIRECT_YAML = Path(os.environ.get("MLLM_REDIRECT_YAML",
                                    "/opt/mllm/rig/redirect.yaml"))

# --------------------------------------------------------------------------- #
# Redaction — at WRITE time, never at read time.
# --------------------------------------------------------------------------- #

# The value after any key-bearing flag is replaced wholesale.
KEY_FLAGS = re.compile(
    r"^--?(api[-_]?key|token|auth[-_]?token|bearer|secret|password|k)$", re.I)

# Credential SHAPES, matched on the token itself. An allowlist of what we capture
# is used for the environment (below); for argv a shape match is the right tool,
# because a credential can arrive as a bare positional.
CRED_SHAPES = (
    re.compile(r"^sk-[A-Za-z0-9_\-]{16,}$"),
    re.compile(r"^sk-ant-[A-Za-z0-9_\-]{16,}$"),
    re.compile(r"^ghp_[A-Za-z0-9]{20,}$"),
    re.compile(r"^gh[pousr]_[A-Za-z0-9]{20,}$"),
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]{10,}$"),
    re.compile(r"^ey[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+$"),  # JWT
    re.compile(r"^[A-Za-z0-9_\-]{32,}$"),                                  # base64url
)

# The env capture is an ALLOWLIST rather than a denylist. A denylist here would be
# a guess about which variable a future CLI release reads its key from.
ENV_CAPTURE_ALLOW = ("MLLM_", "HOME", "PATH")
ENV_CAPTURE_SUFFIX = ("_BASE_URL",)

# What the CHILD is allowed to inherit. Same reasoning, applied to execution
# rather than to logging — and here the guess would be billable.
#
# The second group is terminal/UI geometry, and it is here because of a measured
# failure rather than for tidiness. `AiderProvider.get_env()` sets
# `COLUMNS=10000` precisely to stop aider hard-wrapping its output; the scrub
# dropped it, aider wrapped at ~78 columns, and the JSON document it printed came
# back with raw newlines inside its string values — invalid JSON, reported three
# layers up as the uninformative "No JSON found in aider output". Dropping what
# the PRODUCT's own adapter deliberately sets means the container tests a
# configuration the product never ships.
#
# These are safe to carry by inspection: none can hold a credential, and
# `tests/test_wrapper_env_contract.py` asserts that every adapter's `get_env()`
# keys are covered either here or by that CLI's `env_extra` row in
# `redirect.yaml` (the route for one that CAN hold a credential, since it is set
# from the rig's own value rather than inherited) — so an adapter that later
# contributes something neither place accounts for fails a test rather than
# being silently scrubbed away.
#
# The third group is claude-code's retry envelope (§7.3): `CLAUDE_CODE_MAX_RETRIES`
# and `API_TIMEOUT_MS` are the CLI's own knobs, set by `ClaudeCodeProvider.get_env()`
# because the shipped default of 10 retries burned a measured 182 s in one
# rate-limited slot. Scrubbed, a container row runs on that 10-retry default —
# the same class of "testing a configuration the product never ships" as the
# aider case above, and the one that decides whether `--fallback-model`'s
# overload path exists at all. Both are integers; neither can hold a credential.
ENV_CARRY_EXACT = (
    "PATH", "TERM", "LANG", "LC_ALL", "TZ",
    "COLUMNS", "LINES", "BROWSER", "NO_COLOR", "FORCE_COLOR",
    "CLAUDE_CODE_MAX_RETRIES", "API_TIMEOUT_MS",
    # §4.2.1's pinned structured-output budget. An integer, never a credential,
    # and load-bearing on cost: without it the container row would run at the
    # CLI's default of five model calls per invocation while the product ships
    # one.
    "MAX_STRUCTURED_OUTPUT_RETRIES",
)
ENV_CARRY_PREFIX = ("MLLM_",)


def _redact(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:8]
    return f"<redacted:sha256:{digest}>"


def redact_argv(argv: list[str]) -> list[str]:
    """Redact an argv for the call log, at write time."""
    out: list[str] = []
    expect_value = False
    for token in argv:
        if expect_value:
            out.append(_redact(token))
            expect_value = False
            continue
        if KEY_FLAGS.match(token):
            out.append(token)
            expect_value = True
            continue
        if "=" in token and token.startswith("-"):
            flag, _, value = token.partition("=")
            if KEY_FLAGS.match(flag) or any(s.match(value) for s in CRED_SHAPES):
                out.append(f"{flag}=" + _redact(value))
                continue
        if any(shape.match(token) for shape in CRED_SHAPES):
            out.append(_redact(token))
            continue
        out.append(token)
    return out


def collapse_prompt(argv: list[str], prompt_min: int = 200) -> list[str]:
    """Reduce a prompt-shaped positional to ``sha256[:8]+length``.

    A prompt is tens of KB of plan text and the assertion reads a normalized flag
    view of the record, never the raw argv — the same reason the audit log does
    this.
    """
    out = []
    for token in argv:
        if len(token) >= prompt_min and not token.startswith("-"):
            digest = hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:8]
            out.append(f"<prompt:sha256:{digest}:len={len(token)}>")
        else:
            out.append(token)
    return out


# --------------------------------------------------------------------------- #
# Redirection data
# --------------------------------------------------------------------------- #

#: Provider ids whose BINARY is named differently. `Path(sys.argv[0]).name` gives
#: us the binary, but `redirect.yaml` rows and `home_seed/<id>/` are keyed by the
#: provider id — so without this map a `claude` invocation looks up a row and a
#: seed directory that do not exist, and silently gets neither its redirect nor
#: its synthesized config.
BINARY_TO_CLI = {"claude": "claude-code"}


def resolve_cli_id(binary: str) -> str:
    """The provider id for an invoked binary name."""
    return BINARY_TO_CLI.get(binary, binary)


def load_row(cli: str) -> dict:
    """The CLI's ``redirect.yaml`` row, or an empty row if the file is absent."""
    if not REDIRECT_YAML.exists():
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover - the image always ships pyyaml
        return {}
    data = yaml.safe_load(REDIRECT_YAML.read_text(encoding="utf-8")) or {}
    for row in data.get("clis", []) or []:
        if row.get("cli") == cli or row.get("binary") == cli:
            return row
    return {}


def stub_base_url(exec_id: str, row: dict) -> str:
    """The base URL this exec should use.

    A per-exec id baked into the path is the primary session key: exact, needing
    no cooperation from the client, and costing one path segment the stub strips
    before dispatch. A CLI recorded as rejecting a path prefix gets the bare form
    and falls back to the coarse ``(model, surface, pattern)`` tuple.
    """
    port = os.environ.get("MLLM_STUB_PORT", "8099")
    root = f"http://127.0.0.1:{port}"
    if row.get("session_prefix", True):
        return f"{root}/s/{exec_id}/v1"
    return f"{root}/v1"


def build_child_env(cli: str, exec_id: str, row: dict, invocation_root: Path) -> dict:
    """Build the child's environment **from scratch** (rules 1-3)."""
    env: dict[str, str] = {}

    # 1. Start empty; carry over an allowlist.
    for key, value in os.environ.items():
        if key in ENV_CARRY_EXACT or key.startswith(ENV_CARRY_PREFIX):
            env[key] = value

    # 3. Point every config root at a fresh per-invocation directory. Done before
    #    rule 2 so a mechanism that writes a config file lands inside it.
    home = invocation_root
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["XDG_STATE_HOME"] = str(home / ".local" / "state")
    env["TMPDIR"] = str(home / "tmp")
    for var, rel in (row.get("config_roots") or {}).items():
        env[var] = str(home / rel)

    # 2. Inject only FAKE stub credentials. A CLI that refuses to start without a
    #    key finds one; a CLI that ships it to a vendor ships a string that buys
    #    nothing, because it is meaningful only on 127.0.0.1:$MLLM_STUB_PORT.
    session_key = f"mllm-stub-{exec_id}"
    base_url = stub_base_url(exec_id, row)
    for var in (row.get("credential_vars") or []):
        env[var] = session_key
    # Two spellings, because the CLIs disagree about where the version segment
    # belongs: aider/goose want the `/v1` base, while claude-code and gemini take
    # the bare origin and append their own path. Both are substituted here and in
    # the seeded config files, so a row can use whichever its CLI expects.
    url_tokens = {
        "{base_url}": base_url,
        "{base_url_root}": base_url.rsplit("/v1", 1)[0],
    }
    for var, template in (row.get("base_url_vars") or {}).items():
        if not template:
            env[var] = base_url
            continue
        value = template
        for token, replacement in url_tokens.items():
            value = value.replace(token, replacement)
        env[var] = value

    # Static per-row environment: the image sets CLINE_NO_AUTO_UPDATE=1,
    # OPENCODE_DISABLE_MODELS_FETCH=1 and friends image-wide, but this function
    # builds the child's environment FROM SCRATCH and carries over only the
    # allowlist — so an image-wide variable reaches the entrypoint and the build
    # and never the CLI it was set for. Without this the pins C9 verifies at build
    # time are silently un-held at run time, which is the failure mode where a
    # green build is followed by a CLI that updated itself mid-row.
    for var, value in (row.get("env_extra") or {}).items():
        env[var] = str(value)

    env["MLLM_EXEC_ID"] = exec_id
    env["MLLM_STUB_BASE_URL"] = base_url
    return env


def seed_invocation_root(cli: str, root: Path, row: dict) -> None:
    """Create the per-invocation config root, seeded from the per-CLI home.

    Rule 3 closes a hole that has nothing to do with secrets. Step 2's "write a
    config file" writes into whatever ``$HOME`` it inherits — which in a host-lane
    run is the ``$HOME`` the entrypoint wired for the **host**. Without a private
    root, a provider ``codex`` rewrites the host's ``~/.codex/config.toml``
    mid-run, and two provider ``codex`` subprocesses in one fan-out race each
    other for it: a corrupted host brain and a nondeterministic provider brain
    from the same bug. Seeding from the per-CLI home is what carries the
    synthesized credential file the ``auth_check`` probes read, so two concurrent
    dispatches of the same CLI both still pass their probe.
    """
    source = HOMES_ROOT / cli
    root.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, root, dirs_exist_ok=True)
    for rel in (".config", ".local/share", ".cache", "tmp"):
        (root / rel).mkdir(parents=True, exist_ok=True)

    # Substitute the PER-EXEC base URL into the copied seed templates. This has to
    # happen here rather than once in the entrypoint, and the difference is the
    # session key: the entrypoint knows only the stub's origin, while the exec id
    # that keys the turn counter is minted per invocation. Six of the eleven
    # verified CLIs carry their base URL in a config FILE rather than an env var
    # (codex, cline, kilocode, pi, opencode, muse's auth), so without this they
    # would all collapse onto the coarse `(model, surface, pattern)` session key —
    # and muse, which makes 17 requests per generate call, cannot be disambiguated
    # that way at all.
    base_url = os.environ.get("MLLM_STUB_BASE_URL", "")
    replacements = {
        "{{base_url}}": base_url,
        "{{base_url_root}}": base_url.rsplit("/v1", 1)[0],
        "{{stub_session_key}}": f"mllm-stub-{Path(root).name.rsplit('-', 1)[-1]}",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "{{" not in text:
            continue
        for token, value in replacements.items():
            text = text.replace(token, value)
        path.write_text(text, encoding="utf-8")

    # A mechanism that needs bytes on disk rather than a variable renders them here.
    for rel, template in (row.get("config_files") or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            template.replace("{base_url}", os.environ.get("MLLM_STUB_BASE_URL", "")),
            encoding="utf-8",
        )


def write_call_record(cli: str, argv: list[str], env: dict, exec_id: str,
                      invocation_root: Path) -> None:
    """One JSONL line per invocation, flushed, redacted at write time."""
    log = os.environ.get("MLLM_CALL_LOG")
    if not log:
        return
    captured = {
        key: value for key, value in env.items()
        if key.startswith(ENV_CAPTURE_ALLOW) or key.endswith(ENV_CAPTURE_SUFFIX)
    }
    record = {
        "cli": cli,
        "exec_id": exec_id,
        "row_id": os.environ.get("MLLM_ROW_ID"),
        "t": round(time.time(), 3),
        "cwd": os.getcwd(),
        "argv": collapse_prompt(redact_argv(argv)),
        # The SCRUB RECEIPT: variable NAMES only, never values, plus the config
        # root. A row whose receipt shows a name outside the carry-over allowlist,
        # or a config root that is the host's, fails — that is the detection
        # posture applied to the credential rather than to the route.
        "child_env_names": sorted(env),
        "captured_env": {k: v for k, v in captured.items() if not k.endswith("_KEY")},
        "config_root": str(invocation_root),
    }
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


# --------------------------------------------------------------------------- #
# The `uv` wrapper: receipts (C6c)
# --------------------------------------------------------------------------- #

def run_uv(argv: list[str]) -> int:
    """Tee the dispatcher's stdout and write a run-id-correlated receipt.

    Every shim's dispatcher command is ``uv run --project "{SKILL_DIR}" -- python
    multi_llm.py …``, resolved **by name** through PATH, so this one wrapper sees
    every dispatcher invocation any host makes — including the ones a host makes
    that we did not ask for.

    The shim tells the *agent* to read ``MULTI_LLM_RESULT`` out of its own bash
    tool result. What reaches the harness is the host's event stream, in which
    that line is at best a JSON-escaped field inside a nested tool-result
    envelope, and at worst truncated or absent because the agent paraphrased
    instead of quoting. Asserting on it there would turn every host row into a
    test of a model's willingness to echo a string. So the receipt is the anchor:
    a host that ran our command and narrated it badly still passes, while a host
    that never ran it produces **no receipt at all** and fails naming the row.
    """
    receipt_dir = os.environ.get("MLLM_RECEIPT_DIR")
    real = str(REAL_DIR / "uv")

    env = dict(os.environ)
    # Offline by environment rather than by discipline: a `uv run` that decides it
    # needs to sync has no index to reach under --network none, and the row dies
    # in resolution with a stack that looks nothing like the bug it is.
    env.setdefault("UV_OFFLINE", "1")
    env.setdefault("UV_PROJECT_ENVIRONMENT", "/opt/mllm/skill/.venv")
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")

    forwarded = list(argv)
    if forwarded[:1] == ["run"]:
        for flag in ("--frozen", "--no-sync"):
            if flag not in forwarded:
                forwarded.insert(1, flag)

    started = time.time()
    proc = subprocess.Popen([real, *forwarded], env=env, stdout=subprocess.PIPE,
                            stderr=None, text=True, bufsize=1)
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    code = proc.wait()

    if receipt_dir:
        _write_receipt(Path(receipt_dir), forwarded, captured, code, started)
    return code


def _write_receipt(receipt_dir: Path, argv: list[str], captured: list[str],
                   code: int, started: float) -> None:
    """Parse the last result line with the product's OWN reader, not a second one."""
    receipt_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/opt/mllm/skill")
    payload = None
    try:
        from utils.cli_common import parse_result_line  # type: ignore

        for line in reversed(captured):
            parsed = parse_result_line(line)
            if parsed is not None:
                payload = parsed
                break
    except Exception:  # noqa: BLE001 - a receipt is evidence, never a failure path
        payload = None

    # Correlate launcher and worker BY RUN ID, not by adjacency: `run.state` and
    # the top-level phase state are separate machines on purpose, and a --detach
    # invocation returns 0 with `state: done` before the workflow has run at all.
    run_block = None
    if isinstance(payload, dict):
        for artifact in payload.get("artifacts") or []:
            status = Path(str(artifact))
            status = status if status.name == ".status.json" else status.parent / ".status.json"
            if status.is_file():
                try:
                    block = json.loads(status.read_text(encoding="utf-8")).get("run")
                except (OSError, ValueError):
                    block = None
                if isinstance(block, dict):
                    run_block = {k: block.get(k) for k in
                                 ("run_id", "state", "exit_code")}
                    break

    fd, seq = _claim_receipt_slot(receipt_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(
            {
                "row_id": os.environ.get("MLLM_ROW_ID"),
                "seq": seq,
                "argv": collapse_prompt(redact_argv(argv)),
                "cwd": os.getcwd(),
                "exit": code,
                # A dispatcher invocation that emitted NO line writes a receipt
                # with result: null, which is itself a finding rather than a
                # silent gap — every mode emits one.
                "result": payload,
                "run": run_block,
                "started_at": round(started, 3),
                "ended_at": round(time.time(), 3),
            },
            indent=2,
            sort_keys=True,
        ))


def _claim_receipt_slot(receipt_dir: Path) -> tuple[int, int]:
    """Claim the next free ``NNN.receipt.json`` atomically: ``(fd, seq)``.

    The glob count is where the search STARTS, never the claim. Receipts are not
    written by one serialized writer: on a detaching row the detached worker
    writes its own receipt through this same wrapper long after the launcher
    returned, so it can land at the same instant as the agent's `status --wait`
    invocation. Two processes counting the same directory compute the same seq,
    and the second write silently truncates the first — losing a receipt that C8
    makes the row's verdict anchor, and reporting a host that *did* run our
    command as one that never did.

    ``O_CREAT | O_EXCL`` makes the claim atomic against any number of concurrent
    wrappers; a loser simply advances to the next number. 0644 rather than the
    executor's usual 0600 because these are read back on the HOST, by a developer
    whose uid is not the container's 10001.
    """
    seq = len(list(receipt_dir.glob("*.receipt.json"))) + 1
    while True:
        try:
            fd = os.open(receipt_dir / f"{seq:03d}.receipt.json",
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            seq += 1
            continue
        return fd, seq


# --------------------------------------------------------------------------- #

def inject_argv(argv: list[str], row: dict, base_url: str) -> list[str]:
    """Insert a redirect flag into the child's argv (measured: `muse`).

    The three mechanisms C2 anticipated were an env var, a config file, or both.
    `muse` has neither: its base-URL override is **argv-only** (`muse exec
    --base-url`), and Phase 1 proved the negative for six config keys across
    `auth.json` and `settings.json` — all six left the request on the real Meta
    endpoint with zero stub hits. Since `MuseProvider.build_command` emits no
    `--base-url`, the wrapper is the only place the flag can come from without
    changing shipping code to carry a test concern.

    Injection is positional rather than appended: the prompt is the last argv
    element on most adapters, so appending would put a flag after it.
    """
    spec = row.get("argv_inject")
    if not spec:
        return argv
    extra = [token.replace("{base_url}", base_url) for token in spec.get("args", [])]
    if not extra:
        return argv
    after = spec.get("after_token")
    index = 0
    if after and after in argv:
        index = argv.index(after) + 1
    return argv[:index] + extra + argv[index:]


def run_child(real: Path, binary: str, argv: list[str], env: dict,
              invocation_root: Path) -> int:
    """Run the real binary as a child, then remove the per-invocation root.

    C2 rule 3 makes the per-invocation directory "created before the exec and
    removed after it", and ``os.execve`` leaves nobody to do the removing: from
    that call on, the wrapper process *is* the CLI. Nothing else prunes
    ``/tmp/mllm-inv/<cli>-<exec_id>`` during a row, and every exec copies the
    whole per-CLI home into it — for opencode the ~63 MB baked ``node_modules``
    tree — onto a tmpfs whose pages count against the container's memory limit.
    A long chain then accumulates gigabytes of dead config roots and dies of OOM
    somewhere that looks nothing like the bug.

    So the wrapper forks. What the fork has to preserve is everything the exec
    gave for free:

    * **argv[0], fds, cwd and the terminal** — inherited unchanged, which is why
      nothing is piped here. ``run_uv`` tees because it must read the result
      line; this path must not touch the stream at all, or a CLI that renders a
      TUI, or whose adapter reads its stdout, sees a different process than the
      one the product ships to.
    * **signals** — the executor spawns provider CLIs with ``start_new_session``,
      so a ``kill_tree`` SIGTERM reaches the whole group and hits the child
      directly. The forwarding handler covers the narrower case where only the
      wrapper's pid is signalled: without it the child would outlive its parent
      and leak the very root this function exists to remove.
    * **the exit status** — a signalled child is reported as ``128 + signal``,
      the shell convention, so a timeout still reads as a kill and not as a
      clean exit.

    A SIGKILLed wrapper is the one residual leak, and it is bounded by the
    container's lifetime — the tmpfs dies with it.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # the child: never return, never run an atexit handler
        try:
            os.execve(str(real), [binary, *argv], env)
        except BaseException:  # noqa: BLE001 - the child has no caller to raise to
            os._exit(127)

    def _forward(signum, _frame):
        try:
            os.kill(pid, signum)
        except OSError:
            pass

    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        try:
            previous[signum] = signal.signal(signum, _forward)
        except (OSError, ValueError):  # pragma: no cover - not a main thread
            pass
    try:
        _, status = os.waitpid(pid, 0)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        shutil.rmtree(invocation_root, ignore_errors=True)

    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


def main(argv: list[str]) -> int:
    binary = Path(sys.argv[0]).name
    if binary == "uv":
        return run_uv(argv)

    # The real binary is found by the name we were invoked as; the row and the
    # seeded home are found by the provider id, which is not always the same
    # string (`claude` -> `claude-code`).
    real = REAL_DIR / binary
    if not real.exists():
        sys.stderr.write(f"mllm-wrapper: no real binary at {real}\n")
        return 127

    cli = resolve_cli_id(binary)
    exec_id = secrets.token_hex(6)
    row = load_row(cli)
    invocation_root = Path(os.environ.get("MLLM_INVOCATION_ROOT",
                                          "/tmp/mllm-inv")) / f"{cli}-{exec_id}"
    base_url = stub_base_url(exec_id, row)
    os.environ["MLLM_STUB_BASE_URL"] = base_url
    seed_invocation_root(cli, invocation_root, row)
    env = build_child_env(cli, exec_id, row, invocation_root)
    argv = inject_argv(argv, row, base_url)
    write_call_record(cli, [str(real), *argv], env, exec_id, invocation_root)

    # argv[0] is the BINARY name the caller used, never the provider id: a CLI
    # that inspects its own argv[0] (for usage text, or to pick a mode) must see
    # what it would have seen unwrapped.
    return run_child(real, binary, argv, env, invocation_root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
