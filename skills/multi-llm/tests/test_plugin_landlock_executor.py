"""The plugin-landlock rung in the executor lane (plan codex-sandbox-unavailable).

Plan ``codex-sandbox-unavailable-findings-and-fixes`` §6 F1 ("Where it sits",
"Dispatch shape", "How a dispatch receives its grant", "Write profiles on the
rung", Limits), §6 F5, §7 steps 2–4 and 6, §8.2 and §8.3 items 3, 7, 10 and 13
(2026-09-30). What this file pins is the **executor's** share:

* the walk: one outcome per scripted wrapper result (taken, absent on 126/125
  and on a refused grant, rejected on a landed write or a changed seed,
  unavailable on a failed positive control), the stop rules (timeout,
  unstartable, spent deadline, setup failure) that never reach the
  uncontained rung, the one shared deadline, and the skip switch;
* the gate: ``contained`` and ``off`` never reach the rung (``off`` spawns no
  rung probe), ``agent-write`` takes the rung alone and only under ``auto``,
  ``repo-wide`` takes it with the enumeration's controls (step 3b), F5's
  consent comes below the rung for the step-3 scopes alone and is
  never carried over, and the step-4 refusal leads with F6 and names its case;
* the spawn: the wrapped argv for every read kind, no prefix off the rung,
  a fresh ``TMPDIR`` per attempt that is gone afterwards, the process-group
  kill reaching the exec'd child, the ``agent-write`` backstop, a
  ``[JAIL_DENIED]`` line that is never retried, an ordinary permission error,
  command-output denials that fail nothing and the warn-only home check;
* previews, budgets, the kind→scope map and path-free caches.

Tier 1 throughout (§8.1): the wrapper is a fake under ``tmp_path``
(``landlock_dispatch.WRAPPER_PATHS`` is monkeypatched), which the suite's
probe guard runs for real, and codex is the repo's mock launcher. No kernel,
no real codex, no network.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import landlock_dispatch  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers.base import (  # noqa: E402
    HOST_SANDBOX_POSTURE_ENV,
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    SANDBOX_CANARY_DENIED,
    SANDBOX_CANARY_NOT_DENIED,
    SANDBOX_FALLBACK_PROFILES,
    SKIP_RUNG_ENV,
    JailRequest,
    WritePostureConsent,
)
from utils.providers.codex import (  # noqa: E402
    PLUGIN_LANDLOCK_RUNG,
    UNSANDBOXED_RUNG,
    CodexProvider,
)
from utils.subagent_tasks import READ_KINDS, WRITE_KINDS, build_inputs, make_write_task  # noqa: E402
from tests.test_subagent_executor import (  # noqa: E402,F401
    _pid_alive,
    _validation_batch,
    runner_env,
)

BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
F6_LEAD = (
    "To run this now, set `driver.subagent_runner` to a runner that is not "
    "Codex (for example `claude-code:<model>`) in `.multi-llm/providers.yaml`."
)
ABSENT_NOTE = "landlock_exec: unavailable: Landlock ABI none (ENOSYS); the floor is 3"
ABI2_NOTE = "landlock_exec: unavailable: Landlock ABI 2 found; the floor is 3"
REQUIRED_NOTE = "landlock_exec: required-entry: dir:/nowhere/codex-home: missing"


# ---------------------------------------------------------------------------
# The fake wrapper (§8.1 tier 1): records its argv and env, then does what the
# test scripted — for a probe (`codex sandbox …`) and for a dispatch apart.
# ---------------------------------------------------------------------------

FAKE_WRAPPER = r'''
import json, os, subprocess, sys, time
LOG = {log!r}
MODES = {modes!r}
args = sys.argv[1:]
sep = args.index("--")
allow, cmd = args[:sep], args[sep + 1:]
probe = len(cmd) > 1 and cmd[1] == "sandbox"
record = {{
    "argv": sys.argv, "orig_argv": getattr(sys, "orig_argv", None),
    "executable": sys.executable, "allow": allow, "cmd": cmd, "probe": probe,
    "pid": os.getpid(), "pgid": os.getpgid(0),
    "env": {{k: os.environ.get(k) for k in (
        "TMPDIR", "MULTI_LLM_CANARY_SEEDED", "MULTI_LLM_CANARY_WRITE_FILE",
        "MULTI_LLM_CANARY_WRITE_DIR", "MULTI_LLM_CANARY_DENY_DIRS", "CODEX_HOME",
        "GIT_OPTIONAL_LOCKS", "UV_CACHE_DIR", "PYTEST_ADDOPTS")}},
    "tmpdir_exists": os.path.isdir(os.environ.get("TMPDIR") or "/nonexistent"),
}}
with open(LOG, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
with open(MODES, encoding="utf-8") as handle:
    modes = json.load(handle)
mode = modes["probe" if probe else "dispatch"]
kind, _, arg = mode.partition(":")

def say(text, code, err=True):
    print(text, file=sys.stderr if err else sys.stdout, flush=True)
    sys.exit(code)

if kind == "exec":
    os.execvp(cmd[0], cmd)
if kind == "deny":
    say({denied!r}, 0, err=False)
if kind == "absent":
    say({absent!r}, 126)
if kind == "abi2":
    say({abi2!r}, 126)
if kind == "required":
    say({required!r}, 125)
if kind == "not_denied":
    say({not_denied!r}, 3)
if kind == "seed":
    os.truncate(os.environ["MULTI_LLM_CANARY_SEEDED"], 0)
    say({denied!r}, 0, err=False)
if kind == "positive":
    say("multi-llm sandbox canary: an allowed write failed", 5)
if kind == "exit0":
    sys.exit(0)
if kind == "startup":
    say("Error: codex could not start: Permission denied (os error 13)", 1)
if kind == "hang":
    time.sleep(30)
    sys.exit(0)
if kind == "fail":
    sys.exit(1)
if kind == "eacces":
    say("Error: failed to write " + arg.replace("$TMPDIR", os.environ.get("TMPDIR", "")) +
        ": Permission denied (os error 13)", 1)
if kind == "eacces_nopath":
    say("Error: Permission denied (os error 13)", 1)
if kind == "tree":
    os.execvp("sh", ["sh", "-c", "sleep 300 & echo $! > " + arg + "; wait"])
if kind == "home":
    home = os.environ["CODEX_HOME"]
    with open(os.path.join(home, "config.toml"), "a", encoding="utf-8") as handle:
        handle.write(arg.replace("\\n", "\n"))
    os.execvp(cmd[0], cmd)
if kind == "link":
    # uv traffic on a shared inode: a name made and removed (ctime only).
    time.sleep(0.02)
    os.link(arg, arg + ".spare")
    os.unlink(arg + ".spare")
    os.execvp(cmd[0], cmd)
if kind == "linkwrite":
    # The same inode rewritten at the same size, mtime put back.
    info = os.stat(arg)
    with open(arg, "r+b") as handle:
        data = handle.read()
        handle.seek(0)
        handle.write(data.replace(b"6.0.3", b"6.6.6"))
    os.utime(arg, ns=(info.st_atime_ns, info.st_mtime_ns))
    os.execvp(cmd[0], cmd)
if kind == "cmdeny":
    done = subprocess.run(cmd, capture_output=True, text=True)
    item = {{"type": "item.completed", "item": {{
        "type": "command_execution", "command": "sh -c 'echo x > /etc/stray'",
        "aggregated_output": "sh: 1: cannot create " + arg + ": Permission denied\n",
        "exit_code": 2, "status": "failed"}}}}
    print(json.dumps(item), flush=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    sys.exit(done.returncode)
say("fake wrapper: unknown mode " + mode, 99)
'''


class FakeWrapper:
    """A fake ``landlock_exec.py`` under ``tmp_path`` and its JSONL log."""

    def __init__(self, root: Path, monkeypatch, probe: str = "deny", dispatch: str = "exec"):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "landlock_exec.py"
        self.log = root / "wrapper.jsonl"
        self.modes = root / "modes.json"
        self.set(probe=probe, dispatch=dispatch)
        self.path.write_text(FAKE_WRAPPER.format(
            log=str(self.log), modes=str(self.modes),
            denied=f"{SANDBOX_CANARY_DENIED}", absent=ABSENT_NOTE, abi2=ABI2_NOTE,
            required=REQUIRED_NOTE, not_denied=SANDBOX_CANARY_NOT_DENIED,
        ), encoding="utf-8")
        monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(self.path))

    def set(self, *, probe: str = None, dispatch: str = None) -> None:
        current = {"probe": "deny", "dispatch": "exec"}
        if self.modes.exists():
            current = json.loads(self.modes.read_text(encoding="utf-8"))
        if probe is not None:
            current["probe"] = probe
        if dispatch is not None:
            current["dispatch"] = dispatch
        self.modes.write_text(json.dumps(current), encoding="utf-8")

    def records(self, probe=None):
        if not self.log.exists():
            return []
        rows = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        return [r for r in rows if probe is None or r["probe"] is probe]


def install_fake_wrapper(root: Path, monkeypatch, **modes) -> FakeWrapper:
    """For other test files: a scripted fake wrapper in place of the real one."""
    return FakeWrapper(root, monkeypatch, **modes)


@pytest.fixture
def jail_home(tmp_path, monkeypatch):
    """A temp ``$HOME`` with an existing ``$CODEX_HOME`` inside it (the rung
    refuses a missing home and never creates one)."""
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    return codex_home


@pytest.fixture
def repo(tmp_path, monkeypatch, jail_home):
    root = tmp_path / "repo"
    (root / "plans").mkdir(parents=True)
    (root / "plans" / "plan.md").write_text("# Plan\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.delenv(SKIP_RUNG_ENV, raising=False)
    monkeypatch.delenv(HOST_SANDBOX_POSTURE_ENV, raising=False)
    return root


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    return FakeWrapper(tmp_path / "wrapper", monkeypatch)


def _slot_dirs():
    base = os.path.realpath(tempfile.gettempdir())
    return {name for name in os.listdir(base) if name.startswith(landlock_dispatch.SLOT_PREFIX)}


# ---------------------------------------------------------------------------
# A stand-in adapter: codex's rungs, a broken standard probe, a policy word.
# ---------------------------------------------------------------------------

class _JailLadder:
    """Codex's ladder around the real plugin-landlock rung, with a standard
    probe that fails like bwrap does where it cannot start."""

    internal_retries = None
    name = "fakecodex"

    def __init__(self, policy="auto", opt_in=True, profile_probe=None):
        self.policy = policy
        self.opt_in = opt_in
        self.profile_probe = profile_probe or f"echo '{BWRAP_LINE}' >&2; exit 1"

    def sandbox_probe_command(self, profile):
        return ["sh", "-c", self.profile_probe]

    def sandbox_fallback_policy(self, profile, anchor=None):
        if profile in SANDBOX_FALLBACK_PROFILES:
            return self.policy
        if profile == PROFILE_AGENT_WRITE and self.opt_in:
            return self.policy
        return None

    def sandbox_fallback_rungs(self, profile):
        if profile in SANDBOX_FALLBACK_PROFILES:
            return (PLUGIN_LANDLOCK_RUNG, UNSANDBOXED_RUNG)
        if profile == PROFILE_AGENT_WRITE and self.opt_in:
            return (PLUGIN_LANDLOCK_RUNG,)
        return ()


def _plan_request(repo_root: Path, *, scope="plan-file", kind="apply_suggestions"):
    plan = repo_root / "plans" / "plan.md"
    return JailRequest(
        repo_root=str(repo_root), kind=kind, write_scope=scope,
        target=str(plan) if scope != "repo-wide" else None,
        output_dir=str(repo_root / "plans" / "plan") if scope == "repo-wide" else None,
    )


@pytest.fixture
def gate(monkeypatch):
    """`_detect_sandbox` with a stand-in adapter and a clean cache."""

    def run(adapter, profile=PROFILE_AGENT_READONLY, request=None):
        monkeypatch.setattr(executor, "get_provider", lambda _name: adapter)
        kwargs = {"jail_request": request} if request is not None else {}
        return executor._detect_sandbox(
            "fakecodex", key="driver.subagent_runner", spec="fakecodex:m",
            profile=profile, **kwargs,
        )
    return run


# ---------------------------------------------------------------------------
# The walk (F1 "Where it sits" item 3)
# ---------------------------------------------------------------------------

def _walk(repo, **kwargs):
    tails = []
    rung, walk = executor._walk_fallback_rungs(
        (PLUGIN_LANDLOCK_RUNG, UNSANDBOXED_RUNG), cwd=str(repo),
        deadline=kwargs.pop("deadline", time.monotonic() + 10), tails=tails, **kwargs,
    )
    return rung, walk, dict(tails)


class TestTheWalk:
    def test_a_passing_probe_takes_the_rung_and_cleans_up(self, repo, wrapper):
        before = _slot_dirs()
        rung, walk, tails = _walk(repo)
        assert (rung, walk) == ("plugin-landlock", (("plugin-landlock", "taken"),))
        [probe] = wrapper.records(probe=True)
        # The probe is wrapped exactly like a dispatch: the interpreter under
        # -I -S, the wrapper, the allow entries, `--`, then codex's pinned tail.
        assert probe["orig_argv"][:4] == [sys.executable, "-I", "-S", str(wrapper.path)]
        assert probe["cmd"][1:] == list(PLUGIN_LANDLOCK_RUNG.probe_argv[1:])
        assert "--allow" in probe["allow"]
        assert probe["tmpdir_exists"] is True
        assert f"dir:{probe['env']['TMPDIR']}" in probe["allow"]
        # The seed, the probe's TMPDIR and the walk's own scratch dir are gone.
        assert not os.path.exists(probe["env"]["MULTI_LLM_CANARY_SEEDED"])
        assert not os.path.exists(probe["env"]["TMPDIR"])
        assert not (repo / ".multi-llm").exists()
        assert _slot_dirs() <= before

    @pytest.mark.parametrize("mode, outcome, note", [
        ("absent", "absent", ABSENT_NOTE),
        ("abi2", "absent", ABI2_NOTE),
        ("required", "absent", REQUIRED_NOTE),
        ("not_denied", "rejected", None),
        ("seed", "rejected", None),
        ("positive", "unavailable", None),
        ("exit0", "absent", None),
        ("startup", "unavailable", None),
    ])
    def test_a_measured_failure_moves_on_to_the_next_rung(
        self, repo, wrapper, mode, outcome, note,
    ):
        wrapper.set(probe=mode)
        rung, walk, tails = _walk(repo)
        assert walk == (("plugin-landlock", outcome), ("unsandboxed", "taken"))
        assert rung == "unsandboxed"
        if note:
            assert tails["plugin-landlock"] == note

    def test_a_refused_grant_is_absent_and_names_the_home(self, repo, wrapper, monkeypatch, tmp_path):
        missing = tmp_path / "no-such-home"
        monkeypatch.setenv("CODEX_HOME", str(missing))
        rung, walk, tails = _walk(repo)
        assert walk[0] == ("plugin-landlock", "absent")
        assert str(missing) in tails["plugin-landlock"] or "no-such-home" in tails["plugin-landlock"]
        assert "codex login" in tails["plugin-landlock"]
        assert not missing.exists()
        assert wrapper.records() == []

    def test_a_timeout_stops_the_walk_with_no_rung(self, repo, wrapper):
        wrapper.set(probe="hang")
        rung, walk, _ = _walk(repo, deadline=time.monotonic() + 0.5)
        assert rung is None
        assert walk in ((("plugin-landlock", "unprobed"),), (("plugin-landlock", "unknown"),))
        assert "unsandboxed" not in dict(walk)

    def test_an_unstartable_probe_stops_the_walk(self, repo, wrapper, monkeypatch):
        monkeypatch.setattr(executor, "_run_sandbox_probe", lambda argv, **kw: None)
        rung, walk, _ = _walk(repo)
        assert (rung, walk) == (None, (("plugin-landlock", "unknown"),))

    def test_a_spent_deadline_starts_nothing(self, repo, wrapper):
        rung, walk, _ = _walk(repo, deadline=time.monotonic() - 1)
        assert (rung, walk) == (None, (("plugin-landlock", "unprobed"),))
        assert wrapper.records() == []

    def test_a_setup_failure_stops_the_walk(self, repo, wrapper, monkeypatch):
        def broken(repo_root=None):
            raise OSError("no temp space")

        monkeypatch.setattr(landlock_dispatch, "slot_tmpdir", broken)
        rung, walk, _ = _walk(repo)
        assert (rung, walk) == (None, (("plugin-landlock", "unknown"),))
        assert wrapper.records() == []

    def test_a_temp_base_inside_the_repo_stops_the_walk_never_falls_through(
        self, repo, wrapper, monkeypatch,
    ):
        """slot_tmpdir's ``tmpdir-in-repo`` refusal is a probe TMPDIR the
        parent cannot prepare: ``unknown`` with no rung, never ``absent`` —
        which let ``auto`` run a read-only dispatch on ``unsandboxed``."""
        inside = repo / ".tmp"
        inside.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(inside))
        rung, walk, tails = _walk(repo)
        assert (rung, walk) == (None, (("plugin-landlock", "unknown"),))
        assert "could not make the probe's TMPDIR" in tails["plugin-landlock"]
        assert wrapper.records() == []
        # Only the probe's mutex file, never unlinked by design, may remain.
        assert [n for n in os.listdir(inside)
                if not n.startswith(executor.PROBE_WORKDIR_LOCK_PREFIX)] == []

    def test_an_unwritable_negative_target_stops_the_walk(self, repo, wrapper):
        (repo / ".multi-llm").mkdir()
        (repo / ".multi-llm" / "tmp").write_text("not a dir", encoding="utf-8")
        rung, walk, _ = _walk(repo)
        assert (rung, walk) == (None, (("plugin-landlock", "unknown"),))
        assert wrapper.records() == []

    def test_the_rung_gets_what_is_left_of_the_one_deadline(self, repo, wrapper, monkeypatch):
        seen = []
        real = executor._run_sandbox_probe

        def spy(argv, **kwargs):
            seen.append((landlock_dispatch.is_wrapped(argv), kwargs["timeout"]))
            return real(argv, **kwargs)

        monkeypatch.setattr(executor, "_run_sandbox_probe", spy)
        adapter = _JailLadder(profile_probe=f"sleep 0.6; echo '{BWRAP_LINE}' >&2; exit 1")
        verdict = executor.sandbox_verdict(adapter, PROFILE_AGENT_READONLY, cwd=str(repo), timeout=2.0)
        assert verdict.rung == "plugin-landlock"
        (first_wrapped, first), (rung_wrapped, rung_budget) = seen
        assert first == 2.0 and not first_wrapped
        assert rung_wrapped and 0 < rung_budget <= 1.45

    def test_the_skip_switch_skips_the_rung_without_a_probe(self, repo, wrapper, monkeypatch):
        monkeypatch.setenv(SKIP_RUNG_ENV, "plugin-landlock")
        rung, walk, _ = _walk(repo)
        assert walk == (("plugin-landlock", "skipped-by-switch"), ("unsandboxed", "taken"))
        assert executor.RUNG_SKIPPED_BY_SWITCH == "skipped-by-switch"
        assert executor.rung_retirements(executor.SandboxProbeResult("unavailable", "", rung, walk)) == []
        assert wrapper.records() == []

    def test_an_unknown_skip_value_is_refused_never_ignored(self, repo, wrapper, monkeypatch, gate):
        monkeypatch.setenv(SKIP_RUNG_ENV, "landlock")
        with pytest.raises(executor.RunnerRejected) as excinfo:
            gate(_JailLadder())
        assert SKIP_RUNG_ENV in str(excinfo.value)
        assert wrapper.records() == []


# ---------------------------------------------------------------------------
# The gate (F1 items 4 and 6; step 3; step 6)
# ---------------------------------------------------------------------------

class TestTheReadOnlyGate:
    def test_auto_takes_the_rung_with_one_incidental_line(self, repo, wrapper, gate, capsys):
        assert gate(_JailLadder()) == "plugin-landlock"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_READONLY) == "plugin-landlock"
        out = capsys.readouterr().out
        assert out.count("[SANDBOX_INCIDENTAL]") == 1
        assert "[SANDBOX_BYPASSED]" not in out and "[SANDBOX_FALLBACK]" not in out
        line = next(ln for ln in out.splitlines() if "[SANDBOX_INCIDENTAL]" in ln)
        assert line.startswith("[exec-subagents] [SANDBOX_INCIDENTAL] fakecodex:")
        assert BWRAP_LINE in line and "'plugin-landlock'" in line
        assert "read-only phase" in line
        # The executor lane runs no mode check, and its home check only warns.
        assert "a changed file mode is neither stopped nor checked" in line
        assert "reported after this dispatch, not failed" in line
        assert "contained" not in line and "sandboxed" not in line
        # Replayed, never re-printed.
        assert gate(_JailLadder()) == "plugin-landlock"
        assert capsys.readouterr().out == ""

    def test_contained_refuses_when_only_the_rung_is_usable(self, repo, wrapper, gate):
        with pytest.raises(executor.RunnerRejected) as excinfo:
            gate(_JailLadder(policy="contained"))
        message = str(excinfo.value)
        assert message.startswith(F6_LEAD)
        assert "does not take this job: providers.fakecodex.sandbox_fallback=contained" in message
        assert executor._SANDBOX_DECISIONS == {}

    def test_off_spawns_no_rung_probe(self, repo, wrapper, gate):
        with pytest.raises(executor.RunnerRejected) as excinfo:
            gate(_JailLadder(policy="off"))
        assert "sandbox_fallback=off reaches no fallback rung" in str(excinfo.value)
        assert wrapper.records() == []
        assert executor._SANDBOX_DECISIONS == {}

    def test_absent_falls_to_the_uncontained_rung_under_auto(self, repo, wrapper, gate, capsys):
        wrapper.set(probe="absent")
        assert gate(_JailLadder()) == "bypassed"
        out = capsys.readouterr().out
        assert out.count("[SANDBOX_BYPASSED]") == 1
        assert "fallback rung 'plugin-landlock' absent" in out

    def test_a_stopped_walk_refuses_and_never_renders_the_uncontained_rung(
        self, repo, wrapper, gate, monkeypatch,
    ):
        wrapper.set(probe="hang")
        monkeypatch.setattr(executor, "SANDBOX_PROBE_TIMEOUT_S", 0.8)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            gate(_JailLadder())
        assert "its probe gave no answer" in str(excinfo.value)
        assert executor._SANDBOX_DECISIONS == {}


class TestTheWriteGate:
    def test_auto_takes_the_rung_for_a_plan_file_grant(self, repo, wrapper, gate, capsys):
        request = _plan_request(repo)
        assert gate(_JailLadder(), PROFILE_AGENT_WRITE, request) == "plugin-landlock"
        assert executor.sandbox_rung(
            "fakecodex:m", PROFILE_AGENT_WRITE, write_scope="plan-file",
        ) == "plugin-landlock"
        # Never replayed for another scope, and never as a read decision.
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="repo-wide") is None
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE) is None
        assert any(key.endswith("|fakecodex:m|agent-write|plan-file") for key in executor._SANDBOX_DECISIONS)
        [probe] = wrapper.records(probe=True)
        # The positive control writes the grant's own target.
        assert probe["env"]["MULTI_LLM_CANARY_WRITE_FILE"] == os.path.realpath(request.target)
        assert f"file:{os.path.realpath(request.target)}" in probe["allow"]
        # The write line is the scheduler's, printed before each batch from
        # `write_rung_notice`; the preflight prints none of its own, so batch
        # 0 never shows it twice.
        assert "[SANDBOX_INCIDENTAL]" not in capsys.readouterr().out
        line = executor.write_rung_notice("fakecodex:m", request)
        assert "write phase, scope plan-file" in line and "read-only" not in line
        assert "network access is not confined" in line

    def test_the_scheduler_s_notice_names_scope_and_target(self, repo, wrapper, gate, monkeypatch):
        request = _plan_request(repo)
        monkeypatch.setattr(executor, "get_provider", lambda _name: _JailLadder())
        gate(_JailLadder(), PROFILE_AGENT_WRITE, request)
        line = executor.write_rung_notice("fakecodex:m", request)
        assert line.startswith("[SANDBOX_INCIDENTAL] fakecodex:")
        assert "scope plan-file" in line and "plan.md" in line
        other = _plan_request(repo, scope="tasks", kind="apply_task_suggestions")
        assert executor.write_rung_notice("fakecodex:m", other) is None
        assert executor.write_rung_notice("fakecodex:m", None) is None

    @pytest.mark.parametrize("leg", [
        "contained", "off", "opt-in-off", "absent", "abi2", "required",
        "not_denied", "seed", "positive", "startup", "exit0", "home-missing",
    ])
    def test_every_other_case_is_refused_and_never_falls_through(
        self, repo, wrapper, gate, monkeypatch, tmp_path, leg,
    ):
        adapter = _JailLadder()
        if leg in ("contained", "off"):
            adapter = _JailLadder(policy=leg)
        elif leg == "opt-in-off":
            adapter = _JailLadder(opt_in=False)
        elif leg == "home-missing":
            monkeypatch.setenv("CODEX_HOME", str(tmp_path / "gone"))
        else:
            wrapper.set(probe=leg)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            gate(adapter, PROFILE_AGENT_WRITE, _plan_request(repo))
        rejected = excinfo.value
        assert rejected.reason_code == "sandbox_unavailable"
        assert rejected.message.startswith(F6_LEAD)
        assert "The plugin's Landlock jail" in rejected.message
        assert rejected.message.endswith("Nothing was dispatched.")
        assert "install the AppArmor profile it prints" not in rejected.message
        assert executor._SANDBOX_DECISIONS == {}
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="plan-file") is None
        if leg in ("off", "opt-in-off"):
            assert wrapper.records() == []

    def test_a_timed_out_rung_probe_refuses_the_write(self, repo, wrapper, gate, monkeypatch):
        wrapper.set(probe="hang")
        monkeypatch.setattr(executor, "SANDBOX_PROBE_TIMEOUT_S", 0.8)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo))
        assert "its probe gave no answer" in str(excinfo.value)
        assert executor._SANDBOX_DECISIONS == {}

    def test_repo_wide_takes_the_rung_with_the_enumerations_controls(self, repo, wrapper, gate):
        """Step 3b: a repo-wide grant walks the rung like the step-3 scopes.
        Its probe hands the canary no positive target in the tree (the
        $TMPDIR write is the control, 2026-10-02) and its extra negative
        targets are the protected directories that exist (`.multi-llm/`
        here: the probe's own workdir makes it)."""
        (repo / "src").mkdir()
        request = _plan_request(repo, scope="repo-wide", kind="implement")
        assert gate(_JailLadder(), PROFILE_AGENT_WRITE, request) == "plugin-landlock"
        assert executor.sandbox_rung(
            "fakecodex:m", PROFILE_AGENT_WRITE, write_scope="repo-wide",
        ) == "plugin-landlock"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="plan-file") is None
        [probe] = wrapper.records(probe=True)
        assert probe["env"]["MULTI_LLM_CANARY_WRITE_DIR"] is None
        assert os.path.realpath(repo / ".multi-llm") in probe["env"]["MULTI_LLM_CANARY_DENY_DIRS"]
        assert probe["env"]["MULTI_LLM_CANARY_WRITE_FILE"] is None
        assert probe["env"]["UV_CACHE_DIR"] == os.path.join(probe["env"]["TMPDIR"], "uv-cache")
        assert f"dir:{os.path.realpath(repo / 'src')}" in probe["allow"]
        # `plans/` holds the output folder: descended, never ruled; the root never.
        assert f"dir:{os.path.realpath(repo / 'plans')}" not in probe["allow"]
        assert f"file:{os.path.realpath(repo / 'plans' / 'plan.md')}" in probe["allow"]
        assert f"dir:{os.path.realpath(repo)}" not in probe["allow"]
        line = executor.write_rung_notice("fakecodex:m", request)
        assert "write phase, scope repo-wide" in line and "read-only" not in line
        assert ".git, .codex, .multi-llm and the run's output folder" in line

    def test_plan_file_and_repo_wide_are_decided_separately(self, repo, wrapper, gate):
        assert gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo)) == "plugin-landlock"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="repo-wide") is None
        assert gate(_JailLadder(), PROFILE_AGENT_WRITE,
                    _plan_request(repo, scope="repo-wide", kind="implement")) == "plugin-landlock"
        assert len(wrapper.records(probe=True)) == 2
        assert sorted(k.rsplit("|", 1)[1] for k in executor._SANDBOX_DECISIONS) == ["plan-file", "repo-wide"]

    def test_a_standard_probe_that_passes_keeps_codex_s_own_sandbox(self, repo, wrapper, gate):
        assert gate(_JailLadder(profile_probe="exit 0"), PROFILE_AGENT_WRITE, _plan_request(repo)) == "ok"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="plan-file") is None
        assert wrapper.records() == []


class TestF5BelowTheRung:
    """Plan step 6: a full-access chat is consent — only below the jail."""

    def test_the_rung_still_comes_first(self, repo, wrapper, gate, monkeypatch):
        monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        assert gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo)) == "plugin-landlock"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="plan-file") == "plugin-landlock"

    def test_the_uncontained_rung_only_when_the_jail_cannot_take_it(
        self, repo, wrapper, gate, monkeypatch, capsys,
    ):
        wrapper.set(probe="absent")
        monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        assert gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo)) == "bypassed"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="plan-file") == "unsandboxed"
        assert executor.SANDBOX_DECISION_WRITE_BY_CONSENT in executor._SANDBOX_DECISIONS.values()
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "[SANDBOX_BYPASSED]" in ln)
        assert "F5" in line and "danger-full-access sandbox posture" in line
        assert "write phase" in line and "read-only phase" not in line
        # The decision never outlives the posture: without it, nothing is
        # replayed and the gate refuses.
        monkeypatch.delenv(HOST_SANDBOX_POSTURE_ENV)
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_WRITE, write_scope="plan-file") is None
        with pytest.raises(executor.RunnerRejected):
            gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo))

    @pytest.mark.parametrize("posture", [None, "workspace-write", "read-only", "DANGER-FULL-ACCESS"])
    def test_any_other_posture_leaves_the_refusal(self, repo, wrapper, gate, monkeypatch, posture):
        wrapper.set(probe="absent")
        if posture is not None:
            monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, posture)
        with pytest.raises(executor.RunnerRejected):
            gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo))

    @pytest.mark.parametrize("policy", ["contained", "off"])
    def test_never_under_contained_or_off(self, repo, wrapper, gate, monkeypatch, policy):
        wrapper.set(probe="absent")
        monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        with pytest.raises(executor.RunnerRejected):
            gate(_JailLadder(policy=policy), PROFILE_AGENT_WRITE, _plan_request(repo))
        assert executor._SANDBOX_DECISIONS == {}

    def test_never_on_a_walk_that_measured_nothing(self, repo, wrapper, gate, monkeypatch):
        wrapper.set(probe="hang")
        monkeypatch.setattr(executor, "SANDBOX_PROBE_TIMEOUT_S", 0.8)
        monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        with pytest.raises(executor.RunnerRejected):
            gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo))

    def test_never_for_repo_wide_edits(self, repo, wrapper, gate, monkeypatch):
        # Step 3b (2026-10-02) put repo-wide edits on the jail, never on the
        # uncontained rung: with the jail unusable the chat's posture does not
        # open rung 3 for them (`WRITE_SCOPES_BY_CONSENT`), the safer reading.
        wrapper.set(probe="absent")
        monkeypatch.setenv(HOST_SANDBOX_POSTURE_ENV, "danger-full-access")
        with pytest.raises(executor.RunnerRejected):
            gate(_JailLadder(), PROFILE_AGENT_WRITE,
                 _plan_request(repo, scope="repo-wide", kind="implement"))
        assert executor.SANDBOX_DECISION_WRITE_BY_CONSENT not in executor._SANDBOX_DECISIONS.values()


class TestStepFourWording:
    """Plan §7 "Step 4: the refusal's wording in each state"."""

    def _refusal(self, gate, repo, adapter=None, request=True):
        with pytest.raises(executor.RunnerRejected) as excinfo:
            gate(adapter or _JailLadder(), PROFILE_AGENT_WRITE,
                 _plan_request(repo) if request else None)
        return excinfo.value.message

    def test_the_parts_come_in_order(self, repo, wrapper, gate):
        wrapper.set(probe="absent")
        message = self._refusal(gate, repo)
        lead = message.index(F6_LEAD)
        reason = message.index("the runner's own sandbox cannot start under this repo")
        rung = message.index("The plugin's Landlock jail")
        tail = message.index("Probe tail:")
        end = message.index("Nothing was dispatched.")
        assert lead == 0 and lead < reason < rung < tail < end

    def test_kept_off_names_its_case(self, repo, wrapper, gate, monkeypatch):
        message = self._refusal(gate, repo, _JailLadder(opt_in=False))
        assert "does not take this job: write jobs are not enabled on it" in message
        message = self._refusal(gate, repo, request=False)
        assert "does not take this job: the write dispatch carries no write grant" in message
        executor.reset_sandbox_cache()
        monkeypatch.setenv(SKIP_RUNG_ENV, "plugin-landlock")
        message = self._refusal(gate, repo)
        assert f"does not take this job: skipped by {SKIP_RUNG_ENV}" in message

    @pytest.mark.parametrize("mode, note", [("absent", ABSENT_NOTE), ("abi2", ABI2_NOTE)])
    def test_landlock_unavailable_names_the_note_and_the_sudo_repair(self, repo, wrapper, gate, mode, note):
        wrapper.set(probe=mode)
        message = self._refusal(gate, repo)
        assert "cannot run here either: this kernel or container lacks Landlock (" in message
        assert note.split("landlock_exec: ", 1)[1] in message
        assert "which needs sudo" in message

    def test_a_failed_self_test_names_the_control_and_doctor_without_sudo(self, repo, wrapper, gate):
        wrapper.set(probe="positive")
        message = self._refusal(gate, repo)
        assert "present but failed its self-test (" in message
        assert "`plugin-landlock` row" in message
        assert "sudo" not in message

    def test_a_newer_codex_names_both_versions(self, repo, wrapper, gate, monkeypatch):
        wrapper.set(probe="not_denied")
        monkeypatch.setitem(executor._VERSION_CACHE, "fakecodex", "99.0.0")
        message = self._refusal(gate, repo)
        measured = PLUGIN_LANDLOCK_RUNG.measured_through[0]
        assert "99.0.0" in message and measured in message
        assert "likely cause" in message

    def test_the_nested_exec_refusal_leads_with_f6_too(self, monkeypatch):
        rejection = executor._sandbox_unavailable_rejection(
            key="driver.subagent_runner (host capabilities)", spec="codex:x",
            tail=BWRAP_LINE, trailer="Host: h. ",
        )
        assert rejection.message.startswith(F6_LEAD)
        assert "Fix:" not in rejection.message


# ---------------------------------------------------------------------------
# The spawn (F1 "Dispatch shape", "How a dispatch receives its grant")
# ---------------------------------------------------------------------------

def _codex():
    return registry.get_provider("codex")


def _spawn_on_rung(repo, kind, *, timeout=30, profile=PROFILE_AGENT_READONLY, request=None):
    return executor._spawn(
        provider=_codex(), model="gpt-5.5", prompt="Reply with []", profile=profile,
        transport="stdin", timeout=timeout, cwd=str(repo),
        sandbox_fallback="plugin-landlock",
        jail_request=request or JailRequest(repo_root=str(repo), kind=kind),
    )


class TestTheWrappedSpawn:
    @pytest.mark.parametrize("kind", sorted(READ_KINDS))
    def test_every_read_kind_spawns_the_wrapped_codex(self, repo, wrapper, runner_env, jail_home, kind):
        runner_env.use("codex:gpt-5.5")
        before = _slot_dirs()
        capture = _spawn_on_rung(repo, kind)
        assert capture.exit_code == 0, capture.stderr
        [row] = wrapper.records(probe=False)
        assert row["orig_argv"][:4] == [sys.executable, "-I", "-S", str(wrapper.path)]
        codex_path = str(runner_env.bin_dir / "codex")
        assert row["cmd"][0] == codex_path and row["cmd"][1] == "exec"
        assert row["cmd"][row["cmd"].index("--sandbox") + 1] == "danger-full-access"
        assert any(a.startswith("developer_instructions=") for a in row["cmd"])
        allow = row["allow"]
        assert allow[allow.index("--allow") + 1] == f"dir:{os.path.realpath(jail_home)}"
        assert f"dir:{row['env']['TMPDIR']}" in allow and "file:/dev/null" in allow
        assert row["tmpdir_exists"] is True
        assert row["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        # The capture records the wrapped argv; the TMPDIR is gone after.
        assert capture.argv[:4] == [sys.executable, "-I", "-S", str(wrapper.path)]
        assert capture.sandbox_rung == "plugin-landlock"
        assert not os.path.exists(row["env"]["TMPDIR"])
        assert _slot_dirs() <= before
        assert os.environ.get("TMPDIR") != row["env"]["TMPDIR"]

    def test_no_prefix_off_the_rung(self, repo, wrapper, runner_env):
        runner_env.use("codex:gpt-5.5")
        for fallback in (None, "unsandboxed"):
            capture = executor._spawn(
                provider=_codex(), model="gpt-5.5", prompt="Reply with []",
                profile=PROFILE_AGENT_READONLY, transport="stdin", timeout=30,
                cwd=str(repo), sandbox_fallback=fallback,
            )
            assert not landlock_dispatch.is_wrapped(capture.argv)
        assert wrapper.records() == []

    @pytest.mark.parametrize("state", ["ok", "unknown", "not-required"])
    def test_a_plain_standard_verdict_hands_the_dispatch_no_rung(self, repo, state):
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, state)
        assert executor.sandbox_rung("codex:gpt-5.5", PROFILE_AGENT_READONLY) is None

    def test_a_passing_standard_probe_never_reaches_the_wrapper(self, repo, wrapper, runner_env):
        """Standard sandbox first (user decision 2026-09-29): the mock's own
        `codex sandbox` passes, so the batch runs codex's own sandbox."""
        runner_env.use("codex:gpt-5.5")
        runner_env.echo("validation")
        phase = repo / "phase"
        phase.mkdir()
        report = executor.run_tasks(
            [_validation_batch(phase)], phase_dir=str(phase), kind="validation", cwd=str(repo),
        )
        assert report.ok, report.failure_summary()
        assert wrapper.records() == []

    def test_a_broken_standard_sandbox_runs_the_batch_on_the_rung(
        self, repo, wrapper, runner_env, monkeypatch, capsys,
    ):
        """Acceptance §8.3 item 3, tier 1: the batch's dispatch goes through the
        jail, its preflight printed one `[SANDBOX_INCIDENTAL]` line and no
        `[SANDBOX_BYPASSED]` line."""
        runner_env.use("codex:gpt-5.5")
        runner_env.echo("validation")
        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        phase = repo / "phase"
        phase.mkdir()
        report = executor.run_tasks(
            [_validation_batch(phase)], phase_dir=str(phase), kind="validation", cwd=str(repo),
        )
        assert report.ok, report.failure_summary()
        assert len(wrapper.records(probe=True)) == 1
        assert len(wrapper.records(probe=False)) == 1
        out = capsys.readouterr().out
        assert out.count("[SANDBOX_INCIDENTAL]") == 1
        assert "[SANDBOX_BYPASSED]" not in out

    def test_a_rung_spawn_without_a_grant_spawns_nothing(self, repo, wrapper, runner_env):
        runner_env.use("codex:gpt-5.5")
        before = _slot_dirs()
        capture = executor._spawn(
            provider=_codex(), model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_READONLY,
            transport="stdin", timeout=30, cwd=str(repo), sandbox_fallback="plugin-landlock",
        )
        assert capture.launch_error and "grant" in capture.launch_error
        assert wrapper.records() == [] and _slot_dirs() <= before

    def test_the_group_kill_reaches_the_exec_d_child(self, repo, wrapper, runner_env, tmp_path):
        runner_env.use("codex:gpt-5.5")
        pidfile = tmp_path / "grandchild.pid"
        wrapper.set(dispatch=f"tree:{pidfile}")
        capture = _spawn_on_rung(repo, "validation", timeout=1.5)
        assert capture.timed_out
        [row] = wrapper.records(probe=False)
        # The wrapper exec'd in place: its pid led its own group, the one killed.
        assert row["pid"] == row["pgid"]
        grandchild = int(pidfile.read_text().strip())
        deadline = time.monotonic() + 5
        while _pid_alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(grandchild)
        assert not os.path.exists(row["env"]["TMPDIR"])


class TestTheBackstop:
    @staticmethod
    def _stub(monkeypatch, argv):
        provider = _codex()
        monkeypatch.setattr(
            type(provider), "build_agent_command_for_transport",
            lambda self, *a, **k: (list(argv), None),
        )
        spawned = []

        class NoPopen:
            def __init__(self, *args, **kwargs):
                spawned.append(args)
                raise OSError("stubbed")

        monkeypatch.setattr(executor.subprocess, "Popen", NoPopen)
        return provider, spawned

    def test_an_unwrapped_write_argv_with_full_access_raises_before_popen(self, repo, monkeypatch):
        provider, spawned = self._stub(
            monkeypatch, ["codex", "exec", "--sandbox", "danger-full-access", "p"],
        )
        with pytest.raises(executor.SandboxBackstopViolation):
            executor._spawn_in_context(
                provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_WRITE,
                transport="argv", timeout=5, cwd=str(repo),
            )
        assert spawned == []

    @pytest.mark.parametrize("scope", ["plan-file", "tasks"])
    def test_f5_consent_passes_the_backstop(self, repo, monkeypatch, scope):
        provider, spawned = self._stub(
            monkeypatch, ["codex", "exec", "--sandbox", "danger-full-access", "p"],
        )
        capture = executor._spawn_in_context(
            provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_WRITE,
            transport="argv", timeout=5, cwd=str(repo),
            write_consent=WritePostureConsent("danger-full-access", write_scope=scope),
        )
        assert capture.launch_error == "stubbed" and len(spawned) == 1

    @pytest.mark.parametrize("scope", ["repo-wide", None])
    def test_f5_consent_for_a_scope_the_rung_does_not_serve_is_refused(
        self, repo, monkeypatch, scope,
    ):
        provider, spawned = self._stub(
            monkeypatch, ["codex", "exec", "--sandbox", "danger-full-access", "p"],
        )
        with pytest.raises(executor.SandboxBackstopViolation) as excinfo:
            executor._spawn_in_context(
                provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_WRITE,
                transport="argv", timeout=5, cwd=str(repo),
                write_consent=WritePostureConsent("danger-full-access", write_scope=scope),
            )
        assert repr(scope) in str(excinfo.value) and "F5" in str(excinfo.value)
        assert spawned == []

    def test_a_wrapped_write_argv_without_an_in_scope_grant_is_refused(self, repo, monkeypatch):
        wrapper_py = landlock_dispatch.WRAPPER_PATHS["landlock_exec"]
        provider, spawned = self._stub(
            monkeypatch,
            [sys.executable, "-I", "-S", wrapper_py, "--",
             "codex", "exec", "--sandbox", "danger-full-access", "p"],
        )
        with pytest.raises(executor.SandboxBackstopViolation) as excinfo:
            executor._spawn_in_context(
                provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_WRITE,
                transport="argv", timeout=5, cwd=str(repo),
            )
        assert "Landlock jail" in str(excinfo.value)
        assert spawned == []

    def test_a_wrapped_repo_wide_grant_passes_the_backstop(self, repo, monkeypatch, wrapper):
        """Step 3b: the jail serves repo-wide, so a wrapped argv under a
        repo-wide grant is not what the backstop refuses (the stub spawns)."""
        wrapper_py = landlock_dispatch.WRAPPER_PATHS["landlock_exec"]
        provider, spawned = self._stub(
            monkeypatch,
            [sys.executable, "-I", "-S", wrapper_py, "--",
             "codex", "exec", "--sandbox", "danger-full-access", "p"],
        )
        request = _plan_request(repo, scope="repo-wide", kind="implement")
        jail = executor._JailSpawn(rung=PLUGIN_LANDLOCK_RUNG, request=request, tmpdir=str(repo / "slot"))
        (repo / "slot").mkdir()
        executor._spawn_in_context(
            provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_WRITE,
            transport="argv", timeout=5, cwd=str(repo), jail=jail,
        )
        assert len(spawned) == 1

    def test_the_render_refuses_f5_for_a_scope_the_rung_does_not_serve(self):
        provider = CodexProvider()
        base = provider.build_agent_command("p", "gpt-5.5", PROFILE_AGENT_WRITE)
        for scope in ("repo-wide", None):
            with pytest.raises(ValueError, match="write scopes"):
                provider.apply_sandbox_fallback(
                    list(base), PROFILE_AGENT_WRITE, "unsandboxed",
                    consent=WritePostureConsent("danger-full-access", write_scope=scope),
                )
        rendered = provider.apply_sandbox_fallback(
            list(base), PROFILE_AGENT_WRITE, "unsandboxed",
            consent=WritePostureConsent("danger-full-access", write_scope="plan-file"),
        )
        assert executor._carries_full_access(rendered)

    def test_a_prompt_that_names_the_value_is_not_the_value(self, repo, monkeypatch):
        provider, spawned = self._stub(
            monkeypatch,
            ["codex", "exec", "--sandbox", "workspace-write", "explain danger-full-access"],
        )
        executor._spawn_in_context(
            provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_WRITE,
            transport="argv", timeout=5, cwd=str(repo),
        )
        assert len(spawned) == 1

    def test_every_write_render_is_wrapped_or_refused(self, repo, jail_home):
        """No agent-write render path yields danger-full-access unwrapped."""
        provider = CodexProvider()
        for rung in ("landlock", "unsandboxed", "plugin-landlock", None, "bogus"):
            try:
                argv, _ = provider.build_agent_command_for_transport(
                    "p", "gpt-5.5", PROFILE_AGENT_WRITE, "argv", sandbox_fallback=rung,
                )
            except ValueError:
                continue
            assert not executor._carries_full_access(argv), rung


def _decide(runner, profile, rung, scope=None):
    executor._SANDBOX_DECISIONS[executor._sandbox_decision_key(runner, profile, scope)] = rung


class TestTheDispatchOnTheRung:
    def test_a_validation_dispatch_runs_through_the_wrapper(self, repo, wrapper, runner_env):
        runner_env.use("codex:gpt-5.5")
        runner_env.echo("validation")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        seen = {}
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=0, after_result=lambda ctx: seen.update(ctx),
        )
        assert outcome.ok, outcome.reason
        assert len(wrapper.records(probe=False)) == 1
        assert seen["sandbox_rung"] == "plugin-landlock"
        assert seen["jail_denials"] is None and seen["jail_evidence"] is None

    def test_each_retry_gets_a_fresh_tmpdir_that_is_gone_after(self, repo, wrapper, runner_env):
        runner_env.use("codex:gpt-5.5")
        wrapper.set(dispatch="fail")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=1, sleep=lambda _s: None,
        )
        assert not outcome.ok
        dirs = [row["env"]["TMPDIR"] for row in wrapper.records(probe=False)]
        assert len(dirs) == 2 and dirs[0] != dirs[1]
        assert not any(os.path.exists(d) for d in dirs)

    def test_a_jail_eacces_is_one_line_and_never_retried(self, repo, wrapper, runner_env, jail_home, capsys):
        runner_env.use("codex:gpt-5.5")
        denied = Path(os.environ["HOME"]) / "elsewhere" / "state.db"
        denied.parent.mkdir(parents=True)
        wrapper.set(dispatch=f"eacces:{denied}")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=2, sleep=lambda _s: None,
        )
        assert outcome.status == executor.STATUS_REJECTED
        assert len(outcome.attempts) == 1
        assert outcome.reason.startswith("[JAIL_DENIED] codex:gpt-5.5 on rung 'plugin-landlock'")
        assert "denied ~/elsewhere/state.db" in outcome.reason
        assert str(Path(os.environ["HOME"])) + os.sep not in outcome.reason
        assert "allow-list: dir:~/.codex" in outcome.reason
        assert PLUGIN_LANDLOCK_RUNG.measured_through[0] in outcome.reason
        assert "github.com/beastlabai/multi-llm-plugin/issues" in outcome.reason
        assert f"{SKIP_RUNG_ENV}=plugin-landlock" in outcome.reason
        out = capsys.readouterr().out
        assert out.count("[JAIL_DENIED]") == 1
        assert any("[JAIL_DENIED]" in line for line in executor.jail_summary_lines())

    def test_a_denial_that_names_no_path_is_a_probable_jail_denial(self, repo, wrapper, runner_env):
        runner_env.use("codex:gpt-5.5")
        wrapper.set(dispatch="eacces_nopath")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=1, sleep=lambda _s: None,
        )
        assert outcome.status == executor.STATUS_REJECTED
        assert "path not named" in outcome.reason and len(outcome.attempts) == 1

    def test_an_ordinary_permission_error_is_not_a_jail_denial(self, repo, wrapper, runner_env, capsys):
        runner_env.use("codex:gpt-5.5")
        # Inside the attempt's own TMPDIR: an allow entry, so not the jail's.
        wrapper.set(dispatch="eacces:$TMPDIR/locked")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=0,
        )
        assert not outcome.ok
        assert outcome.status != executor.STATUS_REJECTED
        assert "not a jail denial" in outcome.reason
        assert "[JAIL_DENIED]" not in capsys.readouterr().out

    def test_a_command_output_denial_fails_nothing(self, repo, wrapper, runner_env, capsys):
        runner_env.use("codex:gpt-5.5")
        runner_env.echo("validation")
        wrapper.set(dispatch="cmdeny:/etc/stray")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=0,
        )
        assert outcome.ok, outcome.reason
        out = capsys.readouterr().out
        assert "1 command write(s) outside the grant denied (/etc/stray)" in out
        assert "[JAIL_DENIED]" not in out

    def test_the_home_check_warns_and_keeps_the_verdict(self, repo, wrapper, runner_env, jail_home, capsys):
        runner_env.use("codex:gpt-5.5")
        runner_env.echo("validation")
        (jail_home / "config.toml").write_text('model = "x"\n', encoding="utf-8")
        wrapper.set(dispatch='home:\\n[mcp_servers.x]\\ncommand = "evil-value"\\n')
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=0,
        )
        assert outcome.ok, outcome.reason
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "[CODEX_HOME_CHECK]" in ln]
        assert len(lines) == 1
        assert "config.toml" in lines[0] and "mcp_servers" in lines[0]
        assert "evil-value" not in lines[0]
        assert "Warn-only" in lines[0]
        assert "mcp_servers.x" in (jail_home / "config.toml").read_text(encoding="utf-8")

    def _seed_linked_venv(self, tmp_path, jail_home):
        """A plugin's uv ``.venv`` file under the home, hardlinked from a uv
        cache outside it (the shape that tripped the check, 2026-09-30)."""
        cache = tmp_path / "uv-cache" / "yaml"
        cache.mkdir(parents=True)
        cached = cache / "__init__.py"
        cached.write_text("__version__ = '6.0.3'\n", encoding="utf-8")
        venv = jail_home / "plugins" / "cache" / "m" / "multi-llm" / "0.1.0" / ".venv" / "yaml"
        venv.mkdir(parents=True)
        os.link(cached, venv / "__init__.py")
        return cached, venv / "__init__.py"

    @pytest.mark.parametrize("dispatch, warned", [("link", False), ("linkwrite", True)])
    def test_the_home_check_reads_content_not_ctime(
        self, repo, wrapper, runner_env, jail_home, tmp_path, capsys, dispatch, warned
    ):
        """The executor lane shares the snapshot: a hardlink made and removed
        on a watched venv inode (ctime only) warns nothing; a rewrite of the
        same inode through the cache's name, size and mtime unchanged, warns."""
        runner_env.use("codex:gpt-5.5")
        runner_env.echo("validation")
        cached, watched = self._seed_linked_venv(tmp_path, jail_home)
        ctime_before = os.lstat(watched).st_ctime_ns
        wrapper.set(dispatch=f"{dispatch}:{cached}")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=0,
        )
        assert outcome.ok, outcome.reason
        assert os.lstat(watched).st_ctime_ns != ctime_before
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "[CODEX_HOME_CHECK]" in ln]
        if warned:
            assert len(lines) == 1 and "__init__.py" in lines[0], lines
        else:
            assert lines == []

    def test_the_closing_summary_repeats_the_lines(self, repo, wrapper, runner_env, jail_home, capsys):
        runner_env.use("codex:gpt-5.5")
        wrapper.set(dispatch="eacces_nopath")
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        monkey_task = _validation_batch(phase)
        report = executor.run_tasks([monkey_task], phase_dir=str(phase), kind="validation", cwd=str(repo))
        assert not report.ok
        out = capsys.readouterr().out
        assert "[exec-subagents] closing summary: [JAIL_DENIED]" in out


def _write_task(phase: Path, kind="apply_suggestions"):
    return make_write_task(
        kind=kind, task_id=f"{kind}-0", prompt="do it\n",
        inputs=build_inputs(grouped_file=str(phase / "grouped.json")),
        batch_index=0, prior_changes_id="sec-1", phase_dir=str(phase),
        declared_files=["plans/plan.md"],
    )


class TestTheGrant:
    def test_the_kind_to_scope_map(self):
        assert executor.WRITE_SCOPE_BY_KIND == {
            "apply_suggestions": "plan-file",
            "apply_task_suggestions": "tasks",
            "apply_code_fixes": "repo-wide",
            "implement": "repo-wide",
        }
        assert set(executor.WRITE_SCOPE_BY_KIND) == set(WRITE_KINDS)

    @pytest.fixture
    def no_spawn(self, monkeypatch):
        def boom(**kwargs):
            raise AssertionError("a dispatch with no complete grant was spawned")

        monkeypatch.setattr(executor, "_spawn", boom)

    @pytest.mark.parametrize("case", ["none", "kind", "scope", "target"])
    def test_a_write_dispatch_without_its_grant_is_refused(self, repo, runner_env, no_spawn, case):
        runner_env.use("codex:gpt-5.5")
        _decide("codex:gpt-5.5", PROFILE_AGENT_WRITE, "plugin-landlock", "plan-file")
        phase = repo / "phase"
        phase.mkdir()
        plan = str(repo / "plans" / "plan.md")
        request = {
            "none": None,
            "kind": JailRequest(str(repo), "apply_task_suggestions", "plan-file", plan),
            "scope": JailRequest(str(repo), "apply_suggestions", "tasks", plan),
            "target": JailRequest(str(repo), "apply_suggestions", "plan-file", None),
        }[case]
        outcome = executor.dispatch_task(
            _write_task(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), prompt="do it\n", jail_request=request,
        )
        assert outcome.status == executor.STATUS_REJECTED and outcome.rung == 7
        assert "grant" in outcome.reason and "Nothing was dispatched." in outcome.reason

    def test_an_unmapped_write_kind_is_refused_never_widened(self, repo, runner_env, no_spawn, monkeypatch):
        runner_env.use("codex:gpt-5.5")
        monkeypatch.setattr(executor, "WRITE_SCOPE_BY_KIND", {})
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _write_task(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), prompt="do it\n",
        )
        assert outcome.status == executor.STATUS_REJECTED
        assert "no write scope" in outcome.reason

    def test_a_read_kind_never_carries_a_grant(self, repo, runner_env, no_spawn):
        runner_env.use("codex:gpt-5.5")
        phase = repo / "phase"
        phase.mkdir()
        outcome = executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), jail_request=_plan_request(repo),
        )
        assert outcome.status == executor.STATUS_REJECTED
        assert "a read kind never carries one" in outcome.reason

    def test_the_hook_context_carries_denials_for_a_write_on_the_rung_only(self):
        denials = landlock_dispatch.JailDenials(out_of_scope_count=1)
        capture = executor._Capture(sandbox_rung="plugin-landlock", jail_denials=denials)
        request = JailRequest("/r", "apply_suggestions", "plan-file", "/r/plans/p.md")
        ctx = executor._jail_hook_context(capture, request, "plugin-landlock")
        assert ctx == {"sandbox_rung": "plugin-landlock", "jail_denials": denials, "jail_evidence": None}
        read = executor._jail_hook_context(capture, JailRequest("/r", "validation"), "plugin-landlock")
        assert read["jail_denials"] is None
        off = executor._jail_hook_context(executor._Capture(), request, None)
        assert off == {"sandbox_rung": None, "jail_denials": None, "jail_evidence": None}


class TestTheJailDeniedNextStep:
    """The `[JAIL_DENIED]` skip step is worded per lane: skipping the only
    write rung refuses a write unless F5's consent is present."""

    @staticmethod
    def _line(**lane):
        return executor.jail_denied_line(
            spec="codex:gpt-5.5", rung=PLUGIN_LANDLOCK_RUNG, path=None,
            excerpt="Permission denied", allow_list="", **lane,
        )

    def test_a_read_dispatch_keeps_the_one_run_skip(self):
        line = self._line(profile=PROFILE_AGENT_READONLY)
        assert f"or else set {SKIP_RUNG_ENV}=plugin-landlock for one run." in line
        assert line == self._line()

    @pytest.mark.parametrize("scope", ["plan-file", "tasks"])
    def test_a_write_dispatch_says_the_skip_is_a_refusal_without_f5(self, scope):
        line = self._line(profile=PROFILE_AGENT_WRITE, write_scope=scope)
        assert "for one run" not in line
        assert f"Setting {SKIP_RUNG_ENV}=plugin-landlock does not run it" in line
        assert f"(scope {scope})" in line
        assert "refused unless the chat itself runs in danger-full-access" in line
        assert HOST_SANDBOX_POSTURE_ENV in line and "F5" in line
        assert "not Codex (F6)" in line


class TestNoSpawnRejectionsCloseTheAudit:
    """A rejection after `before_attempt` (which writes the audit's
    `dispatched` line) records an attempt, so the scheduler's `_audit_terminal`
    closes that line instead of reading the empty list as "never dispatched"."""

    @pytest.mark.parametrize("exc", [
        executor.SandboxBackstopViolation("backstop: nothing was spawned"),
        ValueError("the rung needs its grant"),
    ])
    def test_the_rejection_is_recorded_as_an_attempt(self, repo, runner_env, monkeypatch, exc):
        runner_env.use("codex:gpt-5.5")
        _decide("codex:gpt-5.5", PROFILE_AGENT_WRITE, "plugin-landlock", "plan-file")
        phase = repo / "phase"
        phase.mkdir()
        plan = str(repo / "plans" / "plan.md")

        def refuse(**kwargs):
            raise exc

        monkeypatch.setattr(executor, "_spawn", refuse)
        dispatched, observed = [], []
        outcome = executor.dispatch_task(
            _write_task(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), prompt="do it\n",
            jail_request=JailRequest(str(repo), "apply_suggestions", "plan-file", plan),
            before_attempt=lambda ctx: dispatched.append(ctx["attempt"]),
            after_attempt=observed.append,
        )
        assert dispatched == [1]
        assert outcome.status == executor.STATUS_REJECTED and outcome.rung == 7
        [attempt] = outcome.attempts
        assert attempt.status == executor.STATUS_REJECTED
        assert attempt.exit_code is None and attempt.index == 1
        assert attempt.error == outcome.reason
        assert observed == [attempt]


class TestPreviewsAndBudgets:
    def test_the_dry_run_preview_shows_the_wrapped_shape_and_creates_nothing(self, repo, wrapper):
        before = _slot_dirs()
        argv, refusal = executor._preview_argv(
            _codex(), prompt="p", model="gpt-5.5", profile=PROFILE_AGENT_READONLY,
            schema=None, sandbox_fallback="plugin-landlock",
            jail_request=JailRequest(str(repo), "validation"),
        )
        assert refusal is None
        assert argv[:4] == [sys.executable, "-I", "-S", str(wrapper.path)]
        assert f"dir:{landlock_dispatch.PLACEHOLDER_TMPDIR}" in argv
        assert f"dir:{landlock_dispatch.PLACEHOLDER_CODEX_HOME}" in argv
        tail = argv[argv.index("--") + 1:]
        assert tail[0] == "codex" and tail[tail.index("--sandbox") + 1] == "danger-full-access"
        assert _slot_dirs() <= before and wrapper.records() == []

    def test_the_budget_measures_the_wrapped_argv(self, repo, wrapper, monkeypatch):
        from utils import subagent_tasks

        seen = {}

        def budget(*, transport, non_prompt_argv, is_batch_shim):
            seen["argv"] = list(non_prompt_argv)
            return None

        monkeypatch.setattr(subagent_tasks, "prompt_budget_utf16", budget)
        executor._prompt_over_budget(
            _codex(), "gpt-5.5", "p", "argv", PROFILE_AGENT_READONLY,
            sandbox_fallback="plugin-landlock",
            jail_request=JailRequest(str(repo), "validation"),
        )
        assert seen["argv"][:4] == [sys.executable, "-I", "-S", str(wrapper.path)]
        assert _slot_dirs() is not None


class TestNoPathInSharedState:
    def test_no_decision_or_cache_value_holds_a_path(
        self, repo, wrapper, gate, runner_env, tmp_path, monkeypatch,
    ):
        runner_env.use("codex:gpt-5.5")
        runner_env.echo("validation")
        request = _plan_request(repo)
        gate(_JailLadder(), PROFILE_AGENT_WRITE, request)
        gate(_JailLadder())
        monkeypatch.setattr(executor, "get_provider", registry.get_provider)
        _decide("codex:gpt-5.5", PROFILE_AGENT_READONLY, "plugin-landlock")
        phase = repo / "phase"
        phase.mkdir()
        executor.dispatch_task(
            _validation_batch(phase), phase_dir=str(phase), runner="codex:gpt-5.5",
            cwd=str(repo), retries=0,
        )
        values = repr(list(executor._SANDBOX_DECISIONS.values())) + repr(
            list(executor._SANDBOX_PROBE_CACHE.values())
        )
        assert str(tmp_path) not in values
        assert landlock_dispatch.SLOT_PREFIX not in values
        assert "plan.md" not in values
        for key in executor._SANDBOX_PROBE_CACHE:
            assert not any(landlock_dispatch.SLOT_PREFIX in str(part) for part in key)
        for key in executor._SANDBOX_DECISIONS:
            assert str(tmp_path) not in key


def test_a_changed_seed_is_named_as_the_truncation_control(repo, wrapper, gate):
    """The parent's byte comparison caught a truncation the canary reported as
    denied: the refusal names the truncation control, never the canary's own
    DENIED marker, which reads as the opposite (acceptance review, 2026-09-30)."""
    wrapper.set(probe="seed")
    with pytest.raises(executor.RunnerRejected) as excinfo:
        gate(_JailLadder(), PROFILE_AGENT_WRITE, _plan_request(repo))
    message = excinfo.value.message
    assert "the truncation control changed the seeded file" in message
    assert "the write was denied" not in message


class TestTheProbeWorkdirIsSerialized:
    """Adversarial review, 2026-10-02 (`repro_edge.py`, twelve `status` polls
    at once): each rung probe removed only the directories it created, so two
    probes of one repository left an empty `.multi-llm/` and
    `.multi-llm/tmp/` behind — the first one's `rmdir` failed on the second
    one's seed, and the second had created nothing. The canary directory is
    now held under an interprocess mutex for the probe's duration."""

    @pytest.fixture(autouse=True)
    def _temp_base(self, tmp_path, monkeypatch):
        base = tmp_path / "tmpbase"
        base.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(base))

    def test_a_second_probe_waits_its_turn_and_the_last_one_out_removes_the_dirs(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        events: list = []
        a_inside = threading.Event()
        a_release = threading.Event()

        def first():
            with executor._probe_workdir(str(repo), ".multi-llm/tmp", time.monotonic() + 10) as created:
                events.append(("a-in", list(created)))
                seed = repo / ".multi-llm" / "tmp" / "seed"
                seed.write_text("x")
                a_inside.set()
                a_release.wait(10)
                seed.unlink()
                events.append("a-out")

        def second():
            a_inside.wait(10)
            with executor._probe_workdir(str(repo), ".multi-llm/tmp", time.monotonic() + 10) as created:
                events.append(("b-in", list(created)))
            events.append("b-out")

        ta, tb = threading.Thread(target=first), threading.Thread(target=second)
        ta.start(); tb.start()
        assert a_inside.wait(10)
        time.sleep(0.3)
        assert not any(e[0] == "b-in" for e in events if isinstance(e, tuple)), events
        a_release.set()
        ta.join(10); tb.join(10)
        assert [e if isinstance(e, str) else e[0] for e in events] == ["a-in", "a-out", "b-in", "b-out"]
        # Each took the lock in turn, created both directories afresh, and
        # removed them: nothing is left in the repository.
        for _label, created in (e for e in events if isinstance(e, tuple)):
            assert [os.path.basename(p) for p in created] == ["tmp", ".multi-llm"]
        assert not (repo / ".multi-llm").exists()
        assert os.path.exists(executor._probe_workdir_lock_path(str(repo)))

    def test_a_held_mutex_past_the_deadline_reads_as_unprobed(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        inside = threading.Event()
        release = threading.Event()

        def holder():
            with executor._probe_workdir(str(repo), ".multi-llm/tmp", time.monotonic() + 10):
                inside.set()
                release.wait(10)

        thread = threading.Thread(target=holder)
        thread.start()
        assert inside.wait(10)
        try:
            started = time.monotonic()
            with executor._probe_workdir(str(repo), ".multi-llm/tmp", time.monotonic() + 0.3) as created:
                assert created is None
            assert time.monotonic() - started < 5
        finally:
            release.set()
            thread.join(10)
        assert not (repo / ".multi-llm").exists()
