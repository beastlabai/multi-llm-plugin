"""The fan-out lanes on the plugin's own Landlock jail (plan
codex-sandbox-unavailable-findings-and-fixes, step 2; tests 2026-09-30).

Tier 1, zero spend. The sandbox probe verdict is pinned (the walk is the
executor package's), and the dispatch is either a stubbed ``subprocess.run``
(the argv, env and read-back are what is under test) or a real process pair
under ``tmp_path``: a fake wrapper (``landlock_dispatch.WRAPPER_PATHS``) that
records what it was handed and ``exec``s a fake ``codex`` script. The fake
wrapper does not jail, so a ``chmod`` or a ``$CODEX_HOME`` write lands exactly
as it does inside the real jail (Landlock has no right for either).

Covered here:

* §8.3 item 1 — the §2a replay: a sibling run's log and a user edit inside a
  jailed slot's window cost nothing; the control on ``unsandboxed`` still
  fails closed on a heredoc stream.
* §8.3 item 5 — the mode check (``chmod +x`` → ``[MODE_CHECK]``, fail closed,
  bit left set; clean stream → kept with a warning; ``touch`` → nothing;
  ``core.fileMode false`` → no check).
* §8.3 item 6 — the home check (an ``[mcp_servers.x]`` table →
  ``[CODEX_HOME_CHECK]``, fail closed, table left; the run's own trust entry →
  nothing; a foreign trust entry → closing summary only).
* §8.2 "The porcelain check is skipped for the new rung only" — the predicate
  pins that ``tests/test_subagent_executor.py``'s
  ``test_the_porcelain_gate_reads_the_rung_id`` would gain (that file is
  another package's), the sibling counts, and ``mode_check_applies``.
* §8.2 "Carrying the grant" — one TMPDIR per attempt, the timeout and the
  cancelled-coroutine lifecycles, ``$CODEX_HOME`` as the child sees it, no
  path left on the adapter.
* §8.2 "A jail denial is told apart and explained" — ``[JAIL_DENIED]`` in the
  log and the closing summary, ``jail-denied`` on the record, no retry; the
  same text inside a command's output fails nothing.
* §7 step-2 subtask item 5 — a mixed fan-out's ``[SANDBOX_BYPASSED]`` count.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import review_plan_orchestrator as rpo  # noqa: E402
from utils import landlock_dispatch  # noqa: E402
from utils import llm_client  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import status_file  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.provider_registry import get_provider  # noqa: E402
from utils.providers.base import (  # noqa: E402
    CODEX_HOME_CHECK_MARKER,
    JAIL_DENIED_MARKER,
    MODE_CHECK_MARKER,
    SANDBOX_BYPASSED_MARKER,
    SANDBOX_FALLBACK_MARKER,
    SANDBOX_INCIDENTAL_MARKER,
)
from utils.providers.codex import PLUGIN_LANDLOCK_RUNG, CodexProvider  # noqa: E402
from utils.subagent_executor import SandboxProbeResult  # noqa: E402

# The fan-out's jailed slots are measured through POSIX things: a shebang fake
# codex on PATH, 0700 slot directories, tracked file modes, hardlinks on a
# venv inode. Windows runs no plugin jail (codex has its own sandbox there).
pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="shebang fake codex, POSIX file modes and hardlinks; Windows runs no plugin jail",
)

SPEC = "codex:gpt-5.5"
RUNG = "plugin-landlock"
#: The codex version the faked walk was judged against: the rung's own
#: measured-through version, read from the constant.
WALK_VERSION = PLUGIN_LANDLOCK_RUNG.measured_through[0]
BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
ABSENT_TAIL = "landlock_exec: unavailable: this kernel has no Landlock (ABI 0; the floor is 3)"
FINDINGS = [{
    "title": "Add input validation",
    "desc": "The plan does not validate the new email field.",
    "importance": "HIGH",
    "reference": "Overview",
    "type": "addition",
}]

#: A plugin's uv ``.venv`` package dir under ``$CODEX_HOME`` (a watched tree).
VENV_REL = os.path.join("plugins", "cache", "market", "multi-llm", "0.1.0",
                        ".venv", "lib", "python3.11", "site-packages", "yaml")

#: A command the stream audit rejects (a heredoc), and one it accepts.
HEREDOC_COMMAND = "/bin/bash -lc \"python3 - <<'PY'\nprint(1)\nPY\""
CLEAN_COMMAND = "/bin/bash -lc 'rg -n validation plans'"


# ---------------------------------------------------------------------------
# Fixtures: the verdict, the scratch repo, the fake wrapper and fake codex
# ---------------------------------------------------------------------------


def _pin_verdict(monkeypatch, rung):
    """The fan-out's probe answers ``unavailable``; the walk (the executor
    package's) is faked: the legacy rung skipped by version, then ``rung``."""
    if rung == RUNG:
        verdict = SandboxProbeResult(
            "unavailable", BWRAP_LINE, RUNG,
            (("landlock", "skipped"), (RUNG, "taken")), (), WALK_VERSION,
        )
    else:
        verdict = SandboxProbeResult(
            "unavailable", BWRAP_LINE, "unsandboxed",
            (("landlock", "skipped"), (RUNG, "absent"), ("unsandboxed", "taken")),
            ((RUNG, ABSENT_TAIL),), WALK_VERSION,
        )
    monkeypatch.setattr(base, "_fanout_sandbox_verdict", lambda _p, _cwd: verdict)
    return verdict


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=str(repo), check=True, capture_output=True, text=True,
    )


def _stream(command=None, *, output="", findings=True):
    lines = [json.dumps({"type": "thread.started", "thread_id": "t1"})]
    if command is not None:
        lines.append(json.dumps({"type": "item.completed", "item": {
            "id": "item_0", "type": "command_execution", "command": command,
            "aggregated_output": output, "exit_code": 0, "status": "completed",
        }}))
    if findings:
        lines.append(json.dumps({"type": "item.completed", "item": {
            "id": "item_1", "type": "agent_message", "text": json.dumps(FINDINGS),
        }}))
    lines.append(json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": 10, "output_tokens": 5,
    }}))
    return "\n".join(lines) + "\n"


FAKE_WRAPPER = r'''
import json, os, sys
args = sys.argv[1:]
sep = args.index("--")
allows, rest = args[:sep], args[sep + 1:]
log = os.environ.get("FAKE_WRAPPER_LOG")
if log:
    tmp = os.environ.get("TMPDIR", "")
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "allows": allows, "tmpdir": tmp, "tmpdir_exists": os.path.isdir(tmp),
            "argv": rest,
        }) + "\n")
os.execv(rest[0], rest)
'''

FAKE_CODEX = r'''
import json, os, pathlib, sys, time
mode = os.environ.get("FAKE_CODEX_MODE", "")
count = os.environ.get("FAKE_CODEX_COUNT")
if count:
    with open(count, "a", encoding="utf-8") as handle:
        handle.write("1\n")
hs = os.environ.get("FAKE_CODEX_HANDSHAKE")
if hs:
    # The window: tell the second process the slot is running, then wait for
    # it to say its writes are done. The poll only paces the wait; the order
    # is the handshake's.
    pathlib.Path(hs, "ready").write_text("1")
    deadline = time.time() + 60
    while not os.path.exists(os.path.join(hs, "done")) and time.time() < deadline:
        time.sleep(0.02)
cwd = os.getcwd()
home = os.environ.get("CODEX_HOME", "")
if mode == "chmod":
    os.chmod(os.path.join(cwd, "tests", "run.sh"), 0o755)
elif mode == "touch":
    os.utime(os.path.join(cwd, "tests", "run.sh"), (1, 1))
elif mode == "home":
    with open(os.path.join(home, "config.toml"), "a", encoding="utf-8") as handle:
        handle.write('\n[mcp_servers.x]\ncommand = "secret-launcher-value"\n')
elif mode == "trust":
    with open(os.path.join(home, "config.toml"), "a", encoding="utf-8") as handle:
        handle.write('\n[projects."%s"]\ntrust_level = "trusted"\n' % cwd)
elif mode == "foreign":
    with open(os.path.join(home, "config.toml"), "a", encoding="utf-8") as handle:
        handle.write('\n[projects."/elsewhere/other-repo"]\ntrust_level = "trusted"\n')
elif mode == "subdir":
    with open(os.path.join(home, "config.toml"), "a", encoding="utf-8") as handle:
        handle.write('\n[projects."%s"]\ntrust_level = "trusted"\n'
                     % os.path.join(cwd, "packages", "web"))
elif mode == "link":
    # What any `uv run` elsewhere does to the uv cache: link and unlink a
    # name of an inode the home's plugin venv shares (moves only its ctime).
    # The sleep only clears a coarse timestamp tick after the before-snapshot.
    src = os.environ["FAKE_LINK_SRC"]
    time.sleep(0.02)
    os.link(src, src + ".spare")
    os.unlink(src + ".spare")
elif mode == "link-write":
    # A real change to the same inode, through the cache's name, at the same
    # size and with the mtime put back.
    src = os.environ["FAKE_LINK_SRC"]
    info = os.stat(src)
    with open(src, "r+b") as handle:
        data = handle.read()
        handle.seek(0)
        handle.write(data.replace(b"6.0.3", b"6.6.6"))
    os.utime(src, ns=(info.st_atime_ns, info.st_mtime_ns))
elif mode == "venv":
    # What uv does to a plugin cache on its first `uv run` there: a new
    # `.venv` of ~40 files (racing re-run 3, opencode 01).
    venv = os.path.join(home, "plugins", "cache", "market", "multi-llm", "0.1.0",
                        "skills", "multi-llm", ".venv")
    for n in range(40):
        target = os.path.join(venv, "lib", "f%02d.py" % n)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("x = %d\n" % n)
elif mode == "eacces":
    sys.stderr.write(
        "Error: failed to write %s: Permission denied (os error 13)\n"
        % os.environ["FAKE_DENIED_PATH"]
    )
    sys.exit(1)
sys.stdout.write(os.environ.get("FAKE_CODEX_STREAM", ""))
sys.exit(0)
'''


def _write_script(path, body):
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A scratch repo with a tracked source file and a tracked script, a temp
    ``CODEX_HOME``, a temp base for the attempts' scratch dirs (outside the
    repo), and HOME pointed at ``tmp_path`` so no real home is ever read."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "plans").mkdir()
    (repo / "src" / "app.py").write_text("print('app')\n", encoding="utf-8")
    run_sh = repo / "tests" / "run.sh"
    run_sh.write_text("echo run\n", encoding="utf-8")
    run_sh.chmod(0o644)
    (repo / "plans" / "feature.md").write_text(
        "# Plan\n\nAdd an email field with validation.\n", encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "add", "src/app.py", "tests/run.sh")
    _git(repo, "commit", "-q", "-m", "init")

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    slot_base = tmp_path / "slot-tmp"
    slot_base.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(slot_base))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    # The retry ladder's backoff, if one fires, is an exact 10 ms.
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    return {
        "tmp": tmp_path, "repo": repo, "plan": repo / "plans" / "feature.md",
        "home": codex_home, "slot_base": slot_base,
    }


@pytest.fixture
def fake_codex(scratch, monkeypatch):
    """A real fake wrapper and a real fake ``codex`` on PATH."""
    tmp = scratch["tmp"]
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    _write_script(bin_dir / "codex", FAKE_CODEX)
    wrapper = tmp / "fake_landlock_exec.py"
    wrapper.write_text(FAKE_WRAPPER, encoding="utf-8")
    monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(wrapper))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    log = tmp / "wrapper.log"
    count = tmp / "codex.count"
    monkeypatch.setenv("FAKE_WRAPPER_LOG", str(log))
    monkeypatch.setenv("FAKE_CODEX_COUNT", str(count))
    monkeypatch.setenv("FAKE_CODEX_STREAM", _stream(HEREDOC_COMMAND))
    monkeypatch.setattr(base, "PROVIDER_STAGGER_DELAY", 0.0)

    def wrapper_calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line]

    def codex_calls():
        return len(count.read_text().splitlines()) if count.exists() else 0

    scratch["wrapper_calls"] = wrapper_calls
    scratch["codex_calls"] = codex_calls
    return scratch


def _run_review_plan(scratch, *, expect_exit=0):
    """``review_plan_orchestrator.main()`` for real over one codex slot.
    Returns ``(phase_dir, status)``."""
    argv = [
        "--plan-file", str(scratch["plan"]), "--models", SPEC, "--skip-validation",
        "--timeout", "3600", "--max-parallel", "1", "--rerun-all", "--force",
    ]
    if expect_exit == 0:
        assert asyncio.run(rpo.main(argv)) == 0
    else:
        with pytest.raises(SystemExit) as exited:
            asyncio.run(rpo.main(argv))
        assert exited.value.code == expect_exit
    phase_dir = base.get_phase_dir(scratch["plan"], "review-plan")
    return Path(phase_dir), status_file.read_status(str(phase_dir))


def _slot_lines(out, marker):
    """The lines a slot printed under ``marker`` (its display-name prefix)."""
    return [line for line in out.splitlines() if line.startswith(f"[{SPEC}] {marker}")]


def _result_file(phase_dir):
    return phase_dir / "codex_gpt-5_5.json"


# ---------------------------------------------------------------------------
# §8.3 item 1: the §2a replay, tier 1
# ---------------------------------------------------------------------------


def _second_process(scratch):
    """The other writer in the window: a sibling run's log and a user edit to
    a tracked file, made only once the slot says it is running."""
    hs = scratch["tmp"] / "handshake"
    hs.mkdir()
    sibling = scratch["repo"] / "plans" / "other" / "review-plan" / "log_other.txt"
    script = (
        "import os, pathlib, time\n"
        f"hs = {str(hs)!r}\n"
        "deadline = time.time() + 60\n"
        "while not os.path.exists(os.path.join(hs, 'ready')) and time.time() < deadline:\n"
        "    time.sleep(0.02)\n"
        f"p = pathlib.Path({str(sibling)!r}); p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text('sibling run log\\n')\n"
        f"with open({str(scratch['repo'] / 'src' / 'app.py')!r}, 'a') as f: f.write('# user edit\\n')\n"
        "pathlib.Path(hs, 'done').write_text('1')\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script])
    return hs, proc, sibling


class TestTheSection2aReplay:
    def test_a_jailed_slot_survives_a_sibling_run_and_a_user_edit(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        hs, proc, sibling = _second_process(fake_codex)
        monkeypatch.setenv("FAKE_CODEX_HANDSHAKE", str(hs))
        try:
            phase_dir, status = _run_review_plan(fake_codex)
        finally:
            proc.wait(timeout=60)
        out = capsys.readouterr().out

        # Both writes really landed inside the slot's window.
        assert sibling.exists()
        assert "# user edit" in (fake_codex["repo"] / "src" / "app.py").read_text()
        # The slot ran through the wrapper, on the rung's argv.
        (call,) = fake_codex["wrapper_calls"]()
        assert call["argv"][call["argv"].index("--sandbox") + 1] == "danger-full-access"

        # Kept and counted.
        assert json.loads(_result_file(phase_dir).read_text()) == FINDINGS
        assert status["models_completed"] == [SPEC]
        assert status["models_failed"] == {}

        incidental = [line for line in out.splitlines() if SANDBOX_INCIDENTAL_MARKER in line]
        assert len(incidental) == 1
        assert f"'{RUNG}'" in incidental[0] and BWRAP_LINE in incidental[0]
        for marker in (SANDBOX_BYPASSED_MARKER, "[PORCELAIN_CHECK]", MODE_CHECK_MARKER,
                       SANDBOX_FALLBACK_MARKER, "fails closed"):
            assert marker not in out, marker

    def test_control_on_the_uncontained_rung_the_heredoc_stream_is_discarded(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, "unsandboxed")
        hs, proc, _sibling = _second_process(fake_codex)
        monkeypatch.setenv("FAKE_CODEX_HANDSHAKE", str(hs))
        try:
            phase_dir, status = _run_review_plan(fake_codex, expect_exit=3)
        finally:
            proc.wait(timeout=60)
        out = capsys.readouterr().out

        # No wrapper on the uncontained rung, and the porcelain check fired.
        assert fake_codex["wrapper_calls"]() == []
        assert not _result_file(phase_dir).exists()
        assert SPEC in status["models_failed"]
        assert "fails closed" in status["models_failed"][SPEC]
        assert SANDBOX_INCIDENTAL_MARKER not in out
        assert _slot_lines(out, SANDBOX_BYPASSED_MARKER)


# ---------------------------------------------------------------------------
# §8.3 item 5: the mode check
# ---------------------------------------------------------------------------


class TestTheModeCheck:
    def _mode(self, fake_codex):
        out = subprocess.run(
            ["git", "ls-files", "-s", "tests/run.sh"], cwd=str(fake_codex["repo"]),
            capture_output=True, text=True,
        )
        return os.stat(fake_codex["repo"] / "tests" / "run.sh").st_mode & 0o100, out

    def test_a_flipped_bit_fails_the_slot_closed_and_is_left_set(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "chmod")
        phase_dir, status = _run_review_plan(fake_codex, expect_exit=3)
        out = capsys.readouterr().out

        lines = _slot_lines(out, MODE_CHECK_MARKER)
        assert len(lines) == 1
        assert "tests/run.sh 100644 → 100755" in lines[0]
        assert "fails closed" in lines[0]
        assert SANDBOX_BYPASSED_MARKER not in out
        assert not _result_file(phase_dir).exists()
        assert MODE_CHECK_MARKER in status["models_failed"][SPEC]
        assert "porcelain" not in status["models_failed"][SPEC].lower()
        # Never flipped back: the user may be the one who changed it.
        assert os.stat(fake_codex["repo"] / "tests" / "run.sh").st_mode & 0o100

    def test_a_clean_stream_keeps_the_output_under_a_warning(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "chmod")
        monkeypatch.setenv("FAKE_CODEX_STREAM", _stream(CLEAN_COMMAND))
        phase_dir, status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out

        (line,) = _slot_lines(out, MODE_CHECK_MARKER)
        assert "not attributed" in line and "kept" in line
        assert json.loads(_result_file(phase_dir).read_text()) == FINDINGS
        assert status["progress"]["models"][SPEC]["warning"].startswith(MODE_CHECK_MARKER)

    def test_a_touch_is_outside_the_contract(self, fake_codex, monkeypatch, capsys):
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "touch")
        phase_dir, _status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out
        assert MODE_CHECK_MARKER not in out
        assert _result_file(phase_dir).exists()

    def test_core_filemode_false_runs_no_check(self, fake_codex, monkeypatch, capsys):
        _git(fake_codex["repo"], "config", "core.fileMode", "false")
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "chmod")
        assert base.mode_snapshot(str(fake_codex["repo"])) is None
        phase_dir, _status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out
        assert MODE_CHECK_MARKER not in out
        assert _result_file(phase_dir).exists()

    def test_the_snapshot_reads_only_tracked_regular_files(self, scratch):
        (scratch["repo"] / "untracked.sh").write_text("x", encoding="utf-8")
        snap = base.mode_snapshot(str(scratch["repo"]))
        assert snap == {"src/app.py": "100644", "tests/run.sh": "100644"}
        assert base.mode_snapshot(None) is None
        assert base.mode_snapshot(str(scratch["tmp"] / "slot-tmp")) is None


# ---------------------------------------------------------------------------
# §8.3 item 6: the home check
# ---------------------------------------------------------------------------


class TestTheHomeCheck:
    def test_an_mcp_server_table_fails_the_slot_closed_and_stays(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "home")
        phase_dir, status = _run_review_plan(fake_codex, expect_exit=3)
        out = capsys.readouterr().out

        (line,) = _slot_lines(out, CODEX_HOME_CHECK_MARKER)
        assert "config.toml (mcp_servers.x)" in line
        assert "secret-launcher-value" not in out
        assert "fails closed" in line
        # Repeated, as it is, in the run's closing summary.
        summary = out.split("=== Sandbox and jail checks")[-1]
        assert line[len(f"[{SPEC}] "):] in summary
        assert not _result_file(phase_dir).exists()
        assert CODEX_HOME_CHECK_MARKER in status["models_failed"][SPEC]
        # Never reverted.
        assert "[mcp_servers.x]" in (fake_codex["home"] / "config.toml").read_text()

    def test_the_runs_own_trust_entry_is_expected(self, fake_codex, monkeypatch, capsys):
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "trust")
        phase_dir, _status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out
        assert CODEX_HOME_CHECK_MARKER not in out
        assert _result_file(phase_dir).exists()

    @pytest.mark.parametrize("mode, named", [
        ("foreign", "/elsewhere/other-repo"),
        ("subdir", os.path.join("packages", "web")),
    ])
    def test_a_trust_entry_added_during_the_run_is_named_in_the_closing_summary_only(
        self, fake_codex, monkeypatch, capsys, mode, named
    ):
        """Fix round 17: a jailed slot can write ``$CODEX_HOME`` itself, so a
        table for another repository may be this run's own; round 16 named
        only one inside the repository (run 6, #11a), which hid it. Named in
        the closing summary only, blaming no one, and never a failure."""
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", mode)
        phase_dir, _status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out
        assert not _slot_lines(out, CODEX_HOME_CHECK_MARKER)
        lines = [line for line in out.splitlines() if CODEX_HOME_CHECK_MARKER in line]
        assert len(lines) == 1
        assert named in lines[0] and "not a failure" in lines[0]
        assert "by this run or by another codex session" in lines[0]
        assert _result_file(phase_dir).exists()

    def test_a_trust_entry_gone_again_by_the_closing_summary_is_not_named(
        self, fake_codex, monkeypatch, capsys
    ):
        """Run 6's noise: a concurrent sibling run's own clean-up removed the
        entry it added before this run's summary printed."""
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "foreign")
        config = fake_codex["home"] / "config.toml"
        real_print = base.print_closing_summary

        def sibling_cleans_up_first(key):
            text = config.read_text(encoding="utf-8")
            config.write_text(text.replace(
                '[projects."/elsewhere/other-repo"]\ntrust_level = "trusted"\n', ""),
                encoding="utf-8")
            return real_print(key)

        monkeypatch.setattr(base, "print_closing_summary", sibling_cleans_up_first)
        phase_dir, _status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out
        assert "/elsewhere/other-repo" not in out
        assert CODEX_HOME_CHECK_MARKER not in out
        assert _result_file(phase_dir).exists()

    def test_a_long_change_is_summarized_and_listed_in_full_elsewhere(
        self, fake_codex, monkeypatch, capsys
    ):
        """Racing re-run 3 (opencode 01): a new uv `.venv` in the plugin cache
        put ~40 paths on one line, and the slot's progress warning carried it
        into a 9 700-character tool result."""
        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setenv("FAKE_CODEX_MODE", "venv")
        _phase_dir, status = _run_review_plan(fake_codex, expect_exit=3)
        out = capsys.readouterr().out

        (line,) = _slot_lines(out, CODEX_HOME_CHECK_MARKER)
        venv = os.path.join("plugins", "cache", "market", "multi-llm", "0.1.0",
                            "skills", "multi-llm", ".venv", "lib")
        assert f"40 paths under {venv}{os.sep}, e.g. f00.py, f01.py, f02.py and 37 more" in line
        assert "fails closed" in line and len(line) < 900
        listed = line.split("full list: ")[1].split(")")[0]
        listed = os.path.expanduser(listed)
        assert listed.startswith(str(os.environ["XDG_STATE_HOME"]))
        names = open(listed, encoding="utf-8").read().split()
        assert names == [os.path.join(venv, "f%02d.py" % n) for n in range(40)]
        assert line[len(f"[{SPEC}] "):] in status["models_failed"][SPEC]

    def _seed_linked_venv(self, scratch):
        """A plugin's uv ``.venv`` under the home whose package file is a
        hardlink into a uv cache outside it (the 2026-09-30 shape)."""
        cache = scratch["tmp"] / "uv-cache" / "yaml"
        cache.mkdir(parents=True)
        cached = cache / "__init__.py"
        cached.write_text("__version__ = '6.0.3'\n", encoding="utf-8")
        venv = scratch["home"] / VENV_REL
        venv.mkdir(parents=True)
        os.link(cached, venv / "__init__.py")
        return cached, venv / "__init__.py"

    def test_hardlink_traffic_on_a_venv_inode_does_not_fail_the_slot(
        self, fake_codex, monkeypatch, capsys
    ):
        """BUG (2026-09-30, plans/next-phase-command/review-plan): the
        ``codex:gpt-6-astra:xhigh`` slot failed closed after 5m24s naming 23
        pyyaml files of the plugin's own ``.venv`` whose bytes and mtime were
        untouched — their inodes are hardlinks into the uv cache, and a
        ``uv run`` elsewhere moved only their ctime."""
        _pin_verdict(monkeypatch, RUNG)
        cached, watched = self._seed_linked_venv(fake_codex)
        ctime_before = os.lstat(watched).st_ctime_ns
        monkeypatch.setenv("FAKE_CODEX_MODE", "link")
        monkeypatch.setenv("FAKE_LINK_SRC", str(cached))
        phase_dir, status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out

        assert os.lstat(watched).st_ctime_ns != ctime_before  # the window really moved it
        assert CODEX_HOME_CHECK_MARKER not in out
        assert json.loads(_result_file(phase_dir).read_text()) == FINDINGS
        assert status["models_failed"] == {}

    def test_a_content_change_through_the_cache_name_still_fails_closed(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        cached, _watched = self._seed_linked_venv(fake_codex)
        monkeypatch.setenv("FAKE_CODEX_MODE", "link-write")
        monkeypatch.setenv("FAKE_LINK_SRC", str(cached))
        phase_dir, status = _run_review_plan(fake_codex, expect_exit=3)
        out = capsys.readouterr().out

        (line,) = _slot_lines(out, CODEX_HOME_CHECK_MARKER)
        assert os.path.join(VENV_REL, "__init__.py") in line
        assert "fails closed" in line
        assert not _result_file(phase_dir).exists()
        assert CODEX_HOME_CHECK_MARKER in status["models_failed"][SPEC]


# ---------------------------------------------------------------------------
# The jail's own failure, and denials inside a command's output
# ---------------------------------------------------------------------------


class TestAJailDenialIsToldApart:
    def test_a_codex_internal_eacces_is_a_jail_failure_named_twice(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        outside = fake_codex["tmp"] / "outside"
        outside.mkdir()
        denied = outside / "state.db"
        monkeypatch.setenv("FAKE_CODEX_MODE", "eacces")
        monkeypatch.setenv("FAKE_DENIED_PATH", str(denied))
        # An installed codex one minor release past what the jail was measured
        # through — derived from the constant, never hard-coded.
        measured = PLUGIN_LANDLOCK_RUNG.measured_through[0]
        major, minor = measured.split(".")[:2]
        installed = f"{major}.{int(minor) + 1}.0"
        monkeypatch.setitem(executor._PROBE_VERSION_CACHE, "codex", installed)
        phase_dir, status = _run_review_plan(fake_codex, expect_exit=3)
        out = capsys.readouterr().out

        (line,) = _slot_lines(out, JAIL_DENIED_MARKER)
        notice = line[len(f"[{SPEC}] "):]
        # HOME is tmp_path here, so the path renders under `~`.
        assert "~/outside/state.db" in notice and str(fake_codex["tmp"]) not in notice
        assert SPEC in notice and f"'{RUNG}'" in notice
        assert f"installed {installed}" in notice
        assert f"measured through {measured}" in notice
        assert "the likely cause" in notice
        assert "allow list: dir:~/codex-home" in notice
        assert "multi_llm.py doctor" in notice
        assert "github.com/beastlabai/multi-llm-plugin/issues" in notice
        assert "MULTI_LLM_CODEX_SKIP_RUNG=plugin-landlock" in notice
        summary = out.split("=== Sandbox and jail checks")[-1]
        assert notice in summary
        # Named with the other failed models; never retried, never re-rung.
        assert "denied by the plugin's jail: ~/outside/state.db" in status["models_failed"][SPEC]
        assert fake_codex["codex_calls"]() == 1
        assert status["models_served"][SPEC] == {
            "sandbox": "fallback-plugin-landlock",
            "failure_reason": "jail-denied",
            "jail_denied_path": "~/outside/state.db",
        }

    def test_the_same_text_inside_a_command_fails_nothing_and_is_counted(
        self, fake_codex, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        target = fake_codex["tmp"] / "outside" / "cache.txt"
        monkeypatch.setenv("FAKE_CODEX_STREAM", _stream(
            "/bin/bash -lc 'echo x > %s'" % target,
            output=f"/bin/bash: line 1: {target}: Permission denied\n",
        ))
        phase_dir, status = _run_review_plan(fake_codex)
        out = capsys.readouterr().out
        assert JAIL_DENIED_MARKER not in out
        assert _result_file(phase_dir).exists()
        (line,) = [line for line in out.splitlines() if "inside the model's own commands" in line]
        assert "denied 1 write " in line and "not a failure" in line
        assert "~/outside/cache.txt" in line
        assert "failure_reason" not in status["models_served"][SPEC]


# ---------------------------------------------------------------------------
# The porcelain predicate, the sibling counts and the mode predicate
# ---------------------------------------------------------------------------


V = base.SandboxVerdict


class TestThePorcelainCheckIsSkippedForTheNewRungOnly:
    def test_the_porcelain_gate_reads_the_rung_id(self):
        # The leg `test_subagent_executor.py`'s test of this name would gain.
        assert base.porcelain_check_applies(V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", "unsandboxed"))
        assert not base.porcelain_check_applies(V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", RUNG))
        assert not base.porcelain_check_applies(
            V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", RUNG, (), True)
        )
        assert not base.porcelain_check_applies(V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", "landlock"))
        assert not base.porcelain_check_applies(V(base.SANDBOX_DISPATCH, None))
        assert not base.porcelain_check_applies(V(base.SANDBOX_REFUSE, "x"))

    def test_the_mode_and_home_checks_are_its_mirror_image(self):
        for predicate in (base.mode_check_applies, base.home_check_applies):
            assert predicate(V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", RUNG))
            assert predicate(V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", "some-rung", (), True))
            assert not predicate(V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", "unsandboxed"))
            assert not predicate(V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", "landlock"))
            assert not predicate(V(base.SANDBOX_DISPATCH, None))
            assert not predicate(V(base.SANDBOX_REFUSE, "x"))

    def _siblings(self, monkeypatch, verdicts):
        """Four sibling providers, each answering the gate as scripted."""
        states = {name: ("not-required" if v == "none" else "unavailable")
                  for name, v in verdicts.items()}
        monkeypatch.setattr(
            base, "_fanout_sandbox_verdict",
            lambda provider, _cwd: SandboxProbeResult(states[provider], ""),
        )

        def gate(provider, _cwd):
            rung = verdicts[provider]
            if rung == "none":
                return V(base.SANDBOX_DISPATCH, None)
            return V(base.SANDBOX_DISPATCH_UNSANDBOXED, "x", rung, (), rung == RUNG)

        monkeypatch.setattr(base, "sandbox_refusal", gate)
        monkeypatch.setattr(
            base, "parse_model_spec", lambda spec, anchor=None: tuple(spec.split(":", 1)),
        )

    def _count(self, fn, roster, spec):
        async def slot():
            return fn(spec, None, None)

        async def main():
            return await base.spawn_fanout_slot(roster, slot())

        return asyncio.run(main())

    def test_a_sibling_on_the_jail_is_not_boundaryless_but_can_chmod(self, monkeypatch):
        self._siblings(monkeypatch, {
            "self": RUNG, "jailed": RUNG, "bare": "unsandboxed", "plain": "none",
        })
        roster = ["self:m", "jailed:m", "bare:m", "plain:m"]
        # `bare` (uncontained) and `plain` (not-required) count; `jailed` does not.
        assert self._count(base.boundaryless_sibling_count, roster, "self:m") == 2
        # The mode check's count adds the sibling on the jail.
        assert self._count(base.mode_check_sibling_count, roster, "self:m") == 3


# ---------------------------------------------------------------------------
# The [SANDBOX_INCIDENTAL] line, and a mixed fan-out's [SANDBOX_BYPASSED] count
# ---------------------------------------------------------------------------


class TestTheIncidentalNotice:
    def test_the_gate_picks_the_incidental_line_by_grade(self, monkeypatch):
        _pin_verdict(monkeypatch, RUNG)
        verdict = base.sandbox_refusal("codex", None)
        assert verdict.rung == RUNG and verdict.incidental
        assert base.sandbox_fallback_kwargs(verdict) == {"sandbox_fallback": RUNG}
        line = verdict.reason
        assert line.startswith(SANDBOX_INCIDENTAL_MARKER)
        assert BWRAP_LINE in line and f"'{RUNG}'" in line
        # The walk clause of the rung before it: the legacy rung, by version.
        assert "'landlock' not tried" in line
        assert "read-only phase" in line and "stops stray writes, not the model" in line
        assert "checked after this slot" in line
        for word in (SANDBOX_BYPASSED_MARKER, SANDBOX_FALLBACK_MARKER, "contained", "sandboxed"):
            assert word not in line.replace("sandbox_fallback", ""), word

    def test_a_mixed_fanout_counts_only_the_uncontained_slots(
        self, scratch, monkeypatch, capsys
    ):
        _pin_verdict(monkeypatch, RUNG)
        real_gate = base.sandbox_refusal

        def gate(provider, cwd):
            if provider == "claude-code":
                return V(
                    base.SANDBOX_DISPATCH_UNSANDBOXED,
                    base.sandbox_bypassed_notice("claude-code"), "unsandboxed",
                )
            return real_gate(provider, cwd)

        def invoke(**kwargs):
            rung = kwargs.get("sandbox_fallback")
            return {
                "success": True, "data": FINDINGS, "source": "stdout_fallback",
                "output_file": "", "details": {"sandbox": "bypassed" if rung == "unsandboxed" else rung},
            }

        monkeypatch.setattr(base, "sandbox_refusal", gate)
        monkeypatch.setattr(base, "invoke_with_file_output", invoke)
        monkeypatch.setattr(base, "porcelain_snapshot", lambda *_a, **_k: None)
        monkeypatch.setattr(base, "PROVIDER_STAGGER_DELAY", 0.0)
        argv = [
            "--plan-file", str(scratch["plan"]), "--models", SPEC, "claude-code:sonnet",
            "--skip-validation", "--timeout", "3600", "--max-parallel", "2",
            "--rerun-all", "--force",
        ]
        assert asyncio.run(rpo.main(argv)) == 0
        out = capsys.readouterr().out
        bypassed = [line for line in out.splitlines() if SANDBOX_BYPASSED_MARKER in line]
        assert len(bypassed) == 1 and bypassed[0].startswith("[claude-code:sonnet]"), out
        incidental = [line for line in out.splitlines() if SANDBOX_INCIDENTAL_MARKER in line]
        assert len(incidental) == 1 and incidental[0].startswith(f"[{SPEC}]")
        status = status_file.read_status(str(base.get_phase_dir(scratch["plan"], "review-plan")))
        assert status["progress"]["sandbox_bypassed"] == ["claude-code:sonnet"]
        assert status["progress"]["sandbox_fallback"] == {SPEC: RUNG}


# ---------------------------------------------------------------------------
# Carrying the grant: the invoker owns the attempt's TMPDIR and home read-back
# ---------------------------------------------------------------------------


def _completed(stdout, returncode=0, stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _invoke(scratch, **kwargs):
    return llm_client.invoke_with_provider(
        "prompt", SPEC, timeout=60, cwd=str(scratch["repo"]),
        sandbox_fallback=RUNG, **kwargs,
    )


def _tmpdir_of(argv):
    entries = [a for a in argv if a.startswith("dir:") and "multi-llm-slot-" in a]
    assert len(entries) == 1, argv
    return entries[0][len("dir:"):]


class TestCarryingTheGrant:
    @pytest.fixture(autouse=True)
    def _which(self, scratch):
        with patch("shutil.which", return_value="/usr/bin/codex"):
            yield

    def test_each_attempt_gets_its_own_scratch_dir(self, scratch, monkeypatch):
        """Five concurrent slots — every one alive at once, held at a barrier
        inside its "child" — each with a directory of its own."""
        seen = []
        lock = threading.Lock()
        barrier = threading.Barrier(5, timeout=30)
        parent_tmpdir = os.environ.get("TMPDIR")

        def fake_run(cmd, **kwargs):
            env = kwargs["env"]
            tmpdir = _tmpdir_of(cmd)
            record = {
                "argv_tmpdir": tmpdir, "env_tmpdir": env["TMPDIR"],
                "exists": os.path.isdir(tmpdir),
                "mode": os.stat(tmpdir).st_mode & 0o777,
                "home_entry": f"dir:{os.path.realpath(scratch['home'])}" in cmd,
            }
            barrier.wait()
            with lock:
                seen.append(record)
            return _completed(_stream())

        results = []
        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            threads = [
                threading.Thread(target=lambda: results.append(_invoke(scratch)))
                for _ in range(5)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)
        assert len(results) == 5
        assert all(r["success"] for r in results)
        assert all(r["details"]["sandbox"] == RUNG for r in results)
        assert len({s["argv_tmpdir"] for s in seen}) == 5
        for s in seen:
            assert s["exists"] and s["mode"] == 0o700 and s["home_entry"]
            assert s["env_tmpdir"] == s["argv_tmpdir"]
            assert os.path.dirname(s["argv_tmpdir"]) == os.path.realpath(scratch["slot_base"])
            assert os.path.basename(s["argv_tmpdir"]).startswith(f"multi-llm-slot-{os.getpid()}-")
            # Removed by the invoker once the child is gone.
            assert not os.path.exists(s["argv_tmpdir"])
        # The parent's own TMPDIR never changes.
        assert os.environ.get("TMPDIR") == parent_tmpdir

    def test_a_timed_out_attempts_dir_is_gone_after_the_kill(self, scratch):
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(_tmpdir_of(cmd))
            raise subprocess.TimeoutExpired(cmd, 60, output=_stream(findings=False), stderr="")

        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            result = _invoke(scratch)
        assert result["error_code"] == llm_client.ERROR_TIMEOUT
        assert result["details"]["sandbox"] == RUNG
        assert not os.path.exists(seen[0])

    def test_a_cancelled_coroutine_leaves_the_dir_until_the_child_is_reaped(self, scratch):
        release = threading.Event()
        started = threading.Event()
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(_tmpdir_of(cmd))
            started.set()
            release.wait(30)
            return _completed(_stream())

        async def main():
            task = asyncio.ensure_future(asyncio.to_thread(_invoke, scratch))
            # Bounded: an invoke that never reaches the spawn (macOS, 2026-10-23)
            # must fail this row, not hold the whole leg until its cap.
            deadline = time.monotonic() + 30
            while not started.is_set():
                if task.done() or time.monotonic() > deadline:
                    pytest.fail(f"the invoke never reached the spawn: {task.done() and task.result()!r}")
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # The awaiting coroutine is gone; the thread and its child are not.
            assert os.path.isdir(seen[0])
            release.set()

        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            asyncio.run(main())
            deadline = time.time() + 30
            while os.path.exists(seen[0]) and time.time() < deadline:
                time.sleep(0.02)
        assert not os.path.exists(seen[0])

    def test_codex_home_is_read_as_the_child_sees_it(self, scratch, monkeypatch):
        custom = scratch["tmp"] / "custom-home"
        custom.mkdir()
        (custom / "config.toml").write_text("", encoding="utf-8")
        # The adapter's own env delta, `~`-expanded by the spawn site.
        monkeypatch.setattr(CodexProvider, "get_env", lambda self, model: {"CODEX_HOME": "~/custom-home"})
        argv_seen = []

        def fake_run(cmd, **kwargs):
            argv_seen.append(cmd)
            with open(custom / "AGENTS.md", "w", encoding="utf-8") as handle:
                handle.write("injected\n")
            return _completed(_stream())

        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            result = _invoke(scratch)
        assert f"dir:{os.path.realpath(custom)}" in argv_seen[0]
        assert f"dir:{os.path.realpath(scratch['home'])}" not in argv_seen[0]
        check = result["details"]["home_check"]
        assert check["home"] == "~/custom-home" and check["changed"] == ["AGENTS.md"]

    def test_no_adapter_attribute_holds_a_path_afterwards(self, scratch):
        tmpdirs = []

        def fake_run(cmd, **kwargs):
            tmpdirs.append(_tmpdir_of(cmd))
            return _completed(_stream())

        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            _invoke(scratch)
        provider = get_provider("codex")
        text = repr(vars(provider))
        assert tmpdirs[0] not in text
        assert str(scratch["home"]) not in text
        assert "multi-llm-slot-" not in text

    def test_a_missing_codex_home_is_a_jail_failure_with_nothing_spawned(
        self, scratch, monkeypatch
    ):
        monkeypatch.setenv("CODEX_HOME", str(scratch["tmp"] / "no-such-home"))
        fake_run = MagicMock()
        before = set(os.listdir(scratch["slot_base"]))
        with patch("utils.llm_client.subprocess.run", fake_run):
            result = _invoke(scratch)
        fake_run.assert_not_called()
        details = result["details"]
        assert details["failure_reason"] == "jail-denied" and details["retryable"] is False
        assert details["jail_failure"]["path"].endswith("no-such-home")
        assert JAIL_DENIED_MARKER in details["jail_failure"]["notice"]
        assert "denied by the plugin's jail: ~/no-such-home" in result["error"]
        assert not (scratch["tmp"] / "no-such-home").exists()
        assert set(os.listdir(scratch["slot_base"])) == before

    def test_a_temp_base_inside_the_repo_is_a_launch_failure(self, scratch, monkeypatch):
        inside = scratch["repo"] / ".tmp"
        inside.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(inside))
        fake_run = MagicMock()
        with patch("utils.llm_client.subprocess.run", fake_run):
            result = _invoke(scratch)
        fake_run.assert_not_called()
        assert result["details"]["retryable"] is False
        assert result["details"]["failure_reason"] == "jail-setup-failed"
        # The fan-out ladder maps it to `jail_denied`, never retried.
        assert base.is_jail_denied(result["details"])
        assert os.listdir(inside) == []

    def test_a_required_entry_gone_since_the_probe_is_a_jail_failure(self, scratch):
        gone = scratch["tmp"] / "vanished"

        def fake_run(cmd, **kwargs):
            return _completed("", 125, f"landlock_exec: required-entry: dir:{gone}: missing\n")

        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            result = _invoke(scratch)
        details = result["details"]
        assert details["failure_reason"] == "jail-denied" and details["retryable"] is False
        assert details["jail_failure"]["path"] == str(gone)
        assert details["sandbox"] == RUNG

    def test_an_ordinary_permission_error_says_it_is_not_a_jail_denial(self, scratch):
        def fake_run(cmd, **kwargs):
            inside_tmp = os.path.join(_tmpdir_of(cmd), "x")
            return _completed("", 1, f"Error: Permission denied (os error 13) {inside_tmp}\n")

        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            result = _invoke(scratch)
        assert "not a jail denial" in result["error"]
        assert "jail_failure" not in result["details"]
        assert result["details"].get("failure_reason") != "jail-denied"

    def test_a_timed_out_attempt_stays_a_timeout_whatever_it_printed(self, scratch):
        """As on the executor lane: an EACCES codex printed before the deadline
        never turns the kill into a never-retried jail denial."""
        denied = scratch["tmp"] / "outside.txt"

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(
                cmd, 60, output="",
                stderr=f"Error: failed to write {denied}: Permission denied (os error 13)\n",
            )

        with patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            result = _invoke(scratch)
        details = result["details"]
        assert result["error_code"] == llm_client.ERROR_TIMEOUT
        assert details.get("failure_reason") != "jail-denied"
        assert details.get("retryable") is not False
        assert "jail_failure" not in details
        assert "denied by the plugin's jail" not in result["error"]

    @staticmethod
    def _answer_stream(*messages, failed=None):
        lines = [json.dumps({"type": "thread.started", "thread_id": "t1"})]
        for i, text in enumerate(messages):
            lines.append(json.dumps({"type": "item.completed", "item": {
                "id": f"item_{i}", "type": "agent_message", "text": text,
            }}))
        if failed is not None:
            lines.append(json.dumps({"type": "turn.failed", "error": {"message": failed}}))
        else:
            lines.append(json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 10, "output_tokens": 5,
            }}))
        return "\n".join(lines) + "\n"

    def test_an_exit_0_prose_answer_is_never_rewritten_as_jail_denied(self, scratch):
        """codex's ``parse_output`` succeeds only on JSON, so a finished prose
        answer reads as a failed parse; an EACCES-looking line on codex's
        stderr must not turn it into a never-retried jail denial."""
        denied = scratch["tmp"] / "outside.txt"
        stream = self._answer_stream("The plan covers the email field; it looks sound.")
        stderr = f"Error: failed to write {denied}: Permission denied (os error 13)\n"
        with patch("utils.llm_client.subprocess.run",
                   side_effect=lambda cmd, **kw: _completed(stream, 0, stderr)):
            result = _invoke(scratch, want_answer_text=True)
        details = result["details"]
        assert details.get("failure_reason") != "jail-denied"
        assert "jail_failure" not in details
        assert result.get("error_code") != llm_client.ERROR_SUBPROCESS_FAILED
        assert "denied by the plugin's jail" not in (result.get("error") or "")
        assert "it looks sound" in details[llm_client.ANSWER_TEXT_KEY]

    def test_an_exit_0_with_no_answer_is_still_read_as_a_jail_failure(self, scratch):
        denied = scratch["tmp"] / "outside.txt"
        stream = self._answer_stream()
        stderr = f"Error: failed to write {denied}: Permission denied (os error 13)\n"
        with patch("utils.llm_client.subprocess.run",
                   side_effect=lambda cmd, **kw: _completed(stream, 0, stderr)):
            result = _invoke(scratch)
        details = result["details"]
        assert details["failure_reason"] == "jail-denied" and details["retryable"] is False
        assert details["jail_failure"]["path"] == str(denied)
        assert result["error_code"] == llm_client.ERROR_SUBPROCESS_FAILED

    def test_an_exit_0_turn_failed_after_narration_is_a_jail_failure(self, scratch):
        """A ``turn.failed`` means the turn never finished: the narration
        before it is not an answer."""
        denied = scratch["tmp"] / "outside.txt"
        stream = self._answer_stream(
            "Let me read the plan first.",
            failed=f"failed to write {denied}: Permission denied (os error 13)",
        )
        with patch("utils.llm_client.subprocess.run",
                   side_effect=lambda cmd, **kw: _completed(stream, 0, "")):
            result = _invoke(scratch)
        details = result["details"]
        assert details["failure_reason"] == "jail-denied"
        assert details["jail_failure"]["where"] == "event"
        assert details["jail_failure"]["path"] == str(denied)

    def test_the_readonly_audit_runs_on_the_jail_and_not_on_a_contained_rung(self, scratch):
        with patch("utils.llm_client.subprocess.run",
                   side_effect=lambda cmd, **kw: _completed(_stream(CLEAN_COMMAND))):
            jailed = _invoke(scratch)
            contained = llm_client.invoke_with_provider(
                "prompt", SPEC, timeout=60, cwd=str(scratch["repo"]),
                sandbox_fallback="landlock",
            )
        assert jailed["details"]["readonly_audit"]["clean"] is True
        assert "readonly_audit" not in contained["details"]
        assert "home_check" not in contained["details"]
        assert "jail_denials" not in contained["details"]


# ---------------------------------------------------------------------------
# The other two fan-out lanes apply the same verdicts (code review, ask)
# ---------------------------------------------------------------------------


NOT_CLEAN = {"clean": False, "reason": "command_execution: heredoc", "actions": 1}
HOME_CHANGE = {
    "home": "~/codex-home", "changed": ["config.toml"], "config_keys": ["mcp_servers.x"],
    "foreign_trust": [], "other_live_children": 1,
}


class TestTheOtherTwoLanes:
    def _jailed_reply(self, scratch, *, answer):
        os.chmod(scratch["repo"] / "tests" / "run.sh", 0o755)
        return {
            "success": True, "data": answer, "source": "stdout_fallback", "output_file": "",
            "details": {
                "sandbox": RUNG, "readonly_audit": dict(NOT_CLEAN),
                "home_check": dict(HOME_CHANGE),
                "answer_text": answer if isinstance(answer, str) else "",
            },
        }

    def test_code_review_runs_the_mode_check_and_the_home_verdict(
        self, scratch, monkeypatch, capsys
    ):
        import code_review_orchestrator as cro

        _pin_verdict(monkeypatch, RUNG)
        monkeypatch.setattr(cro, "get_project_root", lambda _p: str(scratch["repo"]))
        porcelain = MagicMock(side_effect=AssertionError("no porcelain check on the jail"))
        monkeypatch.setattr(cro, "porcelain_snapshot", porcelain)
        seen = []

        def invoke(**kwargs):
            seen.append(kwargs.get("sandbox_fallback"))
            return self._jailed_reply(scratch, answer=[{"title": "t"}])

        monkeypatch.setattr(cro, "invoke_with_file_output", invoke)
        served = {}
        out_dir = scratch["repo"] / "plans" / "feature"
        success, issues, error = asyncio.run(cro.run_single_review(
            SPEC, scratch["plan"], out_dir, served_models=served,
        ))
        out = capsys.readouterr().out
        assert seen == [RUNG]
        assert success is False and issues is None
        assert MODE_CHECK_MARKER in error and CODEX_HOME_CHECK_MARKER in error
        assert len(_slot_lines(out, MODE_CHECK_MARKER)) == 1
        (home_line,) = _slot_lines(out, CODEX_HOME_CHECK_MARKER)
        assert "config.toml (mcp_servers.x)" in home_line
        assert "1 other codex child of this run was live" in home_line
        assert SANDBOX_BYPASSED_MARKER not in out
        assert served == {SPEC: {"sandbox": "fallback-plugin-landlock"}}
        porcelain.assert_not_called()
        # The home line waits in this phase's closing summary.
        phase_dir = base.get_phase_dir(scratch["plan"], "code-review")
        summary = base.print_closing_summary(str(phase_dir))
        assert summary == [home_line[len(f"[{SPEC}] "):]]

    def test_ask_sets_the_answer_aside_on_a_fail_closed_verdict(
        self, scratch, monkeypatch, capsys
    ):
        import ask_orchestrator

        _pin_verdict(monkeypatch, RUNG)
        seen = []

        def invoke(prompt, model_spec, timeout=None, log_file=None, cwd=None,
                   anchor=None, want_answer_text=False, **kwargs):
            seen.append(kwargs)
            return self._jailed_reply(scratch, answer="An answer.")

        with patch.object(ask_orchestrator, "invoke_with_provider", invoke):
            rc = asyncio.run(ask_orchestrator.main([
                "--plan-file", str(scratch["plan"]), "--question", "Q", "--models", SPEC,
            ]))
        out = capsys.readouterr().out
        assert seen == [{"sandbox_fallback": RUNG}]
        assert rc != 0
        assert len(_slot_lines(out, MODE_CHECK_MARKER)) == 1
        home_lines = [line for line in out.splitlines() if CODEX_HOME_CHECK_MARKER in line
                      and not line.startswith("  - ")]
        # Once for the slot, once in the closing summary.
        assert len([line for line in home_lines if line.startswith(f"[{SPEC}]")]) == 1
        assert any(line.startswith(CODEX_HOME_CHECK_MARKER) for line in home_lines)
        assert SANDBOX_INCIDENTAL_MARKER in out and SANDBOX_BYPASSED_MARKER not in out


class TestTheJailDeniedLine:
    def _notice(self, installed, path="/elsewhere/x"):
        return llm_client.jail_denied_notice(
            "codex", SPEC, PLUGIN_LANDLOCK_RUNG,
            {"where": "stderr", "line": "Permission denied (os error 13)", "path": path,
             "class": "jail"},
            "dir:~/.codex", installed,
        )

    def test_the_likely_cause_only_when_codex_is_newer(self):
        measured = PLUGIN_LANDLOCK_RUNG.measured_through[0]
        assert "the likely cause" not in self._notice(measured)
        assert "the likely cause" not in self._notice("")
        assert "installed unknown" in self._notice("")
        major, minor = measured.split(".")[:2]
        assert "the likely cause" in self._notice(f"codex-cli {major}.{int(minor) + 1}.0")

    def test_an_unnamed_path_is_a_probable_denial(self):
        notice = llm_client.jail_denied_notice(
            "codex", SPEC, PLUGIN_LANDLOCK_RUNG,
            {"where": "event", "line": "Permission denied", "path": None, "class": "probable"},
            "", "",
        )
        assert notice.startswith(JAIL_DENIED_MARKER)
        assert "path not named (Codex did not name the path" in notice

    def test_the_quoted_line_is_redacted(self):
        """The line is codex's raw stderr; the notice reaches stdout, the run
        log and the closing summary, so it goes through the redaction boundary."""
        key = "sk-proj-" + "A1b2C3d4" * 6
        notice = llm_client.jail_denied_notice(
            "codex", SPEC, PLUGIN_LANDLOCK_RUNG,
            {"where": "stderr", "path": "/elsewhere/x", "class": "jail",
             "line": f"Permission denied (os error 13) /elsewhere/x token {key}"},
            "", "",
        )
        assert key not in notice and "<redacted:" in notice


class TestAJailDeniedSlotInAPartialRun:
    def test_the_report_header_and_models_served_name_the_denial(
        self, scratch, monkeypatch, capsys
    ):
        """One slot denied by the jail beside one that answered: the run keeps
        the other slot, names the denied one with the failed models, and its
        record reaches the report through ``models_served``."""
        _pin_verdict(monkeypatch, RUNG)
        real_gate = base.sandbox_refusal

        def gate(provider, cwd):
            if provider == "claude-code":
                return V(base.SANDBOX_DISPATCH, None)
            return real_gate(provider, cwd)

        calls = []

        def invoke(**kwargs):
            spec = kwargs["model_spec"]
            calls.append(spec)
            if spec == SPEC:
                return {
                    "success": False, "error": "codex exited with code 1: denied by "
                    "the plugin's jail: ~/outside/state.db",
                    "error_code": "SUBPROCESS_FAILED", "output_file": "",
                    "details": {
                        "sandbox": RUNG, "failure_reason": "jail-denied",
                        "retryable": False,
                        "jail_failure": {"path": str(scratch["tmp"] / "outside" / "state.db"),
                                         "notice": f"{JAIL_DENIED_MARKER} codex ({SPEC}): …"},
                    },
                }
            return {"success": True, "data": FINDINGS, "source": "stdout_fallback",
                    "output_file": "", "details": {}}

        monkeypatch.setattr(base, "sandbox_refusal", gate)
        monkeypatch.setattr(base, "invoke_with_file_output", invoke)
        argv = [
            "--plan-file", str(scratch["plan"]), "--models", SPEC, "claude-code:sonnet",
            "--skip-validation", "--timeout", "3600", "--max-parallel", "2",
            "--rerun-all", "--force",
        ]
        assert asyncio.run(rpo.main(argv)) == 0
        out = capsys.readouterr().out
        # Never retried, never moved to another rung.
        assert calls.count(SPEC) == 1
        phase_dir = Path(base.get_phase_dir(scratch["plan"], "review-plan"))
        status = status_file.read_status(str(phase_dir))
        assert status["models_served"][SPEC] == {
            "sandbox": "fallback-plugin-landlock", "failure_reason": "jail-denied",
            "jail_denied_path": "~/outside/state.db",
        }
        assert SPEC in status["models_failed"]
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "denied by the plugin's jail: ~/outside/state.db" in report
        # The `[JAIL_DENIED]` line: once for the slot, once in the summary.
        assert len(_slot_lines(out, JAIL_DENIED_MARKER)) == 1
        summary = out.split("=== Sandbox and jail checks")[-1]
        assert f"{JAIL_DENIED_MARKER} codex ({SPEC})" in summary
