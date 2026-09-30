"""I12, option B: a run removes only the codex trust entries it added.

Plan ``codex-sandbox-unavailable-findings-and-fixes`` §5 ("I12, the trust
entry: fix and test"), step 5 (2026-09-30). codex 0.158.0–0.159.1 writes
``[projects."<git toplevel>"] trust_level = "trusted"`` into
``$CODEX_HOME/config.toml`` for an untrusted repo, via a temp file and a rename,
forcing mode 0600. The byte shapes below are the ones measured 2026-09-30 on
0.159.0 (append with a leading blank line; inserted after the last projects
table; a file with no trailing newline; a file codex created; a literal-quoted
key). Option A's override was measured to narrow what the child loads, so the
plugin restores after the run instead.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from utils import codex_trust


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(codex_trust, "ENABLED", True)
    codex_trust.reset()
    yield
    codex_trust.reset()


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _home(tmp_path: Path, config: bytes = None, mode: int = 0o600) -> Path:
    home = tmp_path / "codex-home"
    home.mkdir()
    if config is not None:
        path = home / "config.toml"
        path.write_bytes(config)
        os.chmod(path, mode)
    return home


def _codex_trusts(home: Path, key: str) -> None:
    """What codex does: rewrite the whole file with the table appended after a
    blank line, through a temp file and a rename, mode forced to 0600."""
    path = home / "config.toml"
    old = path.read_bytes() if path.exists() else b""
    table = f'[projects."{key}"]\ntrust_level = "trusted"\n'.encode()
    if not old:
        new = table
    elif old.endswith(b"\n"):
        new = old + b"\n" + table
    else:
        new = old + b"\n\n" + table
    tmp = home / ".tmpABC123"
    tmp.write_bytes(new)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


SEED = b"# a comment that must survive\n[tui]\nnotifications = false\n"


def test_an_added_entry_is_removed_byte_for_byte_and_the_mode_restored(tmp_path, armed):
    repo = _repo(tmp_path)
    home = _home(tmp_path, SEED, mode=0o644)
    codex_trust.arm(str(home), [str(repo)])
    _codex_trusts(home, os.path.realpath(repo))
    assert (home / "config.toml").read_bytes() != SEED
    assert codex_trust.restore_all() == []
    assert (home / "config.toml").read_bytes() == SEED
    assert (os.stat(home / "config.toml").st_mode & 0o777) == 0o644


def test_a_file_codex_created_is_removed_again(tmp_path, armed):
    repo = _repo(tmp_path)
    home = _home(tmp_path)
    codex_trust.arm(str(home), [str(repo)])
    _codex_trusts(home, os.path.realpath(repo))
    codex_trust.restore_all()
    assert not (home / "config.toml").exists()


def test_a_final_line_without_newline_is_restored_exactly(tmp_path, armed):
    repo = _repo(tmp_path)
    seed = b"[tui]\nnotifications = false"
    home = _home(tmp_path, seed)
    codex_trust.arm(str(home), [str(repo)])
    _codex_trusts(home, os.path.realpath(repo))
    codex_trust.restore_all()
    assert (home / "config.toml").read_bytes() == seed


def test_an_entry_inserted_after_existing_projects_keeps_every_other_byte(tmp_path, armed):
    repo = _repo(tmp_path)
    other = b'[projects."/some/other/project"]\ntrust_level = "trusted"\n'
    before = b"# seed\n\n" + other + b"\n# a comment between tables\n[tui]\nnotifications = false\n"
    key = os.path.realpath(repo)
    after = (
        b"# seed\n\n" + other + b"\n"
        + f'[projects."{key}"]\ntrust_level = "trusted"\n'.encode()
        + b"\n# a comment between tables\n[tui]\nnotifications = false\n"
    )
    home = _home(tmp_path, before)
    codex_trust.arm(str(home), [str(repo)])
    (home / "config.toml").write_bytes(after)
    codex_trust.restore_all()
    assert (home / "config.toml").read_bytes() == before


def test_a_table_present_before_the_run_is_left_alone(tmp_path, armed):
    repo = _repo(tmp_path)
    key = os.path.realpath(repo)
    seed = SEED + f'\n[projects."{key}"]\ntrust_level = "trusted"\n'.encode()
    home = _home(tmp_path, seed)
    codex_trust.arm(str(home), [str(repo)])
    codex_trust.restore_all()
    assert (home / "config.toml").read_bytes() == seed


def test_a_table_changed_during_the_run_is_kept_with_a_warning(tmp_path, armed, capsys):
    repo = _repo(tmp_path)
    home = _home(tmp_path, SEED)
    codex_trust.arm(str(home), [str(repo)])
    _codex_trusts(home, os.path.realpath(repo))
    path = home / "config.toml"
    path.write_bytes(path.read_bytes() + b'extra_key = "someone added this"\n')
    changed = path.read_bytes()
    warnings = codex_trust.restore_all()
    assert path.read_bytes() == changed
    assert warnings and "left as it is" in warnings[0]
    assert "left as it is" in capsys.readouterr().err


def test_a_literal_quoted_key_is_matched(tmp_path, armed):
    repo = _repo(tmp_path)
    key = os.path.realpath(repo)
    home = _home(tmp_path, SEED)
    codex_trust.arm(str(home), [str(repo)])
    path = home / "config.toml"
    path.write_bytes(SEED + f"\n[projects.'{key}']\ntrust_level = \"trusted\"\n".encode())
    codex_trust.restore_all()
    assert path.read_bytes() == SEED


def test_another_repositorys_new_entry_is_not_this_runs(tmp_path, armed):
    repo = _repo(tmp_path)
    other = _repo(tmp_path, "other")
    home = _home(tmp_path, SEED)
    codex_trust.arm(str(home), [str(repo)])
    _codex_trusts(home, os.path.realpath(other))
    written = (home / "config.toml").read_bytes()
    codex_trust.restore_all()
    assert (home / "config.toml").read_bytes() == written


def test_the_key_is_the_git_toplevel_of_a_subdirectory(tmp_path):
    repo = _repo(tmp_path)
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    assert codex_trust.repo_keys(str(sub))[0] == os.path.realpath(repo)
    assert codex_trust.repo_keys(str(tmp_path / "nowhere-in-git")) == []


def test_arm_is_off_under_the_suite_default(tmp_path):
    repo = _repo(tmp_path)
    home = _home(tmp_path, SEED)
    codex_trust.arm(str(home), [str(repo)])
    assert codex_trust._RECORDS == {}


_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys, time
    sys.path.insert(0, {skill!r})
    import multi_llm
    from utils import codex_trust
    codex_trust.ENABLED = True
    home, repo, how = sys.argv[1], sys.argv[2], sys.argv[3]
    previous = multi_llm._install_foreground_sigterm()
    try:
        codex_trust.arm(home, [repo])
        path = os.path.join(home, "config.toml")
        with open(path, "ab") as handle:
            handle.write(('\\n[projects."' + os.path.realpath(repo) + '"]\\ntrust_level = "trusted"\\n').encode())
        os.chmod(path, 0o600)
        if how == "exception":
            raise RuntimeError("boom")
        if how == "sigterm":
            print("ready", flush=True)
            time.sleep(30)
    finally:
        multi_llm._restore_codex_trust()
        multi_llm._uninstall_foreground_sigterm(previous)
    """
)


@pytest.mark.parametrize("how", ["normal", "exception", "sigterm"])
def test_every_terminal_path_restores_the_file(tmp_path, how):
    skill = str(Path(__file__).resolve().parent.parent)
    repo = _repo(tmp_path)
    home = _home(tmp_path, SEED, mode=0o600)
    script = tmp_path / "run.py"
    script.write_text(_SCRIPT.format(skill=skill))
    proc = subprocess.Popen(
        [sys.executable, str(script), str(home), str(repo), how],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if how == "sigterm":
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=60)
    assert (home / "config.toml").read_bytes() == SEED
    assert (os.stat(home / "config.toml").st_mode & 0o777) == 0o600


#: A real detached worker driving the production ``RunHandle``: it arms a temp
#: codex home, writes the trust table the way codex does, and then waits to be
#: ended from outside. Neither ending under test reaches ``multi_llm.main``'s
#: ``finally`` — both ``os._exit`` — so the restore has to come from ``detach``.
_DETACHED_WORKER = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {skill!r})
    from utils import codex_trust, detach, process_control
    codex_trust.ENABLED = True
    PHASE, HOME, REPO, READY = {phase!r}, {home!r}, {repo!r}, {ready!r}
    process_control.enter_detached_worker(detach.log_path(PHASE))
    handle = detach.RunHandle(
        PHASE, run_id={run_id!r}, mode="review-plan", plan_file={plan!r},
        host="test-host",
    )
    handle.start()
    codex_trust.arm(HOME, [REPO])
    path = os.path.join(HOME, "config.toml")
    with open(path, "ab") as handle_:
        handle_.write(('\\n[projects."' + os.path.realpath(REPO) + '"]\\ntrust_level = "trusted"\\n').encode())
    os.chmod(path, 0o600)
    open(READY, "w").close()
    time.sleep(60)
    handle.finish(0)
    sys.exit(0)
    """
)


def _await(predicate, timeout: float = 30.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _spawn_detached_worker(tmp_path: Path, run_id: str):
    """``(pid, plan, phase_dir, home)`` for a worker that has armed and written."""
    from utils import detach, process_control

    skill = str(Path(__file__).resolve().parent.parent)
    repo = _repo(tmp_path)
    home = _home(tmp_path, SEED, mode=0o600)
    plan = tmp_path / "a plan.md"
    plan.write_text("# Plan\n\n## Tasks\n- [ ] 1. Do the thing\n", encoding="utf-8")
    phase = tmp_path / "a_plan" / "review-plan"
    phase.mkdir(parents=True)
    ready = tmp_path / "worker-ready"
    script = tmp_path / "detached-worker.py"
    script.write_text(_DETACHED_WORKER.format(
        skill=skill, phase=str(phase), home=str(home), repo=str(repo),
        ready=str(ready), run_id=run_id, plan=str(plan),
    ), encoding="utf-8")
    env = dict(os.environ)
    env[detach.WORKER_ENV] = "1"
    env["MULTI_LLM_HEARTBEAT_INTERVAL_S"] = "0.4"
    env["CODEX_HOME"] = str(home)  # never the real ~/.codex
    pid = process_control.spawn_detached(
        [sys.executable, str(script)], cwd=str(tmp_path), env=env,
        log_path=str(detach.log_path(str(phase))),
    )
    assert _await(ready.exists), "the detached worker never armed"
    assert (home / "config.toml").read_bytes() != SEED
    return pid, plan, phase, home


@pytest.mark.skipif(os.name == "nt", reason="POSIX detach (setsid, SIGTERM)")
def test_aborting_a_detached_worker_restores_the_file(tmp_path, capsys):
    """``status --abort`` SIGTERMs the worker, whose handler sweeps the CLIs,
    records ``aborted`` and ``os._exit``s — no ``finally`` runs, so the restore
    has to happen in the handler. The Codex host is ``detach: required``, so
    this is how every Codex-host run is stopped."""
    from utils import detach, process_control, run_status, status_file

    pid, plan, phase, home = _spawn_detached_worker(tmp_path, "run-trust-abort-1")
    assert run_status.main(["--plan-file", str(plan), "--abort"]) == 0
    capsys.readouterr()
    assert _await(lambda: not process_control.process_alive(pid))
    assert status_file.read_run(str(phase)).get("state") == "aborted"
    log = Path(detach.log_path(str(phase))).read_text(encoding="utf-8")
    assert (home / "config.toml").read_bytes() == SEED, log[-2000:]
    assert (os.stat(home / "config.toml").st_mode & 0o777) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX detach (setsid)")
def test_a_detached_worker_that_loses_its_record_restores_the_file(tmp_path):
    """The heartbeat-lost exit: the run was closed out under the worker, which
    stops beating and ``os._exit``s from the heartbeat thread."""
    from utils import detach, process_control, status_file

    run_id = "run-trust-lost-01"
    pid, _plan, phase, home = _spawn_detached_worker(tmp_path, run_id)
    assert status_file.update_run(
        str(phase), {"state": "failed", "error": "closed out by the test"},
        expect_run_id=run_id,
    )
    assert _await(lambda: not process_control.process_alive(pid))
    log = Path(detach.log_path(str(phase))).read_text(encoding="utf-8")
    assert "no longer ours to write" in log, log[-2000:]
    assert (home / "config.toml").read_bytes() == SEED, log[-2000:]


def test_every_codex_dispatch_context_arms_the_restore(tmp_path, armed, monkeypatch):
    """Both spawn sites enter ``llm_client.provider_dispatch_context``; for codex
    that calls ``note_dispatch``, which records the trust state of the home the
    child will see before the child starts (never a refusal, never argv/env)."""
    from utils.llm_client import provider_dispatch_context
    from utils.providers.codex import CodexProvider

    repo = _repo(tmp_path)
    home = _home(tmp_path, SEED)
    monkeypatch.setenv("CODEX_HOME", str(home))
    with provider_dispatch_context(CodexProvider(), "gpt-5.5", "agent-write",
                                   anchor=str(repo)) as dispatch:
        assert dispatch.refusal is None
    record = codex_trust._RECORDS[os.path.realpath(home)]
    assert record.keys[os.path.realpath(repo)] is False
    _codex_trusts(home, os.path.realpath(repo))
    codex_trust.restore_all()
    assert (home / "config.toml").read_bytes() == SEED
