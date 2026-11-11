"""The helper layer both release gates share: ``packaging/common/gate.py`` and
the kill routine under it, ``packaging/common/procgroup.py``.

The families' own suites (``test_codex_package.py``,
``test_opencode_release_gate.py``) grade each gate end to end; this file holds
the shared contract once, and checks that both gates use it rather than a copy.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGING = REPO_ROOT / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

from common import gate as common_gate  # noqa: E402
from common import procgroup  # noqa: E402

posix_only = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX; the "
                                "gates run on Linux and macOS")


def _load(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None and Path(module.__file__).resolve() == path.resolve():
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def codex_gate():
    return _load("codex_release_gate_common_test", PACKAGING / "codex" / "release_gate.py")


@pytest.fixture(scope="module")
def opencode_gate():
    return _load("opencode_release_gate", PACKAGING / "opencode" / "release_gate.py")


# ---------------------------------------------------------------------------
# Both gates stand on the one layer
# ---------------------------------------------------------------------------

class TestBothGatesShareIt:
    @pytest.mark.parametrize("family", ["codex_gate", "opencode_gate"])
    def test_the_vocabulary_is_the_shared_one(self, family, request):
        gate = request.getfixturevalue(family)
        assert gate.GateError is common_gate.GateError
        assert gate._STAMP is common_gate.STAMP_RE
        assert gate._read_json is common_gate.read_json
        assert gate._records_for is common_gate.records_for
        assert gate.SESSION_NAME == common_gate.SESSION_NAME

    @pytest.mark.parametrize("path", ["codex/release_gate.py", "opencode/release_gate.py"])
    def test_neither_gate_keeps_its_own_copy(self, path):
        source = (PACKAGING / path).read_text(encoding="utf-8")
        for copied in ("class GateError", "def _records_for", "def _read_json", "os.killpg",
                       "subprocess.Popen(", "uuid.uuid4"):
            assert copied not in source, f"{path} re-implements {copied!r}"

    def test_both_gates_bound_a_step(self, codex_gate, opencode_gate):
        assert codex_gate.STEP_GRACE_S > 0
        assert opencode_gate.step_grace_s() > 0


# ---------------------------------------------------------------------------
# The vocabulary and the session
# ---------------------------------------------------------------------------

class TestRecords:
    def test_records_since_a_stamp_oldest_first(self, tmp_path):
        for name in ("20261101T000000Z-row.json", "20261102T000000Z-row.json",
                     "20261103T000000Z-row.json", "20261102T000000Z-other.json",
                     "latest-row.json"):
            (tmp_path / name).write_text("{}")
        found = common_gate.records_for("row", "20261102T000000Z", tmp_path)
        assert [p.name for p in found] == ["20261102T000000Z-row.json",
                                           "20261103T000000Z-row.json"]

    def test_unreadable_and_invalid_json_are_none(self, tmp_path):
        (tmp_path / "bad.json").write_text("{")
        assert common_gate.read_json(tmp_path / "bad.json") is None
        assert common_gate.read_json(tmp_path / "missing.json") is None


class TestSession:
    def test_begin_deletes_the_earlier_evidence_then_asks_for_the_fields(self, tmp_path):
        (tmp_path / "old-row.json").write_text("{}")
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "x.log").write_text("x")
        (tmp_path / "keep.txt").write_text("kept")
        seen = []

        def fields(moment):
            seen.append(sorted(p.name for p in tmp_path.iterdir()))
            return {"started_stamp": common_gate.compact(moment), "extra": True}

        session = common_gate.begin_session(tmp_path, fields=fields, clear=("logs", "raw"),
                                            create=("logs",))
        assert seen == [["keep.txt", "logs"]], "the fields were read before the deletion"
        assert not list((tmp_path / "logs").iterdir())
        assert list(session) == ["id", "started_at", "started_stamp", "extra"]
        on_disk = json.loads((tmp_path / common_gate.SESSION_NAME).read_text())
        assert on_disk == session
        assert common_gate.current_session(tmp_path) == session

    def test_a_session_without_an_id_is_none(self, tmp_path):
        (tmp_path / common_gate.SESSION_NAME).write_text(json.dumps({"started_at": "x"}))
        assert common_gate.current_session(tmp_path) is None

    def test_codex_keeps_its_id_only_session(self, tmp_path, codex_gate):
        session = codex_gate.begin_session(tmp_path)
        assert sorted(session) == ["id", "started_at"]
        assert codex_gate.current_session(tmp_path) == session["id"]


# ---------------------------------------------------------------------------
# The bounded step
# ---------------------------------------------------------------------------

class TestStep:
    def test_split_keeps_the_gate_flags_off_the_command(self):
        words, command = common_gate.split_step_command(
            ["--evidence-dir", "d", "step", "n", "--timeout", "3", "--", "cmd", "--timeout", "9"])
        assert words == ["--evidence-dir", "d", "step", "n", "--timeout", "3"]
        assert command == ["cmd", "--timeout", "9"]
        assert common_gate.split_step_command(["summary", "--", "x"]) == (["summary", "--", "x"], [])

    def test_stdin_is_the_named_file(self, tmp_path):
        source = tmp_path / "in.txt"
        source.write_text("FROM scratch\n")
        code = common_gate.run_step("cat", [sys.executable, "-c",
                                            "import sys; print(sys.stdin.read().strip())"],
                                    timeout=30, grace=1, directory=tmp_path, stdin=source,
                                    capture=tmp_path / "cap.txt")
        assert code == 0
        assert (tmp_path / "cap.txt").read_text() == "FROM scratch\n"
        assert "FROM scratch" in (tmp_path / "logs" / "cat.log").read_text()

    def test_stdin_defaults_to_nothing(self, tmp_path):
        code = common_gate.run_step("empty", [sys.executable, "-c",
                                              "import sys; assert sys.stdin.read() == ''"],
                                    timeout=30, grace=1, directory=tmp_path)
        assert code == 0

    def test_a_missing_stdin_file_or_binary_cannot_start(self, tmp_path, capsys):
        assert common_gate.run_step("nostdin", [sys.executable, "-c", "pass"], timeout=30,
                                    grace=1, directory=tmp_path,
                                    stdin=tmp_path / "absent") == 127
        assert common_gate.run_step("nobin", [str(tmp_path / "no-such-binary")], timeout=30,
                                    grace=1, directory=tmp_path) == 127
        err = capsys.readouterr().err
        assert "could not open its stdin" in err and "could not start" in err

    @posix_only
    def test_the_bound_ends_the_step_with_124(self, tmp_path, capsys):
        started = time.monotonic()
        code = common_gate.run_step("hang", [sys.executable, "-c", "import time; time.sleep(60)"],
                                    timeout=1, grace=1, directory=tmp_path)
        assert code == 124 and time.monotonic() - started < 30
        assert "exceeded its 1 s bound" in capsys.readouterr().err

    def test_the_codex_gate_runs_a_step_through_its_cli(self, tmp_path, codex_gate, monkeypatch):
        monkeypatch.setattr(codex_gate, "EVIDENCE_DIR", tmp_path)
        source = tmp_path / "Dockerfile"
        source.write_text("ARG CODEX_VERSION\n")
        code = codex_gate.main(["step", "demo", "--timeout", "30", "--stdin", str(source), "--",
                                sys.executable, "-c", "import sys; sys.exit(sys.stdin.read() == '')"])
        assert code == 0
        assert (tmp_path / "logs" / "demo.log").is_file()
        code = codex_gate.main(["step", "fails", "--timeout", "30", "--",
                                sys.executable, "-c", "raise SystemExit(7)"])
        assert code == 7


# ---------------------------------------------------------------------------
# The kill routine
# ---------------------------------------------------------------------------

def _spawn_stubborn(tmp_path, name: str, *, ignore_term: bool):
    marker = tmp_path / f"{name}.pid"
    script = textwrap.dedent(f"""
        import os, signal, subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.{'SIG_IGN' if ignore_term else 'SIG_DFL'}); time.sleep(60)"])
        open({str(marker)!r}, "w").write(str(child.pid))
        time.sleep(60)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    for _ in range(200):
        if marker.exists() and marker.read_text():
            break
        time.sleep(0.02)
    return proc, int(marker.read_text())


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True,
                            text=True, timeout=5).stdout.strip()
    return not status or status.startswith("Z")


class TestStop:
    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError):
            procgroup.stop(None, grace=1, until="never")

    @posix_only
    @pytest.mark.parametrize("until", ["group", "leader"])
    def test_a_term_ignoring_descendant_is_killed(self, tmp_path, until):
        proc, grandchild = _spawn_stubborn(tmp_path, until, ignore_term=True)
        try:
            procgroup.stop(proc, grace=0.5, until=until)
            assert proc.returncode is not None
            for _ in range(100):
                if _gone(grandchild):
                    break
                time.sleep(0.02)
            else:
                pytest.fail("the TERM-ignoring descendant outlived the stop")
        finally:
            for pid in (proc.pid, grandchild):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @posix_only
    def test_the_group_strategy_sends_no_kill_to_a_group_that_ends(self, tmp_path, monkeypatch):
        proc, _grandchild = _spawn_stubborn(tmp_path, "cooperative", ignore_term=False)
        sent = []
        real = os.killpg

        def recording(pgid, sig):
            if sig:
                sent.append(sig)
            return real(pgid, sig)

        monkeypatch.setattr(procgroup.os, "killpg", recording)
        procgroup.stop(proc, grace=10, until="group")
        assert sent == [signal.SIGTERM]
        assert proc.returncode is not None

    @posix_only
    def test_the_leader_strategy_kills_what_its_leader_left(self, tmp_path, monkeypatch):
        """The probe's sweep: the leader's exit is the grace's end, and the rest
        of the session gets SIGKILL straight away."""
        proc, grandchild = _spawn_stubborn(tmp_path, "leader-left", ignore_term=False)
        sent = []
        real = os.killpg

        def recording(pgid, sig):
            if sig:
                sent.append(sig)
            return real(pgid, sig)

        monkeypatch.setattr(procgroup.os, "killpg", recording)
        procgroup.stop(proc, grace=10, session=True, until="leader")
        assert sent == [signal.SIGTERM, signal.SIGKILL]

    @pytest.mark.skipif(not Path("/proc").is_dir(), reason="session members are read from /proc")
    def test_session_members_include_this_session(self):
        assert os.getpid() in procgroup.session_members(os.getsid(0))
