"""``utils/advisory.py``: silent by default, a traceback on stderr under
``MULTI_LLM_DEBUG``, a re-raise under ``MULTI_LLM_STRICT_ADVISORY``.

The last class drives a real ``status`` over a checkpoint whose clock raises:
stdout and the exit code must be byte-identical with and without the debug
switch, and only stderr gains the swallowed traceback.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import advisory, run_status, status_file, status_text  # noqa: E402


@pytest.fixture
def switches(monkeypatch):
    """Both switches off, whatever the conftest set; the test turns one on."""
    monkeypatch.delenv(advisory.DEBUG_ENV, raising=False)
    monkeypatch.delenv(advisory.STRICT_ENV, raising=False)
    return monkeypatch


def _swallow(label="test: a site", error=None):
    """An advisory site as the converted modules write it."""
    try:
        raise error or ValueError("the advisory read failed")
    except Exception as exc:  # noqa: BLE001 - the shape under test
        advisory.note_swallowed(label, exc)
        return "fallback"


class TestNoteSwallowed:
    def test_default_is_silent(self, switches, capfd):
        assert _swallow() == "fallback"
        assert capfd.readouterr() == ("", "")

    @pytest.mark.parametrize("word", ["1", "true", "YES", " on "])
    def test_debug_writes_the_label_and_traceback_to_stderr_only(
        self, switches, capfd, word
    ):
        switches.setenv(advisory.DEBUG_ENV, word)
        assert _swallow("status: native implement view") == "fallback"
        out, err = capfd.readouterr()
        assert out == ""
        assert err.startswith(
            "multi-llm debug: swallowed in status: native implement view:\n"
            "Traceback (most recent call last):\n")
        assert err.rstrip().endswith("ValueError: the advisory read failed")

    @pytest.mark.parametrize("word", ["", "0", "false", "2"])
    def test_other_values_stay_silent(self, switches, capfd, word):
        switches.setenv(advisory.DEBUG_ENV, word)
        switches.setenv(advisory.STRICT_ENV, word)
        assert _swallow() == "fallback"
        assert capfd.readouterr() == ("", "")

    def test_strict_re_raises_the_same_exception(self, switches):
        switches.setenv(advisory.STRICT_ENV, "1")
        error = KeyError("phase")
        with pytest.raises(KeyError) as caught:
            _swallow(error=error)
        assert caught.value is error

    def test_strict_outside_a_pytest_process_is_silent(self, switches, capfd):
        """The variable alone is not enough: a process that is not pytest
        keeps the default guard (simulated by hiding pytest from
        ``sys.modules``; the next test runs a real interpreter)."""
        switches.setenv(advisory.STRICT_ENV, "1")
        switches.delitem(sys.modules, "pytest")
        assert advisory.strict_enabled() is False
        assert _swallow() == "fallback"
        assert capfd.readouterr() == ("", "")

    def test_a_child_interpreter_that_inherits_strict_is_silent(self, switches):
        """What a detached worker a test spawns is: a fresh interpreter with
        the variable in its environment and no pytest imported."""
        import subprocess

        switches.setenv(advisory.STRICT_ENV, "1")
        script = (
            "import sys\n"
            "from utils import advisory\n"
            "try:\n"
            "    raise ValueError('the advisory read failed')\n"
            "except Exception as exc:\n"
            "    advisory.note_swallowed('test: child', exc)\n"
            "print('pytest' in sys.modules, advisory.strict_enabled(), 'fallback')\n"
        )
        child = subprocess.run(
            [sys.executable, "-c", script], cwd=str(Path(__file__).parent.parent),
            capture_output=True, text=True, timeout=60,
        )
        assert (child.returncode, child.stdout, child.stderr) == (
            0, "False False fallback\n", "")

    def test_strict_wins_over_debug(self, switches, capfd):
        switches.setenv(advisory.STRICT_ENV, "1")
        switches.setenv(advisory.DEBUG_ENV, "1")
        with pytest.raises(ValueError):
            _swallow()
        assert capfd.readouterr().err == ""

    def test_the_switch_is_read_at_the_catch(self, switches, capfd):
        """No import-time snapshot: a test's ``monkeypatch.setenv`` and a
        long-lived process both see the current environment."""
        _swallow()
        switches.setenv(advisory.DEBUG_ENV, "1")
        _swallow("test: second")
        assert "swallowed in test: second" in capfd.readouterr().err


class TestTheReportNeverRaises:
    def test_a_broken_stderr_falls_back_to_the_descriptor(self, switches, capfd):
        """A closed or reentrant ``sys.stderr`` costs nothing: the message goes
        straight onto fd 2 through ``stream_bootstrap._FdWriter``."""
        switches.setenv(advisory.DEBUG_ENV, "1")
        broken = io.StringIO()
        broken.close()
        switches.setattr(sys, "stderr", broken)
        assert _swallow("test: closed stderr") == "fallback"
        assert "swallowed in test: closed stderr" in capfd.readouterr().err

    def test_nothing_left_to_write_to_is_still_silent(self, switches):
        switches.setenv(advisory.DEBUG_ENV, "1")
        broken = io.StringIO()
        broken.close()
        switches.setattr(sys, "stderr", broken)

        def refuse(fd, data):
            raise OSError("fd 2 is gone")

        switches.setattr(os, "write", refuse)
        assert _swallow() == "fallback"

    def test_an_exception_whose_str_raises_still_gets_its_label_out(
        self, switches, capfd
    ):
        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no text")

        switches.setenv(advisory.DEBUG_ENV, "1")
        assert _swallow("test: unprintable", Unprintable()) == "fallback"
        assert "swallowed in test: unprintable" in capfd.readouterr().err


class TestTheContextManager:
    def test_swallows_like_the_helper(self, switches, capfd):
        ran = []
        with advisory.advisory("test: block"):
            ran.append(1)
            raise RuntimeError("inside the block")
        assert ran == [1]
        assert capfd.readouterr() == ("", "")

    def test_debug_and_strict(self, switches, capfd):
        switches.setenv(advisory.DEBUG_ENV, "1")
        with advisory.advisory("test: block"):
            raise RuntimeError("inside the block")
        assert "swallowed in test: block" in capfd.readouterr().err
        switches.setenv(advisory.STRICT_ENV, "1")
        with pytest.raises(RuntimeError, match="inside the block"):
            with advisory.advisory("test: block"):
                raise RuntimeError("inside the block")

    def test_base_exceptions_pass_through(self, switches):
        with pytest.raises(KeyboardInterrupt):
            with advisory.advisory("test: block"):
                raise KeyboardInterrupt


class TestAStatusCallUnderDebug:
    """``MULTI_LLM_DEBUG=1 multi_llm.py status`` keeps stdout and the exit code
    byte-identical; the swallowed traceback is the only addition, on stderr."""

    @staticmethod
    def _checkpoint(tmp_path: Path) -> Path:
        plan = tmp_path / "a plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase = tmp_path / "a_plan" / "review-plan"
        phase.mkdir(parents=True)
        beat = status_file.now_iso()
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {
                "step": "fanout", "models_total": 1, "models_done": 0,
                "models_succeeded": 0,
                "models": {"codex:gpt-5.6-sol": {
                    "state": "running", "started_at": beat, "updated_at": beat,
                    "timeout_s": 1800, "attempt": 1,
                }},
            },
        })
        return plan

    def test_stdout_is_unchanged_and_stderr_names_the_site(
        self, tmp_path, switches, capsys
    ):
        plan = self._checkpoint(tmp_path)

        def explode(entry, now=None):
            raise TypeError("can't subtract offset-naive and offset-aware datetimes")

        switches.setattr(status_text, "entry_elapsed_s", explode)
        argv = ["--plan-file", str(plan), "--view", "run"]

        quiet_code = run_status.main(argv)
        quiet = capsys.readouterr()
        switches.setenv(advisory.DEBUG_ENV, "1")
        loud_code = run_status.main(argv)
        loud = capsys.readouterr()

        assert loud_code == quiet_code
        assert loud.out == quiet.out
        assert quiet.err == ""
        assert "multi-llm debug: swallowed in status: model elapsed time:" in loud.err
        assert "TypeError: can't subtract offset-naive" in loud.err
