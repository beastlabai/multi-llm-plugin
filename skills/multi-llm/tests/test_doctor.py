#!/usr/bin/env python3
"""`multi_llm.py doctor` (D6 step 5): one line per provider, no model calls,
exit 0 or 3, and the AppArmor profile printed verbatim when a sandbox row fails.

Every provider CLI on the test PATH is a mock: `codex` is `tests/mocks/mock_llm.py`
(whose `sandbox` emulation replays the real machine under
`MOCK_LLM_SANDBOX_BROKEN=1`), `grok` and `muse` are two-line scripts that answer
`--version` and nothing else. The dispatch entry points are patched to raise, so
"no model calls" is observed rather than assumed.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import doctor  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"
BWRAP_BROKEN_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"


def _script(bin_dir: Path, name: str, body: str) -> None:
    target = bin_dir / name
    target.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def clinic(tmp_path, monkeypatch):
    """A PATH carrying mock `codex`, `grok` and `muse` only, and no model path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "codex").symlink_to(MOCK_LLM)
    _script(bin_dir, "grok", 'echo "grok 0.2.93 (abc1234)"')
    _script(bin_dir, "muse", 'echo "Muse Code 0.1.0 (0.1.0-R708.1)"')
    interpreter_dir = str(Path(sys.executable).parent)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(bin_dir), interpreter_dir, "/usr/bin", "/bin"])
    )
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.delenv("MOCK_LLM_SANDBOX_BROKEN", raising=False)
    monkeypatch.delenv("MULTI_LLM_PROVIDERS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    registry._config = None
    registry._config_key = None
    executor.reset_version_cache()
    for name in ("codex", "grok", "muse"):
        resolved = shutil.which(name)
        assert resolved and resolved.startswith(str(bin_dir)), resolved

    def no_model_calls(*args, **kwargs):
        raise AssertionError("doctor dispatched a model call")

    monkeypatch.setattr(executor, "_spawn", no_model_calls)
    from utils import llm_client

    monkeypatch.setattr(llm_client, "invoke_with_provider", no_model_calls)
    yield bin_dir
    registry._config = None
    registry._config_key = None


def _broken_bwrap(monkeypatch):
    monkeypatch.setattr(
        doctor, "_bwrap_self_test", lambda timeout: ("unavailable", BWRAP_BROKEN_LINE)
    )


def _row(report, name):
    return next(r for r in report["providers"] if r["provider"] == name)


class TestDoctorOnABrokenSandbox:

    def test_json_names_the_bwrap_line_prints_the_profile_and_exits_3(
        self, clinic, monkeypatch, capsys
    ):
        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        _broken_bwrap(monkeypatch)
        assert doctor.main(["--json", "--budget", "10"]) == 3
        report = json.loads(capsys.readouterr().out)
        assert report["exit"] == 3 and report["warn_only"] is False
        assert report["sandbox_broken"] is True
        codex = _row(report, "codex")
        assert codex["present"] is True
        assert codex["sandbox"]["state"] == "unavailable"
        assert codex["sandbox"]["probe"] == "runner"
        assert codex["sandbox"]["ok"] is False
        assert BWRAP_BROKEN_LINE in codex["sandbox"]["detail"]
        assert codex["auth"]["state"] == "ok"
        assert codex["version"]["version"]
        assert any(f.startswith("codex: sandbox") for f in report["failures"])
        repair = report["repair"]
        assert repair["apparmor_profile"] == doctor.APPARMOR_PROFILE
        assert "profile bwrap /usr/bin/bwrap flags=(unconfined)" in repair["apparmor_profile"]
        assert "userns," in repair["apparmor_profile"]
        assert repair["sysctl_alternative_recommended"] is False
        assert "apparmor_restrict_unprivileged_userns=0" in repair["sysctl_alternative"]

    def test_text_prints_the_profile_verbatim_and_the_sysctl_caveat(
        self, clinic, monkeypatch, capsys
    ):
        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        _broken_bwrap(monkeypatch)
        assert doctor.main(["--budget", "10"]) == 3
        out = capsys.readouterr().out
        assert "sandbox BROKEN: " + BWRAP_BROKEN_LINE in out
        assert "FAIL: codex: sandbox" in out
        for line in doctor.APPARMOR_PROFILE.rstrip("\n").splitlines():
            assert line in out, line
        assert "sudo apparmor_parser -r /etc/apparmor.d/bwrap" in out
        assert "not recommended" in out
        assert "kernel.apparmor_restrict_unprivileged_userns=0" in out

    def test_warn_only_reports_the_same_and_exits_0(self, clinic, monkeypatch, capsys):
        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        _broken_bwrap(monkeypatch)
        assert doctor.main(["--warn-only", "--budget", "10"]) == 0
        out = capsys.readouterr().out
        assert "WARN: codex: sandbox" in out
        assert "FAIL:" not in out
        assert "profile bwrap /usr/bin/bwrap flags=(unconfined)" in out

    def test_a_bypassed_sandbox_is_reported_never_failed(self, clinic, monkeypatch, capsys):
        """muse names bubblewrap but runs `--disable-sandbox`: the broken
        self-test is on the row, the repair is printed, the exit stays 0."""
        _broken_bwrap(monkeypatch)
        assert doctor.main(["--json", "--budget", "10"]) == 0
        report = json.loads(capsys.readouterr().out)
        muse = _row(report, "muse")
        assert muse["sandbox"]["state"] == "unavailable"
        assert muse["sandbox"]["bypassed_by"] == "--disable-sandbox"
        assert muse["sandbox"]["probe"] == "bwrap"
        assert muse["sandbox"]["ok"] is True
        assert report["sandbox_broken"] is True and report["repair"] is not None
        assert not any(f.startswith("muse") for f in report["failures"])


class TestDoctorOnAHealthyMachine:

    def test_every_present_provider_passes_and_exits_0(self, clinic, monkeypatch, capsys):
        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        assert doctor.main(["--json", "--budget", "10"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["failures"] == []
        assert report["repair"] is None
        codex = _row(report, "codex")
        assert codex["sandbox"] == {
            "state": "ok", "detail": codex["sandbox"]["detail"], "ok": True,
            "mechanism": "bubblewrap", "bypassed_by": "", "probe": "runner",
        }
        # The mock's `codex sandbox` really ran the shell script here, and the
        # script left nothing behind.
        assert not (Path.cwd() / ".multi-llm").exists()
        absent = [r for r in report["providers"] if not r["present"]]
        assert absent and all("version" not in r for r in absent)

    def test_grok_prints_trust_untested(self, clinic, monkeypatch, capsys):
        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        assert doctor.main(["--budget", "10"]) == 0
        out = capsys.readouterr().out
        grok_line = next(l for l in out.splitlines() if l.strip().startswith("grok"))
        assert "trust: untested" in grok_line
        assert "version 0.2.93" in grok_line
        assert _row(json.loads(json.dumps(doctor.collect(budget_s=10))), "grok")["trust"] == "untested"

    def test_a_logged_out_cli_is_a_failure_with_its_login_command(
        self, clinic, monkeypatch, capsys
    ):
        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        monkeypatch.setenv("MOCK_LLM_LOGGED_OUT", "1")
        assert doctor.main(["--json", "--budget", "10"]) == 3
        report = json.loads(capsys.readouterr().out)
        codex = _row(report, "codex")
        assert codex["auth"]["state"] == "not-logged-in"
        assert "codex login" in codex["auth"]["detail"]
        assert any("codex: auth" in f for f in report["failures"])

    def test_a_slow_probe_is_unprobed_not_a_blown_budget(self, clinic, monkeypatch, capsys):
        """A `grok --version` that sleeps past the budget is reported
        `unprobed (slow…)`; the report still ends at the budget, and a slow
        probe is never counted as a failure."""
        import time

        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        _script(clinic, "grok", 'sleep 5\necho "grok 0.2.93 (abc1234)"')
        started = time.monotonic()
        assert doctor.main(["--json", "--budget", "0.5"]) == 0
        assert time.monotonic() - started < 4.0
        report = json.loads(capsys.readouterr().out)
        grok = _row(report, "grok")
        assert grok["present"] is True
        assert grok["version"]["state"] == "unprobed"
        assert grok["version"]["detail"].startswith("slow")
        assert grok["version"]["ok"] is True
        assert grok["trust"] == "untested"
        assert report["failures"] == []
        assert report["elapsed_s"] < 4.0


class TestDoctorIsAMode:

    def test_the_row_is_dispatcher_driven_and_not_a_workflow_flag(self):
        mode = mode_table.resolve("doctor")
        assert mode is not None
        assert mode.phase is None
        assert mode.instruction_file is None
        assert mode.user_facing is False
        assert mode.supports_detach is False
        assert mode.target_module == "utils.doctor"

    def test_help_and_a_warn_only_run_go_through_the_dispatcher(
        self, clinic, monkeypatch, capsys
    ):
        assert multi_llm.main(["doctor", "--help"]) == 0
        assert "--warn-only" in capsys.readouterr().out
        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        assert multi_llm.main(["doctor", "--warn-only", "--budget", "10"]) == 0
        out = capsys.readouterr().out
        assert "multi-llm doctor" in out
        assert '"mode": "doctor"' in out and '"phase": null' in out

    def test_doctor_takes_no_lock_and_writes_nothing(self, clinic, monkeypatch, capsys):
        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        before = sorted(p.name for p in Path.cwd().iterdir())
        assert doctor.main(["--budget", "10"]) == 0
        capsys.readouterr()
        assert sorted(p.name for p in Path.cwd().iterdir()) == before
