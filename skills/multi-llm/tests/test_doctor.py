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
    # Hermetic: the constraint-2 check reads codex's own config, never the
    # developer's real ~/.codex/config.toml.
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
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

    def test_the_default_fallback_is_named_beside_broken_and_changes_nothing_else(
        self, clinic, monkeypatch, capsys
    ):
        """D6 override (2026-09-17): the shipped `sandbox_fallback: auto` —
        nothing pinned here, the base config is what `doctor` reads — renders
        its note on the BROKEN row so the operator sees what the next run will
        do on this machine, and nothing else moves: the row is still a failure
        (the write profile is refused whatever the key says), the repair is
        still printed, the exit is still 3."""
        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        _broken_bwrap(monkeypatch)
        assert doctor.main(["--budget", "10"]) == 3
        out = capsys.readouterr().out
        # The mock replays a machine where the landlock rung's probe dies on
        # the same bwrap line, so the walk reaches rung 2 — and the row names
        # the contained rung it measured and could not use.
        assert (
            "sandbox BROKEN: " + BWRAP_BROKEN_LINE
            + " (fallback: auto — codex runs unsandboxed for read-only phases"
            "; rung landlock is not usable here (unavailable) — measured at "
            "codex 0.155.1: "
        ) in out
        assert "→ landlock" not in out
        assert "FAIL: codex: sandbox" in out
        assert "profile bwrap /usr/bin/bwrap flags=(unconfined)" in out

        assert doctor.main(["--json", "--budget", "10"]) == 3
        report = json.loads(capsys.readouterr().out)
        codex = _row(report, "codex")
        assert codex["sandbox"]["state"] == "unavailable"
        assert codex["sandbox"]["fallback"] == "auto"
        assert codex["sandbox"]["fallback_rung"] == "unsandboxed"
        assert codex["sandbox"]["fallback_walk"] == [
            ["landlock", "unavailable"], ["unsandboxed", "taken"],
        ]
        assert codex["sandbox"]["ok"] is False

    def test_off_says_refused_by_config_and_probes_no_rung(
        self, clinic, monkeypatch, capsys
    ):
        """Landlock rung (2026-09-22): `off` is still a refusal, and the row
        now says so — refused by config, not merely unavailable. No rung is
        reachable under `off`, so the read-only probe is never spent."""
        from utils.providers import codex as codex_module
        from utils.providers.base import PROFILE_GENERATE

        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        _broken_bwrap(monkeypatch)
        monkeypatch.setattr(
            codex_module, "sandbox_fallback_policy", lambda anchor=None: "off"
        )
        real = executor.sandbox_verdict
        profiles = []

        def spy(provider, profile, **kwargs):
            profiles.append(profile)
            return real(provider, profile, **kwargs)

        monkeypatch.setattr(executor, "sandbox_verdict", spy)
        assert doctor.main(["--budget", "10"]) == 3
        out = capsys.readouterr().out
        assert (
            "sandbox BROKEN: " + BWRAP_BROKEN_LINE
            + " (fallback: off — codex is refused by config for read-only "
            "phases too, not merely unavailable)\n"
        ) in out
        assert "unsandboxed" not in out
        assert "landlock" not in out
        assert PROFILE_GENERATE not in profiles

        assert doctor.main(["--json", "--budget", "10"]) == 3
        report = json.loads(capsys.readouterr().out)
        assert _row(report, "codex")["sandbox"]["fallback"] == "off"

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

    def test_one_providers_three_probes_run_concurrently(self, monkeypatch):
        import threading
        from types import SimpleNamespace

        provider = SimpleNamespace(is_available=lambda: True)
        monkeypatch.setattr(registry, "load_config", lambda **kwargs: {"providers": {"fake": {}}})
        monkeypatch.setattr(registry, "get_provider", lambda name: provider)
        monkeypatch.setattr(doctor, "_probe_cwd", lambda anchor: None)
        barrier = threading.Barrier(3, timeout=2)

        def probe(*args):
            barrier.wait()
            return {"ok": True, "state": "unprobed"}

        for kind in ("version", "auth", "sandbox"):
            monkeypatch.setattr(doctor, f"_probe_{kind}", probe)
        report = doctor.collect(budget_s=3)
        assert report["present"] == 1
        assert report["failures"] == []

    def test_exhausted_budget_does_not_start_another_probe(self, monkeypatch):
        def forbidden(*args, **kwargs):
            pytest.fail("an exhausted report budget started a subprocess")

        monkeypatch.setattr(doctor.subprocess, "run", forbidden)
        monkeypatch.setattr(executor, "sandbox_verdict", forbidden)
        assert doctor._run(["unused"], timeout=0) is None
        from types import SimpleNamespace

        result = doctor._probe_sandbox(SimpleNamespace(), 0, None)
        assert result["state"] == "unprobed"
        assert result["ok"] is True

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
            # D6 override (2026-09-17): the resolved policy rides the row; the
            # shipped default is `auto`, and a healthy sandbox renders no note
            # (the fallback only means anything where the sandbox cannot start).
            "fallback": "auto",
            # Landlock rung (2026-09-22): the read-only probe runs only when
            # the row is BROKEN, so a healthy row carries it unspent.
            "fallback_probe": "", "fallback_walk": [], "fallback_rung": None,
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


# ---------------------------------------------------------------------------
# Landlock rung (2026-09-22, T6): doctor reports the ladder it probed.
# ---------------------------------------------------------------------------

LANDLOCK_SENTENCE = "`[features].use_legacy_landlock` is deprecated and will be removed soon."


def _fake_verdicts(monkeypatch, generate_verdict, calls=None):
    """`agent-write` answers the broken bwrap line; `generate` answers
    ``generate_verdict`` — the one place a rung can come from."""
    from utils.providers.base import PROFILE_GENERATE

    def fake(provider, profile, **kwargs):
        if calls is not None:
            calls.append((profile, kwargs.get("timeout")))
        if profile == PROFILE_GENERATE:
            return generate_verdict
        return executor.SandboxProbeResult("unavailable", BWRAP_BROKEN_LINE)

    monkeypatch.setattr(executor, "sandbox_verdict", fake)


def _codex_line(out):
    return next(ln for ln in out.splitlines() if ln.strip().startswith("codex"))


class TestDoctorReportsTheLadder:

    def test_a_rung_1_verdict_is_what_puts_landlock_on_screen(
        self, clinic, monkeypatch, capsys
    ):
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "landlock", (("landlock", "taken"),),
        ))
        assert doctor.main(["--budget", "10"]) == 3
        out = capsys.readouterr().out
        line = _codex_line(out)
        assert (
            "sandbox BROKEN: " + BWRAP_BROKEN_LINE + " (fallback: auto → landlock — "
            "codex runs read-only phases under a real sandbox; agent-write still refused"
        ) in line
        # `deprecated_at` rides the row: the version it was measured at and the
        # sentence upstream prints.
        assert "measured at codex 0.155.1: " + LANDLOCK_SENTENCE in line
        assert "unsandboxed" not in line
        # The verdict and the exit code do not move: agent-write is refused.
        assert "FAIL: codex: sandbox" in out
        assert "profile bwrap /usr/bin/bwrap flags=(unconfined)" in out

        assert doctor.main(["--json", "--budget", "10"]) == 3
        sandbox = _row(json.loads(capsys.readouterr().out), "codex")["sandbox"]
        assert sandbox["state"] == "unavailable" and sandbox["ok"] is False
        assert sandbox["fallback_rung"] == "landlock"
        # The split (plan §3.3): two keys, and the old one is gone.
        assert sandbox["fallback_deprecated_at"] == ["0.155.1", LANDLOCK_SENTENCE]
        removed_version, evidence = sandbox["fallback_removed_at"]
        assert removed_version == "0.156.1" and "2026-09-24" in evidence
        assert "fallback_retires_when" not in sandbox

    def test_no_row_names_a_rung_the_probe_did_not_return(
        self, clinic, monkeypatch, capsys
    ):
        """The key says `auto`, which *could* reach landlock — but the probe
        selected rung 2, so the row renders today's unsandboxed sentence and
        never the word `landlock →`."""
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "unsandboxed",
            (("landlock", "absent"), ("unsandboxed", "taken")),
        ))
        assert doctor.main(["--budget", "10"]) == 3
        line = _codex_line(capsys.readouterr().out)
        assert "(fallback: auto — codex runs unsandboxed for read-only phases" in line
        assert "→ landlock" not in line
        assert "rung landlock is not usable here (absent)" in line

    def test_a_none_rung_after_a_complete_walk_is_a_refusal_not_a_bypass(
        self, clinic, monkeypatch, capsys
    ):
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, None, (("landlock", "rejected"),),
        ))
        assert doctor.main(["--budget", "10"]) == 3
        line = _codex_line(capsys.readouterr().out)
        assert "(fallback: auto — no rung usable here: codex is refused" in line
        assert "runs unsandboxed" not in line
        assert "→ landlock" not in line

    def test_an_unprobed_rung_is_its_own_row(self, clinic, monkeypatch, capsys):
        """Unprobed and unavailable are different rows: one says this machine
        has no contained rung, the other says the report did not find out."""
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, None, (("landlock", "unprobed"),),
        ))
        assert doctor.main(["--budget", "10"]) == 3
        line = _codex_line(capsys.readouterr().out)
        assert line.endswith(
            "(fallback: auto — rung unprobed: the report budget was exhausted)"
        )
        assert "unsandboxed" not in line and "not usable" not in line

    def test_contained_names_the_disabled_bypass_and_its_own_rung(
        self, clinic, monkeypatch, capsys
    ):
        from utils.providers import codex as codex_module

        monkeypatch.setattr(
            codex_module, "sandbox_fallback_policy", lambda anchor=None: "contained"
        )
        # The walk is policy-blind: it reaches rung 2 — which `contained` never takes.
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "unsandboxed",
            (("landlock", "unavailable"), ("unsandboxed", "taken")),
        ))
        assert doctor.main(["--budget", "10"]) == 3
        line = _codex_line(capsys.readouterr().out)
        assert "(fallback: contained — bypass disabled: no contained rung is usable here" in line
        assert "runs unsandboxed" not in line

        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "landlock", (("landlock", "taken"),),
        ))
        assert doctor.main(["--budget", "10"]) == 3
        assert "(fallback: contained → landlock — " in _codex_line(capsys.readouterr().out)

    def test_the_second_probe_gets_only_what_is_left_of_the_budget(self, monkeypatch):
        """A slow `agent-write` probe truncates the read-only probe rather than
        lengthening the report: with nothing left it is never started, and
        the row says unprobed."""
        import time
        from utils.providers import codex as codex_module
        from utils.providers.base import PROFILE_GENERATE

        from utils.provider_registry import get_provider

        provider = get_provider("codex")
        monkeypatch.setattr(
            codex_module, "sandbox_fallback_policy", lambda anchor=None: "auto"
        )
        calls = []

        def slow(provider, profile, **kwargs):
            calls.append((profile, kwargs.get("timeout")))
            if profile == PROFILE_GENERATE:
                pytest.fail("the read-only probe started on an exhausted budget")
            time.sleep(0.25)
            return executor.SandboxProbeResult("unavailable", BWRAP_BROKEN_LINE)

        monkeypatch.setattr(executor, "sandbox_verdict", slow)
        started = time.monotonic()
        row = doctor._probe_sandbox(provider, 0.2, None)
        assert time.monotonic() - started < 1.0
        assert row["state"] == "unavailable" and row["ok"] is False
        assert row["fallback_probe"] == "unprobed"
        assert doctor._fallback_note("codex", row) == (
            " (fallback: auto — rung unprobed: the report budget was exhausted)"
        )

        # With budget left, the second probe is handed the remainder, not the whole.
        calls.clear()
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "landlock", (("landlock", "taken"),),
        ), calls=calls)
        doctor._probe_sandbox(provider, 5.0, None)
        assert [profile for profile, _ in calls][-1] == PROFILE_GENERATE
        assert calls[-1][1] <= calls[0][1]


class TestDoctorNamesTheRetiredRung:
    """Plan §3.4 (2026-09-24): codex 0.156.1 measured the landlock rung
    unavailable. The BROKEN row's *fallback clause* says so — and offers the
    read-phase choice — while the write-profile repair block stays exactly
    as it was."""

    PANIC = (
        "thread 'main' panicked at linux-sandbox/src/linux_run_main.rs:410:5: "
        "filesystem-restricted execution requires bubblewrap to isolate "
        "app-server sockets"
    )

    def _repair_is_unchanged(self, out, report):
        assert "profile bwrap /usr/bin/bwrap flags=(unconfined)" in out
        assert "Repair (yours to install, with sudo — the plugin only prints it):" in out
        assert report["repair"] == doctor._repair_block()
        assert report["repair"]["sysctl_alternative_recommended"] is False

    def test_a_retired_rung_names_the_measured_version_and_the_choice(
        self, clinic, monkeypatch, capsys
    ):
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "unsandboxed",
            (("landlock", "retired"), ("unsandboxed", "taken")),
            (("landlock", self.PANIC),), "0.156.1",
        ))
        assert doctor.main(["--budget", "10"]) == 3
        out = capsys.readouterr().out
        line = _codex_line(out)
        assert "sandbox BROKEN: " + BWRAP_BROKEN_LINE in line
        assert "(fallback: auto — codex runs unsandboxed for read-only phases" in line
        assert "→ landlock unavailable: measured unavailable from codex 0.156.1" in line
        assert "`sandbox_fallback: contained` refuses them like writes" in line
        assert "`auto` runs them under --sandbox danger-full-access" in line
        # The clause carries no AppArmor or /etc advice; the repair block does.
        clause = line.split("(fallback:", 1)[1]
        assert "apparmor" not in clause.lower() and "/etc" not in clause
        assert "FAIL: codex: sandbox" in out

        assert doctor.main(["--budget", "10", "--json"]) == 3
        report = json.loads(capsys.readouterr().out)
        sandbox = _row(report, "codex")["sandbox"]
        lost = sandbox["fallback_lost"]
        assert lost["rung"] == "landlock" and lost["outcome"] == "retired"
        assert lost["deprecated_at"][0] == "0.155.1"
        assert lost["removed_at"][0] == "0.156.1"
        assert "retires_when" not in lost
        self._repair_is_unchanged(
            "\n".join(doctor.render_text(report, warn_only=False)), report,
        )

    @pytest.mark.parametrize("walk, shown", [
        ((("landlock", "unprobed"),), "rung unprobed: the report budget was exhausted"),
        ((("landlock", "unknown"),), "a rung probe gave no answer"),
        ((("landlock", "unavailable"), ("unsandboxed", "taken")),
         "rung landlock is not usable here (unavailable)"),
    ])
    def test_a_timeout_or_other_failure_is_shown_as_itself(
        self, clinic, monkeypatch, capsys, walk, shown
    ):
        rung = "unsandboxed" if walk[-1] == ("unsandboxed", "taken") else None
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, rung, walk,
        ))
        assert doctor.main(["--budget", "10"]) == 3
        out = capsys.readouterr().out
        line = _codex_line(out)
        assert shown in line
        assert "measured unavailable from" not in line
        assert "profile bwrap /usr/bin/bwrap flags=(unconfined)" in out
        assert doctor.main(["--budget", "10", "--json"]) == 3
        assert json.loads(capsys.readouterr().out)["repair"] == doctor._repair_block()

    def test_a_pass_at_or_above_removed_at_is_flagged_not_hidden(
        self, clinic, monkeypatch, capsys
    ):
        """`removed_at` only skips the per-run probe; doctor still probes, so a
        build that restored the rung is a visible mismatch."""
        monkeypatch.setenv("MOCK_LLM_VERSION", "codex-cli 0.156.1")
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "landlock", (("landlock", "taken"),),
        ))
        assert doctor.main(["--budget", "10"]) == 3
        line = _codex_line(capsys.readouterr().out)
        assert "(fallback: auto → landlock — " in line
        assert (
            "mismatch: removed_at says unavailable from codex 0.156.1, but the "
            "probe passed on 0.156.1"
        ) in line
        assert doctor.main(["--budget", "10", "--json"]) == 3
        sandbox = _row(json.loads(capsys.readouterr().out), "codex")["sandbox"]
        assert sandbox["fallback_removed_mismatch"] is True

    def test_doctor_never_skips_the_rung_probe_by_version(self, monkeypatch):
        """Doctor hands `sandbox_verdict` no version, so the walk probes rung 1
        whatever `removed_at` says."""
        seen = []

        def spy(provider, profile, **kwargs):
            seen.append(kwargs)
            return executor.SandboxProbeResult("unavailable", BWRAP_BROKEN_LINE)

        monkeypatch.setattr(executor, "sandbox_verdict", spy)
        from utils.provider_registry import get_provider
        from utils.providers import codex as codex_module

        monkeypatch.setattr(
            codex_module, "sandbox_fallback_policy", lambda anchor=None: "auto"
        )
        doctor._probe_sandbox(get_provider("codex"), 5.0, None)
        assert seen and all(kwargs.get("version") is None for kwargs in seen)

    def test_a_rung_with_deprecated_at_but_no_removed_at(
        self, clinic, monkeypatch, capsys
    ):
        from dataclasses import replace

        from utils.providers import codex as codex_module

        kept = replace(codex_module.LANDLOCK_RUNG, removed_at=None)
        monkeypatch.setattr(
            codex_module.CodexProvider, "sandbox_fallback_rungs",
            lambda self, profile: (kept, codex_module.UNSANDBOXED_RUNG),
        )
        _fake_verdicts(monkeypatch, executor.SandboxProbeResult(
            "unavailable", BWRAP_BROKEN_LINE, "landlock", (("landlock", "taken"),),
        ))
        assert doctor.main(["--json", "--budget", "10"]) == 3
        sandbox = _row(json.loads(capsys.readouterr().out), "codex")["sandbox"]
        assert sandbox["fallback_deprecated_at"] == ["0.155.1", LANDLOCK_SENTENCE]
        assert "fallback_removed_at" not in sandbox
        assert "fallback_retires_when" not in sandbox


class TestDoctorDetectsAGlobalLandlockFlag:
    """Constraint 2: `use_legacy_landlock` enabled in codex's own config makes
    every workspace-write dispatch panic. Detected by a plain file read."""

    def _write(self, directory: Path, text: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "config.toml"
        target.write_text(text, encoding="utf-8")
        return target

    def _run(self, monkeypatch, capsys):
        import subprocess

        argvs = []
        real = subprocess.run

        def recording(argv, *args, **kwargs):
            argvs.append(list(argv))
            return real(argv, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", recording)
        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        doctor.main(["--budget", "10"])
        out = capsys.readouterr().out
        # Never `codex features enable` (nor any `features` subcommand): that
        # would set the very thing being looked for.
        assert not any("features" in argv for argv in argvs), argvs
        return out

    def test_codex_home_config_with_the_flag_warns_naming_the_file(
        self, clinic, monkeypatch, capsys
    ):
        target = self._write(
            Path(os.environ["CODEX_HOME"]),
            'model = "gpt-5"\n\n[features]\nuse_legacy_landlock = true\n',
        )
        out = self._run(monkeypatch, capsys)
        warn = next(ln for ln in out.splitlines() if "use_legacy_landlock" in ln)
        assert warn.startswith("WARN: codex: " + str(target))
        assert "agent-write" in warn and "panic" in warn

        assert doctor.main(["--json", "--budget", "10"]) in (0, 3)
        report = json.loads(capsys.readouterr().out)
        assert any(str(target) in w for w in report["warnings"])

    def test_home_dot_codex_is_the_fallback_location(
        self, clinic, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.delenv("CODEX_HOME", raising=False)
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        target = self._write(home / ".codex", "features.use_legacy_landlock = true\n")
        out = self._run(monkeypatch, capsys)
        assert "WARN: codex: " + str(target) in out

    @pytest.mark.parametrize("text", [
        None,  # missing file
        "[features]\nuse_legacy_landlock = false\n",
        "[features]\nother = true\n",
        "[features\nuse_legacy_landlock = = true\n",  # unparseable
    ])
    def test_silent_without_the_flag_missing_or_unparseable(
        self, clinic, monkeypatch, capsys, text
    ):
        if text is not None:
            self._write(Path(os.environ["CODEX_HOME"]), text)
        out = self._run(monkeypatch, capsys)
        assert "use_legacy_landlock" not in out
        assert "OK: every present provider passed its checks." in out

    @pytest.mark.parametrize("text,warns", [
        ('model = "gpt-5"\n\n[features]\nuse_legacy_landlock = true # on\n', True),
        ("features.use_legacy_landlock = true\n", True),
        ("features = { other = 1, use_legacy_landlock = true }\n", True),
        ("[features]\nuse_legacy_landlock = false\n", False),
        ("[profiles.x.features]\nuse_legacy_landlock = true\n", False),
        ("[features]\nx = 1\n[other]\nuse_legacy_landlock = true\n", False),
        ("# [features]\n# use_legacy_landlock = true\n", False),
    ])
    def test_still_detected_with_no_toml_parser(
        self, clinic, monkeypatch, capsys, text, warns
    ):
        """Python 3.10 under `uv run --no-dev` has neither `tomllib` nor
        `tomli` (the runtime dependency surface is pyyaml alone); the WARN
        must not go silent there."""
        monkeypatch.setattr(doctor, "_toml_module", lambda: None)
        target = self._write(Path(os.environ["CODEX_HOME"]), text)
        out = self._run(monkeypatch, capsys)
        assert ("WARN: codex: " + str(target) in out) is warns


# ---------------------------------------------------------------------------
# Plan §8 T8.6 / T8.13 — `config-only`, the reader's budget, and the scope
# ---------------------------------------------------------------------------

_OC_DEBUG = json.loads(
    (Path(__file__).parent / "fixtures" / "opencode" / "debug_config.1.18.32.json")
    .read_text(encoding="utf-8")
)["cases"]
_OC_AUTH = json.loads(
    (Path(__file__).parent / "fixtures" / "opencode" / "auth_list.1.18.32.json")
    .read_text(encoding="utf-8")
)["cases"]


class TestDoctorConfigOnly:
    """`doctor` reports discovery, never readiness: a declared provider is
    `config-only` with `ok: False`, never a clearance."""

    @pytest.fixture(autouse=True)
    def _opencode(self, tmp_path, monkeypatch):
        from utils.providers import opencode as oc

        self.oc = oc
        home = tmp_path / "user-config"
        (home / "opencode").mkdir(parents=True)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
        oc.reset_hermetic_state()
        self.provider = registry.get_provider("opencode")
        self.reads = []
        monkeypatch.setattr(
            doctor, "_run",
            lambda argv, **kw: (0, _OC_AUTH["empty"]["stdout"], _OC_AUTH["empty"]["stderr"]),
        )
        yield
        oc.reset_hermetic_state()

    def _serve(self, monkeypatch, case):
        import subprocess as _sp

        def spawn(argv, env, cwd, timeout):
            self.reads.append({"argv": list(argv), "env": env, "timeout": timeout})
            if case is None:
                return None
            return _sp.CompletedProcess(list(argv), 0, _OC_DEBUG[case]["stdout"], "")

        monkeypatch.setattr(self.oc, "_spawn_debug_config", spawn)

    def test_a_keyless_custom_provider_is_config_only_and_named(self, monkeypatch, tmp_path):
        self._serve(monkeypatch, "custom_keyless")
        row = doctor._probe_auth(self.provider, 5.0, str(tmp_path), None)
        assert row["state"] == "config-only"
        assert row["ok"] is False
        assert "keyless" in row["detail"]
        assert "not that every runner is cleared" in row["detail"]
        # The user-config read: no hermetic delta, no `--pure`.
        assert self.reads[0]["argv"] == ["opencode", "debug", "config"]
        assert self.reads[0]["env"] is None

    def test_user_config_only_vs_reachable_by_hermetic_validators(
        self, monkeypatch, tmp_path
    ):
        self._serve(monkeypatch, "custom_keyless")
        row = doctor._probe_auth(self.provider, 5.0, str(tmp_path), None)
        assert "user config only: keyless" in row["detail"]
        assert "hermetic validators refuse these" in row["detail"]
        assert doctor.AUTH_SCOPE_NOTE not in row["detail"]

        path = tmp_path / "cfg.yaml"
        path.write_text(
            "providers:\n  opencode:\n    config_providers:\n      keyless:\n"
            "        npm: '@ai-sdk/openai-compatible'\n", encoding="utf-8",
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
        registry._config = None
        registry._config_key = None
        self.oc.reset_hermetic_state()
        row = doctor._probe_auth(self.provider, 5.0, str(tmp_path), None)
        assert row["state"] == "config-only" and row["ok"] is False
        assert "reachable by hermetic validators: keyless" in row["detail"]
        assert "user config only" not in row["detail"]
        assert doctor.AUTH_SCOPE_NOTE not in row["detail"]

    def test_a_keyless_builtin_override_stays_not_logged_in(self, monkeypatch, tmp_path):
        self._serve(monkeypatch, "builtin_override_keyless")
        row = doctor._probe_auth(self.provider, 5.0, str(tmp_path), None)
        assert row["state"] == "not-logged-in"
        assert "opencode auth login" in row["detail"]
        assert doctor.AUTH_SCOPE_NOTE in row["detail"]

    def test_a_failed_read_stays_not_logged_in(self, monkeypatch, tmp_path):
        self._serve(monkeypatch, None)
        row = doctor._probe_auth(self.provider, 5.0, str(tmp_path), None)
        assert row["state"] == "not-logged-in"

    def test_the_detail_carries_no_key(self, monkeypatch, tmp_path):
        for case in _OC_DEBUG:
            self.oc.reset_hermetic_state()
            self._serve(monkeypatch, case)
            row = doctor._probe_auth(self.provider, 5.0, str(tmp_path), None)
            assert "sk-canary" not in json.dumps(row), case

    def test_the_reader_gets_what_is_left_of_the_budget(self, monkeypatch, tmp_path):
        self._serve(monkeypatch, "custom_keyless")
        doctor._probe_auth(self.provider, 2.0, str(tmp_path), None)
        assert 0 < self.reads[0]["timeout"] <= 2.0

    def test_an_exhausted_budget_spawns_no_reader(self, monkeypatch, tmp_path):
        import time as _time

        self._serve(monkeypatch, "custom_keyless")

        def slow_run(argv, **kw):
            _time.sleep(0.05)
            return 0, _OC_AUTH["empty"]["stdout"], ""

        monkeypatch.setattr(doctor, "_run", slow_run)
        row = doctor._probe_auth(self.provider, 0.01, str(tmp_path), None)
        assert row["state"] == "not-logged-in"
        assert self.reads == []

    def test_the_credential_leg_runs_under_the_validator_env(self, monkeypatch, tmp_path):
        seen = {}

        def run(argv, **kw):
            seen.update(kw)
            return 0, _OC_AUTH["stored"]["stdout"], ""

        monkeypatch.setattr(doctor, "_run", run)
        row = doctor._probe_auth(self.provider, 5.0, str(tmp_path), None)
        assert row["state"] == "ok"
        assert row["detail"] == doctor.AUTH_SCOPE_NOTE
        assert seen["cwd"] == str(tmp_path)
        # `resolve_env` always composes the git block, so the env is built.
        assert isinstance(seen["env"], dict)


@pytest.mark.debug_config
class TestDoctorStalledDebugConfig:
    """A `debug config` that hangs cannot blow `doctor`'s budget."""

    def test_a_sleeping_reader_returns_within_budget_as_not_logged_in(
        self, clinic, monkeypatch, capsys
    ):
        import time as _time

        from utils.providers import opencode as oc

        oc.reset_hermetic_state()
        monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
        auth = _OC_AUTH["empty"]["stdout"].replace("'", "'\\''")
        _script(clinic, "opencode", (
            'case "$1 $2" in\n'
            '  "--version ") echo 1.18.32 ;;\n'
            f"  \"auth list\") printf '%s' '{auth}' ;;\n"
            '  "debug config") exec sleep 5 ;;\n'
            'esac'
        ))
        xdg = clinic.parent / "oc-xdg"
        (xdg / "opencode").mkdir(parents=True)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        started = _time.monotonic()
        report = doctor.collect(budget_s=1.5)
        assert _time.monotonic() - started < 4.0
        row = _row(report, "opencode")
        assert row["present"] is True
        assert row["auth"]["state"] == "not-logged-in", row["auth"]
        oc.reset_hermetic_state()
