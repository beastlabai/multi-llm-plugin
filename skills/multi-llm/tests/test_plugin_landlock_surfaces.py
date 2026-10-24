#!/usr/bin/env python3
"""The plugin-landlock rung on its read-back surfaces: `doctor`, the HTML/markdown
report and `status` (plan codex-sandbox-unavailable-findings-and-fixes §6 F1
"Where it sits" items 3, 4 and 6; §6 F7; §7 "Step 2 subtask: messaging and
markers" items 3 and 5; §7 step 3, option (b); §8.2; 2026-09-30).

What is pinned here:

* `doctor` lists the rung with its probe result, its grade in words ("stops
  stray writes, not the model"), the codex version its allow-list was measured
  through beside the installed one (a newer installed codex flagged), the
  resolved base allow-list, and — in every state — the switch that skips it
  alone; while the switch is set the rung is shown skipped and no rung probe
  spawns; a bad switch value is reported, never a crash; the agent-write row
  names the write rung, its scopes (repo-wide included since step 3b), the open
  network and F5's clause, and the verdict and exit code do not move;
* where the rung probes ok under `auto` the AppArmor/sysctl repair is re-headed
  as a system-level alternative and printed in full; where it is absent (the
  wrapper's own exit 126) the heading is today's;
* no surface for the rung calls it "contained" or "sandboxed" (the policy value
  `sandbox_fallback: contained`, where named, is exempt);
* the report label and the `status` text for `fallback-plugin-landlock`, with
  `fallback-landlock` unchanged, and a `jail-denied` slot rendered as "denied by
  the plugin's jail: <path>" on both;
* `multi_llm.py` rejects a bad `MULTI_LLM_CODEX_SKIP_RUNG` at start-up (exit 3)
  while `doctor` reports it.

The doctor tests run the executor's real walk: `codex` is the mock CLI
(`MOCK_LLM_SANDBOX_BROKEN=1` replays the broken bwrap line) and the wrapper is a
fake under tmp_path (`landlock_dispatch.WRAPPER_PATHS`, plan §8.1) that logs
every invocation, so "no rung probe spawned" is observed rather than assumed.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import doctor  # noqa: E402
from utils import html_report_generator as report  # noqa: E402
from utils import landlock_dispatch  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import run_status  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers import codex as codex_module  # noqa: E402
from utils.providers.base import (  # noqa: E402
    SANDBOX_CANARY_DENIED,
    SKIP_RUNG_ENV,
)

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"
BWRAP_BROKEN_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
SWITCH = f"{SKIP_RUNG_ENV}=plugin-landlock"
MEASURED = codex_module.PLUGIN_LANDLOCK_RUNG.measured_through[0]
WRAPPER_126_NOTE = "landlock_exec: unavailable: Landlock ABI 0 found, the floor is 3"
INCIDENTAL_LABEL = "stray writes stopped by plugin-landlock; not a boundary against the model"


def _version_after(version: str) -> str:
    parts = [int(p) for p in version.split(".")]
    parts[-1] += 1
    return ".".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jailed(tmp_path, monkeypatch):
    """A PATH carrying the mock `codex` only, a broken bwrap, a codex home
    outside the probe's repo root, and a settable fake wrapper.

    Returns ``set_wrapper(passes) -> log path``: a fake `landlock_exec.py` that
    logs each invocation and either answers as a jail whose canary held (exit
    0 and the affirmative marker, the seeded file untouched) or as a kernel
    without Landlock (the wrapper's own exit 126 and its note)."""
    if os.name == "nt":
        pytest.skip(
            "symlinked shebang mock codex on a replaced PATH, which Windows can "
            "neither find nor run; Windows runs no plugin jail"
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "codex").symlink_to(MOCK_LLM)
    interpreter_dir = str(Path(sys.executable).parent)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(bin_dir), interpreter_dir, "/usr/bin", "/bin"])
    )
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
    monkeypatch.setenv("MOCK_LLM_VERSION", f"codex-cli {MEASURED}")
    monkeypatch.delenv("MULTI_LLM_PROVIDERS_CONFIG", raising=False)
    monkeypatch.delenv(SKIP_RUNG_ENV, raising=False)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    registry._config = None
    registry._config_key = None
    executor.reset_version_cache()
    assert shutil.which("codex") == str(bin_dir / "codex")
    monkeypatch.setattr(
        doctor, "_bwrap_self_test", lambda timeout: ("unavailable", BWRAP_BROKEN_LINE)
    )

    def no_model_calls(*args, **kwargs):
        raise AssertionError("doctor dispatched a model call")

    monkeypatch.setattr(executor, "_spawn", no_model_calls)
    from utils import llm_client

    monkeypatch.setattr(llm_client, "invoke_with_provider", no_model_calls)

    def set_wrapper(passes: bool) -> Path:
        log = tmp_path / "wrapper.log"
        target = tmp_path / "fake_landlock_exec.py"
        body = (
            "import sys\n"
            f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        )
        if passes:
            body += f"print({SANDBOX_CANARY_DENIED!r})\nsys.exit(0)\n"
        else:
            body += f"sys.stderr.write({WRAPPER_126_NOTE + chr(10)!r})\nsys.exit(126)\n"
        target.write_text(body, encoding="utf-8")
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setitem(landlock_dispatch.WRAPPER_PATHS, "landlock_exec", str(target))
        return log

    set_wrapper.codex_home = codex_home  # type: ignore[attr-defined]
    yield set_wrapper
    registry._config = None
    registry._config_key = None


def _run(capsys, *argv):
    code = doctor.main(["--budget", "10", *argv])
    return code, capsys.readouterr().out


def _codex_line(out):
    return next(ln for ln in out.splitlines() if ln.strip().startswith("codex"))


def _sub_row(out, head):
    return next(ln for ln in out.splitlines() if ln.startswith(f"    {head}"))


def _spawned(log: Path) -> str:
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _slot_dirs():
    base = tempfile.gettempdir()
    return glob.glob(os.path.join(base, f"multi-llm-slot-{os.getpid()}-*"))


_POLICY_VALUE = re.compile(r"`?sandbox_fallback[:=] ?`?contained`?|fallback: contained")


def _calls_it_contained_or_sandboxed(text: str) -> bool:
    """`contained`/`sandboxed` as words, after the exempt policy-value mentions."""
    return bool(re.search(r"\b(contained|sandboxed)\b", _POLICY_VALUE.sub("", text)))


# ---------------------------------------------------------------------------
# doctor: the rung probes ok
# ---------------------------------------------------------------------------


class TestDoctorWithAPassingJail:

    def test_the_rung_row_the_heading_and_an_unchanged_verdict(self, jailed, capsys):
        """F1 item 6 / F7 / §7 step 2: the rung is probed through the real walk
        (the fake wrapper ran), its row says ok with the grade, both versions,
        the base allow-list and the switch; the repair is re-headed as a
        system-level alternative and printed in full; BROKEN and exit 3 stay."""
        log = jailed(passes=True)
        before = set(_slot_dirs())
        code, out = _run(capsys)
        assert code == 3
        assert _spawned(log), "the rung probe never reached the wrapper"
        # The probe's scratch dirs (the walk's and the allow-list's) are gone.
        assert set(_slot_dirs()) <= before

        line = _codex_line(out)
        assert "sandbox BROKEN: " + BWRAP_BROKEN_LINE in line
        assert (
            " (fallback: auto → plugin-landlock — codex runs read-only phases on "
            "the plugin's Landlock jail, which stops stray writes, not the model"
        ) in line

        row = _sub_row(out, "plugin-landlock:")
        assert row.startswith("    plugin-landlock: probe ok — stops stray writes, not the model; ")
        assert f"measured through codex {MEASURED}, installed codex {MEASURED}" in row
        assert "newer than the jail" not in row
        home = os.path.realpath(str(jailed.codex_home))
        assert f"base allow-list: dir:{landlock_dispatch.render_home(home)}, dir:" in row
        assert "multi-llm-slot-<pid>-…" in row and "file:/dev/null" in row
        assert row.endswith(f"; {SWITCH} skips this rung alone")

        write_row = _sub_row(out, "agent-write:")
        assert write_row.startswith(
            "    agent-write: runs on plugin-landlock (probe ok) — plugin-landlock "
            "serves the plan-file, tasks and repo-wide scopes (apply-suggestions, "
            "apply-task-suggestions, apply-code-fixes, implement)"
        )
        for clause in (
            "it stops stray writes, not the model",
            "network access is not confined on this rung, unlike codex's own workspace-write",
            "repo-wide edits (apply-code-fixes, implement) run on it too, with "
            "the whole repository writable — .git/, .codex/, .multi-llm/ and the "
            "run's output folder included; the model is told to leave them alone "
            "and the batch is checked afterwards, nothing is backed up (root "
            "grant, 2026-10-05)",
            "a chat in danger-full-access posture may run writes unsandboxed only when this rung fails",
        ):
            assert clause in write_row, clause
        assert write_row.endswith(
            "the row stays BROKEN (exit 3 without --warn-only): codex's own sandbox cannot start"
        )

        assert "FAIL: codex: sandbox" in out
        assert doctor.REPAIR_HEADING_SYSTEM_ALTERNATIVE in out
        assert doctor.REPAIR_HEADING not in out
        for text in doctor.APPARMOR_PROFILE.rstrip("\n").splitlines():
            assert text in out, text
        assert doctor.APPARMOR_LOAD_COMMAND in out
        assert doctor.SYSCTL_ALTERNATIVE in out and "not recommended" in out

    def test_the_json_carries_the_same_facts_and_the_same_repair(self, jailed, capsys):
        jailed(passes=True)
        code, out = _run(capsys, "--json")
        assert code == 3
        data = json.loads(out)
        sandbox = next(r for r in data["providers"] if r["provider"] == "codex")["sandbox"]
        assert sandbox["state"] == "unavailable" and sandbox["ok"] is False
        assert sandbox["fallback_rung"] == "plugin-landlock"
        assert sandbox["fallback_contained"] is False
        assert sandbox["fallback_containment"] == "incidental"
        assert ["plugin-landlock", "taken"] in sandbox["fallback_walk"]
        rung = sandbox["plugin_landlock"]
        assert rung["outcome"] == "taken"
        assert rung["measured_through"] == MEASURED
        assert rung["installed"] == MEASURED
        assert rung["newer_than_measured"] is False
        assert rung["switch"] == SWITCH and rung["switch_set"] is False
        assert rung["allow_list"] and rung["allow_list_error"] is None
        write = sandbox["agent_write"]
        assert write["status"] == doctor.WRITE_RUNS_ON_RUNG
        assert write["scopes"] == ["plan-file", "tasks", "repo-wide"]
        assert write["refused_scopes"] == []
        # The repair's text is unchanged; only its heading moves.
        assert data["repair"] == doctor._repair_block()
        assert data["repair_is_system_alternative"] is True
        assert data["repair_heading"] == doctor.REPAIR_HEADING_SYSTEM_ALTERNATIVE
        assert any(f.startswith("codex: sandbox") for f in data["failures"])
        assert data["exit"] == 3

    def test_warn_only_moves_the_exit_and_nothing_else(self, jailed, capsys):
        jailed(passes=True)
        code, out = _run(capsys, "--warn-only")
        assert code == 0
        assert "WARN: codex: sandbox" in out
        assert doctor.REPAIR_HEADING_SYSTEM_ALTERNATIVE in out

    def test_a_newer_installed_codex_is_flagged(self, jailed, capsys, monkeypatch):
        """F1 item 3: the measured-through version rides the row beside the
        installed one, like `removed_at`'s evidence; a newer codex is named as
        the likely cause of a jail denial, never refused."""
        newer = _version_after(MEASURED)
        monkeypatch.setenv("MOCK_LLM_VERSION", f"codex-cli {newer}")
        jailed(passes=True)
        code, out = _run(capsys)
        assert code == 3
        row = _sub_row(out, "plugin-landlock:")
        assert f"measured through codex {MEASURED}, installed codex {newer}" in row
        assert (
            "(codex is newer than the jail was measured with, the likely cause of "
            "a jail denial)"
        ) in row
        assert row.startswith("    plugin-landlock: probe ok")

    def test_no_surface_calls_the_rung_contained_or_sandboxed(self, jailed, capsys):
        """§7 step 2 subtask item 5: the doctor line's fallback clause, the rung
        row, the agent-write row, the report label and the `status` text."""
        jailed(passes=True)
        _, out = _run(capsys)
        clause = _codex_line(out).split("(fallback:", 1)[1]
        surfaces = {
            "fallback clause": clause,
            "rung row": _sub_row(out, "plugin-landlock:"),
            "agent-write row": _sub_row(out, "agent-write:"),
            "report label": report.sandbox_fallback_label("fallback-plugin-landlock"),
            "served label": report.served_model_label(
                "codex:gpt-5.5", {"codex:gpt-5.5": {"sandbox": "fallback-plugin-landlock"}}
            ),
            "status text": run_status._progress_text(
                {"sandbox_fallback": {"codex:gpt-5.5": "plugin-landlock"}}
            ),
        }
        for name, text in surfaces.items():
            assert not _calls_it_contained_or_sandboxed(text), (name, text)
            assert "stray writes" in text or name == "fallback clause" or "stray-writes" in text, (name, text)
        assert "stray writes, not the model" in clause


# ---------------------------------------------------------------------------
# doctor: the rung is absent, skipped, or the switch is bad
# ---------------------------------------------------------------------------


class TestDoctorWithAnUnusableJail:

    def test_a_landlock_less_host_keeps_todays_heading(self, jailed, capsys):
        """The wrapper's own exit 126 (fake): the rung is absent, the walk falls
        through to rung 3, the row names the wrapper's note, and the repair
        keeps today's heading — on a host without Landlock the system-level
        repair is the only sandbox left (§7, "Step 4", the doctor half)."""
        log = jailed(passes=False)
        code, out = _run(capsys)
        assert code == 3
        assert _spawned(log)
        line = _codex_line(out)
        assert "(fallback: auto — codex runs unsandboxed for read-only phases" in line
        assert "rung plugin-landlock is not usable here (absent)" in line
        row = _sub_row(out, "plugin-landlock:")
        assert row.startswith(
            "    plugin-landlock: absent: the plugin's Landlock jail cannot run here"
        )
        assert WRAPPER_126_NOTE in row
        assert f"measured through codex {MEASURED}, installed codex {MEASURED}" in row
        assert row.endswith(f"; {SWITCH} skips this rung alone")
        write_row = _sub_row(out, "agent-write:")
        assert write_row.startswith(
            "    agent-write: refused — plugin-landlock serves the plan-file, tasks "
            "and repo-wide scopes (apply-suggestions, apply-task-suggestions, "
            "apply-code-fixes, implement), and its probe here says absent"
        )
        assert "a chat in danger-full-access posture may run writes unsandboxed only when this rung fails" in write_row
        assert doctor.REPAIR_HEADING in out
        assert doctor.REPAIR_HEADING_SYSTEM_ALTERNATIVE not in out
        assert not _calls_it_contained_or_sandboxed(row)

        code, out = _run(capsys, "--json")
        data = json.loads(out)
        assert data["repair_is_system_alternative"] is False
        assert data["repair"] == doctor._repair_block()

    def test_the_switch_skips_the_rung_and_spawns_no_probe(self, jailed, capsys, monkeypatch):
        """F1 item 6 "Run the job anyway": with the switch set the walk records
        the rung skipped without a probe — the fake wrapper that would have
        passed never runs — read phases fall through to rung 3, writes are
        refused, and the repair keeps today's heading."""
        log = jailed(passes=True)
        monkeypatch.setenv(SKIP_RUNG_ENV, "plugin-landlock")
        code, out = _run(capsys)
        assert code == 3
        assert _spawned(log) == ""
        line = _codex_line(out)
        assert "(fallback: auto — codex runs unsandboxed for read-only phases" in line
        assert f"rung plugin-landlock skipped by {SWITCH}" in line
        row = _sub_row(out, "plugin-landlock:")
        assert row.startswith(
            f"    plugin-landlock: skipped: {SWITCH} is set, so no probe ran"
        )
        assert row.endswith(
            f"; {SWITCH} is set: this rung is skipped for this process (unset it to walk it)"
        )
        assert "measured through codex" in row and "base allow-list: " in row
        write_row = _sub_row(out, "agent-write:")
        assert write_row.startswith("    agent-write: refused — ")
        assert f"skipped: {SWITCH} is set" in write_row
        assert doctor.REPAIR_HEADING in out
        assert doctor.REPAIR_HEADING_SYSTEM_ALTERNATIVE not in out

        code, out = _run(capsys, "--json")
        sandbox = next(r for r in json.loads(out)["providers"] if r["provider"] == "codex")["sandbox"]
        assert ["plugin-landlock", "skipped-by-switch"] in sandbox["fallback_walk"]
        assert sandbox["plugin_landlock"]["switch_set"] is True
        assert sandbox["plugin_landlock"]["outcome"] == "skipped-by-switch"
        assert _spawned(log) == ""

    def test_a_bad_switch_value_is_reported_never_a_crash(self, jailed, capsys, monkeypatch):
        """Every other mode refuses to start on it (`multi_llm.py`); doctor is
        where a user looks, so the row names the value and the accepted one."""
        jailed(passes=True)
        monkeypatch.setenv(SKIP_RUNG_ENV, "bogus")
        code, out = _run(capsys)
        assert code == 3
        row = _sub_row(out, "plugin-landlock:")
        assert f"{SKIP_RUNG_ENV}='bogus' is not recognised" in row
        assert "every other mode refuses to start until it is unset or set to plugin-landlock" in row
        assert "FAIL: codex: sandbox" in out

    def test_the_row_names_the_switch_in_every_state(self, jailed, capsys, monkeypatch):
        """F1 item 4: the switch is on the rung's row whatever the probe said —
        ok, absent, skipped — so a user whose turns fail finds it first."""
        for passes, switch in ((True, None), (False, None), (True, "plugin-landlock")):
            jailed(passes=passes)
            if switch:
                monkeypatch.setenv(SKIP_RUNG_ENV, switch)
            else:
                monkeypatch.delenv(SKIP_RUNG_ENV, raising=False)
            _, out = _run(capsys)
            assert SWITCH in _sub_row(out, "plugin-landlock:"), (passes, switch)


# ---------------------------------------------------------------------------
# multi_llm.py: a bad switch value is rejected at start-up
# ---------------------------------------------------------------------------


class TestTheDispatcherRejectsABadSwitch:

    def test_any_other_value_is_exit_3_naming_the_variable(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(SKIP_RUNG_ENV, "bogus")
        monkeypatch.chdir(tmp_path)
        assert multi_llm.main(["status", str(tmp_path / "plan.md")]) == 3
        captured = capsys.readouterr()
        assert f"{SKIP_RUNG_ENV}='bogus' is not recognised" in captured.err
        assert "'plugin-landlock'" in captured.err
        assert '"exit": 3' in captured.out

    def test_the_accepted_value_and_unset_pass_the_gate(self, monkeypatch):
        for value in (None, "", "plugin-landlock"):
            if value is None:
                monkeypatch.delenv(SKIP_RUNG_ENV, raising=False)
            else:
                monkeypatch.setenv(SKIP_RUNG_ENV, value)
            multi_llm._validate_skip_rung("status")

    def test_doctor_reports_rather_than_dies(self, jailed, capsys, monkeypatch):
        jailed(passes=True)
        monkeypatch.setenv(SKIP_RUNG_ENV, "bogus")
        assert multi_llm.main(["doctor", "--warn-only", "--budget", "10"]) == 0
        out = capsys.readouterr().out
        assert f"{SKIP_RUNG_ENV}='bogus' is not recognised" in out
        assert '"mode": "doctor"' in out


# ---------------------------------------------------------------------------
# The report and `status`
# ---------------------------------------------------------------------------

SPEC = "codex:gpt-5.5"


class TestTheReportLabel:

    def test_the_incidental_word_names_its_grade(self):
        assert report.sandbox_fallback_label("fallback-plugin-landlock") == INCIDENTAL_LABEL
        label = report.served_model_label(SPEC, {SPEC: {"sandbox": "fallback-plugin-landlock"}})
        assert label == f"{SPEC} ({INCIDENTAL_LABEL})"
        assert "unsandboxed" not in label and report.SANDBOX_BYPASSED_LABEL not in label

    def test_fallback_landlock_renders_as_today(self):
        assert report.sandbox_fallback_label("fallback-landlock") == "sandboxed by fallback rung: landlock"
        assert report.served_model_label(SPEC, {SPEC: {"sandbox": "fallback-landlock"}}) == (
            f"{SPEC} (sandboxed by fallback rung: landlock)"
        )

    def test_a_jail_denied_slot_names_the_path(self):
        failed = {SPEC: {
            "failure_reason": "jail-denied", "jail_denied_path": "/home/u/.cache/codex/x",
            "error": "codex exited 1: Permission denied (os error 13)",
        }}
        line = report.failed_models_line(failed, 3)
        assert line == (
            f"**Models failed:** 1 of 3 requested — {SPEC}: denied by the plugin's "
            f"jail: /home/u/.cache/codex/x"
        )
        notices = report.build_report_notices([], failed, 3)
        assert notices[0]["message"].endswith("denied by the plugin's jail: /home/u/.cache/codex/x")
        unnamed = {SPEC: {"failure_reason": "jail-denied"}}
        assert report.failed_models_line(unnamed).endswith(
            "denied by the plugin's jail: path not named"
        )
        # Every other reason renders as before, string or record.
        assert report.failed_models_line({SPEC: "timed out after 60s"}, 1) == (
            f"**Models failed:** 1 of 1 requested — {SPEC}: timed out after 60s"
        )
        assert report.failed_models_line({SPEC: {"error": "boom"}}).endswith(f"{SPEC}: boom")

    def test_the_models_served_record_names_the_denial_over_the_error_string(self):
        """The fan-out keeps `failed_models` a string map and puts
        `failure_reason`/`jail_denied_path` on the slot's `models_served` entry
        (`review_orchestrator_base.jail_record_fields`); the line reads the
        record, so the words survive `--reaggregate` and outlive the log."""
        failed = {SPEC: "codex exited 1: Permission denied (os error 13)", "pi:x": "timed out"}
        served = {SPEC: {
            "sandbox": "fallback-plugin-landlock", "failure_reason": "jail-denied",
            "jail_denied_path": "~/.cache/codex/new",
        }}
        line = report.failed_models_line(failed, 4, served_models=served)
        assert line == (
            f"**Models failed:** 2 of 4 requested — {SPEC}: denied by the plugin's "
            f"jail: ~/.cache/codex/new; codex exited 1: Permission denied (os error 13); "
            f"pi:x: timed out"
        )
        notice = report.build_report_notices([], failed, 4, served_models=served)[0]
        assert notice["message"] == line.replace("**", "")
        # Without the record, the reason string renders as before.
        assert report.failed_models_line(failed, 4).startswith(
            f"**Models failed:** 2 of 4 requested — {SPEC}: codex exited 1"
        )

    def test_the_jail_label_never_hides_the_recorded_reason(self):
        """The label is prepended, not substituted: a later check violation
        or code-review's spend clause stays on the line, and a reason that
        already names the denial is not repeated."""
        served = {SPEC: {"failure_reason": "jail-denied", "jail_denied_path": "/x/y"}}
        label = "denied by the plugin's jail: /x/y"
        later = "[CODEX_HOME_CHECK] violation: config.toml changed (spent $0.12)"
        line = report.failed_models_line({SPEC: later}, served_models=served)
        assert line.endswith(f"{SPEC}: {label}; {later}")
        carried = f"{label} (fallback rung 'plugin-landlock'; see the log) (spent $0.40)"
        line = report.failed_models_line({SPEC: carried}, served_models=served)
        assert line.endswith(f"{SPEC}: {carried}")
        assert line.count(label) == 1
        assert report.failed_models_line({SPEC: ""}, served_models=served).endswith(
            f"{SPEC}: {label}"
        )

    def test_the_served_entry_carries_the_denial_beside_the_rung(self):
        entry = {
            "sandbox": "fallback-plugin-landlock", "failure_reason": "jail-denied",
            "jail_denied_path": "/tmp/x",
        }
        assert report.served_model_label(SPEC, {SPEC: entry}) == (
            f"{SPEC} ({INCIDENTAL_LABEL}; denied by the plugin's jail: /tmp/x)"
        )


class TestTheStatusText:

    def test_the_incidental_rung_is_stray_writes_only(self):
        text = run_status._progress_text({"sandbox_fallback": {
            SPEC: "plugin-landlock", "codex:other": "landlock",
        }})
        assert f"stray-writes-only plugin-landlock {SPEC}" in text
        assert "sandbox-fallback plugin-landlock" not in text
        # The contained rung renders as today.
        assert "sandbox-fallback landlock codex:other" in text
        assert "unsandboxed" not in text

    def test_a_jail_denied_row_names_the_path(self):
        entry = {
            "state": "failed", "failure_reason": "jail-denied",
            "jail_denied_path": "/home/u/.config/x", "error": "codex exited 1",
        }
        details = run_status._model_detail_text(entry)
        assert "denied by the plugin's jail: /home/u/.config/x" in details
        assert "codex exited 1" not in details
        assert "denied by the plugin's jail: path not named" in run_status._model_detail_text(
            {"state": "failed", "failure_reason": "jail-denied"}
        )
        # Any other failure keeps its own error text.
        assert "codex exited 1" in run_status._model_detail_text(
            {"state": "failed", "error": "codex exited 1"}
        )

    def test_a_jail_denied_slot_that_finished_during_a_wait(self):
        views = [{
            "phase_dir": "p",
            "progress": {"models": {SPEC: {
                "state": "failed", "ended_at": "2026-09-30T00:00:10+00:00",
                "elapsed_s": 10, "failure_reason": "jail-denied",
                "jail_denied_path": "/x/y",
            }}},
        }]
        run_status._attach_finished_during_wait({"p": {}}, views)
        finished = views[0]["finished_during_wait"]
        assert finished[0]["error"] == "denied by the plugin's jail: /x/y"
        assert "denied by the plugin's jail: /x/y" in run_status._just_finished_text(views[0])
