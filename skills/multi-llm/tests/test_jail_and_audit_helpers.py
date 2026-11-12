#!/usr/bin/env python3
"""The helpers both dispatch lanes share, pinned by their raw output.

The executor (``dispatch_spawn``) and the fan-out (``llm_client``, the review
lanes) used to keep private copies of these, and the copies drifted. Each now
has one home: the ``[JAIL_DENIED]`` line and the denial count
(``landlock_dispatch.jail_denied_text`` / ``jail_denials_text``, over one
``JailDenials``), ``$HOME`` rendering (``render_home_text``), the
"newer than measured" test (``providers.base.version_newer``) and the audit
argv (``dispatch_spawn.render_audit_argv``). Each lane keeps its own wording:
the strings below are each lane's whole line, byte for byte, so a change to
the shared code that moves one lane's text fails here, not in a user's log.
"""

import json
import os
import sys
import types
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import dispatch_spawn, landlock_dispatch, probe_state, task_batch  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils.provider_registry import get_provider  # noqa: E402
from utils.providers.base import (  # noqa: E402
    INCIDENTAL_LANE_EXECUTOR,
    INCIDENTAL_LANE_FANOUT,
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    JailRequest,
    WritePostureConsent,
    version_newer,
)
from utils.providers.codex import PLUGIN_LANDLOCK_RUNG  # noqa: E402

#: The plugin-landlock rung with a fixed measured-through version, so the pins
#: below do not move when the real rung is re-measured.
RUNG = types.SimpleNamespace(name="plugin-landlock", measured_through=("0.163.0", ""))
SPEC = "codex:gpt-5.5"
NEXT = (
    "Next: compare with `multi_llm.py doctor`'s `plugin-landlock` row; report it "
    "at github.com/beastlabai/multi-llm-plugin/issues with this line and that row; "
)
FANOUT_NEXT = (
    "next steps: (1) compare with `multi_llm.py doctor` (its plugin-landlock row "
    "lists the allow-list, both versions and the switch); (2) report it at "
    "github.com/beastlabai/multi-llm-plugin/issues with this line and that row (the "
    "allow-list is not configurable); (3) to run the job anyway, set this key to a "
    "runner that is not Codex (for example claude-code:<model>) in "
    ".multi-llm/providers.yaml, or for one run set "
    "MULTI_LLM_CODEX_SKIP_RUNG=plugin-landlock"
)


def _executor(evidence, *, allow_list="dir:~/.codex", installed="unknown", **lane):
    return landlock_dispatch.jail_denied_text(
        spec=SPEC, rung=RUNG, evidence=evidence, allow_list=allow_list,
        installed=installed, lane=INCIDENTAL_LANE_EXECUTOR, **lane,
    )


def _fanout(evidence, *, allow_list="dir:~/.codex", installed=""):
    return landlock_dispatch.jail_denied_text(
        spec=SPEC, rung=RUNG, evidence=evidence, allow_list=allow_list,
        installed=installed, lane=INCIDENTAL_LANE_FANOUT,
    )


class TestTheExecutorJailDeniedLine:
    def test_a_read_dispatch_with_a_named_path(self):
        assert _executor(
            {"path": "/srv/x", "line": "open: Permission denied (os error 13)"},
            profile=PROFILE_AGENT_READONLY,
        ) == (
            "[JAIL_DENIED] codex:gpt-5.5 on rung 'plugin-landlock': denied /srv/x; "
            "excerpt: 'open: Permission denied (os error 13)'; allow-list: "
            "dir:~/.codex; codex unknown installed, the jail measured through "
            "0.163.0. This is a jail failure, not a model error, and is not retried "
            "on another rung. " + NEXT + "to run the job anyway, set "
            "`driver.subagent_runner` to a runner that is not Codex (F6), or else set "
            "MULTI_LLM_CODEX_SKIP_RUNG=plugin-landlock for one run."
        )

    def test_a_write_dispatch_with_no_path_on_a_newer_codex(self):
        assert _executor(
            {"path": None, "line": "Permission denied"}, allow_list="",
            installed="codex-cli 0.199.0", profile=PROFILE_AGENT_WRITE,
            write_scope="plan-file",
        ) == (
            "[JAIL_DENIED] codex:gpt-5.5 on rung 'plugin-landlock': denied path not "
            "named (Codex did not name the path; a probable jail denial); excerpt: "
            "'Permission denied'; allow-list: (not computed); codex codex-cli 0.199.0 "
            "installed, the jail measured through 0.163.0; Codex is newer than the "
            "jail was measured with, the likely cause. This is a jail failure, not a "
            "model error, and is not retried on another rung. " + NEXT + "to run the "
            "job anyway, set `driver.subagent_runner` to a runner that is not Codex "
            "(F6). Setting MULTI_LLM_CODEX_SKIP_RUNG=plugin-landlock does not run it: "
            "this is a write dispatch (scope plan-file) and 'plugin-landlock' is its "
            "only write rung, so with it skipped the write is refused unless the chat "
            "itself runs in danger-full-access (MULTI_LLM_HOST_SANDBOX_POSTURE; F5, "
            "under sandbox_fallback=auto), which then runs it with no OS boundary."
        )

    def test_an_unnamed_path_reads_as_probable_whatever_the_class(self):
        """The executor's wording for a wrapper or setup refusal that names no
        path, kept as it was (the fan-out names the class instead)."""
        line = _executor({"path": None, "line": "x", "class": "jail"})
        assert "denied path not named (Codex did not name the path; a probable" in line

    def test_the_quote_is_redacted_before_its_300_character_cut(self):
        key = "sk-proj-" + "A1b2C3d4" * 6
        line = _executor({"path": None, "line": "z" * 290 + f" token {key} tail"})
        assert "A1b2" not in line and "sk-proj" not in line
        excerpt = line.split("excerpt: '", 1)[1].split("'; allow-list", 1)[0]
        assert len(excerpt) == landlock_dispatch.EXCERPT_CHARS


class TestTheFanoutJailDeniedLine:
    def test_a_named_path(self):
        assert _fanout({
            "where": "stderr", "path": "/srv/x", "class": "jail",
            "line": "open: Permission denied (os error 13)",
        }) == (
            "[JAIL_DENIED] codex (codex:gpt-5.5): the plugin's Landlock jail (fallback "
            "rung 'plugin-landlock') denied a write codex itself needed, so the turn "
            "failed: /srv/x; stderr: \"open: Permission denied (os error 13)\"; allow "
            "list: dir:~/.codex; codex installed unknown, jail measured through "
            "0.163.0; " + FANOUT_NEXT
        )

    def test_a_probable_denial_on_a_newer_codex(self):
        assert _fanout(
            {"where": "event", "path": None, "class": "probable", "line": "Permission denied"},
            allow_list="", installed="codex-cli 0.199.0",
        ) == (
            "[JAIL_DENIED] codex (codex:gpt-5.5): the plugin's Landlock jail (fallback "
            "rung 'plugin-landlock') denied a write codex itself needed, so the turn "
            "failed: path not named (Codex did not name the path; a probable jail "
            "denial); event: \"Permission denied\"; allow list: (unavailable); codex "
            "installed codex-cli 0.199.0, jail measured through 0.163.0; Codex is "
            "newer than the jail was measured with, the likely cause; " + FANOUT_NEXT
        )

    def test_an_unnamed_jail_denial_and_no_where(self):
        line = _fanout({"where": None, "path": None, "class": "jail", "line": "Permission denied"})
        assert "so the turn failed: path not named; stderr: \"Permission denied\";" in line

    def test_the_real_rung_and_an_older_codex_name_no_cause(self):
        measured = PLUGIN_LANDLOCK_RUNG.measured_through[0]
        line = landlock_dispatch.jail_denied_text(
            spec=SPEC, rung=PLUGIN_LANDLOCK_RUNG, evidence={"line": "x"}, allow_list="",
            installed="codex-cli 0.1.0", lane=INCIDENTAL_LANE_FANOUT,
        )
        assert f"jail measured through {measured}; next steps:" in line


class TestHomeRendering:
    """``$HOME`` shows as ``~`` where it starts a path or is quoted whole, on
    both lanes (the fan-out's copy skipped the quoted case)."""

    def test_both_shapes(self):
        home = os.path.expanduser("~")
        if home in ("~", os.sep):
            pytest.skip("no home to render")
        text = f"x {home}{os.sep}a '{home}' {home}x"
        assert landlock_dispatch.render_home_text(text) == (
            f"x ~{os.sep}a '~' {home}x"
        )
        assert landlock_dispatch.render_home_text("") == ""
        assert landlock_dispatch.render_home_text(None) == ""

    def test_a_quoted_path_that_ends_in_home_keeps_its_head(self):
        """Only a whole quoted ``'$HOME'`` becomes ``'~'``; a path quoted
        whole that merely ends in ``$HOME`` is another folder."""
        home = os.path.expanduser("~")
        if home in ("~", os.sep):
            pytest.skip("no home to render")
        other = f"{os.sep}mnt{os.sep}backup{home}"
        text = f"open '{other}' failed: Permission denied; open '{home}' failed"
        assert landlock_dispatch.render_home_text(text) == (
            f"open '{other}' failed: Permission denied; open '~' failed"
        )

    @pytest.mark.parametrize("raw, shown", [
        # SP-6: a home that ends a token it does not start renders as before.
        ("failed: 'HOME=/home/u'", "failed: 'HOME=~'"),
        ("error writing '--cd=/home/u': EACCES", "error writing '--cd=~': EACCES"),
        ("PATH=/usr/bin:/home/u/bin", "PATH=/usr/bin:~/bin"),
        ('open("/home/u") failed', 'open("~") failed'),
        ("EACCES /home/u: denied", "EACCES ~: denied"),
        ("cd /home/u/.codex/x", "cd ~/.codex/x"),
        # A path that merely contains the home is another folder.
        ("EACCES /mnt/backup/home/u/.codex/x", "EACCES /mnt/backup/home/u/.codex/x"),
        ("open '/mnt/backup/home/u' failed", "open '/mnt/backup/home/u' failed"),
        ("ls //home/u", "ls //home/u"),
        # So is one whose name only starts with it.
        ("/home/u2/x /home/u.bak /home/u-old", "/home/u2/x /home/u.bak /home/u-old"),
    ])
    def test_the_home_renders_only_where_it_starts_a_path(self, monkeypatch, raw, shown):
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.setenv("USERPROFILE", "/home/u")  # expanduser's source on Windows
        assert landlock_dispatch.render_home_text(raw) == shown

    @pytest.mark.parametrize("escape", ["\n", "\t", "\r", '"'])
    def test_an_escaped_boundary_in_a_cli_event_hides_the_home(self, monkeypatch, escape):
        """Codex review 5 (F03): a CLI's JSON event quotes its tool output
        escaped, so the character before a path can be the ``n`` of ``\\n``.
        That escape is a boundary like a real one, on the whole line as the
        user sees it (``jail_denied_text``), while a backup copy of the home
        inside the same escaped text keeps its head."""
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.setenv("USERPROFILE", "/home/u")  # expanduser's source on Windows
        hidden = json.dumps({"error": f"Permission denied{escape}/home/u/.cache/item"})
        backup = json.dumps({"error": f"Permission denied{escape}/mnt/backup/home/u/x"})
        for lane in (_executor, _fanout):
            line = lane({"where": "stderr", "class": "probable", "line": hidden})
            assert "/home/u/.cache/item" not in line, line
            assert "~/.cache/item" in line, line
            kept = lane({"where": "stderr", "class": "probable", "line": backup})
            assert "/mnt/backup/home/u/x" in kept, kept

    def test_the_fanout_line_renders_a_quoted_home(self):
        home = os.path.expanduser("~")
        if home in ("~", os.sep):
            pytest.skip("no home to render")
        line = _fanout({"where": "stderr", "class": "probable",
                        "line": f"open '{home}' failed: Permission denied"})
        assert "stderr: \"open '~' failed: Permission denied\";" in line


@pytest.mark.parametrize("installed, measured, newer", [
    ("codex-cli 0.170.0", "0.163.0", True),
    ("0.163.0", "0.163.0", False),
    ("0.163.1", "0.163", True),
    ("", "0.163.0", False),
    ("unknown", "0.163.0", False),
    ("9", "0.163.0", False),
    ("0.170.0", "", False),
    ("0.1.0", "unknown", True),
    ("0.170.0", None, False),
])
def test_version_newer(installed, measured, newer):
    assert version_newer(installed, measured) is newer


# ---------------------------------------------------------------------------
# The denial count: one JailDenials, each lane's words
# ---------------------------------------------------------------------------

DENIED = landlock_dispatch.JailDenials(
    failed_file_changes=[["/f/a", "/f/b"]], out_of_scope_count=4,
    out_of_scope_paths=["/f/a", "/o/c", "/o/d"],
)


def _fanout_record(denials):
    """``details["jail_denials"]`` as ``llm_client`` records it."""
    record = denials.to_dict()
    record["total"] = denials.total
    record["paths"] = denials.named_paths()
    return record


class TestTheDenialCount:
    def test_the_executor_line(self, capsys):
        dispatch_spawn._report_jail_outcome(
            "T001", types.SimpleNamespace(jail_denied_line=None, jail_denials=DENIED),
        )
        assert capsys.readouterr().out == (
            "[exec-subagents] T001: the plugin's Landlock jail stopped 4 command "
            "write(s) outside the grant denied (/f/a, /o/c, /o/d); 1 failed "
            "file_change item(s) — stray writes stopped by the jail\n"
        )

    def test_the_fanout_line_from_the_recorded_dict(self, capsys):
        base.report_jail_outcome(SPEC, {"jail_denials": _fanout_record(DENIED)}, "u37c-a")
        assert capsys.readouterr().out == (
            "[codex:gpt-5.5] the plugin's Landlock jail denied 5 writes inside the "
            "model's own commands (not a failure): /f/a, /f/b, /o/c\n"
        )

    def test_one_write_is_singular_and_nothing_prints_nothing(self, capsys):
        one = landlock_dispatch.JailDenials(in_scope_paths=["/in"])
        base.report_jail_outcome("x", {"jail_denials": _fanout_record(one)}, "u37c-b")
        dispatch_spawn._report_jail_outcome(
            "T002", types.SimpleNamespace(jail_denied_line=None, jail_denials=one),
        )
        base.report_jail_outcome("x", {"jail_denials": _fanout_record(
            landlock_dispatch.JailDenials())}, "u37c-c")
        assert capsys.readouterr().out == (
            "[x] the plugin's Landlock jail denied 1 write inside the model's own "
            "commands (not a failure)\n"
        )

    def test_the_jail_line_goes_to_each_lanes_closing_summary(self, capsys):
        line = "[JAIL_DENIED] codex: simulated"
        start = probe_state.jail_summary_count()
        dispatch_spawn._report_jail_outcome(
            "T003", types.SimpleNamespace(jail_denied_line=line, jail_denials=None),
        )
        assert probe_state.jail_summary_lines(start) == [line]
        base.report_jail_outcome(SPEC, {"jail_failure": {"notice": line}}, "u37c-d")
        assert capsys.readouterr().out == (
            f"[exec-subagents] {line}\n[{SPEC}] {line}\n"
        )
        assert base.print_closing_summary("u37c-d") == [line]

    def test_the_record_round_trips_to_the_same_denials(self):
        record = _fanout_record(DENIED)
        back = landlock_dispatch.as_jail_denials(record)
        assert back == DENIED and back.named_paths() == record["paths"]
        assert landlock_dispatch.as_jail_denials(DENIED) is DENIED
        assert landlock_dispatch.as_jail_denials(None) is None
        assert landlock_dispatch.as_jail_denials("text") is None


# ---------------------------------------------------------------------------
# The audit argv
# ---------------------------------------------------------------------------


class _Refusing:
    name = "refusing"

    def build_agent_command(self, prompt, model, profile):
        raise ValueError("no such profile")


class TestTheAuditArgv:
    def test_the_jail_wraps_the_resolved_command(self):
        codex = get_provider("codex")
        request = JailRequest(repo_root="/repo", kind="validation")
        argv = dispatch_spawn.render_audit_argv(
            codex, "gpt-5.5", PROFILE_AGENT_READONLY, "p", "plugin-landlock", request,
            launcher=lambda exe: [f"/resolved/{exe}"],
        )
        assert argv[:3] == [sys.executable, "-I", "-S"]
        inner = argv[argv.index("--") + 1:]
        assert inner[0] == "/resolved/codex"
        assert inner[inner.index("--sandbox") + 1] == "danger-full-access"
        assert f"dir:{landlock_dispatch.PLACEHOLDER_TMPDIR}" in argv

    def test_the_jail_with_no_request_renders_the_bare_command(self):
        codex = get_provider("codex")
        bare = list(codex.build_agent_command("p", "gpt-5.5", PROFILE_AGENT_READONLY))
        assert dispatch_spawn.render_audit_argv(
            codex, "gpt-5.5", PROFILE_AGENT_READONLY, "p", "plugin-landlock", None,
        ) == bare

    def test_f5_renders_with_its_consent_and_falls_back_without_it(self):
        codex = get_provider("codex")
        consent = WritePostureConsent(posture="danger-full-access", write_scope="plan-file")
        argv = dispatch_spawn.render_audit_argv(
            codex, "gpt-5.5", PROFILE_AGENT_WRITE, "p", "unsandboxed", None, consent,
        )
        assert argv[argv.index("--sandbox") + 1] == "danger-full-access"
        with pytest.raises(Exception):
            dispatch_spawn.render_audit_argv(
                codex, "gpt-5.5", PROFILE_AGENT_WRITE, "p", "unsandboxed", None, None,
            )
        bare = list(codex.build_agent_command("p", "gpt-5.5", PROFILE_AGENT_WRITE))
        assert dispatch_spawn.audit_argv(
            codex, "codex", "gpt-5.5", PROFILE_AGENT_WRITE, "p", rung="unsandboxed",
        ) == bare

    def test_a_refused_profile_names_the_runner(self):
        assert dispatch_spawn.audit_argv(
            _Refusing(), "refusing", "m", PROFILE_AGENT_READONLY, "p", rung="landlock",
        ) == ["refusing", "<profile unsupported>"]


class TestTheJudgesAuditLine:
    """The judge's ``dispatched`` line renders the rung its dispatch takes
    (the recorded one, replayed as ``_dispatch_grant`` does for a read kind),
    as the write scheduler's does: the audit record never shows a sandbox the
    child did not get."""

    @staticmethod
    def _audit_argv(tmp_path, rung):
        runner, profile = "codex:gpt-5.5", PROFILE_AGENT_READONLY
        if rung is not None:
            probe_state._SANDBOX_DECISIONS[
                probe_state._sandbox_decision_key(runner, profile, None)
            ] = rung
        hook = task_batch._judge_audit_before_attempt(str(tmp_path), None)
        hook({
            "task": {"task_id": "J1"}, "task_id": "J1", "kind": "human_decision_judge",
            "attempt": 1, "runner": runner, "profile": profile, "cwd": str(tmp_path),
            "prompt": "judge it",
        })
        [record] = [
            json.loads(line)
            for line in (tmp_path / "subagent_audit.jsonl").read_text().splitlines()
        ]
        return record["argv"]

    def test_no_recorded_rung_keeps_codexs_own_sandbox(self, tmp_path):
        argv = self._audit_argv(tmp_path, None)
        assert argv[0] == "codex" and argv[argv.index("--sandbox") + 1] == "read-only"

    def test_the_uncontained_rung_is_shown(self, tmp_path):
        argv = self._audit_argv(tmp_path, "unsandboxed")
        assert argv[argv.index("--sandbox") + 1] == "danger-full-access"
        assert "read-only" not in argv

    def test_the_jail_is_shown_wrapped(self, tmp_path):
        argv = self._audit_argv(tmp_path, "plugin-landlock")
        assert argv[:3] == [sys.executable, "-I", "-S"]
        inner = argv[argv.index("--") + 1:]
        assert inner[0] == "codex"
        assert inner[inner.index("--sandbox") + 1] == "danger-full-access"
