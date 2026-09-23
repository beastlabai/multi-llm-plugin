"""The sandbox fallback ladder (landlock rung, 2026-09-22) — T1 to T4.

On a machine where bubblewrap cannot start, codex's read-only phases used to
fall straight to `--sandbox danger-full-access`. The ladder puts a contained
rung — `--sandbox read-only` plus a per-dispatch `-c
features.use_legacy_landlock=true` — above that bypass. This file guards:

* the vocabulary (`SandboxRung`, the rung ids, the three-valued key);
* codex's two rungs and the argv they render — the `--sandbox` slot stays at
  index 2, the prompt stays last, the feature pair lands once at the measured
  splice site and never beside `workspace-write`;
* `sandbox_verdict`'s walk, one test per row of the plan's three-way table,
  plus the single shared deadline;
* the key's table (`auto` / `contained` / `off`) at both gates;
* the rung id surviving every carrier to the argv — never a bool, never the
  word `bypassed` for a contained rung.

Reporting (T5), doctor (T6) and the container lane (T8) have their own guards.
"""

import shlex
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import review_orchestrator_base as rob  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers import base as base_module  # noqa: E402
from utils.providers import codex as codex_module  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    SANDBOX_CANARY_DENIED,
    SANDBOX_CANARY_NOT_DENIED,
    SANDBOX_FALLBACK_PROFILES,
    SandboxRung,
    sandbox_rung_allowed,
    sandbox_rung_id,
)
from utils.providers.codex import (  # noqa: E402
    LANDLOCK_CANARY_SCRIPT,
    LANDLOCK_FEATURE_PAIR,
    LANDLOCK_PROBE_ARGV,
    LANDLOCK_RUNG,
    UNSANDBOXED_RUNG,
    CodexProvider,
)

BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
PROMPT = "Review the plan. PROMPT-SENTINEL"
FLAG_VALUE = LANDLOCK_FEATURE_PAIR[1]

#: Stand-in probe scripts, one per row of the walk table. The `sh` is real.
CANARY_DENIED = f"echo '{SANDBOX_CANARY_DENIED}'; exit 0"
#: Exit 0 with no marker: a CLI that printed usage and never ran the canary
#: (the plan's walk-table row 2), and a bare `exit 0` (a flag accepted as a
#: no-op). Neither measured a boundary.
USAGE_EXIT_0 = "echo 'Usage: codex sandbox [OPTIONS] -- <COMMAND>...'; exit 0"
BARE_EXIT_0 = "exit 0"
#: A CLI that echoes its whole argv back (the canary script included) and
#: exits 0 — the marker must not be forgeable from the script's own text.
ECHOES_THE_CANARY = f"printf '%s\\n' {shlex.quote(LANDLOCK_CANARY_SCRIPT)}; exit 0"
CANARY_LANDED = f"echo '{SANDBOX_CANARY_NOT_DENIED}' >&2; exit 3"
SANDBOX_BROKEN = f"echo '{BWRAP_LINE}' >&2; exit 1"
FLAG_UNKNOWN = "echo 'error: unexpected argument --enable found' >&2; exit 2"
HANGS = "sleep 5"


def _landlock_stand_in(script: str) -> SandboxRung:
    """Codex's rung 1 with its probe swapped for ``script`` — same name, same
    rewrite, same containment claim, so the adapter still renders it."""
    return SandboxRung(
        name=LANDLOCK_RUNG.name,
        probe_argv=("sh", "-c", script),
        argv_rewrite=LANDLOCK_RUNG.argv_rewrite,
        contained=True,
        retires_when=LANDLOCK_RUNG.retires_when,
    )


class _Ladder:
    """An adapter whose own probe fails like codex's does where bwrap cannot
    start, declaring codex's two-rung ladder with a scripted rung-1 probe."""

    internal_retries = None

    def __init__(self, policy="auto", landlock=CANARY_DENIED, profile_probe=SANDBOX_BROKEN):
        self.policy = policy
        self.landlock = landlock
        self.profile_probe = profile_probe

    def sandbox_probe_command(self, profile):
        return ["sh", "-c", self.profile_probe]

    def sandbox_fallback_policy(self, profile, anchor=None):
        return self.policy if profile in SANDBOX_FALLBACK_PROFILES else None

    def sandbox_fallback_rungs(self, profile):
        if profile not in SANDBOX_FALLBACK_PROFILES:
            return ()
        return (_landlock_stand_in(self.landlock), UNSANDBOXED_RUNG)


# ---------------------------------------------------------------------------
# T1 — the vocabulary
# ---------------------------------------------------------------------------


class TestTheRungVocabulary:
    def test_the_key_gains_contained_and_keeps_off_and_auto(self):
        assert base_module.SANDBOX_FALLBACK_POLICIES == ("off", "auto", "contained")

    @pytest.mark.parametrize("policy, contained, uncontained", [
        ("auto", True, True),
        ("contained", True, False),
        ("off", False, False),
        (None, False, False),
        ("garbage", False, False),
    ])
    def test_the_configuration_table(self, policy, contained, uncontained):
        """`auto` both rungs, `contained` rung 1 only, `off` none."""
        assert sandbox_rung_allowed(LANDLOCK_RUNG, policy) is contained
        assert sandbox_rung_allowed(UNSANDBOXED_RUNG, policy) is uncontained

    def test_a_carrier_value_normalises_to_a_rung_id(self):
        assert sandbox_rung_id(None) is None
        assert sandbox_rung_id(False) is None
        assert sandbox_rung_id("") is None
        assert sandbox_rung_id("landlock") == "landlock"
        # The pre-rung bool always meant the uncontained rung, never another.
        assert sandbox_rung_id(True) == "unsandboxed"

    def test_the_base_class_declares_no_ladder(self):
        class Plain(base_module.LLMProvider):
            name = "plain"
            default_timeout = 1

            def build_command(self, prompt, model, **kwargs):
                return ["plain", prompt]

            def parse_output(self, stdout, stderr):
                return stdout

            def is_available(self):
                return True

        plain = Plain()
        for profile in (PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
            assert plain.sandbox_fallback_rungs(profile) == ()
        argv = ["plain", "p"]
        assert plain.apply_sandbox_fallback(list(argv), PROFILE_GENERATE, "landlock") == argv

    def test_a_rung_is_frozen_data(self):
        with pytest.raises(Exception):
            LANDLOCK_RUNG.contained = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# T2 — codex's two rungs and the argv they render
# ---------------------------------------------------------------------------


class TestCodexDeclaresTwoRungs:
    def test_the_ladder_is_landlock_then_unsandboxed_for_the_read_only_profiles(self):
        provider = CodexProvider()
        for profile in (PROFILE_GENERATE, PROFILE_AGENT_READONLY):
            assert [r.name for r in provider.sandbox_fallback_rungs(profile)] == [
                "landlock", "unsandboxed",
            ]
        assert provider.sandbox_fallback_rungs(PROFILE_AGENT_WRITE) == ()

    def test_rung_one_is_contained_and_says_when_it_retires(self):
        assert LANDLOCK_RUNG.contained is True
        assert UNSANDBOXED_RUNG.contained is False
        version, sentence = LANDLOCK_RUNG.retires_when
        assert version == "0.155.1"
        assert "use_legacy_landlock" in sentence and "deprecated" in sentence
        assert UNSANDBOXED_RUNG.probe_argv is None
        assert UNSANDBOXED_RUNG.argv_rewrite == ("--sandbox", "danger-full-access")

    def test_the_probe_is_pinned_in_the_dispatch_spelling_with_a_canary(self):
        """Probe and dispatch carry the feature the same way, and the probe's
        command is the canary — never `exit 0`, never the profile probe's
        write-a-byte script, whose exit code would come out inverted."""
        assert LANDLOCK_PROBE_ARGV[:2] == ("codex", "sandbox")
        assert LANDLOCK_PROBE_ARGV[2:4] == LANDLOCK_FEATURE_PAIR
        assert LANDLOCK_PROBE_ARGV[4:6] == ("-c", 'sandbox_mode="read-only"')
        assert "--enable" not in LANDLOCK_PROBE_ARGV
        assert LANDLOCK_PROBE_ARGV[-1] == LANDLOCK_CANARY_SCRIPT
        assert LANDLOCK_RUNG.probe_argv == LANDLOCK_PROBE_ARGV
        generate_probe = CodexProvider().sandbox_probe_command(PROFILE_GENERATE)
        assert generate_probe[-1] != LANDLOCK_CANARY_SCRIPT
        assert LANDLOCK_PROBE_ARGV != tuple(generate_probe)

    def test_the_canary_fails_when_its_write_lands(self, tmp_path):
        """Run unsandboxed the write lands: exit 3, the marker, nothing left."""
        done = subprocess.run(
            ["sh", "-c", LANDLOCK_CANARY_SCRIPT], cwd=tmp_path,
            capture_output=True, text=True,
        )
        assert done.returncode == 3
        assert SANDBOX_CANARY_NOT_DENIED in done.stderr
        assert SANDBOX_CANARY_DENIED not in f"{done.stdout}{done.stderr}"
        assert list(tmp_path.iterdir()) == []

    def test_a_landed_write_leaves_pre_existing_dirs_alone(self, tmp_path):
        """Fail-open with `.multi-llm/tmp` already there (the walk's own
        scratch dir): the canary removes only its file, never those dirs."""
        workdir = tmp_path / ".multi-llm" / "tmp"
        workdir.mkdir(parents=True)
        done = subprocess.run(
            ["sh", "-c", LANDLOCK_CANARY_SCRIPT], cwd=tmp_path,
            capture_output=True, text=True,
        )
        assert done.returncode == 3
        assert workdir.is_dir()
        assert list(workdir.iterdir()) == []

    def test_the_script_never_carries_its_success_marker_verbatim(self):
        """A CLI that echoes its argv must not be able to forge the marker:
        only a shell that ran the script prints it."""
        assert SANDBOX_CANARY_DENIED not in LANDLOCK_CANARY_SCRIPT
        assert SANDBOX_CANARY_DENIED not in " ".join(LANDLOCK_PROBE_ARGV)

    @pytest.mark.skipif(
        hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
        reason="root ignores the directory mode that stands in for the denial",
    )
    def test_the_canary_passes_when_its_write_is_denied(self, tmp_path):
        """A read-only (pre-existing) directory stands in for the kernel's
        denial of the write, and the canary's success is exit 0."""
        root = tmp_path / "root"
        workdir = root / ".multi-llm" / "tmp"
        workdir.mkdir(parents=True)
        dirs = [root, root / ".multi-llm", workdir]
        for d in dirs:
            d.chmod(0o555)
        try:
            done = subprocess.run(
                ["sh", "-c", LANDLOCK_CANARY_SCRIPT], cwd=root,
                capture_output=True, text=True,
            )
        finally:
            for d in dirs:
                d.chmod(0o755)
        assert done.returncode == 0, done.stderr
        assert SANDBOX_CANARY_NOT_DENIED not in done.stderr
        assert SANDBOX_CANARY_DENIED in done.stdout

    @pytest.mark.skipif(
        hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
        reason="root ignores the directory mode that stands in for the denial",
    )
    def test_a_denied_mkdir_is_a_setup_failure_not_containment(self, tmp_path):
        """A `mkdir` that fails never reaches the write: that is a setup
        failure — its own exit code, no marker — never the denial."""
        tmp_path.chmod(0o555)
        try:
            done = subprocess.run(
                ["sh", "-c", LANDLOCK_CANARY_SCRIPT], cwd=tmp_path,
                capture_output=True, text=True,
            )
        finally:
            tmp_path.chmod(0o755)
        assert done.returncode == codex_module.LANDLOCK_CANARY_SETUP_EXIT
        assert codex_module.LANDLOCK_CANARY_SETUP_FAILED in done.stderr
        assert SANDBOX_CANARY_DENIED not in f"{done.stdout}{done.stderr}"
        assert SANDBOX_CANARY_NOT_DENIED not in done.stderr

    def test_a_non_directory_in_the_way_is_a_setup_failure(self, tmp_path):
        (tmp_path / ".multi-llm").mkdir()
        (tmp_path / ".multi-llm" / "tmp").write_text("not a dir", encoding="utf-8")
        done = subprocess.run(
            ["sh", "-c", LANDLOCK_CANARY_SCRIPT], cwd=tmp_path,
            capture_output=True, text=True,
        )
        assert done.returncode == codex_module.LANDLOCK_CANARY_SETUP_EXIT
        assert SANDBOX_CANARY_DENIED not in f"{done.stdout}{done.stderr}"
        assert (tmp_path / ".multi-llm" / "tmp").read_text(encoding="utf-8") == "not a dir"

    def test_the_rung_declares_the_canary_workdir(self):
        assert LANDLOCK_RUNG.probe_workdir == codex_module.LANDLOCK_CANARY_WORKDIR
        assert UNSANDBOXED_RUNG.probe_workdir is None


def _variants():
    """Every argv shape a rung has to survive: plain, with an effort suffix
    (`-c model_reasoning_effort=…`) and with a materialized schema."""
    return [("gpt-5.5", False), ("gpt-5.5:high", False), ("gpt-5.5:high", True)]


@pytest.fixture
def schema_file(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text("{}", encoding="utf-8")
    return base_module.StructuredSchema({}, path=str(path), name="s")


class TestTheRungArgvInvariants:
    """`--sandbox` at index 2, the prompt last, the pair once at the measured
    site — on every rung, both profiles, both transports."""

    @staticmethod
    def _render(profile, rung, model, schema, transport):
        provider = CodexProvider()
        if profile == PROFILE_GENERATE:
            return provider.build_command_for_transport(
                PROMPT, model, transport, schema=schema, sandbox_fallback=rung,
            )
        return provider.build_agent_command_for_transport(
            PROMPT, model, profile, transport, schema=schema, sandbox_fallback=rung,
        )

    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    @pytest.mark.parametrize("rung", ["landlock", "unsandboxed"])
    @pytest.mark.parametrize("transport", ["argv", "stdin"])
    @pytest.mark.parametrize("model, with_schema", _variants())
    def test_the_positions_hold_on_every_rung(
        self, profile, rung, transport, model, with_schema, schema_file
    ):
        schema = schema_file if with_schema else None
        argv, stdin = self._render(profile, rung, model, schema, transport)
        assert argv.index("--sandbox") == 2
        if transport == "argv":
            assert argv[-1] == PROMPT and stdin is None
        else:
            assert stdin == PROMPT and PROMPT not in argv
        count = sum(
            1 for i in range(len(argv) - 1)
            if (argv[i], argv[i + 1]) == LANDLOCK_FEATURE_PAIR
        )
        if rung == "landlock":
            assert argv[3] == "read-only"
            assert count == 1
            # The measured splice site: straight after `--skip-git-repo-check`,
            # so ahead of `--output-schema` and `--model` — never appended.
            at = argv.index("--skip-git-repo-check") + 1
            assert tuple(argv[at:at + 2]) == LANDLOCK_FEATURE_PAIR
            assert argv.index(FLAG_VALUE) < argv.index("--model")
            if with_schema:
                assert argv.index(FLAG_VALUE) < argv.index("--output-schema")
        else:
            assert argv[3] == "danger-full-access"
            assert count == 0 and FLAG_VALUE not in argv

    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    def test_the_feature_never_rides_beside_workspace_write(self, profile):
        """`workspace-write` + landlock is the exit-101 panic: rung 1 renders
        `read-only` whatever the profile's own row is."""
        argv, _ = self._render(profile, "landlock", "gpt-5.5", None, "argv")
        assert "workspace-write" not in argv
        assert argv[argv.index("--sandbox") + 1] == "read-only"

    def test_the_healthy_machine_argv_is_untouched(self):
        """No rung: `generate` still renders `workspace-write` and no feature."""
        provider = CodexProvider()
        argv, _ = provider.build_command_for_transport(PROMPT, "gpt-5.5", "argv")
        assert argv[3] == "workspace-write" and FLAG_VALUE not in argv
        agent = provider.build_agent_command(PROMPT, "gpt-5.5", PROFILE_AGENT_READONLY)
        assert agent[3] == "read-only" and FLAG_VALUE not in agent
        assert codex_module._SANDBOX_BY_PROFILE[PROFILE_GENERATE] == "workspace-write"

    def test_an_agent_leg_through_both_calls_carries_the_pair_once(self):
        """`build_agent_command` then `apply_sandbox_fallback` (and a second
        dispatch-time pass for good measure): the insert is idempotent."""
        provider = CodexProvider()
        argv = provider.build_agent_command(PROMPT, "gpt-5.5:high", PROFILE_AGENT_READONLY)
        argv = provider.apply_sandbox_fallback(argv, PROFILE_AGENT_READONLY, "landlock")
        argv = provider.apply_sandbox_fallback(argv, PROFILE_AGENT_READONLY, "landlock")
        assert argv.count(FLAG_VALUE) == 1
        assert argv.index("--sandbox") == 2 and argv[-1] == PROMPT

    def test_a_prompt_that_spells_the_value_does_not_suppress_the_insert(self):
        provider = CodexProvider()
        argv = provider.build_command(FLAG_VALUE, "gpt-5.5")
        argv = provider.apply_sandbox_fallback(argv, PROFILE_GENERATE, "landlock")
        assert argv[-1] == FLAG_VALUE
        at = argv.index("--skip-git-repo-check") + 1
        assert tuple(argv[at:at + 2]) == LANDLOCK_FEATURE_PAIR

    @pytest.mark.parametrize("rung", [None, "", "bogus"])
    def test_a_missing_or_unknown_rung_raises_rather_than_rendering_one(self, rung):
        provider = CodexProvider()
        argv = provider.build_command(PROMPT, "gpt-5.5")
        with pytest.raises(ValueError, match="no sandbox fallback rung"):
            provider.apply_sandbox_fallback(argv, PROFILE_GENERATE, rung)

    @pytest.mark.parametrize("rung", ["landlock", "unsandboxed"])
    def test_the_write_profile_reaches_no_rung(self, rung):
        provider = CodexProvider()
        argv = provider.build_command(PROMPT, "gpt-5.5")
        with pytest.raises(ValueError, match="agent-write"):
            provider.apply_sandbox_fallback(argv, PROFILE_AGENT_WRITE, rung)
        with pytest.raises(ValueError, match="agent-write"):
            provider.build_agent_command_for_transport(
                PROMPT, "gpt-5.5", PROFILE_AGENT_WRITE, "argv", sandbox_fallback=rung,
            )


# ---------------------------------------------------------------------------
# T3 — the verdict carries the rung, and the walk is three-way
# ---------------------------------------------------------------------------


class TestTheWalkTable:
    """One test per row of the plan's walk table."""

    def test_canary_denied_takes_the_rung(self):
        verdict = executor.sandbox_verdict(_Ladder(landlock=CANARY_DENIED), PROFILE_GENERATE)
        assert (verdict.state, verdict.rung) == ("unavailable", "landlock")
        assert BWRAP_LINE in verdict.tail
        assert verdict.walk == (("landlock", "taken"),)

    def test_a_canary_whose_write_landed_rejects_the_rung_and_moves_on(self):
        verdict = executor.sandbox_verdict(_Ladder(landlock=CANARY_LANDED), PROFILE_GENERATE)
        assert verdict.rung == "unsandboxed"
        assert verdict.walk == (("landlock", "rejected"), ("unsandboxed", "taken"))

    @pytest.mark.parametrize("script", [USAGE_EXIT_0, BARE_EXIT_0])
    def test_an_exit_0_without_the_canary_marker_is_not_containment(self, script):
        """Exit 0 is not containment: a CLI that exits 0 without running the
        canary to completion never measured a boundary, so the rung is not
        taken and the walk moves on exactly as for a probe it did not understand."""
        verdict = executor.sandbox_verdict(_Ladder(landlock=script), PROFILE_GENERATE)
        assert verdict.rung == "unsandboxed"
        assert verdict.walk == (("landlock", "absent"), ("unsandboxed", "taken"))

    def test_an_echoed_canary_script_cannot_forge_the_marker(self):
        """A CLI that prints its argv (the canary's source) and exits 0 ran
        nothing: whatever else it reads as, the landlock rung is not taken."""
        verdict = executor.sandbox_verdict(
            _Ladder(landlock=ECHOES_THE_CANARY), PROFILE_GENERATE,
        )
        assert verdict.rung == "unsandboxed"
        assert verdict.walk[0][0] == "landlock"
        assert verdict.walk[0][1] != "taken"

    def test_a_real_sandbox_failure_moves_on(self):
        verdict = executor.sandbox_verdict(_Ladder(landlock=SANDBOX_BROKEN), PROFILE_GENERATE)
        assert verdict.rung == "unsandboxed"
        assert verdict.walk == (("landlock", "unavailable"), ("unsandboxed", "taken"))

    def test_an_inconclusive_tail_skips_that_rung_only(self):
        """An older codex that does not know the flag reaches rung 2 — today."""
        verdict = executor.sandbox_verdict(_Ladder(landlock=FLAG_UNKNOWN), PROFILE_GENERATE)
        assert verdict.rung == "unsandboxed"
        assert verdict.walk == (("landlock", "absent"), ("unsandboxed", "taken"))

    def test_a_timeout_stops_the_walk_with_no_rung(self):
        verdict = executor.sandbox_verdict(
            _Ladder(landlock=HANGS), PROFILE_GENERATE, timeout=0.5,
        )
        assert verdict.state == "unavailable"
        assert verdict.rung is None
        assert verdict.walk == (("landlock", "unprobed"),)

    def test_a_spawn_failure_stops_the_walk_with_no_rung(self, monkeypatch):
        real = executor._run_sandbox_probe

        def unspawnable_rung(argv, **kwargs):
            if argv[-1] == CANARY_DENIED:
                return None
            return real(argv, **kwargs)

        monkeypatch.setattr(executor, "_run_sandbox_probe", unspawnable_rung)
        verdict = executor.sandbox_verdict(_Ladder(landlock=CANARY_DENIED), PROFILE_GENERATE)
        assert verdict.rung is None
        assert verdict.walk == (("landlock", "unknown"),)

    @staticmethod
    def _workdir_rung(script):
        return SandboxRung(
            name=LANDLOCK_RUNG.name,
            probe_argv=("sh", "-c", script),
            argv_rewrite=LANDLOCK_RUNG.argv_rewrite,
            contained=True,
            probe_workdir=codex_module.LANDLOCK_CANARY_WORKDIR,
        )

    def test_the_walk_sets_the_canary_dir_up_outside_and_removes_what_it_made(
        self, tmp_path
    ):
        """The walk creates `.multi-llm/tmp` before the probe (so the probe
        sees an existing, writable directory) and removes only what it made."""
        script = f"[ -d .multi-llm/tmp ] || exit 9; {CANARY_DENIED}"
        rung, walk = executor._walk_fallback_rungs(
            (self._workdir_rung(script), UNSANDBOXED_RUNG),
            cwd=str(tmp_path), deadline=time.monotonic() + 10,
        )
        assert (rung, walk) == ("landlock", (("landlock", "taken"),))
        assert list(tmp_path.iterdir()) == []

    def test_the_walk_leaves_a_pre_existing_canary_dir_alone(self, tmp_path):
        (tmp_path / ".multi-llm" / "tmp").mkdir(parents=True)
        rung, _ = executor._walk_fallback_rungs(
            (self._workdir_rung(CANARY_DENIED),),
            cwd=str(tmp_path), deadline=time.monotonic() + 10,
        )
        assert rung == "landlock"
        assert (tmp_path / ".multi-llm" / "tmp").is_dir()

    def test_a_non_directory_in_the_way_is_never_probed_nor_taken(self, tmp_path):
        (tmp_path / ".multi-llm").mkdir()
        (tmp_path / ".multi-llm" / "tmp").write_text("x", encoding="utf-8")
        ran = tmp_path / "probe-ran"
        rung, walk = executor._walk_fallback_rungs(
            (self._workdir_rung(f": > {shlex.quote(str(ran))}; {CANARY_DENIED}"),
             UNSANDBOXED_RUNG),
            cwd=str(tmp_path), deadline=time.monotonic() + 10,
        )
        assert rung is None
        assert walk == (("landlock", "unknown"),)
        assert not ran.exists()

    @pytest.mark.skipif(
        hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
        reason="root ignores the directory mode",
    )
    def test_a_canary_dir_unwritable_outside_the_sandbox_is_not_taken(self, tmp_path):
        """Plain permissions deny the write too — the walk proves the directory
        writable outside the sandbox first, so that denial never reads as a
        boundary."""
        workdir = tmp_path / ".multi-llm" / "tmp"
        workdir.mkdir(parents=True)
        workdir.chmod(0o555)
        try:
            rung, walk = executor._walk_fallback_rungs(
                (self._workdir_rung(CANARY_DENIED), UNSANDBOXED_RUNG),
                cwd=str(tmp_path), deadline=time.monotonic() + 10,
            )
        finally:
            workdir.chmod(0o755)
        assert rung is None
        assert walk == (("landlock", "unknown"),)

    def test_an_in_sandbox_setup_failure_is_never_taken(self):
        """The canary's own setup-failure exit carries no marker: not taken."""
        script = (
            f"echo '{codex_module.LANDLOCK_CANARY_SETUP_FAILED}' >&2; "
            f"exit {codex_module.LANDLOCK_CANARY_SETUP_EXIT}"
        )
        verdict = executor.sandbox_verdict(_Ladder(landlock=script), PROFILE_GENERATE)
        assert verdict.walk[0] == ("landlock", "unavailable")
        assert verdict.rung == "unsandboxed"

    @pytest.mark.parametrize("profile_probe, state", [
        ("exit 0", "ok"), (FLAG_UNKNOWN, "unknown"),
    ])
    def test_no_walk_unless_the_profile_probe_said_unavailable(
        self, profile_probe, state, monkeypatch
    ):
        calls = []
        real = executor._run_sandbox_probe

        def counting(argv, **kwargs):
            calls.append(list(argv))
            return real(argv, **kwargs)

        monkeypatch.setattr(executor, "_run_sandbox_probe", counting)
        verdict = executor.sandbox_verdict(
            _Ladder(profile_probe=profile_probe), PROFILE_GENERATE,
        )
        assert (verdict.state, verdict.rung, verdict.walk) == (state, None, ())
        assert len(calls) == 1

    def test_the_write_profile_walks_nothing(self):
        verdict = executor.sandbox_verdict(_Ladder(), PROFILE_AGENT_WRITE)
        assert (verdict.state, verdict.rung, verdict.walk) == ("unavailable", None, ())


class TestTheVerdictShape:
    def test_a_two_value_unpack_raises_instead_of_dropping_the_rung(self):
        verdict = executor.sandbox_verdict(_Ladder(), PROFILE_GENERATE)
        with pytest.raises(ValueError):
            state, tail = verdict  # noqa: F841 - the unpack is the assertion
        state, tail, rung, walk = verdict
        assert (state, rung) == ("unavailable", "landlock")
        assert verdict._fields == ("state", "tail", "rung", "walk")

    def test_a_verdict_with_nothing_to_probe_still_has_every_field(self):
        class NoProbe:
            def sandbox_probe_command(self, profile):
                return None

        verdict = executor.sandbox_verdict(NoProbe(), PROFILE_GENERATE)
        assert verdict == executor.SandboxProbeResult("not-required", "", None, ())


class TestTheWalkSharesOneDeadline:
    def test_a_slow_first_probe_leaves_the_rung_unprobed_inside_one_budget(self):
        """The profile probe spends most of the budget; the rung probe gets
        only what is left of it, and the whole call stays inside one budget."""
        budget = 1.0
        ladder = _Ladder(
            profile_probe=f"sleep 0.7; {SANDBOX_BROKEN}", landlock=HANGS,
        )
        started = time.monotonic()
        verdict = executor.sandbox_verdict(ladder, PROFILE_GENERATE, timeout=budget)
        elapsed = time.monotonic() - started
        assert verdict.state == "unavailable"
        assert verdict.rung is None
        assert verdict.walk == (("landlock", "unprobed"),)
        assert elapsed < budget + 0.5, elapsed
        # And the uncontained rung was never reached on exhausted time.
        assert "unsandboxed" not in dict(verdict.walk)

    def test_a_rung_with_no_time_left_is_not_started(self, monkeypatch):
        started = []
        monkeypatch.setattr(
            executor, "_run_sandbox_probe",
            lambda argv, **kwargs: started.append(argv),
        )
        rung, walk = executor._walk_fallback_rungs(
            (_landlock_stand_in(CANARY_DENIED), UNSANDBOXED_RUNG),
            cwd=None, deadline=time.monotonic() - 1.0,
        )
        assert rung is None
        assert walk == (("landlock", "unprobed"),)
        assert started == []

    def test_the_default_budget_is_one_probe_timeout_for_the_whole_walk(self, monkeypatch):
        seen = []

        def recording(argv, *, cwd=None, timeout=None):
            seen.append(timeout)
            return subprocess.CompletedProcess(argv, 1, "", BWRAP_LINE)

        monkeypatch.setattr(executor, "_run_sandbox_probe", recording)
        monkeypatch.setattr(executor, "SANDBOX_PROBE_TIMEOUT_S", 10.0)
        executor.sandbox_verdict(_Ladder(landlock=SANDBOX_BROKEN), PROFILE_GENERATE)
        assert seen[0] == 10.0
        assert 0 < seen[1] <= 10.0
        assert len(seen) == 2


# ---------------------------------------------------------------------------
# T4 — the two gates, one test per key value
# ---------------------------------------------------------------------------


@pytest.fixture
def executor_gate(monkeypatch):
    """`_detect_sandbox` with a stand-in adapter and a clean cache."""
    monkeypatch.setattr(executor, "_SANDBOX_CACHE", {})

    def run(adapter, profile=PROFILE_AGENT_READONLY):
        monkeypatch.setattr(executor, "get_provider", lambda _name: adapter)
        return executor._detect_sandbox(
            "fakecodex", key="driver.subagent_runner", spec="fakecodex:m",
            profile=profile,
        )
    return run


class TestTheExecutorGate:
    def test_auto_takes_the_contained_rung_and_records_its_id(self, executor_gate, capsys):
        state = executor_gate(_Ladder(policy="auto"))
        assert state == "landlock"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_READONLY) == "landlock"
        assert not executor.sandbox_bypassed("fakecodex:m", PROFILE_AGENT_READONLY)
        out = capsys.readouterr().out
        assert "[SANDBOX_FALLBACK] fakecodex" in out
        assert "[SANDBOX_BYPASSED]" not in out
        # Cached: replayed without re-probing or re-printing.
        assert executor_gate(_Ladder(policy="auto", landlock=HANGS)) == "landlock"
        assert capsys.readouterr().out == ""

    def test_auto_falls_to_the_uncontained_rung_when_rung_one_fails(self, executor_gate, capsys):
        state = executor_gate(_Ladder(policy="auto", landlock=SANDBOX_BROKEN))
        assert state == "bypassed"
        assert executor.sandbox_rung("fakecodex:m", PROFILE_AGENT_READONLY) == "unsandboxed"
        assert executor.sandbox_bypassed("fakecodex:m", PROFILE_AGENT_READONLY)
        out = capsys.readouterr().out
        assert out.count("[SANDBOX_BYPASSED]") == 1
        assert "--sandbox danger-full-access" in out

    def test_contained_takes_rung_one(self, executor_gate, capsys):
        assert executor_gate(_Ladder(policy="contained")) == "landlock"
        assert "sandbox_fallback=contained" in capsys.readouterr().out

    def test_contained_never_takes_rung_two(self, executor_gate, capsys):
        with pytest.raises(executor.RunnerRejected) as excinfo:
            executor_gate(_Ladder(policy="contained", landlock=SANDBOX_BROKEN))
        assert excinfo.value.reason_code == "sandbox_unavailable"
        assert excinfo.value.exit_code == 3
        assert "sandbox_fallback=contained" in str(excinfo.value)
        assert executor._SANDBOX_CACHE == {}
        assert "[SANDBOX_BYPASSED]" not in capsys.readouterr().out

    @pytest.mark.parametrize("landlock", [CANARY_DENIED, SANDBOX_BROKEN])
    def test_off_reaches_no_rung(self, executor_gate, capsys, landlock):
        with pytest.raises(executor.RunnerRejected) as excinfo:
            executor_gate(_Ladder(policy="off", landlock=landlock))
        assert excinfo.value.reason_code == "sandbox_unavailable"
        assert executor._SANDBOX_CACHE == {}
        out = capsys.readouterr().out
        assert "[SANDBOX_BYPASSED]" not in out and "[SANDBOX_FALLBACK]" not in out

    def test_a_stopped_walk_refuses_under_auto_and_says_why(self, executor_gate, monkeypatch):
        monkeypatch.setattr(executor, "SANDBOX_PROBE_TIMEOUT_S", 0.5)
        with pytest.raises(executor.RunnerRejected) as excinfo:
            executor_gate(_Ladder(policy="auto", landlock=HANGS))
        assert "'landlock' was unprobed" in str(excinfo.value)

    @pytest.mark.parametrize("policy", ["auto", "contained"])
    def test_agent_write_reaches_no_rung(self, executor_gate, policy):
        with pytest.raises(executor.RunnerRejected):
            executor_gate(_Ladder(policy=policy), profile=PROFILE_AGENT_WRITE)
        assert executor._SANDBOX_CACHE == {}


@pytest.fixture
def fanout_gate(monkeypatch):
    def run(adapter):
        monkeypatch.setattr(rob, "_FANOUT_SANDBOX_CACHE", {})
        monkeypatch.setattr(rob, "get_provider", lambda _name: adapter)
        return rob.sandbox_refusal("fakecodex", None)
    return run


class TestTheFanoutGate:
    def test_auto_takes_the_contained_rung_which_is_not_a_bypass(self, fanout_gate):
        verdict = fanout_gate(_Ladder(policy="auto"))
        assert verdict.verdict == rob.SANDBOX_DISPATCH_UNSANDBOXED
        assert verdict.rung == "landlock"
        assert not verdict.bypass and not verdict.refused
        assert verdict.reason.startswith("[SANDBOX_FALLBACK] fakecodex")
        assert rob.sandbox_fallback_kwargs(verdict) == {"sandbox_fallback": "landlock"}

    def test_auto_falls_to_rung_two_exactly_as_before(self, fanout_gate):
        verdict = fanout_gate(_Ladder(policy="auto", landlock=SANDBOX_BROKEN))
        assert verdict.rung == "unsandboxed" and verdict.bypass
        assert verdict.reason == rob.sandbox_bypassed_notice("fakecodex")
        assert rob.sandbox_fallback_kwargs(verdict) == {"sandbox_fallback": "unsandboxed"}

    def test_contained_takes_rung_one(self, fanout_gate):
        verdict = fanout_gate(_Ladder(policy="contained"))
        assert verdict.rung == "landlock" and not verdict.bypass

    def test_contained_never_takes_rung_two(self, fanout_gate):
        verdict = fanout_gate(_Ladder(policy="contained", landlock=SANDBOX_BROKEN))
        assert verdict.refused
        assert verdict.reason.startswith("[SANDBOX_UNAVAILABLE]")
        assert "sandbox_fallback=contained" in verdict.reason
        assert rob.sandbox_fallback_kwargs(verdict) == {}

    @pytest.mark.parametrize("landlock", [CANARY_DENIED, SANDBOX_BROKEN])
    def test_off_still_refuses(self, fanout_gate, landlock):
        verdict = fanout_gate(_Ladder(policy="off", landlock=landlock))
        assert verdict.refused and verdict.rung is None
        assert verdict.reason == rob.sandbox_unavailable_reason("fakecodex", BWRAP_LINE)
        assert rob.sandbox_fallback_kwargs(verdict) == {}

    def test_the_cache_holds_the_rung(self, fanout_gate):
        fanout_gate(_Ladder(policy="auto"))
        (cached,) = rob._FANOUT_SANDBOX_CACHE.values()
        assert cached.rung == "landlock"


# ---------------------------------------------------------------------------
# The rung survives every carrier to the argv
# ---------------------------------------------------------------------------


def _completed(stdout):
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = stdout
    proc.stderr = ""
    return proc


CODEX_STREAM = "\n".join([
    '{"type": "item.completed", "item": {"id": "item_0", "type": "error", '
    '"message": "`[features].use_legacy_landlock` is deprecated and will be '
    'removed soon."}}',
    '{"type": "item.completed", "item": {"type": "agent_message", "text": "[]"}}',
    '{"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}',
])


class TestTheRungSurvivesTheCarriers:
    def test_a_landlock_slot_renders_its_rung_and_is_never_called_bypassed(self):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["argv"] = list(cmd)
            return _completed(CODEX_STREAM)

        from utils.llm_client import invoke_with_provider

        with patch("shutil.which", return_value="/usr/bin/codex"), \
             patch("utils.llm_client.subprocess.run", side_effect=fake_run):
            result = invoke_with_provider(
                "p", "codex:gpt-5.5", **rob.sandbox_fallback_kwargs(
                    rob.SandboxVerdict(rob.SANDBOX_DISPATCH_UNSANDBOXED, "n", "landlock")
                ),
            )
        assert result["success"] is True
        assert result["details"]["sandbox"] == "landlock"
        assert result["details"]["sandbox"] != "bypassed"
        argv = seen["argv"]
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert argv.count(FLAG_VALUE) == 1

    def test_every_apply_sandbox_fallback_call_site_passes_the_rung(self, monkeypatch):
        """The four call sites: base argv leg, base agent leg, codex's own
        stdin leg, and the executor's dry-run preview."""
        calls = []
        real = CodexProvider.apply_sandbox_fallback

        def recording(self, argv, profile, rung=None):
            calls.append(rung)
            return real(self, argv, profile, rung)

        monkeypatch.setattr(CodexProvider, "apply_sandbox_fallback", recording)
        provider = CodexProvider()
        provider.build_command_for_transport(PROMPT, "gpt-5.5", "argv", sandbox_fallback="landlock")
        provider.build_command_for_transport(PROMPT, "gpt-5.5", "stdin", sandbox_fallback="landlock")
        provider.build_agent_command_for_transport(
            PROMPT, "gpt-5.5", PROFILE_AGENT_READONLY, "stdin", sandbox_fallback="landlock",
        )
        argv, refusal = executor._preview_argv(
            provider, prompt=PROMPT, model="gpt-5.5", profile=PROFILE_AGENT_READONLY,
            schema=None, sandbox_fallback="landlock",
        )
        assert refusal is None
        assert calls == ["landlock"] * 4
        assert argv[3] == "read-only" and argv.count(FLAG_VALUE) == 1

    def test_the_preflight_rung_reaches_the_spawn_as_its_id(self, monkeypatch, tmp_path):
        monkeypatch.setattr(executor, "_SANDBOX_CACHE", {})
        monkeypatch.setattr(executor, "get_provider", lambda _name: _Ladder())
        executor._detect_sandbox(
            "codex", key="k", spec="codex:gpt-5.5", profile=PROFILE_AGENT_READONLY,
        )
        rung = executor.sandbox_rung("codex:gpt-5.5", PROFILE_AGENT_READONLY)
        assert rung == "landlock"
        # …and that id, handed on, renders rung 1 — not rung 2's constant.
        argv, _ = CodexProvider().build_agent_command_for_transport(
            PROMPT, "gpt-5.5", PROFILE_AGENT_READONLY, "argv", sandbox_fallback=rung,
        )
        assert argv[3] == "read-only" and FLAG_VALUE in argv
        assert "danger-full-access" not in argv
