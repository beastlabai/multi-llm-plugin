"""codex 0.156.1 retired the landlock rung — make the loss loud (plan §3,
2026-09-24).

Guards, in both lanes (the fan-out gate `sandbox_refusal` and the executor
preflight `_detect_sandbox`):

* 3.1 — the `[SANDBOX_BYPASSED]` line keeps its fixed stem and appends the
  rung clause (each rung tried, its outcome, its probe tail), on a plan's
  **first** run with a cold cache, and a cache hit prints the same tail;
* 3.2 — at or above `removed_at` rung 1 is not probed at all and the
  retirement line is printed once per run; below it the probe's outcome is
  authoritative, and only the rung's own removal panic earns the line;
* 3.3 — that panic is classified **by text**, never by exit code;
* 3.6 — the two lanes share one probe cache: one rung probe per version.
"""

import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import review_orchestrator_base as rob  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_GENERATE,
    SANDBOX_CANARY_DENIED,
    SANDBOX_FALLBACK_PROFILES,
    SANDBOX_RUNG_RETIRED_MARKER,
    SandboxRung,
)
from utils.providers.codex import (  # noqa: E402
    LANDLOCK_GLOBAL_PANIC,
    LANDLOCK_GLOBAL_PANIC_REASON,
    LANDLOCK_NEEDS_BWRAP_PANIC,
    LANDLOCK_NEEDS_BWRAP_REASON,
    LANDLOCK_RUNG,
    UNSANDBOXED_RUNG,
    CodexProvider,
)

BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
SANDBOX_BROKEN = f"echo '{BWRAP_LINE}' >&2; exit 1"
CANARY_DENIED = f"echo '{SANDBOX_CANARY_DENIED}'; exit 0"
PANIC_LINE = (
    "thread 'main' panicked at linux-sandbox/src/linux_run_main.rs:410:5: "
    + LANDLOCK_NEEDS_BWRAP_PANIC
)
#: codex 0.156.1's rung-1 probe, as measured 2026-09-24: exit 101 and the panic.
RUNG_PANICS = f"echo \"{PANIC_LINE}\" >&2; exit 101"
#: Exit 101 without the sentence: not the retirement, whatever the exit says.
EXIT_101_OTHER = "echo 'thread main panicked: something else' >&2; exit 101"
UNRELATED_ERROR = "echo 'error: permission denied opening socket' >&2; exit 1"
HANGS = "sleep 5"
SPEC = "fakecodex:m"


def _rung(script):
    return SandboxRung(
        name=LANDLOCK_RUNG.name,
        probe_argv=("sh", "-c", script),
        argv_rewrite=LANDLOCK_RUNG.argv_rewrite,
        contained=True,
        deprecated_at=LANDLOCK_RUNG.deprecated_at,
        removed_at=LANDLOCK_RUNG.removed_at,
        removed_signature=LANDLOCK_RUNG.removed_signature,
    )


class _Codexish:
    """codex's shape: a broken profile probe, the two-rung ladder with a
    scripted rung-1 probe, and a `version_cmd` answering ``version``."""

    internal_retries = None

    def __init__(self, version="0.155.1", landlock=RUNG_PANICS, policy="auto"):
        self.version = version
        self.landlock = landlock
        self.policy = policy
        self.version_cmd = (
            ["sh", "-c", f"echo 'codex-cli {version}'"] if version else []
        )

    def parse_version(self, stdout, stderr):
        return CodexProvider().parse_version(stdout, stderr)

    def sandbox_probe_command(self, profile):
        return ["sh", "-c", SANDBOX_BROKEN]

    def sandbox_fallback_policy(self, profile, anchor=None):
        return self.policy if profile in SANDBOX_FALLBACK_PROFILES else None

    def sandbox_fallback_rungs(self, profile):
        if profile not in SANDBOX_FALLBACK_PROFILES:
            return ()
        return (_rung(self.landlock), UNSANDBOXED_RUNG)


@pytest.fixture
def lanes(monkeypatch):
    """Both gates bound to one stand-in adapter, the shared cache clean, and
    every sandbox spawn counted."""
    spawned = []
    real = executor._run_sandbox_probe

    def counting(argv, **kwargs):
        spawned.append(list(argv))
        return real(argv, **kwargs)

    monkeypatch.setattr(executor, "_run_sandbox_probe", counting)
    executor.reset_sandbox_cache()
    executor._VERSION_CACHE.pop("fakecodex", None)

    class Lanes:
        adapter = None
        probes = spawned

        def use(self, adapter):
            self.adapter = adapter
            monkeypatch.setattr(rob, "get_provider", lambda _name: adapter)
            monkeypatch.setattr(executor, "get_provider", lambda _name: adapter)
            return self

        def fanout(self):
            # The repo root the executor derives from its anchor — the cwd a
            # real fan-out pins its dispatch (and its probe) to.
            return rob.sandbox_refusal("fakecodex", executor._probe_cwd(None))

        def executor(self, spec=SPEC):
            return executor._detect_sandbox(
                "fakecodex", key="driver.subagent_runner", spec=spec,
                profile=PROFILE_AGENT_READONLY,
            )

        def rung_probes(self):
            return [argv for argv in spawned if argv[-1] == self.adapter.landlock]

    yield Lanes()
    executor.reset_sandbox_cache()


def _retired_lines(text):
    return [line for line in text.splitlines() if SANDBOX_RUNG_RETIRED_MARKER in line]


# ---------------------------------------------------------------------------
# 3.1 — the bypass line names the rung that was tried
# ---------------------------------------------------------------------------


class TestTheBypassLineNamesTheRungTried:
    def test_a_first_run_with_a_cold_cache_and_no_prior_record(self, lanes, capsys):
        lanes.use(_Codexish(version="0.155.1", landlock=UNRELATED_ERROR))
        verdict = lanes.fanout()
        assert verdict.bypass
        assert verdict.reason == (
            rob.sandbox_bypassed_notice("fakecodex")
            + "; fallback rung 'landlock' unavailable (fakecodex 0.155.1: "
            "\"error: permission denied opening socket\")"
        )
        # No prior record at all, and still no silence: the per-slot line is
        # the one the transition notice used to be the only carrier of.
        assert rob.announce_retired_rungs(None, {SPEC: {"sandbox": "bypassed"}}) == []

    def test_a_cache_hit_prints_the_same_probe_tail(self, lanes):
        lanes.use(_Codexish(version="0.155.1", landlock=UNRELATED_ERROR))
        first = lanes.fanout()
        second = lanes.fanout()
        assert second.reason == first.reason
        assert "permission denied opening socket" in second.reason
        assert len(lanes.rung_probes()) == 1

    def test_the_executor_prints_the_same_clause(self, lanes, capsys):
        lanes.use(_Codexish(version="0.155.1", landlock=UNRELATED_ERROR))
        assert lanes.executor() == "bypassed"
        out = capsys.readouterr().out
        (line,) = [ln for ln in out.splitlines() if "[SANDBOX_BYPASSED]" in ln]
        assert line == (
            "[exec-subagents] " + rob.sandbox_bypassed_notice("fakecodex")
            + "; fallback rung 'landlock' unavailable (fakecodex 0.155.1: "
            "\"error: permission denied opening socket\")"
        )


# ---------------------------------------------------------------------------
# 3.2 — removed_at skips the probe; below it the outcome is authoritative
# ---------------------------------------------------------------------------


class TestAtOrAboveRemovedAt:
    @pytest.mark.parametrize("version", ["0.156.1", "0.157.0"])
    def test_the_fanout_spawns_no_rung_probe_and_says_so_once(
        self, lanes, capsys, version
    ):
        lanes.use(_Codexish(version=version, landlock=CANARY_DENIED))
        first = lanes.fanout()
        second = lanes.fanout()
        assert lanes.rung_probes() == []
        assert first.walk == (("landlock", "skipped"), ("unsandboxed", "taken"))
        assert first.bypass and second.reason == first.reason
        assert first.reason.endswith(
            "; fallback rung 'landlock' not tried: unavailable from fakecodex "
            "0.156.1, measured"
        )
        capsys.readouterr()
        served = {SPEC: {"sandbox": "bypassed"}, "fakecodex:n": {"sandbox": "bypassed"}}
        # Printed from the version alone — no prior record needed.
        lines = rob.announce_retired_rungs(None, served)
        assert len(lines) == 1
        (line,) = lines
        assert line.startswith(SANDBOX_RUNG_RETIRED_MARKER)
        assert "fakecodex:m, fakecodex:n" in line
        assert "measured unavailable from fakecodex 0.156.1" in line
        assert f"this is fakecodex {version}" in line
        assert _retired_lines(capsys.readouterr().out) == [line]

    def test_the_executor_spawns_no_rung_probe_and_says_so_once(self, lanes, capsys):
        lanes.use(_Codexish(version="0.156.1", landlock=CANARY_DENIED))
        assert lanes.executor(SPEC) == "bypassed"
        assert lanes.executor("fakecodex:n") == "bypassed"
        assert lanes.rung_probes() == []
        out = capsys.readouterr().out
        assert "not tried: unavailable from fakecodex 0.156.1, measured" in out
        assert len(_retired_lines(out)) == 1

    def test_an_unread_version_probes_and_the_outcome_is_authoritative(
        self, lanes, capsys
    ):
        lanes.use(_Codexish(version="", landlock=RUNG_PANICS))
        verdict = lanes.fanout()
        assert len(lanes.rung_probes()) == 1
        assert verdict.walk[0] == ("landlock", "retired")
        lines = rob.announce_retired_rungs(None, {SPEC: {"sandbox": "bypassed"}})
        assert len(lines) == 1 and "measured unavailable from fakecodex 0.156.1" in lines[0]


class TestUnderContained:
    """`sandbox_fallback: contained` refuses the slot — and still says why."""

    def test_both_lanes_name_the_skipped_rung_and_the_executor_announces_once(
        self, lanes, capsys
    ):
        lanes.use(_Codexish(version="0.156.1", landlock=CANARY_DENIED, policy="contained"))
        verdict = lanes.fanout()
        assert verdict.refused
        clause = "fallback rung 'landlock' not tried: unavailable from fakecodex 0.156.1, measured"
        assert clause in verdict.reason
        for spec in (SPEC, "fakecodex:n"):
            with pytest.raises(executor.RunnerRejected) as excinfo:
                lanes.executor(spec)
            assert clause in str(excinfo.value)
        (line,) = _retired_lines(capsys.readouterr().out)
        assert "measured unavailable from fakecodex 0.156.1" in line
        assert "fell to the uncontained rung" not in line
        assert "this run refused fakecodex:m" in line


class TestBetweenDeprecatedAtAndRemovedAt:
    def test_a_passing_probe_uses_the_rung_and_announces_nothing(self, lanes, capsys):
        lanes.use(_Codexish(version="0.155.1", landlock=CANARY_DENIED))
        verdict = lanes.fanout()
        assert verdict.rung == "landlock" and not verdict.bypass
        assert lanes.executor() == "landlock"
        assert rob.announce_retired_rungs(None, {SPEC: {"sandbox": "fallback-landlock"}}) == []
        assert _retired_lines(capsys.readouterr().out) == []

    @pytest.mark.parametrize("lane", ["fanout", "executor"])
    def test_the_removal_panic_earns_the_retirement_line(self, lanes, capsys, lane):
        lanes.use(_Codexish(version="0.155.1", landlock=RUNG_PANICS))
        if lane == "fanout":
            verdict = lanes.fanout()
            assert verdict.bypass
            assert "requires bubblewrap in this fakecodex build" in verdict.reason
            assert LANDLOCK_NEEDS_BWRAP_PANIC in verdict.reason
            lines = rob.announce_retired_rungs(None, {SPEC: {"sandbox": "bypassed"}})
            assert len(lines) == 1
        else:
            assert lanes.executor() == "bypassed"
            assert len(_retired_lines(capsys.readouterr().out)) == 1

    @pytest.mark.parametrize("script, outcome", [
        (UNRELATED_ERROR, "unavailable"),
        (EXIT_101_OTHER, "unavailable"),
    ])
    def test_any_other_failure_is_named_as_itself(self, lanes, capsys, script, outcome):
        lanes.use(_Codexish(version="0.155.1", landlock=script))
        verdict = lanes.fanout()
        assert verdict.walk[0] == ("landlock", outcome)
        assert f"fallback rung 'landlock' {outcome}" in verdict.reason
        assert "not tried" not in verdict.reason
        assert rob.announce_retired_rungs(None, {SPEC: {"sandbox": "bypassed"}}) == []
        assert lanes.executor() == "bypassed"
        assert _retired_lines(capsys.readouterr().out) == []

    def test_a_timeout_is_named_as_itself(self, lanes, monkeypatch, capsys):
        monkeypatch.setattr(executor, "SANDBOX_PROBE_TIMEOUT_S", 0.5)
        lanes.use(_Codexish(version="0.155.1", landlock=HANGS))
        verdict = lanes.fanout()
        assert verdict.refused
        assert "'landlock' was unprobed" in verdict.reason
        with pytest.raises(executor.RunnerRejected) as excinfo:
            lanes.executor()
        assert "'landlock' was unprobed" in str(excinfo.value)
        assert _retired_lines(capsys.readouterr().out) == []

    def test_a_canary_setup_failure_is_named_as_itself(self, lanes, monkeypatch, capsys):
        monkeypatch.setattr(executor, "_prepare_probe_workdir", lambda cwd, workdir: None)
        adapter = _Codexish(version="0.155.1", landlock=CANARY_DENIED)
        rung = replace(_rung(CANARY_DENIED), probe_workdir=".multi-llm/tmp")
        monkeypatch.setattr(
            adapter, "sandbox_fallback_rungs",
            lambda profile: (rung, UNSANDBOXED_RUNG) if profile in SANDBOX_FALLBACK_PROFILES else (),
        )
        lanes.use(adapter)
        verdict = lanes.fanout()
        assert verdict.refused
        assert "'landlock' was unknown" in verdict.reason
        assert rob.announce_retired_rungs(None, {SPEC: {"sandbox": "bypassed"}}) == []
        assert _retired_lines(capsys.readouterr().out) == []


# ---------------------------------------------------------------------------
# 3.3 — the panic is its own outcome, told apart by text
# ---------------------------------------------------------------------------


class TestThePanicIsClassifiedByText:
    def test_describe_failure_names_the_rung_panic_apart_from_the_global_one(self):
        codex = CodexProvider()
        assert codex.describe_failure("", PANIC_LINE) == LANDLOCK_NEEDS_BWRAP_REASON
        assert "landlock rung requires bubblewrap in this codex build" in (
            LANDLOCK_NEEDS_BWRAP_REASON
        )
        assert codex.describe_failure("", LANDLOCK_GLOBAL_PANIC) == LANDLOCK_GLOBAL_PANIC_REASON
        assert codex.describe_failure("", "thread main panicked: other") != (
            LANDLOCK_NEEDS_BWRAP_REASON
        )

    def test_the_sentence_wrapped_across_lines_still_matches(self):
        wrapped = PANIC_LINE.replace(" requires ", "\n  requires ")
        assert CodexProvider().describe_failure("", wrapped) == LANDLOCK_NEEDS_BWRAP_REASON

    @pytest.mark.parametrize("script, outcome", [
        (RUNG_PANICS, "retired"),
        (f"echo \"{PANIC_LINE}\" >&2; exit 1", "retired"),
        (EXIT_101_OTHER, "unavailable"),
    ])
    def test_the_walk_reads_the_text_not_the_exit_code(self, script, outcome):
        tails = []
        rung, walk = executor._walk_fallback_rungs(
            (_rung(script), UNSANDBOXED_RUNG), cwd=None,
            deadline=time.monotonic() + 5, tails=tails,
        )
        assert walk[0] == ("landlock", outcome)
        if outcome == "retired":
            assert tails[0] == ("landlock", PANIC_LINE)


# ---------------------------------------------------------------------------
# 3.6 — one probe cache for both lanes
# ---------------------------------------------------------------------------


class TestOneProbePerVersion:
    def test_both_lanes_in_one_process_spawn_exactly_one_rung_probe(self, lanes):
        lanes.use(_Codexish(version="0.155.1", landlock=UNRELATED_ERROR))
        assert lanes.fanout().bypass
        assert lanes.executor() == "bypassed"
        assert lanes.fanout().bypass
        assert len(lanes.rung_probes()) == 1
        # One version read too: the preflight's own, or the probe's.
        assert sum(1 for argv in lanes.probes if "codex-cli" in argv[-1]) == 1

    def test_a_new_version_is_a_new_probe(self, lanes):
        lanes.use(_Codexish(version="0.155.1", landlock=UNRELATED_ERROR))
        lanes.fanout()
        executor._PROBE_VERSION_CACHE.clear()
        lanes.use(_Codexish(version="0.155.2", landlock=UNRELATED_ERROR))
        lanes.fanout()
        assert len(lanes.rung_probes()) == 2

    def test_concurrent_specs_read_the_version_once(self, lanes):
        import threading

        adapter = _Codexish(version="0.155.1")
        adapter.version_cmd = ["sh", "-c", "sleep 0.3; echo 'codex-cli 0.155.1'"]
        results = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    executor._sandbox_cli_version("fakecodex", adapter)
                )
            )
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == ["0.155.1"] * 3
        assert sum(1 for argv in lanes.probes if "codex-cli" in argv[-1]) == 1

    def test_a_preflight_unknown_is_settled(self, lanes):
        executor._VERSION_CACHE["fakecodex"] = "unknown"
        try:
            assert executor._sandbox_cli_version("fakecodex", _Codexish()) == ""
            assert lanes.probes == []
        finally:
            executor._VERSION_CACHE.pop("fakecodex", None)

    def test_the_fanout_cache_is_gone(self):
        assert not hasattr(rob, "_FANOUT_SANDBOX_CACHE")
        assert not hasattr(executor, "_SANDBOX_CACHE")

    def test_a_real_codex_version_read_goes_through_the_guarded_runner(self, monkeypatch):
        """The version read uses the sandbox probe's runner, so the suite's
        guard on real CLIs answers it and no real codex is spawned."""
        seen = []
        monkeypatch.setattr(
            executor, "_run_sandbox_probe",
            lambda argv, **kwargs: seen.append(list(argv)) or subprocess.CompletedProcess(
                argv, 0, "codex-cli 0.156.1\n", ""
            ),
        )
        executor.reset_sandbox_cache()
        executor._VERSION_CACHE.pop("codex", None)
        assert executor._sandbox_cli_version("codex", CodexProvider()) == "0.156.1"
        assert seen and seen[0][0] == "codex"
        executor.reset_sandbox_cache()
