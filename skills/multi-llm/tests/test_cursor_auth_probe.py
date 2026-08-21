#!/usr/bin/env python3
"""M13 (§5.2, §5.3) — cursor-agent's real auth probe, and `about`'s demotion.

The donor plans proposed `cursor-agent about --format json` as one call doing
four jobs. Measured 2026-08-16 it does two, and **auth is not one of them**:
`about`'s `userEmail` and `subscriptionTier` round-trip the network, so a
logged-in HOME pointed at an unreachable endpoint returns them both `null` —
byte-identical to a logged-out scratch HOME, at exit 0 in all six states
measured. Keying auth on `userEmail == null` reports a laptop in flight mode as
logged out, which is the defect this module exists to keep out.

Three ship gates, and they are the §6.1 matrix's M13 row:

1. **The probe separates offline-logged-in from logged-out.** Those are
   different states; a probe that conflates them is the defect. Only
   `status --format json`'s local token field tells them apart, so the verdict
   is a JSON field or nothing — never the exit code, which is 0 either way.
2. **Redaction at the probe boundary, over the nine-key envelope.** Three of
   `about`'s nine keys are §4.5.1 P13 identity fields. They are replaced once,
   where the envelope is read, rather than at each place it might be rendered;
   the other six survive, or a redacted artifact stops being usable.
3. **The availability row is upgraded.** M10's rule is that `READY` requires a
   *positive* verdict and is never inferred from silence, so an adapter
   declaring no probe reports `UNKNOWN` — which is what this adapter did until
   this task. It now reaches `READY` and `NEEDS_AUTH` from a real CLI answer.

The registry-wide sweep at the bottom (§5.3's install/login/status metadata)
parametrizes over the live registry per M0.a: no test below names a roster or
an adapter count. The cursor-specific tests name cursor-agent because it is the
adapter under test, which is the one place a name is not a roster.

Plain CI: no container, no Docker, no network, no subprocess.
"""

import json
import sys
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import (  # noqa: E402
    Availability,
    SetupHints,
)
from utils.providers.cursor_agent import (  # noqa: E402
    CURSOR_ABOUT_ARGS,
    CURSOR_ABOUT_IDENTITY_KEYS,
    CURSOR_ABOUT_KEYS,
    CURSOR_STATUS_ARGS,
    CursorAbout,
    CursorAgentProvider,
    read_about,
    redact_probe_output,
    status_auth_verdict,
)
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.write_safety import redact_text  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parents[1]

#: Any non-None `shutil.which` result: `is_available()` is a bare `which` in
#: every adapter, and these tests are about the *auth* leg, not about whether
#: the machine running CI happens to have cursor-agent installed.
_RESOLVED = "/usr/bin/cursor-agent"

PROVIDER = CursorAgentProvider()


# ---------------------------------------------------------------------------
# The measured envelopes (2026-08-16). Values are synthetic; shapes are not.
# ---------------------------------------------------------------------------

#: Logged in, network reachable.
STATUS_ONLINE = json.dumps(
    {
        "isAuthenticated": True,
        "userEmail": "ada@example.com",
        "message": "Logged in",
    }
)

#: Logged in, endpoint unreachable — the state `about` cannot see. The message
#: is the measured one.
STATUS_OFFLINE_LOGGED_IN = json.dumps(
    {
        "isAuthenticated": True,
        "message": "Logged in (unable to fetch user details)",
    }
)

#: Logged out.
STATUS_LOGGED_OUT = json.dumps(
    {"status": "unauthenticated", "message": "Not logged in"}
)

#: What a build without the flag prints: stderr, empty stdout, exit 1, and
#: commander.js wording — never clap's `unexpected argument`.
STATUS_NO_FLAG_STDERR = "error: unknown option '--format'"


def _about(**overrides):
    """The nine-key `about --format json` envelope, keyed off the declaration.

    Built from :data:`CURSOR_ABOUT_KEYS` rather than from a second literal list,
    so a key added to the adapter's declaration and not to this fixture is a
    failure here rather than a silently unredacted field.
    """
    values = {
        "cliVersion": "2026.08.11-e8db854",
        "model": "GPT-5.6 Sol 272K Extra High",
        "subscriptionTier": "pro",
        "osPlatform": "linux",
        "osArch": "x64",
        "userEmail": "ada@example.com",
        "terminalProgram": "iTerm.app",
        "shell": "/bin/zsh",
        "lastRequestId": "req_01HQ8Z4K2M",
    }
    assert set(values) == set(CURSOR_ABOUT_KEYS), (
        "the fixture and the adapter's declared envelope keys have drifted"
    )
    values.update(overrides)
    return json.dumps(values)


#: Offline: the two network-backed fields come back null. This is the envelope
#: that is indistinguishable from a logged-out HOME.
ABOUT_OFFLINE = _about(userEmail=None, subscriptionTier=None)


# ---------------------------------------------------------------------------
# Gate 1 — the probe separates offline-logged-in from logged-out
# ---------------------------------------------------------------------------


class TestTheProbeSeparatesOfflineLoggedInFromLoggedOut:
    """The two states the donor names, and the ways they can be conflated."""

    def test_the_declared_probe_is_status_not_about(self):
        assert PROVIDER.auth_check == CURSOR_STATUS_ARGS
        assert "status" in PROVIDER.auth_check
        assert "about" not in PROVIDER.auth_check
        assert PROVIDER.auth_check != CURSOR_ABOUT_ARGS

    def test_offline_logged_in_is_authenticated(self):
        """Logged in with the endpoint unreachable is **logged in**."""
        assert PROVIDER.classify_auth_probe(STATUS_OFFLINE_LOGGED_IN, "", 0) is True

    def test_logged_out_is_not(self):
        assert PROVIDER.classify_auth_probe(STATUS_LOGGED_OUT, "", 0) is False

    def test_the_two_states_do_not_collapse(self):
        """Stated as the inequality, so a probe that conflates them is red.

        This is the whole task in one line: a single classifier returning the
        same answer for both would pass every other test in this class.
        """
        offline = PROVIDER.classify_auth_probe(STATUS_OFFLINE_LOGGED_IN, "", 0)
        out = PROVIDER.classify_auth_probe(STATUS_LOGGED_OUT, "", 0)
        assert offline is not out
        assert {offline, out} == {True, False}

    def test_online_and_offline_logged_in_agree(self):
        """Reachability is a network fact, never an auth fact."""
        assert PROVIDER.classify_auth_probe(
            STATUS_ONLINE, "", 0
        ) is PROVIDER.classify_auth_probe(STATUS_OFFLINE_LOGGED_IN, "", 0)

    @pytest.mark.parametrize("exit_code", [0, 1, 2, 127])
    def test_the_exit_code_carries_no_verdict(self, exit_code):
        """cursor-agent exited 0 in all six auth states measured (§5.3).

        So the verdict must move with the *field* and not with the exit code,
        in both directions.
        """
        assert PROVIDER.classify_auth_probe(
            STATUS_OFFLINE_LOGGED_IN, "", exit_code
        ) is True
        assert PROVIDER.classify_auth_probe(STATUS_LOGGED_OUT, "", exit_code) is False

    def test_a_clean_exit_with_no_field_is_never_ready(self):
        """The base rule reads a clean exit as authenticated; here it may not.

        Silence is not a positive verdict for this CLI — that is precisely the
        rule M10's `UNKNOWN` state exists to hold.
        """
        assert PROVIDER.classify_auth_probe("", "", 0) is None
        assert PROVIDER.classify_auth_probe("Cursor Agent", "", 0) is None

    def test_a_build_without_the_flag_is_unknown_not_logged_out(self):
        """Exit ≠ 0 plus a decode failure is the feature-sniff, not a refusal."""
        assert PROVIDER.classify_auth_probe("", STATUS_NO_FLAG_STDERR, 1) is None

    def test_a_plain_text_login_notice_is_still_recognised(self):
        """Leg 2: the shared `auth_error_patterns` over a build with no JSON."""
        notice = (
            "Error: Authentication required. Please run 'agent login' first."
        )
        assert PROVIDER.classify_auth_probe("", notice, 1) is False

    def test_the_verdict_is_a_tri_state_and_nothing_else(self):
        """No probe text escapes through the classifier's return value."""
        for stdout, stderr, code in (
            (STATUS_ONLINE, "", 0),
            (STATUS_LOGGED_OUT, "", 0),
            ("", STATUS_NO_FLAG_STDERR, 1),
        ):
            verdict = PROVIDER.classify_auth_probe(stdout, stderr, code)
            assert verdict is True or verdict is False or verdict is None

    def test_the_function_and_the_method_agree(self):
        """The module-level reader is what the adapter method delegates to."""
        for payload in (STATUS_ONLINE, STATUS_OFFLINE_LOGGED_IN, STATUS_LOGGED_OUT):
            assert status_auth_verdict(payload, "", 0) == PROVIDER.classify_auth_probe(
                payload, "", 0
            )

    def test_the_envelope_may_arrive_on_stderr(self):
        """Some builds banner to stderr; the verdict must not depend on which."""
        assert PROVIDER.classify_auth_probe("", STATUS_LOGGED_OUT, 0) is False
        assert PROVIDER.classify_auth_probe("", STATUS_OFFLINE_LOGGED_IN, 0) is True


# ---------------------------------------------------------------------------
# Gate 1b — `about` is demoted, and the demotion is executable
# ---------------------------------------------------------------------------


class TestAboutIsDemoted:
    """`about` keeps the two jobs it does and loses the two it cannot do."""

    def test_about_cannot_tell_the_two_states_apart(self):
        """The measurement, as a test: offline-logged-in ≡ logged-out here.

        A logged-out scratch HOME produces the same two nulls an unreachable
        endpoint does, so anything keyed on them is keyed on nothing.
        """
        logged_out_looking = _about(userEmail=None, subscriptionTier=None)
        assert ABOUT_OFFLINE == logged_out_looking
        offline = read_about(ABOUT_OFFLINE, "", 0)
        scratch = read_about(logged_out_looking, "", 0)
        assert offline == scratch

    def test_the_readout_has_no_field_that_could_carry_a_verdict(self):
        """Structural, not conventional: there is nowhere to put an auth flag."""
        names = {f.name for f in fields(CursorAbout)}
        assert names == {"version", "plan_tier", "parsed_json", "redacted"}
        assert not any("auth" in name for name in names)
        assert not any("login" in name for name in names)

    def test_the_default_model_is_never_read_back(self):
        """`about`'s `model` is a last-used display label, not a `--model` slug."""
        assert "model" not in {f.name for f in fields(CursorAbout)}
        about = read_about(_about(), "", 0)
        assert "GPT-5.6" not in about.version
        assert "GPT-5.6" not in about.plan_tier

    def test_the_two_jobs_it_does_deliver(self):
        about = read_about(_about(), "", 0)
        assert about.parsed_json is True
        assert about.version == "2026.08.11-e8db854"
        assert about.plan_tier == "pro"

    def test_the_tier_is_empty_offline_and_that_is_not_an_auth_signal(self):
        about = read_about(ABOUT_OFFLINE, "", 0)
        assert about.version == "2026.08.11-e8db854"
        assert about.plan_tier == ""

    def test_a_clean_run_with_no_json_is_not_guessed_at(self):
        """Exit 0 and no object: the sniff needs *both* halves, so nothing is read."""
        about = read_about("Cursor Agent", "", 0)
        assert about.parsed_json is False
        assert about.version == ""
        assert about.plan_tier == ""

    def test_the_text_fallback_runs_only_on_a_failed_run(self):
        """§5.2's fallback: exit ≠ 0 **plus** a decode failure, then `Key  Value`."""
        text = "Version        2026.08.11-e8db854\nSubscription   pro\n"
        about = read_about(text, STATUS_NO_FLAG_STDERR, 1)
        assert about.parsed_json is False
        assert about.version == "2026.08.11-e8db854"
        assert about.plan_tier == "pro"

    def test_the_version_probe_is_not_wired_to_about(self):
        """The demotion holds: ``about`` is not, and never became, the version probe.

        M13 recorded this as ``version_cmd == ()`` — the honest shape while the
        shared version gate did not exist and M13 declined to open a second,
        unowned one. **M14 opened the owned one**, and measured the probe it
        needed rather than reaching for the envelope M13 had just demoted:
        ``cursor-agent --version`` exits 0 and prints ``2026.08.11-e8db854`` on
        stdout, so that is what the adapter declares.

        What the assertion is really defending is unchanged and is now stated
        directly: the version read is the CLI's own ``--version``, never
        ``about``. Wiring `about` in — the one thing M13's demotion removed —
        still fails here.
        """
        assert PROVIDER.version_cmd == ("cursor-agent", "--version")
        assert "about" not in PROVIDER.version_cmd, (
            "the version read must not be wired back to the `about` envelope: "
            "that is the demotion M13 landed"
        )
        # And it is a *version* read, not a second auth probe wearing one.
        assert PROVIDER.version_cmd[0] == PROVIDER.auth_check[0]
        assert PROVIDER.version_cmd != PROVIDER.auth_check


# ---------------------------------------------------------------------------
# Gate 2 — redaction at the probe boundary, over the nine-key envelope
# ---------------------------------------------------------------------------


class TestTheNineKeyEnvelopeIsRedactedAtTheProbeBoundary:
    """§4.5.1 P13, applied where the envelope is read rather than where it is shown."""

    IDENTITY_VALUES = ("ada@example.com", "iTerm.app", "/bin/zsh")
    SURVIVING_VALUES = (
        "2026.08.11-e8db854",
        "GPT-5.6 Sol 272K Extra High",
        "pro",
        "linux",
        "x64",
        "req_01HQ8Z4K2M",
    )

    def test_the_envelope_really_has_nine_keys(self):
        """The donor plans said five; the CLI returns nine, and three are PII."""
        assert len(CURSOR_ABOUT_KEYS) == 9
        assert len(set(CURSOR_ABOUT_KEYS)) == 9
        assert CURSOR_ABOUT_IDENTITY_KEYS <= set(CURSOR_ABOUT_KEYS)
        assert len(CURSOR_ABOUT_IDENTITY_KEYS) == 3

    def test_every_identity_value_is_gone_at_the_boundary(self):
        out = read_about(_about(), "", 0).redacted
        for value in self.IDENTITY_VALUES:
            assert value not in out, f"{value!r} survived the probe boundary"

    def test_every_other_value_survives(self):
        """A redacted artifact that lost its version and platform is not usable."""
        out = read_about(_about(), "", 0).redacted
        for value in self.SURVIVING_VALUES:
            assert value in out, f"{value!r} was redacted and should not have been"

    def test_the_keys_survive_so_the_artifact_stays_diffable(self):
        out = read_about(_about(), "", 0).redacted
        for key in CURSOR_ABOUT_KEYS:
            assert key in out
        assert "<redacted:" in out

    def test_the_boundary_and_the_shared_redactor_agree(self):
        """One redaction implementation (§4.5), reached from the adapter."""
        envelope = _about()
        assert redact_probe_output(envelope) == redact_text(envelope)
        assert read_about(envelope, "", 0).redacted == redact_text(envelope)

    def test_the_boundary_is_idempotent(self):
        once = redact_probe_output(_about())
        assert redact_probe_output(once) == once

    def test_a_status_envelope_carrying_an_email_is_redacted_too(self):
        """The auth probe reads the same identity fields; same boundary."""
        out = redact_probe_output(STATUS_ONLINE)
        assert "ada@example.com" not in out
        assert "isAuthenticated" in out

    def test_the_verdict_path_carries_no_envelope_text_at_all(self):
        """The strongest form: a tri-state cannot leak a field."""
        verdict = PROVIDER.classify_auth_probe(STATUS_ONLINE, "", 0)
        assert verdict is True
        assert "ada@example.com" not in repr(verdict)

    def test_an_embedded_log_line_is_redacted_the_same_way(self):
        line = f"cursor-agent about --format json -> {_about()}"
        out = redact_probe_output(line)
        for value in self.IDENTITY_VALUES:
            assert value not in out
        assert "cursor-agent about" in out


# ---------------------------------------------------------------------------
# Gate 3 — the availability row M10 could only ever report as UNKNOWN
# ---------------------------------------------------------------------------


class TestTheAvailabilityRowIsUpgraded:
    """M10 ships the states; M13 is what lets this adapter reach two of them."""

    def _report(self, stdout, stderr="", exit_code=0, provider=PROVIDER):
        verdict = provider.classify_auth_probe(stdout, stderr, exit_code)
        with patch("shutil.which", return_value=_RESOLVED):
            return verdict, provider.describe_availability(auth=verdict)

    def test_offline_logged_in_now_reaches_ready(self):
        verdict, report = self._report(STATUS_OFFLINE_LOGGED_IN)
        assert verdict is True
        assert report.state is Availability.READY
        assert report.launchable is True

    def test_logged_out_now_reaches_needs_auth_and_names_the_fix(self):
        verdict, report = self._report(STATUS_LOGGED_OUT)
        assert verdict is False
        assert report.state is Availability.NEEDS_AUTH
        assert PROVIDER.login_cmd in report.detail
        # D3c: installed is not authenticated, and needs-auth still dispatches.
        assert report.launchable is True

    def test_an_inconclusive_probe_still_fails_open(self):
        verdict, report = self._report("", STATUS_NO_FLAG_STDERR, 1)
        assert verdict is None
        assert report.state is Availability.UNKNOWN

    def test_no_probe_output_reaches_the_rendered_detail(self):
        """`AvailabilityReport.detail` is declared data only (§4.5.1)."""
        _verdict, report = self._report(STATUS_ONLINE)
        assert "ada@example.com" not in report.detail

    def test_without_the_declaration_the_row_could_only_say_unknown(self):
        """The before/after, as one comparison rather than as a claim.

        M10's rule is that `READY` requires a positive verdict and is never
        inferred from silence, so an adapter declaring no probe is `UNKNOWN` by
        construction. That was this adapter's row until the declaration above
        existed — asserted here against a copy of it with the declaration
        removed, so the proof does not depend on any other adapter.
        """
        class _Undeclared(CursorAgentProvider):
            auth_check = ()

        undeclared = _Undeclared()
        with patch("shutil.which", return_value=_RESOLVED):
            before = undeclared.describe_availability()
        assert before.state is Availability.UNKNOWN
        assert "no auth probe" in before.detail

        _verdict, after = self._report(STATUS_OFFLINE_LOGGED_IN)
        assert after.state is Availability.READY

    def test_the_probe_is_declared_and_never_spawned_from_the_adapter(self):
        """The shared D3c preflight owns the spawn; the adapter owns the argv.

        A probe run from inside the adapter would turn a `shutil.which` in a
        picker loop into a fan-out of subprocesses, which is exactly what
        `describe_availability`'s "it never spawns a child" rule forbids.
        """
        import ast

        path = SKILL_DIR / "utils/providers/cursor_agent.py"
        source = path.read_text(encoding="utf-8")
        probes = {"CURSOR_STATUS_ARGS", "CURSOR_ABOUT_ARGS"}
        spawned = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            rendered = ast.dump(node.func)
            if not any(
                token in rendered
                for token in ("subprocess", "Popen", "system", "check_output")
            ):
                continue
            args = ast.dump(ast.Module(body=[ast.Expr(node)], type_ignores=[]))
            spawned += [name for name in probes if f"id='{name}'" in args]
        assert not spawned, f"the adapter spawns its own probe: {spawned}"

        head, _, tail = source.partition("class CursorAgentProvider")
        assert "CURSOR_STATUS_ARGS: Tuple[str, ...] = (" in head
        assert "auth_check: Tuple[str, ...] = CURSOR_STATUS_ARGS" in tail


# ---------------------------------------------------------------------------
# §5.3's install / login / status metadata — over the registry (M0.a)
# ---------------------------------------------------------------------------


def _registry_names():
    """Every provider id in the registry, so this sweep cannot miss a new one.

    Resolved at class-definition time because `@pytest.mark.parametrize`
    evaluates its argument while the class body runs — the same constraint that
    forces `test_providers.py`'s and `test_availability.py`'s local twins.

    **No test in this section may name an adapter or a count** (M0.a): §5.3's
    metadata rows were specified for thirteen adapters and the registry holds
    more, so a sweep written against a literal roster silently stops covering
    whatever shipped after it.
    """
    return sorted(_PROVIDERS)


class TestTheSetupMetadataIsDeclaredByEveryAdapter:
    """§5.3's four rows, derived from one owner each."""

    @pytest.mark.parametrize("name", _registry_names())
    def test_every_adapter_reports_the_four_rows(self, name):
        hints = _PROVIDERS[name].describe_setup()
        assert isinstance(hints, SetupHints)
        assert isinstance(hints.install_hint, str)
        assert isinstance(hints.auth_hint, str)
        assert isinstance(hints.login_args, tuple)
        assert isinstance(hints.status_args, tuple)

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_status_row_is_the_argv_the_preflight_really_runs(self, name):
        """Derived, never re-declared: one owner per fact.

        A second copy of the probe argv is a copy that can disagree with the
        one `subagent_executor._detect_auth` spawns, and then the metadata
        describes a probe nobody runs.
        """
        provider = _PROVIDERS[name]
        assert provider.describe_setup().status_args == tuple(provider.auth_check)

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_login_row_is_the_command_a_refusal_names(self, name):
        provider = _PROVIDERS[name]
        assert provider.describe_setup().login_args == tuple(
            provider.login_cmd.split()
        )

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_declared_probe_comes_with_the_fix_for_its_refusal(self, name):
        """A `False` verdict that names no repair is a dead end for a human."""
        hints = _PROVIDERS[name].describe_setup()
        if hints.status_args:
            assert hints.login_args, (
                "this adapter declares an auth probe but no login command, so a "
                "recognised login notice would refuse the run without saying "
                "how to fix it"
            )

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_declared_probe_is_a_status_subcommand_not_a_dispatch(self, name):
        """§5.3: probe by running the CLI's own status subcommand.

        The tokens rejected here are the ones that make an invocation a *turn* —
        a probe that dispatches costs a billed request every preflight.
        """
        hints = _PROVIDERS[name].describe_setup()
        if not hints.status_args:
            return
        launcher = hints.status_args[0]
        assert launcher and not launcher.startswith("-")
        assert "/" not in launcher and "\\" not in launcher, (
            "the probe names a path rather than a binary on PATH"
        )
        assert not any(
            token in hints.status_args for token in ("exec", "run", "-p", "--prompt")
        )

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_declared_install_hint_is_printable(self, name):
        """It is rendered into a one-line refusal, so it must be one line."""
        hint = _PROVIDERS[name].describe_setup().install_hint
        if hint:
            assert "\n" not in hint
            assert hint == hint.strip()

    def test_no_install_hint_is_ever_executed(self):
        """The trust boundary §5.3 names, asserted over this repo's own source.

        `install_hint` is a sentence for a human. A module that reads it into a
        subprocess call would be an auto-installer, which is the disposition
        M16 was declined on.
        """
        offenders = []
        for path in sorted((SKILL_DIR / "utils").rglob("*.py")):
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if "install_hint" not in line:
                    continue
                if any(
                    token in line
                    for token in ("subprocess", "Popen", "os.system", "shell=True")
                ):
                    offenders.append(f"{path.relative_to(SKILL_DIR)}:{number}")
        assert not offenders, f"install_hint reaches a spawn at {offenders}"

    def test_the_hints_carry_declared_data_only(self):
        """Safe to print into `--init` output, a log or a failure artifact.

        Asserted the only way it can be: every field is either empty or equal
        to the class attribute it was derived from, so nothing from an
        environment, a probe or a credential can appear in one.
        """
        for name in _registry_names():
            provider = _PROVIDERS[name]
            hints = provider.describe_setup()
            assert hints.install_hint == (provider.install_hint or "").strip()
            assert hints.auth_hint == (provider.auth_hint or "").strip()
            assert " ".join(hints.login_args) == " ".join(
                (provider.login_cmd or "").split()
            )

    def test_this_adapters_rows_are_populated(self):
        """The M13 row itself: all four, for the adapter this module is about."""
        hints = PROVIDER.describe_setup()
        assert hints.status_args == CURSOR_STATUS_ARGS
        assert hints.login_args == ("cursor-agent", "login")
        assert hints.install_hint
        # The measured credential path, not the redirected profile dir:
        # `auth.json` follows neither CURSOR_CONFIG_DIR nor CURSOR_DATA_DIR.
        assert "auth.json" in hints.auth_hint
        assert "$XDG_CONFIG_HOME/cursor/auth.json" in hints.auth_hint


# ---------------------------------------------------------------------------
# The readout is *executed* — the half of `about`'s demotion that was missing
# ---------------------------------------------------------------------------


class TestTheReadoutIsExecutedByTheStatusPath:
    """A demoted probe still has to run, or its two remaining jobs go undone.

    `about` was correctly kept out of the auth path, and then out of every
    other path too: the argv and :func:`read_about` existed, were tested, and
    were called by nothing that ships — so the plan tier, the one fact no other
    probe reports, reached no surface. The wiring is a declaration
    (:attr:`CursorAgentProvider.setup_probe`) plus one spawner on the *status*
    path (``install-host --check`` row 5, through
    ``subagent_executor.probe_setup_metadata``), and every test below is
    written so that removing either half fails.
    """

    def test_the_readout_is_declared_and_is_neither_other_probe(self):
        assert PROVIDER.setup_probe == CURSOR_ABOUT_ARGS
        assert PROVIDER.setup_probe != PROVIDER.auth_check
        assert PROVIDER.setup_probe != PROVIDER.version_cmd
        # It is a readout, not a turn: the tokens that make an invocation a
        # billed dispatch are the ones §5.3 rejects for a probe.
        assert not any(
            token in PROVIDER.setup_probe
            for token in ("exec", "run", "-p", "--prompt")
        )

    def test_the_reader_yields_the_plan_tier(self):
        assert PROVIDER.read_setup_probe(_about(), "", 0) == {"plan-tier": "pro"}

    def test_the_version_is_left_to_the_version_probe(self):
        """One fact, one owner: row 5's version comes from `--version` (M14)."""
        facts = PROVIDER.read_setup_probe(_about(), "", 0)
        assert "2026.08.11-e8db854" not in " ".join(facts.values())
        assert not any("version" in key for key in facts)

    def test_an_unreadable_readout_is_silence_and_never_a_verdict(self):
        """Offline, logged out, and a build without the flag all report nothing.

        The three states differ, and none of them is an auth verdict — which is
        why they must all render as *one less fact on the row* rather than as
        anything a caller could branch on.
        """
        assert PROVIDER.read_setup_probe(ABOUT_OFFLINE, "", 0) == {}
        assert PROVIDER.read_setup_probe("", STATUS_NO_FLAG_STDERR, 1) == {}
        assert PROVIDER.read_setup_probe("", "", 0) == {}

    def test_nothing_but_the_tier_crosses_the_boundary(self):
        """The other eight keys, identity fields included, have no path out."""
        rendered = " ".join(
            f"{k}={v}" for k, v in PROVIDER.read_setup_probe(_about(), "", 0).items()
        )
        for value in (
            "ada@example.com", "iTerm.app", "/bin/zsh",
            "GPT-5.6 Sol 272K Extra High", "req_01HQ8Z4K2M", "linux", "x64",
        ):
            assert value not in rendered

    def test_a_surprising_tier_cannot_print_whatever_it_likes(self):
        """The value is CLI output bound for a terminal and a log."""
        facts = PROVIDER.read_setup_probe(
            _about(subscriptionTier="pro\nsecret: sk-" + "z" * 80), "", 0
        )
        tier = facts["plan-tier"]
        assert "\n" not in tier and ":" not in tier
        assert len(tier) <= 32
        assert tier.startswith("pro")

    def test_the_status_path_really_spawns_it(self):
        """The defect, as a test: something in production runs the argv."""
        from types import SimpleNamespace

        from utils import subagent_executor as se

        spawned = []

        def _fake(argv):
            spawned.append(list(argv))
            return SimpleNamespace(stdout=_about(), stderr="", returncode=0)

        with patch.object(se, "_run_probe", _fake):
            facts = se.probe_setup_metadata("cursor-agent")

        assert spawned == [list(CURSOR_ABOUT_ARGS)]
        assert facts == {"plan-tier": "pro"}

    def test_a_readout_that_will_not_run_reports_nothing(self):
        from utils import subagent_executor as se

        with patch.object(se, "_run_probe", lambda argv: None):
            assert se.probe_setup_metadata("cursor-agent") == {}

    def test_an_adapter_declaring_no_readout_spawns_nothing(self):
        """Empty is the default, and the default must cost no subprocess."""
        from utils import subagent_executor as se

        spawned = []
        silent = [
            name for name in _registry_names()
            if not getattr(_PROVIDERS[name], "setup_probe", ())
        ]
        assert silent, "every adapter declares a readout; this sweep is dead"
        with patch.object(
            se, "_run_probe", lambda argv: spawned.append(list(argv))
        ):
            for name in silent:
                assert se.probe_setup_metadata(name) == {}
        assert spawned == []

    def test_the_readout_is_never_spawned_by_a_dispatch_preflight(self):
        """It decides nothing, so no phase may pay a subprocess for it."""
        from utils import subagent_executor as se

        source = Path(se.__file__).read_text(encoding="utf-8")
        body = source.partition("def preflight_runner")[2].partition("\ndef ")[0]
        assert "setup_probe" not in body
        assert "probe_setup_metadata" not in body

    def test_the_check_path_carries_it_end_to_end(self):
        """`--check` row 5 → the executor's spawner → this adapter's reader.

        The chain, with only the spawn faked, so a break anywhere along it —
        the declaration, the spawner, the reader, or the key row 5 renders —
        fails here rather than showing up as a status row that quietly lost a
        fact.
        """
        from types import SimpleNamespace

        from utils import host_install as hi
        from utils import subagent_executor as se

        with patch.object(
            se, "preflight_runner",
            lambda *a, **kw: {"auth": "ok", "version": "2026.08.11"},
        ), patch.object(
            se, "_run_probe",
            lambda argv: SimpleNamespace(stdout=_about(), stderr="", returncode=0),
        ):
            facts = hi.probe_runner("cursor-agent:gpt-5.6-sol-xhigh")

        assert facts["metadata"] == "plan-tier=pro"
        assert facts["auth"] == "ok"

    def test_the_row_five_verdict_ignores_the_readout(self):
        """Context for a verdict, never an input to one (M10's rule)."""
        from utils import host_install as hi

        with patch("shutil.which", return_value=_RESOLVED):
            without = hi._runner_state(
                "cursor-agent", {"auth": "ok", "version": "", "problem": ""}
            )
            with_tier = hi._runner_state(
                "cursor-agent",
                {
                    "auth": "ok", "version": "", "problem": "",
                    "metadata": "plan-tier=pro",
                },
            )
        assert without == with_tier == Availability.READY.value
