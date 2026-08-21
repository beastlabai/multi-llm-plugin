"""M10 (§5.1) — the typed availability status that sits *beside* the bool.

`is_available()` was a bare `shutil.which` in every adapter, which collapses
"not installed" / "logged out" / "too old" into one message. M10's answer is
**additive**: a status type (`Availability`), a report (`AvailabilityReport`)
and one new method (`describe_availability()`), with `is_available() -> bool`
keeping its signature, its name and its meaning as the binary-only launch gate.

Three things can regress, and they are the §6.1 matrix's three ship-gate tests:

* **the compatibility identity** — `is_available() == (state is not
  BINARY_MISSING)` — which is what keeps the new typed answer and the old bool
  from ever disagreeing. Parametrized over the **live registry** per M0.a, never
  over a name list or a count: the roster has grown from thirteen to seventeen
  once already, and a sweep written against a literal roster silently stops
  covering whatever shipped after it.
* **the `classify_auth_probe` → state mapping**, including its fail-open edge:
  `None` and "no probe declared" are the same amount of evidence, which is none,
  and neither may become a verdict.
* **the launch gate**, which must still admit a logged-out-but-installed CLI —
  D3c's *installed is not authenticated*. An auth or version problem surfaces at
  run time carrying the CLI's own message; it never vanishes into a preflight
  refusal.

Everything here is parametrized over `_registry_names()`. That helper is the
local twin of `test_providers._registered_provider_names()`, forced by the same
parametrize-at-class-body-time problem, and its docstring is the requirement.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import (  # noqa: E402
    AVAILABILITY_BY_STATUS,
    NAMED_UNAVAILABILITY,
    Availability,
    AvailabilityReport,
    LLMProvider,
    availability_from_failure,
)
from utils.provider_registry import _PROVIDERS  # noqa: E402


def _registry_names():
    """Every provider id in the registry, so the sweeps below cannot miss a new one.

    Resolved at class-definition time because `@pytest.mark.parametrize`
    evaluates its argument while the class body runs — the same constraint that
    forces `test_providers.py`'s local twin at its own class bodies.

    **No test in this module may name an adapter or a count** (M0.a): the
    specifications M10 came from were written against thirteen adapters and the
    registry holds seventeen, so a literal roster is under-specified rather than
    merely stale.
    """
    return sorted(_PROVIDERS)


#: A path that `shutil.which` can plausibly have returned. Its value is never
#: read — only its truthiness — so it carries no platform assumption.
_RESOLVED = "/usr/bin/some-cli"

SKILL_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Ship gate 1 — the compatibility identity, over the live registry
# ---------------------------------------------------------------------------


class TestTheCompatibilityIdentity:
    """`is_available() == describe_availability().launchable`, always.

    This is the assertion that makes M10 additive rather than a semantic change
    wearing an additive costume. It holds *by construction* — the first thing
    `describe_availability` does is ask `is_available()` and return
    `BINARY_MISSING` when, and only when, it says no — and this class is what
    fails if an adapter ever overrides that leg.
    """

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_identity_holds_when_the_binary_resolves(self, name):
        provider = _PROVIDERS[name]
        with patch("shutil.which", return_value=_RESOLVED):
            available = provider.is_available()
            report = provider.describe_availability()

        assert available is True
        assert report.state is not Availability.BINARY_MISSING
        assert available == (report.state is not Availability.BINARY_MISSING)
        assert available == report.launchable

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_identity_holds_when_the_binary_is_missing(self, name):
        provider = _PROVIDERS[name]
        with patch("shutil.which", return_value=None):
            available = provider.is_available()
            report = provider.describe_availability()

        assert available is False
        assert report.state is Availability.BINARY_MISSING
        assert available == (report.state is not Availability.BINARY_MISSING)
        assert available == report.launchable

    @pytest.mark.parametrize("name", _registry_names())
    @pytest.mark.parametrize("auth", [True, False, None])
    @pytest.mark.parametrize("version", [None, "", "unknown", "0.0.1", "9999.0.0"])
    def test_no_evidence_can_move_the_gate(self, name, auth, version):
        """No probe result, of any shape, may flip the launch gate either way.

        The point of the sweep is the *cross product*: `NEEDS_AUTH`,
        `VERSION_TOO_LOW` and `UNKNOWN` are all still launchable, and no
        combination of a clean probe and a fresh version can make a missing
        binary launchable.
        """
        provider = _PROVIDERS[name]
        for resolved, expected in ((_RESOLVED, True), (None, False)):
            with patch("shutil.which", return_value=resolved):
                available = provider.is_available()
                report = provider.describe_availability(auth=auth, version=version)

            assert available is expected
            assert report.launchable is expected
            assert available == (report.state is not Availability.BINARY_MISSING)

    @pytest.mark.parametrize("name", _registry_names())
    def test_no_adapter_overrides_the_method(self, name):
        """M10 asks nothing of the seventeen subclasses, and this pins that.

        `describe_availability` is implemented once, on the base, from data the
        adapters already declare. An override is the one way the identity above
        can be broken, so an override has to be a deliberate, visible change to
        this test rather than a quiet one in an adapter.
        """
        provider = _PROVIDERS[name]

        assert type(provider).describe_availability is LLMProvider.describe_availability


# ---------------------------------------------------------------------------
# Ship gate 2 — the classify_auth_probe → state mapping
# ---------------------------------------------------------------------------


class TestTheAuthProbeMapping:
    """The two tri-states **map, they do not merge**.

    `classify_auth_probe` keeps its `Optional[bool]` and stays the only producer
    of `NEEDS_AUTH`. The four inputs and their four states are the whole
    contract: `True → READY`, `False → NEEDS_AUTH`, `None → UNKNOWN`, and
    no-probe-declared `→ UNKNOWN`.
    """

    @pytest.mark.parametrize("name", _registry_names())
    @pytest.mark.parametrize(
        "verdict,expected",
        [
            (True, Availability.READY),
            (False, Availability.NEEDS_AUTH),
            (None, Availability.UNKNOWN),
        ],
    )
    def test_each_verdict_maps_to_its_state(self, name, verdict, expected):
        provider = _PROVIDERS[name]
        with patch("shutil.which", return_value=_RESOLVED):
            report = provider.describe_availability(auth=verdict)

        assert report.state is expected

    @pytest.mark.parametrize("name", _registry_names())
    def test_no_probe_declared_is_unknown_never_a_verdict(self, name):
        """Absence of evidence is not a negative verdict — and not a positive one.

        This is the fourth input of the mapping, and the fail-open edge. An
        adapter that declares no auth probe reports `UNKNOWN`: never
        `NEEDS_AUTH` (which would refuse a runner nothing measured) and never
        `READY` (which would manufacture a verdict the CLI never gave). The
        adapters this currently covers are read off the registry, not listed.
        """
        provider = _PROVIDERS[name]
        with patch("shutil.which", return_value=_RESOLVED):
            report = provider.describe_availability()

        assert report.state is Availability.UNKNOWN
        assert report.state is not Availability.NEEDS_AUTH
        assert report.state is not Availability.READY
        if not provider.auth_check:
            assert "no auth probe" in report.detail

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_login_notice_travels_from_the_probe_to_the_state(self, name):
        """End to end: the probe's own classifier is what produces NEEDS_AUTH.

        Asserting the mapping on a hand-written `False` would pass even if the
        two surfaces had drifted apart, so the verdict here is the one
        `classify_auth_probe` really returns for a login notice.
        """
        provider = _PROVIDERS[name]
        verdict = provider.classify_auth_probe("", "Error: not logged in.", 1)
        assert verdict is False

        with patch("shutil.which", return_value=_RESOLVED):
            report = provider.describe_availability(auth=verdict)

        assert report.state is Availability.NEEDS_AUTH

    @pytest.mark.parametrize("name", _registry_names())
    def test_an_unrecognised_non_zero_exit_is_unknown(self, name):
        """The probe's own `None` leg, carried through unchanged.

        An older CLI without the subcommand, a broken pipe and an offline probe
        all exit non-zero saying nothing about credentials. `classify_auth_probe`
        answers `None` there, and `None` may not become `NEEDS_AUTH`.
        """
        provider = _PROVIDERS[name]
        verdict = provider.classify_auth_probe("", "usage: whatever", 2)
        assert verdict is None

        with patch("shutil.which", return_value=_RESOLVED):
            report = provider.describe_availability(auth=verdict)

        assert report.state is Availability.UNKNOWN

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_clean_probe_is_ready(self, name):
        """`True` is READY for every adapter — and only `True` may be.

        Two legs, because a clean stream is not a positive verdict on every
        CLI. The base rule is "clean exit, no notice → `True`", and for most
        CLIs the exit code *is* the verdict. An adapter whose CLI was measured
        exiting **0 in every auth state** overrides `classify_auth_probe` to
        read a verdict field instead, and then a generic notice-free stream is
        honestly `None` — never `True` (which would manufacture the verdict
        M10's `READY` exists to require) and never `False` (which would refuse
        a runner nothing measured).

        So the mapping leg is asserted on `True` for every adapter, which is
        the ship gate; the classifier leg asserts only what holds registry-wide
        — a clean, notice-free probe is never a refusal, and it reaches `READY`
        exactly when the adapter really said yes. Neither leg names an adapter
        (M0.a).
        """
        provider = _PROVIDERS[name]
        verdict = provider.classify_auth_probe("logged in as someone", "", 0)
        assert verdict is not False

        with patch("shutil.which", return_value=_RESOLVED):
            declared = provider.describe_availability(auth=True)
            report = provider.describe_availability(auth=verdict)

        assert declared.state is Availability.READY
        assert report.state in (Availability.READY, Availability.UNKNOWN)
        assert (report.state is Availability.READY) is (verdict is True)

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_missing_binary_outranks_every_auth_verdict(self, name):
        """`BINARY_MISSING` is decided first and nothing downstream can rewrite it."""
        provider = _PROVIDERS[name]
        for verdict in (True, False, None):
            with patch("shutil.which", return_value=None):
                report = provider.describe_availability(auth=verdict)
            assert report.state is Availability.BINARY_MISSING

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_version_leg_is_a_label_not_a_second_gate(self, name):
        """A build below `min_version` is named, and still launchable.

        §5.1: `VERSION_TOO_LOW` borrows the enum member for a message issued by
        the preflight's own per-flag gate. It does **not** flow back into
        `is_available()`, which stays binary-only. Adapters that declare no
        floor keep whatever the auth axis said — which is the honest answer, not
        a gap.
        """
        provider = _PROVIDERS[name]
        with patch("shutil.which", return_value=_RESOLVED):
            report = provider.describe_availability(auth=True, version="0.0.1")
            available = provider.is_available()

        if (provider.min_version or "").strip():
            assert report.state is Availability.VERSION_TOO_LOW
            assert provider.min_version in report.detail
        else:
            assert report.state is Availability.READY
        assert report.launchable is True
        assert available is True, "a too-old build is a label, not a launch refusal"

    @pytest.mark.parametrize("name", _registry_names())
    @pytest.mark.parametrize("version", [None, "", "unknown"])
    def test_an_unread_version_never_produces_version_too_low(self, name, version):
        """"Not read" is not "too low" — the same absence-of-evidence rule."""
        provider = _PROVIDERS[name]
        with patch("shutil.which", return_value=_RESOLVED):
            report = provider.describe_availability(auth=True, version=version)

        assert report.state is Availability.READY


# ---------------------------------------------------------------------------
# Ship gate 3 — the launch gate
# ---------------------------------------------------------------------------


class TestTheLaunchGate:
    """A logged-out-but-installed CLI still dispatches (D3c).

    This is the assertion that a typed status did not quietly become a fourth
    refusal. Only `BINARY_MISSING` refuses; an auth or version failure must
    surface at run time carrying the CLI's own message.
    """

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_logged_out_runner_is_still_launchable(self, name):
        provider = _PROVIDERS[name]
        with patch("shutil.which", return_value=_RESOLVED):
            report = provider.describe_availability(auth=False)
            assert provider.is_available() is True

        assert report.state is Availability.NEEDS_AUTH
        assert report.launchable is True

    @pytest.mark.parametrize("name", _registry_names())
    def test_only_binary_missing_refuses(self, name):
        """Enumerated over the enum, so a new member cannot silently refuse."""
        for state in Availability:
            report = AvailabilityReport(state)
            assert report.launchable is (state is not Availability.BINARY_MISSING)

    def test_lane_a_dispatches_a_logged_out_cli_rather_than_refusing_it(self):
        """The gate in the real caller: a login notice reaches the CLI, not a refusal.

        `llm_client` is where the `ERROR_BINARY_NOT_FOUND` preflight lives. A
        logged-out runner must get past it and fail on the child's own words, so
        the assertion is that a child was **spawned** and that the error code is
        the subprocess one — not that a string appeared.
        """
        from utils.llm_client import (
            ERROR_BINARY_NOT_FOUND,
            ERROR_SUBPROCESS_FAILED,
            invoke_with_provider,
        )

        failed = MagicMock()
        failed.returncode = 1
        failed.stdout = ""
        failed.stderr = "Error: not logged in. Please run `login`."

        with patch("shutil.which", return_value=_RESOLVED):
            with patch("utils.llm_client.subprocess.run", return_value=failed) as run:
                result = invoke_with_provider(
                    prompt="Test", model_spec="cursor-agent:gpt-4"
                )

        assert run.called, "a logged-out CLI must still be dispatched"
        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert result["error_code"] != ERROR_BINARY_NOT_FOUND

    def test_lane_a_refuses_only_when_the_binary_is_missing(self):
        """The other side of the same gate, with the details now carrying the state."""
        from utils.llm_client import ERROR_BINARY_NOT_FOUND, invoke_with_provider

        with patch("shutil.which", return_value=None):
            with patch("utils.llm_client.subprocess.run") as run:
                result = invoke_with_provider(
                    prompt="Test", model_spec="cursor-agent:gpt-4"
                )

        assert not run.called, "a missing binary must not be dispatched"
        assert result["error_code"] == ERROR_BINARY_NOT_FOUND
        assert result["details"]["availability_state"] == (
            Availability.BINARY_MISSING.value
        )
        assert result["details"]["availability_detail"]


# ---------------------------------------------------------------------------
# The bool contract, asserted rather than assumed
# ---------------------------------------------------------------------------


class TestIsAvailableKeepsItsBoolContract:
    """`is_available() -> bool` did not widen, and could not have.

    The donor design M10 came from proposed returning
    `True | False | "binary-missing" | "needs-auth" | "version-too-low"` from
    this very method. That is rejected in §5.1 and this class is why: all three
    strings are truthy, so every `if provider.is_available():` in the tree would
    have started silently admitting a logged-out or too-old CLI, and every
    `if not provider.is_available():` would have stopped refusing a missing one.
    """

    def test_the_abstract_declaration_still_returns_bool(self):
        signature = inspect.signature(LLMProvider.is_available)

        assert signature.return_annotation is bool
        assert list(signature.parameters) == ["self"]

    @pytest.mark.parametrize("name", _registry_names())
    def test_every_adapter_returns_a_real_bool(self, name):
        """Not merely truthy — `bool`, on both sides of the gate."""
        provider = _PROVIDERS[name]
        for resolved in (_RESOLVED, None):
            with patch("shutil.which", return_value=resolved):
                answer = provider.is_available()
            assert isinstance(answer, bool)
            assert answer is (resolved is not None)

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_new_method_is_the_sibling_not_the_replacement(self, name):
        """Both surfaces exist, and the typed one is the additive one."""
        provider = _PROVIDERS[name]

        assert callable(provider.is_available)
        assert callable(provider.describe_availability)
        signature = inspect.signature(LLMProvider.describe_availability)
        assert signature.return_annotation is AvailabilityReport
        # Keyword-only evidence, both optional: an existing caller passes
        # nothing and gets an answer built from declared data alone.
        assert [p.kind for p in list(signature.parameters.values())[1:]] == [
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.KEYWORD_ONLY,
        ]

    @pytest.mark.parametrize(
        "relative",
        [
            "utils/llm_client.py",
            "utils/subagent_executor.py",
            "utils/provider_registry.py",
            "init_config.py",
            "utils/interactive.py",
            "utils/host_install.py",
        ],
    )
    def test_every_call_surface_still_asks_the_bool(self, relative):
        """§5.1's call-site table, walked.

        Each of these modules gates on `is_available()` and M10 left every one
        of those gates a yes/no. A site that swapped the bool for the typed
        report would be a semantic change to a dispatch decision, which is
        exactly what this task is not.
        """
        source = (SKILL_DIR / relative).read_text(encoding="utf-8")

        assert "is_available()" in source


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


class TestTheReport:
    """`AvailabilityReport` is a frozen value, and its `detail` is safe to store."""

    def test_the_report_is_frozen(self):
        report = AvailabilityReport(Availability.READY, "fine")
        with pytest.raises(Exception):
            report.state = Availability.UNKNOWN  # type: ignore[misc]

    def test_the_detail_defaults_to_empty(self):
        assert AvailabilityReport(Availability.UNKNOWN).detail == ""

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_detail_never_carries_probe_output(self, name):
        """§4.5.1's boundary, honoured by construction rather than by redaction.

        `detail` reaches `--init` output, logs and failure artifacts, so it is
        composed only from data the adapter *declares*. Feeding a probe's
        stdout/stderr through the evidence parameters must not put a byte of it
        into the sentence.
        """
        provider = _PROVIDERS[name]
        secret = "sk-live-DEADBEEFCAFE"
        noisy = (f"1.0.0 {secret}", f"{secret} 1.0.0", secret)
        with patch("shutil.which", return_value=_RESOLVED):
            for verdict in (True, False, None):
                for version in noisy:
                    report = provider.describe_availability(
                        auth=verdict, version=version
                    )
                    assert secret not in report.detail
                    assert "DEADBEEF" not in report.detail
                    assert "sk-live" not in report.detail

    def test_the_states_render_as_their_wire_strings(self):
        """A member interpolated into a message is the hyphenated word, not a repr."""
        for state in Availability:
            assert str(state) == state.value
            assert f"{state.value}" == state.value
            assert state == state.value

    def test_the_vocabulary_is_the_five_states_and_no_more(self):
        """The closed set. Adding a member is a deliberate edit here first."""
        assert {state.value for state in Availability} == {
            "ready",
            "binary-missing",
            "needs-auth",
            "version-too-low",
            "unknown",
        }


# ---------------------------------------------------------------------------
# The status-consumer sites (§5.1's two `describe_failure` rows)
# ---------------------------------------------------------------------------


class TestTheStatusConsumers:
    """A failed run's own streams can carry an availability verdict — and only one.

    §5.1: the two tri-states map, they do not merge. Exactly one run status says
    anything about whether the CLI is usable, and it is the one drawn from a
    positive login-notice match. Everything else stays silent, because a state
    appended to every failure is noise rather than diagnosis.
    """

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_login_notice_in_a_failed_run_names_the_state(self, name):
        provider = _PROVIDERS[name]

        assert (
            availability_from_failure(provider, "", "Error: not logged in.", 1)
            is Availability.NEEDS_AUTH
        )

    @pytest.mark.parametrize("name", _registry_names())
    def test_an_ordinary_failure_says_nothing(self, name):
        provider = _PROVIDERS[name]

        assert availability_from_failure(provider, "", "boom: file not found", 1) is None

    @pytest.mark.parametrize("name", _registry_names())
    def test_a_raising_classifier_is_silence_not_a_crash(self, name):
        """A diagnostic may never become the reason a failure cannot be reported."""
        provider = _PROVIDERS[name]
        with patch.object(
            type(provider), "classify_failure", side_effect=RuntimeError("boom")
        ):
            assert availability_from_failure(provider, "", "whatever", 1) is None

    def test_only_named_negatives_are_ever_returned(self):
        """`READY` and `UNKNOWN` are not failure diagnoses and never surface here."""
        assert set(AVAILABILITY_BY_STATUS.values()) <= set(NAMED_UNAVAILABILITY)
        assert Availability.READY not in NAMED_UNAVAILABILITY
        assert Availability.UNKNOWN not in NAMED_UNAVAILABILITY

    def test_lane_a_appends_the_state_to_a_logged_out_failure(self):
        from utils.llm_client import invoke_with_provider

        failed = MagicMock()
        failed.returncode = 1
        failed.stdout = ""
        failed.stderr = "Error: not logged in. Please run `login`."

        with patch("shutil.which", return_value=_RESOLVED):
            with patch("utils.llm_client.subprocess.run", return_value=failed):
                result = invoke_with_provider(
                    prompt="Test", model_spec="cursor-agent:gpt-4"
                )

        assert "exited with code 1" in result["error"]
        assert f"[{Availability.NEEDS_AUTH.value}]" in result["error"]

    def test_lane_a_leaves_an_ordinary_failure_alone(self):
        """No state word on a failure that carries no availability verdict."""
        from utils.llm_client import invoke_with_provider

        failed = MagicMock()
        failed.returncode = 1
        failed.stdout = ""
        failed.stderr = "Command failed: invalid arguments"

        with patch("shutil.which", return_value=_RESOLVED):
            with patch("utils.llm_client.subprocess.run", return_value=failed):
                result = invoke_with_provider(
                    prompt="Test", model_spec="cursor-agent:gpt-4"
                )

        assert "exited with code 1" in result["error"]
        for state in Availability:
            assert f"[{state.value}]" not in result["error"]


# ---------------------------------------------------------------------------
# The remaining call surfaces §5.1 tabulates
# ---------------------------------------------------------------------------


class TestTheAnnotatingCallSurfaces:
    """The sites that render a sentence now have one to render."""

    def test_init_selection_is_derived_from_the_typed_walk(self):
        """`--init`'s bool selection *is* `describe_availability().launchable`.

        One walk, two halves: the selection and the annotation cannot disagree
        about the same CLI because they are computed from the same reports.
        """
        import init_config

        config = {name: {} for name in _registry_names()}
        rows = init_config._availability_rows(config)

        assert [name for name, _ in rows] == _registry_names()
        assert init_config._available_providers(config) == [
            name for name, report in rows if report.launchable
        ]

    def test_init_annotates_an_installed_but_unusable_runner(self):
        """The status half: "detected" only ever meant "the binary resolves"."""
        import init_config

        provider = _PROVIDERS[_registry_names()[0]]
        name = provider.name
        with patch.object(
            type(provider),
            "describe_availability",
            return_value=AvailabilityReport(
                Availability.NEEDS_AUTH, "it is logged out."
            ),
        ):
            lines = init_config._unusable_notices([name])

        assert lines == [f"{name}: it is logged out."]

    def test_init_never_re_reports_a_missing_binary_as_unusable(self):
        """A row in the detected list is one whose binary resolved, by definition."""
        import init_config

        provider = _PROVIDERS[_registry_names()[0]]
        with patch.object(
            type(provider),
            "describe_availability",
            return_value=AvailabilityReport(Availability.BINARY_MISSING, "gone."),
        ):
            assert init_config._unusable_notices([provider.name]) == []

    def test_the_interactive_warning_can_finally_say_why(self):
        """`Provider 'X' is not available` gained the half it could never say."""
        from utils import interactive

        provider = _PROVIDERS[_registry_names()[0]]
        with patch("shutil.which", return_value=None):
            why = interactive._availability_reason(provider.name)

        assert why
        assert "not on PATH" in why

    def test_the_interactive_warning_is_silent_on_an_unknown_provider(self):
        from utils import interactive

        assert interactive._availability_reason("definitely-not-a-provider") == ""

    def test_the_binary_absent_refusal_names_which_binary(self):
        """§5.1: the refusal stays a bool; `_render_runner_rejection`'s text gains the state."""
        from utils import subagent_executor as se

        name = _registry_names()[0]
        with patch("shutil.which", return_value=None):
            rejection = se._render_runner_rejection(
                key="driver.subagent_runner",
                spec=f"{name}:some-model",
                reason_code="binary_absent",
                profile="agent-readonly",
            )

        assert rejection.reason_code == "binary_absent"
        assert "not on PATH" in str(rejection)

    def test_the_refusal_detail_is_silence_for_an_unresolvable_spec(self):
        from utils import subagent_executor as se

        assert se._availability_detail("definitely-not-a-provider:m") == ""
        assert se._availability_detail("") == ""

    def test_row_five_reports_the_state_beside_auth_and_version(self):
        """`install-host --check` already ran both probes; the state costs nothing."""
        from utils import host_install as hi

        name = _registry_names()[0]
        with patch("shutil.which", return_value=_RESOLVED):
            ready = hi._runner_state(name, {"auth": "ok", "version": "", "problem": ""})
            unknown = hi._runner_state(
                name, {"auth": "unknown", "version": "", "problem": ""}
            )
            logged_out = hi._runner_state(
                name,
                {
                    "auth": "unknown",
                    "version": "",
                    "problem": "Refusing to dispatch: x (reason: not_authenticated)",
                },
            )

        assert ready == Availability.READY.value
        assert unknown == Availability.UNKNOWN.value
        assert logged_out == Availability.NEEDS_AUTH.value

    def test_row_five_state_is_silence_for_an_unknown_provider(self):
        from utils import host_install as hi

        assert hi._runner_state("definitely-not-a-provider", {}) == ""
