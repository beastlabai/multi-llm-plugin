#!/usr/bin/env python3
"""Probe output quoted into a refusal is redacted at that boundary (§4.5).

Three of ``subagent_executor``'s exit-3 refusals tail a preflight probe's own
stdout/stderr into their message so a human can see *why* the runner was
refused. Those messages are printed to a terminal and to
``orchestrator-run.log``; they reach **neither** of §4.5.3's redacting write
sinks (``subagent_batch_report.json``'s ``_redact_json``, ``failure_summary``'s
``_redact``), because a ``RunnerRejected`` is raised before any batch report
exists. They were the one path that quoted a probe verbatim.

M13 made that concrete: ``cursor-agent status --format json`` answers with an
identity-bearing envelope whose nine keys include ``userEmail``,
``terminalProgram`` and ``shell``. The adapter redacts it at its own boundary
(``providers.cursor_agent.redact_probe_output``), but that boundary is on the
*adapter's* readout — the preflight tails the raw ``CompletedProcess`` from the
other direction and never passes through it.

The standard these tests hold the fix to is M13's, not "drop the tail":

* the three identity **values** go;
* the other six values survive, so the refusal still says which build, which
  plan tier and which platform was refused;
* every **key name** survives, so two runs stay diffable;
* redaction runs **before** the 500-byte tail, or a cut landing inside
  ``"userEmail"`` leaves a value the pattern can no longer match;
* a redactor that raises drops the text (``<redacted:probe-output>``) rather
  than falling back to the raw stream.
"""

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor  # noqa: E402
from utils.providers.cursor_agent import (  # noqa: E402
    CURSOR_ABOUT_IDENTITY_KEYS,
    CURSOR_ABOUT_KEYS,
    redact_probe_output,
)
from utils.retry_ladder import STATUS_AUTH_FAILED  # noqa: E402
from utils.write_safety import redact_text  # noqa: E402

# ---------------------------------------------------------------------------
# The envelope. Values are synthetic; the shape and the key set are not.
# ---------------------------------------------------------------------------

#: The nine keys, and the three that are §4.5.1 P13 identity fields, are read
#: from the adapter's declaration rather than re-listed here: a key added there
#: and not here would otherwise be a silently unredacted field.
_ENVELOPE_VALUES = {
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

assert set(_ENVELOPE_VALUES) == set(CURSOR_ABOUT_KEYS), (
    "the fixture and the adapter's declared envelope keys have drifted"
)

#: The identity values that must never reach a message.
IDENTITY_VALUES = tuple(
    _ENVELOPE_VALUES[key] for key in sorted(CURSOR_ABOUT_IDENTITY_KEYS)
)

PROBE_ENVELOPE = json.dumps(_ENVELOPE_VALUES)

#: The measured three-key `status` envelope of a logged-in CLI — the shape the
#: D3c preflight really reads, and the one that carries an address on the
#: shortest possible stream.
STATUS_LOGGED_IN = json.dumps(
    {
        "isAuthenticated": True,
        "userEmail": "ada@example.com",
        "message": "Logged in",
    }
)

#: A network denial, so the third refusal can be reached with the same envelope
#: riding along on the stream. Multi-word on purpose: `providers.base.matches_any`
#: requires an error-context word near a *bare* token, and the point of this
#: fixture is the refusal, not the classifier.
NETWORK_DENIAL = "Error: connect to api.cursor.sh failed: connection refused"


def assert_diagnostic_and_clean(message: str) -> None:
    """The one standard all three refusals are held to (M13's).

    Asserted as whole rendered ``"key": "value"`` pairs rather than as bare
    values: ``"pro"`` is a substring of half the English language, and a test
    that passes on an accident is not a test.
    """
    for value in IDENTITY_VALUES:
        assert value not in message, (
            f"the refusal leaked the identity value {value!r}"
        )
    for key in sorted(CURSOR_ABOUT_IDENTITY_KEYS):
        assert f'"{key}": "<redacted:identity-json>"' in message, (
            f"{key!r} was dropped rather than redacted in place"
        )
    for key, value in _ENVELOPE_VALUES.items():
        if key in CURSOR_ABOUT_IDENTITY_KEYS:
            continue
        assert f'"{key}": "{value}"' in message, (
            f"redaction gutted the refusal: {key!r} is diagnostic, not identity"
        )


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _Provider:
    """The narrow surface the two refusal paths ask a provider for."""

    auth_check = ("cursor-agent", "status", "--format", "json")
    login_cmd = "cursor-agent login"

    def __init__(self, *, auth_verdict=False, failure=None):
        self._auth_verdict = auth_verdict
        self._failure = failure

    def classify_auth_probe(self, stdout, stderr, exit_code):
        return self._auth_verdict

    def classify_failure(self, stdout, stderr, exit_code):
        return self._failure


@pytest.fixture(autouse=True)
def _clean_preflight_caches():
    """Both refusals are cached per process; a stale entry would skip them."""
    subagent_executor._AUTH_CACHE.clear()
    subagent_executor._NESTED_CACHE.clear()
    yield
    subagent_executor._AUTH_CACHE.clear()
    subagent_executor._NESTED_CACHE.clear()


def _capture(stdout="", stderr=""):
    capture = subagent_executor._Capture()
    capture.stdout = stdout
    capture.stderr = stderr
    capture.exit_code = 1
    return capture


def _detect_auth_refusal(monkeypatch, stream, *, on_stderr=False):
    """Drive `_detect_auth` to its `not_authenticated` refusal."""
    monkeypatch.setattr(
        subagent_executor, "get_provider", lambda name: _Provider()
    )
    completed = subprocess.CompletedProcess(
        args=list(_Provider.auth_check),
        returncode=1,
        stdout="" if on_stderr else stream,
        stderr=stream if on_stderr else "",
    )
    monkeypatch.setattr(subagent_executor, "_run_probe", lambda argv: completed)
    with pytest.raises(subagent_executor.RunnerRejected) as excinfo:
        subagent_executor._detect_auth(
            "cursor-agent",
            key="subagent_runner",
            spec="cursor-agent:auto",
            profile="agent-readonly",
        )
    assert excinfo.value.reason_code == "not_authenticated"
    return str(excinfo.value)


def _nested_refusal(monkeypatch, stream, *, reason_code):
    """Drive `_probe_nested_exec` to its credentials or network refusal."""
    failure = STATUS_AUTH_FAILED if reason_code == "nested_exec_credentials" else None
    monkeypatch.setattr(
        subagent_executor,
        "get_provider",
        lambda name: _Provider(failure=failure),
    )
    monkeypatch.setattr(
        subagent_executor, "_spawn", lambda **kwargs: _capture(stderr=stream)
    )
    with pytest.raises(subagent_executor.RunnerRejected) as excinfo:
        subagent_executor._probe_nested_exec(
            "cursor-agent",
            "auto",
            key="subagent_runner",
            spec="cursor-agent:auto",
            host_id="codex",
            legs=("credentials",),
            notes="",
        )
    assert excinfo.value.reason_code == reason_code
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# Gate 1 — the helper itself
# ---------------------------------------------------------------------------


class TestTheProbeTailHelper:
    """`_redact_probe_tail` is the boundary the three refusals now share."""

    def test_identity_values_go_and_the_other_six_stay(self):
        tail = subagent_executor._redact_probe_tail(PROBE_ENVELOPE, 5000)
        assert_diagnostic_and_clean(tail)

    def test_it_agrees_with_the_adapter_boundary_on_the_same_envelope(self):
        """Three boundaries, one rule — M13's readout and this one must match.

        Compared on the *untailed* text: the only difference between them is
        that this one also caps, and a cap is not a redaction decision.
        """
        assert subagent_executor._redact_probe_tail(
            PROBE_ENVELOPE, 5000
        ) == redact_probe_output(PROBE_ENVELOPE).strip()
        assert subagent_executor._redact_probe_tail(
            PROBE_ENVELOPE, 5000
        ) == redact_text(PROBE_ENVELOPE).strip()

    def test_redaction_runs_before_the_cap(self):
        """§4.5 before §4.4, at the cap the refusals really use.

        The stream is sized so the 500-byte tail lands *inside* ``"userEmail"``.
        Tailing first would hand the pattern ``serEmail": "ada@example.com"`` —
        no opening quote, no key, no match — and the address would survive into
        the refusal.
        """
        cut_at = PROBE_ENVELOPE.index("Email") + 2
        prefix = "z" * (500 - (len(PROBE_ENVELOPE) - cut_at))
        stream = prefix + PROBE_ENVELOPE
        assert len(stream) > 500

        naive = subagent_executor._tail(stream, 500)
        assert "ada@example.com" in naive, "the fixture no longer straddles the cut"

        tail = subagent_executor._redact_probe_tail(stream, 500)
        assert "ada@example.com" not in tail
        assert "<redacted:identity-json>" in tail

    def test_a_raising_redactor_drops_the_text_rather_than_leaking_it(
        self, monkeypatch
    ):
        def _boom(_text):
            raise RuntimeError("write_safety is unavailable")

        monkeypatch.setattr(subagent_executor, "_redact", _boom)
        tail = subagent_executor._redact_probe_tail(PROBE_ENVELOPE, 500)
        assert tail == "<redacted:probe-output>"
        for value in IDENTITY_VALUES:
            assert value not in tail

    def test_empty_input_is_empty_output_not_a_redaction_marker(self):
        assert subagent_executor._redact_probe_tail("", 500) == ""
        assert subagent_executor._redact_probe_tail(None, 500) == ""


# ---------------------------------------------------------------------------
# Gate 2 — the three refusals that quote a probe
# ---------------------------------------------------------------------------


class TestTheNotAuthenticatedRefusal:
    """`_detect_auth` — D3c's batch-wide exit 3."""

    def test_the_nine_key_envelope_is_redacted_in_place(self, monkeypatch):
        message = _detect_auth_refusal(monkeypatch, PROBE_ENVELOPE)
        assert_diagnostic_and_clean(message)

    def test_the_refusal_still_names_the_probe_and_the_repair(self, monkeypatch):
        """Redaction may not cost the message its two actionable halves."""
        message = _detect_auth_refusal(monkeypatch, PROBE_ENVELOPE)
        assert "cursor-agent status --format json" in message
        assert "cursor-agent login" in message
        assert "not_authenticated" in message
        assert "Nothing was dispatched." in message

    def test_the_measured_status_envelope_loses_only_the_address(
        self, monkeypatch
    ):
        message = _detect_auth_refusal(monkeypatch, STATUS_LOGGED_IN)
        assert "ada@example.com" not in message
        assert "userEmail" in message
        assert "isAuthenticated" in message
        assert "Logged in" in message

    def test_a_stderr_only_probe_is_redacted_too(self, monkeypatch):
        """The site reads `stderr or stdout`; both are provider output."""
        message = _detect_auth_refusal(
            monkeypatch, PROBE_ENVELOPE, on_stderr=True
        )
        assert_diagnostic_and_clean(message)

    def test_a_raising_redactor_does_not_fall_back_to_the_raw_stream(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            subagent_executor,
            "_redact",
            lambda _text: (_ for _ in ()).throw(RuntimeError("no redactor")),
        )
        message = _detect_auth_refusal(monkeypatch, PROBE_ENVELOPE)
        assert "<redacted:probe-output>" in message
        for value in IDENTITY_VALUES:
            assert value not in message
        for key in CURSOR_ABOUT_KEYS:
            assert f'"{key}"' not in message, (
                "a failed redactor must drop the tail, not emit it"
            )
        # The refusal itself is unharmed: it still names the fix.
        assert "cursor-agent login" in message


class TestTheNestedExecCredentialsRefusal:
    """`_probe_nested_exec` — the throwaway dispatch's credentials leg."""

    def test_the_nine_key_envelope_is_redacted_in_place(self, monkeypatch):
        message = _nested_refusal(
            monkeypatch, PROBE_ENVELOPE, reason_code="nested_exec_credentials"
        )
        assert_diagnostic_and_clean(message)

    def test_the_refusal_keeps_its_host_trailer_and_repair(self, monkeypatch):
        message = _nested_refusal(
            monkeypatch, PROBE_ENVELOPE, reason_code="nested_exec_credentials"
        )
        assert "Host: codex" in message
        assert "cursor-agent login" in message
        assert "Nothing was dispatched." in message

    def test_a_raising_redactor_does_not_fall_back_to_the_raw_stream(
        self, monkeypatch
    ):
        # `_classify_failed_run` is stubbed rather than driven: it has its own
        # `_redact` call, whose message reaches only §4.5.3's redacting sinks,
        # and letting a raising redactor abort *there* would test a different
        # site than the refusal boundary this asserts.
        monkeypatch.setattr(
            subagent_executor,
            "_classify_failed_run",
            lambda provider, capture, model="": (STATUS_AUTH_FAILED, "stubbed"),
        )
        monkeypatch.setattr(
            subagent_executor,
            "_redact",
            lambda _text: (_ for _ in ()).throw(RuntimeError("no redactor")),
        )
        message = _nested_refusal(
            monkeypatch, PROBE_ENVELOPE, reason_code="nested_exec_credentials"
        )
        assert "<redacted:probe-output>" in message
        for value in IDENTITY_VALUES:
            assert value not in message
        for key in CURSOR_ABOUT_KEYS:
            assert f'"{key}"' not in message


class TestTheNestedExecNetworkRefusal:
    """`_probe_nested_exec` — the egress leg, same tail, same boundary."""

    def test_the_envelope_riding_the_denial_is_redacted(self, monkeypatch):
        stream = f"{NETWORK_DENIAL}\n{PROBE_ENVELOPE}"
        message = _nested_refusal(
            monkeypatch, stream, reason_code="nested_exec_network"
        )
        assert_diagnostic_and_clean(message)
        assert NETWORK_DENIAL in message, (
            "the denial itself is the whole reason this refusal quotes a stream"
        )

    def test_a_raising_redactor_does_not_fall_back_to_the_raw_stream(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            subagent_executor,
            "_redact",
            lambda _text: (_ for _ in ()).throw(RuntimeError("no redactor")),
        )
        stream = f"{NETWORK_DENIAL}\n{PROBE_ENVELOPE}"
        message = _nested_refusal(
            monkeypatch, stream, reason_code="nested_exec_network"
        )
        assert "<redacted:probe-output>" in message
        for value in IDENTITY_VALUES:
            assert value not in message
        for key in CURSOR_ABOUT_KEYS:
            assert f'"{key}"' not in message


# ---------------------------------------------------------------------------
# Gate 3 — no refusal in this module tails a stream without the boundary
# ---------------------------------------------------------------------------


#: The only ``_tail(...)`` call in ``subagent_executor`` allowed to be handed
#: an unredacted stream, keyed by ``(enclosing function, ``ast.unparse``
#: rendering)`` and valued by how many times that exact site occurs. It builds
#: ``outcome.reason`` in ``dispatch_task``, which is written through §4.5.3's
#: redacting sinks — ``BatchReport.write``'s ``_redact_json`` and
#: ``failure_summary``'s ``_redact`` — never straight to a terminal, which is
#: what separates it from the three refusals above. An entry earns its place by
#: naming the sink that redacts it; a new bare tail with no such sink is the
#: leak this module exists to stop.
#:
#: The ``after_result`` payload's ``stderr_tail`` used to share this entry, on
#: the same justification. It did not hold: that value leaves the module through
#: a caller's callback, so there is no sink to name — it was safe only because
#: the one shipped callback ignores the key, and a future one that logged its
#: context would have leaked a raw stream with this guard still green. It goes
#: through ``_redact_probe_tail`` now, which is why the count below is 1.
#:
#: Both halves of the key are load-bearing, and so is the count. A *text*
#: allowlist would wave through a second ``_tail(capture.stderr, 2000)``
#: anywhere in the module — including one interpolated into a user-facing
#: message — with no test change at all, and the enclosing name alone would
#: wave through a different tail added to the same function.
_TAIL_WITHOUT_REDACTION_ALLOWLIST = {
    ("dispatch_task", "_tail(capture.stderr, 2000)"): 1,
}


class TestNoRefusalQuotesAStreamRaw:
    """A source guard, so a fourth site cannot land unredacted.

    The three refusals were found by reading, not by a failing test, which is
    exactly the way a fourth one would arrive. This walks the module's AST
    instead of its text: every ``_tail`` call whose first argument is not itself
    a redaction call has to be on the allowlist above, with the sink that covers
    it written down beside it.
    """

    @staticmethod
    def _tail_calls_without_redaction():
        """``(lineno, enclosing function, rendering)`` per bare ``_tail`` call.

        The enclosing name is half a site's identity: the rendering on its own
        cannot tell two allowlisted calls apart from a third that happens to be
        spelled the same way.
        """
        source = Path(subagent_executor.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        # `ast.walk` is breadth-first, so an inner `def` is visited after the
        # one enclosing it and its assignment wins: every node ends up owned by
        # the *nearest* function around it.
        owner_of = {}
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                owner_of[id(node)] = function.name
        found = []
        for node in ast.walk(tree):
            # `_redact_probe_tail` is the boundary itself: its own `_tail` runs
            # on text `_redact` has already returned, one statement earlier,
            # which no single-expression rule can see.
            owner = owner_of.get(id(node), "<module>")
            if owner == "_redact_probe_tail":
                continue
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "_tail"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            redacted = (
                isinstance(first, ast.Call)
                and isinstance(first.func, ast.Name)
                and first.func.id.startswith("_redact")
            )
            if not redacted:
                found.append((node.lineno, owner, ast.unparse(node)))
        return found

    def test_every_bare_tail_is_a_declared_and_covered_one(self):
        offenders = [
            f"{subagent_executor.__file__}:{lineno}: {owner}: {text}"
            for lineno, owner, text in self._tail_calls_without_redaction()
            if (owner, text) not in _TAIL_WITHOUT_REDACTION_ALLOWLIST
        ]
        assert not offenders, (
            "these tail a stream with no redaction; if the value reaches a "
            "message or a terminal, route it through `_redact_probe_tail`, and "
            "if it reaches a redacting sink, add it to the allowlist with the "
            "sink named: " + "; ".join(offenders)
        )

    def test_the_allowlist_pins_each_declared_site_exactly(self):
        """Each entry still matches, and matches exactly as often as declared.

        A subset check over rendered text can see neither half of the drift it
        exists to catch: a third identically-spelled tail rides in on the
        permission the first two earned, and deleting one of them leaves the
        entry matching its surviving twin. Counting per site turns both red,
        which is what makes adding a tail site a deliberate edit here.
        """
        live = Counter(
            (owner, text)
            for _lineno, owner, text in self._tail_calls_without_redaction()
        )
        drifted = {
            site: (declared, live.get(site, 0))
            for site, declared in _TAIL_WITHOUT_REDACTION_ALLOWLIST.items()
            if live.get(site, 0) != declared
        }
        assert not drifted, (
            "an allowlisted bare tail no longer occurs the declared number of "
            "times (site: declared vs live); a count that dropped means the "
            "entry outlived its site — drop it rather than leaving a permission "
            "for code that is gone — and a count that grew means a new bare "
            "tail was spelled like an approved one, which needs the sink that "
            "covers it named before the count moves: "
            + "; ".join(
                f"{site}: {declared} vs {found}"
                for site, (declared, found) in sorted(drifted.items())
            )
        )

    @pytest.mark.parametrize(
        "reason_code",
        ["not_authenticated", "nested_exec_credentials", "nested_exec_network"],
    )
    def test_the_refusal_that_quotes_a_probe_goes_through_the_boundary(
        self, reason_code
    ):
        """Pinned per refusal, so a reworded message cannot quietly drop it.

        Scoped to the enclosing function rather than to the message expression:
        one of the three binds its tail to a local first, and a guard that only
        understands the inline spelling would go green on a refactor that
        reintroduced the leak one line higher.
        """
        source = Path(subagent_executor.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        owners = []
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            raises_it = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "RunnerRejected"
                and any(
                    keyword.arg == "reason_code"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == reason_code
                    for keyword in node.keywords
                )
                for node in ast.walk(function)
            )
            if raises_it:
                owners.append(function)
        assert owners, f"no RunnerRejected raises {reason_code!r} any more"
        for function in owners:
            called = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert "_redact_probe_tail" in called, (
                f"{function.name} raises {reason_code!r} while quoting a probe "
                f"stream that never reached the redaction boundary"
            )
            assert "_tail" not in called, (
                f"{function.name} still tails a stream directly"
            )
