"""The outcome rules for a provider CLI run, read without a child process.

``utils/provider_outcome.py`` turns a :class:`Capture` into the result
``invoke_with_provider`` returns. These are its structural guards; the outcome
of every committed fixture row on both lanes is graded in
``test_provider_fixture_rows.py`` (``expect.lane_status``), and the paths
through the real spawn stay in ``test_llm_client.py``.
"""

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import provider_outcome  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.provider_outcome import Capture, outcome_from_capture  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.retry_ladder import STATUS_OK, STATUS_SPAWN_FAILED, STATUS_TIMEOUT  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent

#: The helpers that attach a key a child that ran carries. Each is called from
#: the one tail only, so no path can attach a key the others miss.
TAIL_HELPERS = frozenset({
    "_attach_stdout_excerpt", "_attach_salvage_text", "_attach_answer_text",
    "_attach_usage", "_readonly_audit", "_attach_readonly_audit",
    "_attach_sandbox_word", "_note_stream_cap", "_salvage_hook_text",
    "_served_model_attribution", "_adapter_failure_category",
})

#: The fields both lanes' captures share, the stream half.
STREAM_FIELDS = (
    "stdout", "stderr", "exit_code", "timed_out", "duration", "launch_error",
    "capped_streams",
)


def _calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in TAIL_HELPERS:
                yield name


def test_only_the_outcome_module_attaches_the_tail_keys():
    """A second copy of the tail anywhere in the skill fails here, by name."""
    found = {}
    for path in sorted(SKILL_DIR.rglob("*.py")):
        parts = set(path.relative_to(SKILL_DIR).parts)
        if parts & {"tests", ".venv", "__pycache__"}:
            continue
        names = sorted(set(_calls(path)))
        if names:
            found[str(path.relative_to(SKILL_DIR))] = names
    assert set(found) == {"utils/provider_outcome.py"}, found


def test_a_capture_is_a_frozen_value():
    capture = Capture(stdout="x", exit_code=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        capture.stdout = "y"  # type: ignore[misc]


def test_lane_bs_capture_carries_the_same_stream_fields():
    """The executor's capture stays mutable (the spawn fills it in steps, the
    jail and the parse write to it after), so it shares the stream half by
    field: same names, same defaults."""
    shared = {f.name: f for f in dataclasses.fields(Capture)}
    lane_b = {f.name: f for f in dataclasses.fields(executor._Capture)}
    for name in STREAM_FIELDS:
        assert name in shared and name in lane_b, name
        assert shared[name].default == lane_b[name].default, name


@pytest.mark.parametrize("capture,status", [
    (Capture(launch_error="No such file"), STATUS_SPAWN_FAILED),
    (Capture(stdout="partial", timed_out=True), STATUS_TIMEOUT),
    (Capture(stdout="[]", exit_code=0), STATUS_OK),
])
def test_lane_bs_readers_take_the_shared_capture(capture, status):
    assert executor._classify_pre_parse(capture)[0] == status


def test_lane_bs_failed_run_reader_takes_the_shared_capture():
    capture = Capture(stdout="", stderr="429 Too Many Requests: rate limit", exit_code=1)
    assert executor._classify_failed_run(_PROVIDERS["opencode"], capture) == (
        executor._classify_failed_run(
            _PROVIDERS["opencode"],
            executor._Capture(stdout="", stderr=capture.stderr, exit_code=1),
        )
    )


def test_a_launch_failure_carries_no_key_of_a_child_that_ran():
    result = outcome_from_capture(
        _PROVIDERS["pi"], Capture(launch_error="boom", duration=0.5),
        provider_name="pi", model="m", want_answer_text=True,
        sandbox_fallback="unsandboxed",
    )
    assert result == {
        "success": False,
        "error": "Failed to launch pi CLI: boom",
        "error_code": provider_outcome.ERROR_SUBPROCESS_FAILED,
        "details": {"provider": "pi", "model": "m", "duration_seconds": 0.5},
    }


def test_a_jail_denial_wins_over_the_category():
    """The read-back is the caller's callable; what it returns replaces the
    error line, which then carries no category tag."""
    calls = []

    def read_back(details, **kwargs):
        calls.append(kwargs)
        details["failure_reason"] = provider_outcome.FAILURE_JAIL_DENIED
        details["retryable"] = False
        return "denied by the plugin's jail: ~/.codex/x", ""

    result = outcome_from_capture(
        _PROVIDERS["codex"],
        Capture(stdout="", stderr="429 Too Many Requests", exit_code=1),
        provider_name="codex", model="m", jail=read_back,
    )
    assert result["error"] == (
        "codex exited with code 1: denied by the plugin's jail: ~/.codex/x"
    )
    assert result["details"]["failure_reason"] == provider_outcome.FAILURE_JAIL_DENIED
    assert calls == [{
        "stdout": "", "stderr": "429 Too Many Requests", "returncode": 1,
        "failed": True, "timed_out": False,
    }]


@pytest.mark.parametrize("capture,order", [
    (Capture(stdout="", stderr="EACCES", exit_code=1), ["jail_read_back", "audit"]),
    (Capture(stdout="", stderr="EACCES", timed_out=True), ["audit", "jail_read_back"]),
    (Capture(stdout="", stderr="", exit_code=0), ["audit", "jail_read_back"]),
], ids=["nonzero", "timeout", "exit0"])
def test_the_jail_read_back_keeps_its_place_around_the_audit(monkeypatch, capture, order):
    """The read-back samples live state (the home, the other children still
    running), so where it runs decides what a finished attempt is charged
    with: a non-zero exit reads back before the read-only audit, a timeout and
    an exit 0 after it. Its keys land right after ``sandbox`` on every path."""
    calls = []
    audit = provider_outcome._readonly_audit

    def recording_audit(*args, **kwargs):
        calls.append("audit")
        return audit(*args, **kwargs)

    def read_back(details, **_kwargs):
        calls.append("jail_read_back")
        details["home_check"] = {"changed": ["config.toml"]}
        return None, ""

    monkeypatch.setattr(provider_outcome, "_readonly_audit", recording_audit)
    result = outcome_from_capture(
        _PROVIDERS["codex"], capture, provider_name="codex", model="m",
        sandbox_fallback="plugin-landlock", jail=read_back,
    )

    assert calls == order
    keys = list(result["details"])
    assert keys[keys.index("sandbox") + 1] == "home_check"


@pytest.mark.parametrize("capture,defer,logged", [
    (Capture(launch_error="boom"), False, 1),
    (Capture(stdout="x", exit_code=1), False, 1),
    (Capture(stdout="x", exit_code=1), True, 1),
    (Capture(stdout="x", timed_out=True), True, 1),
    (Capture(stdout="[]", exit_code=0), False, 1),
    (Capture(stdout="[]", exit_code=0), True, 0),
], ids=["launch", "nonzero", "nonzero-defer", "timeout-defer", "exit0", "exit0-defer"])
def test_the_log_writer_runs_once_unless_an_exit_0_is_deferred(capture, defer, logged):
    """A deferred exit 0 hands its capture back on ``_deferred_log`` instead;
    a hard failure is never deferred."""
    seen = []
    result = outcome_from_capture(
        _PROVIDERS["kilocode"], capture, provider_name="kilocode", model="m",
        defer_log=defer, log=lambda cap, res: seen.append((cap, res)),
    )

    assert seen == [(capture, result)] * logged
    assert ("_deferred_log" in result["details"]) is (defer and logged == 0)
