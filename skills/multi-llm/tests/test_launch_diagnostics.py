"""Where each adapter's launch diagnostics live (``LLMProvider.LAUNCH_DIAGNOSTICS``).

Two verdicts fail the executor's whole batch from text alone,
``sandbox_unavailable`` and ``spawn_failed``'s stderr rows, so they are read
only from stderr a model or a tool cannot write. Each adapter states which
kind of stderr its CLI prints, from its captures; the matrix of what that does
to both lanes is ``tests/test_exit_zero_prose_matrix.py``.
See docs/development/providers.md#launch-diagnostics.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.providers.base import (  # noqa: E402
    CATEGORY_SANDBOX_UNAVAILABLE,
    LAUNCH_DIAGNOSTICS_NONE,
    LAUNCH_DIAGNOSTICS_STDERR,
    LAUNCH_DIAGNOSTICS_VALUES,
    STATUS_SPAWN_FAILED,
    classify_streams,
    match_streams,
)

#: The adapters whose stderr relays a model's or a tool's words on lines that
#: cannot be told apart from the CLI's own; each declaration names its capture.
RELAYED = {"fx", "gemini", "kilocode", "opencode", "pi"}

BWRAP = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n"


@pytest.mark.parametrize("provider", sorted(_PROVIDERS))
def test_every_adapter_declares_where_its_launch_diagnostics_live(provider):
    """A reviewed claim about the CLI's stderr, so each adapter states it."""
    declared = vars(type(_PROVIDERS[provider])).get("LAUNCH_DIAGNOSTICS")
    assert declared in LAUNCH_DIAGNOSTICS_VALUES, provider


def test_the_relaying_adapters_are_the_reviewed_set():
    relayed = {
        name for name, adapter in _PROVIDERS.items()
        if adapter.LAUNCH_DIAGNOSTICS == LAUNCH_DIAGNOSTICS_NONE
    }
    assert relayed == RELAYED


@pytest.mark.parametrize("provider", sorted(_PROVIDERS))
def test_the_hook_hands_the_fatal_rows_stderr_or_nothing(provider):
    adapter = _PROVIDERS[provider]
    expected = "" if adapter.LAUNCH_DIAGNOSTICS == LAUNCH_DIAGNOSTICS_NONE else BWRAP
    assert adapter.launch_diagnostics(BWRAP) == expected


@pytest.mark.parametrize("provider", sorted(set(_PROVIDERS) - {"fx"}))
def test_exit_127_is_spawn_failed_whatever_stderr_holds(provider):
    """The shell's own verdict needs no text, so no declaration moves it.
    (fx classifies from its envelope alone, exit code included.)"""
    assert _PROVIDERS[provider].classify_failure("", "", 127) == STATUS_SPAWN_FAILED


@pytest.mark.parametrize("provider", sorted(RELAYED))
def test_a_relaying_adapter_reads_no_fatal_verdict_off_stderr(provider):
    """kilocode's tool trace: the model chose the path and
    the command, and the tool printed the refusal."""
    adapter = _PROVIDERS[provider]
    for stderr in (
        "> code · stub-model\n| Read docs/bubblewrap.md\n" + BWRAP,
        "> code · stub-model\n| Bash cat missing\nbash: command not found\n",
    ):
        assert adapter.classify_failure("", stderr, 1) not in (
            CATEGORY_SANDBOX_UNAVAILABLE, STATUS_SPAWN_FAILED,
        ), stderr


@pytest.mark.parametrize("provider", sorted(
    name for name, adapter in _PROVIDERS.items()
    if adapter.LAUNCH_DIAGNOSTICS == LAUNCH_DIAGNOSTICS_STDERR
))
def test_a_cli_stderr_adapter_keeps_the_measured_verdicts(provider):
    adapter = _PROVIDERS[provider]
    assert adapter.classify_failure("", BWRAP, 1) == CATEGORY_SANDBOX_UNAVAILABLE
    assert adapter.classify_failure("", "sh: 1: npx: command not found\n", 1) == (
        STATUS_SPAWN_FAILED
    )


@pytest.mark.parametrize("stdout,stderr,launch", [
    ("", BWRAP, ""),
    ("", BWRAP, None),
    ("", BWRAP, " " * len(BWRAP)),
    ("", "bash: npx: command not found\nrate limit exceeded", ""),
    ("", "bash: npx: command not found\nrate limit exceeded", None),
    ("rate limit exceeded", BWRAP, ""),
])
def test_match_streams_reads_the_same_launch_text_as_classify_streams(stdout, stderr, launch):
    """The quote's re-scan (``subagent_executor._quote_verdict``) reads what
    the verdict read, so the quote lands on the text that decided it."""
    expected = classify_streams(stdout, stderr, 1, launch_stderr=launch)
    match = match_streams(stdout, stderr, 1, launch_stderr=launch)
    assert (None if match is None else match.status) == expected
    if expected == CATEGORY_SANDBOX_UNAVAILABLE:
        assert (match.channel, match.offset) == ("stderr", 0)
