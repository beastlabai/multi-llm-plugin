#!/usr/bin/env python3
"""The file channel, made load-bearing on the fake-binary lane.

**What was wrong.** `invoke_with_file_output` has two channels — the model
writes `{output_json_path}` and the caller reads it, or the caller falls back to
parsing stdout — and until this file existed the whole suite only ever exercised
the second one. The container stub never emits a tool call, so no real CLI on
the provider lane can move a byte; and `mock_llm.py` wrote a file only when
`MOCK_OUTPUT_PATH` or an `OUTPUT_FILE:` directive said so, neither of which any
orchestrator has ever emitted. So every fake-lane result arrived through the
fallback, and **an adapter that cannot write a file was indistinguishable from
one that can** — which is exactly how cursor-agent shipped for a release under
`--mode ask`, spending whole review slots explaining that it could not write the
file the prompt demanded.

`mock_llm.py` now reads the SHIPPED instruction
(`llm_client.FILE_OUTPUT_LOCATION`), so the primary channel is live on this lane
and the two halves of the `supports_file_output` contract are both observable:

* a file-capable identity writes the file and the result reports `source ==
  "file"`;
* a read-only identity is asked for stdout, writes nothing, and is served from
  the fallback — and if a prompt ever demands a file from it again, it answers
  with the sentence the real cursor-agent was measured answering with, and the
  slot fails here instead of in a user's review.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mock_launchers import install_mock_llm  # noqa: E402
from utils.llm_client import (  # noqa: E402
    FILE_OUTPUT_LOCATION,
    STDOUT_OUTPUT_LOCATION,
    invoke_with_file_output,
)

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"

#: A template shaped like the three shipped review prompts: some instruction,
#: then the `{output_location}` block `invoke_with_file_output` renders from the
#: adapter's declaration.
TEMPLATE = "Review this.\n\n## Output Location\n{output_location}\n"


@pytest.fixture
def mock_path(tmp_path, monkeypatch):
    """Put the fake binaries for both identities first on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for launcher in ("cursor-agent", "claude"):
        install_mock_llm(bin_dir, launcher)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    return bin_dir


def _invoke(model_spec, out_dir, template=TEMPLATE):
    return invoke_with_file_output(
        prompt_template=template,
        model_spec=model_spec,
        prompt_context={"prefix": "channel"},
        output_dir=out_dir,
        phase="channel",
        timeout=60,
        prefer_arrays=True,
    )


class TestTheFileChannelIsReallyUsed:
    """A file-capable adapter's result comes from the FILE, not the fallback."""

    def test_claude_code_writes_the_file_and_is_served_from_it(
        self, mock_path, tmp_path
    ):
        result = _invoke("claude-code:sonnet", tmp_path / "out")

        assert result["success"] is True, result.get("error")
        assert result["source"] == "file", (
            "the file channel is dead again: the shipped file instruction did "
            "not produce a file, so this result came from the stdout fallback "
            "and no test in the suite covers the primary path"
        )
        assert Path(result["output_file"]).exists()

    def test_the_file_the_prompt_named_is_the_file_that_was_read(
        self, mock_path, tmp_path
    ):
        result = _invoke("claude-code:sonnet", tmp_path / "out")

        written = Path(result["output_file"])
        assert written.parent == tmp_path / "out" / "channel"
        assert written.read_text(encoding="utf-8").strip().startswith("[")


class TestAReadOnlyAdapterIsServedFromStdout:
    """cursor-agent is asked for stdout, writes nothing, and still succeeds."""

    def test_cursor_succeeds_through_the_fallback(self, mock_path, tmp_path):
        result = _invoke("cursor-agent:auto", tmp_path / "out")

        assert result["success"] is True, result.get("error")
        assert result["source"] == "stdout_fallback"
        assert isinstance(result["data"], list)

    def test_cursor_writes_no_file(self, mock_path, tmp_path):
        result = _invoke("cursor-agent:auto", tmp_path / "out")

        assert not Path(result["output_file"]).exists(), (
            "cursor-agent declares supports_file_output = False; a file here "
            "means the prompt asked for one anyway"
        )

    def test_the_two_identities_really_take_different_channels(
        self, mock_path, tmp_path
    ):
        """The whole point, in one assertion: the declaration changed the run."""
        cursor = _invoke("cursor-agent:auto", tmp_path / "cursor")
        claude = _invoke("claude-code:sonnet", tmp_path / "claude")

        assert cursor["success"] and claude["success"]
        assert {cursor["source"], claude["source"]} == {"stdout_fallback", "file"}


class TestTheRegressionItselfIsCovered:
    """Hand cursor the file block again and the measured refusal comes back."""

    def test_a_file_demand_to_cursor_reproduces_the_shipped_failure(
        self, mock_path, tmp_path
    ):
        """This is the bug, reproduced: a *successful* CLI envelope carrying a
        refusal sentence, no file, and the slot dying on the parse."""
        forced_file_block = (
            "Review this.\n\n## Output Location\n"
            + FILE_OUTPUT_LOCATION.format(path=tmp_path / "out" / "forced.json")
            + "\n"
        )

        result = _invoke("cursor-agent:auto", tmp_path / "out", forced_file_block)

        assert result["success"] is False
        assert result["error"] == "JSON parsing failed from both file and stdout"
        assert not (tmp_path / "out" / "forced.json").exists()
        # ...and the sentence that caused it is readable from the failure, which
        # is the diagnosis that cost three 450 s runs the first time.
        assert "Ask mode" in result["details"]["stdout_excerpt"]

    def test_the_shipped_prompt_does_not_do_that(self, mock_path, tmp_path):
        """...and the prompt we actually ship never asks cursor for a file."""
        result = _invoke("cursor-agent:auto", tmp_path / "out")

        assert result["success"] is True
        assert STDOUT_OUTPUT_LOCATION.splitlines()[0].startswith("CRITICAL: Return")


#: Adapters that declare `supports_file_output = False` for a reason that is
#: NOT a read-only mode, and whose fake identity therefore keeps writing when a
#: prompt hands it the file block. codex (D6, 2026-09-11): its CLI writes
#: wherever bubblewrap can start; the declaration moves the review off a sandbox
#: that a stock Ubuntu 24.04 cannot start, and a mock that refused the write
#: would model a CLI that does not exist on a working machine. What guards the
#: review is the prompt — `test_llm_client.TestOutputLocationBlock` pins that
#: codex is asked for stdout — not a refusal sentence here.
DECLARED_FALSE_BUT_WRITES_WHEN_ASKED = frozenset({"codex"})


class TestTheMockRosterTracksTheDeclaration:
    """`mock_llm.READ_ONLY_IDENTITIES` cannot drift from the adapters."""

    def test_every_read_only_identity_is_a_declared_read_only_adapter(self):
        from tests.test_providers import TestDeclaredOutputChannel

        sys.path.insert(0, str(MOCK_LLM.parent))
        import mock_llm

        assert mock_llm.READ_ONLY_IDENTITIES <= (
            TestDeclaredOutputChannel.CANNOT_WRITE_FILES
        )

    def test_every_declared_adapter_with_an_identity_is_in_the_roster(self):
        """The other direction, over the identities the mock actually has.

        An adapter that declares it cannot write files and HAS a fake identity
        must refuse there too — otherwise the fake lane models a CLI that does
        not exist.
        """
        from tests.harness.skill_runner import SkillRunner
        from tests.test_providers import TestDeclaredOutputChannel

        sys.path.insert(0, str(MOCK_LLM.parent))
        import mock_llm

        with_identities = TestDeclaredOutputChannel.CANNOT_WRITE_FILES & set(
            SkillRunner.PROVIDERS
        )
        assert (
            with_identities - DECLARED_FALSE_BUT_WRITES_WHEN_ASKED
            == mock_llm.READ_ONLY_IDENTITIES
        )

    def test_the_exception_is_a_declared_false_adapter_with_an_identity(self):
        """The exception cannot outlive what it excuses: every member must be
        on the declared roster, have a fake identity, and NOT refuse there."""
        from tests.harness.skill_runner import SkillRunner
        from tests.test_providers import TestDeclaredOutputChannel

        sys.path.insert(0, str(MOCK_LLM.parent))
        import mock_llm

        assert DECLARED_FALSE_BUT_WRITES_WHEN_ASKED <= (
            TestDeclaredOutputChannel.CANNOT_WRITE_FILES
        )
        assert DECLARED_FALSE_BUT_WRITES_WHEN_ASKED <= set(SkillRunner.PROVIDERS)
        assert not (DECLARED_FALSE_BUT_WRITES_WHEN_ASKED & mock_llm.READ_ONLY_IDENTITIES)

    def test_codex_is_asked_for_stdout_and_still_succeeds(self, mock_path, tmp_path):
        """D6, end to end on the fake lane: the shipped prompt asks codex for
        stdout, the mock writes nothing, and the slot is served from the
        fallback — the same per-model result either way."""
        bin_dir = mock_path
        install_mock_llm(bin_dir, "codex")

        result = _invoke("codex:gpt-5.5", tmp_path / "out")

        assert result["success"] is True, result.get("error")
        assert result["source"] == "stdout_fallback"
        assert not Path(result["output_file"]).exists()
