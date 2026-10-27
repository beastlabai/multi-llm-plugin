"""What a host says at the end of a finished run (live run 8, #9).

* "Workflow complete." said while the workflow was not complete: OpenCode
  racing 04 after review-tasks (its result named `--implement`), and the
  Claude Code stop path's accept 02 after implement ("Workflow complete: no
  manual tasks are owed." above `--review-code`). The tool description said
  "only then" since fix round 11; the routes' own relay rules and Claude
  Code's report rules did not.
* OpenCode racing 08: a finished apply-code-fixes result carried, in
  `result.notice`, that nothing ran the repository's tests and that two
  invalid fixes were not applied, and the host's reply said neither. No
  route's text named the key: the stop's line (`error`) is relayed by rule,
  a finished run's `notice` was not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
for entry in (str(SKILL_ROOT), str(REPO_ROOT / "packaging")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import modes  # noqa: E402

from tests.test_next_step_relay_rules import RELAY_SURFACES, _flat  # noqa: E402

#: Fix round 24 (live run 10, OpenCode 08 wrote it twice): the copied
#: section says it, and only when complete; the host never writes it.
COMPLETE_ONLY = 'says "Workflow complete.": never write those words yourself.'



class TestWorkflowCompleteOnlyWhenComplete:

    @pytest.mark.parametrize("path", RELAY_SURFACES)
    def test_every_routes_relay_rule_says_so(self, path):
        assert COMPLETE_ONLY in _flat(path), path

    def test_the_claude_code_report_rule_says_so_beside_a_command(self):
        text = _flat("references/summary-format.md")
        assert ("`user_next_step.command` alone in a fenced block, copied character "
                "for character, and never \"Workflow complete.\" beside it") in text

    def test_the_tool_description_still_does(self):
        line = " ".join(modes.TOOL_USER_NEXT_STEP_LINE.split())
        # Fix round 22: the `markdown` it says to copy says it, and only then.
        assert "It says \"Workflow complete.\" only when it is `complete`" in line


class TestAFinishedRunsNoticeIsRelayed:

    @pytest.mark.parametrize("host", ["codex", "opencode"])
    def test_the_tool_description_names_result_notice(self, host):
        from common import surface

        description = surface.cli_surface(host)["tool_description"]
        assert modes.TOOL_NOTICE_LINE in description
        assert "`result` can carry a `notice`" in modes.TOOL_NOTICE_LINE
        assert "nothing ran this repository's tests" in modes.TOOL_NOTICE_LINE
        assert "every part of it" in modes.TOOL_NOTICE_LINE
        assert modes.TOOL_COMMIT_BLOCKS_LINE in description
        assert "`commit_blocks`" in modes.TOOL_COMMIT_BLOCKS_LINE
        assert "verbatim and once" in modes.TOOL_COMMIT_BLOCKS_LINE

    @pytest.mark.parametrize("path", [
        "hosts/codex/plugin-skill.md.tmpl", "hosts/opencode/plugin-command.md.tmpl",
    ])
    def test_the_package_routes_name_result_notice(self, path):
        text = _flat(path)
        assert ("When the result carries a `notice` (`result.notice`, a finished "
                "run's: what it left out or could not check") in text, path
        assert "report what it says, every part of it, as plain text." in text, path

    @pytest.mark.parametrize("path", [
        "hosts/codex/shim.md.tmpl", "hosts/opencode/shim.md.tmpl",
    ])
    def test_the_file_routes_read_it_off_the_result_line(self, path):
        text = _flat(path)
        assert ("When it carries a `notice` (a finished run's: what it left out or "
                "could not check") in text, path

    def test_claude_code_reads_it_off_the_result_line(self):
        assert ("report what its `notice` says, when it has one, every part of it, "
                "as plain text") in _flat("SKILL.md")
