"""What the hosts are told to relay (fix round 18).

From the seventh live run (2026-10-06):

* #11, every host's phase 04: the optional `--apply-task-suggestions` step
  was shown first on Codex, after the primary on Claude Code, and without its
  `why` on OpenCode. Every route is now told the one order of
  plans/next-phase-command.md §2.1 ("Primary first": the `command`, then
  "Optional, before that:") and to give each optional entry's `why`, which
  says what skipping it means (tests/test_optional_step_why.py).
* #13a, Codex racing 08: the needs-human message named `argv` as what to
  report, and the host told the user the tool's own `--exec-subagents
  --detach`.
* #13c, Codex racing 06: the host put the manual-task note's prose in a
  ```text fence.
* #13d, Claude Code stop fix path: the host's hand-written summary.md said
  `retries: 2` after the user had fixed the config to 3; it described the
  file from T001's prompt and prior-changes line.
* #13e, Claude Code stop accept 01: the host offered "remove `src/vendor/**`
  from the deny list" as a way forward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

import modes  # noqa: E402


def _flat(path: str) -> str:
    return " ".join((SKILL_ROOT / path).read_text(encoding="utf-8").split())


#: Every route's relay rule for `user_next_step`: the tool description both
#: package hosts carry, their two plugin instructions, the two file shims and
#: Claude Code's SKILL.md.
RELAY_SURFACES = (
    "hosts/codex/plugin-skill.md.tmpl",
    "hosts/opencode/plugin-command.md.tmpl",
    "hosts/codex/shim.md.tmpl",
    "hosts/opencode/shim.md.tmpl",
    "SKILL.md",
)


def _relay_texts():
    yield "modes.TOOL_USER_NEXT_STEP_LINE", " ".join(modes.TOOL_USER_NEXT_STEP_LINE.split())
    for path in RELAY_SURFACES:
        yield path, _flat(path)


class TestEveryRouteRelaysTheOptionalStepWithItsWhy:

    @pytest.mark.parametrize("name,text", list(_relay_texts()))
    def test_command_first_then_each_optional_entry_with_its_why(self, name, text):
        assert "the `command` first" in text, name
        assert '"Optional, before that:"' in text, name
        assert "`why` as plain text above its command" in text, name

    def test_summary_format_keeps_the_same_order(self):
        text = _flat("references/summary-format.md")
        assert ("Any `optional_before` entries follow it, never precede it: the "
                "line \"Optional, before that:\"") in text
        assert "its `why` as plain text outside the fence" in text


class TestTheManualTaskNoteIsProse:

    @pytest.mark.parametrize("name,text", list(_relay_texts()))
    def test_every_route_keeps_the_note_out_of_a_fence(self, name, text):
        assert "fenced block holds one command and nothing else" in text, name

    @pytest.mark.parametrize("path", [
        "references/summary-format.md", "instructions/implement.md"])
    def test_the_claude_code_report_rules_say_so_too(self, path):
        text = _flat(path)
        assert ("as plain text, never inside a fenced block" in text
                or "as plain text and never in a fenced block" in text), path


class TestTheStopNamesThePhaseNotTheToolsArgv:

    def test_the_message_asks_for_the_phase_and_plan_and_forbids_argv(self):
        text = modes.NEEDS_HUMAN_MESSAGE["template"]
        assert "Say what was being run as its phase and its plan" in text
        assert "never quote `argv` or its flags to the user" in text
        for flag in ("--exec-subagents", "--detach"):
            assert flag in text

    @pytest.mark.parametrize("key", ["error_unknown", "phase_unknown"])
    def test_no_clause_sends_the_model_back_to_argv(self, key):
        assert "argv" not in modes.NEEDS_HUMAN_MESSAGE[key]


class TestTheWritePolicyIsNeverAWayForward:

    def test_the_tool_routes_are_told(self):
        text = " ".join(modes.TOOL_NO_COMPOSED_COMMAND_LINE.split())
        assert "never suggest loosening the write policy" in text
        assert "`driver.write_paths`" in text

    def test_the_native_leg_is_told_at_a_stop(self):
        text = _flat("instructions/implement.md")
        assert "Never suggest loosening the write policy for a task it refused" in text
        assert "the policy is the user's own" in text


def test_the_native_summary_describes_files_as_they_are_now():
    text = _flat("instructions/implement.md")
    assert ("Describe each file under `## Files Modified` and `## Files Created` "
            "as it is on disk now: when you quote a value it holds, read the file "
            "for it.") in text
    assert "a value copied from them can be out of date" in text
