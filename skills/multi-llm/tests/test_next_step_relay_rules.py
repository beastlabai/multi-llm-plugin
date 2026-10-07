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

import re
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


#: The rule of every route that copies `user_next_step.markdown`, the
#: section already written (fix round 22: Codex 04 left both commands
#: unfenced, and the hosts folded the manual close command into prose).
#: Fix round 24: Claude Code copies it too (the lookup prints it between
#: `[NEXT_STEP_MARKDOWN]` lines; racing 06 put the manual tasks first), in
#: its words and order (OpenCode 03 reordered it under captions of its own).
MARKDOWN_RULE = (
    "copied as it is (its words and order, each command alone in its fenced "
    "block, each why and note plain text, the manual tasks it lists included, "
    "no caption of your own)")


class TestEveryRouteRelaysTheOptionalStepWithItsWhy:

    @pytest.mark.parametrize("name,text", list(_relay_texts()))
    def test_command_first_then_each_optional_entry_with_its_why(self, name, text):
        # The order and the whys are the markdown's own
        # (tests/test_next_step_markdown.py).
        assert MARKDOWN_RULE in text, name

    def test_summary_format_keeps_the_same_order(self):
        text = _flat("references/summary-format.md")
        assert ("Any `optional_before` entries follow it, never precede it: the "
                "line \"Optional, before that:\"") in text
        assert "its `why` as plain text outside the fence" in text


class TestTheManualTaskNoteIsProse:

    @pytest.mark.parametrize("name,text", list(_relay_texts()))
    def test_every_route_keeps_the_note_out_of_a_fence(self, name, text):
        assert MARKDOWN_RULE in text, name

    @pytest.mark.parametrize("path,said", [
        ("references/summary-format.md",
         "verbatim as plain text and its `manual_tasks_command` (the "
         "`[MANUAL_TASK_COMMAND]` line) alone in its own fenced block"),
        ("instructions/implement.md",
         "as plain text and never in a fenced block"),
    ])
    def test_the_claude_code_report_rules_say_so_too(self, path, said):
        assert said in _flat(path), path


#: A word that asks the host to tell the user something.
_REPORTING = {"report", "say", "says", "tell", "relay", "quote", "name", "names",
              "show", "give", "print", "mention"}
#: A negation that binds the `argv` right after it ("never quote `argv`").
_NEGATIONS = {"never", "not", "don't", "no"}


def _words(text: str) -> list:
    words = (word.strip("`()*,:\"'").lower() for word in text.split())
    return [word for word in words if word]


def _argv_reporting_clauses(text: str) -> list:
    """Each place *text* asks for `argv` — or "what was being run", which on
    a file route is the same command line — to be reported: a reporting word
    in the eight words of its sentence before it (``report … what was being
    run (`argv`)``) or in the three after it (```argv` says what ran``), and
    no negation in the six words before it. Code spans other than `argv`
    itself are not words here, and a line wrap ends nothing."""
    text = " ".join(re.sub(
        r"`([^`\n]*)`",
        lambda span: span.group(0) if span.group(1) == "argv" else " ",
        text).split())
    found = []
    for match in re.finditer(r"\bargv\b|\bwhat was being run\b", text, re.I):
        head = re.split(r"[.;:!?](?:\s|$)", text[:match.start()])[-1]
        tail = re.split(r"[.;:!?](?:\s|$)", text[match.end():])[0]
        before, after = _words(head), _words(tail)
        if _NEGATIONS & set(before[-6:]):
            continue
        if _REPORTING & set(before[-8:]) or _REPORTING & set(after[:3]):
            found.append(text[max(0, match.start() - 120):match.end() + 80])
    return found


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _host_facing_texts():
    """Every text a host reads: the mode table's strings, both tool routes'
    generated surfaces, and the skill's, shims' and cores' markdown."""
    for name, value in vars(modes).items():
        if not name.startswith("__"):
            for text in _strings(value):
                yield f"modes.{name}", text
    repo = SKILL_ROOT.parent.parent
    if str(repo / "packaging") not in sys.path:
        sys.path.insert(0, str(repo / "packaging"))
    from common import surface

    for host in ("codex", "opencode"):
        for text in _strings(surface.cli_surface(host)):
            yield f"{host} surface", text
    for pattern in ("SKILL.md", "hosts/*/*.tmpl", "hosts/snippets/*.md",
                    "instructions/*.md", "references/*.md"):
        for path in sorted(SKILL_ROOT.glob(pattern)):
            yield str(path.relative_to(SKILL_ROOT)), path.read_text(encoding="utf-8")


class TestTheStopNamesThePhaseNotTheToolsArgv:

    def test_the_message_asks_for_the_phase_and_plan_and_forbids_argv(self):
        text = modes.NEEDS_HUMAN_MESSAGE["template"]
        # Fix round 22 (live run 9, OpenCode accept 01): "which phase was
        # running" came back as "Phase running: implement" for a stopped one.
        assert "Report to the user which phase stopped, on which plan" in text
        assert "was running" not in text
        assert "Never quote `argv` or its flags to the user" in text
        for flag in ("--exec-subagents", "--detach"):
            assert flag in text

    def test_the_file_routes_rule_asks_for_the_phase_and_plan_too(self):
        assert "report to the user which phase stopped, on which plan" in (
            modes.NEEDS_HUMAN_RULE)
        assert "was running" not in modes.NEEDS_HUMAN_RULE

    def test_no_host_facing_text_asks_for_argv(self):
        """Fix round 19: the merge of round 18 kept "Report to the user what
        was being run (`argv`)" as the message's opening beside the later
        "never quote `argv`", so both tool routes were told both, and the
        file routes' and Claude Code's needs-human rule asked for "what was
        being run", the dispatcher command with the flags the shim composed.
        Any text a host reads that asks for either fails here."""
        offenders = [
            f"{where}: {clause}"
            for where, text in _host_facing_texts()
            for clause in _argv_reporting_clauses(text)
        ]
        assert not offenders, "\n".join(offenders)

    @pytest.mark.parametrize("clause", [
        "Report to the user what was being run (`argv`), that line whole",
        "`log` and `argv` say what ran",
        "tell the user the argv it ran",
        "On `3` stop: report to the user what was being run, the last error line",
    ])
    def test_the_scan_catches_an_argv_report(self, clause):
        assert _argv_reporting_clauses(clause), clause

    @pytest.mark.parametrize("clause", [
        "Never quote `argv` or its flags to the user",
        "The plan path, as exactly one argv element — never split",
    ])
    def test_the_scan_passes_a_prohibition_and_a_plain_mention(self, clause):
        assert not _argv_reporting_clauses(clause), clause

    @pytest.mark.parametrize("key", ["error_unknown", "phase_unknown"])
    def test_no_clause_sends_the_model_back_to_argv(self, key):
        assert "argv" not in modes.NEEDS_HUMAN_MESSAGE[key]


def _hosts_texts():
    """Every text a host reads its stop rule in: both tool descriptions and
    exit-3 messages, the exit-3 rule every file route and plugin instruction
    renders (`{NEEDS_HUMAN_RULE}`), Claude Code's SKILL.md, and implement's
    stop rule (the native leg's stop exits 0)."""
    sys.path.insert(0, str(SKILL_ROOT.parent.parent / "packaging"))
    from common import surface

    for host in ("codex", "opencode"):
        rendered = surface.cli_surface(host)
        yield f"{host} tool description", " ".join(rendered["tool_description"].split())
        yield (f"{host} needs-human message",
               " ".join(rendered["needs_human_message"]["template"].split()))
    yield "modes.NEEDS_HUMAN_RULE", " ".join(modes.NEEDS_HUMAN_RULE.split())
    for path in ("SKILL.md", "instructions/implement.md"):
        yield path, _flat(path)


class TestTheWritePolicyIsTheUsersCall:
    """Fix round 20 (live run 8 #4; the user's decision: permissive). Round
    18 told every host "never suggest loosening driver.write_paths", while
    the plugin's own stop line offered allowing the file and the hosts were
    told to relay that line whole. The plugin keeps offering it as the user's
    call; the hosts never edit the setting themselves and relay the ways
    forward the stop names, as it words them."""

    @pytest.mark.parametrize(
        "name,text", list(_hosts_texts()), ids=[n for n, _t in _hosts_texts()])
    def test_every_host_reads_the_one_rule(self, name, text):
        rule = " ".join(modes.STOP_POLICY_RULE.split())
        assert rule in text, name
        assert "loosen" not in text.lower(), name

    def test_the_rule_never_forbids_the_way_the_plugin_offers(self):
        rule = modes.STOP_POLICY_RULE
        assert "Never edit `driver.write_paths`" in rule
        assert ("when it names a change to `driver.write_paths` that lets a "
                "refused file be written (adding it to `allow`, or removing or "
                "narrowing a deny rule that covers it), that is one of them "
                "and the user's call") in rule
        assert "add no advice about the write policy of your own" in rule

    def test_it_agrees_with_what_the_plugin_says(self):
        """The rule's two kinds of change are the clauses every stop,
        refusal and status line uses (fix round 21: per rule)."""
        from utils.write_safety import (
            PolicyHit, policy_allow_clause, policy_skip_clause,
        )

        assert policy_allow_clause([PolicyHit("src/vendor/stamp.js", deny=("src/vendor/**",))]) == (
            "if you want src/vendor/stamp.js written, remove or narrow the "
            "deny rule `src/vendor/**` in driver.write_paths "
            "(.multi-llm/providers.yaml) so the rule no longer covers it")
        assert policy_allow_clause([PolicyHit("a.js", allowed=False),
                                    PolicyHit("b.js", allowed=False)]) == (
            "if you want a.js and b.js written, add them to "
            "driver.write_paths.allow (.multi-llm/providers.yaml)")
        assert policy_skip_clause("T005") == (
            "if not, skip T005: add --human-task T005=skip to the resume "
            "command (T005 and the tasks after it stay undone)")

    def test_the_native_leg_relays_the_refusal_and_the_answers(self):
        text = _flat("instructions/implement.md")
        assert ("its refusal (the emit's `refused_tasks` line, or "
                "`[WRITE_PROMPT_REFUSED]`) names the change to "
                "`driver.write_paths` that lets its file be written (none for "
                "a file outside the repository, which no rule can allow) and "
                "resuming") in text
        assert "relay them as the user's choice, in those words" in text


def test_the_native_summary_describes_files_as_they_are_now():
    text = _flat("instructions/implement.md")
    assert ("Describe each file under `## Files Modified` and `## Files Created` "
            "as it is on disk now: when you quote a value it holds, read the file "
            "for it.") in text
    assert "a value copied from them can be out of date" in text


class TestTheStopsWaysForwardAreItsOwn:
    """Live run 9, OpenCode stop accept 01: beside the stop's line and its
    commands, the host wrote a paraphrased "Ways forward (as named)" list,
    and headed every answer "Optional, before that:", the label of an
    optional phase. The answers clause says each answer's why goes above
    it, that the answers are alternatives to `resume_command`, and that no
    list of the host's own goes beside them."""

    def test_the_answers_clause(self):
        clause = " ".join(modes.NEEDS_HUMAN_MESSAGE["answers"].split())
        assert "Instead of `resume_command` the user can type any of these" in clause
        assert "its why as plain text above it" in clause
        assert "no list or summary of them of your own" in clause
        assert "{answers}" in clause
