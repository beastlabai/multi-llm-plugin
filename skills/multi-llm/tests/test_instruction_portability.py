#!/usr/bin/env python3
"""Portability regression guards for the instruction/driver layer.

Two generations of guard live here.

**Windows/macOS quoting (original).** Phase 1 of the portability plan swept every
shell-argument occurrence of ``${CLAUDE_SKILL_DIR}`` in the instruction/reference
markdown into the double-quoted form (Windows install paths routinely contain
spaces, e.g. ``C:\\Users\\John Smith\\...``, so an unquoted expansion word-splits
and the command dies before Python starts), and audited the ``allowed-tools``
glob syntax. Those two machine greps are ``TestSkillDirQuoting`` and
``TestAllowedToolsGlobSyntax``.

**Host-harness portability (the substitution validator).** The driver layer is
being made host-neutral: the shared instruction core refers to the skill dir as a
single ``{SKILL_DIR}`` **template token** — *not shell syntax*, since a driving
agent that copies ``uv run --project {SKILL_DIR}`` verbatim gets
``{SKILL_DIR}: No such file or directory`` — and the token is substituted by a
command (``multi_llm.py util instructions <mode>``) before an agent ever reads it,
never by the reader. The classes below enforce the six rules that keep that true:

1. *Site inventory* — every ``{SKILL_DIR}`` occurrence sits inside the invariant
   ``uv run --project … multi_llm.py`` prefix; an occurrence anywhere else fails.
2. *Corpus completeness* — a checked-in per-file manifest, asserted exhaustive
   over the corpus in **both** directions, recording the expected occurrence count
   per file so a regression that reintroduces a second substitution site is a
   failure rather than silent drift.
3. *Render-and-parse* — every command block in every file, rendered against a
   fixture skill dir **containing a space**, carries no brace token, ``shlex``
   splits cleanly, and yields the skill dir as exactly one argv element.
4. *The served core is token-free* — ``util instructions <mode>`` emits the source
   file with the prefix already substituted, for every mode that has one.
5. *Printed follow-ups are resolved at the source* — the follow-up commands the
   orchestrators print carry no placeholder, name an existing path, and keep the
   allow-rule prefix Claude Code matches ``Bash(...)`` permissions on.
6. *Negative cases* — ``${SKILL_DIR}``/``$SKILL_DIR`` are never defined and are a
   failure wherever they appear; the ``${CLAUDE_SKILL_DIR}`` counts are pinned.

**What this asserts today vs. after the corpus rewrite.** The instruction markdown
has *not* yet been rewritten to ``{SKILL_DIR}`` (a later change owns that sweep),
so the manifest currently records today's non-zero ``${CLAUDE_SKILL_DIR}`` counts
and a zero ``{SKILL_DIR}`` count per file. Every rule above is enforced *now* on
whatever occurrences exist, and rules 3–5 are already fully load-bearing (they run
against the current ``${CLAUDE_SKILL_DIR}`` corpus, because the renderer
substitutes every spelling of the placeholder). When the corpus is rewritten, the
only edit needed here is flipping each manifest row's two counts — at which point
rule 1 stops being vacuous and rule 6's "the only residual
``${CLAUDE_SKILL_DIR}`` is the Claude Code ``SKILL.md`` shim" becomes assertable
by setting every other row's first number to 0.

Known blind spot (accepted by the plan): the unquoted-path-argument pattern stops
scanning a line at its first double quote, so a mixed line with one quoted and one
later-unquoted occurrence can pass. That residual case is a manual-review concern;
these tests block the shapes that actually occurred.
"""

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common  # noqa: E402


def _scanned_files():
    files = sorted(SKILL_ROOT.glob("instructions/*.md"))
    files += sorted(SKILL_ROOT.glob("references/*.md"))
    files.append(SKILL_ROOT / "SKILL.md")
    # `hosts/` does not exist yet (a later change adds the per-host shims); the
    # manifest tests below fail loudly the moment it appears without rows.
    files += sorted(SKILL_ROOT.glob("hosts/**/*.md"))
    files += sorted(SKILL_ROOT.glob("hosts/**/*.md.tmpl"))
    return files


def _matches(pattern):
    """Return (file, line_no, line) for every regex match across the docs."""
    hits = []
    for path in _scanned_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(pattern, line):
                hits.append((path.name, i, line.strip()))
    return hits


class TestSkillDirQuoting:
    def test_no_unquoted_project_flag(self):
        """``--project ${CLAUDE_SKILL_DIR}`` must always be quoted:
        ``--project "${CLAUDE_SKILL_DIR}"``."""
        hits = _matches(r"--project \$\{CLAUDE_SKILL_DIR\}")
        assert not hits, (
            "Unquoted `--project ${CLAUDE_SKILL_DIR}` (must be "
            '`--project "${CLAUDE_SKILL_DIR}"`) found at:\n'
            + "\n".join(f"  {f}:{ln}: {txt}" for f, ln, txt in hits)
        )

    def test_no_unquoted_skill_dir_command_argument(self):
        """No ``uv run``/``python`` command line may pass an unquoted
        ``${CLAUDE_SKILL_DIR}`` argument (e.g. ``python
        ${CLAUDE_SKILL_DIR}/script.py``)."""
        hits = _matches(r'(uv run|python)[^"]*[ =]\$\{CLAUDE_SKILL_DIR\}')
        assert not hits, (
            "Unquoted ${CLAUDE_SKILL_DIR} in shell-argument position "
            '(must be wrapped in double quotes, e.g. '
            '`"${CLAUDE_SKILL_DIR}/script.py"`) found at:\n'
            + "\n".join(f"  {f}:{ln}: {txt}" for f, ln, txt in hits)
        )


class TestAllowedToolsGlobSyntax:
    def test_no_regex_syntax_in_bash_rules(self):
        """Claude Code `allowed-tools` Bash patterns are glob-style prefix
        rules, not regexes: a rule like ``Bash(grep:.*)`` is inert (it matches
        the literal prefix ``grep:.``) and silently reintroduces permission
        prompts. Only ``:*`` glob prefixes or exact strings are valid."""
        skill_md = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        offending = [
            line.strip()
            for line in skill_md.splitlines()
            if "Bash(" in line and re.search(r"\.\*|\.\+", line)
        ]
        assert not offending, (
            "Regex syntax inside Bash(...) allowed-tools rules (glob-style "
            "`:*` prefix or exact string only):\n"
            + "\n".join(f"  {line}" for line in offending)
        )


# ===========================================================================
# The {SKILL_DIR} substitution validator
# ===========================================================================

#: Per-file expected placeholder counts: ``path -> (${CLAUDE_SKILL_DIR}, {SKILL_DIR})``.
#:
#: Asserted exhaustive in both directions, so a file added to (or removed from)
#: the corpus without a manifest row is a failure — the test cannot pass by
#: silently skipping the file that was missed.
#:
#: The second number is the *substitution-site* count the design budgets for:
#: after the corpus rewrite each file collapses to at most the one invariant
#: ``uv run --project {SKILL_DIR} -- python {SKILL_DIR}/multi_llm.py`` prefix per
#: command, so a row whose count grows unexpectedly is a regression that
#: reintroduced a second site.
#:
#: **The rewrite has landed**, so every row is now ``(0, n)`` — except
#: ``SKILL.md``, the one file that keeps ``${CLAUDE_SKILL_DIR}`` because it *is*
#: the Claude Code shim and the host expands the variable at load time. That makes
#: "the only residual ``${CLAUDE_SKILL_DIR}`` in the tree is the Claude Code
#: ``SKILL.md``" directly assertable rather than aspirational: see
#: ``test_claude_skill_dir_is_only_in_the_shim``.
PLACEHOLDER_MANIFEST = {
    # 28 ${CLAUDE_SKILL_DIR} -> 28 {SKILL_DIR}: the same commands, now naming the
    # single entry point behind the invariant prefix.
    "instructions/apply-code-fixes.md": (0, 28),
    "instructions/apply-suggestions.md": (0, 28),
    "instructions/apply-task-suggestions.md": (0, 26),
    # 4 -> 6: the hand-rolled question-file dance (write the file, re-resolve the
    # git root inline, launch) collapsed into the dispatcher's two positionals,
    # and the mode gained its `--detach` + `status` poll pair.
    "instructions/ask.md": (0, 6),
    # 50 -> 48: D6 row 12 deleted the no-TTY branch, and with it the
    # `check_workflow_prerequisites.py --skip` command it carried. 48 -> 36: the
    # Phase-3 sequencing (the inlined subagent prompt, its file-write steps, the
    # per-phase skip-recording commands and row 12's question) moved into
    # `full_workflow_orchestrator.py`. 36 -> 30: six "load instructions from
    # <path>" lines became one shared `util instructions <mode>` command. A count
    # that *drops* is the corpus shrinking as designed; only a rise is a
    # regression.
    "instructions/full-workflow.md": (0, 30),
    # 8 -> 0: gate, materialization, dispatch, the in-process `update_plan_tasks`
    # call and the TASKS_FILE verification are all Python now, so the file names no
    # command at all.
    "instructions/generate-tasks.md": (0, 0),
    # 24 -> 20: the skill's own pytest invocation (a dogfooding special case) and
    # the inline `python -c` that inserted the summary reference are gone — the
    # latter is `util finalize-tracking --plan-file --summary-file-relative`.
    "instructions/implement.md": (0, 20),
    # 3 -> 2: the "substitute {SKILL_DIR} yourself" preamble is obsolete; the core
    # is served pre-substituted.
    "instructions/init-config.md": (0, 2),
    "instructions/review-code.md": (0, 24),
    "instructions/review-plan.md": (0, 26),
    "instructions/review-tasks.md": (0, 16),
    # The shared shim partials (D5). Authored once, interpolated into a rendered
    # shim by `install-host`, and still counted against that shim's rendered line
    # budget because an agent reads them. `poll-loop.md` carries the one `status`
    # command; `uv-preflight.md` names no path at all — its message arrives through
    # the `{{uv_missing_text}}` slot, so the wording stays single-sourced in
    # hosts/uv-missing.txt.
    "hosts/snippets/poll-loop.md": (0, 2),
    "hosts/snippets/uv-preflight.md": (0, 0),
    # The per-host shim templates (D7 renders them; `install-host` bakes the
    # prefix). Three commands each — `util instructions <mode>`, the one dispatcher
    # command, and the `status … --view run --wait` poll — so six token sites, all
    # inside the invariant prefix. The uv install message is *named*, not inlined:
    # a shim prints `hosts/uv-missing.txt` (whose absolute path arrives through the
    # `{{uv_missing_file}}` slot) exactly as the reference shim does, because
    # inlining 31 lines of install text into a 100-line shim would spend a third of
    # the budget on a message an agent reads once.
    "hosts/codex/shim.md.tmpl": (0, 6),
    "hosts/opencode/shim.md.tmpl": (0, 6),
    # Generated from hosts/capabilities.yaml (D5b). Carries host tool names by
    # design — it is human documentation of the matrix — and therefore no
    # skill-dir placeholder of either spelling.
    "references/host-capabilities.md": (0, 0),
    "references/human-decision-batch.md": (0, 0),
    "references/resume-detection.md": (0, 0),
    "references/salvage-handling.md": (0, 0),
    # Moved out of the Claude Code shim in the D5c cutover: what a run reports is
    # workflow prose, identical on every host, so it belongs in the core.
    "references/summary-format.md": (0, 0),
    # 1 -> 0: the illustrative `uv run --project "<dir>" -- python …` fragment was
    # not a command anyone runs, and a token outside the invariant prefix is a
    # failure by rule 1.
    "references/uv-check.md": (0, 0),
    "references/wave-batching.md": (0, 0),
    # 7 -> 12 when the read step became a dispatcher command; 12 -> 8 with the
    # cutover, which slimmed the shim to its host-specific contract (the mode list,
    # the provider-config prose, the critical rules and the argv-scanning shell
    # preamble all left). Still non-zero, and permanently so: this file *is* the
    # Claude Code shim.
    "SKILL.md": (8, 0),
}

#: The one file allowed to keep ``${CLAUDE_SKILL_DIR}``: it *is* the Claude Code
#: shim, and the host expands the variable at load time.
CLAUDE_SHIM = "SKILL.md"

#: A fixture skill dir with a space in every segment that can carry one, so a
#: quoting bug cannot hide behind a space-free path. Contains the marker word
#: "fixture" in exactly one segment, which the argv assertions key on.
FIXTURE_SKILL_DIR = "/tmp/multi llm fixture/skill dir"

#: The invariant prefix a ``{SKILL_DIR}`` occurrence is allowed to sit inside.
INVARIANT_PREFIX_RE = re.compile(
    r"uv run --project \"?\{SKILL_DIR\}\"? -- python \"?\{SKILL_DIR\}/multi_llm\.py\"?"
)

UV_COMMAND_RE = re.compile(r"^(PYTHONUNBUFFERED=1\s+)?uv\s+run\b")

#: Any brace-delimited token that names a skill dir, in any spelling. Scoped this
#: way because the corpus legitimately carries authoring placeholders
#: (``{total_batches}``, ``{plan_stem}``) that a reader fills in.
SKILL_DIR_BRACE_RE = re.compile(r"\$?\{[A-Za-z_]*SKILL_DIR\}")

#: Placeholder spellings that must never survive into text an agent reads.
FORBIDDEN_IN_RENDERED = (
    "{SKILL_DIR}",
    "${SKILL_DIR}",
    "$SKILL_DIR",
    "${CLAUDE_SKILL_DIR}",
    "$MULTI_LLM_SKILL_DIR",
    "${MULTI_LLM_SKILL_DIR}",
)


def _rel(path: Path) -> str:
    return path.relative_to(SKILL_ROOT).as_posix()


def _command_units(text: str):
    """Yield ``(line_no, command)`` for every shell command unit in ``text``.

    A "unit" is one logical command: successive physical lines are joined while a
    backslash continuation is open *or* a double quote is still unbalanced, so a
    multi-line ``python -c "…"`` heredoc-style block is one unit rather than a
    truncated fragment that could never parse.
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = re.sub(r"^[-*+]\s+", "", lines[index].strip())
        if not UV_COMMAND_RE.match(stripped):
            index += 1
            continue
        start = index + 1
        unit = stripped
        while index + 1 < len(lines) and (
            unit.rstrip().endswith("\\") or unit.count('"') % 2 == 1
        ):
            if unit.rstrip().endswith("\\"):
                unit = unit.rstrip()[:-1].rstrip()
            index += 1
            unit += " " + lines[index].strip()
        yield start, unit
        index += 1


def _assert_command_is_clean(where: str, command: str, base: str):
    """The three render-and-parse assertions, shared by every caller."""
    for token in FORBIDDEN_IN_RENDERED:
        assert token not in command, (
            f"{where}: unexpanded placeholder {token} survived rendering:\n"
            f"  {command[:200]}"
        )
    # Scoped to skill-dir tokens rather than every brace: the corpus also
    # carries authoring placeholders an agent fills in ({total_batches},
    # {plan_stem}), which are not this rule's concern.
    residue = SKILL_DIR_BRACE_RE.search(command)
    assert residue is None, (
        f"{where}: skill-dir brace token {residue.group(0)} left in a rendered "
        f"command:\n  {command[:200]}"
    )
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        pytest.fail(f"{where}: rendered command does not shlex.split ({exc}):\n  {command[:200]}")

    assert argv and argv[0] in ("uv", "PYTHONUNBUFFERED=1"), (
        f"{where}: the allow-rule prefix is part of the printed command's "
        f"contract — a command whose first token is not `uv` (or "
        f"`PYTHONUNBUFFERED=1 uv`) stops matching Claude Code's Bash(uv:*) glob "
        f"and turns every follow-up into a permission prompt:\n  {command[:200]}"
    )
    assert "--project" in argv, f"{where}: no --project in {argv[:6]}"
    project = argv[argv.index("--project") + 1]
    assert project == base, (
        f"{where}: the skill dir must be exactly one argv element in the "
        f"--project slot, got {project!r} (word-splitting on a space):\n"
        f"  {command[:200]}"
    )
    marker = "fixture" if "fixture" in base else base
    for element in argv:
        if marker in element:
            assert element == base or element.startswith(base + "/"), (
                f"{where}: skill-dir fragment {element!r} is a separate argv "
                f"element — the path word-split on a space:\n  {command[:200]}"
            )


class TestPlaceholderManifest:
    """Rule 2: the per-file manifest is exhaustive and its counts are pinned."""

    def test_manifest_covers_every_corpus_file(self):
        on_disk = {_rel(p) for p in _scanned_files()}
        missing = on_disk - set(PLACEHOLDER_MANIFEST)
        assert not missing, (
            "instruction/shim file on disk with no PLACEHOLDER_MANIFEST row "
            f"(add one, with its expected placeholder counts): {sorted(missing)}"
        )

    def test_manifest_has_no_stale_rows(self):
        on_disk = {_rel(p) for p in _scanned_files()}
        stale = set(PLACEHOLDER_MANIFEST) - on_disk
        assert not stale, (
            f"PLACEHOLDER_MANIFEST names files that no longer exist: {sorted(stale)}"
        )

    @pytest.mark.parametrize("rel", sorted(PLACEHOLDER_MANIFEST))
    def test_placeholder_counts_match_manifest(self, rel):
        expected_claude, expected_token = PLACEHOLDER_MANIFEST[rel]
        text = (SKILL_ROOT / rel).read_text(encoding="utf-8")
        assert text.count("${CLAUDE_SKILL_DIR}") == expected_claude, (
            f"{rel}: ${{CLAUDE_SKILL_DIR}} count changed "
            f"({text.count('${CLAUDE_SKILL_DIR}')} != {expected_claude}). The "
            "corpus rewrite lowers these numbers; nothing else should raise one."
        )
        assert text.count("{SKILL_DIR}") == expected_token, (
            f"{rel}: {{SKILL_DIR}} count changed "
            f"({text.count('{SKILL_DIR}')} != {expected_token}). Each file "
            "budgets one substitution site per command; a second one is a "
            "regression."
        )

    def test_claude_skill_dir_is_only_in_the_shim(self):
        """The corpus rewrite's headline invariant, asserted on disk.

        Only ``SKILL.md`` may carry ``${CLAUDE_SKILL_DIR}``: it *is* the Claude
        Code shim, and the host expands the variable at load time. Every other
        file in the corpus reaches 0, which is what makes the placeholder a
        host-specific detail of one file rather than a coupling spread over 5k
        lines. Asserted against the files themselves, not only the manifest, so a
        row edited to match a regression still fails.
        """
        assert PLACEHOLDER_MANIFEST[CLAUDE_SHIM][0] > 0, (
            "SKILL.md is the Claude Code shim and resolves the token via "
            "${CLAUDE_SKILL_DIR}; losing every occurrence means the shim broke."
        )
        offenders = {
            _rel(path): (path.read_text(encoding="utf-8").count("${CLAUDE_SKILL_DIR}"))
            for path in _scanned_files()
            if _rel(path) != CLAUDE_SHIM
            and "${CLAUDE_SKILL_DIR}" in path.read_text(encoding="utf-8")
        }
        assert not offenders, (
            "${CLAUDE_SKILL_DIR} is a Claude-Code-specific pre-expansion and may "
            "exist in exactly one file, the Claude Code shim. Use the "
            "{SKILL_DIR} template token (substituted by `multi_llm.py util "
            f"instructions <mode>`) instead. Found in: {offenders}"
        )
        assert all(
            counts[0] == 0
            for rel, counts in PLACEHOLDER_MANIFEST.items()
            if rel != CLAUDE_SHIM
        ), "a manifest row outside SKILL.md still budgets for ${CLAUDE_SKILL_DIR}"

    def test_the_token_corpus_is_not_vacuous(self):
        """The rewrite moved the sites, it did not delete them."""
        total = sum(counts[1] for counts in PLACEHOLDER_MANIFEST.values())
        assert total >= 150, (
            f"only {total} {{SKILL_DIR}} sites remain across the corpus, which "
            "would mean the commands went missing rather than being rewritten."
        )


class TestPlaceholderSites:
    """Rule 1: a ``{SKILL_DIR}`` occurrence may only sit in the invariant prefix."""

    @pytest.mark.parametrize("rel", sorted(PLACEHOLDER_MANIFEST))
    def test_token_only_inside_invariant_prefix(self, rel):
        text = (SKILL_ROOT / rel).read_text(encoding="utf-8")
        offenders = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            occurrences = line.count("{SKILL_DIR}")
            if not occurrences:
                continue
            covered = sum(
                match.group(0).count("{SKILL_DIR}")
                for match in INVARIANT_PREFIX_RE.finditer(line)
            )
            if covered < occurrences:
                offenders.append((line_no, line.strip()))
        assert not offenders, (
            f"{rel}: {{SKILL_DIR}} outside the invariant "
            "`uv run --project {SKILL_DIR} -- python {SKILL_DIR}/multi_llm.py` "
            "prefix. The token belongs nowhere else — not in prompts, not in "
            "task JSON, not in printed follow-ups:\n"
            + "\n".join(f"  line {ln}: {txt}" for ln, txt in offenders)
        )


class TestRenderAndParse:
    """Rule 3: every command block renders clean under a space-bearing skill dir."""

    def test_the_probe_is_not_vacuous(self):
        total = sum(
            len(list(_command_units((SKILL_ROOT / rel).read_text(encoding="utf-8"))))
            for rel in PLACEHOLDER_MANIFEST
        )
        assert total >= 50, f"suspiciously few uv command units found: {total}"

    @pytest.mark.parametrize("rel", sorted(PLACEHOLDER_MANIFEST))
    def test_command_blocks_render_and_split(self, rel):
        source = (SKILL_ROOT / rel).read_text(encoding="utf-8")
        rendered = cli_common.render_instruction_text(source, FIXTURE_SKILL_DIR)
        for line_no, command in _command_units(rendered):
            _assert_command_is_clean(f"{rel}:{line_no}", command, FIXTURE_SKILL_DIR)


class TestServedCoreIsTokenFree:
    """Rule 4: ``util instructions <mode>`` serves a token-free core file."""

    SERVED = [m.name for m in mode_table.MODES if m.instruction_file]

    def test_every_available_mode_has_a_core_file_on_disk(self):
        for mode in mode_table.MODES:
            if mode.instruction_file is None:
                continue
            path = SKILL_ROOT / mode_table.instruction_path(mode.name)
            assert path.exists(), (
                f"mode '{mode.name}' declares core file {mode.instruction_file} "
                "but it is missing on disk — the served surface would silently "
                "diverge from the dispatch table."
            )

    def test_served_surface_is_not_vacuous(self):
        assert len(self.SERVED) >= 11, self.SERVED

    @pytest.mark.parametrize("mode", SERVED)
    def test_rendered_core_is_token_free(self, mode):
        """Rendered under a fixture skill dir *containing a space*."""
        source = (SKILL_ROOT / mode_table.instruction_path(mode)).read_text(
            encoding="utf-8"
        )
        rendered = cli_common.render_instruction_text(source, FIXTURE_SKILL_DIR)
        for token in FORBIDDEN_IN_RENDERED:
            assert token not in rendered, (
                f"util instructions {mode}: {token} survived rendering"
            )
        # Byte-for-byte identical to the source *except* at the placeholders:
        # the line count is unchanged and every placeholder-free line matches.
        source_lines = source.splitlines()
        rendered_lines = rendered.splitlines()
        assert len(rendered_lines) == len(source_lines)
        for line_no, (before, after) in enumerate(
            zip(source_lines, rendered_lines), start=1
        ):
            if any(token in before for token in FORBIDDEN_IN_RENDERED):
                assert FIXTURE_SKILL_DIR in after, f"{mode}:{line_no}"
            else:
                assert after == before, (
                    f"util instructions {mode}:{line_no} changed a line that "
                    f"carried no placeholder:\n  - {before}\n  + {after}"
                )

    @pytest.mark.parametrize("mode", SERVED)
    def test_cli_serves_token_free_core(self, mode, capsys):
        exit_code = multi_llm.main(["util", "instructions", mode])
        out = capsys.readouterr().out
        assert exit_code == 0, out
        body = out.rsplit(cli_common.RESULT_MARKER, 1)[0]
        for token in FORBIDDEN_IN_RENDERED:
            assert token not in body, f"util instructions {mode}: {token} in output"
        assert str(cli_common.skill_dir()) in body or "${" not in body

    def test_missing_mode_is_an_error(self, capsys):
        assert multi_llm.main(["util", "instructions", "no-such-mode"]) == 3
        capsys.readouterr()

    #: Modes with no core instruction file *by design* — they are pure flag
    #: surfaces with no workflow to narrate. Pinned so the set cannot grow
    #: silently: a new mode landing here means a shim's read step now says
    #: nothing useful about it, which is a review decision, not a default.
    DISPATCHER_DRIVEN = ["status", "install-host", "uninstall-host"]

    def test_dispatcher_driven_set_is_exactly_the_pinned_one(self):
        actual = sorted(m.name for m in mode_table.MODES if not m.instruction_file)
        assert actual == sorted(self.DISPATCHER_DRIVEN), (
            "the set of modes with no core instruction file changed. Every shim "
            "names `util instructions <mode>` as its unconditional read step, so "
            "each of these must still answer usefully rather than error."
        )

    @pytest.mark.parametrize("mode", DISPATCHER_DRIVEN)
    def test_dispatcher_driven_mode_answers_instead_of_erroring(self, mode, capsys):
        """Regression: this exited 3, so a host agent's first step failed.

        Observed live on OpenCode — the shim's read step returned exit 3 for
        `status` and the agent had to improvise past it. A mode with no core
        file is not an error, it is a mode whose flags are the interface.
        """
        exit_code = multi_llm.main(["util", "instructions", mode])
        out = capsys.readouterr().out
        assert exit_code == 0, out
        body = out.rsplit(cli_common.RESULT_MARKER, 1)[0]
        assert mode in body
        # It has to point somewhere, not just decline.
        assert f"{mode} --help" in body
        for token in FORBIDDEN_IN_RENDERED:
            assert token not in body, f"util instructions {mode}: {token} in output"

    @pytest.mark.parametrize("mode", [m.name for m in mode_table.MODES])
    def test_every_mode_word_is_answerable(self, mode, capsys):
        """No mode in the dispatch table may fail the shims' read step."""
        exit_code = multi_llm.main(["util", "instructions", mode])
        out = capsys.readouterr().out
        assert exit_code == 0, f"util instructions {mode} exited {exit_code}:\n{out}"
        body = out.rsplit(cli_common.RESULT_MARKER, 1)[0].strip()
        assert body, f"util instructions {mode} produced no output"

    def test_alias_serves_the_canonical_file(self, capsys):
        assert multi_llm.main(["util", "instructions", "code-review"]) == 0
        aliased = capsys.readouterr().out
        assert multi_llm.main(["util", "instructions", "review-code"]) == 0
        canonical = capsys.readouterr().out
        assert aliased == canonical


class TestPrintedFollowUps:
    """Rule 5: the orchestrators' printed follow-ups are resolved at the source."""

    @pytest.fixture
    def fixture_skill_dir(self, tmp_path, monkeypatch):
        """A real directory whose path contains a space, so "names an existing
        path" is checkable rather than notional."""
        base = tmp_path / "multi llm fixture" / "skill dir"
        base.mkdir(parents=True)
        (base / "multi_llm.py").write_text("# fixture\n", encoding="utf-8")
        monkeypatch.setattr(cli_common, "_SKILL_DIR", base)
        return base

    @pytest.mark.parametrize(
        "script",
        ["review_plan_orchestrator.py", "review_tasks_orchestrator.py",
         "code_review_orchestrator.py"],
    )
    def test_validation_reaggregate_command(self, script, fixture_skill_dir, tmp_path):
        """``utils/validation.py``'s reaggregate hint (the ``:956`` site).

        The task document stores only the durable ``reaggregate_mode`` + plan
        pair (D1c); the resolved command is rebuilt at print time, so that
        rebuild is what has to be clean.
        """
        from utils.validation import (
            prepare_batched_validation_tasks,
            resolved_reaggregate_command,
        )

        plan = tmp_path / "a plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        result = prepare_batched_validation_tasks(
            groups=[{
                "group_id": "g1",
                "importance": "HIGH",
                "suggestions": [{"id": "s1", "description": "d", "section": "s"}],
            }],
            context="ctx",
            output_dir=str(tmp_path),
            plan_file=str(plan),
            orchestrator=script,
        )
        assert "reaggregate_command" not in result, (
            "a persisted task document must never store a skill-dir-absolute "
            "command: the plugin cache is keyed by commit hash (D1c)"
        )
        command = resolved_reaggregate_command(result)
        _assert_command_is_clean("validation.py", command, str(fixture_skill_dir))
        argv = shlex.split(command)
        assert argv[argv.index("--") + 2] == str(fixture_skill_dir / "multi_llm.py")
        assert Path(argv[argv.index("--") + 2]).exists()
        assert str(plan) in argv
        assert "--reaggregate" in argv
        assert script not in command, (
            "the printed follow-up must name the single entry point, never a "
            "second orchestrator path"
        )

    def test_code_review_print_sites(self, fixture_skill_dir):
        """``code_review_orchestrator.py``'s two print sites (``:1565``, ``:1642``).

        Both now build their command through the shared helper, so driving the
        helper for the mode they pass is equivalent to driving the print sites —
        and a source scan below proves neither site can regress to a literal.
        """
        command = cli_common.reaggregate_command("review-code", "/tmp/a plan.md")
        _assert_command_is_clean("code_review_orchestrator.py", command,
                                 str(fixture_skill_dir))
        assert " review-code " in command

    def test_paths_are_shell_data_never_shell_source(self, tmp_path, monkeypatch):
        """A printed command may not re-enter the shell through its own paths.

        Both halves are filesystem-derived — the skill dir the plugin cache
        handed us, the plan path the user typed — so a checkout under
        ``/tmp/$(id).d`` has to round-trip as *that literal directory name*.
        Double quoting (what this replaced) does not: ``$VAR``, backticks and
        ``$(…)`` all stay live inside them, which turned a copy-pasteable
        follow-up into arbitrary command execution.
        """
        base = tmp_path / "skill $(id) `whoami` dir"
        base.mkdir(parents=True)
        (base / "multi_llm.py").write_text("# fixture\n", encoding="utf-8")
        monkeypatch.setattr(cli_common, "_SKILL_DIR", base)
        plan = tmp_path / "a $(touch pwned) plan.md"

        command = cli_common.reaggregate_command("review-code", str(plan))
        argv = shlex.split(command)
        assert argv[argv.index("--project") + 1] == str(base)
        assert argv[argv.index("--") + 2] == str(base / "multi_llm.py")
        assert str(plan) in argv

        # `shlex.split` does not expand, so the decisive check is a real shell:
        # every argument must reach argv byte-identical, and nothing may run.
        bash = shutil.which("bash")
        if bash is None:  # pragma: no cover - POSIX shell absent (native Windows)
            pytest.skip("no POSIX shell available to check expansion")
        for hostile in (str(base), str(plan), "$HOME", "`id`", "a'b\"c", "x; id"):
            proc = subprocess.run(
                [bash, "-c", f"printf %s {cli_common._quote(hostile)}"],
                capture_output=True, text=True, cwd=str(tmp_path),
            )
            assert proc.returncode == 0, proc.stderr
            assert proc.stdout == hostile, (
                f"{hostile!r} was rewritten by the shell to {proc.stdout!r}"
            )
        assert not (tmp_path / "pwned").exists(), (
            "the plan path's command substitution executed"
        )

    def test_no_production_print_emits_a_placeholder(self):
        """A stdout-wide scan: no ``print`` may carry a skill-dir placeholder.

        This is what makes the change durable — the golden regression bundle
        compares phase artifacts, never stdout, so a printed command that stopped
        working would otherwise be unguarded.
        """
        offenders = []
        for path in _production_python_files():
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "CLAUDE_SKILL_DIR" not in line and "{SKILL_DIR}" not in line:
                    continue
                if "print(" in line or "[REAGGREGATE" in line:
                    offenders.append((_rel(path), line_no, line.strip()))
        assert not offenders, (
            "orchestrator stdout must contain no unexpanded placeholder — print "
            "a path resolved from __file__ instead:\n"
            + "\n".join(f"  {f}:{ln}: {txt}" for f, ln, txt in offenders)
        )

    def test_validation_module_has_no_placeholder_left(self):
        text = (SKILL_ROOT / "utils" / "validation.py").read_text(encoding="utf-8")
        assert "CLAUDE_SKILL_DIR" not in text, (
            "utils/validation.py builds a printed follow-up; it must resolve the "
            "skill dir from __file__ and carry no placeholder at all."
        )

    def test_result_next_carries_no_placeholder_or_internal_flag(self):
        ctx = cli_common.ResultContext(mode="review-code", phase="code-review")
        ctx.set_next(cli_common.reaggregate_command("review-code", "/tmp/a plan.md"))
        payload = ctx.payload(0)
        for token in FORBIDDEN_IN_RENDERED:
            assert token not in payload["next"]
        fresh = cli_common.ResultContext(mode="review-code")
        fresh.set_next(
            "uv run ... multi_llm.py review-code p.md --internal-worker"
        )
        assert fresh.payload(0)["next"] is None, (
            "an --internal-* flag must never be rendered into a follow-up"
        )


# ---------------------------------------------------------------------------
# Rule 5b: `--help` is stdout too
# ---------------------------------------------------------------------------
#
# Rule 5 covers the three *follow-up* print sites. `--help` is the other stdout
# surface that ships commands to a user, and argparse writes its `epilog` there
# verbatim — so an epilog example carrying a skill-dir placeholder hands a Codex
# or OpenCode user a command that dies with
# `${CLAUDE_SKILL_DIR}: No such file or directory` before Python ever starts (no
# shell but Claude Code's expands that variable, and *nothing* expands the brace
# token). That makes D1's acceptance row — "a stdout scan across every mode finds
# no ${CLAUDE_SKILL_DIR} and no {SKILL_DIR}" — an assertion about the help surface
# as well, and the scan below is parametrized over every mode word, every `util`
# subcommand and every orchestrator's own parser so a new epilog cannot be added
# unguarded.

def _dispatcher_help_argv():
    """Every ``--help`` stdout surface reachable through the dispatcher."""
    argvs = [["--help"], ["util", "--help"]]
    argvs += [[mode.name, "--help"] for mode in mode_table.MODES]
    argvs += [["util", name, "--help"] for name in multi_llm.UTIL_COMMANDS]
    return argvs


def _orchestrator_parse_args():
    """``(script, callable)`` for every ``*_orchestrator.py``'s own parser.

    Discovered rather than listed, and asserted exhaustive over the scripts on
    disk (``test_every_orchestrator_parser_is_covered``), so a new orchestrator
    joins this scan without an edit here. Two shapes exist: a module-level
    ``parse_args`` (the review/ask/implement family) and a ``parse_args``
    classmethod on the ``ApplyOrchestratorBase`` subclass (the apply family).
    """
    import importlib
    import inspect

    found = []
    for path in sorted(SKILL_ROOT.glob("*_orchestrator.py")):
        module = importlib.import_module(path.stem)
        target = getattr(module, "parse_args", None)
        if target is None:
            for _name, obj in vars(module).items():
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    candidate = getattr(obj, "parse_args", None)
                    if candidate is not None:
                        target = candidate
                        break
        assert target is not None, (
            f"{path.name} exposes no parse_args, so its --help epilog cannot be "
            "scanned for an unexpanded placeholder"
        )
        found.append((path.name, target))
    return found


#: Computed at import so the parametrization ids name the surface under test.
DISPATCHER_HELP_ARGV = _dispatcher_help_argv()


def _help_body(out: str) -> str:
    """Captured stdout minus the terminal ``MULTI_LLM_RESULT`` line."""
    return out.rsplit(cli_common.RESULT_MARKER, 1)[0]


def _assert_help_is_clean(where: str, text: str, base: str) -> int:
    """The help-surface assertions. Returns the command-unit count it checked."""
    for token in ("${CLAUDE_SKILL_DIR}", "{SKILL_DIR}", "$SKILL_DIR",
                  "${SKILL_DIR}", "$MULTI_LLM_SKILL_DIR"):
        assert token not in text, (
            f"{where}: --help is stdout, and {token} reaches the user unexpanded "
            "on every host but Claude Code. Interpolate "
            "`cli_common.dispatcher_prefix()` into the epilog instead."
        )
    units = list(_command_units(text))
    for line_no, command in units:
        _assert_command_is_clean(f"{where} (help line {line_no})", command, base)
        assert "_orchestrator.py" not in command, (
            f"{where}: a printed example must name the single entry point "
            f"(`multi_llm.py <mode>`), never a second, unadvertised one:\n"
            f"  {command[:200]}"
        )
        argv = shlex.split(command)
        script = argv[argv.index("--") + 2]
        assert script == str(Path(base) / "multi_llm.py"), (
            f"{where}: the script slot must be the resolved dispatcher path, "
            f"got {script!r}"
        )
        word = argv[argv.index("--") + 3]
        assert (
            word == mode_table.UTIL_NAMESPACE
            or mode_table.resolve(word) is not None
        ), (
            f"{where}: '{word}' follows multi_llm.py in a printed example but is "
            "neither a mode word nor the util namespace, so the example does not "
            "run"
        )
    return len(units)


class TestHelpSurfaceIsResolved:
    """Rule 5b: no ``--help`` surface ships an unexpanded skill-dir placeholder.

    Everything runs under a fixture skill dir whose every segment carries a
    space, so a quoting bug cannot hide behind a space-free path — the same
    fixture rules 3-5 use.
    """

    @pytest.fixture
    def fixture_skill_dir(self, tmp_path, monkeypatch):
        base = tmp_path / "multi llm fixture" / "skill dir"
        base.mkdir(parents=True)
        (base / "multi_llm.py").write_text("# fixture\n", encoding="utf-8")
        monkeypatch.setattr(cli_common, "_SKILL_DIR", base)
        return base

    def test_the_help_surface_is_not_vacuous(self):
        """Every mode word and every util command is actually reached."""
        flattened = {" ".join(argv) for argv in DISPATCHER_HELP_ARGV}
        for mode in mode_table.MODES:
            assert f"{mode.name} --help" in flattened, mode.name
        for name in multi_llm.UTIL_COMMANDS:
            assert f"util {name} --help" in flattened, name

    @pytest.mark.parametrize(
        "argv", DISPATCHER_HELP_ARGV, ids=lambda argv: " ".join(argv)
    )
    def test_dispatcher_help_is_resolved(self, argv, fixture_skill_dir, capsys):
        exit_code = multi_llm.main(list(argv))
        out = capsys.readouterr().out
        assert exit_code == 0, out
        _assert_help_is_clean(
            f"multi_llm.py {' '.join(argv)}", _help_body(out),
            str(fixture_skill_dir),
        )

    @pytest.mark.parametrize(
        "script,parse", _orchestrator_parse_args(), ids=lambda v: getattr(v, "__name__", v)
    )
    def test_direct_parser_help_is_resolved(self, script, parse,
                                            fixture_skill_dir, capsys):
        """The direct surface too: same parser, same epilog, same stdout."""
        with pytest.raises(SystemExit) as excinfo:
            parse(["--help"])
        assert excinfo.value.code == 0
        _assert_help_is_clean(script, capsys.readouterr().out,
                              str(fixture_skill_dir))

    def test_every_orchestrator_parser_is_covered(self):
        scripts = {p.name for p in SKILL_ROOT.glob("*_orchestrator.py")}
        assert {name for name, _parse in _orchestrator_parse_args()} == scripts

    def test_the_scan_finds_real_commands(self, fixture_skill_dir, capsys):
        """A guard against an over-narrow ``_command_units``.

        Without it, a regex that stopped matching would turn every assertion
        above into a no-op that scanned an empty list.
        """
        total = 0
        for argv in DISPATCHER_HELP_ARGV:
            assert multi_llm.main(list(argv)) == 0
            total += len(list(_command_units(_help_body(capsys.readouterr().out))))
        assert total >= 20, (
            f"only {total} uv command units found across every --help surface; "
            "the epilog examples went missing rather than being resolved"
        )

    def test_epilog_examples_survive_a_space_in_the_skill_dir(self,
                                                             fixture_skill_dir,
                                                             capsys):
        """The concrete failure the fixture exists for, asserted end to end."""
        assert multi_llm.main(["review-plan", "--help"]) == 0
        body = _help_body(capsys.readouterr().out)
        commands = [command for _line, command in _command_units(body)]
        assert commands, body
        for command in commands:
            argv = shlex.split(command)
            assert argv[argv.index("--project") + 1] == str(fixture_skill_dir)
            assert " " in str(fixture_skill_dir)

    def test_a_placeholder_in_an_epilog_would_fail(self, fixture_skill_dir):
        """The assertion fires on the shape this change deleted."""
        base = str(fixture_skill_dir)
        with pytest.raises(AssertionError):
            _assert_help_is_clean(
                "synthetic",
                'Example:\n  uv run --project "${CLAUDE_SKILL_DIR}" -- python '
                '"${CLAUDE_SKILL_DIR}/multi_llm.py" review-plan p.md\n',
                base,
            )
        with pytest.raises(AssertionError):
            _assert_help_is_clean(
                "synthetic",
                f'Example:\n  uv run --project {base} -- python '
                f'"{base}/multi_llm.py" review-plan p.md\n',
                base,
            )
        with pytest.raises(AssertionError):
            _assert_help_is_clean(
                "synthetic",
                f'Example:\n  uv run --project "{base}" -- python '
                f'"{base}/review_plan_orchestrator.py" --plan-file p.md\n',
                base,
            )


class TestNegativeSpellings:
    """Rule 6: the illegal spellings stay illegal."""

    @pytest.mark.parametrize("token", ["${SKILL_DIR}", "$SKILL_DIR"])
    def test_illegal_variable_spelling_is_absent(self, token):
        hits = [
            (_rel(path), i, line.strip())
            for path in _scanned_files()
            for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if token in line
        ]
        assert not hits, (
            f"`{token}` is never defined — the brace token `{{SKILL_DIR}}` is a "
            "template placeholder and the only legal *shell* spelling is "
            '`"$MULTI_LLM_SKILL_DIR"`. Found at:\n'
            + "\n".join(f"  {f}:{ln}: {txt}" for f, ln, txt in hits)
        )

    def test_renderer_does_not_define_the_illegal_spellings(self):
        assert "${SKILL_DIR}" not in cli_common._SKILL_DIR_TOKENS
        assert "$SKILL_DIR" not in cli_common._SKILL_DIR_TOKENS

    def test_instructions_and_shims_name_only_the_entry_point(self):
        """The instruction core names exactly one script, and the debt is now zero.

        Direct ``*_orchestrator.py`` invocation stays supported-but-unadvertised
        (the test suite and the ``--internal-*`` re-entry rely on it); what the
        core and the shims name is ``multi_llm.py``, so a host resolves one path
        once and routes nothing itself.
        """
        hits = [
            _rel(path)
            for path in _scanned_files()
            if re.search(r"_orchestrator\.py", path.read_text(encoding="utf-8"))
        ]
        assert hits == [], (
            "the instruction core and the shims name only multi_llm.py; a direct "
            f"*_orchestrator.py reference reappeared in: {hits}"
        )


# ===========================================================================
# The legacy escape hatch
# ===========================================================================

class TestLegacyInstructionHatch:
    """A way back from the ~5k-line rewrite that does not revert the Python work.

    The golden guard proves the modes its fixture exercises, not the interactive
    corners a real user hits on day two — so the pre-refactor core is *moved*, not
    deleted, and one environment variable switches the read path back to it.
    """

    LEGACY = SKILL_ROOT / "legacy"

    def test_every_served_mode_has_a_frozen_copy(self):
        for mode in mode_table.MODES:
            if mode.instruction_file is None:
                continue
            assert (self.LEGACY / mode_table.instruction_path(mode.name)).exists(), (
                f"no frozen pre-refactor copy for mode '{mode.name}'; the escape "
                "hatch would silently serve the rewritten core instead"
            )

    def test_the_frozen_copies_are_the_pre_refactor_text(self):
        """They still name the host's tools — that is the point of keeping them."""
        legacy = (self.LEGACY / "instructions" / "review-plan.md").read_text(
            encoding="utf-8"
        )
        assert "${CLAUDE_SKILL_DIR}" in legacy
        assert "AskUserQuestion" in legacy
        assert "run_in_background" in legacy

    def test_the_hatch_is_closed_by_default(self, monkeypatch, capsys):
        monkeypatch.delenv(multi_llm.LEGACY_INSTRUCTIONS_ENV, raising=False)
        assert multi_llm.main(["util", "instructions", "review-plan"]) == 0
        captured = capsys.readouterr()
        assert "${CLAUDE_SKILL_DIR}" not in captured.out
        assert "AskUserQuestion" not in captured.out
        assert "LEGACY_INSTRUCTIONS" not in captured.err

    def test_the_hatch_serves_the_frozen_core_rendered(self, monkeypatch, capsys):
        monkeypatch.setenv(multi_llm.LEGACY_INSTRUCTIONS_ENV, "1")
        assert multi_llm.main(["util", "instructions", "review-plan"]) == 0
        captured = capsys.readouterr()
        assert "AskUserQuestion" in captured.out, "the legacy text was not served"
        # Rendered, not raw: the renderer substitutes every spelling, so the
        # legacy corpus's `${CLAUDE_SKILL_DIR}` arrives resolved too.
        for token in FORBIDDEN_IN_RENDERED:
            assert token not in captured.out, f"{token} survived legacy rendering"
        assert str(cli_common.skill_dir()) in captured.out
        # The notice goes to stderr: stdout is the core, and a host may pipe it
        # straight into a prompt.
        assert "[LEGACY_INSTRUCTIONS]" in captured.err

    def test_an_unset_value_other_than_one_does_not_open_it(self, monkeypatch,
                                                            capsys):
        for value in ("0", "", "true", "yes"):
            monkeypatch.setenv(multi_llm.LEGACY_INSTRUCTIONS_ENV, value)
            assert multi_llm.main(["util", "instructions", "review-plan"]) == 0
            captured = capsys.readouterr()
            assert "AskUserQuestion" not in captured.out, value

    def test_legacy_is_outside_every_lint_and_manifest_scan(self):
        """A frozen copy that had to pass the new rules would not be a copy."""
        scanned = {_rel(p) for p in _scanned_files()}
        assert not any(rel.startswith("legacy/") for rel in scanned), scanned
        assert not any(
            _rel(p).startswith("legacy/") for p in _lint_corpus_files()
        )
        assert not any(rel.startswith("legacy/") for rel in PLACEHOLDER_MANIFEST)

    def test_the_hatch_is_documented(self):
        readme = (self.LEGACY / "README.md").read_text(encoding="utf-8")
        assert multi_llm.LEGACY_INSTRUCTIONS_ENV in readme
        assert "Claude Code only" in readme


# ===========================================================================
# Host-neutrality lint (D5)
# ===========================================================================
#
# The instruction core was rewritten to capability verbs, and after that rewrite
# nothing but a test keeps host tool names from creeping back into the shared
# core — so the refactor's central invariant is enforced mechanically, not by
# review. Runs on every PR, not only under `-m live`.

#: The closed three-verb vocabulary. Identical in all three places it appears: the
#: wording in ``instructions/*.md``, the three ``capabilities.yaml`` fields and the
#: three shim slots. A core sentence reaching for a fourth verb fails.
CAPABILITY_VERBS = ("spawn subagents", "ask the user", "run detached")

#: Tokens matched anywhere in a line. Each is paired with the verb that replaces
#: it, so a failure is fixable without re-reading the design.
FORBIDDEN_TOKENS = {
    "AskUserQuestion": "ask the user",
    "subagent_type": "spawn subagents",
    "run_in_background": "run detached",
    "TodoWrite": "(no replacement — the core tracks progress in state.json)",
    "${CLAUDE_SKILL_DIR}": "{SKILL_DIR}",
    ".claude/": "(no replacement — a host-specific config path)",
}

#: Tokens whose ordinary English senses appear constantly in this corpus ("read
#: the plan file", "write the summary"), so they are matched *only* in their tool
#: spellings: inside a code span, capitalized and followed by ``(``, or followed by
#: the word "tool"/"subagent". "Read the plan file" passes; "use the Read tool"
#: fails. The narrower rule is deliberate — a lint that cries wolf gets suppressed,
#: and a suppressed lint guards nothing.
AMBIGUOUS_TOOL_TOKENS = {
    "Task": "spawn subagents",
    "Read": "(plain English: read the file)",
    "Write": "(plain English: write the file)",
    "Edit": "(plain English: edit the file)",
    "Glob": "(plain English: list/match files)",
    "Grep": "(plain English: search the files)",
}

#: Files whose *job* is to name host tools, and therefore exempt: the Claude Code
#: shim (D5c), the ``hosts/**`` shim templates, and the generated capability
#: reference (D5b). Anything else needs a per-file, per-token entry below.
LINT_EXEMPT_FILES = ("SKILL.md", "references/host-capabilities.md")
LINT_EXEMPT_PREFIXES = ("hosts/",)

#: Per-file, per-token exceptions with a one-line rationale. Every entry is
#: asserted to still match something (``test_no_stale_lint_allowlist_entries``), so
#: the allowlist cannot decay into a blanket ignore.
LINT_ALLOWLIST: dict = {}


def _ambiguous_patterns(token):
    return (
        rf"`{token}`",
        rf"\b{token}\s*\(",
        rf"\b{token}[ -][Tt]ool\b",
        rf"\b{token} subagents?\b",
    )


def _lint_corpus_files():
    files = sorted(SKILL_ROOT.glob("instructions/*.md"))
    files += sorted(SKILL_ROOT.glob("references/*.md"))
    return [
        path for path in files
        if _rel(path) not in LINT_EXEMPT_FILES
        and not any(_rel(path).startswith(p) for p in LINT_EXEMPT_PREFIXES)
    ]


def _lint_violations():
    hits = []
    for path in _lint_corpus_files():
        rel = _rel(path)
        allowed = LINT_ALLOWLIST.get(rel, {})
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for token, verb in FORBIDDEN_TOKENS.items():
                if token in line and token not in allowed:
                    hits.append((rel, line_no, token, verb, line.strip()[:110]))
            for token, verb in AMBIGUOUS_TOOL_TOKENS.items():
                if token in allowed:
                    continue
                if any(re.search(p, line) for p in _ambiguous_patterns(token)):
                    hits.append((rel, line_no, token, verb, line.strip()[:110]))
    return hits


class TestHostNeutralityLint:
    def test_the_corpus_is_not_vacuous(self):
        files = _lint_corpus_files()
        assert len(files) >= 15, [_rel(p) for p in files]
        assert any(_rel(p) == "instructions/implement.md" for p in files)

    def test_no_host_tool_names_in_the_shared_core(self):
        hits = _lint_violations()
        assert not hits, (
            "the shared instruction core must name capability verbs, never a "
            "host's tool. Replace each with the verb named below:\n"
            + "\n".join(
                f"  {rel}:{line_no}: `{token}` -> {verb}\n      {text}"
                for rel, line_no, token, verb, text in hits
            )
        )

    def test_the_lint_actually_fires(self):
        """A synthetic violation of each class is detected.

        Without this, an over-narrowed pattern would silently pass the corpus.
        """
        for line, token in (
            ("Use AskUserQuestion to pick models", "AskUserQuestion"),
            ("call the Read tool on the plan", "Read"),
            ('spawn Task(subagent_type="general-purpose")', "Task"),
            ("read `${CLAUDE_SKILL_DIR}/providers.yaml`", "${CLAUDE_SKILL_DIR}"),
            ("spawn a Task subagent per batch", "Task"),
        ):
            fired = token in line and token in FORBIDDEN_TOKENS
            if not fired:
                fired = any(
                    re.search(p, line) for p in _ambiguous_patterns(token)
                )
            assert fired, f"the lint would not catch: {line!r}"

    def test_ordinary_english_passes(self):
        """The narrow rule: prose about reading and writing is not a violation."""
        for line in (
            "Read the plan file and note the marker.",
            "Write the summary to the path the orchestrator printed.",
            "Each task carries a complete prompt; run it as a subagent.",
            "| Task | Status |",
            "Task completion status is recorded in state.json.",
        ):
            for token in AMBIGUOUS_TOOL_TOKENS:
                assert not any(
                    re.search(p, line) for p in _ambiguous_patterns(token)
                ), f"false positive on {line!r} for {token}"

    def test_no_stale_lint_allowlist_entries(self):
        stale = []
        for rel, tokens in LINT_ALLOWLIST.items():
            path = SKILL_ROOT / rel
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            for token in tokens:
                present = token in text or any(
                    re.search(p, text) for p in _ambiguous_patterns(token)
                )
                if not present:
                    stale.append(f"{rel}:{token}")
        assert not stale, (
            "host-neutrality allowlist entries that no longer match anything "
            f"(delete them, or the allowlist decays into a blanket ignore): {stale}"
        )

    @pytest.mark.parametrize("verb", CAPABILITY_VERBS)
    def test_every_capability_verb_is_used_by_the_core(self, verb):
        """The vocabulary is closed *and* live: each verb appears in the corpus."""
        corpus = "\n".join(
            path.read_text(encoding="utf-8") for path in _lint_corpus_files()
        ).lower()
        assert verb in corpus, (
            f"the core never says {verb!r}; a verb nothing uses is not a "
            "vocabulary, it is a comment"
        )

    def test_the_verb_vocabulary_is_closed_at_three(self):
        assert len(CAPABILITY_VERBS) == 3, CAPABILITY_VERBS
        assert len(set(CAPABILITY_VERBS)) == 3


# ===========================================================================
# The mode table is the one owner
# ===========================================================================

class TestModeTable:
    def test_argument_hint_renders_from_the_table(self):
        text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        line = next(
            l for l in text.splitlines() if l.startswith("argument-hint:")
        )
        actual = line.split("argument-hint:", 1)[1].strip()
        assert actual == mode_table.render_argument_hint(), (
            "SKILL.md's argument-hint no longer matches modes.py. It is rendered "
            "from the table, never copied: add the mode as a row and regenerate."
        )

    def test_every_mode_word_is_dispatchable(self):
        for mode in mode_table.MODES:
            assert mode_table.resolve(mode.name) is mode
            if mode.unavailable is None:
                assert mode.target_module, mode.name

    def test_aliases_resolve_and_are_closed(self):
        for alias, target in mode_table.ALIASES.items():
            assert mode_table.resolve(alias).name == target
            assert alias not in mode_table.mode_names(), (
                f"alias {alias} shadows a real mode word"
            )
        # No prefix or fuzzy matching.
        assert mode_table.resolve("review-pl") is None
        assert mode_table.resolve("REVIEW-PLAN") is None

    def test_util_is_not_a_mode(self):
        assert mode_table.resolve(mode_table.UTIL_NAMESPACE) is None
        for mode in mode_table.MODES:
            assert mode_table.UTIL_NAMESPACE != mode.name
        # Excluded from --help's *mode list* (it is still named once, as the
        # pointer to `multi_llm.py util --help`).
        help_text = multi_llm.render_help()
        mode_block = help_text.split("modes:", 1)[1].split("aliases:", 1)[0]
        listed = [
            line.strip().split()[0]
            for line in mode_block.splitlines()
            if line.startswith("  ") and line.strip()
            and not line.strip().startswith("(")
        ]
        assert mode_table.UTIL_NAMESPACE not in listed, listed
        assert set(listed) == set(mode_table.mode_names())

    def test_internal_flags_are_hidden_from_help(self):
        help_text = multi_llm.render_help() + multi_llm.render_util_help()
        for flag in multi_llm.INTERNAL_FLAGS:
            assert flag not in help_text, f"{flag} must not appear in --help"

    def test_script_to_mode_covers_every_orchestrator(self):
        scripts = {p.name for p in SKILL_ROOT.glob("*_orchestrator.py")}
        assert set(mode_table.SCRIPT_TO_MODE) == scripts, (
            "every *_orchestrator.py needs a mode word so a printed follow-up "
            "can name the entry point instead of the script"
        )

    def test_prerequisite_phases_match_the_script(self):
        assert list(mode_table.PREREQUISITE_PHASES) == [
            "review-plan", "apply-suggestions", "generate-tasks", "review-tasks",
            "apply-task-suggestions", "implement", "code-review",
            "apply-code-fixes",
        ]


# ===========================================================================
# Import must be side-effect free
# ===========================================================================

def _production_python_files():
    excluded = {"tests", "__pycache__", ".venv"}
    return sorted(
        p
        for p in SKILL_ROOT.rglob("*.py")
        if not any(part in excluded or part.startswith(".") for part in
                   p.relative_to(SKILL_ROOT).parts)
    )


def _dispatch_target_modules():
    """Every module the dispatcher can import: modes plus the util namespace."""
    targets = {
        m.target_module for m in mode_table.MODES if m.target_module
    }
    targets |= {
        module for module, _attr, _summary in multi_llm.UTIL_COMMANDS.values()
        if module
    }
    return sorted(targets)


class TestDispatchTargetsImportCleanly:
    """Import is side-effect free, so two targets can share one interpreter.

    In-process dispatch plus machine-owned continuations means a single
    invocation can import and run more than one target; a module that parsed
    argv, printed, or wrote a file at import would make that unsafe.
    """

    def test_target_discovery_is_not_vacuous(self):
        targets = _dispatch_target_modules()
        assert len(targets) >= 14, targets
        assert "review_plan_orchestrator" in targets
        assert "utils.metrics" in targets

    def test_importing_every_target_is_silent_and_writes_nothing(self, tmp_path):
        """One process imports them all: no stdout, no file writes, no argv use."""
        targets = _dispatch_target_modules()
        probe = tmp_path / "probe.py"
        probe.write_text(
            textwrap.dedent(
                f"""
                import json, os, sys
                sys.path.insert(0, {str(SKILL_ROOT)!r})
                # A bogus argv: a module that reads sys.argv at import time or
                # parses it eagerly cannot survive this.
                sys.argv = ["import-probe"]
                before = sorted(os.listdir("."))
                import importlib
                for name in {targets!r}:
                    importlib.import_module(name)
                after = sorted(os.listdir("."))
                sys.stderr.write(json.dumps({{"before": before, "after": after}}))
                """
            ),
            encoding="utf-8",
        )
        workdir = tmp_path / "work"
        workdir.mkdir()
        proc = subprocess.run(
            [sys.executable, str(probe)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "", (
            "importing a dispatch target wrote to stdout, which would corrupt "
            f"the MULTI_LLM_RESULT contract:\n{proc.stdout}"
        )
        listing = json.loads(proc.stderr)
        assert listing["before"] == listing["after"] == [], (
            f"importing a dispatch target created files in the CWD: {listing}"
        )

    @pytest.mark.parametrize("module", _dispatch_target_modules())
    def test_no_module_level_argparse_or_argv(self, module):
        """AST guard: argparse and argv stay inside ``main(argv)``."""
        path = SKILL_ROOT / (module.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders = []

        def walk(node, inside_function):
            for child in ast.iter_child_nodes(node):
                nested = inside_function or isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                            ast.ClassDef)
                )
                if not nested and not _is_main_guard(child):
                    if isinstance(child, ast.Attribute) and child.attr == "argv":
                        offenders.append(("sys.argv", child.lineno))
                    if isinstance(child, ast.Call):
                        name = getattr(child.func, "attr", None) or getattr(
                            child.func, "id", None
                        )
                        if name in ("ArgumentParser", "parse_args"):
                            offenders.append((name, child.lineno))
                if not _is_main_guard(child):
                    walk(child, nested)

        walk(tree, False)
        assert not offenders, (
            f"{module}: argument parsing must live inside main(argv); found "
            f"{offenders} at module scope."
        )

    @pytest.mark.parametrize("module", _dispatch_target_modules())
    def test_main_accepts_an_argv_list(self, module):
        """``main(argv)`` is the contract; the __main__ block just exits on it."""
        import importlib
        import inspect

        target = getattr(importlib.import_module(module), "main")
        signature = inspect.signature(target)
        assert len(signature.parameters) >= 1, (
            f"{module}.main must accept an argv list so the dispatcher can hand "
            "it the residual argv verbatim"
        )


def _is_main_guard(node) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left, comparators = node.test.left, node.test.comparators
    names = {n.id for n in [left, *comparators] if isinstance(n, ast.Name)}
    consts = {n.value for n in [left, *comparators] if isinstance(n, ast.Constant)}
    return "__name__" in names and "__main__" in consts


# ===========================================================================
# The dispatcher's public grammar
# ===========================================================================

def _capturing_target(store):
    """A ``_load_target`` stand-in that records the residual argv verbatim."""

    def loader(module_path, attr):
        def target(argv):
            store["argv"] = list(argv)
            store["module"] = module_path
            return 0

        return target

    return loader


class TestDispatcherGrammar:
    def test_help_exits_zero_and_lists_every_mode(self, capsys):
        assert multi_llm.main(["--help"]) == 0
        out = capsys.readouterr().out
        for name in mode_table.mode_names():
            assert name in out

    def test_missing_mode_is_exit_three(self, capsys):
        assert multi_llm.main([]) == 3
        captured = capsys.readouterr()
        assert "review-plan" in captured.err

    def test_unknown_mode_is_exit_three_and_lists_modes(self, capsys):
        assert multi_llm.main(["reviewplan"]) == 3
        err = capsys.readouterr().err
        assert "unknown mode 'reviewplan'" in err
        assert "review-plan" in err

    def test_a_flag_in_the_mode_slot_is_exit_three(self, capsys):
        assert multi_llm.main(["--review-plan", "p.md"]) == 3
        assert "is not a mode" in capsys.readouterr().err

    def test_extra_positional_is_exit_three(self, capsys):
        assert multi_llm.main(["review-plan", "a.md", "b.md"]) == 3
        assert "positional" in capsys.readouterr().err

    def test_init_takes_no_positional(self, capsys):
        assert multi_llm.main(["init", "plans/x.md"]) == 3
        capsys.readouterr()

    @pytest.mark.parametrize(
        "argv",
        [
            ["review-plan", "p.md", "--yes", "--interactive"],
            ["review-plan", "p.md", "--non-interactive", "-i"],
            ["review-code", "p.md", "-y", "--interactive"],
            ["implement", "p.md", "--yes", "--force-strategy", "pause-and-ask"],
            ["implement", "p.md", "--yes", "--force-strategy=pause-and-ask"],
        ],
    )
    def test_contradictory_flags_are_exit_three(self, argv, capsys):
        assert multi_llm.main(argv) == 3
        err = capsys.readouterr().err
        assert "contradictory" in err or "forbids" in err

    def test_no_mode_is_unavailable_any_more(self):
        """The list is empty, and that is the point.

        ``generate-tasks`` and ``full`` left it when their orchestrators landed;
        ``install-host``/``uninstall-host`` left it when the D7 install contract
        did. Every mode word in the table now runs, so the dispatcher boundary a
        shim sees is the real one everywhere.
        """
        stubbed = [m.name for m in mode_table.MODES if m.unavailable]
        assert not stubbed, stubbed
        for mode in mode_table.MODES:
            assert mode.target_module, mode.name

    def test_the_install_modes_dispatch_to_the_installer(self, monkeypatch,
                                                         capsys):
        """Routed, not run: the real thing writes into a host config dir.

        ``tests/test_install_host.py`` exercises the installer itself against a
        temp ``$HOME``; here the only question is whether the two mode words reach
        it with the host id forwarded as a positional.
        """
        seen = {}

        def loader(module_path, attr):
            def target(argv):
                seen[attr] = (module_path, list(argv))
                return 0

            return target

        monkeypatch.setattr(multi_llm, "_load_target", loader)
        assert multi_llm.main(["install-host", "codex", "--check"]) == 0
        assert multi_llm.main(["uninstall-host", "codex", "--dry-run"]) == 0
        capsys.readouterr()
        assert seen["main"] == ("utils.host_install", ["codex", "--check"])
        assert seen["uninstall_main"] == (
            "utils.host_install", ["codex", "--dry-run"]
        )

    def test_no_workflow_mode_is_unavailable(self):
        for mode in mode_table.MODES:
            if not mode.user_facing:
                continue
            assert mode.unavailable is None, (
                f"user-facing mode '{mode.name}' is dispatchable but refuses to "
                "run; the dispatcher boundary a shim sees must be the real one"
            )
            assert mode.target_module, mode.name

    # -- alias table ----------------------------------------------------
    @pytest.mark.parametrize(
        "alias,expected",
        [("review", "review-plan"), ("code-review", "review-code"),
         ("tasks", "generate-tasks")],
    )
    def test_aliases_dispatch_to_the_canonical_mode(self, alias, expected,
                                                    monkeypatch, capsys):
        seen = {}

        def fake_load(module_path, attr):
            seen["module"] = module_path
            return lambda argv: 0

        monkeypatch.setattr(multi_llm, "_load_target", fake_load)
        exit_code = multi_llm.main([alias, "p.md"])
        result = cli_common.parse_result_line(capsys.readouterr().out)
        assert result["mode"] == expected
        assert exit_code == 0
        assert seen["module"] == mode_table.resolve(expected).target_module

    # -- positional normalization ---------------------------------------
    def test_plan_positional_becomes_plan_file(self, monkeypatch, capsys):
        captured = {}

        monkeypatch.setattr(multi_llm, "_load_target", _capturing_target(captured))
        assert multi_llm.main(["review-plan", "plans/a plan.md", "--quick"]) == 0
        capsys.readouterr()
        assert captured["argv"] == ["--plan-file", "plans/a plan.md", "--quick"]

    def test_status_dispatches_to_the_two_view_command(self, monkeypatch, capsys):
        """`status` is a real command now, not a `--status` injection.

        The instruction-only mode is retired: the plan positional is normalized
        the same way every plan mode's is, and nothing is injected — the views
        (`--view workflow|run|all`) replace the old flag.
        """
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capturing_target(captured))
        assert multi_llm.main(["status", "plans/a.md"]) == 0
        capsys.readouterr()
        assert captured["argv"] == ["--plan-file", "plans/a.md"]
        assert captured["module"] == "utils.run_status"
        assert mode_table.resolve("status").injected_flags == ()

    @pytest.mark.parametrize("alias", ["abort", "cancel", "stop"])
    def test_abort_alias_expands_to_status_plus_a_flag(self, alias, monkeypatch,
                                                      capsys):
        """The one alias that carries a flag, and the table owns the expansion."""
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capturing_target(captured))
        assert multi_llm.main([alias, "plans/a.md"]) == 0
        capsys.readouterr()
        assert captured["argv"] == ["--abort", "--plan-file", "plans/a.md"]
        assert captured["module"] == "utils.run_status"
        assert mode_table.alias_expansion(alias) == "status --abort"
        assert f"{alias} -> status --abort" in multi_llm.render_help()

    def test_short_ask_question_stays_on_argv(self, monkeypatch, capsys):
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capturing_target(captured))
        assert multi_llm.main(["ask", "plans/a.md", "Is the rollback ok?"]) == 0
        capsys.readouterr()
        assert captured["argv"] == [
            "--plan-file", "plans/a.md", "--question", "Is the rollback ok?",
        ]

    def test_long_ask_question_falls_back_to_question_file(self, monkeypatch,
                                                           capsys):
        question = "why? " * 2000
        assert len(question) > multi_llm.MAX_ARGV_QUESTION_CHARS
        captured = {}

        def fake_main(argv):
            captured["argv"] = list(argv)
            path = Path(argv[argv.index("--question-file") + 1])
            captured["content"] = path.read_text(encoding="utf-8")
            captured["existed"] = path.exists()
            captured["path"] = path
            return 0

        monkeypatch.setattr(multi_llm, "_load_target", lambda m, a: fake_main)
        assert multi_llm.main(["ask", "plans/a.md", question]) == 0
        capsys.readouterr()
        assert "--question" not in captured["argv"]
        assert "--question-file" in captured["argv"]
        assert captured["content"] == question
        assert captured["existed"]
        # The dispatcher owns the temp file's whole lifetime.
        assert not captured["path"].exists()

    # -- pass-through ---------------------------------------------------
    def test_flags_are_forwarded_verbatim_as_a_list(self, monkeypatch, capsys):
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capturing_target(captured))
        argv = [
            "review-code", "plans/my plan.md", "--models", "cursor-agent:auto",
            "gemini:gemini-2.5-flash", "--base-ref", "HEAD~1",
            "--host", "codex", "--internal-validation",
        ]
        assert multi_llm.main(argv) == 0
        capsys.readouterr()
        assert captured["argv"] == ["--plan-file", "plans/my plan.md", *argv[2:]]

    def test_printed_follow_up_round_trips_through_the_dispatcher(self, monkeypatch,
                                                                capsys):
        """The command the orchestrators print is one the dispatcher accepts.

        Closes the loop between printer and parser: a follow-up that no longer
        parses is the exact failure the resolved-path change could introduce, and
        nothing else in the suite compares the two ends.
        """
        plan = "/tmp/a plan with spaces.md"
        command = cli_common.reaggregate_command("review-code", plan)
        argv = shlex.split(command)
        # Drop `uv run --project <dir> -- python <dir>/multi_llm.py`.
        residual = argv[argv.index("--") + 3:]
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capturing_target(captured))
        assert multi_llm.main(residual) == 0
        capsys.readouterr()
        assert captured["argv"] == ["--plan-file", plan, "--reaggregate"]
        assert captured["module"] == "code_review_orchestrator"

    def test_no_subprocess_on_the_dispatch_path(self):
        source = (SKILL_ROOT / "multi_llm.py").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert "import subprocess" not in code and "subprocess." not in code, (
            "dispatch is in-process by design: a subprocess per mode pays a "
            "second uv resolution, turns a self-contained continuation back "
            "into a re-exec and reintroduces argv quoting and signal plumbing."
        )

    # -- exit codes -----------------------------------------------------
    @pytest.mark.parametrize("code", [0, 1, 2, 3, 7])
    def test_target_exit_codes_are_not_remapped(self, code, monkeypatch, capsys):
        monkeypatch.setattr(
            multi_llm, "_load_target", lambda m, a: lambda argv: code
        )
        assert multi_llm.main(["review-plan", "p.md"]) == code
        assert cli_common.parse_result_line(capsys.readouterr().out)["exit"] == code

    def test_target_system_exit_is_honoured(self, monkeypatch, capsys):
        def raiser(argv):
            raise SystemExit(2)

        monkeypatch.setattr(multi_llm, "_load_target", lambda m, a: raiser)
        assert multi_llm.main(["review-plan", "p.md"]) == 2
        result = cli_common.parse_result_line(capsys.readouterr().out)
        assert result["exit"] == 2
        assert result["state"] == "nothing-to-do"

    # -- MULTI_LLM_RESULT ----------------------------------------------
    def test_result_line_is_last_and_parseable_on_success(self, monkeypatch,
                                                          capsys):
        def target(argv):
            print("human readable progress")
            cli_common.record_artifact("/tmp/a plan/review-plan/grouped.json")
            return 0

        monkeypatch.setattr(multi_llm, "_load_target", lambda m, a: target)
        assert multi_llm.main(["review-plan", "p.md"]) == 0
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.strip()]
        assert lines[-1].startswith(cli_common.RESULT_MARKER)
        assert sum(l.startswith(cli_common.RESULT_MARKER) for l in lines) == 1
        payload = cli_common.parse_result_line(out)
        assert payload == {
            "mode": "review-plan",
            "phase": "review-plan",
            "state": "done",
            "exit": 0,
            "artifacts": ["/tmp/a plan/review-plan/grouped.json"],
            "next": None,
        }

    def test_result_line_is_present_and_parseable_on_a_crash(self, monkeypatch,
                                                             capsys):
        def boom(argv):
            print("partial output")
            raise RuntimeError("kaboom")

        monkeypatch.setattr(multi_llm, "_load_target", lambda m, a: boom)
        assert multi_llm.main(["review-code", "p.md"]) == 1
        captured = capsys.readouterr()
        payload = cli_common.parse_result_line(captured.out)
        assert payload["exit"] == 1
        assert payload["state"] == "error"
        assert payload["mode"] == "review-code"
        assert payload["phase"] == "code-review"
        assert captured.out.strip().splitlines()[-1].startswith(
            cli_common.RESULT_MARKER
        )
        assert "kaboom" in captured.err, "the traceback belongs on stderr"

    def test_result_keys_are_exactly_the_contract(self, monkeypatch, capsys):
        monkeypatch.setattr(multi_llm, "_load_target", lambda m, a: lambda argv: 0)
        multi_llm.main(["ask", "p.md", "q?"])
        payload = cli_common.parse_result_line(capsys.readouterr().out)
        assert list(payload) == [
            "mode", "phase", "state", "exit", "artifacts", "next",
        ]

    def test_every_invocation_emits_exactly_one_result_line(self, capsys):
        for argv in ([], ["--help"], ["bogus"], ["util", "--help"],
                     ["review-plan", "a.md", "b.md"]):
            multi_llm.main(list(argv))
            out = capsys.readouterr().out
            assert sum(
                l.startswith(cli_common.RESULT_MARKER) for l in out.splitlines()
            ) == 1, argv

    # -- util namespace -------------------------------------------------
    def test_util_help_lists_every_command(self, capsys):
        assert multi_llm.main(["util", "--help"]) == 0
        out = capsys.readouterr().out
        for name in multi_llm.UTIL_COMMANDS:
            assert name in out

    def test_unknown_util_command_is_exit_three(self, capsys):
        assert multi_llm.main(["util", "nope"]) == 3
        assert "unknown util command" in capsys.readouterr().err

    def test_util_routes_in_process_to_the_script(self, tmp_path, capsys):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        assert multi_llm.main([
            "util", "metrics", "start", "--state-file", str(state),
            "--phase", "review-plan", "--total-batches", "3",
        ]) == 0
        capsys.readouterr()
        assert "metrics" in json.loads(state.read_text(encoding="utf-8"))


class TestCrossCuttingFlagsAreShared:
    """The cross-cutting flags exist on the direct surface too, unchanged."""

    #: The orchestrators whose parser is reachable as ``parse_args(argv)``.
    #: ``check_workflow_prerequisites`` builds its parser inside ``main()``, so the
    #: dedicated test below drives its CLI surface instead.
    DIRECT_PARSERS = [
        "review_plan_orchestrator", "review_tasks_orchestrator",
        "code_review_orchestrator", "ask_orchestrator",
        "implement_orchestrator",
    ]

    @pytest.mark.parametrize("module", DIRECT_PARSERS)
    def test_direct_invocation_accepts_the_shared_flags(self, module):
        import importlib

        mod = importlib.import_module(module)
        argv = ["--plan-file", "p.md", "--host", "codex", "--detach",
                "--exec-subagents", "--yes"]
        if module == "ask_orchestrator":
            argv += ["--question", "q"]
        args = mod.parse_args(argv)
        assert args.host == "codex"
        assert args.detach is True
        assert args.exec_subagents is True
        assert args.yes is True

    @pytest.mark.parametrize("module", DIRECT_PARSERS)
    def test_interactive_keeps_its_short_flag_and_semantics(self, module):
        import importlib

        mod = importlib.import_module(module)
        for spelling in (["--interactive"], ["-i"]):
            args = mod.parse_args(["--plan-file", "p.md", *spelling]
                                  + (["--question", "q"] if module == "ask_orchestrator" else []))
            assert args.interactive is True

    def test_status_mode_accepts_the_shared_flags(self, tmp_path, capsys):
        """The `status` row of the dispatch table, end-to-end.

        Its parser is built inside ``main()``, so drive it through the mode. The
        workflow view is the retired ``--status`` mode's own implementation, so its
        JSON (and its exit 0) survive for a caller that never detaches.
        """
        plan = tmp_path / "a plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        exit_code = multi_llm.main([
            "status", str(plan), "--view", "workflow", "--json",
            "--host", "codex", "--yes", "--exec-subagents",
        ])
        out = capsys.readouterr().out
        assert exit_code == 0, out
        body = out.rsplit(cli_common.RESULT_MARKER, 1)[0]
        assert json.loads(body)["phases"]
        assert list(json.loads(body)["phases"]) == list(
            mode_table.PREREQUISITE_PHASES
        )

    def test_prerequisites_script_still_accepts_the_shared_flags(self, tmp_path,
                                                                capsys):
        """`check_workflow_prerequisites.py --status` is untouched by the move.

        It survives as the workflow view's implementation and stays directly
        callable — same flags, same JSON, same exit code.
        """
        import check_workflow_prerequisites as prereq

        plan = tmp_path / "a plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            prereq.main([
                "--plan-file", str(plan), "--status", "--host", "codex",
                "--yes", "--detach", "--exec-subagents",
            ])
        assert excinfo.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert list(payload["phases"]) == list(mode_table.PREREQUISITE_PHASES)

    def test_apply_parsers_keep_yes_and_its_short_flag(self):
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator

        for spelling in ("--yes", "-y", "--non-interactive"):
            args = ApplySuggestionsOrchestrator.parse_args(
                ["--plan-file", "p.md", spelling]
            )
            assert args.yes is True

    def test_parent_parser_defines_each_flag_once(self):
        parent = cli_common.build_parent_parser()
        options = [o for action in parent._actions for o in action.option_strings]
        assert len(options) == len(set(options)), options
        assert {"--host", "--yes", "-y", "--non-interactive", "--interactive",
                "-i", "--detach", "--exec-subagents"} <= set(options)

    def test_skill_dir_preserves_symlinks(self):
        """``abspath``, never ``realpath``: resolving would rewrite a plugin
        invocation into a checkout path the user never named."""
        source = (SKILL_ROOT / "utils" / "cli_common.py").read_text(encoding="utf-8")
        assert "os.path.abspath" in source
        assert "os.path.realpath" not in source
        assert ".resolve()" not in source
        assert cli_common.skill_dir().is_absolute()
        assert cli_common.skill_dir() == Path(
            os.path.abspath(str(SKILL_ROOT / "utils" / "..")))
