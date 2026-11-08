"""The conditional runbooks a core names stay reachable from it.

A core instruction file (``instructions/<mode>.md``) is the text a host keeps in
context for a whole phase, so the steps a clean run never takes live in
``references/``: the core keeps one trigger sentence that says when to read the
file and names it. That trade is safe only while three things hold, and each one
can rot without any other test noticing:

1. **Every ``references/<name>.md`` a core names exists.** A renamed or deleted
   runbook leaves a trigger pointing at nothing, and the host improvises the
   branch.
2. **Every moved block keeps its trigger in the core, and only its trigger.**
   The trigger names the file *and* every condition that sends a host there
   (each marker, flag or state the moved block handled, pending work
   included), so a host that meets any of them finds the trigger by searching
   for it; the block's own text is gone from the core, so there is one copy to
   keep right. The inventory below is checked against the markers each
   reference handles, so a marker cannot sit in a runbook without the test
   saying how a host reaches it, and a sentence saying when to skip the branch
   must name every way in: a skip clause that names some of them excludes the
   rest.
3. **A reference carries no template token.** A host reads a reference as it
   is, never through ``render_instruction_text``, so a ``{SKILL_DIR}`` or
   ``{UV_RUN_FLAGS}`` there would reach the agent unsubstituted. Commands in a
   reference are spelled behind a ``<prefix>`` the file itself defines.

See the "Host-neutral instruction core" section of ``docs/development/hosts.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import re
import sys
from pathlib import Path
from typing import NamedTuple, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import apply_code_fixes_orchestrator  # noqa: E402
import apply_commit_fixes_orchestrator  # noqa: E402
import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, host_capabilities  # noqa: E402

INSTRUCTIONS = SKILL_ROOT / "instructions"
REFERENCES = SKILL_ROOT / "references"

#: A reference named in prose: ``references/<name>.md``.
REFERENCE_RE = re.compile(r"references/([A-Za-z0-9_-]+\.md)")

#: Every token ``render_instruction_text`` substitutes, in each spelling an
#: authored file has used. None may appear in a file a host reads raw.
TEMPLATE_TOKENS = (
    "{SKILL_DIR}",
    "${SKILL_DIR}",
    "${CLAUDE_SKILL_DIR}",
    "{UV_RUN_FLAGS}",
    "{DEFAULT_MODE}",
    "{NEEDS_HUMAN_RULE}",
)


class Moved(NamedTuple):
    """One conditional block moved out of a core into a reference.

    ``enters`` is every condition that sends a host into the reference: a
    marker, flag or state it sees, any one of which is enough. The core's
    trigger, the reference's opening and every sentence of either that says
    when to skip the branch name all of them. ``inline`` is what the trigger
    carries besides, because a host acts on it without the read: the branch's
    `--yes` resolution (its D6 row), the notices the trigger answers itself,
    the flags it names as the way forward. ``inside`` holds the markers a host
    meets only once it is following the reference (a retry's own lines, a
    reply check). Every marker the reference handles is in one of the three.
    ``anchor`` is a phrase of the moved block, which must now be in the
    reference and no longer in the core.
    """

    core: str
    reference: str
    enters: Tuple[str, ...]
    inline: Tuple[str, ...]
    inside: Tuple[str, ...]
    anchor: str


#: Consolidation is entered two ways: a recommendation starts a pass, and an
#: explicit `--consolidate` (below the threshold too) or a pending marker
#: continues one that is already decided.
_CONSOLIDATION = Moved(
    "review-plan.md", "consolidation.md",
    enters=("[CONSOLIDATION_RECOMMENDED]", "consolidation_recommended", "--consolidate",
            "[CONSOLIDATION_PENDING]", "[CONSOLIDATION_BATCHES_PENDING]"),
    inline=("--yes", "--non-interactive", "D6 row 4"),
    inside=(),
    anchor="--reaggregate-consolidation",
)


def _interrupted(core: str) -> Moved:
    return Moved(
        core, "interrupted-run-recovery.md",
        enters=("interrupted", "orchestrator-run.log"),
        inline=("--yes", "--non-interactive", "D6 row 3"),
        inside=("[SKIP]", "[PARTIAL_RESULTS:"),
        anchor="`models_complete` / `grouping_complete`",
    )


def _revalidation(core: str) -> Moved:
    """Failed items print no marker until a retry is requested, so they are a
    way in of their own beside the two pending markers."""
    return Moved(
        core, "revalidation.md",
        enters=("validation_failed", "[REVALIDATION_PENDING]", "[REVALIDATION_BATCHES_PENDING]"),
        inline=("validation_failed_count", "--revalidate", "--revalidate-model",
                "--approve-validation-failed"),
        inside=("[REPLY_INVALID]", "[REPLY_MISSING]"),
        anchor="`item_indices` and `original_validation`",
    )


#: The Python write executor's ways in: a flag on the command that runs the
#: phase, never a marker. `--detach` counts because the dispatcher writes the
#: `--exec-subagents` it implies out, and `--dry-run-subagents` because it is a
#: preflight through the same scheduler. `TestTheWriteExecutorIsReachable`
#: holds this list to the real dispatcher on every shipped host: a host's
#: capability decides none of it (docs/development/hosts.md, "Host-neutral
#: instruction core").
WRITE_EXECUTOR_WAYS_IN = ("--exec-subagents", "--detach", "--dry-run-subagents")


def _write_executor(core: str, reference: str, inline: Tuple[str, ...], anchor: str) -> Moved:
    """The trigger also carries what a host must not miss even on the default
    route: an abbreviation of a flag counts, and a run on this route (a dry run
    included, which dispatched nothing) leaves no work for the core's own
    steps, each named in *inline*. Those rules live in the trigger alone; the
    reference holds the rest (`KEPT_RULES`), so the route carries each once."""
    return Moved(
        core, reference,
        enters=WRITE_EXECUTOR_WAYS_IN,
        inline=("which implies it", "abbreviation", "dispatched nothing and leaves no work")
        + inline,
        inside=(),
        anchor=anchor,
    )


_APPLY_EXECUTOR_ANCHOR = "the prior-batch context bound late per dispatch"
#: The native steps an executor run already did, which the host must not run again.
_APPLY_NOT_TWICE = ("do not run the loop below", "step 3's `util decisions`",
                    "summary report", "`--mark-completed`")

#: The references the executor sections moved to.
EXECUTOR_REFERENCES = ("apply-write-executor.md", "implement-write-executor.md")

MOVED = (
    _write_executor("apply-suggestions.md", "apply-write-executor.md",
                    _APPLY_NOT_TWICE, _APPLY_EXECUTOR_ANCHOR),
    _write_executor("apply-task-suggestions.md", "apply-write-executor.md",
                    _APPLY_NOT_TWICE, _APPLY_EXECUTOR_ANCHOR),
    _write_executor("apply-code-fixes.md", "apply-write-executor.md",
                    _APPLY_NOT_TWICE + ("step 7's verification",), _APPLY_EXECUTOR_ANCHOR),
    _write_executor("implement.md", "implement-write-executor.md",
                    ("run none of steps 5 to 10",
                     "no subagent, metrics, summary, tracking or `--mark-completed`",
                     "step 11"),
                    "re-dispatches only the unfinished batches"),
    _CONSOLIDATION,
    _interrupted("review-plan.md"),
    _interrupted("review-tasks.md"),
    Moved("review-code.md", "review-code-apply-fixes.md",
          enters=("--apply-fixes",),
          inline=("fix_tasks.json",),
          inside=("[ETA]",),
          anchor="Implementation Status: NOT FIXED"),
    _revalidation("apply-suggestions.md"),
    _revalidation("apply-code-fixes.md"),
    Moved("implement.md", "implement-prerequisites.md",
          enters=("[PREREQUISITE_UNMET]", "[PREREQUISITE_CHECK]", "[TASKS_MISSING]",
                  "[TASK_SUGGESTIONS_ADVISORY]"),
          inline=("--yes", "--no-confirm", "[PREREQUISITE_SKIPPED]",
                  "[PREREQUISITE_SKIPPED_UPSTREAM", "[AUTO_RESOLVED] task-suggestion-advisory"),
          inside=(),
          anchor="consent to run unattended is not consent to stale input"),
)

#: Every rule the executor sections held that their trigger does not carry, as
#: a phrase the reference must still hold: each flag they named, then the words
#: of each rule, so a later edit cannot drop one without this list saying
#: which. What a host must not run again on this route is the trigger's
#: (`_write_executor`'s *inline*), never repeated here.
KEPT_RULES = {
    "apply-write-executor.md": (
        "--exec-subagents", "--dry-run-subagents", "--detach", "--phase-dir",
        "--task-id", "--all-failed", "--force",
        # one batch at a time, each verified and recorded before the next
        "ascending `batch_index`", "recorded **before** the next is dispatched",
        "bound late per dispatch",
        # concurrency limits, per mode
        "pinned to 1", "provably disjoint",
        "marked completed only after every batch verified",
        # the closing steps the run does itself, apply-code-fixes' verification included
        "prints step 3's breakdown itself", "the review report",
        "runs `driver.verify_command` once after the last batch",
        "ends the phase at exit 3 when it fails",
        # a dry run dispatches nothing, and its file is what is reported
        "**zero dispatch**", "report what that file holds",
        # detach is never wrapped in a host's own background mechanism
        "never passes `--detach`",
        # rollback is a human's call, and a changed file is refused
        "A partial application never reverts itself", "unless `--force` is passed",
    ),
    "implement-write-executor.md": (
        "--force-strategy", "--yes", "--human-task", "--resume", "--force",
        "--phase-dir", "--task-id", "--dry-run-subagents", "--detach",
        # scheduling
        "ascending `batch_index`", "pairwise disjoint", "a task declaring no file set runs alone",
        # the human-task strategy, resolved once and persisted
        "resolved **once**", "persisted in `state.json`", "rather than blocking on a stdin",
        # finalization is part of the contract
        "**failed** run, not a partial one",
        # a stop keeps what a human needs and resumes only what is unfinished
        "keeps every pre-batch snapshot", "re-dispatches only the unfinished batches",
        "account for every task of the decomposition",
        # a failed check: retry, accept or skip, and no dispatch on an unchanged tree
        "`--human-task <id>=done`", "`--human-task <id>=retry`", "`--human-task <id>=skip`",
        "no dispatch is spent", "is not a stop",
        # a dry run dispatches nothing, and its file is what is reported
        "starts no child", "report what that file holds",
        "Never combine it with a host's own background mechanism",
    ),
}

#: A printed marker, up to its first non-name character: `[PARTIAL_RESULTS: m]`
#: and `[PREREQUISITE_SKIPPED_UPSTREAM apply-suggestions: …]` are their stems.
MARKER_RE = re.compile(r"\[[A-Z][A-Z0-9_]+")

#: A sentence that says when a branch is not taken ("skip this step", "never
#: needs this file", "with none of these four"), not one that merely mentions
#: a recorded skip.
SKIP_RE = re.compile(
    r"\bskips? (?:this|it|that|consolidation|the (?:step|reference|branch|file))\b"
    r"|\bnever needs? (?:this|it)\b|\bwith (?:none|neither)\b",
    re.IGNORECASE,
)

CODE_SPAN_RE = re.compile(r"`[^`]*`")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _blocks(text: str):
    """Blank-line separated blocks, whitespace flattened: a trigger and the
    list of its cases are one block."""
    return [_flat(block) for block in re.split(r"\n[ \t]*\n", text) if block.strip()]


def _naming_blocks(text: str, reference: str):
    return [block for block in _blocks(text) if f"references/{reference}" in block]


def _trigger_blocks(text: str, reference: str):
    return [block for block in _naming_blocks(text, reference) if "in full" in block]


def _opening(reference_text: str) -> str:
    """A reference's text before its first section or its `<prefix>` note."""
    head = re.split(r"^(?:## |\*\*Commands\.\*\*)", reference_text, maxsplit=1, flags=re.M)[0]
    return _flat(head)


def _skip_sentences(flat_text: str):
    """Sentences that say when the branch is not taken. Code spans are blanked
    first, so `--skip-prerequisites` or `[PREREQUISITE_SKIPPED]` is no skip."""
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", flat_text)
            if SKIP_RE.search(CODE_SPAN_RE.sub("", sentence))]


def _stems(items) -> set:
    return {match.group(0) for item in items for match in [MARKER_RE.match(item)] if match}


def _reference_files():
    return sorted(REFERENCES.glob("*.md"))


def _naming_files():
    return sorted(INSTRUCTIONS.glob("*.md")) + _reference_files() + [SKILL_ROOT / "SKILL.md"]


class TestEveryNamedReferenceExists:
    def test_the_scan_is_not_vacuous(self):
        named = {m for path in _naming_files() for m in REFERENCE_RE.findall(_read(path))}
        assert {moved.reference for moved in MOVED} <= named
        assert len(named) >= 10, sorted(named)

    @pytest.mark.parametrize("path", _naming_files(), ids=lambda p: p.name)
    def test_each_named_reference_is_on_disk(self, path):
        missing = sorted(
            name for name in set(REFERENCE_RE.findall(_read(path)))
            if not (REFERENCES / name).is_file()
        )
        assert not missing, f"{path.name} names references that do not exist: {missing}"


def _missing(text: str, items) -> list:
    return [item for item in items if item not in text]


class TestEveryMovedBlockKeepsItsTrigger:
    @pytest.mark.parametrize("moved", MOVED, ids=lambda m: f"{m.core}->{m.reference}")
    def test_the_core_names_the_reference_with_every_condition(self, moved):
        core = _read(INSTRUCTIONS / moved.core)
        assert _naming_blocks(core, moved.reference), (
            f"{moved.core} no longer names references/{moved.reference}")
        triggers = _trigger_blocks(core, moved.reference)
        assert triggers, (
            f"{moved.core}: the trigger for references/{moved.reference} must say "
            "to read the file in full"
        )
        wanted = moved.enters + moved.inline
        assert any(not _missing(block, wanted) for block in triggers), (
            f"{moved.core}: the trigger for references/{moved.reference} omits "
            f"{min((_missing(block, wanted) for block in triggers), key=len)}; a "
            "trigger that leaves out a way in is one the host cannot match to "
            "what it sees"
        )

    @pytest.mark.parametrize("moved", MOVED, ids=lambda m: f"{m.core}->{m.reference}")
    def test_the_reference_opening_admits_every_way_in(self, moved):
        opening = _opening(_read(REFERENCES / moved.reference))
        missing = _missing(opening, moved.enters)
        assert not missing, (
            f"references/{moved.reference} opens by saying when to read it but "
            f"leaves out {missing}: a host sent there that way is told the file "
            "is not for it"
        )

    @pytest.mark.parametrize("moved", MOVED, ids=lambda m: f"{m.core}->{m.reference}")
    def test_a_skip_sentence_names_every_way_in(self, moved):
        """A sentence saying when the branch is not taken names every way in.
        One about the markers alone ("a run that printed neither marker skips
        this") under a trigger that also enters on failed items, which no
        marker announces, would exclude them."""
        texts = [(f"{moved.core}", block)
                 for block in _naming_blocks(_read(INSTRUCTIONS / moved.core), moved.reference)]
        texts.append((f"references/{moved.reference} (opening)",
                      _opening(_read(REFERENCES / moved.reference))))
        bad = [(where, sentence, _missing(sentence, moved.enters))
               for where, text in texts
               for sentence in _skip_sentences(text) if _missing(sentence, moved.enters)]
        assert not bad, "\n".join(
            f"{where}: {sentence!r} leaves out {missing}" for where, sentence, missing in bad)

    @pytest.mark.parametrize("moved", MOVED, ids=lambda m: f"{m.core}->{m.reference}")
    def test_the_trigger_and_the_opening_say_when_to_skip(self, moved):
        """Keeps the skip-sentence check from passing on nothing to check."""
        triggers = _trigger_blocks(_read(INSTRUCTIONS / moved.core), moved.reference)
        assert any(_skip_sentences(block) for block in triggers), (
            f"{moved.core}: the trigger for references/{moved.reference} never says "
            "when the branch is not taken")
        assert _skip_sentences(_opening(_read(REFERENCES / moved.reference))), (
            f"references/{moved.reference} never says when a host does not need it")

    @pytest.mark.parametrize("moved", MOVED, ids=lambda m: f"{m.core}->{m.reference}")
    def test_the_inventory_accounts_for_every_marker_the_reference_handles(self, moved):
        handled = set(MARKER_RE.findall(_read(REFERENCES / moved.reference)))
        listed = _stems(moved.enters + moved.inline + moved.inside)
        unlisted = sorted(handled - listed)
        assert not unlisted, (
            f"references/{moved.reference} handles {unlisted}, which the inventory "
            "here does not place: add each to `enters` (and name it in the trigger), "
            "to `inline`, or to `inside` if a host meets it only inside the runbook"
        )
        stale = sorted(_stems(moved.enters + moved.inside) - handled)
        assert not stale, (
            f"references/{moved.reference} no longer handles {stale}; drop them here")

    @pytest.mark.parametrize("moved", MOVED, ids=lambda m: f"{m.core}->{m.reference}")
    def test_the_block_lives_in_the_reference_alone(self, moved):
        assert moved.anchor in _flat(_read(REFERENCES / moved.reference)), (
            f"references/{moved.reference} lost the block it holds ({moved.anchor!r})"
        )
        assert moved.anchor not in _flat(_read(INSTRUCTIONS / moved.core)), (
            f"{moved.core} carries the moved block again ({moved.anchor!r}); keep "
            f"it in references/{moved.reference} and only the trigger here"
        )

    @pytest.mark.parametrize("moved", MOVED, ids=lambda m: f"{m.core}->{m.reference}")
    def test_the_served_core_keeps_the_trigger(self, moved):
        """What `util instructions` serves is the rendered core: the trigger
        survives rendering, with every condition intact."""
        rendered = cli_common.render_instruction_text(
            _read(INSTRUCTIONS / moved.core), "/skill dir")
        wanted = moved.enters + moved.inline
        assert any(not _missing(block, wanted)
                   for block in _trigger_blocks(rendered, moved.reference))


class TestReferencesAreReadAsTheyAre:
    @pytest.mark.parametrize("path", _reference_files(), ids=lambda p: p.name)
    def test_no_template_token(self, path):
        found = [token for token in TEMPLATE_TOKENS if token in _read(path)]
        assert not found, (
            f"references/{path.name} carries {found}: a host reads this file "
            "raw, so the token would reach the agent unsubstituted. Spell a "
            "command behind `<prefix>` instead"
        )

    @pytest.mark.parametrize("name", sorted({m.reference for m in MOVED}))
    def test_a_runbook_says_when_to_read_it(self, name):
        head = _read(REFERENCES / name).split("\n\n", 2)
        assert "Read this only when" in " ".join(head[:2]), (
            f"references/{name} must open by saying when a host reads it"
        )

    @pytest.mark.parametrize(
        "path", [p for p in _reference_files() if "<prefix> " in _read(p)],
        ids=lambda p: p.name)
    def test_a_prefix_command_is_defined(self, path):
        assert "`<prefix>` stands for the dispatcher prefix every command in the core" in _flat(
            _read(path)), (
            f"references/{path.name} runs `<prefix>` commands without saying what "
            "`<prefix>` is"
        )


class TestTheExecutorRulesStayReachable:
    @pytest.mark.parametrize("name", sorted(KEPT_RULES))
    def test_each_rule_is_in_its_reference(self, name):
        text = _flat(_read(REFERENCES / name))
        lost = [phrase for phrase in KEPT_RULES[name] if phrase not in text]
        assert not lost, f"references/{name} no longer carries {lost}"

    def test_every_executor_reference_has_its_rules(self):
        assert set(KEPT_RULES) == set(EXECUTOR_REFERENCES)
        assert {m.reference for m in MOVED if m.enters == WRITE_EXECUTOR_WAYS_IN} == set(
            EXECUTOR_REFERENCES)


# ---------------------------------------------------------------------------
# The executor's ways in, held to the real dispatcher
# ---------------------------------------------------------------------------

#: An argparse abbreviation of each way in: the dispatcher reads flags the way
#: argparse does (`cli_common.flag_present`), so each is a way in too.
ABBREVIATIONS = {"--exec-subagents": "--exec", "--detach": "--det",
                 "--dry-run-subagents": "--dry-run-sub"}

#: The modes whose cores carry an executor trigger, each with that core.
EXECUTOR_CORES = {
    Path(moved.core).stem: moved.core
    for moved in MOVED if moved.reference in EXECUTOR_REFERENCES
}


def _parse(mode: str, argv) -> argparse.Namespace:
    """*argv* parsed by *mode*'s own orchestrator, as its ``main`` parses it."""
    if mode == "apply-commit-fixes":
        # The commit front door forwards what it does not own to
        # apply-code-fixes' parser, the core it loads.
        forwarded = apply_commit_fixes_orchestrator.parse_args(list(argv)).remaining
        return apply_code_fixes_orchestrator.ApplyCodeFixesOrchestrator.parse_args(
            list(forwarded) + ["--plan-file", "brief.md"])
    module = importlib.import_module(mode_table.resolve(mode).target_module)
    owners = [value for value in vars(module).values()
              if isinstance(value, type) and value.__module__ == module.__name__
              and "parse_args" in vars(value)]
    assert len(owners) <= 1, owners
    return (owners[0] if owners else module).parse_args(list(argv))


def _takes_the_executor(args: argparse.Namespace) -> bool:
    """The two flags every write orchestrator's executor leg reads
    (``run_write_executor`` in the apply base, ``build_handoff`` in implement)."""
    return bool(getattr(args, "exec_subagents", False)
                or getattr(args, "dry_run_subagents", False))


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    """Run the real dispatcher with a recording target in place of the mode's
    orchestrator, and return the argv that target would have parsed (``None``
    when the dispatcher refused the command before reaching it)."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    seen = {}

    def loader(module_path, attr):
        def target(argv):
            seen["argv"] = list(argv)
            return 0

        return target

    monkeypatch.setattr(multi_llm, "_load_target", loader)

    def run(mode, host, flags):
        seen.clear()
        head = [mode] if mode == "apply-commit-fixes" else [mode, str(plan)]
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                multi_llm.main(head + ["--host", host] + list(flags))
            except SystemExit:
                pass
        return seen.get("argv")

    yield run
    cli_common.clear_invocation()


def _routes(mode, argv) -> bool:
    if argv is None:
        return False
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return _takes_the_executor(_parse(mode, argv))
    except SystemExit:  # the mode's own parser refuses it: no route at all
        return False


def _mode_parser(mode: str, monkeypatch) -> argparse.ArgumentParser:
    """The parser *mode*'s orchestrator builds, caught as it starts parsing."""
    caught = {}

    class Caught(Exception):
        pass

    def grab(self, *args, **kwargs):
        caught["parser"] = self
        raise Caught()

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", grab)
        with pytest.raises(Caught):
            _parse(mode, ["--plan-file", "plan.md"])
    return caught["parser"]


class TestTheWriteExecutorIsReachable:
    """Each way in a trigger names really takes the executor, through the real
    dispatcher, on every shipped host — Claude Code's ``subagents: native`` is
    its default, not a prohibition — and no other option does. A way in this
    misses is one the trigger would let a host skip."""

    @pytest.mark.parametrize("host", host_capabilities.host_ids())
    @pytest.mark.parametrize("mode", sorted(EXECUTOR_CORES))
    @pytest.mark.parametrize("flag", WRITE_EXECUTOR_WAYS_IN)
    def test_each_way_in_takes_the_executor(self, dispatch, mode, host, flag):
        assert _routes(mode, dispatch(mode, host, [flag])), (
            f"{mode} {flag} on {host} no longer reaches the executor; drop it "
            "from WRITE_EXECUTOR_WAYS_IN and from the trigger")
        assert _routes(mode, dispatch(mode, host, [ABBREVIATIONS[flag]]))

    @pytest.mark.parametrize("host", host_capabilities.host_ids())
    @pytest.mark.parametrize("mode", sorted(EXECUTOR_CORES))
    def test_the_default_route_does_not(self, dispatch, mode, host):
        argv = dispatch(mode, host, [])
        assert argv is not None and not _routes(mode, argv)

    @pytest.mark.parametrize("flag", WRITE_EXECUTOR_WAYS_IN)
    def test_the_commit_front_door_forwards_each_way_in(self, dispatch, flag):
        """apply-commit-fixes runs apply-code-fixes' phase and loads its core."""
        assert _routes("apply-commit-fixes", dispatch("apply-commit-fixes", "claude-code", [flag]))
        assert not _routes("apply-commit-fixes", dispatch("apply-commit-fixes", "claude-code", []))

    @pytest.mark.parametrize("mode", sorted(EXECUTOR_CORES))
    def test_no_other_option_takes_it(self, dispatch, mode, monkeypatch):
        parser = _mode_parser(mode, monkeypatch)
        routed, tried = set(), 0
        for action in parser._actions:
            longs = [o for o in action.option_strings if o.startswith("--")]
            if not longs or isinstance(action, argparse._HelpAction):
                continue
            option = longs[0]
            flags = [option] if action.nargs == 0 else [option, "1"]
            tried += 1
            if _routes(mode, dispatch(mode, "claude-code", flags)):
                routed.add(option)
        assert tried > 15, f"{mode}: the option sweep found only {tried} options"
        assert routed == set(WRITE_EXECUTOR_WAYS_IN), (
            f"{mode}: the options that reach the executor are {sorted(routed)}; "
            "every one is a way in the trigger and the reference must name")
