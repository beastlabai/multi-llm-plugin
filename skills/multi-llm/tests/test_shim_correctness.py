#!/usr/bin/env python3
"""Shim correctness (D5/D5c) — no host CLI required, runs on every PR.

The per-host live smokes are `-m live` and therefore local-only, so on their own
nothing would catch a shim regression or host-doc drift between releases. These
assertions read the shim *files* and are parametrized over every shipped shim, so a
shim broken by a host-doc edit, a capability-data change or an orchestrator rename
fails CI. Today the only shipped shim is the reference host's `SKILL.md`; the
per-host `.tmpl` files are the install design's to add, and the parametrization
picks them up with no edit here.

The assertions, per shim:

* **Verb resolution is baked, not deferred.** Each shim contains exactly one filled
  resolution per capability verb, **byte-identical** to what the renderer emits for
  that host's entry — including Claude Code's hand-authored `SKILL.md`, which is
  compared against a render of the `claude-code` entry it is never installed from.
  So the reference host cannot drift from the data that governs every other host.
* **Nothing is deferred to the agent.** No unfilled slot, no "if your host supports
  …", and no instruction to consult the capability data or the generated matrix.
* **Capability adherence.** The `--host` id it bakes in is a key in the data, and
  the shim never requests what its entry does not grant.
* **Commands resolve to real modes.** Every mode word and flag a shim emits or shows
  in an example is looked up in the dispatcher's real parser, so a renamed mode fails
  the shim test on the same PR that renames it.
* **`{SKILL_DIR}` resolves absolute**, including under a skill dir containing a
  space, with no placeholder surviving into the emitted command.
* **The uv preflight precedes the first `uv run`.**
* **No environment assignment**, no `unset`, no `env -u`: the only env assignment a
  shim may make is `MULTI_LLM_SKILL_DIR`, because a shim that hand-rolls guard
  neutralization pins one host's variable names into markdown where no test can see
  them.
* **The core is loaded, never inlined**: no shared paragraph is duplicated from the
  instruction core into a shim.
* **The rendered line count** against the effective cap (a tripwire, not the
  correctness property).
"""

import re
import shlex
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes as mode_table  # noqa: E402
from utils import host_capabilities as caps  # noqa: E402

#: The ~100 rendered-line budget a shim is held to unless its entry raises it.
DEFAULT_SHIM_MAX_LINES = 100

#: Fixture skill dir with a space in every segment that can carry one.
FIXTURE_SKILL_DIR = "/tmp/multi llm fixture/skill dir"

#: Every spelling of a skill-dir placeholder that must not survive rendering.
PLACEHOLDERS = ("{SKILL_DIR}", "${SKILL_DIR}", "$SKILL_DIR", "$ARGUMENTS",
                "{{spawn_subagents}}", "{{ask_user}}", "{{run_detached}}")

UV_COMMAND_RE = re.compile(r"^(PYTHONUNBUFFERED=1\s+)?uv\s+run\b")


def _shipped_shims():
    """``(host_id, path)`` for every shim file on disk, from the capability data."""
    found = []
    for host_id in sorted(caps.host_ids()):
        entry = caps.entry(host_id)
        template = entry["shim"]["template"]
        path = SKILL_ROOT / (template or entry["shim"]["installed"])
        if path.exists():
            found.append((host_id, path))
    return found


SHIMS = _shipped_shims()
SHIM_IDS = [host_id for host_id, _ in SHIMS]


def _rendered(path: Path, base: str = FIXTURE_SKILL_DIR) -> str:
    """The shim as an agent reads it: skill dir substituted, snippets inlined.

    ``install-host`` does this for real; here it is done the same way so the
    assertions run against what an agent actually sees, not what the `.tmpl`
    happens to contain.
    """
    from utils.cli_common import render_instruction_text

    text = path.read_text(encoding="utf-8")
    for snippet in sorted((SKILL_ROOT / "hosts" / "snippets").glob("*.md")):
        token = "{{" + snippet.stem.replace("-", "_") + "}}"
        if token in text:
            text = text.replace(token, snippet.read_text(encoding="utf-8"))
    uv_text = (SKILL_ROOT / "hosts" / "uv-missing.txt").read_text(encoding="utf-8")
    text = text.replace("{{uv_missing_text}}", uv_text)
    return render_instruction_text(text, base)


def _effective_cap(host_id: str) -> int:
    return int(caps.entry(host_id).get("shim_max_lines") or DEFAULT_SHIM_MAX_LINES)


def _body(text: str) -> str:
    """The shim minus its host frontmatter — what the line budget is about."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        try:
            end = lines.index("---", 1)
        except ValueError:
            return text
        return "\n".join(lines[end + 1:])
    return text


def _command_units(text: str):
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = re.sub(r"^[-*+]\s+", "", line.strip())
        if UV_COMMAND_RE.match(stripped):
            yield line_no, stripped


def test_at_least_the_reference_shim_is_shipped():
    assert SHIM_IDS, "no shim files found; the parametrization would be vacuous"
    assert "claude-code" in SHIM_IDS


@pytest.mark.parametrize("host_id,path", SHIMS, ids=SHIM_IDS)
class TestShim:
    # -- verb resolution ------------------------------------------------
    def test_the_three_verb_sentences_are_byte_identical_to_the_render(
        self, host_id, path
    ):
        """The one shim nobody renders is asserted instead (D5).

        Claude Code's `SKILL.md` is ``installable: false``, so its sentences are
        hand-written — and they are compared, byte for byte, against what the
        renderer would emit for the ``claude-code`` entry. Without this the
        reference host is the one host free to drift from the data.
        """
        text = path.read_text(encoding="utf-8")
        for verb, sentence in zip(caps.VERBS, caps.verb_sentences(host_id)):
            assert sentence in text, (
                f"{path.name}: the *{caps.VERB_WORDING[verb]}* sentence is not "
                "byte-identical to the renderer's output for this host's entry.\n"
                f"  expected: {sentence}\n"
                "  Re-render it rather than paraphrasing; a shim that drifts from "
                "the data is a shim nobody can reason about."
            )

    def test_each_verb_is_resolved_exactly_once(self, host_id, path):
        text = path.read_text(encoding="utf-8")
        for verb, sentence in zip(caps.VERBS, caps.verb_sentences(host_id)):
            assert text.count(sentence) == 1, (
                f"{path.name}: the *{caps.VERB_WORDING[verb]}* resolution appears "
                f"{text.count(sentence)} times; exactly one per verb"
            )

    def test_no_slot_is_left_unfilled(self, host_id, path):
        rendered = _rendered(path)
        for slot in caps.VERB_SLOTS.values():
            assert slot not in rendered, f"{path.name}: unfilled slot {slot}"

    def test_nothing_is_deferred_to_the_agent(self, host_id, path):
        """No condition to evaluate, and no pointer at the capability matrix."""
        lowered = path.read_text(encoding="utf-8").lower()
        for phrase in ("if your host supports", "if the host supports",
                       "depending on your host", "capabilities.yaml",
                       "host-capabilities.md"):
            assert phrase not in lowered, (
                f"{path.name} says {phrase!r}. The host was chosen when the shim "
                "was installed; there is nothing left to disambiguate at runtime, "
                "and an agent must never consult the matrix mid-run."
            )

    # -- capability adherence -------------------------------------------
    def test_the_baked_host_id_is_a_key_in_the_data(self, host_id, path):
        text = path.read_text(encoding="utf-8")
        baked = re.findall(r"--host\s+([a-z][a-z0-9-]*)", text)
        assert baked, f"{path.name} bakes no --host id into its dispatcher command"
        assert set(baked) == {host_id}, (
            f"{path.name} bakes --host {sorted(set(baked))} but is the shim for "
            f"'{host_id}'"
        )
        caps.entry(host_id)  # exit 3 if unknown

    def test_the_shim_never_requests_what_its_entry_does_not_grant(
        self, host_id, path
    ):
        text = path.read_text(encoding="utf-8")
        entry = caps.entry(host_id)
        if entry["subagents"] == "native":
            assert "Never pass `--exec-subagents`" in text
            assert not re.search(r"^[^|]*\s--exec-subagents\s+[^`]", text, re.M), (
                f"{path.name} passes --exec-subagents although its entry "
                "dispatches subagents natively"
            )
        if entry["detach"] == "native":
            assert "Never pass `--detach`" in text
        if entry["ask"] in ("free-text", "none"):
            ask_tool = (entry.get("native_tool") or {}).get("ask")
            if ask_tool:
                assert ask_tool not in text, (
                    f"{path.name} names the structured ask tool although its entry "
                    f"is ask: {entry['ask']}"
                )

    def test_an_installable_shim_bakes_the_unattended_default(self, host_id, path):
        """D6b: unattended is the default everywhere except the reference host."""
        entry = caps.entry(host_id)
        if not entry["installable"]:
            return
        assert "--yes" in path.read_text(encoding="utf-8"), (
            f"{path.name} is installable but never passes --yes; unattended is the "
            "default on every host that is not the interactive reference host"
        )

    # -- commands resolve ------------------------------------------------
    def test_every_mode_word_the_shim_names_exists(self, host_id, path):
        text = path.read_text(encoding="utf-8")
        named = set(re.findall(r"`--([a-z][a-z0-9-]*)`", text))
        flags = {
            "models", "quick", "interactive", "yes", "non-interactive", "force",
            "dir", "gitignore", "template-only", "detach", "exec-subagents",
            "host", "resume", "reaggregate", "rerun-all", "claude-decide",
            "no-confirm", "consolidate", "skip-prerequisites", "review-tasks",
            "force-strategy", "human-task", "apply-fixes", "init",
            "apply-suggestions", "apply-task-suggestions",
        }
        for word in named - flags:
            assert mode_table.resolve(word) is not None, (
                f"{path.name} names `--{word}`, which is neither a mode word nor a "
                "known flag; a shim can never document a command the dispatcher "
                "does not have"
            )

    def test_every_example_command_parses_into_a_real_mode(self, host_id, path):
        text = path.read_text(encoding="utf-8")
        examples = re.findall(r"^/multi-llm:multi-llm\s+(.*)$", text, re.M)
        assert examples, f"{path.name} shows no worked example"
        for example in examples:
            argv = shlex.split(example.split("#", 1)[0].strip())
            leading = [tok for tok in argv if tok.startswith("--")]
            mode_flags = [
                tok for tok in leading
                if mode_table.resolve(tok.lstrip("-")) is not None
            ]
            # No mode flag means review-plan (the documented default).
            words = [tok.lstrip("-") for tok in mode_flags] or ["review-plan"]
            for word in words:
                assert mode_table.resolve(word) is not None, (example, word)
            positionals = [tok for tok in argv if not tok.startswith("--")]
            mode = mode_table.resolve(words[0])
            assert len(positionals) <= len(mode.positionals), (
                f"{path.name}: example {example!r} passes {len(positionals)} "
                f"positional(s) but mode '{mode.name}' takes "
                f"{len(mode.positionals)}"
            )

    # -- skill-dir resolution -------------------------------------------
    def test_the_skill_dir_resolves_absolute_even_with_a_space(self, host_id, path):
        rendered = _rendered(path)
        for placeholder in PLACEHOLDERS:
            assert placeholder not in rendered, (
                f"{path.name}: {placeholder} survived rendering"
            )
        units = list(_command_units(rendered))
        assert units, f"{path.name} emits no `uv run` command"
        for line_no, command in units:
            argv = shlex.split(command)
            assert argv[0] in ("uv", "PYTHONUNBUFFERED=1"), (
                f"{path.name}:{line_no}: the allow-rule prefix is part of the "
                f"command's contract: {command[:120]}"
            )
            assert "--project" in argv, f"{path.name}:{line_no}: {command[:120]}"
            project = argv[argv.index("--project") + 1]
            assert project == FIXTURE_SKILL_DIR, (
                f"{path.name}:{line_no}: the skill dir word-split on a space "
                f"({project!r})"
            )
            assert Path(project).is_absolute()

    # -- the uv preflight ------------------------------------------------
    def test_the_uv_probe_precedes_the_first_uv_run(self, host_id, path):
        rendered = _rendered(path)
        probe = rendered.find("command -v uv")
        assert probe >= 0, f"{path.name} carries no uv probe"
        first_run = min(
            (rendered.find(marker) for marker in ("uv run --project",)
             if rendered.find(marker) >= 0),
            default=-1,
        )
        assert first_run >= 0
        assert probe < first_run, (
            f"{path.name}: the uv probe must come before the first `uv run`, or a "
            "missing uv surfaces as a bare `command not found`"
        )

    def test_the_install_text_is_single_sourced(self, host_id, path):
        """Either the message is interpolated, or the shim names its file."""
        text = path.read_text(encoding="utf-8")
        rendered = _rendered(path)
        uv_text = (SKILL_ROOT / "hosts" / "uv-missing.txt").read_text(
            encoding="utf-8"
        )
        names_file = "hosts/uv-missing.txt" in text
        interpolated = uv_text.splitlines()[0] in rendered
        assert names_file or interpolated, (
            f"{path.name} neither names hosts/uv-missing.txt nor interpolates it; "
            "the wording lives in exactly one place and may not be paraphrased"
        )
        if not names_file:
            assert uv_text.rstrip("\n") in rendered, (
                f"{path.name} interpolates a *modified* copy of the install text"
            )

    # -- environment discipline ------------------------------------------
    def test_no_environment_assignment_or_unset(self, host_id, path):
        """The only env assignment a shim may make is MULTI_LLM_SKILL_DIR (D1b).

        A shim that hand-rolls guard neutralization (`CLAUDECODE= claude -p …`,
        `env -u …`) pins one host's variable names into markdown, where no test can
        see them and no provenance field governs them.
        """
        offenders = []
        for line_no, line in enumerate(
            _rendered(path).splitlines(), start=1
        ):
            for match in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})=", line):
                name = match.group(1)
                if name in ("MULTI_LLM_SKILL_DIR", "PYTHONUNBUFFERED"):
                    continue
                offenders.append(f"{line_no}: {name}=")
            if re.search(r"\benv\s+-u\b|^\s*unset\s+[A-Z]", line):
                offenders.append(f"{line_no}: {line.strip()[:60]}")
        assert not offenders, (
            f"{path.name}: environment manipulation in a shim. Guard handling is "
            f"Python's (adapter + capability data), never markdown's: {offenders}"
        )

    # -- the core is loaded, never inlined -------------------------------
    def test_no_core_paragraph_is_duplicated_into_the_shim(self, host_id, path):
        """D5c's no-shared-paragraph check — the guard against re-duplicating
        exactly what the host-neutral core just de-duplicated."""
        shim_paragraphs = {
            " ".join(block.split())
            for block in re.split(r"\n\s*\n", _body(path.read_text(encoding="utf-8")))
            if len(block.split()) >= 25
        }
        offenders = []
        for core in sorted(SKILL_ROOT.glob("instructions/*.md")):
            core_text = " ".join(core.read_text(encoding="utf-8").split())
            for paragraph in shim_paragraphs:
                if paragraph in core_text:
                    offenders.append(f"{core.name}: {paragraph[:80]}…")
        assert not offenders, (
            f"{path.name} copies core prose instead of naming the file to read: "
            f"{offenders}"
        )

    def test_the_shim_names_the_read_command_for_the_core(self, host_id, path):
        assert "util instructions" in path.read_text(encoding="utf-8"), (
            f"{path.name} never tells the agent how to load the core; the core is "
            "loaded, never inlined"
        )

    # -- the tripwire ----------------------------------------------------
    def test_the_rendered_shim_fits_its_effective_cap(self, host_id, path):
        rendered = _rendered(path)
        cap = _effective_cap(host_id)
        count = len(rendered.splitlines())
        assert count <= cap, (
            f"{path.name} renders {count} lines against an effective cap of {cap}. "
            "Ask where the excess belongs before raising the cap: workflow prose → "
            "the core, a per-host fact → the capability data, boilerplate a second "
            "host shares → a snippet. Only when none applies does that host's entry "
            "take `shim_max_lines: <n>` with a rationale."
        )

    def test_the_body_fits_the_hundred_line_contract(self, host_id, path):
        """The budget is about instructions, not a host's mandatory manifest."""
        count = len(_body(_rendered(path)).splitlines())
        assert count <= DEFAULT_SHIM_MAX_LINES, (
            f"{path.name}'s body is {count} lines, over the "
            f"{DEFAULT_SHIM_MAX_LINES}-line contract even excluding frontmatter. "
            "`shim_max_lines` covers verbose frontmatter, not verbose instructions."
        )


class TestReferenceShimSpecifics:
    """The Claude Code shim: what the cutover has to have removed."""

    SKILL_MD = SKILL_ROOT / "SKILL.md"

    def test_the_argv_scanning_shell_preamble_is_gone(self):
        """Three `!`-prefixed bash blocks re-implemented argv parsing in shell.

        They were the largest host-coupled thing in the shim: an argv scanner
        duplicating `modes.py` and the dispatcher's own normalization, in a syntax
        only one host runs. The dispatcher rejects a bad plan path or an empty
        question itself, with exit 3 and the offending token named.
        """
        text = self.SKILL_MD.read_text(encoding="utf-8")
        assert "skip_vals" not in text
        assert "PLAN_PATH=\"\"" not in text
        assert not re.search(r"^!\s", text, re.M), (
            "a `!`-prefixed inline bash block survived the cutover"
        )
        assert "ASK=0" not in text

    def test_the_dispatcher_command_bakes_the_host(self):
        text = self.SKILL_MD.read_text(encoding="utf-8")
        assert "--host claude-code" in text

    def test_the_argument_hint_still_renders_from_the_mode_table(self):
        line = next(
            l for l in self.SKILL_MD.read_text(encoding="utf-8").splitlines()
            if l.startswith("argument-hint:")
        )
        assert line.split("argument-hint:", 1)[1].strip() == (
            mode_table.render_argument_hint()
        )

    def test_the_permission_grants_survived_the_slimming(self):
        """Losing an `allowed-tools` entry turns a follow-up into a prompt."""
        text = self.SKILL_MD.read_text(encoding="utf-8")
        for grant in ("Bash(uv:*)", "Bash(PYTHONUNBUFFERED=1 uv:*)",
                      "Bash(command -v uv)", "Task", "AskUserQuestion"):
            assert grant in text, grant

    def test_the_shim_is_the_only_place_naming_the_host_placeholder(self):
        text = self.SKILL_MD.read_text(encoding="utf-8")
        assert "${CLAUDE_SKILL_DIR}" in text
        assert "{SKILL_DIR}" not in text.replace("${CLAUDE_SKILL_DIR}", "")

    def test_the_shim_carries_no_provider_config_reference_or_summary_prose(self):
        """What moved out: the README owns configuration, the core owns workflow."""
        text = self.SKILL_MD.read_text(encoding="utf-8")
        for gone in ("### providers.yaml Format", "## Critical Rules",
                     "## Summary Format", "### Mode Detection Logic",
                     "## Pre-execution Validation"):
            assert gone not in text, f"{gone} did not move out of the shim"


# ===========================================================================
# The full pass of every mode from the reference host (7c)
# ===========================================================================

class TestEveryModeFromTheReferenceHost:
    """Every mode word, invoked the way the slimmed shim tells the host to.

    The cutover's acceptance condition: the exact command shape `SKILL.md` now
    prints — `multi_llm.py <mode> "<plan>" --host claude-code` — reaches every
    mode's own parser. This is what catches the failure the slimming could
    introduce: a shim that documents a command the dispatcher no longer accepts.

    Asserted per mode: the dispatcher does **not** reject the invocation itself
    (no unknown-mode, unknown-host, bad-positional or capability refusal), the
    `--host` token is forwarded verbatim rather than consumed, and exactly one
    `MULTI_LLM_RESULT` line is emitted.
    """

    MODES = [m.name for m in mode_table.MODES if m.user_facing]

    @pytest.mark.parametrize("mode", MODES)
    def test_the_shim_command_reaches_the_mode(self, mode, monkeypatch, capsys,
                                               tmp_path):
        import multi_llm
        from utils.cli_common import parse_result_line

        seen = {}

        def loader(module_path, attr):
            def target(argv):
                seen["argv"] = list(argv)
                seen["module"] = module_path
                return 0

            return target

        monkeypatch.setattr(multi_llm, "_load_target", loader)
        plan = tmp_path / "a plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        argv = [mode]
        if mode_table.resolve(mode).positionals:
            argv.append(str(plan))
            if len(mode_table.resolve(mode).positionals) > 1:
                argv.append("Is the rollback strategy sufficient?")
        argv += ["--host", "claude-code"]

        exit_code = multi_llm.main(argv)
        out = capsys.readouterr().out
        assert exit_code == 0, f"{mode}: the shim's command shape was rejected"
        assert seen["module"] == mode_table.resolve(mode).target_module
        assert "--host" in seen["argv"] and "claude-code" in seen["argv"], (
            f"{mode}: --host must be forwarded verbatim, not consumed"
        )
        payload = parse_result_line(out)
        assert payload["mode"] == mode
        assert out.count("MULTI_LLM_RESULT:") == 1

    @pytest.mark.parametrize("mode", MODES)
    def test_every_mode_serves_its_core_or_declares_it_has_none(self, mode,
                                                               capsys):
        """Step 4 of the shim, for real: `util instructions <mode>`."""
        import multi_llm
        from utils import cli_common

        expected = mode_table.instruction_path(mode)
        exit_code = multi_llm.main(["util", "instructions", mode])
        out = capsys.readouterr().out
        if expected is None:
            assert exit_code == 3, f"{mode} has no core file but did not say so"
            return
        assert exit_code == 0, out
        body = out.rsplit(cli_common.RESULT_MARKER, 1)[0]
        assert body.strip(), f"{mode}: the served core is empty"
        for token in ("{SKILL_DIR}", "${CLAUDE_SKILL_DIR}", "$SKILL_DIR"):
            assert token not in body, f"{mode}: {token} survived the render"
