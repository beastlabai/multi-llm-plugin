"""The bracket-marker registry (``docs/development/markers.md``) and its guards.

The orchestrators print ``[BATCH_DONE]``-style lines; hosts, the opencode and
codex servers and the instructions parse them by string. Renaming one used to
fail nothing. ``tests/tools/render_markers.py`` renders the join between the
emitting sources and the citing instructions into the registry page, and this
module holds the page and the join to account:

1. the committed page is a fresh render;
2. every marker ``SKILL.md``, ``instructions/`` or ``references/`` cites is
   emitted by a source (a renamed marker fails here);
3. every emitted marker is cited by one of those documents or is on the
   ``NOT_FOR_HOST`` list with a reason, so a *new* marker forces a decision;
4. no printed marker is built from upper-case text and a run-time value, so
   the scan can list every one (the construction rule).

The scanner's own rules are pinned on scratch trees laid out like the
repository (``skills/multi-llm/`` beside ``packaging/``). Pure file reads, no
spend.
"""

from __future__ import annotations

import ast
import inspect
import shutil
import textwrap
from pathlib import Path

from .tools import render_markers as rm


def _skill(tmp_path: Path, source: str = "", instruction: str = "") -> Path:
    """A scratch repository whose skill holds ``utils/emit.py`` and one instruction.

    Returns the skill root; ``packaging/`` is its grandparent's child, as in the
    real tree.
    """
    skill = tmp_path / "skills" / "multi-llm"
    (skill / "utils").mkdir(parents=True)
    (skill / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (skill / "utils" / "emit.py").write_text(textwrap.dedent(source), encoding="utf-8")
    (skill / "instructions").mkdir()
    (skill / "instructions" / "mode.md").write_text(instruction, encoding="utf-8")
    (skill / "SKILL.md").write_text("", encoding="utf-8")
    return skill


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def _emitted(skill: Path) -> set:
    return set(rm.scan_emissions(skill))


class TestTheRegistryPageIsFresh:
    def test_the_committed_page_matches_a_fresh_render(self):
        assert rm.check() is None, rm.check()

    def test_a_missing_page_is_stale_not_absent(self, tmp_path):
        problem = rm.check(tmp_path / "markers.md")
        assert problem is not None and "render_markers.py" in problem

    def test_a_hand_edit_is_stale(self, tmp_path):
        page = tmp_path / "markers.md"
        page.write_text(rm.render() + "\nhand-written\n", encoding="utf-8")
        assert "stale" in (rm.check(page) or "")

    def test_the_cli_check_exits_one_on_a_stale_page(self, monkeypatch, tmp_path, capsys):
        page = tmp_path / "markers.md"
        page.write_text("old\n", encoding="utf-8")
        monkeypatch.setattr(rm, "OUTPUT", page)
        assert rm.main(["--check"]) == 1
        assert "stale" in capsys.readouterr().err
        assert rm.main([]) == 0
        assert rm.main(["--check"]) == 0


class TestEveryCitedMarkerIsEmitted:
    def test_a_marker_the_documents_cite_is_emitted_by_a_source(self):
        emitted, cited = rm.scan_emissions(), rm.scan_citations()
        # Non-vacuity: an empty scan on either side would pass the assertion below.
        assert len(cited) >= 80 and len(emitted) >= 150, (len(cited), len(emitted))
        missing = {m: files for m, files in cited.items() if m not in emitted}
        assert not missing, (
            "markers cited by SKILL.md, instructions/ or references/ that no source "
            "prints (a marker was renamed or removed without its citations; hosts and "
            f"the opencode/codex servers parse these strings): {missing}"
        )


class TestEveryEmittedMarkerIsHandledOrDeclaredNotForTheHost:
    def test_a_new_marker_is_cited_or_listed_with_a_reason(self):
        emitted, cited = rm.scan_emissions(), rm.scan_citations()
        new = sorted(m for m in emitted if m not in cited and m not in rm.NOT_FOR_HOST)
        assert not new, (
            f"emitted markers no instruction handles and NOT_FOR_HOST does not list: {new}. "
            "Cite each in the instruction or reference that handles it, or add it to "
            "NOT_FOR_HOST in tests/tools/render_markers.py with a reason key, then re-render "
            "docs/development/markers.md."
        )

    def test_every_not_for_host_entry_is_still_emitted_and_still_uncited(self):
        emitted, cited = rm.scan_emissions(), rm.scan_citations()
        stale = sorted(m for m in rm.NOT_FOR_HOST if m not in emitted)
        handled = sorted(m for m in rm.NOT_FOR_HOST if m in cited)
        assert not stale, f"NOT_FOR_HOST entries no source emits any more: {stale}"
        assert not handled, (
            f"NOT_FOR_HOST entries an instruction now cites; drop them from the list: {handled}"
        )

    def test_every_reason_key_is_defined_and_used(self):
        unknown = sorted({k for k in rm.NOT_FOR_HOST.values() if k not in rm.REASONS})
        unused = sorted(set(rm.REASONS) - set(rm.NOT_FOR_HOST.values()))
        assert not unknown, f"reason keys NOT_FOR_HOST uses but REASONS lacks: {unknown}"
        assert not unused, f"REASONS entries no marker uses: {unused}"
        assert all(text.strip() for text in rm.REASONS.values())


class TestNoPrintedMarkerIsBuiltFromParts:
    def test_the_sources_follow_the_construction_rule(self):
        built = rm.scan_constructions()
        assert not built, (
            "printed brackets built from upper-case text and a run-time value; no scan "
            "can list the markers they print. Write each marker whole, as a literal or a "
            f"module constant: {built}"
        )


class TestTheRealTreeEmissionsCodexFoundMissing:
    """Markers the five-character, literal-only, skill-only scan could not see."""

    def test_a_marker_named_through_a_helper_argument(self):
        rows = rm.scan_emissions()["[DEADLINE_SWEEP]"]
        assert rm.Emission("ask_orchestrator.py", "sweep_live_clis()", "") in rows

    def test_the_short_protocol_markers(self):
        emitted = rm.scan_emissions()
        assert {"[WARN]", "[ETA]", "[SKIP]"} <= set(emitted)
        assert "ask_orchestrator.py" in {row.path for row in emitted["[WARN]"]}

    def test_the_next_command_marker_comes_from_its_emitter_not_its_parser(self):
        rows = rm.scan_emissions()["[NEXT_COMMAND]"]
        assert {row.path for row in rows} == {"utils/next_phase.py"}, rows


class TestTheNextStepRuleNamesEveryLineTheStepDoesNotCarry:
    """A host copies ``user_next_step``'s section and nothing else of the block, so
    each block line the step does not carry needs its own rule where the next-step
    relay rule lives: SKILL.md section 6 (Claude Code) and the report format every
    core points at."""

    def test_each_uncarried_block_marker_is_named_by_the_relay_rule(self):
        from utils import next_phase

        tree = ast.parse(textwrap.dedent(inspect.getsource(next_phase.render_next_phase_lines)))
        printed = {getattr(next_phase, node.id) for node in ast.walk(tree)
                   if isinstance(node, ast.Name) and node.id.endswith("_MARKER")}
        uncarried = sorted(printed - set(next_phase.CARRIED_BLOCK_MARKERS))
        assert "[RUNNER_WARNING]" in uncarried and "[TESTS_NOT_RUN]" in uncarried, uncarried
        for document in ("SKILL.md", "references/summary-format.md"):
            text = (rm.SKILL_ROOT / document).read_text(encoding="utf-8")
            missing = [marker for marker in uncarried if marker not in text]
            assert not missing, (
                f"{document} does not say what to do with {missing}: the block prints them, "
                "user_next_step does not carry them, so a host copying the step drops them"
            )


class TestTheScannerRules:
    def test_a_docstring_comment_or_bare_string_is_not_an_emission(self, tmp_path):
        skill = _skill(tmp_path, '''
            """Mentions [MODULE_DOC] only."""

            TAG = "[TAG_ONE]"
            """Says [ATTR_DOC] about TAG."""

            def f():
                """Says [FUNC_DOC] only."""
                # [COMMENT_ONLY] is a comment
                print(TAG)
                return "[REAL_ONE] printed"
            ''')
        assert _emitted(skill) == {"[REAL_ONE]", "[TAG_ONE]"}

    def test_a_constant_carries_its_name_and_its_comment(self, tmp_path):
        skill = _skill(tmp_path, '''
            #: The one line a record prints. Second sentence is dropped.
            MARKER = "[HAS_COMMENT]"
            OTHER = "[NO_COMMENT]"

            def emit(path):
                print(f"{MARKER} {path}")
                print(OTHER)
            ''')
        found = rm.scan_emissions(skill)
        assert found["[HAS_COMMENT]"] == [
            rm.Emission("utils/emit.py", "MARKER", "The one line a record prints")
        ]
        assert found["[NO_COMMENT]"] == [rm.Emission("utils/emit.py", "OTHER", "")]

    def test_a_docstring_after_the_constant_is_its_meaning(self, tmp_path):
        skill = _skill(tmp_path, '''
            TAG = "[DOC_AFTER]"
            """What the tag means. More."""
            print(TAG)
            ''')
        assert rm.scan_emissions(skill)["[DOC_AFTER]"][0].meaning == "What the tag means"

    def test_a_constant_nothing_prints_is_not_an_emission(self, tmp_path):
        skill = _skill(tmp_path, 'UNUSED = "[NEVER_PRINTED]"\n')
        assert _emitted(skill) == set()

    def test_short_markers_count_and_odd_brackets_do_not(self, tmp_path):
        skill = _skill(tmp_path, '''
            print("[WARN] all fallback")
            print("[ETA] 3 min")
            print("[A-Z] [A] [lower_case] [Mixed_Case] [0X]")
            ''')
        assert _emitted(skill) == {"[WARN]", "[ETA]"}

    def test_a_cited_short_marker_with_no_emitter_is_reported(self, tmp_path):
        skill = _skill(tmp_path, 'print("[ETA] 3 min")\n', "Relay the [WARN] line.")
        assert set(rm.scan_citations(skill)) - _emitted(skill) == {"[WARN]"}

    def test_an_f_string_or_concatenation_of_constants_is_resolved(self, tmp_path):
        skill = _skill(tmp_path, '''
            NAME = "BUILT_UP"
            PREFIX = "[GLUED"

            def f(kind):
                label = "LOCAL_NAME"
                print(f"[{NAME}] x")
                print(PREFIX + "_UP] y")
                print("[" + label + "] z")
                for tag in ("LOOP_ONE", "LOOP_TWO"):
                    print("[%s] w" % tag)
                print("[{}] v".format("FORMATTED"))
            ''')
        assert _emitted(skill) == {
            "[BUILT_UP]", "[GLUED_UP]", "[LOCAL_NAME]", "[LOOP_ONE]", "[LOOP_TWO]", "[FORMATTED]",
        }

    def test_a_helper_argument_is_an_emission_at_the_call(self, tmp_path):
        skill = _skill(tmp_path, '''
            from . import helpers
            from .helpers import note

            def f():
                helpers.sweep("HELPER_ARG")
                note(text="[KEYWORD_ARG] done")
            ''')
        _write(skill, "utils/helpers.py", '''
            import sys

            def sweep(tag):
                print(f"[{tag}] cli pid 1", file=sys.stderr)

            def note(text):
                print(text)
            ''')
        found = rm.scan_emissions(skill)
        assert found["[HELPER_ARG]"] == [rm.Emission("utils/emit.py", "sweep()", "")]
        assert found["[KEYWORD_ARG]"] == [rm.Emission("utils/emit.py", "note()", "")]

    def test_a_label_built_from_a_run_time_value_is_not_a_marker(self, tmp_path):
        skill = _skill(tmp_path, '''
            def f(model, spec):
                print(f"[{model}] Completed in 3s")
                print("[" + spec + "] failed")
            ''')
        assert _emitted(skill) == set() and rm.scan_constructions(skill) == []

    def test_the_construction_rule_refuses_a_marker_built_from_parts(self, tmp_path):
        skill = _skill(tmp_path, '''
            def f(kind, n):
                print(f"[BATCH_{kind}] done")
                print("[" + kind + "_DONE] done")
            ''')
        built = rm.scan_constructions(skill)
        assert [(c.path, c.text) for c in built] == [
            ("utils/emit.py", "[BATCH_{…}]"), ("utils/emit.py", "[{…}_DONE]"),
        ]

    def test_a_parser_is_not_an_emission(self, tmp_path):
        skill = _skill(tmp_path, '''
            import re

            PREFIX = "[PARSED_PREFIX] "
            KEYS = ("[TUPLE_ONE]", "[TUPLE_TWO]")
            PATTERN = re.compile(re.escape("[REGEX_ONE]") + r"(.*)")

            def is_tagged(line, tag):
                return line.startswith(tag)

            def read(line, table):
                if line.startswith(PREFIX):
                    return line[len(PREFIX):]
                if "[COMPARED]" in line or line == "[EQUALS]":
                    return None
                if any(line.startswith(key) for key in KEYS):
                    return table.get("[LOOKED_UP]")
                return is_tagged(line, "[VIA_PARSER]")
            ''')
        assert _emitted(skill) == set()

    def test_a_marker_mentioned_mid_line_is_not_an_emission(self, tmp_path):
        skill = _skill(tmp_path, '''
            def reason():
                return ("copy it from its [MENTIONED] line, "
                        "and from the\\n[NEXT_LINE] that follows")
            ''')
        assert _emitted(skill) == {"[NEXT_LINE]"}

    def test_removing_the_real_emitter_leaves_the_real_parser_unable_to_pass(self, tmp_path):
        """Codex's mutation: ``report_check.py`` only parses ``[NEXT_COMMAND]``."""
        real = rm.SKILL_ROOT / "utils"
        skill = _skill(tmp_path, instruction="Copy the [NEXT_COMMAND] line.")
        shutil.copy(real / "report_check.py", skill / "utils" / "report_check.py")
        assert "[NEXT_COMMAND]" not in _emitted(skill)

        emitter = (real / "next_phase.py").read_text(encoding="utf-8")
        _write(skill, "utils/next_phase.py", emitter)
        assert "[NEXT_COMMAND]" in _emitted(skill)

        blanked = emitter.replace('NEXT_COMMAND_MARKER = "[NEXT_COMMAND]"', 'NEXT_COMMAND_MARKER = ""')
        assert blanked != emitter
        _write(skill, "utils/next_phase.py", blanked)
        assert "[NEXT_COMMAND]" in set(rm.scan_citations(skill)) - _emitted(skill)

    def test_tests_and_hidden_directories_are_not_scanned(self, tmp_path):
        skill = _skill(tmp_path, 'print("[REAL_ONE]")\n')
        for sub in ("tests", ".venv"):
            _write(skill, f"{sub}/noise.py", 'print("[NOISE_ONE]")\n')
        assert _emitted(skill) == {"[REAL_ONE]"}

    def test_packaging_sources_are_scanned_and_their_generated_copies_are_not(self, tmp_path):
        skill = _skill(tmp_path, 'print("[REAL_ONE]")\n')
        repo = skill.parents[1]
        _write(repo, "packaging/codex/mcp_server.py", 'print("[SERVER_ONE] ready")\n')
        _write(repo, "packaging/codex/multi-llm/utils/emit.py", 'print("[MIRROR_ONE]")\n')
        _write(repo, "packaging/opencode/src/index.mjs", '''
            // [JS_COMMENT] a comment
            /* [JS_BLOCK] a block */
            const re = /["'\\[]/g
            function f(line, x) {
              if (line.startsWith("[JS_PARSED] ")) return null
              return `[JS_ONE] uv is required, ${x}`
            }
            ''')
        found = rm.scan_emissions(skill)
        assert set(found) == {"[REAL_ONE]", "[SERVER_ONE]", "[JS_ONE]"}
        assert found["[SERVER_ONE]"] == [rm.Emission("packaging/codex/mcp_server.py", "", "")]
        assert found["[JS_ONE]"] == [rm.Emission("packaging/opencode/src/index.mjs", "", "")]

    def test_citations_come_from_skill_md_instructions_and_references(self, tmp_path):
        skill = _skill(tmp_path, 'print("[REAL_ONE]")\n', "Look for [REAL_ONE] and [PREFIX_*].")
        _write(skill, "references/ref.md", "[REAL_ONE]")
        assert rm.scan_citations(skill) == {
            "[REAL_ONE]": ["instructions/mode.md", "references/ref.md"]
        }

    def test_a_cited_but_unemitted_marker_shows_in_the_diff_of_the_two_scans(self, tmp_path):
        skill = _skill(tmp_path, 'print("[REAL_ONE]")\n', "Look for [RENAMED_AWAY].")
        missing = set(rm.scan_citations(skill)) - _emitted(skill)
        assert missing == {"[RENAMED_AWAY]"}

    def test_the_rendered_row_names_file_constant_consumer_and_meaning(self, tmp_path):
        skill = _skill(tmp_path, '''
            #: Prints once. Then more.
            MARKER = "[ROW_ONE]"
            print(MARKER)
            ''', "Act on [ROW_ONE].")
        page = rm.render(skill)
        assert "| `[ROW_ONE]` | `utils/emit.py` `MARKER` | `instructions/mode.md` | Prints once |" in page
