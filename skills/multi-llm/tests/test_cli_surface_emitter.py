"""The shared CLI-surface emitter, ``packaging/common/surface.py`` (D3e).

Plain CI rows: no build, no node, no npm. The emitter is imported and rendered
in-process, the same way ``test_opencode_package.py`` grades the OpenCode
driver, so a red row can never mean "someone forgot to rebuild".

What is graded here and nowhere else:

* **The surface is a function of the source tree alone.** The uv flag list is
  ``unattended.DEFAULT_UV_RUN_ARGS`` — the constant — and never the
  environment-reading ``uv_run_args()``, so ``MULTI_LLM_UV_RUN_ARGS`` unset,
  set, or whitespace-only produces byte-identical output.
* **Schema v4's keys are present and non-empty, for every host**, with
  ``argument_descriptions`` carrying exactly the four tool properties.
* **Host-keyed clauses are keyed on the host id**: the ``@`` sigil clause is
  OpenCode's alone, the write-capable note codex's alone.
* **``writes_worktree`` is derived**, from the executor's write-kind table and
  the mode table's ``sequence`` column, never from a hand list.
* **Schema v6's commit range is generated, never typed**: ``revs`` joins the
  positional kinds and the argument descriptions, the ``revs`` block carries
  the count word, brief suffix, plan refusal, pinned shape and result-line keys
  from their owners, and the range-shaped worked example teaches a range the
  grammar accepts and a context flag the front door defines.
* **Schema v5's route signal is generated, never guessed**:
  ``dispatcher.route_env`` is ``modes.ROUTE_ENV`` plus the one
  ``modes.TOOL_ROUTES`` member the host's ``next_command`` cell carries, and the
  generator raises rather than pick when there is none or more than one. Each
  mode row carries its ``Mode.optional_why``, and the tool description says,
  right under the outcomes line, that ``user_next_step`` is never a call.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent

sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packaging"))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from common import surface  # noqa: E402
from utils import host_capabilities, subagent_tasks, unattended  # noqa: E402

#: Every host a package route ships this surface for.
HOSTS = ("opencode", "codex")

#: The four tool properties of D3 rule 1 — the same on both hosts.
PROPERTIES = {"mode", "plan", "question", "extra_flags"}

#: Schema v6's optional fifth property, the commit modes' range.
REVS = "revs"


@pytest.mark.parametrize("host", HOSTS)
def test_surface_ignores_the_uv_run_args_override(host, monkeypatch):
    """Unset, ``--no-sync`` and whitespace give byte-identical surfaces.

    A surface that read the builder's environment would ship one developer's
    ``--no-sync`` as the default of every installation, against a tree whose
    ``.venv`` was never built (D3 rule 3, D3e).
    """
    monkeypatch.delenv(unattended.UV_RUN_ARGS_ENV, raising=False)
    unset = surface.cli_surface_json(host)
    for value in ("--no-sync", "   ", "\t\n", "--frozen"):
        monkeypatch.setenv(unattended.UV_RUN_ARGS_ENV, value)
        assert surface.cli_surface_json(host) == unset, value

    rendered = json.loads(unset)
    assert rendered["uv"]["default_run_args"] == list(
        unattended.DEFAULT_UV_RUN_ARGS
    )
    # The override's *name* travels so the server can read it at invocation.
    assert rendered["uv"]["run_args_env"] == unattended.UV_RUN_ARGS_ENV


@pytest.mark.parametrize("host", HOSTS)
def test_uninstall_command_is_the_owners_line_with_the_default_flags(
    host, monkeypatch
):
    """``uninstall_command`` is ``host_install.uninstall_line``'s spelling.

    Composed from the constant rather than by calling the owner (which reads
    the environment), so it is held equal to the owner here, with the
    override unset — the one posture where the two must agree.
    """
    from pathlib import PurePosixPath

    from utils import cli_common, host_install

    monkeypatch.delenv(unattended.UV_RUN_ARGS_ENV, raising=False)
    shipped = surface.cli_surface(host)["uninstall_command"]
    # The surface joins its `{SKILL_DIR}` token with `/` on every OS (it ships
    # as data); the owner joins a real skill dir natively. Same code, same joins.
    monkeypatch.setattr(cli_common, "Path", PurePosixPath)
    monkeypatch.setattr(host_install, "Path", PurePosixPath)
    assert shipped == host_install.uninstall_line(host, surface.SKILL_DIR_PLACEHOLDER)


def test_the_emitter_reads_no_environment():
    """No ``os.environ``/``getenv`` read and no ``uv_run_args()`` call."""
    source = (REPO_ROOT / "packaging" / "common" / "surface.py").read_text(
        encoding="utf-8"
    )
    # Parsed rather than grepped: the module docstring names both to explain
    # the rule, and a docstring is not a read.
    offenders = [
        f"line {node.lineno}: {name}"
        for node in ast.walk(ast.parse(source))
        for name in [
            node.attr if isinstance(node, ast.Attribute)
            else node.id if isinstance(node, ast.Name) else None
        ]
        if name in {"environ", "getenv", "uv_run_args"}
    ]
    assert not offenders, offenders


def test_both_hosts_ship_one_schema_version():
    """One emitter, one version: the OpenCode driver re-exports it."""
    sys.path.insert(0, str(REPO_ROOT / "packaging" / "opencode"))
    import build as opencode_build  # noqa: E402

    assert surface.SURFACE_SCHEMA_VERSION == 6
    assert opencode_build.SURFACE_SCHEMA_VERSION == surface.SURFACE_SCHEMA_VERSION
    for host in HOSTS:
        assert surface.cli_surface(host)["schema_version"] == 6
    # The plugin's lockstep constant, compared rather than trusted.
    mjs = (REPO_ROOT / "packaging" / "opencode" / "src" / "index.mjs").read_text(
        encoding="utf-8"
    )
    assert (
        f"const SUPPORTED_SURFACE_VERSION = {surface.SURFACE_SCHEMA_VERSION}\n"
        in mjs
    )


def test_the_opencode_driver_is_a_thin_wrapper():
    """``build.cli_surface_json()`` is the shared emitter's output for opencode."""
    sys.path.insert(0, str(REPO_ROOT / "packaging" / "opencode"))
    import build as opencode_build  # noqa: E402

    assert opencode_build.cli_surface_json() == surface.cli_surface_json(
        "opencode"
    )
    assert opencode_build.cli_surface()["host"] == "opencode"


@pytest.mark.parametrize("host", HOSTS)
def test_v4_keys_are_present_and_non_empty(host):
    rendered = surface.cli_surface(host)
    assert rendered["host"] == host
    assert rendered["tool_name"] == mode_table.TOOL_NAME == "multi_llm"

    description = rendered["tool_description"]
    assert isinstance(description, str) and description.strip()
    assert description.startswith(mode_table.TOOL_DESCRIPTION_LEDE)
    assert mode_table.TOOL_OUTCOMES_LINE in description
    for row in rendered["modes"]:
        assert row["summary"] in description

    described = rendered["argument_descriptions"]
    assert set(described) == PROPERTIES | {REVS}
    for name, text in described.items():
        assert isinstance(text, str) and text.strip(), name
    for flag in rendered["reserved_flags"]["exact"]:
        assert flag in described["extra_flags"]
    for row in rendered["modes"]:
        assert row["name"] in described["mode"]

    refusal = rendered["write_posture_refusal"]
    assert set(refusal) == {
        "template", "posture_unreadable", "lever_env", "lever_value",
    }
    assert refusal["lever_env"] == "MULTI_LLM_CODEX_UNSANDBOXED_WRITES"
    assert refusal["lever_value"] == "1"
    for token in ("{mode}", "{posture}"):
        assert token in refusal["template"]
    assert f"{refusal['lever_env']}={refusal['lever_value']}" in (
        refusal["template"]
    )
    assert refusal["posture_unreadable"].strip()

    assert "{probed}" in rendered["anchor_unresolved"]
    for token in ("{shell}", "{platform}", "{path}"):
        assert token in rendered["shell_missing"]

    assert rendered["runbook"] == {
        "core_end_marker": multi_llm.CORE_END_MARKER,
        "banner_opening": multi_llm.TRUNCATION_BANNER_OPENING,
    }

    # The keys D3 rule 3 composes an argv out of.
    assert rendered["baked_flags"] == ["--host", host, "--yes", "--exec-subagents"]
    assert rendered["detach_flag"] == "--detach"
    assert rendered["dispatcher"]["host_option"] == "--host"
    assert rendered["dispatcher"]["run_id_option"] == "--run-id"
    assert rendered["dispatcher"]["mechanical_continuations"]


def test_the_banner_opening_is_what_util_instructions_prints():
    """The constant the server strips by is the banner's real opening."""
    notice = multi_llm._truncation_notice("review-plan", "x")
    assert notice.startswith(multi_llm.TRUNCATION_BANNER_OPENING + " ")


def test_the_sigil_clause_is_opencodes_alone():
    opencode = surface.cli_surface("opencode")["argument_descriptions"]["plan"]
    codex = surface.cli_surface("codex")["argument_descriptions"]["plan"]
    assert '"@"' in opencode
    assert "@" not in codex
    # The rest of the sentence is shared: dropping the clause changes the text
    # and never the schema.
    assert "Strip at most one layer of matching quotes from what" in codex


def test_the_write_capable_note_is_codexs_alone():
    note_head = mode_table.TOOL_WRITE_CAPABLE_NOTE.split("{modes}")[0]
    assert note_head in surface.cli_surface("codex")["tool_description"]
    assert note_head not in surface.cli_surface("opencode")["tool_description"]
    codex = surface.cli_surface("codex")
    writers = ", ".join(r["name"] for r in codex["modes"] if r["writes_worktree"])
    assert writers in codex["tool_description"]


def test_writes_worktree_is_derived_from_the_write_kinds():
    """The apply family, ``implement``, and the composites that run them — derived.

    A composite writes when **any** phase of its ``sequence`` does: the field
    asks whether the dispatch can reach ``agent-write``, and one writer is enough.
    """
    write_phases = set(subagent_tasks.PHASE_BY_WRITE_KIND.values())
    rows = {r["name"]: r for r in surface.cli_surface("codex")["modes"]}
    for mode in mode_table.user_facing_modes():
        expected = mode.phase in write_phases or (
            mode.is_composite
            and any(phase in write_phases for phase in mode.sequence_phases)
        )
        assert rows[mode.name]["writes_worktree"] is expected, mode.name
    # Spot-check the reading the plan names, so a derivation that went wrong in
    # both places at once still fails.
    assert {n for n, r in rows.items() if r["writes_worktree"]} == {
        "apply-suggestions", "improve-plan", "apply-task-suggestions",
        "apply-code-fixes", "improve-code", "implement", "full",
        # The commit modes (plans/review-commits.md §2.1): the apply front
        # door's phase is apply-code-fixes, and improve-commits runs it.
        "apply-commit-fixes", "improve-commits",
    }


@pytest.mark.parametrize("sequence, expected", [
    # A writer anywhere is enough: not AND, and not the last phase alone.
    (("apply-suggestions", "review-code"), True),
    (("review-code", "apply-code-fixes"), True),
    (("review-plan", "review-code"), False),
])
def test_a_composite_writes_when_any_phase_of_its_sequence_does(
    sequence, expected
):
    row = mode_table.Mode(
        name="composite", positionals=(mode_table.POS_PLAN,), summary="x",
        phase="composite", sequence=sequence,
    )
    assert surface._writes_worktree(row) is expected


@pytest.mark.parametrize("host", HOSTS)
def test_worked_examples_are_selected_by_shape(host):
    description = surface.cli_surface(host)["tool_description"]
    block = description.split("\nExamples:\n", 1)[1]
    examples = [json.loads(line) for line in block.splitlines()]
    ranging = {row.name for row in mode_table.user_facing_modes()
               if mode_table.POS_REVS in row.positionals}
    for example in examples:
        # `revs` is written only where the mode takes a range.
        expected = PROPERTIES | ({REVS} if example["mode"] in ranging else set())
        assert set(example) == expected, example
    by_mode = {e["mode"]: e for e in examples}
    assert by_mode[mode_table.DEFAULT_MODE]["plan"] == mode_table.EXAMPLE_PLAN_PATH
    assert by_mode["init"] == {
        "mode": "init", "plan": "", "question": "", "extra_flags": []
    }
    assert by_mode["ask"]["question"] == mode_table.EXAMPLE_QUESTION
    assert set(mode_table.ALIASES) & set(by_mode), "no alias example"
    # Schema v6: the range-shaped one (plans/review-commits.md §2.9), the
    # first mode declaring a range, with `plan` empty and the context plan
    # in extra_flags.
    first = next(row.name for row in mode_table.user_facing_modes()
                 if mode_table.POS_REVS in row.positionals)
    assert by_mode[first] == {
        "mode": first, "revs": "last 2 commits", "plan": "", "question": "",
        "extra_flags": ["--context-plan", "docs/x.md"],
    }
    assert list(by_mode[first])[:3] == ["mode", "revs", "plan"]


# ---------------------------------------------------------------------------
# Schema v6: the commit modes' range
# ---------------------------------------------------------------------------

def _range_modes():
    return [row.name for row in mode_table.user_facing_modes()
            if mode_table.POS_REVS in row.positionals]


@pytest.mark.parametrize("host", HOSTS)
def test_revs_is_a_positional_kind_and_a_described_argument(host):
    rendered = surface.cli_surface(host)
    assert rendered["positional_kinds"] == {
        "plan": mode_table.POS_PLAN, "question": mode_table.POS_QUESTION,
        "revs": mode_table.POS_REVS,
    }
    revs = rendered["argument_descriptions"][REVS]
    plan = rendered["argument_descriptions"]["plan"]
    for name in _range_modes():
        assert name in revs and name in plan
        assert name not in plan.split("Modes taking a plan: ", 1)[1].split(".", 1)[0]
    # `plan` keeps one meaning, and both say where the context plan goes.
    for text in (revs, plan):
        assert '["--context-plan", "<path>"]' in text
    assert '"revs"' in plan


def test_the_revs_block_is_read_from_its_owners(tmp_path):
    import re

    from utils import commit_range, commit_review

    block = surface.cli_surface("codex")["revs"]
    assert block == surface.cli_surface("opencode")["revs"]
    assert block["count_word"] == multi_llm._RANGE_COUNT_WORD.pattern
    assert re.fullmatch(block["count_word"], "-2")
    assert not re.fullmatch(block["count_word"], "-2h")
    pinned = block["pinned"]
    base, head = "a" * 40, "b" * 40
    assert (base[:pinned["sha_chars"]] + pinned["separator"]
            + head[:pinned["sha_chars"]]) == commit_range.pinned_spelling(base, head)
    assert (block["range_key"], block["fixes_key"], block["lines_key"]) == (
        "commit_range", "commit_fixes", "lines")
    # A brief path ends in the suffix the dispatcher asks about: its owner's
    # one spelling, and the word `_names_a_commit_brief` really decides on.
    suffix = block["brief_suffix"]
    assert suffix and suffix == multi_llm._COMMIT_BRIEF_SUFFIX
    brief = tmp_path / f"b{suffix}"
    other = tmp_path / "b.txt"
    for path in (brief, other):
        path.write_text(commit_review.BRIEF_MARKER + "\n", encoding="utf-8")
    assert multi_llm._names_a_commit_brief(str(brief))
    assert not multi_llm._names_a_commit_brief(str(other))
    refusal = block["plan_refusal"]
    assert "{mode}" in refusal and '"revs"' in refusal
    assert '["--context-plan", "<path>"]' in refusal


def test_the_result_line_keys_are_the_ones_the_dispatcher_writes():
    """``commit_range`` and ``commit_fixes`` (and the block lines inside them)
    are the keys ``cli_common.ResultContext`` emits."""
    from utils import cli_common

    block = surface.cli_surface("codex")["revs"]
    context = cli_common.ResultContext("review-commits")
    context.set_commit_range({"brief": "b.md", block["lines_key"]: ["x"]})
    context.set_commit_fixes({block["lines_key"]: ["y"]})
    payload = context.payload(cli_common.EXIT_OK)
    assert payload[block["range_key"]][block["lines_key"]] == ["x"]
    assert payload[block["fixes_key"]][block["lines_key"]] == ["y"]


def test_the_described_ranges_and_context_flag_are_checked(monkeypatch):
    """A range the grammar stopped accepting, or a context flag the front door
    stopped defining, fails the build instead of teaching a refused call."""
    monkeypatch.setattr(surface, "_DESCRIBED_RANGES", ("not a range at all",))
    with pytest.raises(RuntimeError, match="no longer accepts"):
        surface.cli_surface("codex")
    monkeypatch.undo()
    monkeypatch.setattr(surface, "_CONTEXT_PLAN_FLAG", "--plan-context")
    with pytest.raises(RuntimeError, match="no longer defines"):
        surface.cli_surface("codex")


# ---------------------------------------------------------------------------
# Schema v5: the route signal, optional_why, the user_next_step line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host", HOSTS)
def test_route_env_is_the_owners_name_and_the_hosts_package_route(host):
    """``{"name": modes.ROUTE_ENV, "value": "package"}`` on both package hosts."""
    assert surface.cli_surface(host)["dispatcher"]["route_env"] == {
        "name": mode_table.ROUTE_ENV, "value": "package",
    }


@pytest.mark.parametrize("host", HOSTS)
def test_each_servers_route_literal_is_a_tool_route_with_a_cell(host):
    """The generated value is a ``TOOL_ROUTES`` member this host carries a
    ``next_command`` template for — so a typo in the surface fails CI instead of
    printing a plausible wrong command at run time."""
    value = surface.cli_surface(host)["dispatcher"]["route_env"]["value"]
    assert value in mode_table.TOOL_ROUTES
    assert value in host_capabilities.next_command_routes(host)
    # And the accessor answers it, which is what resolve_route will ask.
    assert host_capabilities.next_command_template(host, value)


@pytest.mark.parametrize("host", HOSTS)
def test_the_surface_ignores_an_exported_route(host, monkeypatch):
    """The route is read from the source tree, never the builder's env."""
    monkeypatch.delenv(mode_table.ROUTE_ENV, raising=False)
    unset = surface.cli_surface_json(host)
    for value in ("skill", "", "plugin", "package"):
        monkeypatch.setenv(mode_table.ROUTE_ENV, value)
        assert surface.cli_surface_json(host) == unset, value


def _fixture_entry(monkeypatch, next_command):
    """Point ``host_capabilities.entry`` at a host whose cell is *next_command*."""
    real = host_capabilities.entry

    def entry(host_id, base=None):
        if host_id == "fixture-host":
            return {"next_command": next_command}
        return real(host_id, base)

    monkeypatch.setattr(host_capabilities, "entry", entry)


def test_the_generator_raises_when_no_tool_route_has_a_cell(monkeypatch):
    _fixture_entry(monkeypatch, {"default": "/x {args}", "skill": "/y {args}"})
    with pytest.raises(ValueError) as raised:
        surface._route_env("fixture-host")
    assert "fixture-host" in str(raised.value)
    assert "found 0" in str(raised.value)
    # The real hosts are unaffected by the fixture.
    assert surface._route_env("codex")["value"] == "package"


def test_the_generator_raises_on_more_than_one_candidate(monkeypatch):
    """Two members with a cell is a guess the build refuses to make."""
    monkeypatch.setattr(mode_table, "TOOL_ROUTES", ("package", "skill"))
    _fixture_entry(monkeypatch, {
        "default": "/x {args}", "package": "/p {args}", "skill": "/s {args}",
    })
    with pytest.raises(ValueError) as raised:
        surface._route_env("fixture-host")
    assert "found 2" in str(raised.value)


def test_a_host_without_a_package_route_cannot_be_built_for():
    """claude-code carries ``default`` and ``skill``: no surface to write."""
    with pytest.raises(ValueError):
        surface.cli_surface("claude-code")


@pytest.mark.parametrize("host", HOSTS)
def test_each_rows_optional_why_is_its_modes(host):
    by_name = {mode.name: mode for mode in mode_table.MODES}
    rows = surface.cli_surface(host)["modes"]
    assert rows
    for row in rows:
        assert "optional_why" in row, row["name"]
        assert row["optional_why"] == by_name[row["name"]].optional_why, row["name"]
    # At least one exposed row carries one, or the copy is untested.
    assert any(row["optional_why"] for row in rows)


@pytest.mark.parametrize("host", HOSTS)
def test_the_user_next_step_line_follows_the_outcomes_line(host):
    """§2.4: said once, in the description, before the model reads any result."""
    description = surface.cli_surface(host)["tool_description"]
    lines = description.split("\n")
    assert lines.count(mode_table.TOOL_USER_NEXT_STEP_LINE) == 1
    at = lines.index(mode_table.TOOL_OUTCOMES_LINE)
    assert lines[at + 1] == mode_table.TOOL_USER_NEXT_STEP_LINE


# ---------------------------------------------------------------------------
# What a failed result says the user can run (racing re-run 2026-10-04)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host", HOSTS)
def test_the_no_composed_command_line_follows_the_user_next_step_line(host):
    """OpenCode #4: after a failed generate-tasks the host model printed a
    "next phase" it composed itself. The description says, before any result,
    that the two command keys and `resume_command` are the only commands."""
    lines = surface.cli_surface(host)["tool_description"].split("\n")
    at = lines.index(mode_table.TOOL_USER_NEXT_STEP_LINE)
    assert lines[at + 1] == mode_table.TOOL_NO_COMPOSED_COMMAND_LINE
    assert lines.count(mode_table.TOOL_NO_COMPOSED_COMMAND_LINE) == 1
    for key in ("user_next_step", "resume_command", "no_command_reason"):
        assert f"`{key}" in mode_table.TOOL_NO_COMPOSED_COMMAND_LINE


@pytest.mark.parametrize("host", HOSTS)
def test_the_no_command_sentences_are_the_tables(host):
    rendered = surface.cli_surface(host)
    assert rendered["no_command_reason"] == mode_table.NO_COMMAND_REASON
    message = rendered["needs_human_message"]
    assert message["no_command"] == mode_table.NEEDS_HUMAN_MESSAGE["no_command"]
    assert "no_command_reason" in message["no_command"]
    # The redaction sentence now says where a redacted command went for the
    # user (its host spelling), and that only `resume_command` is theirs.
    assert "re-spelled" in message["template"]
    assert "only `resume_command` (and each `resume_answers` command) is " \
        "theirs to type" in message["template"]
    # Fix round 14: the stop's answers have a clause of their own.
    assert "{answers}" in message["answers"]
    assert "`resume_answers`" in message["answers"]


@pytest.mark.parametrize("host", HOSTS)
def test_the_redundant_flags_are_derived_from_the_composed_ones(host):
    """OpenCode #1: a restated `--yes` was refused four times in one session.
    The no-op set is every spelling of a valueless composed flag plus `--host`
    with this host's value — read from `baked_flags` and the parent parser."""
    from utils import cli_common

    rendered = surface.cli_surface(host)
    reserved = rendered["reserved_flags"]
    redundant = reserved["redundant"]
    by_dest = {a.dest: list(a.option_strings)
               for a in cli_common.build_parent_parser()._actions}
    assert redundant["flags"] == by_dest["yes"] + by_dest["exec_subagents"]
    assert redundant["valued"] == {"--host": host}
    # Run 6 (#10a/#14): no note — a restated flag means what the tool does.
    assert "note" not in redundant
    assert set(redundant["flags"]) | set(redundant["valued"]) <= set(reserved["exact"])
    # The conflicting ones stay refused.
    for kept_refused in ("--detach", "--if-not-running", "--max-detached"):
        assert kept_refused not in redundant["flags"]
        assert kept_refused not in redundant["valued"]
    assert not set(reserved["named_exception"]) & set(redundant["flags"])
    description = rendered["argument_descriptions"]["extra_flags"]
    assert f"--host {host}" in description
    assert "note" not in description and "ignored" in description


@pytest.mark.parametrize("host", HOSTS)
def test_the_user_command_spelling_is_the_resume_commands(host):
    """`dispatcher.user_command` plus a row's `user_flag` and the plan quoting
    rule spell exactly what `next_phase.resume_command` prints on this route,
    so a server's re-spelled command and the dispatcher's cannot drift."""
    from utils import next_phase

    rendered = surface.cli_surface(host)
    spell = rendered["dispatcher"]["user_command"]
    route = rendered["dispatcher"]["route_env"]["value"]
    assert spell["template"] == host_capabilities.next_command_template(host, route)
    rows = {row["name"]: row for row in rendered["modes"]}
    for row in rendered["modes"]:
        assert row["user_flag"] == f"--{row['name']}"
    workflow = {"phases": {"implement": {"status": "in_progress"}}}
    for plan in ("plans/x.md", "plans/my plan.md", 'plans/a"b.md'):
        expected = next_phase.resume_command(
            plan, host, route, "implement", workflow=workflow)
        args = " ".join([rows["implement"]["user_flag"],
                         next_phase.quote_plan_arg(plan), next_phase.RESUME_FLAG])
        assert spell["template"].replace(spell["args_slot"], args) == expected
