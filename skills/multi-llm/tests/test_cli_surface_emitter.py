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
  the mode table's ``runs_workflow`` column, never from a hand list.
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
from utils import subagent_tasks, unattended  # noqa: E402

#: Every host a package route ships this surface for.
HOSTS = ("opencode", "codex")

#: The four tool properties of D3 rule 1 — the same on both hosts.
PROPERTIES = {"mode", "plan", "question", "extra_flags"}


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
    from utils import host_install

    monkeypatch.delenv(unattended.UV_RUN_ARGS_ENV, raising=False)
    assert surface.cli_surface(host)["uninstall_command"] == (
        host_install.uninstall_line(host, surface.SKILL_DIR_TOKEN)
    )


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

    assert surface.SURFACE_SCHEMA_VERSION == 4
    assert opencode_build.SURFACE_SCHEMA_VERSION == surface.SURFACE_SCHEMA_VERSION
    for host in HOSTS:
        assert surface.cli_surface(host)["schema_version"] == 4
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
    assert set(described) == PROPERTIES
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
    """The apply family, ``implement``, and ``full`` which runs them — derived."""
    write_phases = set(subagent_tasks.PHASE_BY_WRITE_KIND.values())
    rows = {r["name"]: r for r in surface.cli_surface("codex")["modes"]}
    for mode in mode_table.user_facing_modes():
        expected = mode.phase in write_phases or (
            mode.runs_workflow and any(
                m.phase in write_phases
                for m in mode_table.MODES if m.is_workflow_phase
            )
        )
        assert rows[mode.name]["writes_worktree"] is expected, mode.name
    # Spot-check the reading the plan names, so a derivation that went wrong in
    # both places at once still fails.
    assert {n for n, r in rows.items() if r["writes_worktree"]} == {
        "apply-suggestions", "apply-task-suggestions", "apply-code-fixes",
        "implement", "full",
    }


@pytest.mark.parametrize("host", HOSTS)
def test_worked_examples_are_selected_by_shape(host):
    description = surface.cli_surface(host)["tool_description"]
    block = description.split("\nExamples:\n", 1)[1]
    examples = [json.loads(line) for line in block.splitlines()]
    for example in examples:
        assert set(example) == PROPERTIES
    by_mode = {e["mode"]: e for e in examples}
    assert by_mode[mode_table.DEFAULT_MODE]["plan"] == mode_table.EXAMPLE_PLAN_PATH
    assert by_mode["init"] == {
        "mode": "init", "plan": "", "question": "", "extra_flags": []
    }
    assert by_mode["ask"]["question"] == mode_table.EXAMPLE_QUESTION
    assert set(mode_table.ALIASES) & set(by_mode), "no alias example"
