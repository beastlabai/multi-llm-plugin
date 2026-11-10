"""No line a host reads from a file is longer than its read tool shows.

A file form's completeness test is a line count ("read through line ``<n>``"),
and opencode's read tool clips any line over 2,000 characters without changing
that count, so a long line loses its tail where nothing says so. These are the
files a host reads line by line: the core as ``util instructions --to-file``
writes it (the file shims with no host section, the OpenCode package with its
own), and every reference a core sends the host to
(docs/development/hosts.md, "Host-neutral instruction core").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils.cli_common import render_instruction_text  # noqa: E402

#: opencode's read tool: "Any line longer than 2000 characters is truncated".
READ_LINE_LIMIT = 2000

#: A skill directory as long as an install path gets (Windows' 260), with a
#: space, so a line carrying a command or two is measured as a host sees it.
LONG_SKILL_DIR = "/" + "/".join(["plugin cache"] + ["d" * 40] * 5) + "/" + "s" * (
    259 - len("/plugin cache") - 5 * 41 - 1)

#: The host sections a route writes into the file it reads: none (the file
#: shims' plain `--to-file`) and the OpenCode package's. Claude Code's section
#: carries a sentence over the limit, but no route reads it from a file: its shim
#: holds those sentences and reads the core without `--host`.
FILE_HOSTS = (None, "opencode")

#: Generated human documentation of the capability matrix, which no core,
#: reference or shim sends a host to read.
NOT_READ = {"host-capabilities.md"}

CORE_MODES = sorted(
    (mode.name, mode.instruction_file) for mode in mode_table.MODES
    if mode.instruction_file)


def _long_lines(text: str):
    return [(number, len(line)) for number, line in enumerate(text.split("\n"), 1)
            if len(line) > READ_LINE_LIMIT]


def test_the_long_skill_dir_is_install_length():
    assert len(LONG_SKILL_DIR) == 259 and " " in LONG_SKILL_DIR


@pytest.mark.parametrize("host_id", FILE_HOSTS)
@pytest.mark.parametrize("mode, relative", CORE_MODES)
def test_no_core_line_is_clipped(mode, relative, host_id):
    core = render_instruction_text(
        (SKILL_ROOT / "instructions" / relative).read_text(encoding="utf-8"),
        LONG_SKILL_DIR)
    text = core + "\n" + (multi_llm._verb_section(host_id) if host_id else "")
    assert _long_lines(text) == [], (
        f"{relative} (host section: {host_id}) has lines over {READ_LINE_LIMIT} "
        "characters once rendered; wrap them at a sentence, keeping the indent")


@pytest.mark.parametrize(
    "name", sorted(path.name for path in (SKILL_ROOT / "references").glob("*.md")))
def test_no_reference_line_is_clipped(name):
    if name in NOT_READ:
        pytest.skip("generated documentation no host is sent to read")
    text = (SKILL_ROOT / "references" / name).read_text(encoding="utf-8")
    assert _long_lines(text) == [], (
        f"references/{name} has lines over {READ_LINE_LIMIT} characters")


def test_nothing_sends_a_host_to_the_unread_references():
    sources = [*(SKILL_ROOT / "instructions").glob("*.md"),
               *(SKILL_ROOT / "references").glob("*.md"),
               SKILL_ROOT / "SKILL.md", *(SKILL_ROOT / "hosts").rglob("*.tmpl"),
               *(SKILL_ROOT / "hosts" / "snippets").glob("*.md")]
    for name in NOT_READ:
        naming = sorted(str(path.relative_to(SKILL_ROOT)) for path in sources
                        if path.name != name
                        and name in path.read_text(encoding="utf-8"))
        assert naming == [], (name, naming)
