"""``util instructions <mode> --to-file``: the core as a file the host reads.

Claude Code's ``SKILL.md`` and the OpenCode file shim load the core this way,
so a capped tool result never hands them a fragment. The file holds exactly what
the printed form holds after its head banner; the printed form itself is
unchanged, since the package routes read it through a pipe.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils.cli_common import RESULT_MARKER, render_instruction_text, skill_dir  # noqa: E402

CORE_MODES = [mode.name for mode in mode_table.MODES if mode.instruction_file]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.chdir(root)
    return root.resolve()


def _run(argv, capsys):
    code = multi_llm.main(["util", "instructions", *argv])
    captured = capsys.readouterr()
    body = captured.out.rsplit(RESULT_MARKER, 1)[0]
    return code, body, captured.err


def _core_file_line(body: str):
    lines = body.splitlines()
    assert len(lines) == 1, body[:400]
    head, rest = lines[0].split(" ", 1)
    assert head == multi_llm.CORE_FILE_MARKER
    path, lines_part, bytes_part = rest.rsplit(" ", 2)
    return Path(path), int(lines_part.split("=")[1]), int(bytes_part.split("=")[1])


def _printed(mode, capsys, *extra):
    code, body, _err = _run([mode, *extra], capsys)
    assert code == 0
    return body


@pytest.mark.parametrize("mode", CORE_MODES)
def test_the_file_is_the_printed_core_without_its_banner(repo, mode, capsys):
    printed = _printed(mode, capsys)
    code, body, _err = _run([mode, "--to-file"], capsys)
    assert code == 0
    path, lines, size = _core_file_line(body)
    data = path.read_bytes()
    text = data.decode("utf-8")
    assert printed.endswith(text)
    assert printed[: -len(text)].startswith(multi_llm.TRUNCATION_BANNER_OPENING)
    assert (lines, size) == (text.count("\n"), len(data))
    assert text.endswith(f"{multi_llm.CORE_END_MARKER} {mode}\n")
    assert path.parent == repo / ".multi-llm" / "tmp" / "cores"
    assert multi_llm.CORE_FILE_NAME.fullmatch(path.name)
    assert (repo / ".multi-llm" / "tmp" / ".gitignore").read_text(encoding="utf-8") == "*\n"


@pytest.mark.parametrize("mode", ["review-plan", "implement"])
def test_the_printed_form_is_unchanged(repo, mode, capsys):
    """Byte for byte: banner, core, end line, as the package routes parse it."""
    source = SKILL_ROOT / mode_table.instruction_path(mode)
    core = render_instruction_text(source.read_text(encoding="utf-8"), skill_dir())
    body = core if core.endswith("\n") else core + "\n"
    expected = (multi_llm._truncation_notice(mode, core)  # noqa: SLF001
                + body + f"{multi_llm.CORE_END_MARKER} {mode}\n")
    assert _printed(mode, capsys) == expected
    assert not (repo / ".multi-llm").exists(), "the printed form wrote a file"


def test_host_keeps_its_verb_section_in_both_forms(repo, capsys):
    printed = _printed("review-plan", capsys, "--host", "claude-code")
    assert multi_llm._VERB_HEADING in printed  # noqa: SLF001
    code, body, _err = _run(["--to-file", "review-plan", "--host=claude-code"], capsys)
    assert code == 0
    path, _lines, _size = _core_file_line(body)
    text = path.read_text(encoding="utf-8")
    assert printed.endswith(text)
    assert text.index(multi_llm.CORE_END_MARKER) < text.index(multi_llm._VERB_HEADING)  # noqa: SLF001


def test_a_mode_without_a_core_writes_its_answer(repo, capsys):
    printed = _printed("status", capsys)
    code, body, _err = _run(["status", "--to-file"], capsys)
    assert code == 0
    path, _lines, _size = _core_file_line(body)
    assert path.read_text(encoding="utf-8") == printed
    assert printed.endswith(f"{multi_llm.CORE_END_MARKER} status\n")


def test_same_bytes_same_file_and_a_changed_core_a_new_one(repo, capsys, monkeypatch):
    first = _core_file_line(_run(["review-plan", "--to-file"], capsys)[1])[0]
    again = _core_file_line(_run(["review-plan", "--to-file"], capsys)[1])[0]
    assert again == first
    monkeypatch.setattr(multi_llm, "render_instruction_text",
                        lambda text, base: "a different core\n")
    other = _core_file_line(_run(["review-plan", "--to-file"], capsys)[1])[0]
    assert other != first and first.exists() and other.exists()
    assert other.read_text(encoding="utf-8").startswith("a different core\n")


def _aged(path: Path) -> None:
    stamp = time.time() - multi_llm.CORE_FILE_MAX_AGE_S - 60
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def test_core_files_older_than_the_limit_are_pruned(repo, capsys):
    folder = repo / ".multi-llm" / "tmp" / "cores"
    folder.mkdir(parents=True)
    old = folder / "core-ask-000000000000.md"
    old_multiword = folder / "core-apply-code-fixes-0123456789ab.md"
    recent = folder / "core-ask-111111111111.md"
    for path in (old, old_multiword, recent):
        path.write_text("x\n", encoding="utf-8")
    _aged(old)
    _aged(old_multiword)
    _core_file_line(_run(["review-plan", "--to-file"], capsys)[1])
    assert not old.exists() and not old_multiword.exists()
    assert recent.exists()


@pytest.mark.parametrize("name", [
    "core-design-notes.md",          # a user's notes: no twelve-digit hash
    "core-ask-00000000000.md",       # eleven digits
    "core-ask-0000000000000.md",     # thirteen
    "core-ask-00000000000G.md",      # not hex
    "core-ask-000000000000.md.bak",
    "core-Ask-000000000000.md",      # not a mode word
    "notes-ask-000000000000.md",
])
def test_the_sweep_removes_only_the_generated_name_format(repo, capsys, name):
    folder = repo / ".multi-llm" / "tmp" / "cores"
    folder.mkdir(parents=True)
    other = folder / name
    other.write_text("user-owned notes\n", encoding="utf-8")
    _aged(other)
    _core_file_line(_run(["ask", "--to-file"], capsys)[1])
    assert other.read_text(encoding="utf-8") == "user-owned notes\n"


def test_the_sweep_never_reads_the_shared_temp_dir(repo, capsys):
    """``tmp/`` holds every run's scratch files: an old file there of the very
    name format the sweep removes in ``cores/`` stays."""
    tmp = repo / ".multi-llm" / "tmp"
    tmp.mkdir(parents=True)
    neighbours = [tmp / "core-ask-000000000000.md", tmp / "core-design-notes.md",
                  tmp / "implementation_tasks_x_0000.json"]
    for path in neighbours:
        path.write_text("x\n", encoding="utf-8")
        _aged(path)
    path = _core_file_line(_run(["ask", "--to-file"], capsys)[1])[0]
    assert path.parent == tmp / "cores"
    assert all(neighbour.exists() for neighbour in neighbours)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
@pytest.mark.parametrize("target_kind", ["file", "folder", "dangling"])
def test_the_sweep_never_removes_a_link(repo, tmp_path, capsys, target_kind):
    folder = repo / ".multi-llm" / "tmp" / "cores"
    folder.mkdir(parents=True)
    victim = tmp_path / "victim"
    if target_kind == "file":
        victim.write_text("untouched\n", encoding="utf-8")
    elif target_kind == "folder":
        victim.mkdir()
    link = folder / "core-ask-000000000000.md"
    link.symlink_to(victim, target_is_directory=target_kind == "folder")
    _aged(link)
    if target_kind != "dangling":
        _aged(victim)
    _core_file_line(_run(["ask", "--to-file"], capsys)[1])
    assert link.is_symlink()
    if target_kind == "file":
        assert victim.read_text(encoding="utf-8") == "untouched\n"


def test_the_sweep_never_removes_a_folder(repo, capsys):
    folder = repo / ".multi-llm" / "tmp" / "cores"
    inner = folder / "core-ask-000000000000.md"
    inner.mkdir(parents=True)
    (inner / "kept.txt").write_text("x\n", encoding="utf-8")
    _aged(inner)
    _core_file_line(_run(["ask", "--to-file"], capsys)[1])
    assert (inner / "kept.txt").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs")
def test_the_sweep_never_removes_another_kind_of_file(repo, capsys):
    folder = repo / ".multi-llm" / "tmp" / "cores"
    folder.mkdir(parents=True)
    fifo = folder / "core-ask-000000000000.md"
    os.mkfifo(fifo)
    _aged(fifo)
    _core_file_line(_run(["ask", "--to-file"], capsys)[1])
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)


def test_the_folder_reports_a_link_as_a_link(tmp_path):
    """``PromptFolder.lstat``, which the sweep inspects each entry with,
    never follows the link at the name."""
    from utils import prompt_files

    if not hasattr(os, "symlink"):  # pragma: no cover
        pytest.skip("no symlinks")
    (tmp_path / "cores").mkdir()
    target = tmp_path / "target.md"
    target.write_text("x\n", encoding="utf-8")
    (tmp_path / "cores" / "link.md").symlink_to(target)
    pin = prompt_files.PhasePin(root=str(tmp_path), parts=())
    with prompt_files.PromptFolder.open(pin, create=False, folder="cores") as folder:
        assert stat.S_ISLNK(folder.lstat("link.md").st_mode)
        with pytest.raises(FileNotFoundError):
            folder.lstat("missing.md")
        with pytest.raises(OSError):
            folder.lstat("../target.md")


def test_from_a_subdirectory_the_file_goes_under_the_repository_root(repo, capsys, monkeypatch):
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    path = _core_file_line(_run(["ask", "--to-file"], capsys)[1])[0]
    assert path.parent == repo / ".multi-llm" / "tmp" / "cores"


def test_outside_a_repository_the_cwd_stands_in(tmp_path, capsys, monkeypatch):
    here = tmp_path / "plain"
    here.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    path = _core_file_line(_run(["ask", "--to-file"], capsys)[1])[0]
    assert path.parent == here.resolve() / ".multi-llm" / "tmp" / "cores"


@pytest.mark.parametrize("plant,where", [
    ("file", ".multi-llm"), ("link", ".multi-llm"),
    ("file", ".multi-llm/tmp"), ("link", ".multi-llm/tmp"),
    ("file", ".multi-llm/tmp/cores"), ("link", ".multi-llm/tmp/cores"),
])
def test_an_unwritable_folder_prints_the_core_instead(repo, tmp_path, capsys, plant, where):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    planted = repo / where
    planted.parent.mkdir(parents=True, exist_ok=True)
    if plant == "file":
        planted.write_text("not a folder\n", encoding="utf-8")
    else:
        if not hasattr(os, "symlink"):  # pragma: no cover
            pytest.skip("no symlinks")
        planted.symlink_to(elsewhere, target_is_directory=True)
    code, body, err = _run(["review-plan", "--to-file"], capsys)
    assert code == 0
    assert multi_llm.CORE_FILE_MARKER not in body
    assert body == _printed("review-plan", capsys)
    assert "could not write the core file" in err
    assert list(elsewhere.iterdir()) == [], "wrote through the planted link"


def test_the_banner_counts_a_host_truncation_note_as_an_incomplete_core():
    """A host that cuts the middle keeps the end line, so the banner's check
    names the host's own note as well; and the whole banner stays inside the
    first 2 KB that Claude Code's stub keeps."""
    for mode in mode_table.MODES:
        notice = multi_llm._truncation_notice(mode.name, "x" * 99_999)  # noqa: SLF001
        assert len(notice.encode("utf-8")) < 2000, mode.name
        flat = " ".join(notice.split())
        assert ("that it truncated or omitted part of this output means you do not "
                "have the whole core, even with that line there") in flat
        assert notice.endswith("=== the core begins on the next line ===\n")


def test_an_unknown_option_names_every_option(repo, capsys):
    code, _body, err = _run(["review-plan", "--to-files"], capsys)
    assert code == 3
    assert "`--host <id>`, `--to-file`, `--page <i>` and `--page-size <bytes>`" in err
