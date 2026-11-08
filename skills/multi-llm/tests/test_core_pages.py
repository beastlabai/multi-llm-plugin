"""``util instructions <mode> --page <i>``: the core in pages a cutting shell passes whole.

Codex has no file-read tool and keeps only the head and the tail of a long
shell output, so the Codex file shim reads the core page by page
(docs/development/hosts.md, "Paged delivery"). These tests hold the paging to
its contract: every byte of the ``--to-file`` text on exactly one page, each
page framed and bounded, the sha of the whole render on every page, page 1's
banner stating the completeness rule, and the printed and file forms untouched.
Pure in-process calls, no spend.
"""

from __future__ import annotations

import hashlib
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils.cli_common import RESULT_MARKER  # noqa: E402

CORE_MODES = [mode.name for mode in mode_table.MODES if mode.instruction_file]
NO_CORE_MODES = [mode.name for mode in mode_table.MODES if not mode.instruction_file]

#: The floor, odd sizes that land mid-line, the default and one past every core.
SIZES = [multi_llm.CORE_PAGE_SIZE_MIN, 3_001, 7_919, None, 200_000]

HEADER = re.compile(
    re.escape(multi_llm.CORE_PAGE_MARKER)
    + r" mode=(?P<mode>\S+) page=(?P<page>\d+)/(?P<count>\d+)"
    r" sha=(?P<sha>[0-9a-f]{12}) bytes=(?P<bytes>\d+)"
)
END = re.compile(
    re.escape(multi_llm.CORE_PAGE_END_MARKER)
    + r" mode=(?P<mode>\S+) page=(?P<page>\d+)/(?P<count>\d+) sha=(?P<sha>[0-9a-f]{12})"
)
BANNER_CLOSE = "=== the core begins on the next line ===\n"


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
    return code, captured.out.rsplit(RESULT_MARKER, 1)[0], captured.err


def _size_args(size):
    return [] if size is None else ["--page-size", str(size)]


def _parse(out: str, mode: str):
    """``(header, banner, body, end)`` of one printed page; asserts the framing."""
    first, rest = out.split("\n", 1)
    header = HEADER.fullmatch(first)
    assert header, first
    assert out.endswith("\n")
    inner, last = rest[:-1].rsplit("\n", 1)
    inner += "\n"
    end = END.fullmatch(last)
    assert end, last
    banner = ""
    if header["page"] == "1":
        assert inner.startswith(multi_llm.TRUNCATION_BANNER_OPENING), inner[:200]
        cut = inner.index(BANNER_CLOSE) + len(BANNER_CLOSE)
        banner, inner = inner[:cut], inner[cut:]
    else:
        assert multi_llm.TRUNCATION_BANNER_OPENING not in inner[:200]
    assert header["mode"] == end["mode"] == mode
    assert (header["page"], header["count"], header["sha"]) == (
        end["page"], end["count"], end["sha"])
    assert int(header["bytes"]) == len(inner.encode("utf-8"))
    return header, banner, inner, end


def _all_pages(mode, capsys, size=None, *extra):
    """Every page of *mode* at *size*: a list of ``(header, banner, body)``."""
    code, out, err = _run([mode, "--page", "1", *_size_args(size), *extra], capsys)
    assert code == 0, err
    header, banner, body, _end = _parse(out, mode)
    pages = [(header, banner, body)]
    for index in range(2, int(header["count"]) + 1):
        code, out, err = _run([mode, *extra, *_size_args(size), f"--page={index}"], capsys)
        assert code == 0, err
        page_header, page_banner, page_body, _end = _parse(out, mode)
        assert int(page_header["page"]) == index
        pages.append((page_header, page_banner, page_body))
    return pages


def _file_text(mode, capsys, *extra) -> str:
    code, out, _err = _run([mode, "--to-file", *extra], capsys)
    assert code == 0
    path = Path(out.split(" ")[1])
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The pages are the --to-file text, every byte once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"size-{s or 'default'}")
@pytest.mark.parametrize("mode", CORE_MODES + NO_CORE_MODES)
def test_the_pages_joined_are_the_core_exactly(repo, mode, size, capsys):
    pages = _all_pages(mode, capsys, size)
    joined = "".join(body for _header, _banner, body in pages)
    code, printed, _err = _run([mode], capsys)
    assert code == 0
    # The file form's text, and the printed form after its head banner.
    assert joined == _file_text(mode, capsys)
    assert printed.endswith(joined)
    banner = printed[: len(printed) - len(joined)]
    assert banner == "" or banner.startswith(multi_llm.TRUNCATION_BANNER_OPENING)
    assert joined.endswith(f"{multi_llm.CORE_END_MARKER} {mode}\n")
    # One count and one sha on every page: the sha of the whole render, the
    # digest the core file is named by.
    sha = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
    assert {(h["count"], h["sha"]) for h, _b, _p in pages} == {(str(len(pages)), sha)}
    assert all(body.endswith("\n") and body for _h, _b, body in pages)


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"size-{s or 'default'}")
@pytest.mark.parametrize("mode", CORE_MODES)
def test_no_page_prints_more_than_its_size(repo, mode, size, capsys):
    """Between its two lines a page holds at most the page size, page 1's
    banner included, unless one line alone is longer."""
    limit = size or multi_llm.CORE_PAGE_SIZE_DEFAULT
    for header, banner, body in _all_pages(mode, capsys, size):
        printed = len((banner + body).encode("utf-8"))
        one_line = body.count("\n") == 1
        assert printed <= limit or one_line, (header.group(0), printed)


def test_the_host_section_is_paged_with_the_core(repo, capsys):
    pages = _all_pages("review-plan", capsys, 3_001, "--host", "codex")
    joined = "".join(body for _h, _b, body in pages)
    assert joined == _file_text("review-plan", capsys, "--host", "codex")
    assert multi_llm._VERB_HEADING in pages[-1][2]  # noqa: SLF001
    plain = _all_pages("review-plan", capsys, 3_001)
    # The sha covers what is paged, the --host section included.
    assert pages[0][0]["sha"] != plain[0][0]["sha"]
    assert joined.startswith("".join(body for _h, _b, body in plain))


def test_the_default_size_pages_every_core_and_reads_the_biggest_in_four(repo, capsys):
    counts = {mode: int(_all_pages(mode, capsys)[0][0]["count"]) for mode in CORE_MODES}
    assert max(counts.values()) <= 4, counts
    assert counts["implement"] > 1


def test_a_page_past_the_last_is_refused_and_prints_no_core(repo, capsys):
    count = int(_all_pages("implement", capsys)[0][0]["count"])
    code, out, err = _run(["implement", "--page", str(count + 1)], capsys)
    assert code == 3
    assert out == ""
    assert f"--page takes 1 to {count}" in err


def test_the_printed_and_file_forms_take_no_page_framing(repo, capsys):
    """Without --page both other forms are what they were: no page line in
    either, and the printed form still opens with its own banner."""
    code, printed, _err = _run(["implement"], capsys)
    assert code == 0
    assert printed.startswith(multi_llm._truncation_notice(  # noqa: SLF001
        "implement", "")[:60])
    for marker in (multi_llm.CORE_PAGE_MARKER, multi_llm.CORE_PAGE_END_MARKER):
        assert marker not in printed
        assert marker not in _file_text("implement", capsys)


# ---------------------------------------------------------------------------
# Page 1's banner states the rule
# ---------------------------------------------------------------------------

def _banner(mode, capsys, *extra):
    code, out, _err = _run([mode, "--page", "1", *extra], capsys)
    assert code == 0
    return _parse(out, mode)


def test_the_banner_states_the_paged_completeness_rule(repo, capsys):
    header, banner, _body, _end = _banner("implement", capsys)
    count, sha = header["count"], header["sha"]
    flat = " ".join(banner.split())
    assert f"Run the same command for pages 2 to {count}, changing only the --page number" in flat
    assert f"every page from 1 to {count} arrived whole" in flat
    assert f"{multi_llm.CORE_PAGE_MARKER} mode=implement page=<i>/{count} sha={sha}" in banner
    assert f"{multi_llm.CORE_PAGE_END_MARKER} mode=implement page=<i>/{count} sha={sha}" in banner
    assert f"    {multi_llm.CORE_END_MARKER} implement\n" in banner
    assert ("that it truncated or omitted part of that output means the page is "
            "incomplete, even with both of its lines there") in flat
    half = multi_llm.CORE_PAGE_SIZE_DEFAULT // 2
    assert f"Fetch it again with --page-size {half}" in flat
    assert "start again at page 1 with that size" in flat
    assert "never combine pages of two sizes" in flat
    assert "Do not run any command from the core until you hold every page." in flat


def test_the_banner_names_half_the_size_it_was_served_at(repo, capsys):
    _header, banner, _body, _end = _banner("implement", capsys, "--page-size", "9001")
    assert "--page-size 4500" in " ".join(banner.split())


def test_at_the_floor_the_banner_says_to_stop(repo, capsys):
    floor = multi_llm.CORE_PAGE_SIZE_MIN
    _header, banner, _body, _end = _banner("implement", capsys, "--page-size", str(floor))
    flat = " ".join(banner.split())
    assert f"--page-size {floor} is already the smallest" in flat
    assert "stop and tell the user" in flat
    assert "Fetch it again" not in flat
    # Above the floor, half of the size is never below it.
    for size in (floor + 1, 2 * floor - 1):
        _header, banner, _body, _end = _banner("implement", capsys, "--page-size", str(size))
        assert f"Fetch it again with --page-size {floor}." in " ".join(banner.split())


def test_a_one_page_core_says_there_is_nothing_else_to_fetch(repo, capsys):
    header, banner, _body, _end = _banner("init", capsys, "--page-size", "200000")
    assert header["count"] == "1"
    flat = " ".join(banner.split())
    assert "there is no other page to fetch" in flat
    assert "pages 2 to" not in flat


@pytest.mark.parametrize("mode", CORE_MODES + NO_CORE_MODES)
def test_the_banner_is_dropped_like_the_printed_one(repo, mode, capsys):
    """It opens with the printed banner's opening and closes with its closing
    line, so a reader that drops one drops the other; and it is short."""
    _header, banner, _body, _end = _banner(mode, capsys)
    assert banner.startswith(multi_llm.TRUNCATION_BANNER_OPENING)
    assert banner.endswith(BANNER_CLOSE)
    assert banner.count(multi_llm.TRUNCATION_BANNER_OPENING) == 1
    assert len(banner.encode("utf-8")) < 2000


# ---------------------------------------------------------------------------
# The cutter itself
# ---------------------------------------------------------------------------

def _texts():
    rng = random.Random(79)
    words = ["a", "bb", "ccc", "déjà", "→", "日本語", "x" * 40, ""]
    for _ in range(60):
        lines = [" ".join(rng.choice(words) for _ in range(rng.randrange(0, 30)))
                 for _ in range(rng.randrange(0, 40))]
        text = "\n".join(lines)
        yield text + ("\n" if rng.random() < 0.7 else "")
    yield ""
    yield "\n\n\n"
    yield "one line with no newline"
    yield "y" * 9_000 + "\nshort\n" + "z" * 50 + "\n"


@pytest.mark.parametrize("size,first", [(1, 1), (7, 0), (40, 40), (64, 10), (500, 120)])
def test_the_cutter_keeps_every_byte_once_on_line_boundaries(size, first):
    for text in _texts():
        pages = multi_llm._core_pages(text, size, first=first)  # noqa: SLF001
        assert "".join(pages) == text
        assert pages, "always at least one page"
        for index, page in enumerate(pages):
            room = first if index == 0 else size
            one_line = "\n" not in page[:-1]
            assert len(page.encode("utf-8")) <= room or one_line, (index, page[:80])
            if index < len(pages) - 1:
                assert page.endswith("\n"), "a page ends where a line ends"


def test_the_cutter_fills_pages_greedily():
    text = "".join(f"{n:03d}\n" for n in range(100))  # 4 bytes a line
    pages = multi_llm._core_pages(text, 40, first=12)  # noqa: SLF001
    assert [len(p) for p in pages] == [12] + [40] * 9 + [28]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv,needle", [
    (["implement", "--page", "0"], "at least 1, got '0'"),
    (["implement", "--page", "x"], "at least 1, got 'x'"),
    (["implement", "--page", "²"], "at least 1, got '²'"),
    (["implement", "--page"], "`util instructions --page` needs a page number"),
    (["implement", "--page="], "`util instructions --page` needs a page number"),
    (["implement", "--page", "1", "--page-size", "1999"], "at least 2000, got '1999'"),
    (["implement", "--page", "1", "--page-size"], "needs a size in bytes"),
    (["implement", "--page-size", "8000"], "add `--page <i>`"),
    (["implement", "--page", "1", "--to-file"], "`--to-file` or `--page`, not both"),
])
def test_a_bad_page_option_is_exit_3_with_nothing_on_stdout(repo, argv, needle, capsys):
    code, out, err = _run(argv, capsys)
    assert code == 3
    assert out == ""
    assert needle in err
    assert not (repo / ".multi-llm").exists()


def test_an_unknown_option_names_every_option(repo, capsys):
    code, _out, err = _run(["review-plan", "--pages", "1"], capsys)
    assert code == 3
    assert "`--host <id>`, `--to-file`, `--page <i>` and `--page-size <bytes>`" in err


def test_the_help_names_the_page_options(capsys):
    code = multi_llm.main(["util", "instructions", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--page <i>" in out and "--page-size <bytes>" in out
    assert str(multi_llm.CORE_PAGE_SIZE_DEFAULT) in out


# ---------------------------------------------------------------------------
# Who reads which form
# ---------------------------------------------------------------------------

def _rendered_shim(host_id: str) -> str:
    from utils import host_install

    template = host_install.template_path(host_id).read_text(encoding="utf-8")
    return host_install.render(host_id, skill="/fixture/skill", template=template)


def test_the_codex_shim_reads_pages_and_states_the_rule():
    flat = " ".join(_rendered_shim("codex").split())
    assert "util instructions <mode> --page 1" in flat
    assert "--to-file" not in flat
    for needle in (
        "Fetch pages 2 to `<n>` with the same command, changing only `--page`",
        "follow the core only once you hold every page whole",
        f"opens with its `{multi_llm.CORE_PAGE_MARKER}` line",
        f"closes with its `{multi_llm.CORE_PAGE_END_MARKER}` line",
        "all of those lines carry the same `sha=`",
        "a note in any page's output that it truncated or omitted part of it means "
        "that page is incomplete, both lines or not",
        "the smaller `--page-size` that page 1's banner names",
        "starting over at page 1 with that size",
        # The composite cores load phase cores with a host-neutral command.
        "Load every core this way, including a phase's core that a core tells you "
        "to load with `util instructions <phase>`: add `--page 1` to that command",
    ):
        assert needle in flat, needle


def test_every_core_that_loads_a_phase_core_uses_the_plain_command():
    """The shim's "load every core this way" sentence covers these lines: each
    is the host-neutral read command, which the Codex shim pages."""
    loading = {}
    for path in sorted((SKILL_ROOT / "instructions").glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "multi_llm.py\" util instructions " in line:
                loading[path.name] = line.split("util instructions ", 1)[1]
    assert loading, "no core loads another; drop the shim's sentence and this test"
    for name, rest in loading.items():
        assert "--to-file" not in rest and "--page" not in rest, (name, rest)


def test_the_page_sha_is_the_digest_the_core_file_is_named_by(repo, capsys):
    header = _all_pages("review-plan", capsys)[0][0]
    code, out, _err = _run(["review-plan", "--to-file"], capsys)
    assert code == 0
    name = Path(out.split(" ")[1]).name
    assert name == f"core-review-plan-{header['sha']}.md"


def test_the_hosts_that_read_files_keep_the_file_form():
    opencode = " ".join(_rendered_shim("opencode").split())
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for text in (opencode, skill):
        assert "--to-file" in text
        assert "--page" not in text
