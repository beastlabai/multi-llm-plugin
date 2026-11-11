"""The codex MCP server's runbook pages (docs/development/hosts.md, "Paged
delivery on the Codex tool route").

An unfinished run's result carries page 1 of its runbook, framed the way
``util instructions --page`` frames the core, and closes with the rule that says
when the whole runbook is held; a call whose ``extra_flags`` carry the page
option is answered with one page before anything can run. Driven through the
real server over its wire, with the scripted uv of ``test_codex_mcp_server.py``
(which records every workflow spawn), plus the server's paging functions read
straight from its source.

Nothing here is ``package``/``container``/``live``; no provider runs.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys

import pytest

import multi_llm
from tests.test_codex_mcp_server import (  # noqa: F401  (the fixtures)
    CODES, DISPATCHER, MARKER, REVIEW, SERVER_SRC, SKILL_ROOT, STATES, SURFACE,
    dispatcher_argv, harness, launch_step, make_plugin_root, plugin_root,
    range_call, result_line, runbook_text, surface_mod, turn_meta)

PAGES = SURFACE["runbook"]["pages"]
END_OF_CORE = SURFACE["runbook"]["core_end_marker"]
IMPLEMENT = {"mode": "implement", "plan": "plans/p.md", "question": "",
             "extra_flags": []}
WRITE_META = turn_meta(sandbox_mode="workspace-write")
DETACHED_MODES = [row["name"] for row in SURFACE["modes"] if row["supports_detach"]]


def _server_module():
    """The server's own functions, from the file it ships as (stdlib only)."""
    spec = importlib.util.spec_from_file_location("mllm_codex_server_pages", SERVER_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVER = _server_module()

HEADER = re.compile(
    "^" + re.escape(multi_llm.CORE_PAGE_MARKER) + r" mode=(?P<mode>\S+) "
    r"page=(?P<page>[0-9]+)/(?P<pages>[0-9]+) sha=(?P<sha>[0-9a-f]+) "
    r"bytes=(?P<bytes>[0-9]+)$")
END = re.compile(
    "^" + re.escape(multi_llm.CORE_PAGE_END_MARKER) + r" mode=(?P<mode>\S+) "
    r"page=(?P<page>[0-9]+)/(?P<pages>[0-9]+) sha=(?P<sha>[0-9a-f]+)$")


def split_page(framed: str):
    """``(header fields, body)`` of one framed page, its frame checked whole."""
    assert framed.endswith("\n")
    lines = framed.split("\n")
    assert lines[-1] == ""
    head, end = HEADER.match(lines[0]), END.match(lines[-2])
    assert head and end, (lines[0], lines[-2])
    body = "\n".join(lines[1:-2]) + "\n"
    assert int(head["bytes"]) == len(body.encode("utf-8"))
    fields = head.groupdict()
    assert {key: fields[key] for key in ("mode", "page", "pages", "sha")} == end.groupdict()
    return fields, body


def page_call(mode, page, size=None, **extra) -> dict:
    flags = [PAGES["option"], str(page)]
    if size is not None:
        flags += [PAGES["size_option"], str(size)]
    call = {"mode": mode, "plan": "", "question": "", "extra_flags": flags}
    call.update(extra)
    return call


def read_pages(client, mode, size):
    """Every page of *mode*'s runbook at *size*, page 1 first, over one server:
    ``(bodies, sha)``, each page's frame and its ``runbook_pages`` checked."""
    bodies, sha, count, page = [], None, None, 1
    while count is None or page <= count:
        envelope, result = client.envelope(client.start_call(page_call(mode, page, size)))
        assert envelope["ok"] is True and not result["isError"], envelope
        assert list(envelope)[-2:] == ["runbook", "runbook_pages"]
        fields, body = split_page(envelope["runbook"])
        info = envelope["runbook_pages"]
        count = info["pages"] if count is None else count
        assert (int(fields["page"]), int(fields["pages"])) == (page, count)
        assert (info["page"], info["pages"], info["page_size"]) == (page, count, size)
        assert fields["mode"] == mode and fields["sha"] == info["sha"]
        assert sha is None or fields["sha"] == sha  # one sha on every page
        sha = fields["sha"]
        if page < count:
            assert info["next_page"] == page_call(mode, page + 1, size)
        else:
            assert "next_page" not in info
        bodies.append(body)
        page += 1
    return bodies, sha


def codex_token_cut(text: str, tokens: int) -> str:
    """Codex's head-and-tail cut at a *tokens* budget: 4 bytes a token, half
    each side, the middle replaced by its notice (06b; the U79 review reproduced
    the MCP route's cut this way at 1,200 and 10,000 tokens)."""
    budget = tokens * 4
    if len(text) <= budget:
        return text
    half = budget // 2
    return "{}…{} tokens truncated…{}".format(
        text[:half], (len(text) - budget) // 4, text[-(budget - half):])


def padded_runbook(lines: int) -> str:
    """The fixture runbook with *lines* workflow steps: 1,400 is implement's size."""
    return runbook_text().replace(
        "# Core\n", "# Core\n" + "WORKFLOW-PADDING: a step the model owes.\n" * lines)


def blocked_implement(harness, **kwargs):
    blocked = {"exit": CODES["needs_human"], "stdout": [
        "progress line {} ".format(n) + "x" * 80 for n in range(60)] + [
        result_line(mode="implement", phase="implement",
                    state=STATES["needs_human"], exit=CODES["needs_human"])]}
    return harness.run(IMPLEMENT, [launch_step(), blocked], meta=WRITE_META, **kwargs)


def unpaged_surface_text() -> str:
    """The surface as it was before ``runbook.pages``: a server reading it
    attaches the runbook whole and forwards every flag."""
    surface = surface_mod.cli_surface("codex")
    del surface["runbook"]["pages"]
    return json.dumps(surface, indent=2) + "\n"


@pytest.fixture(scope="module")
def unpaged_root(tmp_path_factory):
    return make_plugin_root(tmp_path_factory.mktemp("unpaged-root"),
                            surface_text=unpaged_surface_text())


@pytest.fixture(scope="module")
def real_runbooks():
    """``util instructions <mode> --host <id>`` from the real dispatcher, per
    detachable mode: plain python, no uv, no provider."""
    out = {}
    for mode in DETACHED_MODES:
        done = subprocess.run(
            [sys.executable, str(SKILL_ROOT / DISPATCHER["script"])]
            + list(DISPATCHER["instructions"]) + [
                mode, DISPATCHER["host_option"], SURFACE["host"]],
            capture_output=True, text=True, encoding="utf-8", timeout=120, check=True)
        out[mode] = done.stdout
    return out


# ===========================================================================
# The format: util instructions' cut, lines and sha
# ===========================================================================

class TestTheFormatIsUtilInstructions:
    @pytest.mark.parametrize("size", [2000, 3001, 8000, 16000, 10 ** 6])
    def test_the_cut_is_util_instructions_cut(self, real_runbooks, size):
        for mode, text in real_runbooks.items():
            bodies = SERVER.runbook_page_bodies(text, size)
            assert bodies == multi_llm._core_pages(text, size, first=size), mode
            assert "".join(bodies) == text
            for body in bodies:
                assert len(body.encode("utf-8")) <= size or body.count("\n") <= 1

    def test_a_text_without_a_last_newline_keeps_every_byte(self):
        text = "één\r\nline two\n\nno newline at the end"
        for size in (1, 5, 12, 100):
            assert "".join(SERVER.runbook_page_bodies(text, size)) == text

    def test_the_lines_and_the_sha_are_util_instructions(self, real_runbooks, capsys):
        text = real_runbooks["implement"]
        assert SERVER.runbook_sha(text, PAGES) == multi_llm._core_sha(text)
        assert PAGES["header"] == multi_llm.CORE_PAGE_HEADER
        assert PAGES["end"] == multi_llm.CORE_PAGE_END
        framed = SERVER.frame_runbook_page(PAGES, "implement", "a line\n", 2, 3, "0" * 12)
        fields, body = split_page(framed)
        assert body == "a line\n" and fields["page"] == "2" and fields["pages"] == "3"
        # util instructions' own page parses with the same two patterns.
        assert multi_llm.main(["util", "instructions", "implement", "--page", "2"]) == 0
        printed = capsys.readouterr().out
        lines = [line for line in printed.split("\n") if not line.startswith(MARKER)]
        assert HEADER.match(lines[0]) and END.match([ln for ln in lines if ln][-1])


class TestThePageRequestIsRecognised:
    @pytest.mark.parametrize("flags, page, size", [
        (["--page", "2"], 2, PAGES["size_default"]),
        (["--page=3", "--page-size=8000"], 3, 8000),
        (["--page-size", "4000", "--page", "1"], 1, 4000),
        (["--resume", "--page", "2", "--yes"], 2, PAGES["size_default"]),
        (["--page", "1", "--page", "4"], 4, PAGES["size_default"]),
    ])
    def test_a_page_request(self, flags, page, size):
        request, problem = SERVER.runbook_page_request(
            dict(IMPLEMENT, extra_flags=flags), SURFACE)
        assert problem is None
        assert (request["row"]["name"], request["page"], request["size"]) == (
            "implement", page, size)

    @pytest.mark.parametrize("flags", [
        ["--page"], ["--page", "0"], ["--page", "x"], ["--page", "²"], ["--page="],
        ["--page", "--force"], ["--page-size", "8000"],
        ["--page", "1", "--page-size", str(PAGES["size_min"] - 1)],
    ])
    def test_a_malformed_page_request(self, flags):
        request, problem = SERVER.runbook_page_request(
            dict(IMPLEMENT, extra_flags=flags), SURFACE)
        assert request is None and problem
        assert PAGES["option"] in problem or PAGES["size_option"] in problem

    @pytest.mark.parametrize("args", [
        IMPLEMENT, dict(IMPLEMENT, extra_flags=["--pages", "2"]),
        dict(IMPLEMENT, extra_flags=["--page2"]), dict(IMPLEMENT, mode="nope",
                                                        extra_flags=["--page", "2"]),
        dict(IMPLEMENT, extra_flags="--page 2"), "not an object",
    ])
    def test_every_other_call(self, args):
        assert SERVER.runbook_page_request(args, SURFACE) == (None, None)

    def test_a_surface_without_pages_has_no_page_option(self):
        surface = json.loads(unpaged_surface_text())
        assert SERVER.runbook_page_request(
            dict(IMPLEMENT, extra_flags=["--page", "2"]), surface) == (None, None)


# ===========================================================================
# The result carries page 1 and closes with the rule
# ===========================================================================

class TestTheResultCarriesPageOne:
    def test_codex_repro_a_middle_cut_cannot_pass_for_the_whole_runbook(
            self, harness, unpaged_root):
        """U79 review, point 1: the whole runbook rode in the result, a cut lost
        its middle and kept `MULTI_LLM_CORE_END: implement`, with no rule."""
        harness.runbook.write_text(padded_runbook(1400), encoding="utf-8")
        sentinel = END_OF_CORE + " implement"
        # The closing *line*, as the result's JSON spells a runbook line (the
        # rule names it too, in backticks, which is no line of the runbook).
        line = json.dumps("\n" + sentinel + "\n")[1:-1]
        _, before, _, _ = blocked_implement(harness, root=unpaged_root)
        before_text = before["content"][0]["text"]
        for tokens in (1200, 10000):
            cut = codex_token_cut(before_text, tokens)
            assert cut != before_text and line in cut, tokens  # the repro
            assert '"runbook_pages"' not in cut

        envelope, result, _, _ = blocked_implement(harness)
        text = result["content"][0]["text"]
        info = envelope["runbook_pages"]
        assert info["page"] == 1 and info["pages"] == 4
        assert list(envelope)[-2:] == ["runbook", "runbook_pages"]
        fields, body = split_page(envelope["runbook"])
        assert sentinel not in body  # page 1 of 4 never carries the closing line
        rule = json.dumps(info["rule"], ensure_ascii=False)
        for tokens in (1200, 10000):
            cut = codex_token_cut(text, tokens)
            assert line not in cut, tokens
            # The tail a cut keeps: page 1's end line, then the rule and the calls.
            assert cut.endswith(json.dumps(info, ensure_ascii=False) + "}")
            assert json.dumps(fields["sha"]) in cut and rule in cut
        # At the default budget the paged result now arrives whole.
        assert codex_token_cut(text, 10000) == text
        # At 1,200 tokens it is cut: page 1's header went with the middle, so
        # the page shows incomplete by its own frame as well as by the notice.
        small = codex_token_cut(text, 1200)
        assert envelope["runbook"].split("\n")[0] not in small
        assert "tokens truncated" in small
        assert json.dumps(info["smaller"]) in small

    def test_the_rule_says_how_to_finish_reading(self, harness):
        harness.runbook.write_text(padded_runbook(1400), encoding="utf-8")
        envelope, _, _, _ = blocked_implement(harness)
        info = envelope["runbook_pages"]
        size, sha = PAGES["size_default"], info["sha"]
        values = {"mode": "implement", "page": "1", "pages": "4", "sha": sha,
                  "size": str(size), "smaller": str(size // 2)}
        again = SERVER.fill(PAGES["again"], values)
        assert info["rule"] == SERVER.fill(PAGES["rule"], dict(values, again=again))
        assert info["page_size"] == size
        assert info["next_page"] == page_call("implement", 2, size)
        assert info["smaller"] == page_call("implement", 1, size // 2)
        flat = " ".join(info["rule"].split())
        for needed in ("page 1 of the implement runbook", "4 pages",
                       '"{}", "<i>"'.format(PAGES["option"]),
                       "{} {}".format(PAGES["size_option"], size // 2),
                       "never starts, resumes or repeats a run", "truncated",
                       END_OF_CORE + " implement"):
            assert needed in flat, needed

    def test_a_runbook_on_one_page_is_framed_and_ruled_too(self, harness):
        """review-commits and apply-commit-fixes fit one page (12.6 and 11.6 KB),
        and a cut of a one-page runbook keeps its closing line: measured at the
        06b budget, so the rule comes with every runbook."""
        harness.runbook.write_text(padded_runbook(300), encoding="utf-8")
        envelope, result, _, _ = blocked_implement(harness)
        info = envelope["runbook_pages"]
        assert (info["page"], info["pages"]) == (1, 1) and "next_page" not in info
        fields, body = split_page(envelope["runbook"])
        assert (fields["page"], fields["pages"]) == ("1", "1")
        assert body.split("\n").count(END_OF_CORE + " implement") == 1
        values = {"mode": "implement", "page": "1", "pages": "1", "sha": info["sha"],
                  "size": str(PAGES["size_default"]),
                  "smaller": str(PAGES["size_default"] // 2)}
        assert info["rule"] == SERVER.fill(PAGES["rule_one"], dict(
            values, again=SERVER.fill(PAGES["again"], values)))
        cut = codex_token_cut(result["content"][0]["text"], 1200)
        line = json.dumps("\n" + END_OF_CORE + " implement\n")[1:-1]
        assert line in cut  # kept, as before paging
        assert envelope["runbook"].split("\n")[0] not in cut  # but the header is not
        assert json.dumps(info["rule"], ensure_ascii=False) in cut

    def test_a_surface_without_pages_keeps_the_runbook_whole(self, harness, unpaged_root):
        """Byte for byte the pre-paging result, and its runbook is exactly the
        paged pages' text less the last newline."""
        old, _, _, _ = harness.run(REVIEW, [launch_step(), {"exit": 3}], root=unpaged_root)
        assert "runbook_pages" not in old and list(old)[-1] == "runbook"
        assert not old["runbook"].startswith(multi_llm.CORE_PAGE_MARKER)
        client = harness.client()
        bodies, _sha = read_pages(client, "review-plan", PAGES["size_default"])
        assert client.close() == 0
        assert "".join(bodies) == old["runbook"] + "\n"

    def test_page_one_of_the_result_is_page_one_of_a_request(self, harness):
        harness.runbook.write_text(padded_runbook(1400), encoding="utf-8")
        envelope, _, _, _ = blocked_implement(harness)
        asked, _, _, _ = harness.run(page_call("implement", 1, PAGES["size_default"]))
        assert asked["runbook"] == envelope["runbook"]
        assert asked["runbook_pages"] == envelope["runbook_pages"]

    def test_a_finished_run_still_gets_no_runbook(self, harness):
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), {
            "exit": 0, "stdout": [result_line(exit=0)]}])
        assert "runbook" not in envelope and "runbook_pages" not in envelope

    def test_a_partial_runbook_still_attaches_nothing(self, harness):
        harness.runbook.write_text(runbook_text(sentinel=False), encoding="utf-8")
        envelope, _, _, _ = harness.run(REVIEW, [launch_step(), {"exit": 3}])
        assert "runbook" not in envelope and "runbook_pages" not in envelope
        assert END_OF_CORE in envelope["runbook_error"]


# ===========================================================================
# A page request runs nothing
# ===========================================================================

class TestAPageRequestRunsNothing:
    def _asked_only_the_runbook(self, harness, mode="implement"):
        assert harness.calls() == []  # no launch, poll, resume or abort
        assert harness.resume_calls() == []
        asked = harness.instruction_calls()
        assert len(asked) == 1
        assert dispatcher_argv(asked[0]) == list(DISPATCHER["instructions"]) + [
            mode, DISPATCHER["host_option"], SURFACE["host"]]

    @pytest.mark.parametrize("args", [
        page_call("implement", 2),
        page_call("implement", 2, 8000),
        dict(IMPLEMENT, extra_flags=["--page=2"]),
        # The run's own call with a page flag added: still only the page.
        dict(IMPLEMENT, extra_flags=["--resume", "--force", "--page", "2"]),
        dict(IMPLEMENT, extra_flags=["--detach", "--page", "2"]),
    ])
    def test_a_page_request_returns_the_page_alone(self, harness, args):
        harness.runbook.write_text(padded_runbook(1400), encoding="utf-8")
        envelope, result, _, _ = harness.run(
            args, [launch_step(), {"exit": CODES["needs_human"]}], meta=WRITE_META)
        self._asked_only_the_runbook(harness)
        assert envelope["ok"] is True and not result["isError"]
        assert envelope["state"] == STATES["ok"] and envelope["exit"] == CODES["ok"]
        fields, _body = split_page(envelope["runbook"])
        assert fields["page"] == "2" and envelope["runbook_pages"]["page"] == 2
        assert envelope["message"] == SERVER.fill(PAGES["served"], {
            "page": "2", "pages": fields["pages"], "mode": "implement"})
        for key in ("run_id", "next", "options", "resume_command", "user_next_step",
                    "result", "output"):
            assert key not in envelope, key

    def test_a_read_only_session_still_reads_a_write_modes_pages(self, harness):
        envelope, _, _, _ = harness.run(
            page_call("implement", 1), meta=turn_meta(sandbox_mode="read-only"))
        self._asked_only_the_runbook(harness)
        assert envelope["ok"] is True and "runbook" in envelope

    def test_a_range_mode_page_needs_no_range(self, harness):
        envelope, _, _, _ = harness.run(page_call("review-commits", 1))
        self._asked_only_the_runbook(harness, "review-commits")
        assert split_page(envelope["runbook"])[0]["mode"] == "review-commits"
        with_range, _, _, _ = harness.run(dict(range_call(), extra_flags=["--page", "1"]))
        assert with_range["runbook"] == envelope["runbook"]

    def test_a_page_past_the_last_runs_nothing(self, harness):
        envelope, result, _, _ = harness.run(page_call("implement", 99))
        self._asked_only_the_runbook(harness)
        assert envelope["ok"] is False and result["isError"]
        assert envelope["state"] == STATES["error"]
        assert "runbook" not in envelope and "runbook_pages" not in envelope
        assert "takes 1 to 1 " in envelope["message"]  # the fixture is one page

    def test_a_failed_fetch_runs_nothing(self, harness):
        harness.runbook.write_text(runbook_text(sentinel=False), encoding="utf-8")
        envelope, _, _, _ = harness.run(page_call("implement", 1))
        self._asked_only_the_runbook(harness)
        assert envelope["ok"] is False and END_OF_CORE in envelope["runbook_error"]
        assert "runbook" not in envelope

    @pytest.mark.parametrize("flags", [
        ["--page"], ["--page", "0"], ["--page-size", "8000"],
        ["--page", "1", "--page-size", "10"]])
    def test_a_malformed_page_request_is_refused_before_any_spawn(self, harness, flags):
        envelope, result, calls, _ = harness.run(dict(IMPLEMENT, extra_flags=flags))
        assert envelope["ok"] is False and result["isError"]
        assert envelope["state"] == STATES["error"] and envelope["exit"] is None
        assert calls == [] and harness.instruction_calls() == []

    def test_an_unknown_mode_is_refused_by_validation(self, harness):
        envelope, _, calls, _ = harness.run(page_call("nope", 1))
        assert envelope["ok"] is False and "unknown mode" in envelope["message"]
        assert calls == [] and harness.instruction_calls() == []

    def test_without_pages_a_page_flag_is_forwarded_as_before(self, harness, unpaged_root):
        args = dict(IMPLEMENT, extra_flags=["--page", "2"])
        harness.run(args, [launch_step(), {"exit": 0, "stdout": [result_line(exit=0)]}],
                    meta=WRITE_META, root=unpaged_root)
        argv = dispatcher_argv(harness.calls()[0])
        assert argv[0] == "implement" and harness.instruction_calls() == []
        assert any(argv[i:i + 2] == ["--page", "2"] for i in range(len(argv)))


# ===========================================================================
# Pages join to the whole runbook, every mode, several sizes
# ===========================================================================

class TestPagesJoinToTheWholeRunbook:
    def test_every_mode_at_the_default_size(self, harness, real_runbooks):
        client = harness.client()
        for mode, text in real_runbooks.items():
            harness.runbook.write_text(text, encoding="utf-8")
            (whole,), whole_sha = read_pages(client, mode, 10 ** 6)
            bodies, sha = read_pages(client, mode, PAGES["size_default"])
            assert "".join(bodies) == whole and sha == whole_sha, mode
            closing = END_OF_CORE + " " + mode
            assert whole.split("\n").count(closing) == 1
            assert sum(body.split("\n").count(closing) for body in bodies) == 1
            for body in bodies:
                assert len(body.encode("utf-8")) <= PAGES["size_default"]
        assert client.close() == 0

    @pytest.mark.parametrize("mode", ["implement", "review-commits"])
    @pytest.mark.parametrize("size", [PAGES["size_min"], 5000])
    def test_small_sizes(self, harness, real_runbooks, mode, size):
        harness.runbook.write_text(real_runbooks[mode], encoding="utf-8")
        client = harness.client()
        (whole,), _ = read_pages(client, mode, 10 ** 6)
        bodies, _ = read_pages(client, mode, size)
        assert client.close() == 0
        assert "".join(bodies) == whole and len(bodies) > 1
        for body in bodies:
            assert len(body.encode("utf-8")) <= size or body.count("\n") == 1

    def test_the_smallest_size_offers_no_smaller(self, harness):
        envelope, _, _, _ = harness.run(page_call("implement", 1, PAGES["size_min"]))
        info = envelope["runbook_pages"]
        assert "smaller" not in info
        values = {"mode": "implement", "page": "1", "pages": str(info["pages"]),
                  "sha": info["sha"], "size": str(PAGES["size_min"]),
                  "smaller": str(PAGES["size_min"])}
        assert SERVER.fill(PAGES["floor"], values) in info["rule"]


# ===========================================================================
# The runbook says where its references are
# ===========================================================================

REFERENCE_NAME = re.compile(r"references/([A-Za-z0-9_.-]+\.md)")


def _bare_server(monkeypatch, stdout: str, *, truncated=False, code=None):
    """A :class:`Server` that never spawns: ``run_child`` hands back *stdout*."""
    server = object.__new__(SERVER.Server)
    server.surface = SURFACE
    seen = {}

    def run_child(call, argv, timeout_s, cancellable=True,
                  stdout_keep=SERVER.STDOUT_TAIL_CHARS):
        seen["stdout_keep"] = stdout_keep
        child = SERVER.ChildRun(argv)
        child.code = CODES["ok"] if code is None else code
        child.stdout, child.truncated = stdout, truncated
        return child

    monkeypatch.setattr(server, "run_child", run_child, raising=False)
    call = SERVER.Call(1, {}, None)
    call.hard_deadline = SERVER.time.monotonic() + 60
    return server, call, seen


def _row(mode):
    return next(row for row in SURFACE["modes"] if row["name"] == mode)


class TestTheRunbookSaysWhereItsReferencesAre:
    """The scrub removes every dispatcher command line, the only lines of a
    served core that name the skill directory, so a runbook that names
    ``references/<file>.md`` opens with the folder they are in."""

    def test_the_surface_carries_the_line(self):
        assert SURFACE["runbook"]["references"] == multi_llm.TOOL_REFERENCES_LINE
        assert "{dir}" in SURFACE["runbook"]["references"]

    @pytest.mark.parametrize("mode", DETACHED_MODES)
    def test_every_reference_a_real_runbook_names_resolves(
            self, monkeypatch, real_runbooks, mode):
        monkeypatch.setattr(SERVER, "SKILL_DIR", SKILL_ROOT)
        server, call, _ = _bare_server(monkeypatch, real_runbooks[mode])
        text, problem, _ = server.fetch_runbook(call, _row(mode))
        assert problem is None
        names = set(REFERENCE_NAME.findall(text))
        assert names, mode  # every detachable core names one (summary-format)
        # No command line survives the scrub to name the skill directory ...
        assert str(SKILL_ROOT) not in text.split("\n", 1)[1]
        # ... so the first line names the folder, and every reference is there.
        folder = SKILL_ROOT / "references"
        assert text.split("\n", 1)[0] == SERVER.fill(
            SURFACE["runbook"]["references"], {"dir": str(folder)})
        assert sorted(name for name in names if not (folder / name).is_file()) == []

    def test_a_runbook_naming_no_reference_gets_no_line(self, monkeypatch):
        server, call, _ = _bare_server(
            monkeypatch, runbook_text().replace("{MODE}", "implement"))
        text, problem, _ = server.fetch_runbook(call, _row("implement"))
        assert problem is None and text.startswith("# Core\n")

    def test_the_wire_result_opens_with_the_plugins_own_folder(self, harness):
        harness.runbook.write_text(runbook_text().replace(
            "WORKFLOW-STEP:", "WORKFLOW-STEP: read `references/own-check.md`, then"),
            encoding="utf-8")
        envelope, _, _, _ = harness.run(page_call("implement", 1))
        _fields, body = split_page(envelope["runbook"])
        folder = harness.root.resolve() / "skills" / "multi-llm" / "references"
        assert body.split("\n")[0] == SERVER.fill(
            SURFACE["runbook"]["references"], {"dir": str(folder)})

    def test_a_surface_without_the_line_attaches_the_runbook_as_before(
            self, monkeypatch, real_runbooks):
        surface = json.loads(surface_mod.cli_surface_json("codex"))
        del surface["runbook"]["references"]
        server, call, _ = _bare_server(monkeypatch, real_runbooks["implement"])
        server.surface = surface
        text, problem, _ = server.fetch_runbook(call, _row("implement"))
        assert problem is None and not text.startswith("Every `references/")


# ===========================================================================
# The runbook is fetched whole, or not at all
# ===========================================================================

class TestTheRunbookIsFetchedWhole:
    """A tail keeps the closing line and loses the head, so the fetch keeps the
    whole core and refuses any stdout that was cut anyway."""

    def test_the_fetch_keeps_more_than_the_tail_every_other_child_keeps(
            self, monkeypatch, real_runbooks):
        server, call, seen = _bare_server(monkeypatch, real_runbooks["implement"])
        server.fetch_runbook(call, _row("implement"))
        assert seen["stdout_keep"] == SERVER.RUNBOOK_STDOUT_CHARS
        assert SERVER.RUNBOOK_STDOUT_CHARS > 8 * SERVER.STDOUT_TAIL_CHARS

    def test_a_cut_runbook_is_refused_though_its_closing_line_arrived(
            self, monkeypatch, real_runbooks):
        stdout = real_runbooks["implement"]
        assert (END_OF_CORE + " implement") in stdout
        server, call, _ = _bare_server(monkeypatch, stdout, truncated=True)
        text, problem, code = server.fetch_runbook(call, _row("implement"))
        assert text is None and code == CODES["ok"]
        assert "cut" in problem and "{:,}".format(SERVER.RUNBOOK_STDOUT_CHARS) in problem

    def test_a_runbook_over_the_old_tail_arrives_with_its_head(self, harness):
        """82 KB of runbook through the real server: every page joins to the
        whole core, its first line included (the 64 KiB tail lost the head and
        kept the closing line, so the cut passed for whole)."""
        text = padded_runbook(2000).replace(
            "# Core\n", "# Core\nFIRST-STEP: the step a tail would lose.\n")
        assert len(text) > 80 * 1024
        harness.runbook.write_text(text, encoding="utf-8")
        client = harness.client()
        (whole,), _ = read_pages(client, "implement", 10 ** 6)
        assert client.close() == 0
        assert whole.startswith("# Core\nFIRST-STEP: the step a tail would lose.\n")
        assert whole.count("WORKFLOW-PADDING") == 2000
        assert whole.split("\n").count(END_OF_CORE + " implement") == 1
