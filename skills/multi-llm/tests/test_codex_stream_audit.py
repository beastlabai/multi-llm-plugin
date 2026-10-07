"""Plan §4.2/§4.5 (option B): the read-only audit of a codex stream, and §6.2's fixtures.

Every positive control replays a stream the real codex-cli 0.156.1 emitted
against the localhost Responses stub (``tests/fixtures/codex/0.156.1/``), so
the item shapes the classifier is written against are the binary's, not ours.
The adversarial table is §4.5's list verbatim; each case also runs wrapped in
``/bin/bash -lc '…'`` exactly as codex renders a command.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils import readonly_audit
from utils.providers import codex_stream_audit as audit_mod
from utils.providers.codex import CodexProvider
from utils.providers.codex_stream_audit import audit_codex_stream, classify_command

FIXTURES = Path(__file__).parent / "fixtures" / "codex" / "0.156.1"


def _lines(name):
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


def _events(name):
    return [json.loads(line) for line in _lines(name) if line.strip()]


TURN_COMPLETED = json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
})


def _stream(*events):
    return "\n".join(e if isinstance(e, str) else json.dumps(e) for e in events) + "\n"


def _route(fixture, *item_ids, complete=True):
    """The fixture's frame (thread/turn events) plus only the named items' events."""
    kept = []
    for event in _events(fixture):
        etype = event["type"]
        if etype.startswith("item."):
            if event["item"]["id"] in item_ids:
                kept.append(event)
        elif etype != "turn.completed":
            kept.append(event)
    if complete:
        kept.append(TURN_COMPLETED)
    return _stream(*kept)


def _cmd_item(item_id, command, *, status="completed", exit_code=0):
    return {"type": "item.completed", "item": {
        "id": item_id, "type": "command_execution", "command": command,
        "aggregated_output": "", "exit_code": exit_code, "status": status}}


def _wrapped(inner):
    return "/bin/bash -lc " + shlex.quote(inner)


# --------------------------------------------------------------------------- #
# §6.2: the fixtures really are 0.156.1 streams the parser still reads
# --------------------------------------------------------------------------- #

class TestTheCapturedFixtures:
    """The recapture codex's parser is now verified against (verified_version)."""

    @pytest.mark.parametrize("name", [
        "read_only_clean.jsonl", "quoting_render.jsonl",
        "writes_patches_sessions.jsonl", "exec_session.jsonl",
        "mcp_tool_call.jsonl", "collab_spawn_agent.jsonl",
        "web_search_hosted_replay.jsonl", "web_search_probe_replay.jsonl",
    ])
    def test_parse_output_reads_the_final_agent_message(self, name):
        result = CodexProvider().parse_output((FIXTURES / name).read_text(), "")
        assert result["success"] is True
        assert result["data"] == []

    def test_parse_usage_reads_the_turn_completed_event(self):
        usage = CodexProvider().parse_usage(
            (FIXTURES / "read_only_clean.jsonl").read_text(), "")
        assert usage["input_tokens"] == 7 and usage["output_tokens"] == 7

    def test_the_verified_version_is_the_newest_recapture(self):
        """Run 6 (#11b): "codex 0.160.1 is newer than the release the shipped
        parser fixtures were captured from (0.160.0)" opened every phase's
        log; the rig re-run on the installed 0.160.1 moves the pin. Run 10
        (#10) logged the same for 0.161.0, and the rig moved it again."""
        newest = max(
            (path.name for path in FIXTURES.parent.iterdir()
             if path.is_dir() and path.name[0].isdigit()),
            key=lambda name: tuple(int(part) for part in name.split(".")),
        )
        assert newest == "0.161.0"
        assert CodexProvider().verified_version == newest

    @pytest.mark.parametrize("version", ["0.160.0", "0.160.1", "0.161.0"])
    @pytest.mark.parametrize("name", [
        "read_only_clean.jsonl", "quoting_render.jsonl", "collab_spawn_agent.jsonl",
    ])
    def test_each_recapture_parses_and_keeps_the_0_156_1_event_shapes(
        self, name, version,
    ):
        """2026-10-02 (0.160.0), 2026-10-06 (0.160.1) and 2026-10-08
        (0.161.0): the offline rig re-run on the installed codex (the
        version-lag WARN's cause was the pin trailing the install). Same event
        kinds, same keys."""
        newer = FIXTURES.parent / version / name
        result = CodexProvider().parse_output(newer.read_text(), "")
        assert result["success"] is True and result["data"] == []

        def shapes(path):
            seen = set()
            for line in path.read_text().splitlines():
                if line.startswith("{"):
                    event = json.loads(line)
                    item = event.get("item") or {}
                    seen.add((event["type"], item.get("type"),
                              tuple(sorted(event)), tuple(sorted(item))))
            return seen

        # 0.156.1 also emitted an `error` item (a deprecation notice) that 0.160.0
        # no longer does; nothing the parser reads is new.
        assert shapes(newer) - shapes(FIXTURES / name) == set()

    def test_command_is_one_wrapped_string_and_shlex_recovers_the_argv(self):
        commands = [e["item"]["command"] for e in _events("quoting_render.jsonl")
                    if e["type"] == "item.completed"
                    and e["item"]["type"] == "command_execution"]
        assert all(isinstance(c, str) for c in commands)
        assert [shlex.split(c)[:2] for c in commands] == [
            ["/bin/bash", "-lc"], ["/bin/bash", "-lc"], ["/bin/bash", "-lc"],
            ["/bin/bash", "-c"], ["/usr/bin/sh", "-lc"],
        ]
        assert shlex.split(commands[0])[2] == "rg -n 'line$' AGENTS.md"

    def test_a_non_zero_exit_is_status_failed(self):
        done = [e["item"] for e in _events("read_only_clean.jsonl")
                if e["type"] == "item.completed" and e["item"]["id"] == "item_4"]
        assert done[0]["exit_code"] == 2 and done[0]["status"] == "failed"

    def test_hosted_types_are_web_search_fields_not_item_types(self):
        """§1 called them item types; the persisted 0c477c4d log says otherwise."""
        items = [e["item"] for e in _events("run_0c477c4d_partial.jsonl")]
        assert {i["type"] for i in items} == {"command_execution", "web_search"}
        started, completed = [i for i in items if i["type"] == "web_search"]
        assert started["action"]["type"] == "other"
        assert completed["action"]["type"] == "find_in_page"
        assert [r["type"] for r in completed["results"]] == ["text_result"]

    def test_the_fixtures_carry_no_capture_machine_identity(self):
        for path in FIXTURES.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                assert "/home/" not in text and "claude-1000" not in text, path


# --------------------------------------------------------------------------- #
# §6.2 live part (2026-09-25): what web_search items really carry
# --------------------------------------------------------------------------- #

WEB_SEARCH_FIXTURES = [
    "web_search_hosted.jsonl",          # live, gpt-5.5: hosted web_search tool
    "web_search_hosted_replay.jsonl",   # the same items replayed from the stub
    "web_search_code_mode.jsonl",       # live, gpt-6-astra: tools.web__run
    "web_search_probe_replay.jsonl",    # stub: unknown action, dropped results
]


def _web_searches(name):
    return [e for e in _events(name)
            if e["type"].startswith("item.") and e["item"]["type"] == "web_search"]


class TestTheLiveWebSearchCapture:
    """The shapes §4.2 step 5 classifies, from the live capture and its replay."""

    def test_every_action_the_capture_names_is_covered(self):
        completed = {e["item"]["action"]["type"]
                     for name in WEB_SEARCH_FIXTURES for e in _web_searches(name)
                     if e["type"] == "item.completed"}
        assert completed == {"search", "open_page", "find_in_page", "other"}
        assert completed == set(audit_mod.WEB_SEARCH_INERT_ACTIONS)

    def test_every_started_web_search_is_the_other_placeholder(self):
        for name in WEB_SEARCH_FIXTURES:
            started = [e["item"] for e in _web_searches(name)
                       if e["type"] == "item.started"]
            assert started and all(i["action"] == {"type": "other"} for i in started)
            assert all(i["query"] == "" for i in started)

    def test_only_the_code_mode_route_carries_text_results(self):
        def result_types(name):
            return {r["type"] for e in _web_searches(name)
                    if e["type"] == "item.completed"
                    for r in e["item"].get("results", [])}
        assert result_types("web_search_code_mode.jsonl") == {"text_result"}
        for name in ("web_search_hosted.jsonl", "web_search_hosted_replay.jsonl",
                     "web_search_probe_replay.jsonl"):
            assert all("results" not in e["item"] for e in _web_searches(name)), name
        assert set(audit_mod.WEB_SEARCH_INERT_RESULT_TYPES) == {"text_result"}

    def test_an_unknown_server_action_surfaces_as_other(self):
        """The stub sent `not_a_known_action` (with a url) and a find_in_page
        with `results`; codex rendered `other` with the fields dropped and
        dropped `results`."""
        done = [e["item"] for e in _web_searches("web_search_probe_replay.jsonl")
                if e["type"] == "item.completed"]
        assert done[0]["id"] == "ws_probe_1"
        assert done[0]["action"] == {"type": "other"} and done[0]["query"] == ""
        assert done[1]["action"]["type"] == "find_in_page"
        assert "results" not in done[1]

    def test_the_stub_replay_matches_the_live_hosted_items(self):
        def items(name):
            return [e["item"] for e in _web_searches(name)]
        assert items("web_search_hosted_replay.jsonl") == items("web_search_hosted.jsonl")

    def test_the_item_id_is_emitted_twice(self):
        """`item_N`, then the call id (`ws_…` hosted, `exec-…` code mode)."""
        for name, prefix in (("web_search_hosted.jsonl", '"id":"ws_'),
                             ("web_search_code_mode.jsonl", '"id":"exec-')):
            for line in _lines(name):
                if '"type":"web_search"' in line:
                    assert line.count('"id":') == 2 and prefix in line, line

    @pytest.mark.parametrize("name", WEB_SEARCH_FIXTURES)
    def test_every_web_search_capture_audits_clean(self, name):
        verdict = audit_codex_stream((FIXTURES / name).read_text())
        assert verdict.clean is True, verdict
        assert verdict.actions == len([e for e in _web_searches(name)
                                       if e["type"] == "item.completed"])

    def test_the_code_mode_cell_emits_no_item_but_its_nested_tools_do(self):
        """The `exec` JS cell (rollout excerpt) called apply_patch and
        exec_command twice; the stream has exactly those three items."""
        cell = [json.loads(line)["payload"] for line in
                _lines("code_mode_writes.rollout.jsonl")]
        assert [c["name"] for c in cell] == ["exec"]
        assert cell[0]["input"].count("await tools.") == 3
        types = [e["item"]["type"] for e in _events("code_mode_writes.jsonl")
                 if e["type"] == "item.completed"]
        assert types == ["agent_message", "file_change", "command_execution",
                         "command_execution", "agent_message"]
        verdict = audit_codex_stream((FIXTURES / "code_mode_writes.jsonl").read_text())
        assert verdict.clean is False and "file_change" in verdict.reason
        assert audit_codex_stream(
            _route("code_mode_writes.jsonl", "item_2")).clean is False  # echo > b.txt
        assert audit_codex_stream(
            _route("code_mode_writes.jsonl", "item_3")).clean is True   # cat

    def test_a_nested_mcp_call_in_a_code_mode_cell_is_an_mcp_tool_call(self):
        cell = [json.loads(line)["payload"] for line in
                _lines("code_mode_mcp.rollout.jsonl")]
        assert [c["name"] for c in cell] == ["exec"]
        done = [e["item"] for e in _events("code_mode_mcp.jsonl")
                if e["type"] == "item.completed" and e["item"]["type"] != "agent_message"]
        assert [(i["type"], i["server"], i["tool"]) for i in done] == [
            ("mcp_tool_call", "capfs", "write_file")]
        verdict = audit_codex_stream((FIXTURES / "code_mode_mcp.jsonl").read_text())
        assert verdict.clean is False and "mcp_tool_call" in verdict.reason

    def test_the_code_mode_web_search_is_a_web_run_extension(self):
        rollout = [json.loads(line)["payload"] for line in
                   _lines("web_search_code_mode.rollout.jsonl")]
        assert rollout[0]["type"] == "custom_tool_call" and rollout[0]["name"] == "exec"
        assert rollout[0]["input"].count("tools.web__run(") == 3
        extensions = [p["item"] for p in rollout[1:]]
        assert {(x["type"], x["kind"]) for x in extensions} == {("Extension", "web.search")}


# --------------------------------------------------------------------------- #
# §4.2 steps 1-4: the fail-closed classifier (§4.5's adversarial table)
# --------------------------------------------------------------------------- #

MUST_FAIL = [
    "sed -n '1w out' f",
    "sed -n '1e id' f",
    "sed -n 's/a/b/w x' f",
    "sed -i 's/a/b/' f",
    "sed '1p' -i f",
    "cat $(echo x)",
    "cat `x`",
    "cat <(x)",
    "rg foo | tee x",
    "rg foo > x",
    "rg foo >> x",
    "find . -fprint x",
    "find . -exec rm {} +",
    "find . -delete",
    "git diff --output=x",
    "git -c core.pager=x log",
    "git -c diff.external=x diff",
    "git diff --ext-diff",
    "git diff --textconv",
    "git diff --no-index a b",
    "git grep -O foo",
    "git diff -U --output=x",
    "git log --pretty --output=x",
    "rg --pre x foo",
    "rg -z foo",
    "cat 'unbalanced",
    "cat trailing\\",
    "rg --bogus-flag foo",
    "grep --bogus-flag foo f",
    "echo x > f",
    "cat > h.txt <<'EOF'\nhello\nEOF",
    "cat x &",
    "FOO=1 cat x",
    "python3 -c 'print(1)'",
    "node -e 1",
    "tee x",
    "touch x",
    "xargs cat",
    "env cat x",
    "git commit -m x",
    "cat x#; touch y",
    "cat 'x'#; touch y",
    "find . $X",
    "git log $X",
    "find . $'\\x2ddelete'",
]

MUST_PASS = [
    "grep -i foo f",
    "rg -e '-i' f",
    "sed -n '1,80p' f",
    "sed -n '/a/,/b/p;3p' f",
    "rg foo | head -5",
    "git log --oneline -5",
    "git --no-pager show HEAD:README.md",
    "git diff -U3 HEAD",
    "git status --short",
    "cat f 2>/dev/null",
    "rg -n 'a|b' f 2>&1 | head -40",
    "find . -name '*.py' -type f -maxdepth 2",
    "nl -ba f | sed -n '2,3p'",
    "ls -la && wc -l f",
]


class TestTheClassifier:

    @pytest.mark.parametrize("inner", MUST_FAIL)
    @pytest.mark.parametrize("wrap", [False, True], ids=["bare", "bash-lc"])
    def test_fails_closed(self, inner, wrap):
        command = _wrapped(inner) if wrap else inner
        assert classify_command(command) is not None, command

    @pytest.mark.parametrize("inner", MUST_PASS)
    @pytest.mark.parametrize("wrap", [False, True], ids=["bare", "bash-lc"])
    def test_read_only_commands_pass(self, inner, wrap):
        command = _wrapped(inner) if wrap else inner
        assert classify_command(command) is None, (command, classify_command(command))

    @pytest.mark.parametrize("inner", ["bash -c 'cat x'", "sh -c 'cat x'",
                                       "python3 -c 'import os'"])
    def test_a_shell_or_interpreter_as_the_inner_command_fails(self, inner):
        assert classify_command(_wrapped(inner)) is not None

    def test_a_doubled_wrapper_fails(self):
        assert classify_command(_wrapped(_wrapped("cat x"))) is not None

    def test_a_wrapper_not_at_the_front_fails(self):
        assert classify_command(_wrapped("cat x; bash -lc 'cat y'")) is not None

    def test_only_the_known_shell_binaries_are_unwrapped(self):
        assert classify_command("/tmp/evil/bash -lc 'cat x'") is not None
        assert classify_command("/usr/bin/sh -lc 'cat x'") is None
        assert classify_command("/bin/bash -c 'cat x'") is None
        assert classify_command("/bin/bash -x 'cat x'") is not None

    def test_the_wire_shape_is_checked_first(self):
        assert classify_command(["cat", "x"]) is None
        assert classify_command(["bash", "-lc", "cat x"]) is None
        assert classify_command(["bash", "-lc", "cat x > y"]) is not None
        assert classify_command(["cat", 1]) is not None
        assert classify_command({"cmd": "cat x"}) is not None
        assert classify_command(None) is not None
        assert classify_command("") is not None

    @pytest.mark.parametrize("inner", [
        "pwd",
        "pwd -L",
        "pwd -P",
        # The command a real read-only code review opened with, marked not
        # clean for `pwd` alone (review-run-scope-salvage-and-failure-visibility
        # §7.3, T7.6).
        "pwd && rg --files -g AGENTS.md -g '*opencode*' -g '*validator*' "
        "-g '*test*' -g 'package.json' -g 'pyproject.toml' -g 'README*' "
        "-g '*rate*' -g '*denied*'",
    ])
    @pytest.mark.parametrize("wrap", [False, True], ids=["bare", "bash-lc"])
    def test_pwd_is_accepted_with_either_flag(self, inner, wrap):
        """`pwd`'s only flags choose how the directory is spelled; none writes."""
        command = _wrapped(inner) if wrap else inner
        assert classify_command(command) is None, (command, classify_command(command))

    @pytest.mark.parametrize("inner", [
        # Exactly what the same run's codex slot ran, and was refused for.
        'for p in /AGENTS.md ~ ~/AGENTS.md ~/code/AGENTS.md; do '
        'if [ -f "$p" ]; then cat "$p"; fi; done',
        # The `&&` spelling of the same loop.
        'for p in /AGENTS.md ~/AGENTS.md; do [ -f "$p" ] && cat "$p"; done',
    ])
    @pytest.mark.parametrize("wrap", [False, True], ids=["bare", "bash-lc"])
    def test_a_for_loop_over_a_shell_variable_stays_rejected(self, inner, wrap):
        """T7.7, decision 11.15: a loop reading `$p` is the shell
        programmability the fail-closed classifier exists to refuse, and every
        cell of it only reads. Pinned so accepting `pwd` never grows into
        accepting this; no prompt guidance steers codex away from it."""
        command = _wrapped(inner) if wrap else inner
        assert classify_command(command) is not None, command

    def test_every_0c477c4d_command_passes(self):
        """The acceptance stream's own commands are all read-only."""
        commands = [e["item"]["command"] for e in _events("run_0c477c4d_partial.jsonl")
                    if e["item"]["type"] == "command_execution"]
        assert commands
        for command in commands:
            assert classify_command(command) is None, (command, classify_command(command))


# --------------------------------------------------------------------------- #
# §4.2 step 5: the stream walk
# --------------------------------------------------------------------------- #

class TestTheStreamAudit:

    def test_the_captured_read_only_stream_is_clean(self):
        verdict = audit_codex_stream((FIXTURES / "read_only_clean.jsonl").read_text())
        assert verdict.clean is True, verdict
        assert verdict.actions == 6

    def test_a_truncated_stream_fails(self):
        text = "\n".join(_lines("read_only_clean.jsonl")[:-1])  # no turn.completed
        verdict = audit_codex_stream(text)
        assert verdict.clean is False
        assert "turn.completed" in verdict.reason

    def test_an_empty_stream_fails(self):
        assert audit_codex_stream("").clean is False

    def test_a_sed_i_command_fails(self):
        verdict = audit_codex_stream(_stream(
            _cmd_item("item_0", _wrapped("sed -i 's/a/b/' f")), TURN_COMPLETED))
        assert verdict.clean is False
        assert "item_0" in verdict.reason

    def test_an_unknown_item_type_fails(self):
        verdict = audit_codex_stream(_stream(
            {"type": "item.completed", "item": {"id": "i", "type": "brand_new_tool"}},
            TURN_COMPLETED))
        assert verdict.clean is False

    def test_todo_list_fails_until_captured(self):
        verdict = audit_codex_stream(_stream(
            {"type": "item.completed", "item": {"id": "i", "type": "todo_list",
                                                "items": []}},
            TURN_COMPLETED))
        assert verdict.clean is False

    def test_an_unknown_top_level_event_fails(self):
        assert audit_codex_stream(_stream({"type": "mystery"}, TURN_COMPLETED)).clean is False

    def test_a_non_json_line_fails(self):
        assert audit_codex_stream("not json\n" + TURN_COMPLETED).clean is False

    def test_turn_failed_fails(self):
        assert audit_codex_stream(_stream(
            {"type": "turn.failed", "error": {"message": "x"}}, TURN_COMPLETED,
        )).clean is False

    def test_an_item_started_with_no_completion_fails(self):
        started = {"type": "item.started", "item": {
            "id": "item_9", "type": "command_execution", "command": _wrapped("cat x"),
            "aggregated_output": "", "exit_code": None, "status": "in_progress"}}
        verdict = audit_codex_stream(_stream(started, TURN_COMPLETED))
        assert verdict.clean is False
        assert "unfinished" in verdict.reason

    @pytest.mark.parametrize("item", [
        {"id": "i", "type": "command_execution"},
        {"id": "i", "type": "command_execution", "command": 5},
        {"id": "i", "type": "command_execution", "command": "cat x", "status": 3},
        {"id": "i", "type": "mcp_tool_call", "tool": "read"},
        {"id": "i", "type": "web_search"},
        {"id": "i", "type": "web_search", "action": {"type": "find_in_page"},
         "results": "nope"},
        {"id": "i", "type": "web_search", "action": "search"},
        {"type": "command_execution", "command": "cat x"},
    ])
    def test_a_known_tool_item_with_missing_fields_fails(self, item):
        verdict = audit_codex_stream(_stream(
            {"type": "item.completed", "item": item}, TURN_COMPLETED))
        assert verdict.clean is False, item

    @pytest.mark.parametrize("action,results", [
        ({"type": "run_command", "url": "https://example.org"}, None),  # new variant
        ({"type": "find_in_page", "pattern": "p"},
         [{"type": "image_result", "url": "https://example.org"}]),     # new result
        ({"type": "search", "query": "q"}, [{"url": "https://example.org"}]),
        ({"type": "search", "query": "q"}, ["text_result"]),
        ({"type": "search", "query": "q"}, {"type": "text_result"}),
    ])
    def test_an_unlisted_web_search_action_or_result_fails_closed(self, action,
                                                                  results):
        """§4.2 step 5: only the captured values are inert."""
        item = {"id": "w", "type": "web_search", "query": "q", "action": action}
        if results is not None:
            item["results"] = results
        verdict = audit_codex_stream(_stream(
            {"type": "item.completed", "item": item}, TURN_COMPLETED))
        assert verdict.clean is False, item

    def test_a_find_in_page_with_no_results_is_inert(self):
        item = {"type": "item.completed", "item": {
            "id": "w", "type": "web_search", "query": "q",
            "action": {"type": "find_in_page", "pattern": "p"}, "results": []}}
        assert audit_codex_stream(_stream(item, TURN_COMPLETED)).clean is True

    def test_the_verdict_carries_the_redacted_command_but_never_its_output(self):
        # Built at runtime so the source never holds a key-shaped literal
        # (the committed-secrets scan would flag it).
        secret = "sk-" + "live-" + "0123456789" + "abcdef"
        output_marker = "OUTPUT-THAT-MUST-NOT-LEAK"
        event = _cmd_item("item_0", _wrapped(f"echo {secret} > f"))
        event["item"]["aggregated_output"] = output_marker
        verdict = audit_codex_stream(_stream(event, TURN_COMPLETED))
        assert verdict.clean is False
        dumped = json.dumps(verdict.to_dict())
        assert secret not in dumped
        assert output_marker not in dumped
        assert "ran: echo <redacted:sk-family> > f" in verdict.reason, verdict.reason

    def test_the_rejection_names_what_codex_ran_without_the_wrapper(self):
        verdict = audit_codex_stream(_stream(
            _cmd_item("item_18", _wrapped("docker ps -a")), TURN_COMPLETED))
        assert verdict.clean is False
        assert verdict.reason == (
            "item_18: command_execution: verb 'docker' is not on the read-only "
            "allow-list; ran: docker ps -a"), verdict.reason
        assert "/bin/bash" not in verdict.reason and "-lc" not in verdict.reason

    def test_an_argv_command_is_rendered_without_the_wrapper(self):
        verdict = audit_codex_stream(_stream(
            _cmd_item("item_0", ["/bin/bash", "-lc", "touch x"]), TURN_COMPLETED))
        assert verdict.reason.endswith("; ran: touch x"), verdict.reason

    def test_the_rendered_command_is_bounded(self):
        long_path = "d/" * 400 + "f"
        verdict = audit_codex_stream(_stream(
            _cmd_item("item_0", _wrapped(f"touch {long_path}")), TURN_COMPLETED))
        rendered = verdict.reason.split("; ran: ", 1)[1]
        assert rendered.endswith("\u2026")
        assert len(rendered) == audit_mod.MAX_RENDERED_COMMAND_CHARS + 1
        assert len(verdict.reason) < 400

    def test_the_rendered_command_masks_home_paths_and_newlines(self):
        verdict = audit_codex_stream(_stream(
            _cmd_item("item_0", _wrapped("cat <<EOF > /home/someone/x\nhi\nEOF")),
            TURN_COMPLETED))
        assert verdict.clean is False
        assert "/home/someone" not in verdict.reason
        assert "\n" not in verdict.reason
        assert "ran: cat <<EOF > ~/x\\nhi\\nEOF" in verdict.reason, verdict.reason

    def test_a_clean_stream_longer_than_the_log_cap_passes_on_the_full_stream(self):
        """§4.2: the persisted/excerpted copy is cut at 46,080 B and would lose
        `turn.completed`; the audit reads the whole in-memory stream."""
        big = "x" * 4000
        events = []
        for n in range(20):
            item = _cmd_item(f"item_{n}", _wrapped(f"sed -n '{n + 1},{n + 80}p' plan.md"))
            item["item"]["aggregated_output"] = big
            events.append(item)
        text = _stream({"type": "thread.started", "thread_id": "t"},
                       {"type": "turn.started"}, *events, TURN_COMPLETED)
        assert len(text.encode()) > 46080
        assert audit_codex_stream(text).clean is True
        from utils.llm_client import _bound_stream
        assert audit_codex_stream(_bound_stream(text)).clean is False


# --------------------------------------------------------------------------- #
# §4.5 positive controls: every write route the capture drove fails the audit
# --------------------------------------------------------------------------- #

class TestEveryWriteRouteFails:
    """One route per case, each replayed from the real 0.156.1 capture."""

    @pytest.mark.parametrize("fixture,item_id,what", [
        ("writes_patches_sessions.jsonl", "item_3", "shell operator '>'"),
        ("writes_patches_sessions.jsonl", "item_4", "newline"),
        ("writes_patches_sessions.jsonl", "item_5", "python3"),
        ("writes_patches_sessions.jsonl", "item_6", "file_change"),
        ("writes_patches_sessions.jsonl", "item_7", "file_change"),
        ("exec_session.jsonl", "item_1", "'bash'"),
        ("mcp_tool_call.jsonl", "item_1", "mcp_tool_call capfs/write_file"),
        ("collab_spawn_agent.jsonl", "item_1", "collab_tool_call"),
    ], ids=["echo-redirect", "heredoc", "python-c", "apply-patch-tool",
            "apply-patch-via-shell", "exec-session-write-stdin", "mcp-tool",
            "spawn-agent"])
    def test_the_route_fails(self, fixture, item_id, what):
        verdict = audit_codex_stream(_route(fixture, item_id))
        assert verdict.clean is False
        assert item_id in verdict.reason and what in verdict.reason, verdict.reason

    def test_an_exec_session_left_open_at_turn_end_fails(self):
        # item_8: `sleep 3; echo done > s.txt`, still running when the turn ended.
        verdict = audit_codex_stream(_route("writes_patches_sessions.jsonl", "item_8"))
        assert verdict.clean is False
        assert "unfinished" in verdict.reason

    def test_the_read_items_of_the_writing_capture_are_clean_alone(self):
        """The failures above are the write items, not the frame around them."""
        verdict = audit_codex_stream(
            _route("writes_patches_sessions.jsonl", "item_0", "item_1", "item_2"))
        assert verdict.clean is True, verdict

    def test_an_mcp_tool_on_the_read_only_list_passes(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "READONLY_MCP_TOOLS",
                            frozenset({("capfs", "write_file")}))
        assert audit_codex_stream(_route("mcp_tool_call.jsonl", "item_1")).clean is True

    def test_the_whole_writing_captures_fail(self):
        for name in ("writes_patches_sessions.jsonl", "exec_session.jsonl",
                     "mcp_tool_call.jsonl", "collab_spawn_agent.jsonl",
                     "quoting_render.jsonl"):
            assert audit_codex_stream((FIXTURES / name).read_text()).clean is False, name


# --------------------------------------------------------------------------- #
# Acceptance (§4.2): run 0c477c4d's stream
# --------------------------------------------------------------------------- #

def _acceptance_stream():
    """The pasted 0c477c4d items, plus what the log's 46,080 B cut removed.

    Supplied ONLY here (plan §4.2 acceptance): item_31's completion — its
    started line is intact and its completed line is cut mid-output in the log,
    command identical — and the missing `turn.completed`.
    """
    lines = _lines("run_0c477c4d_partial.jsonl")
    last = json.loads(lines[-1])
    assert last["type"] == "item.started" and last["item"]["id"] == "item_31"
    done = dict(last["item"], status="completed", exit_code=0,
                aggregated_output="[synthetic: cut from the persisted log]")
    return "\n".join(lines + [json.dumps({"type": "item.completed", "item": done}),
                              TURN_COMPLETED]) + "\n"


class TestTheAcceptanceStream:

    def test_it_audits_clean(self):
        """The acceptance criterion: every command is read-only and the one
        web_search (find_in_page, a `text_result`) is inert per the live
        capture."""
        verdict = audit_codex_stream(_acceptance_stream())
        assert verdict.clean is True, verdict
        assert verdict.actions == 5

    def test_it_fails_again_if_text_result_is_unlisted(self, monkeypatch):
        """The classification is what passes it — nothing looser."""
        monkeypatch.setattr(audit_mod, "WEB_SEARCH_INERT_RESULT_TYPES", frozenset())
        verdict = audit_codex_stream(_acceptance_stream())
        assert verdict.clean is False
        assert "text_result" in verdict.reason


# --------------------------------------------------------------------------- #
# §4.3: the AND across attempts, and the switch
# --------------------------------------------------------------------------- #

CLEAN = {"clean": True, "reason": "", "actions": 3}
DIRTY = {"clean": False, "reason": "item_3: file_change", "actions": 1}


class TestCombinationAndSwitch:

    def test_the_switch_is_on_by_default(self):
        assert readonly_audit.STREAM_AUDIT_EXEMPTION_ENABLED is True
        assert readonly_audit.exemption_applies([CLEAN]) is True

    def test_a_clean_attempt_changes_nothing_when_switched_off(self, monkeypatch):
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", False)
        assert readonly_audit.exemption_applies([CLEAN]) is False

    def test_a_single_clean_attempt_clears_when_switched_on(self, monkeypatch):
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", True)
        assert readonly_audit.exemption_applies([CLEAN]) is True

    @pytest.mark.parametrize("attempts", [
        [CLEAN, DIRTY],          # second attempt wrote, first clean
        [DIRTY, CLEAN],          # a writing attempt, then a clean retry
        [{"clean": False, "reason": "no turn.completed"}, CLEAN],  # incomplete
        [None, CLEAN],           # an attempt with no verdict
        [CLEAN, None],
        [],                      # no attempt at all
        [{"clean": "yes"}],      # not exactly True
    ])
    def test_any_non_clean_attempt_blocks(self, attempts, monkeypatch):
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", True)
        assert readonly_audit.exemption_applies(attempts) is False
        assert readonly_audit.combine_audits(attempts).clean is False

    def test_the_combined_reason_names_the_attempt(self):
        assert readonly_audit.combine_audits([CLEAN, DIRTY]).reason.startswith("attempt 2:")


# --------------------------------------------------------------------------- #
# §4.2 wiring: llm_client audits the full stdout and carries only the verdict
# --------------------------------------------------------------------------- #

class TestLlmClientCarriesTheVerdict:

    @staticmethod
    def _run(stdout, returncode=0, tmp_path=None, rung="unsandboxed"):
        from utils.llm_client import invoke_with_provider

        completed = MagicMock()
        completed.returncode = returncode
        completed.stdout = stdout
        completed.stderr = ""
        log = (tmp_path / "log.txt") if tmp_path is not None else None
        with patch("utils.llm_client.subprocess.run", return_value=completed), \
                patch("shutil.which", return_value="/usr/bin/codex"):
            result = invoke_with_provider("p", "codex:gpt-5.5", timeout=60,
                                          log_file=log, sandbox_fallback=rung)
        return result, log

    def _big_clean_stream(self):
        events = []
        for n in range(20):
            item = _cmd_item(f"item_{n}", _wrapped(f"sed -n '{n + 1},{n + 9}p' plan.md"))
            item["item"]["aggregated_output"] = "y" * 4000
            events.append(item)
        events.append({"type": "item.completed", "item": {
            "id": "item_99", "type": "agent_message", "text": "[]"}})
        return _stream({"type": "thread.started", "thread_id": "t"},
                       {"type": "turn.started"}, *events, TURN_COMPLETED)

    def test_a_long_clean_stream_is_audited_whole_and_only_the_verdict_leaves(
        self, tmp_path
    ):
        stdout = self._big_clean_stream()
        assert len(stdout.encode()) > 46080
        result, log = self._run(stdout, tmp_path=tmp_path)
        assert result["success"] is True
        assert result["details"]["readonly_audit"] == {
            "clean": True, "reason": "", "actions": 20}
        # Only the compact verdict — no raw stream key on the envelope.
        assert set(result["details"]["readonly_audit"]) == {"clean", "reason", "actions"}
        assert "aggregated_output" not in json.dumps(result["details"])
        text = log.read_text(encoding="utf-8")
        assert "Readonly audit: clean (20 action(s))" in text
        assert "[stream truncated at" in text  # the log copy is bounded

    def test_a_failed_attempt_carries_its_verdict_too(self):
        stdout = _stream(_cmd_item("item_0", _wrapped("echo x > f")))
        result, _ = self._run(stdout, returncode=1)
        assert result["success"] is False
        assert result["details"]["readonly_audit"]["clean"] is False

    def test_a_default_or_contained_rung_run_carries_no_verdict(self):
        """Only the uncontained rung is bracketed by the porcelain check."""
        for rung in (None, "landlock"):
            result, _ = self._run(self._big_clean_stream(), rung=rung)
            assert "readonly_audit" not in result["details"], rung

    def test_an_adapter_with_no_structured_stream_carries_none(self):
        from utils.llm_client import invoke_with_provider

        completed = MagicMock(returncode=0, stdout='{"type":"result","result":"[]"}',
                              stderr="")
        with patch("utils.llm_client.subprocess.run", return_value=completed), \
                patch("shutil.which", return_value="/usr/bin/cursor-agent"):
            result = invoke_with_provider("p", "cursor-agent:auto", timeout=60,
                                          sandbox_fallback="unsandboxed")
        assert "readonly_audit" not in result.get("details", {})

    def test_a_raising_hook_is_a_missing_verdict(self, monkeypatch):
        def boom(self, raw):
            raise RuntimeError("x")

        monkeypatch.setattr(CodexProvider, "audit_readonly_stream", boom)
        result, _ = self._run(self._big_clean_stream())
        assert "readonly_audit" not in result["details"]


# --------------------------------------------------------------------------- #
# §4.3 call sites: a foreign change beside a clean stream
# --------------------------------------------------------------------------- #

from tests.test_subagent_executor import (  # noqa: E402
    TestTheReviewFanoutGatesOnTheSameProbe as _Fanout,
)


def _audit_of(name_or_stream):
    text = (FIXTURES / name_or_stream).read_text() if name_or_stream.endswith(
        ".jsonl") else name_or_stream
    return audit_codex_stream(text).to_dict()


class TestTheReviewPlanCallSite:
    """review-plan's file-output lane: the retry ladder's attempts are ANDed."""

    @pytest.fixture
    def slot(self, tmp_path, monkeypatch):
        from utils import review_orchestrator_base as rob

        _Fanout._git_repo(tmp_path)
        monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
        monkeypatch.setattr(rob, "get_provider", lambda _n: _Fanout._AutoFallback())
        monkeypatch.setattr(rob, "get_project_root", lambda _p: str(tmp_path))
        monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
        return rob

    @staticmethod
    def _attempts(tmp_path, audits, *, foreign=True):
        """One dispatch per entry: all but the last time out (retryable)."""
        calls = []

        def dispatch(**kwargs):
            n = len(calls)
            calls.append(kwargs)
            details = {"sandbox": "bypassed"}
            if audits[n] is not None:
                details["readonly_audit"] = audits[n]
            if n < len(audits) - 1:
                return {"success": False, "error": "codex timed out after 5s",
                        "error_code": "TIMEOUT", "details": details,
                        "output_file": str(tmp_path / "out.json")}
            if foreign:
                # "Another session" edits the tree during the window.
                (tmp_path / "tracked.txt").write_text("someone else\n",
                                                      encoding="utf-8")
            return {"success": True, "data": [{"title": "x"}],
                    "source": "stdout_fallback",
                    "output_file": str(tmp_path / "out.json"), "details": details}

        return dispatch, calls

    def test_switch_off_a_clean_stream_still_fails_closed(self, slot, tmp_path,
                                                          monkeypatch):
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", False)
        dispatch, _ = self._attempts(tmp_path, [_audit_of("read_only_clean.jsonl")])
        monkeypatch.setattr(slot, "invoke_with_file_output", dispatch)
        result = _Fanout._run_slot(slot, tmp_path, "fakebwrap:m1")
        assert result["success"] is False
        assert "or any other process writing to this repository" in result["error"]

    def test_switch_on_a_clean_stream_passes_with_a_not_attributed_warning(
        self, slot, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", True)
        dispatch, _ = self._attempts(tmp_path, [_audit_of("read_only_clean.jsonl")])
        monkeypatch.setattr(slot, "invoke_with_file_output", dispatch)
        result = _Fanout._run_slot(slot, tmp_path, "fakebwrap:m1")
        assert result["success"] is True, result["error"]
        out = capsys.readouterr().out
        assert "[SANDBOX_BYPASSED] fakebwrap" in out
        assert "not attributed" in out and "(tracked.txt)" in out

    @pytest.mark.parametrize("fixture", [
        "writes_patches_sessions.jsonl", "mcp_tool_call.jsonl",
    ])
    def test_switch_on_a_writing_stream_fails(self, slot, tmp_path, monkeypatch,
                                              fixture):
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", True)
        dispatch, _ = self._attempts(tmp_path, [_audit_of(fixture)])
        monkeypatch.setattr(slot, "invoke_with_file_output", dispatch)
        assert _Fanout._run_slot(slot, tmp_path, "fakebwrap:m1")["success"] is False

    @pytest.mark.parametrize("first", [
        {"clean": False, "reason": "item_3: file_change", "actions": 1},
        {"clean": False, "reason": "no turn.completed", "actions": 2},
        None,
    ], ids=["writing-attempt", "incomplete-attempt", "no-verdict"])
    def test_a_clean_retry_never_excuses_an_earlier_attempt(
        self, slot, tmp_path, monkeypatch, first
    ):
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", True)
        dispatch, calls = self._attempts(
            tmp_path, [first, _audit_of("read_only_clean.jsonl")])
        monkeypatch.setattr(slot, "invoke_with_file_output", dispatch)
        result = _Fanout._run_slot(slot, tmp_path, "fakebwrap:m1")
        assert len(calls) == 2, "the ladder did not retry"
        assert result["success"] is False
        assert "fails closed" in result["error"]

    def test_the_tracker_records_the_verdict_and_the_warning_not_the_stream(
        self, slot, tmp_path, monkeypatch
    ):
        import asyncio

        from utils.progress_tracker import FanoutProgress

        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", True)
        dispatch, _ = self._attempts(tmp_path, [_audit_of("read_only_clean.jsonl")])
        monkeypatch.setattr(slot, "invoke_with_file_output", dispatch)
        phase = tmp_path / "phase"
        phase.mkdir()
        tracker = FanoutProgress(phase, ["fakebwrap:m1"], interval_s=0, timeout=60)
        tracker.seed()
        plan = tmp_path / "plan.md"
        plan.write_text("plan", encoding="utf-8")
        asyncio.run(slot.run_single_model(
            model_spec="fakebwrap:m1", plan_path=str(plan),
            prompt_template="{prefix}{output_json_path}", prompt_context={},
            phase_name="review-plan", invoke_phase="plan_review", timeout=1800,
            out_dir=str(tmp_path), prefix="plan", tracker=tracker,
        ))
        status = json.loads((phase / ".status.json").read_text(encoding="utf-8"))
        entry = status["progress"]["models"]["fakebwrap:m1"]
        assert entry["readonly_audit"] == {"clean": True, "reason": "", "actions": 6}
        assert "not attributed" in entry["warning"]
        assert "aggregated_output" not in json.dumps(status)
