"""The commit modes at the dispatcher (plans/review-commits.md §2.1, §2.2, T8).

``review-commits``, ``apply-commit-fixes`` and ``improve-commits`` take an
optional range (``modes.POS_REVS``) where every other mode takes a plan, and
the dispatcher is where the words become the one flag a front door parses:

* **The leading run.** Every leading bare word is the range, and on these modes
  a ``-N`` word (a hyphen and ASCII digits, nothing else) continues the run
  instead of ending it. Every other mode splits as it always has.
* **The join.** The words become one ``--revs "<joined>"`` value, ahead of the
  extra-positional check: ``last 2 commits`` needs no quotes, and the shims'
  quoted slot (one word, spaces inside) is the same value. Zero words emit
  nothing (``HEAD``). One word that is a commit brief's path becomes
  ``--plan-file``; a plan, a README or a missing file stays range text.
* **The flags the front doors own.** A typed ``--plan-file``, ``--base-ref`` or
  ``--head-ref`` (an abbreviation included) is exit 3
  ``[COMMIT_FLAG_CONFLICT]``, checked on the tokens as typed, so a root range's
  brief path still passes; a detached worker's argv is skipped.
* **``status`` takes the pinned spelling.** Its row stays ``POS_PLAN``;
  ``run_status.main`` maps a pinned ``<base12>..<head12>`` to its brief by name
  and refuses other range text.

``utils/commit_review.py`` and ``utils/commit_range.py`` are other tasks'
modules; the three functions the dispatcher and ``status`` call are faked here
(``is_commit_brief``, ``brief_for_pinned``, ``match_phrase``), so these rows
pin the dispatcher's own decisions whether or not the real modules have
landed. Every dispatch is stubbed at ``multi_llm._load_target``: nothing runs a
real phase.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, run_status  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402

COMMIT_MODES = ("review-commits", "apply-commit-fixes", "improve-commits")
MARKER = "<!-- multi-llm:commit-brief v1 -->"
PINNED = "a1b2c3d4e5f6..e4f5a6b7c8d9"
BRIEF_REL = "plans/commit-reviews/a1b2c3d4e5f6-e4f5a6b7c8d9.md"
#: The phrases the fake matcher knows: enough to stand for the table's forms.
PHRASES = ("today", "yesterday", "branch", "this branch", "last 2 commits",
           "last two days", "-2")


# ---------------------------------------------------------------------------
# Fakes for the two modules other tasks own
# ---------------------------------------------------------------------------

def _fake_commit_review(calls):
    module = types.ModuleType("utils.commit_review")
    module.BRIEF_MARKER = MARKER

    def is_commit_brief(path, *, cwd=None):
        calls.append((str(path), cwd))
        try:
            with open(path, encoding="utf-8") as stream:
                return stream.readline().rstrip("\r\n") == MARKER
        except OSError:
            return False

    def brief_for_pinned(text, *, cwd=None):
        if not re.fullmatch(r"[0-9a-f]{12}\.\.[0-9a-f]{12}", text):
            return None
        base, head = text.split("..")
        return Path(cwd or os.getcwd()) / "plans" / "commit-reviews" / f"{base}-{head}.md"

    module.is_commit_brief = is_commit_brief
    module.brief_for_pinned = brief_for_pinned
    return module


def _fake_commit_range():
    module = types.ModuleType("utils.commit_range")

    def match_phrase(text):
        lowered = " ".join(text.lower().split())
        return lowered if lowered in PHRASES else None

    module.match_phrase = match_phrase
    return module


@pytest.fixture
def fakes(monkeypatch):
    """The fake modules in ``sys.modules`` and on the ``utils`` package (both,
    so ``from utils import commit_review`` and ``from . import commit_review``
    find them whether or not the real one was imported first). Yields the
    ``is_commit_brief`` call log."""
    import utils

    calls = []
    for name, module in (("commit_review", _fake_commit_review(calls)),
                         ("commit_range", _fake_commit_range())):
        monkeypatch.setitem(sys.modules, f"utils.{name}", module)
        monkeypatch.setattr(utils, name, module, raising=False)
    return calls


@pytest.fixture
def here(tmp_path, monkeypatch):
    root = Path(os.path.realpath(tmp_path))
    monkeypatch.chdir(root)
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    return root


@pytest.fixture
def seen(monkeypatch):
    """Every dispatch target records its module, attr and argv, and runs nothing."""
    calls = []

    def loader(module_path, attr):
        def target(argv):
            calls.append({"module": module_path, "attr": attr, "argv": list(argv)})
            return 0

        return target

    monkeypatch.setattr(multi_llm, "_load_target", loader)
    return calls


def _run(argv, capsys):
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _brief(root: Path, rel: str = BRIEF_REL, marker: bool = True) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "# Commit review a1b2c3d..e4f5a6b\n"
    path.write_text((MARKER + "\n" + body) if marker else body, encoding="utf-8")
    return path


def _mode(name):
    return mode_table.resolve(name)


HOST = ["--host", "claude-code"]


# ---------------------------------------------------------------------------
# 1. The leading run: `-N` is a range word there, on the commit modes only
# ---------------------------------------------------------------------------

class TestTheLeadingRun:
    @pytest.mark.parametrize("mode", COMMIT_MODES)
    def test_a_count_word_continues_the_run(self, mode):
        assert multi_llm._split_positionals(["-2"], _mode(mode)) == (["-2"], [])
        assert multi_llm._split_positionals(
            ["last", "-12", "--quick", "-3"], _mode(mode)
        ) == (["last", "-12"], ["--quick", "-3"])

    @pytest.mark.parametrize("token", ["-2h", "--2", "-2.5", "-", "-h", "-x2", "-٣"])
    def test_anything_else_with_a_hyphen_ends_the_run(self, token):
        """`-٣` is an Arabic-Indic digit: ASCII digits only."""
        assert multi_llm._split_positionals(
            [token], _mode("review-commits")
        ) == ([], [token])

    @pytest.mark.parametrize("mode", [None, "review-code", "status", "review-plan"])
    def test_every_other_mode_splits_as_today(self, mode):
        row = _mode(mode) if mode else None
        assert multi_llm._split_positionals(["-2"], row) == ([], ["-2"])
        assert multi_llm._split_positionals(["p.md", "-2"], row) == (["p.md"], ["-2"])

    @pytest.mark.parametrize("mode", COMMIT_MODES)
    def test_minus_n_alone_is_the_range(self, mode, here, seen, capsys):
        code, _out, err = _run([mode, "-2", *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--revs", "-2", *HOST]

    def test_minus_2h_is_forwarded_as_typed(self, here, seen, capsys):
        code, _out, err = _run(["review-commits", "-2h", *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["-2h", *HOST]

    def test_a_count_after_a_flag_is_never_a_range(self, here, seen, capsys):
        """The front door refuses it ([RANGE_AFTER_FLAGS]); the dispatcher
        forwards it verbatim and makes no range of it."""
        code, _out, err = _run(["review-commits", "--quick", "-2", *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--quick", "-2", *HOST]
        assert "--revs" not in seen[0]["argv"]

    def test_review_code_minus_2_is_split_as_today(self, here, seen, capsys):
        code, _out, err = _run(["review-code", "plans/x.md", "-2", *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--plan-file", "plans/x.md", "-2", *HOST]


# ---------------------------------------------------------------------------
# 2. The join: one --revs value, never the extra-positional error
# ---------------------------------------------------------------------------

class TestTheJoin:
    @pytest.mark.parametrize("mode", COMMIT_MODES)
    def test_last_2_commits_is_one_value(self, mode, here, seen, capsys):
        code, _out, err = _run([mode, "last", "2", "commits", *HOST], capsys)
        assert code == 0, err
        assert "positional" not in err
        assert seen[0]["argv"] == ["--revs", "last 2 commits", *HOST]

    @pytest.mark.parametrize("mode", COMMIT_MODES)
    def test_the_shims_quoted_slot_is_the_same_value(self, mode, here, seen, capsys):
        """The codex and opencode shims put the range in the quoted
        ``"<plan-path>"`` slot, so a model following them sends ``last 2
        commits`` as one argv word: it is the leading run's one word and the
        same ``--revs`` the three bare words make, never a plan."""
        code, _out, err = _run([mode, "last 2 commits", *HOST], capsys)
        assert code == 0, err
        assert _run([mode, "last", "2", "commits", *HOST], capsys)[0] == 0
        assert seen[0]["argv"] == seen[1]["argv"] == ["--revs", "last 2 commits", *HOST]
        assert multi_llm._normalize_positionals(
            _mode(mode), ["last 2 commits"], []
        ) == ["--revs", "last 2 commits"]

    @pytest.mark.parametrize("words", [
        ["HEAD~3", "HEAD"], ["A..B", "C..D"], ["-2", "main..HEAD"],
        ["main..HEAD", "-2"], ["last", "2", "-2"],
    ])
    def test_two_ranges_are_joined_for_the_grammar_to_refuse(
        self, words, here, seen, capsys,
    ):
        """One range per run is the grammar's rule ([RANGE_UNRECOGNIZED] … one
        range per run, step 1): the dispatcher hands it the text whole."""
        code, _out, err = _run(["review-commits", *words, *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--revs", " ".join(words), *HOST]

    @pytest.mark.parametrize("mode", COMMIT_MODES)
    def test_no_words_emit_nothing(self, mode, here, seen, capsys):
        code, _out, err = _run([mode, "--quick", *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--quick", *HOST]

    def test_the_target_is_the_rows_front_door(self, here, seen, capsys):
        for mode in COMMIT_MODES:
            assert _run([mode, *HOST], capsys)[0] == 0
        assert [call["module"] for call in seen] == [
            "review_commits_orchestrator", "apply_commit_fixes_orchestrator",
            "improve_commits_orchestrator",
        ]

    def test_the_invocation_is_recorded_under_the_commit_word(
        self, here, seen, capsys, monkeypatch,
    ):
        """What the dispatcher records before the front door runs; step 7
        replaces it with the fronted mode's."""
        recorded = {}

        def loader(module_path, attr):
            def target(argv):
                recorded.update(cli_common.invocation() or {})
                return 0

            return target

        monkeypatch.setattr(multi_llm, "_load_target", loader)
        assert _run(["review-commits", "last", "2", *HOST], capsys)[0] == 0
        assert recorded == {"mode": "review-commits",
                            "argv": ["--revs", "last 2", *HOST]}
        assert cli_common.replay_flags(recorded["mode"], recorded["argv"]) == HOST


# ---------------------------------------------------------------------------
# 3. A brief path, and only a brief, becomes --plan-file
# ---------------------------------------------------------------------------

class TestTheBriefRule:
    @pytest.mark.parametrize("mode", COMMIT_MODES)
    def test_a_brief_by_its_marker(self, mode, here, fakes, seen, capsys):
        _brief(here, "notes/elsewhere.md")
        code, _out, err = _run([mode, "notes/elsewhere.md", *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--plan-file", "notes/elsewhere.md", *HOST]
        assert fakes == [("notes/elsewhere.md", None)]

    def test_the_one_helper_decides(self, here, fakes, seen, capsys, monkeypatch):
        """A brief in the folder with its marker cut is a brief because
        `is_commit_brief` says so (its folder rule); the dispatcher only
        finds the file and asks."""
        _brief(here, marker=False)
        import utils

        monkeypatch.setattr(utils.commit_review, "is_commit_brief",
                            lambda path, *, cwd=None: path == BRIEF_REL)
        code, _out, err = _run(["apply-commit-fixes", BRIEF_REL, *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--plan-file", BRIEF_REL, *HOST]

    @pytest.mark.parametrize("name, body", [
        ("plans/x.md", "# A plan\n"),
        ("README.md", "# Readme\n"),
        ("docs/quoting.md", "# Quotes it below\n" + MARKER + "\n"),
    ])
    def test_any_other_markdown_file_is_range_text(
        self, name, body, here, fakes, seen, capsys,
    ):
        """Never a plan on these modes: the grammar refuses it naming
        `--context-plan` (step 1), or takes a revision of that name."""
        (here / name).parent.mkdir(parents=True, exist_ok=True)
        (here / name).write_text(body, encoding="utf-8")
        code, _out, err = _run(["review-commits", name, *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--revs", name, *HOST]
        assert fakes == [(name, None)]

    @pytest.mark.parametrize("words", [
        ["plans/gone.md"],               # no such file
        ["src/main.py"],                 # a file, not markdown
        [BRIEF_REL, "HEAD"],             # two words
    ])
    def test_the_helper_is_asked_only_about_one_existing_md_word(
        self, words, here, fakes, seen, capsys,
    ):
        _brief(here)
        (here / "src").mkdir()
        (here / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
        code, _out, err = _run(["review-commits", *words, *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--revs", " ".join(words), *HOST]
        assert fakes == []


# ---------------------------------------------------------------------------
# 4. The flags the front doors own: [COMMIT_FLAG_CONFLICT], exit 3
# ---------------------------------------------------------------------------

CONFLICTS = [
    (["--plan-file", "plans/x.md"], "--plan-file", ""),
    (["--plan", "plans/x.md"], "--plan-file", " (typed as --plan)"),
    (["--plan-file=plans/x.md"], "--plan-file", ""),
    (["--base-ref", "HEAD~2"], "--base-ref", ""),
    (["--base=HEAD~2"], "--base-ref", " (typed as --base)"),
    (["--head-ref", "HEAD"], "--head-ref", ""),
]


class TestTheFlagsTheFrontDoorsOwn:
    @pytest.mark.parametrize("mode", COMMIT_MODES)
    @pytest.mark.parametrize("tokens, flag, typed", CONFLICTS)
    def test_a_typed_one_is_refused_before_anything_runs(
        self, mode, tokens, flag, typed, here, seen, capsys,
    ):
        code, out, err = _run([mode, "HEAD~2..HEAD", *tokens, *HOST], capsys)
        assert code == 3
        assert seen == []
        assert f"[COMMIT_FLAG_CONFLICT] {flag}{typed}\n" in out
        assert f"ERROR: --{mode} takes no {flag}" in err
        result = parse_result_line(out)
        assert result["exit"] == 3 and result["mode"] == mode
        assert result["error"].startswith(f"[COMMIT_FLAG_CONFLICT] {flag}")
        assert list(here.iterdir()) == []

    def test_each_names_the_commit_spelling(self, here, seen, capsys):
        _code, _out, err = _run(["review-commits", "--plan-file", "x.md"], capsys)
        assert "`--context-plan <path>`" in err and "`--review-commits <brief>`" in err
        _code, _out, err = _run(["apply-commit-fixes", "--base-ref", "C"], capsys)
        assert "`--apply-commit-fixes C..HEAD`" in err
        _code, _out, err = _run(["improve-commits", "--head-ref", "B"], capsys)
        assert "`--improve-commits A..B`" in err

    def test_the_scan_reads_the_tokens_as_typed(self, here, fakes, seen, capsys):
        """A root range's printed command is a commit mode and its brief path,
        which the dispatcher itself turns into `--plan-file` after the scan."""
        _brief(here)
        code, _out, err = _run(["apply-commit-fixes", BRIEF_REL, *HOST], capsys)
        assert code == 0, err
        assert seen[0]["argv"] == ["--plan-file", BRIEF_REL, *HOST]
        code, out, _err = _run(
            ["apply-commit-fixes", BRIEF_REL, "--plan-file", BRIEF_REL], capsys)
        assert code == 3 and "[COMMIT_FLAG_CONFLICT] --plan-file" in out

    def test_a_detached_worker_is_not_scanned(self, here, seen, capsys):
        argv = ["improve-commits", "--plan-file", BRIEF_REL, "--base-ref", "a" * 40,
                "--head-ref", "b" * 40, "--yes", "--internal-worker"]
        code, _out, err = _run(argv, capsys)
        assert code == 0, err
        assert seen[0]["argv"][:6] == argv[1:7]

    def test_everything_else_is_forwarded(self, here, seen, capsys):
        argv = ["review-commits", "--files", "a.py", "b.py", "--models", "x:y",
                "--focus", "error handling", "--context-plan", "plans/x.md", *HOST]
        code, _out, err = _run(argv, capsys)
        assert code == 0, err
        assert seen[0]["argv"] == argv[1:]

    def test_the_phase_modes_keep_their_flags(self, here, seen, capsys):
        code, _out, err = _run(
            ["review-code", "plans/x.md", "--base-ref", "HEAD~2", *HOST], capsys)
        assert code == 0, err
        assert "--base-ref" in seen[0]["argv"]

    def test_help_is_forwarded_untouched(self, here, seen, capsys):
        code, _out, _err = _run(["review-commits", "--plan-file", "x", "--help"], capsys)
        assert code == 0
        assert seen[0]["argv"] == ["--plan-file", "x", "--help"]


class TestUtilCommitBrief:
    def test_it_is_a_util_command_routed_to_the_front_door_core(self):
        module, attr, summary = multi_llm.UTIL_COMMANDS["commit-brief"]
        assert (module, attr) == ("utils.commit_review", "commit_brief_main")
        assert "commit brief" in summary

    def test_it_takes_the_typed_tokens_verbatim(self, here, seen, capsys):
        argv = ["last", "2", "commits", "--focus", "x", "--quick"]
        code, _out, err = _run(["util", "commit-brief", *argv], capsys)
        assert code == 0, err
        assert seen == [{"module": "utils.commit_review",
                         "attr": "commit_brief_main", "argv": argv}]

    @pytest.mark.parametrize("tokens, flag, typed", CONFLICTS)
    def test_it_refuses_the_same_three(self, tokens, flag, typed, here, seen, capsys):
        code, out, err = _run(["util", "commit-brief", "-2", *tokens], capsys)
        assert code == 3
        assert seen == []
        assert f"[COMMIT_FLAG_CONFLICT] {flag}{typed}\n" in out
        assert f"ERROR: util commit-brief takes no {flag}" in err

    def test_util_help_names_it(self, capsys):
        assert multi_llm.main(["util", "--help"]) == 0
        out = capsys.readouterr().out
        assert "commit-brief" in out and "[COMMIT_BRIEF] <path>" in out


# ---------------------------------------------------------------------------
# 5. `status` takes the pinned spelling, never the grammar
# ---------------------------------------------------------------------------

@pytest.fixture
def status_reads(monkeypatch):
    """`--abort` and the workflow view record the plan path `status` read."""
    reads = []

    def abort(plan_path, args):
        reads.append(("abort", str(plan_path)))
        return 0

    def workflow(plan_path, host_id=None):
        reads.append(("workflow", str(plan_path)))
        return {}

    monkeypatch.setattr(run_status, "_run_abort", abort)
    monkeypatch.setattr(run_status, "_workflow_payload", workflow)
    return reads


class TestStatusTakesThePinnedSpelling:
    def test_bare_status_keeps_its_missing_plan_file_error(self, here, capsys):
        code, _out, err = _run(["status"], capsys)
        assert code == 3
        assert "--plan-file" in err and "required" in err

    def test_status_never_takes_a_count_word(self):
        assert multi_llm._split_positionals(["-2"], _mode("status")) == ([], ["-2"])

    def test_a_phrase_of_several_words_is_the_extra_positional_error(self, here, capsys):
        code, _out, err = _run(["status", "last", "2", "commits"], capsys)
        assert code == 3
        assert "positional" in err

    def test_the_pinned_range_is_its_brief(self, here, fakes, status_reads, capsys):
        brief = _brief(here)
        code, _out, err = _run(["status", PINNED, "--abort"], capsys)
        assert code == 0, err
        code, _out, err = _run(["status", PINNED, "--view", "workflow"], capsys)
        assert code == 0, err
        assert status_reads == [("abort", str(brief)), ("workflow", str(brief))]

    def test_a_missing_brief_is_commit_review_not_found(self, here, fakes, status_reads, capsys):
        code, out, _err = _run(["status", PINNED], capsys)
        assert code == 3
        brief = here / BRIEF_REL
        assert f"[COMMIT_REVIEW_NOT_FOUND] {PINNED}: no commit brief at {brief}\n" in out
        assert f"--review-commits {PINNED}" in out
        assert status_reads == []
        assert parse_result_line(out)["error"].startswith("[COMMIT_REVIEW_NOT_FOUND]")

    def test_json_output_stays_one_object(self, here, fakes, capsys):
        code, out, _err = _run(["status", PINNED, "--json"], capsys)
        assert code == 3
        payload = json.loads(out.rsplit(cli_common.RESULT_MARKER, 1)[0])
        assert payload["error"].startswith(f"[COMMIT_REVIEW_NOT_FOUND] {PINNED}")

    @pytest.mark.parametrize("typed", [
        "today", "branch", "HEAD~2..HEAD", "root..abc1234", "main...HEAD",
        "abc1234..", "..HEAD", "a1b2c3d..e4f5a6b",
    ])
    def test_other_range_text_is_refused(self, typed, here, fakes, status_reads, capsys):
        code, out, _err = _run(["status", typed], capsys)
        assert code == 3
        assert out.startswith(f"[RANGE_UNRECOGNIZED] {typed}\n")
        assert "pinned range <base12>..<head12>" in out
        assert "commit brief's path" in out
        assert status_reads == []

    def test_a_brief_path_and_a_plan_are_unchanged(self, here, fakes, status_reads, capsys):
        brief = _brief(here, "plans/commit-reviews/root-e4f5a6b7c8d9.md")
        plan = here / "plans" / "x.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        for value in ("plans/commit-reviews/root-e4f5a6b7c8d9.md", "plans/x.md"):
            code, _out, err = _run(["status", value, "--abort"], capsys)
            assert code == 0, err
        assert status_reads == [("abort", str(brief)), ("abort", str(plan))]

    @pytest.mark.parametrize("typed", ["plans/nope.md", "../nope.md", "nope"])
    def test_anything_else_keeps_plan_file_not_found(self, typed, here, fakes, capsys):
        code, out, _err = _run(["status", typed], capsys)
        assert code == 1
        assert "Plan file not found" in out
        assert "[RANGE_UNRECOGNIZED]" not in out


# ---------------------------------------------------------------------------
# 6. --help, the result line, the replay
# ---------------------------------------------------------------------------

class TestTheSurfaces:
    def test_help_renders_the_range_kind(self):
        text = multi_llm.render_help()
        for mode in COMMIT_MODES:
            line = next(l for l in text.splitlines() if l.startswith(f"  {mode} "))
            assert mode_table.POS_REVS in line
        assert "take an optional [<range>], HEAD" in " ".join(text.split())

    def test_the_result_line_carries_the_commit_blocks_only_when_set(self):
        ctx = cli_common.ResultContext("review-commits", "code-review")
        assert "commit_range" not in ctx.payload(0)
        assert "commit_fixes" not in ctx.payload(0)
        rng = {"brief": BRIEF_REL, "base": "a" * 40, "head": "b" * 40,
               "commits": [], "more": 0, "full_list": None, "window": None,
               "drifted": []}
        fixes = {"to_apply": 1, "per_commit": [], "skipped": 0}
        cli_common.activate_result(ctx)
        try:
            cli_common.record_commit_range_result(rng)
            cli_common.record_commit_fixes_result(fixes)
        finally:
            cli_common.deactivate_result()
        rng["more"] = 99
        payload = ctx.payload(3)
        assert payload["commit_range"]["more"] == 0
        assert payload["commit_fixes"] == fixes
        cli_common.record_commit_range_result(rng)  # no active run: a no-op

    def test_a_front_door_s_record_reaches_the_line(self, here, capsys, monkeypatch):
        def loader(module_path, attr):
            def target(argv):
                cli_common.record_commit_range_result({"brief": BRIEF_REL, "more": 0})
                return 0

            return target

        monkeypatch.setattr(multi_llm, "_load_target", loader)
        code, out, _err = _run(["review-commits", *HOST], capsys)
        assert code == 0
        assert parse_result_line(out)["commit_range"] == {"brief": BRIEF_REL, "more": 0}

    @pytest.mark.parametrize("tokens", [
        ["--revs", "last two days"], ["--revs=-2"],
        ["--commit-stage", "/r/plans/commit-reviews/x/code-review/.stage-1.json"],
        ["--commit-stage=/s.json"],
    ])
    def test_the_range_words_and_the_stage_are_never_replayed(self, tokens):
        for mode in COMMIT_MODES + ("review-code", "apply-code-fixes"):
            assert cli_common.replay_flags(mode, [*tokens, "--quick"]) == ["--quick"]

    @pytest.mark.parametrize("mode", ["apply-commit-fixes", "improve-commits"])
    def test_yes_implies_the_judge_on_the_commit_apply_modes(self, mode):
        assert mode in cli_common.YES_IMPLIES_JUDGE_MODES
        flags = cli_common.replay_flags(
            mode, ["--revs", PINNED, "--yes", "--agent-decide", "--detach"])
        assert flags == ["--detach"]
        assert "review-commits" not in cli_common.YES_IMPLIES_JUDGE_MODES


# ---------------------------------------------------------------------------
# 7. The thin front-door modules
# ---------------------------------------------------------------------------

class TestTheFrontDoorModules:
    @pytest.mark.parametrize("mode", COMMIT_MODES)
    def test_each_hands_its_word_to_the_core(self, mode, fakes, monkeypatch):
        import importlib
        import utils

        calls = []
        monkeypatch.setattr(utils.commit_review, "front_door",
                            lambda word, argv: calls.append(("front_door", word, argv)) or 7,
                            raising=False)
        monkeypatch.setattr(utils.commit_review, "parse_args",
                            lambda word, argv: calls.append(("parse_args", word, argv)),
                            raising=False)
        module = importlib.import_module(_mode(mode).target_module)
        assert module.MODE == mode
        assert module.main(["-2", "--quick"]) == 7
        module.parse_args(["--help"])
        assert calls == [("front_door", mode, ["-2", "--quick"]),
                         ("parse_args", mode, ["--help"])]
        assert mode_table.mode_for_script(f"{_mode(mode).target_module}.py") == mode
