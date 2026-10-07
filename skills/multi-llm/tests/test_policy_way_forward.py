"""The way past a write-policy refusal works when the user follows it.

Fix round 21 (review of round 20, stops #1 and #2). Round 20 worded every
policy refusal's way forward as "if you want src/vendor/stamp.js written,
allow it in driver.write_paths". The deny list wins over ``allow``
(``WritePolicy.violation``), and live run 8's refusal was a deny hit
(``src/vendor/**``): a user who followed the words — added the file to
``allow`` — resumed into the same refusal with the same advice. The way
forward is now worded per rule: a deny hit names the deny rule to remove or
narrow, an allow miss says to add the file to ``allow``, and a file outside
the repository, which no rule can allow, gets the skip alone, named as the
task declared it (it was named by its resolved absolute path, in ``error``,
``policy_paths``, the mirror, the summary and ``state.json``).

Each story is driven through the real ``implement`` orchestrator on both
legs (``--exec-subagents`` and the native emit, plus ``util write-prompt``)
and checks that doing what the stop says gets past it.
"""

from __future__ import annotations

import json

import pytest

from tests.test_check_gate_fixture import (  # noqa: E402
    CHECK_GATE,
    DENY,
    DENY_CLAUSE,
    PLAN,
    PRINT,
    STAMP,
    _check,
    _policy,
)
from tests.test_implement_stop_account import (  # noqa: E402
    _dispatched_titles,
    _implement,
    _statuses,
    make_repo,
)

import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.cli_common import ERROR_MAX_CHARS  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

#: run 8's deny hit, after the user followed round 20's words literally.
ALLOWED_BUT_DENIED = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**', 'src/vendor/stamp.js']\n"
    "    deny: ['.git/**', '.multi-llm/**', 'src/vendor/**']\n"
)
#: The deny rule narrowed as the stop now says: it no longer covers the file.
NARROWED = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**']\n"
    "    deny: ['.git/**', '.multi-llm/**', 'src/vendor/*.min.js']\n"
)
#: An allow miss: nothing allows src/vendor/stamp.js, nothing denies it.
ALLOW_MISS = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['src/*.py', 'plans/**']\n    deny: ['.git/**', '.multi-llm/**']\n"
)
#: The same, with the file added to `allow` as the stop says.
ALLOW_ADDED = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['src/*.py', 'plans/**', 'src/vendor/stamp.js']\n"
    "    deny: ['.git/**', '.multi-llm/**']\n"
)
ALLOW_EVERYTHING = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**']\n    deny: ['.git/**', '.multi-llm/**']\n"
)
OUTSIDE_PLAN = PLAN.replace(
    "- Files to modify: src/main.py, src/vendor/stamp.js",
    "- Files to modify: src/main.py, ../outside-stamp.js",
)
OUTSIDE_NOTE = (
    "../outside-stamp.js is outside the repository, so no driver.write_paths "
    "rule can allow it"
)
SKIP_ALONE = (
    "skip T005: add --human-task T005=skip to the resume command (T005 and "
    "the tasks after it stay undone)"
)


def _repo(tmp_path, monkeypatch, plan: str, policy: str):
    obj = make_repo(tmp_path, monkeypatch, plan)
    _policy(obj, policy)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        [f"T00{n}" for n in range(1, 8)]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    return obj


@pytest.fixture(autouse=True)
def _fresh_config():
    yield
    registry._config = None
    registry._config_key = None


def _stop(repo, capsys, *flags) -> tuple:
    """Run implement on the executor leg to its stop: (text, result, stop)."""
    rc = multi_llm.main([
        "implement", str(repo.plan), "--yes", "--exec-subagents", *flags,
    ])
    captured = capsys.readouterr()
    assert rc == 3, captured.out[-2000:]
    repo.log.unlink(missing_ok=True)
    line = [l for l in captured.out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
    report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
    return captured.out + captured.err, json.loads(line.split(" ", 1)[1]), report["stop"]


def _reason(repo, task_id: str) -> str:
    return get_or_create_state(repo.plan).state["task_status_reasons"][task_id]


class TestADenyHitNamesTheDenyRule:
    """Run 8's case: `allow: ['**']`, `deny: [… 'src/vendor/**']`."""

    def test_following_the_way_forward_gets_past_the_stop(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = _repo(tmp_path, monkeypatch, PLAN, DENY)
        _check(repo, passed=False)
        text, result, stop = _stop(repo, capsys)
        assert (f"To go on (your call): {DENY_CLAUSE}, then resume this "
                f"phase; if not, {SKIP_ALONE}.") in result["error"], result["error"]
        assert len(result["error"]) <= ERROR_MAX_CHARS
        assert stop["policy_rules"] == [{
            "path": "src/vendor/stamp.js", "deny": ["src/vendor/**"], "allowed": True,
        }]
        for said in (text, result["error"], _reason(repo, "T005")):
            assert "allow it in driver.write_paths" not in said

        # Round 20's words, followed literally: the deny rule still wins, and
        # the stop again names the rule, never "allow it".
        _policy(repo, ALLOWED_BUT_DENIED)
        _text, again, _stop_again = _stop(repo, capsys, "--resume")
        assert _dispatched_titles(repo) == []
        assert DENY_CLAUSE in again["error"], again["error"]

        # The way forward it names: the deny rule narrowed off the file.
        _policy(repo, NARROWED)
        _implement(repo, "--resume")
        capsys.readouterr()
        assert _dispatched_titles(repo)[:2] == [STAMP, PRINT]
        assert _statuses(repo)["T005"] == "completed"

    def test_a_file_both_denied_and_not_allowed_names_both_changes(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = _repo(tmp_path, monkeypatch, PLAN, (
            "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
            "    allow: ['src/*.py', 'plans/**']\n"
            "    deny: ['.git/**', '.multi-llm/**', '**/*.js']\n"
        ))
        _check(repo, passed=False)
        _text, result, stop = _stop(repo, capsys)
        assert (
            "if you want src/vendor/stamp.js written, remove or narrow the deny "
            "rule `**/*.js` in driver.write_paths (.multi-llm/providers.yaml) so "
            "the rule no longer covers it, and add it to driver.write_paths.allow, "
            "then resume this phase"
        ) in result["error"], result["error"]
        assert stop["policy_rules"] == [{
            "path": "src/vendor/stamp.js", "deny": ["**/*.js"], "allowed": False,
        }]


class TestAnAllowMissNamesTheAllowList:

    def test_following_the_way_forward_gets_past_the_stop(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = _repo(tmp_path, monkeypatch, PLAN, ALLOW_MISS)
        _check(repo, passed=False)
        text, result, stop = _stop(repo, capsys)
        clause = (
            "if you want src/vendor/stamp.js written, add it to "
            "driver.write_paths.allow (.multi-llm/providers.yaml)"
        )
        assert (f"To go on (your call): {clause}, then resume this phase; "
                f"if not, {SKIP_ALONE}.") in result["error"], result["error"]
        assert f"  - your call: {clause}, then resume this phase" in text
        assert _reason(repo, "T005").endswith(f"Your call: {clause}, then resume.")
        assert stop["policy_rules"] == [{
            "path": "src/vendor/stamp.js", "deny": [], "allowed": False,
        }]

        _policy(repo, ALLOW_ADDED)
        _implement(repo, "--resume")
        capsys.readouterr()
        assert _dispatched_titles(repo)[:2] == [STAMP, PRINT]


class TestAFileOutsideTheRepository:
    """No `driver.write_paths` rule can allow `../outside-stamp.js`: the stop
    offers the skip alone, and names the file as T005 declared it."""

    def test_the_executor_stop_offers_only_the_skip(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = _repo(tmp_path, monkeypatch, OUTSIDE_PLAN, ALLOW_EVERYTHING)
        _check(repo, passed=False)
        text, result, stop = _stop(repo, capsys)
        error = result["error"]
        assert (f"{OUTSIDE_NOTE[0].upper()}{OUTSIDE_NOTE[1:]}. To go on: "
                f"{SKIP_ALONE}.") in error, error
        assert "your call" not in error and "written, " not in error
        assert f"To go on ({OUTSIDE_NOTE}):" in text, text
        bullet = next(line for line in text.splitlines()
                      if line.startswith("  - go on without T005: "))
        assert bullet.split(": ", 1)[1].startswith("/multi-llm:multi-llm --implement ")
        assert "--resume --human-task T005=skip (T005 and the tasks" in bullet, bullet
        assert "your call" not in text
        assert stop["policy_paths"] == ["../outside-stamp.js"]
        assert stop["policy_rules"] == [{"path": "../outside-stamp.js", "outside": True}]
        absolute = str(repo.root.parent / "outside-stamp.js")
        reason = _reason(repo, "T005")
        assert reason.endswith(f"{OUTSIDE_NOTE[0].upper()}{OUTSIDE_NOTE[1:]}."), reason
        summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
        mirror = (repo.phase_dir / ".status.json").read_text(encoding="utf-8")
        assert "../outside-stamp.js" in summary
        for said in (text, error, reason, json.dumps(stop), summary, mirror):
            assert absolute not in said
            assert "allow it in" not in said and "allow them in" not in said
        # Its answers are still offered: skip, or done by hand.
        flags = [" ".join(entry["flags"]) for entry in stop["answers"]]
        assert "--resume --human-task T005=skip" in flags
        assert "--resume --human-task T005=done" in flags

    def test_the_native_emit_and_write_prompt_offer_only_the_skip(
        self, tmp_path, monkeypatch, capsys,
    ):
        import implement_orchestrator

        repo = _repo(tmp_path, monkeypatch, OUTSIDE_PLAN, ALLOW_EVERYTHING)
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes"]) == 0
        out = capsys.readouterr().out
        path = next(line.split(": ", 1)[1] for line in out.splitlines()
                    if line.startswith("Output file: "))
        handoff = json.loads(open(path, encoding="utf-8").read())
        refused = handoff["refused_tasks"]["T005"]
        assert refused.endswith(
            f"{OUTSIDE_NOTE[0].upper()}{OUTSIDE_NOTE[1:]}. "
            f"{SKIP_ALONE[0].upper()}{SKIP_ALONE[1:]}."
        ), refused
        absolute = str(repo.root.parent / "outside-stamp.js")
        assert absolute not in refused and "If not" not in refused

        # A task file that still hands T005 out (the emit never refuses on a
        # guess): `util write-prompt` refuses it in the same words.
        written_path = repo.phase_dir / "write_tasks.json"
        _policy(repo, ALLOW_EVERYTHING)
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", "--fresh"]) == 0
        capsys.readouterr()
        written = json.loads(written_path.read_text(encoding="utf-8"))
        tasks = written.get("batches") or written.get("tasks")
        assert "T005" not in [t["task_id"] for t in tasks]
        stamp = dict(next(t for t in tasks if t["task_id"] == "T004"))
        stamp.update(task_id="T005", expected_ids=["T005"], prior_changes_id="T005",
                     depends_on=["T004"], task_title=STAMP,
                     declared_files=["src/main.py", "../outside-stamp.js"])
        tasks.append(stamp)
        written_path.write_text(json.dumps(written), encoding="utf-8")
        code = multi_llm.main(["util", "write-prompt", "--phase-dir",
                               str(repo.phase_dir), "--task-id", "T005"])
        err = capsys.readouterr().err
        assert code == 3
        assert (
            "report the way forward as the user's call, in these words: "
            f"{OUTSIDE_NOTE}; skip T005 with --resume --human-task T005=skip "
            "(T005 and the tasks after it stay undone). Never change "
            "driver.write_paths yourself." in err
        ), err
        assert "both ways forward" not in err and absolute not in err


class TestWritePromptWordsTheDenyRule:

    def test_the_native_refusal_names_the_deny_rule(self, tmp_path, monkeypatch, capsys):
        import implement_orchestrator

        repo = _repo(tmp_path, monkeypatch, PLAN, ALLOW_EVERYTHING)
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        # The policy changes after the emit handed T005 out.
        _policy(repo, DENY)
        code = multi_llm.main(["util", "write-prompt", "--phase-dir",
                               str(repo.phase_dir), "--task-id", "T005"])
        err = capsys.readouterr().err
        assert code == 3
        assert f"in these words: {DENY_CLAUSE}, then re-run implement" in err, err
        assert "allow it in driver.write_paths" not in err


class TestTheClauseNamesEachRule:
    """`write_safety.policy_allow_clause` and `WritePolicy.check`, unit level."""

    def test_check_records_each_files_rule(self, tmp_path):
        from utils.write_safety import PolicyHit, PolicyViolation, WritePolicy

        policy = WritePolicy(root=str(tmp_path), allow=["src/**"],
                             deny=["src/vendor/**", "**/*.pem"])
        with pytest.raises(PolicyViolation) as caught:
            policy.check(["src/vendor/a.pem", "docs/x.md", "../up.txt", "src/ok.py"],
                         when="pre-dispatch")
        assert caught.value.paths == ["src/vendor/a.pem", "docs/x.md", "../up.txt"]
        assert caught.value.hits == [
            PolicyHit("src/vendor/a.pem", deny=("src/vendor/**", "**/*.pem")),
            PolicyHit("docs/x.md", allowed=False),
            PolicyHit("../up.txt", outside=True),
        ]
        # A file outside the repository: no policy change lets the task run.
        assert caught.value.message.endswith(
            "../up.txt is outside the repository, so no driver.write_paths rule "
            "can allow it.")
        assert "Your call" not in caught.value.message

    def test_several_files_under_different_rules_each_get_theirs(self):
        from utils.write_safety import PolicyHit, policy_allow_clause

        assert policy_allow_clause([
            PolicyHit("src/vendor/a.js", deny=("src/vendor/**",)),
            PolicyHit("docs/x.md", allowed=False),
        ]) == (
            "if you want src/vendor/a.js and docs/x.md written, remove or narrow "
            "the deny rule `src/vendor/**` in driver.write_paths "
            "(.multi-llm/providers.yaml) so the rule no longer covers "
            "src/vendor/a.js, and add docs/x.md to driver.write_paths.allow"
        )
        assert policy_allow_clause([
            PolicyHit("a.pem", deny=("**/*.pem", "*.pem")),
        ]) == (
            "if you want a.pem written, remove or narrow the deny rules "
            "`**/*.pem` and `*.pem` in driver.write_paths "
            "(.multi-llm/providers.yaml) so the rules no longer cover it"
        )
        many = policy_allow_clause([
            PolicyHit(f"{n}.js", deny=(f"{n}.js",)) for n in "abcde"
        ])
        assert many.endswith(
            "and remove or narrow the 2 other deny rules that cover d.js and "
            "e.js (the refusal's details name each)"), many

    def test_a_record_without_rules_gets_wording_right_for_either(self):
        """A stop recorded before fix round 21 kept `policy_paths` alone."""
        from utils.write_safety import policy_allow_clause

        assert policy_allow_clause(["src/vendor/stamp.js"]) == (
            "if you want src/vendor/stamp.js written, change driver.write_paths "
            "(.multi-llm/providers.yaml) so it allows it: add it to `allow`, and "
            "remove or narrow any deny rule that covers it"
        )

    def test_outside_the_repository_has_no_clause(self):
        from utils.write_safety import PolicyHit, policy_allow_clause

        assert policy_allow_clause([
            PolicyHit("src/a.js", allowed=False), PolicyHit("../b", outside=True),
        ]) is None
