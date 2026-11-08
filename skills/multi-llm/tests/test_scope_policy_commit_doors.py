"""``--scope-policy`` through the commit front doors (plans/review-commits.md §2.1).

The three commit modes front review-code, apply-code-fixes and the
improve-commits composite and rebuild the argv they hand on. The policy is not
a door-owned flag, so every door forwards it verbatim to the phase it fronts,
the re-recorded invocation keeps it (a detached worker re-executes that), and
``util commit-brief`` hands it back on ``[COMMIT_REMAINING_FLAGS]`` for
``util improve-steps``. The fixtures are ``tests/test_commit_front_door.py``'s.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import cli_common, commit_review  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils.cli_common import RESULT_MARKER  # noqa: E402

from tests.support.commit_door import (  # noqa: E402
    FOLDER,
    _brief_of,
    _complete_review,
    _door,
    _review,
    _sha,
    door_fronted,  # noqa: F401 - the fronted fixture
    door_repo,  # noqa: F401 - the repo fixture
    door_seams,  # noqa: F401 - the seams fixture
)

PRESERVE = ["--scope-policy", "preserve"]


def _has(argv, value="preserve"):
    return any(t == "--scope-policy" and argv[i + 1] == value
               for i, t in enumerate(argv[:-1]))


class TestTheDoorsForwardIt:
    def test_review_commits_hands_it_to_review_code(self, repo, fronted, capsys):
        assert _review("--revs", "last 2 commits", *PRESERVE, "--quick") == 0
        [call] = fronted
        assert call["target"] == "review-code"
        assert _has(call["argv"])
        assert call["invocation"]["mode"] == "review-code"
        assert _has(call["invocation"]["argv"])

    def test_apply_commit_fixes_hands_it_to_apply_code_fixes(self, repo, fronted, capsys):
        assert _review("--revs", "last 2 commits") == 0
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        _complete_review(brief)
        fronted.clear()
        assert _door("apply-commit-fixes", "--revs", "last 2 commits", "--yes",
                     "--scope-policy", "expand") == 0
        [call] = fronted
        assert call["target"] == "apply-code-fixes"
        assert call["argv"] == ["--yes", "--scope-policy", "expand",
                                "--plan-file", f"{FOLDER}/{brief.name}"]

    def test_improve_commits_hands_it_to_the_composite_and_records_it(
        self, repo, fronted, capsys
    ):
        base, head = _sha(repo, "HEAD~2"), _sha(repo, "HEAD")
        assert _door("improve-commits", "--revs", "last 2 commits", *PRESERVE) == 0
        [call] = fronted
        assert call["target"] == "entry_point:improve-commits"
        assert _has(call["argv"])
        assert call["invocation"]["argv"] == [
            *PRESERVE, "--revs", f"{base[:12]}..{head[:12]}"]

    def test_the_composite_reads_it_off_the_rebuilt_argv(self, repo, fronted, capsys):
        assert _door("improve-commits", "--revs", "last 2 commits", *PRESERVE) == 0
        [call] = fronted
        args = iw.parse_args("improve-commits", call["argv"])
        assert args.scope_policy == "preserve"
        brief = Path(os.path.abspath(cli_common.flag_value(call["argv"], "--plan-file")))
        word, rest = iw.continuation_args("improve-commits", brief, args)
        assert word == "improve-commits"
        assert rest[-3:] == [*PRESERVE, "--resume"]
        assert "--base-ref" not in rest and "--head-ref" not in rest

    def test_util_commit_brief_hands_it_back_for_the_planner(self, repo, capsys):
        assert commit_review.commit_brief_main(
            ["last", "2", "commits", *PRESERVE, "--quick"]) == 0
        out = capsys.readouterr().out
        line = next(line for line in out.splitlines()
                    if line.startswith("[COMMIT_REMAINING_FLAGS]"))
        tokens = line.split()[1:]
        assert tokens == [*PRESERVE, "--quick"]
        brief = _brief_of(repo, _sha(repo, "HEAD~2"), _sha(repo, "HEAD"))
        assert multi_llm.main(["util", "improve-steps", f"{FOLDER}/{brief.name}",
                               "--mode", "improve-commits", "--json", *tokens]) == 0
        payload = json.loads(capsys.readouterr().out.rsplit(RESULT_MARKER, 1)[0])
        for step in payload["steps"]:
            assert _has(step["argv"]), step
        assert payload["rerun_argv"][-3:] == [*PRESERVE, "--resume"]
