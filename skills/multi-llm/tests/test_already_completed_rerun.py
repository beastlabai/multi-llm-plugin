"""A re-run of a review phase the plan already completed says it is done
and nothing ran, never an error (fix round 28, live run 12 #11).

OpenCode racing 01f retried a review-plan its killed first attempt had
already finished. The plugin answered exit 2, `state: nothing-to-do`, with
"ERROR: Phase 'review-plan' has already been completed for this plan. …
Use --force to re-run, or --reaggregate to reprocess existing results.", and
the host relayed it as an error block above the next command. Nothing had
failed: exit 2 is "nothing to do / phase already complete", which every
route reads as a success (the servers answer `ok: true`). The text now says
the phase is done and nothing ran, where its results are, and the two ways
to run it again; the result's `notice` carries the same words (the field
every route relays as plain text), and `user_next_step` is the next phase
as on any exit 2. Same on review-code.
"""

from __future__ import annotations

import pytest

from tests.test_next_command_emitters import (
    PLUGIN,
    _run,
    build,
    workdir,  # noqa: F401 - the fixture
)
from tests.test_next_phase import CR_FINDING, P, RP_VALID, S, UP_TO_CODE_REVIEW
from utils.review_orchestrator_base import already_completed_lines


@pytest.mark.parametrize("mode,spec,folder,expect", [
    ("review-plan", S(("review-plan",), files=RP_VALID), "review-plan",
     f"--apply-suggestions {P}"),
    ("review-code", S(UP_TO_CODE_REVIEW, files=CR_FINDING), "code-review",
     f"--apply-code-fixes {P}"),
], ids=["review-plan", "review-code"])
def test_a_rerun_of_a_completed_review_says_it_is_done_not_an_error(
        workdir, capsys, mode, spec, folder, expect):
    build(workdir, spec)
    code, out, err, result = _run([mode, P, "--models", "mock:any"], capsys)
    assert code == 2, (out, err)
    assert "ERROR" not in out and "ERROR" not in err, (out, err)
    said = already_completed_lines(mode, f"plans/foo/{folder}")
    for line in said:
        assert line in out, out
    assert "nothing ran" in said[0] and "--force" in said[1]
    # Read as done, not failed, and the way on is the next phase.
    assert result["state"] == "nothing-to-do"
    assert "error" not in result
    assert result["notice"] == " ".join(said)
    assert result["user_next_step"]["command"] == PLUGIN + expect


def test_a_results_folder_outside_the_cwd_is_named_whole(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    folder = tmp_path / "repo" / "plans" / "foo" / "review-plan"
    assert f"its results are in {folder}." in already_completed_lines(
        "review-plan", folder)[0]
