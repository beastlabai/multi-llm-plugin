"""Resume must replay verified write evidence, including interrupted attempts."""

import json
from pathlib import Path

import pytest

from tests import test_write_executor

env = test_write_executor.env


@pytest.mark.parametrize("claim", ["no-op", "applied"])
def test_resume_preserves_no_change_verdict(env, claim):
    task = env.apply_task(0)
    result = {
        "status": claim, "summary": "Nothing changed", "ids": ["s0"],
        "files_touched": ["plans/p.md"],
    }
    first = env.scheduler([task], respond=lambda *_: result, finalize=False).run()
    resumed = env.scheduler(
        [task], respond=lambda *_: pytest.fail("completed task dispatched again"),
        resume=True, finalize=False,
    ).run()
    before, after = first.outcomes[0], resumed.outcomes[0]
    assert after.skipped
    assert after.write_outcome == before.write_outcome
    assert after.files_touched == before.files_touched == []
    assert after.sha256_before == before.sha256_before
    assert after.sha256_after == before.sha256_after


def test_resume_keeps_observed_files_when_runner_omitted_them(env):
    task = env.apply_task(0)

    def respond(*_):
        env.plan.write_text("# Applied\n", encoding="utf-8")
        return {"status": "applied", "summary": "Applied", "ids": ["s0"]}

    first = env.scheduler([task], respond=respond, finalize=False).run()
    resumed = env.scheduler(
        [task], respond=lambda *_: pytest.fail("already done"),
        resume=True, finalize=False,
    ).run()
    assert resumed.outcomes[0].files_touched == first.outcomes[0].files_touched
    assert resumed.outcomes[0].files_touched == ["plans/p.md"]
    assert resumed.outcomes[0].expected_ids == ["s0"]
    assert resumed.outcomes[0].returned_ids == ["s0"]


@pytest.mark.parametrize("old_success", [False, True])
def test_interruption_before_classification_cannot_resume_as_success(env, monkeypatch, old_success):
    task = env.apply_task(0)

    def respond(*_):
        env.plan.write_text(env.plan.read_text() + "Partial edit\n")
        return {"status": "applied", "summary": "Claimed success", "ids": ["s0"]}

    if old_success:
        env.scheduler([task], respond=respond, finalize=False).run()

    interrupted = env.scheduler([task], respond=respond, finalize=False, resume=False)

    def crash(*_, **__):
        raise KeyboardInterrupt("orchestrator stopped before verdict")

    monkeypatch.setattr(interrupted, "_classify", crash)
    with pytest.raises(KeyboardInterrupt):
        interrupted.run()
    data = json.loads(Path(task["output_path"]).read_text())
    assert data["status"] == "applied"
    assert "verdict" not in data
    called = []

    def finish(*_):
        called.append(True)
        return {"status": "no-op", "summary": "Verified existing edit", "ids": ["s0"]}

    resumed = env.scheduler(
        [task], respond=finish, resume=True, finalize=False,
    ).run()
    assert called == [True]
    assert not resumed.outcomes[0].skipped
    assert resumed.exit_code == 0
